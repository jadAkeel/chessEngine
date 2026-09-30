from __future__ import annotations

import argparse
import copy
import gc
import json
import hashlib
import math
import random
from dataclasses import replace
from pathlib import Path

import torch
from torch.amp import GradScaler

from app.cli.common import add_common_runtime_args
from app.evaluation.arena import play_match
from app.infra.config import load_config
from app.infra.device import select_device
from app.infra.logging import setup_logging
from app.infra.runtime import configure_torch_runtime
from app.model.checkpoint import load_checkpoint, save_checkpoint
from app.model.network import ChessNet

# 🔥 الجديد
from app.training.external_samples import load_external_samples_sharded
from app.training.external_samples import _sample_hash

from app.training.replay_buffer import ReplayBuffer
from app.training.trainer import evaluate_model_on_samples, train_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Train from external supervised data (SHARDED STREAMING)')
    add_common_runtime_args(parser)
    parser.set_defaults(config='config/external_training.yaml')
    parser.add_argument('--save-dir', type=str, default=None)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--base-model', type=str, default=None)
    parser.add_argument('--iterations', type=int, default=1)
    parser.add_argument('--max-train-samples', type=int, default=None)
    parser.add_argument('--max-val-samples', type=int, default=None)
    parser.add_argument('--train-steps', type=int, default=None)
    parser.add_argument('--no-save', action='store_true')
    parser.add_argument('--lr-override', type=float, default=None,
                        help='Set every param group to this learning rate after restoring')
    return parser


def _history_path(save_dir: Path, prefix: str) -> Path:
    return save_dir / f'{prefix}_history.json'


def _load_history(path: Path) -> dict:
    default_history = {
        'train_loss': [],
        'val_loss': [],
        'benchmark_win_rate': [],
    }
    if not path.exists():
        return default_history
    try:
        return json.loads(path.read_text())
    except Exception:
        return default_history


def _best_val_loss_from_history(history: dict) -> float:
    best = float("inf")
    start = int(history.get("validation_start_index", 0))
    for value in history.get("val_loss", [])[start:]:
        try:
            val_loss = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(val_loss):
            best = min(best, val_loss)
    return best


def _set_validation_history(history: dict, samples, cfg=None) -> float:
    digest = hashlib.sha256(b'position-split-v1')
    if cfg is not None:
        digest.update(repr((cfg.training.value_loss_coeff, cfg.training.entropy_coeff,
                            cfg.training.policy_label_smoothing)).encode('ascii'))
    for state, policy, value in samples:
        digest.update(_sample_hash(state, int(policy.indices[0]), value))
    fingerprint = digest.hexdigest()
    if history.get('validation_fingerprint') != fingerprint:
        history['validation_start_index'] = len(history.get('val_loss', []))
        history['validation_fingerprint'] = fingerprint
    return _best_val_loss_from_history(history)


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    if args.train_steps is not None:
        if args.train_steps <= 0:
            raise ValueError('--train-steps must be > 0')
        cfg = replace(cfg, training=replace(cfg.training, train_steps_per_iter=int(args.train_steps)))

    logger = setup_logging('training.external')
    device = select_device(args.device or cfg.system.device)
    configure_torch_runtime(cfg, device=str(device), role='training')

    external_cfg = cfg.external
    sample_path = external_cfg.samples_path

    checkpoint_prefix = str(external_cfg.checkpoint_prefix or 'external')
    save_dir = Path(args.save_dir or external_cfg.save_dir)
    if not args.no_save:
        save_dir.mkdir(parents=True, exist_ok=True)

    latest_ckpt = save_dir / f'{checkpoint_prefix}_latest_checkpoint.pth'
    best_ckpt = save_dir / f'{checkpoint_prefix}_best_model.pth'
    history_path = _history_path(save_dir, checkpoint_prefix)

    history = _load_history(history_path)
    overall_best_val_loss = _best_val_loss_from_history(history)

    total_iterations = max(1, int(args.iterations))

    # 🔥 validation set (small, safe)
    val_samples = []
    logger.info("[INIT] Building validation set...")

    val_iter = load_external_samples_sharded(
        sample_path,
        cfg,
        max_samples=int(args.max_val_samples if args.max_val_samples is not None else 50000),
        partition='validation',
    )

    for sample in val_iter:
        val_samples.append(sample)

    logger.info(f"[INIT] Validation samples: {len(val_samples)}")
    if not val_samples:
        raise ValueError("No validation samples; cannot select a best model from an empty validation set")
    overall_best_val_loss = _set_validation_history(history, val_samples, cfg)

    model = ChessNet(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.training.lr_decay_gamma)
    scaler = GradScaler(enabled=bool(cfg.training.use_amp and str(device).startswith('cuda')))
    global_step = 0
    if args.base_model:
        if not Path(args.base_model).exists():
            raise FileNotFoundError(args.base_model)
        load_checkpoint(args.base_model, model=model, device=device)
    elif latest_ckpt.exists():
        restored = load_checkpoint(latest_ckpt, model=model, optimizer=optimizer, scheduler=scheduler,
                                   scaler=scaler, device=device)
        global_step = int(restored['global_step'])
    elif args.resume:
        raise FileNotFoundError(latest_ckpt)
    if args.lr_override is not None:
        # A restored optimizer keeps its saved LR; the Kaggle wrapper sets a
        # per-iteration schedule through this flag.
        for group in optimizer.param_groups:
            group['lr'] = float(args.lr_override)
        logger.info("[LR] override -> %.3g", float(args.lr_override))

    # ======================================
    # 🔁 ITERATIONS LOOP (UNCHANGED LOGIC)
    # ======================================

    for current_iter in range(1, total_iterations + 1):
        logger.info("=" * 60)
        logger.info("ITERATION %s / %s", current_iter, total_iterations)
        logger.info("=" * 60)

        # ======================================
        # 🔥 STREAM → BUFFER (CORE FIX)
        # ======================================

        # Drop the previous iteration's buffer before allocating the next one.
        # ReplayBuffer preallocates capacity upfront (~12.8 GB of states at
        # capacity=5M), so rebinding without releasing first would briefly hold
        # two full buffers and exhaust the machine.
        train_buffer = None
        gc.collect()

        train_buffer = ReplayBuffer(cfg)
        buffer_limit = int(getattr(cfg.training, "buffer_size", 200000))

        logger.info("[ITER %s] Streaming shards into buffer...", current_iter)

        stream_iter = load_external_samples_sharded(
            sample_path,
            cfg,
            max_samples=int(
                args.max_train_samples
                if args.max_train_samples is not None
                else int(external_cfg.max_samples or 0)
            ),
            partition='train',
            shuffle_seed=int(external_cfg.seed) + global_step,
        )

        count = 0
        for state, policy, value in stream_iter:
            train_buffer.add(state, policy, value)
            count += 1

            if count % 50000 == 0:
                logger.info("[STREAM] loaded %s samples...", count)

            if count >= buffer_limit:
                break

        logger.info("[ITER %s] Buffer filled: %s samples", current_iter, count)
        if not len(train_buffer):
            raise ValueError("No training samples after filtering and validation split")

        # ======================================
        # 🔥 TRAIN
        # ======================================

        train_stats = train_model(
            model=model,
            optimizer=optimizer,
            buffer=train_buffer,
            device=device,
            scheduler=scheduler,
            global_step=global_step,
            scaler=scaler,
            cfg=cfg,
        )

        global_step = int(train_stats["global_step"])

        # ======================================
        # 🔥 VALIDATION
        # ======================================

        val_stats = evaluate_model_on_samples(
            model,
            val_samples,
            device=device,
            cfg=cfg,
        )

        current_val_loss = float(val_stats["loss"])

        logger.info(
            "[ITER %s] Train Loss=%.6f | Val Loss=%.6f",
            current_iter,
            float(train_stats["loss"]),
            current_val_loss
        )
        if train_stats.get("nonfinite_steps"):
            logger.warning("[ITER %s] rolled back %s non-finite steps", current_iter, train_stats["nonfinite_steps"])

        # Never overwrite a good checkpoint with NaN weights (iteration 10 of the
        # 2026-09-23 Kaggle run did, leaving a corrupt latest checkpoint).
        if not math.isfinite(current_val_loss) or not math.isfinite(float(train_stats["loss"])):
            raise RuntimeError(
                f"Iteration {current_iter} produced a non-finite loss; checkpoints left untouched"
            )

        # ======================================
        # 🔥 SAVE
        # ======================================

        if not args.no_save:
            save_checkpoint(
                latest_ckpt,
                model=model,
                cfg=cfg,
                global_step=global_step,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                meta={'validation_fingerprint': history['validation_fingerprint']},
            )

        if current_val_loss < overall_best_val_loss:
            overall_best_val_loss = current_val_loss
            logger.info("[ITER %s] NEW BEST MODEL", current_iter)

            if not args.no_save:
                save_checkpoint(
                    best_ckpt,
                    model=model,
                    cfg=cfg,
                    global_step=global_step,
                )

        history["train_loss"].append(float(train_stats["loss"]))
        history["val_loss"].append(current_val_loss)

        if not args.no_save:
            history_path.write_text(json.dumps(history, indent=2))
        del train_buffer

    logger.info("=" * 60)
    logger.info("DONE | Best Val Loss: %.6f", overall_best_val_loss)
    logger.info("=" * 60)

    print({
        "best_val_loss": overall_best_val_loss,
        "best_checkpoint": str(best_ckpt),
    })


if __name__ == "__main__":
    main()
