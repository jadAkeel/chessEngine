from __future__ import annotations

import argparse
import copy
import json
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


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_config(args.config)

    logger = setup_logging('training.external')
    device = select_device(args.device or cfg.system.device)
    configure_torch_runtime(cfg, device=str(device), role='training')

    external_cfg = cfg.external
    sample_path = external_cfg.samples_path

    checkpoint_prefix = str(external_cfg.checkpoint_prefix or 'external')
    save_dir = Path(args.save_dir or external_cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    latest_ckpt = save_dir / f'{checkpoint_prefix}_latest_checkpoint.pth'
    best_ckpt = save_dir / f'{checkpoint_prefix}_best_model.pth'
    history_path = _history_path(save_dir, checkpoint_prefix)

    history = _load_history(history_path)
    overall_best_val_loss = float("inf")

    total_iterations = max(1, int(args.iterations))

    # 🔥 validation set (small, safe)
    val_samples = []
    logger.info("[INIT] Building validation set...")

    val_iter = load_external_samples_sharded(
        sample_path,
        cfg,
        max_samples=50000
    )

    for sample in val_iter:
        val_samples.append(sample)

    logger.info(f"[INIT] Validation samples: {len(val_samples)}")

    # ======================================
    # 🔁 ITERATIONS LOOP (UNCHANGED LOGIC)
    # ======================================

    for current_iter in range(1, total_iterations + 1):
        logger.info("=" * 60)
        logger.info("ITERATION %s / %s", current_iter, total_iterations)
        logger.info("=" * 60)

        model = ChessNet(cfg).to(device)
        baseline_model = ChessNet(cfg).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=cfg.training.lr_decay_gamma,
        )

        scaler = GradScaler(enabled=bool(cfg.training.use_amp))

        global_step = 0

        # 🔹 load weights
        if args.base_model and Path(args.base_model).exists():
            logger.info("[ITER %s] Loading base model", current_iter)
            load_checkpoint(args.base_model, model=model, device=device)
        elif latest_ckpt.exists():
            logger.info("[ITER %s] Loading previous checkpoint", current_iter)
            load_checkpoint(latest_ckpt, model=model, device=device)

        baseline_model.load_state_dict(copy.deepcopy(model.state_dict()))
        baseline_model.eval()

        # ======================================
        # 🔥 STREAM → BUFFER (CORE FIX)
        # ======================================

        train_buffer = ReplayBuffer(cfg)
        buffer_limit = int(getattr(cfg.training, "buffer_size", 200000))

        logger.info("[ITER %s] Streaming shards into buffer...", current_iter)

        stream_iter = load_external_samples_sharded(
            sample_path,
            cfg,
            max_samples=int(external_cfg.max_samples or 0),
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

        # ======================================
        # 🔥 SAVE
        # ======================================

        save_checkpoint(
            latest_ckpt,
            model=model,
            cfg=cfg,
            global_step=global_step,
        )

        if current_val_loss < overall_best_val_loss:
            overall_best_val_loss = current_val_loss
            logger.info("[ITER %s] 🏆 NEW BEST MODEL", current_iter)

            save_checkpoint(
                best_ckpt,
                model=model,
                cfg=cfg,
                global_step=global_step,
            )

        history["train_loss"].append(float(train_stats["loss"]))
        history["val_loss"].append(current_val_loss)

        history_path.write_text(json.dumps(history, indent=2))

    logger.info("=" * 60)
    logger.info("DONE | Best Val Loss: %.6f", overall_best_val_loss)
    logger.info("=" * 60)

    print({
        "best_val_loss": overall_best_val_loss,
        "best_checkpoint": str(best_ckpt),
    })


if __name__ == "__main__":
    main()