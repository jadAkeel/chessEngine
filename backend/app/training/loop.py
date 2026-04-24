from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
from pathlib import Path

import torch
from torch.amp import GradScaler

from app.cli.common import add_common_runtime_args
from app.evaluation.arena import play_match
from app.evaluation.metrics import update_elo_pair
from app.infra.config import load_config, get_current_config
from app.infra.device import select_device
from app.infra.logging import setup_logging
from app.infra.runtime import configure_torch_runtime
from app.model.checkpoint import load_checkpoint, save_checkpoint
from app.model.network import ChessNet
from app.selfplay.generator import generate_self_play_data
from app.training.external_samples import load_external_samples
from app.training.replay_buffer import ReplayBuffer
from app.training.trainer import train_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='AlphaZero-style training loop')
    add_common_runtime_args(parser)
    parser.set_defaults(config=None)
    parser.add_argument('--iterations', type=int, default=50)
    parser.add_argument('--min-buffer-size', type=int, default=512)
    parser.add_argument('--save-dir', type=str, default='models')
    parser.add_argument('--resume', action='store_true')
    return parser


def _load_history(path: Path) -> dict:
    default_history = {
        'iterations': [],
        'train_loss': [],
        'arena_win_rate': [],
        'scheduled_score_rate': [],
        'accepted': [],
        'champion_elo': [],
        'learner_elo': [],
    }
    if not path.exists():
        return default_history
    try:
        with open(path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            for key, value in default_history.items():
                loaded.setdefault(key, value)
            return loaded
    except Exception:
        pass
    return default_history


def _step_scheduler_if_needed(scheduler, optimizer, cfg) -> None:
    if scheduler is None or bool(cfg.training.step_scheduler_per_batch):
        return
    scheduler.step()
    min_lr = float(cfg.training.min_lr)
    for group in optimizer.param_groups:
        group['lr'] = max(float(group['lr']), min_lr)


def _migrate_replay_buffer_if_needed(loaded_buffer, cfg, logger):
    replay_buffer = ReplayBuffer.from_serialized(loaded_buffer, cfg=cfg)
    logger.info(
        'Loaded replay buffer len=%s capacity=%s cfg_capacity=%s',
        len(replay_buffer),
        replay_buffer.capacity,
        cfg.replay.capacity,
    )
    return replay_buffer


def _prefill_replay_buffer_from_external(replay_buffer: ReplayBuffer, cfg, logger) -> None:
    external_path = str(getattr(cfg.training, 'external_samples_path', '') or '').strip()
    external_max = int(getattr(cfg.training, 'external_samples_max', 0) or 0)

    if not external_path:
        logger.info('External prefill disabled: cfg.training.external_samples_path is empty')
        return

    logger.info(
        'External prefill start path=%s max=%s buffer_before=%s',
        external_path,
        external_max if external_max > 0 else 'all',
        len(replay_buffer),
    )

    loaded_count = 0
    added_count = 0
    deduped_count = 0

    for state, policy, value in load_external_samples(external_path, cfg, max_samples=external_max):
        loaded_count += 1
        before_len = len(replay_buffer)
        replay_buffer.add(state, policy, value)
        if len(replay_buffer) > before_len:
            added_count += 1
        else:
            deduped_count += 1

    logger.info(
        'External prefill complete loaded=%s added=%s deduped=%s buffer_after=%s',
        loaded_count,
        added_count,
        deduped_count,
        len(replay_buffer),
    )


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    logger = setup_logging('training.loop')
    device = select_device(args.device or cfg.system.device)
    configure_torch_runtime(cfg, device=str(device), role='training', worker_count=1)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / 'best_model.pth'
    latest_ckpt = save_dir / 'latest_checkpoint.pth'
    buffer_path = save_dir / 'replay_buffer'
    history_path = save_dir / 'history.json'

    model = ChessNet(cfg).to(device)
    best_model = ChessNet(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=cfg.training.lr_decay_gamma,
    )
    scaler = GradScaler(enabled=bool(cfg.training.use_amp) and str(device).startswith('cuda'))
    replay_buffer = ReplayBuffer(cfg)
    global_step = 0
    best_elo = float(cfg.arena.initial_elo)
    learner_elo = best_elo
    history = _load_history(history_path)

    if args.resume and latest_ckpt.exists():
        state = load_checkpoint(
            latest_ckpt,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )
        global_step = int(state.get('global_step', 0))
        latest_meta = state.get('meta') or {}
        best_elo = float(latest_meta.get('best_elo', best_elo))
        learner_elo = float(latest_meta.get('learner_elo', best_elo))

        if best_path.exists():
            load_checkpoint(best_path, model=best_model, device=device)
        else:
            best_model.load_state_dict(copy.deepcopy(model.state_dict()))

        if buffer_path.exists():
            replay_buffer = ReplayBuffer.load_from_path(
                buffer_path,
                cfg=cfg,
                prompt_on_missing_shards=True,
            )
        else:
            logger.info(
                'Replay buffer file not found at %s -> starting with empty buffer capacity=%s',
                buffer_path,
                cfg.replay.capacity,
            )

        logger.info(
            'Resumed training from %s (global_step=%s)',
            latest_ckpt,
            global_step,
        )
    else:
        best_model.load_state_dict(copy.deepcopy(model.state_dict()))

    best_model.eval()

    logger.info(
        'Replay buffer status len=%s maxlen=%s cfg_capacity=%s',
        len(replay_buffer),
        replay_buffer.capacity,
        cfg.replay.capacity,
    )

    _prefill_replay_buffer_from_external(replay_buffer, cfg, logger)

    logger.info(
        'Training start iterations=%s device=%s save_dir=%s min_buffer_size=%s buffer_after_prefill=%s',
        args.iterations,
        device,
        save_dir,
        args.min_buffer_size,
        len(replay_buffer),
    )

    for iteration in range(1, args.iterations + 1):
        logger.info(
            'Iteration start iter=%s/%s buffer=%s global_step=%s',
            iteration,
            args.iterations,
            len(replay_buffer),
            global_step,
        )

        logger.info('Iteration %s: starting self-play', iteration)
        samples, stats = generate_self_play_data(model, device=device, cfg=cfg)
        logger.info('Iteration %s: finished self-play', iteration)

        if samples:
            before_buffer = len(replay_buffer)
            replay_buffer.save_game(samples)
            after_buffer = len(replay_buffer)
            logger.info(
                'Iteration %s: replay buffer update before=%s added=%s after=%s maxlen=%s',
                iteration,
                before_buffer,
                after_buffer - before_buffer,
                after_buffer,
                replay_buffer.capacity,
            )

        logger.info(
            'iter=%s selfplay_samples=%s buffer=%s',
            iteration,
            len(samples),
            len(replay_buffer),
        )

        trained_this_iteration = False
        train_stats = {
            'loss': None,
            'policy_loss': None,
            'value_loss': None,
            'steps': 0,
            'global_step': global_step,
        }

        if len(replay_buffer) >= args.min_buffer_size:
            logger.info(
                'Iteration %s: starting training step buffer=%s',
                iteration,
                len(replay_buffer),
            )
            train_stats = train_model(
                model=model,
                optimizer=optimizer,
                buffer=replay_buffer,
                device=device,
                scheduler=scheduler,
                global_step=global_step,
                scaler=scaler,
                cfg=cfg,
            )
            global_step = train_stats['global_step']
            trained_this_iteration = int(train_stats.get('steps', 0)) > 0
            _step_scheduler_if_needed(scheduler, optimizer, cfg)
            logger.info(
                'Iteration %s: finished training step loss=%s global_step=%s steps=%s',
                iteration,
                f"{train_stats['loss']:.6f}" if train_stats.get('loss') is not None else 'n/a',
                global_step,
                train_stats.get('steps', 0),
            )
        else:
            logger.info(
                'Iteration %s: skipped training buffer=%s < min_buffer_size=%s',
                iteration,
                len(replay_buffer),
                args.min_buffer_size,
            )

        arena_stats = {
            'win_rate': None,
            'scheduled_score_rate': None,
            'games_played': 0,
            'scheduled_games': int(cfg.arena.games),
            'accepted': False,
            'decision': 'skipped_no_training',
        }
        accept = False

        if trained_this_iteration:
            candidate_model = ChessNet(cfg).to(device)
            candidate_model.load_state_dict(copy.deepcopy(model.state_dict()))
            candidate_model.eval()

            logger.info('Iteration %s: starting arena evaluation', iteration)
            arena_stats = play_match(best_model, candidate_model, device=device, cfg=cfg)
            logger.info(
                'Iteration %s: finished arena evaluation games=%s win_rate=%.3f decision=%s',
                iteration,
                arena_stats.get('games_played', 0),
                float(arena_stats.get('win_rate', 0.0) or 0.0),
                arena_stats.get('decision'),
            )

            win_rate = float(arena_stats.get('win_rate', 0.0) or 0.0)
            updated_best_elo, updated_learner_elo = update_elo_pair(
                rating_a=best_elo,
                rating_b=learner_elo,
                score_a=1.0 - win_rate,
                k_factor=float(cfg.arena.elo_k_factor),
            )

            accept = bool(arena_stats.get('accepted', False))
            if accept:
                best_model.load_state_dict(candidate_model.state_dict())
                best_model.eval()
                best_elo = updated_learner_elo
                learner_elo = best_elo
                logger.info('New best model accepted')
            else:
                best_elo = updated_best_elo
                learner_elo = updated_learner_elo
                logger.info('Candidate rejected -> keeping learner for further training, champion unchanged')
        else:
            logger.info('Iteration %s: arena evaluation skipped because no optimization steps ran', iteration)

        history['iterations'].append(int(iteration))
        history['train_loss'].append(float(train_stats['loss']) if train_stats.get('loss') is not None else None)
        history['arena_win_rate'].append(float(arena_stats['win_rate']) if arena_stats.get('win_rate') is not None else None)
        history['scheduled_score_rate'].append(
            float(arena_stats['scheduled_score_rate'])
            if arena_stats.get('scheduled_score_rate') is not None else None
        )
        history['accepted'].append(bool(accept))
        history['champion_elo'].append(float(best_elo))
        history['learner_elo'].append(float(learner_elo))

        logger.info('Iteration %s: saving latest checkpoint -> %s', iteration, latest_ckpt)
        save_checkpoint(
            latest_ckpt,
            model,
            cfg=get_current_config(),
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            global_step=global_step,
            meta={
                'iteration': iteration,
                'best_elo': best_elo,
                'learner_elo': learner_elo,
                'accepted': bool(accept),
                'selfplay_stats': stats,
                'arena_stats': arena_stats,
            },
        )

        logger.info('Iteration %s: saving best checkpoint -> %s', iteration, best_path)
        save_checkpoint(
            best_path,
            best_model,
            cfg=get_current_config(),
            global_step=global_step,
            meta={'iteration': iteration, 'best_elo': best_elo},
        )

        should_save_buffer = (iteration % int(cfg.replay.save_interval) == 0) or (iteration == args.iterations)
        if should_save_buffer:
            logger.info('Iteration %s: saving replay buffer -> %s', iteration, buffer_path)
            replay_buffer.save(str(buffer_path))
        else:
            logger.info('Iteration %s: replay buffer save skipped (save_interval=%s)', iteration, int(cfg.replay.save_interval))

        logger.info('Iteration %s: writing history -> %s', iteration, history_path)
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2)

        logger.info('Iteration complete iter=%s/%s', iteration, args.iterations)

    logger.info('Training complete')


if __name__ == '__main__':
    mp.freeze_support()
    main()
