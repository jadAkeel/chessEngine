from __future__ import annotations

import argparse

from app.cli.common import add_common_runtime_args, configure_runtime
from app.model.checkpoint import CheckpointLoadError, load_checkpoint, load_compatible_weights
from app.model.network import ChessNet
from app.selfplay.generator import generate_self_play_data



def main() -> None:
    parser = argparse.ArgumentParser(description='Generate self-play data')
    add_common_runtime_args(parser)
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--games-per-worker', type=int, default=None)
    args = parser.parse_args()

    cfg, logger, device = configure_runtime(args, 'cli.selfplay')
    model = ChessNet(cfg).to(device)
    if args.model_path:
        try:
            load_checkpoint(args.model_path, model=model, device=device)
        except CheckpointLoadError:
            load_compatible_weights(model, args.model_path, device=device, min_match_ratio=0.95, raise_on_mismatch=True)
    samples, stats = generate_self_play_data(model, device=device, num_workers=args.workers, games_per_worker=args.games_per_worker)
    logger.info('Generated %s samples', len(samples))
    print({'samples': len(samples), 'stats': stats})


if __name__ == '__main__':
    main()
