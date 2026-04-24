from __future__ import annotations

import argparse

from app.cli.common import add_common_runtime_args, configure_runtime
from app.evaluation.arena import benchmark_vs_search
from app.model.checkpoint import CheckpointLoadError, load_checkpoint, load_compatible_weights
from app.model.network import ChessNet



def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate checkpoint against search baseline')
    add_common_runtime_args(parser, require_model_path=True)
    parser.add_argument('--games', type=int, default=None)
    parser.add_argument('--depth', type=int, default=None)
    args = parser.parse_args()

    cfg, logger, device = configure_runtime(args, 'cli.evaluate')
    model = ChessNet(cfg).to(device)

    loaded = False
    try:
        state = load_checkpoint(args.model_path, model=model, device=device)
        loaded = bool(state.get('loaded', False))
    except CheckpointLoadError:
        loaded = load_compatible_weights(model, args.model_path, device=device, min_match_ratio=0.95, raise_on_mismatch=True)

    if not loaded:
        raise RuntimeError(f'Failed to load model checkpoint: {args.model_path}')

    model.eval()
    logger.info('Running benchmark for %s', args.model_path)
    print(benchmark_vs_search(model, device=device, depth=args.depth, games=args.games))


if __name__ == '__main__':
    main()
