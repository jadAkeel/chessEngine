from __future__ import annotations

import argparse

import chess

from app.cli.common import add_common_runtime_args, configure_runtime
from app.core.engine import Engine



def main() -> None:
    parser = argparse.ArgumentParser(description='Play against the engine in terminal')
    add_common_runtime_args(parser)
    args = parser.parse_args()

    cfg, logger, device = configure_runtime(args, 'cli.play')
    engine = Engine(model_path=args.model_path, cfg=cfg, device=device, allow_partial_weights=bool(args.model_path))
    board = chess.Board()
    logger.info('Interactive game started on %s', device)
    print('You are White. Enter moves in UCI format such as e2e4.')
    while not board.is_game_over(claim_draw=False):
        print(board)
        if board.turn == chess.WHITE:
            try:
                move = chess.Move.from_uci(input('Your move: ').strip())
            except ValueError:
                print('Invalid move format.')
                continue
            if move not in board.legal_moves:
                print('Illegal move.')
                continue
        else:
            move = engine.get_best_move(board, add_noise=False, temperature=0.0)
            print('Engine:', move)
        board.push(move)
    print(board.outcome(claim_draw=False))


if __name__ == '__main__':
    main()
