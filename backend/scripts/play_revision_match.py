"""Play paired local revision matches using one checkpoint and equal simulations."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import chess
import chess.pgn


def worker(args):
    import numpy as np
    import torch
    from app.core.engine import Engine
    from app.infra.config import load_config
    torch.manual_seed(0)
    np.random.seed(0)
    engine = Engine(model_path=args.model, cfg=load_config(args.config), device='cpu', cache_size=0)
    for line in sys.stdin:
        request = json.loads(line)
        board = chess.Board()
        for move in request['moves']:
            board.push_uci(move)
        start = time.perf_counter()
        result = engine.analyze(board, num_simulations=args.simulations, temperature=0.05)
        print('MOVE ' + json.dumps({'move': result.best_move.uci() if result.best_move else None,
                                    'seconds': time.perf_counter() - start}), flush=True)


def main(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    processes = {}
    logs = []
    results = []
    try:
        for name, root in [('before', args.before), ('after', str(Path.cwd()))]:
            env = dict(os.environ, PYTHONPATH=root, PYTHONIOENCODING='utf-8')
            log = (out / (name + '.log')).open('w', encoding='utf-8')
            logs.append(log)
            processes[name] = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()),
                '--worker', '--model', args.model, '--config', args.config,
                '--simulations', str(args.simulations)], env=env, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=log, text=True, encoding='utf-8')
        openings = ['e2e4 e7e5 g1f3 b8c6', 'd2d4 d7d5 c2c4 e7e6']
        for index in range(4):
            white = 'after' if index % 2 == 0 else 'before'
            black = 'before' if white == 'after' else 'after'
            board = chess.Board()
            game = chess.pgn.Game()
            game.headers.update(Event='Local revision comparison', White=white, Black=black,
                                TimeControl='-', SimulationBudget=str(args.simulations))
            node = game
            for move in openings[index // 2].split():
                board.push_uci(move)
                node = node.add_variation(board.peek())
            timings = {'before': 0.0, 'after': 0.0}
            while not board.is_game_over(claim_draw=True) and board.ply() < args.max_plies:
                name = white if board.turn else black
                proc = processes[name]
                proc.stdin.write(json.dumps({'moves': [m.uci() for m in board.move_stack]}) + '\n')
                proc.stdin.flush()
                while True:
                    line = proc.stdout.readline()
                    if not line:
                        raise RuntimeError(f'{name} worker exited: {proc.poll()}')
                    if line.startswith('MOVE '):
                        reply = json.loads(line[5:])
                        break
                move = chess.Move.from_uci(reply['move'])
                if move not in board.legal_moves:
                    raise RuntimeError(f'Illegal move {move} from {name}')
                timings[name] += reply['seconds']
                board.push(move)
                node = node.add_variation(move)
                node.comment = f"{reply['seconds']:.3f} seconds"
                (out / f'game_{index + 1}.pgn').write_text(str(game), encoding='utf-8')
                print(json.dumps({'game': index + 1, 'ply': board.ply(), 'side': name, **reply}), flush=True)
            outcome = board.outcome(claim_draw=True)
            result = outcome.result() if outcome else '*'
            termination = outcome.termination.name if outcome else 'PLY_LIMIT_UNFINISHED'
            game.headers['Result'] = result
            game.headers['Termination'] = termination
            (out / f'game_{index + 1}.pgn').write_text(str(game), encoding='utf-8')
            row = dict(game=index + 1, white=white, black=black, result=result,
                       termination=termination, plies=board.ply(), seconds=timings)
            results.append(row)
            (out / 'results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
            print('RESULT ' + json.dumps(row), flush=True)
    finally:
        for proc in processes.values():
            proc.terminate()
            proc.wait()
        for log in logs:
            log.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--before')
    parser.add_argument('--model', default=str(Path('models/best_model.pth').resolve()))
    parser.add_argument('--config', default=str(Path('config/default.yaml').resolve()))
    parser.add_argument('--simulations', type=int, default=64)
    parser.add_argument('--max-plies', type=int, default=200)
    parser.add_argument('--output', default='data/evaluations/local_revision_match')
    arguments = parser.parse_args()
    worker(arguments) if arguments.worker else main(arguments)
