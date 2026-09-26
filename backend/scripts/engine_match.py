"""Paired engine matches for measuring search changes with one fixed checkpoint.

Each side is an engine subprocess with its own code tree (``--a-root``/``--b-root``),
config file and ``section.key=value`` overrides, so the same script compares two
revisions or two settings of one revision. Every opening is played twice with
colours reversed; several games run in parallel. Engines keep their search tree
between moves exactly as in play.

Example (from backend/):

    python scripts/engine_match.py --b-set mcts.penalty_mode=progressive \
        --games 40 --simulations 64 --parallel 3 --threads 3 --output data/evaluations/penalty_mode

The summary reports W/D/L for side B, the score, an Elo estimate with a 95%
interval from the game-pair (pentanomial) variance, and per-game PGNs.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import chess
import chess.pgn

BACKEND = Path(__file__).resolve().parents[1]

# Resign adjudication: both engines must agree the game is decided for this many
# consecutive moves each. Draw adjudication is not used; long games end at --max-plies.
RESIGN_VALUE = 0.92
RESIGN_MOVES = 3


# ----------------------------------------------------------------------------- engine side

def _parse_override(item: str):
    import yaml

    key, _, raw = item.partition("=")
    section, _, field = key.strip().partition(".")
    if not section or not field or not _:
        raise ValueError(f"override must look like section.key=value, got {item!r}")
    return section, field, yaml.safe_load(raw)


def engine_worker(args) -> None:
    from dataclasses import replace

    import numpy as np
    import torch

    from app.infra.config import load_config
    from app.core.engine import Engine

    torch.manual_seed(0)
    np.random.seed(0)
    cfg = load_config(args.config)
    for item in args.set or []:
        section, field, value = _parse_override(item)
        cfg = replace(cfg, **{section: replace(getattr(cfg, section), **{field: value})})
    engine = Engine(model_path=args.model, cfg=cfg, device="cpu", cache_size=0)
    torch.set_num_threads(max(1, int(args.threads)))
    print("READY", flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        if request.get("cmd") == "new_game":
            reset = getattr(engine.mcts, "reset_tree", None)
            if callable(reset):
                reset()
            print("OK", flush=True)
            continue
        board = chess.Board(request.get("fen") or chess.STARTING_FEN)
        for move in request["moves"]:
            board.push_uci(move)
        started = time.perf_counter()
        kwargs = {}
        if request.get("time_limit_sec"):
            kwargs["time_limit_sec"] = float(request["time_limit_sec"])
        sims = int(request["sims"])
        retained = 0
        if args.budget == "topup":
            # Equal effort: every move ends with the same number of root visits, whether
            # or not this engine keeps its tree between moves.
            probe = getattr(engine.mcts, "retained_visits", None)
            retained = int(probe(board)) if callable(probe) else 0
            sims = max(1, sims - retained)
        result = engine.analyze(board, num_simulations=sims, temperature=0.0, **kwargs)
        reply = {
            "move": result.best_move.uci() if result.best_move else None,
            "value": float(result.score),
            "visits": int(sum(result.visit_counts.values())),
            "retained": retained,
            "seconds": time.perf_counter() - started,
        }
        print("MOVE " + json.dumps(reply), flush=True)


class EngineProcess:
    def __init__(self, name: str, root: Path, config: str, overrides: list[str], model: str, threads: int,
                 log_path: Path, budget: str = "topup"):
        env = dict(os.environ, PYTHONPATH=str(root), PYTHONIOENCODING="utf-8")
        cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--engine-worker",
               "--config", config, "--model", model, "--threads", str(threads), "--budget", budget]
        for item in overrides:
            cmd += ["--set", item]
        self.name = name
        self.log = log_path.open("w", encoding="utf-8")
        self.proc = subprocess.Popen(cmd, cwd=str(root), env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=self.log, text=True, encoding="utf-8")
        self._expect("READY")

    def _expect(self, prefix: str) -> str:
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError(f"engine {self.name} exited with {self.proc.poll()} (see {self.log.name})")
            if line.startswith(prefix):
                return line

    def new_game(self) -> None:
        self.proc.stdin.write(json.dumps({"cmd": "new_game"}) + "\n")
        self.proc.stdin.flush()
        self._expect("OK")

    def move(self, start_fen: str, moves: list[str], sims: int, time_limit: float | None) -> dict:
        request = {"fen": start_fen, "moves": moves, "sims": sims}
        if time_limit:
            request["time_limit_sec"] = time_limit
        self.proc.stdin.write(json.dumps(request) + "\n")
        self.proc.stdin.flush()
        return json.loads(self._expect("MOVE ")[5:])

    def close(self) -> None:
        try:
            self.proc.terminate()
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()
        self.log.close()


# ----------------------------------------------------------------------------- match side

def load_openings(path: str | None, count: int) -> list[list[str]]:
    """UCI opening lines. Default: 8-ply lines from the local GM PGN collection, first unique ones."""
    if path and Path(path).suffix == ".json":
        return json.loads(Path(path).read_text(encoding="utf-8"))[:count]
    pgn_path = Path(path) if path else BACKEND / "data" / "downloads" / "all_grandmasters.pgn"
    lines, seen = [], set()
    with pgn_path.open(encoding="utf-8", errors="replace") as handle:
        while len(lines) < count:
            game = chess.pgn.read_game(handle)
            if game is None:
                break
            board = game.board()
            moves = []
            for move in game.mainline_moves():
                board.push(move)
                moves.append(move.uci())
                if len(moves) == 8:
                    break
            key = " ".join(moves)
            if len(moves) == 8 and key not in seen and not board.is_game_over():
                seen.add(key)
                lines.append(moves)
    return lines


def _game_finished(board: chess.Board) -> bool:
    """A real ending only. A draw that the side to move could merely claim is its own
    decision (python-chess's claim_draw=True would end the game on its behalf)."""
    return board.is_game_over(claim_draw=False) or board.is_repetition(3) or board.halfmove_clock >= 100


def play_game(white: EngineProcess, black: EngineProcess, opening: list[str], args) -> dict:
    board = chess.Board()
    for uci in opening:
        board.push_uci(uci)
    white.new_game()
    black.new_game()
    engines = {chess.WHITE: white, chess.BLACK: black}
    # Each engine reports its root value from its own (the mover's) perspective.
    white_view_streak = {chess.WHITE: 0, chess.BLACK: 0}
    last_white_view = {chess.WHITE: 0.0, chess.BLACK: 0.0}
    seconds = {white.name: 0.0, black.name: 0.0}
    visits = {white.name: [], black.name: []}
    adjudicated = None
    node_moves = []
    while not _game_finished(board) and board.ply() < args.max_plies:
        side = board.turn
        engine = engines[side]
        reply = engine.move(chess.STARTING_FEN, [m.uci() for m in board.move_stack], args.simulations, args.time_limit)
        move = chess.Move.from_uci(reply["move"])
        if move not in board.legal_moves:
            raise RuntimeError(f"illegal move {move} from {engine.name}")
        seconds[engine.name] += reply["seconds"]
        visits[engine.name].append(reply["visits"])
        node_moves.append((move, reply))
        board.push(move)
        white_view = float(reply["value"]) if side == chess.WHITE else -float(reply["value"])
        decided = abs(white_view) >= RESIGN_VALUE and (
            white_view_streak[side] == 0 or (white_view > 0) == (last_white_view[side] > 0)
        )
        white_view_streak[side] = white_view_streak[side] + 1 if decided else 0
        last_white_view[side] = white_view
        if (white_view_streak[chess.WHITE] >= RESIGN_MOVES and white_view_streak[chess.BLACK] >= RESIGN_MOVES
                and (last_white_view[chess.WHITE] > 0) == (last_white_view[chess.BLACK] > 0)):
            # Both engines have called the same winner for several consecutive moves.
            adjudicated = chess.WHITE if last_white_view[chess.WHITE] > 0 else chess.BLACK
            break
    if adjudicated is not None:
        result = "1-0" if adjudicated == chess.WHITE else "0-1"
        termination = "ADJUDICATED_RESIGN"
    else:
        outcome = board.outcome(claim_draw=False)
        if outcome is not None:
            result, termination = outcome.result(), outcome.termination.name
        elif board.is_repetition(3):
            result, termination = "1/2-1/2", "THREEFOLD_REPETITION"
        elif board.halfmove_clock >= 100:
            result, termination = "1/2-1/2", "FIFTY_MOVES"
        else:
            result, termination = "1/2-1/2", "MAX_PLIES"
    game = chess.pgn.Game()
    game.headers.update(Event="engine_match", White=white.name, Black=black.name, Result=result,
                        Termination=termination, SimulationBudget=str(args.simulations))
    node = game
    replay = chess.Board()
    for uci in opening:
        node = node.add_variation(chess.Move.from_uci(uci))
        replay.push_uci(uci)
    for move, reply in node_moves:
        node = node.add_variation(move)
        node.comment = f"v={reply['value']:+.2f} n={reply['visits']} r={reply.get('retained', 0)} t={reply['seconds']:.2f}"
    return {
        "white": white.name, "black": black.name, "result": result, "termination": termination,
        "plies": board.ply(), "seconds": seconds,
        "mean_visits": {k: (sum(v) / len(v) if v else 0.0) for k, v in visits.items()},
        "pgn": str(game),
    }


def score_for(name: str, row: dict) -> float:
    if row["result"] == "1/2-1/2":
        return 0.5
    white_won = row["result"] == "1-0"
    return 1.0 if (row["white"] == name) == white_won else 0.0


def elo_summary(rows: list[dict], name: str) -> dict:
    scores = [score_for(name, row) for row in rows]
    n = len(scores)
    wins, draws = scores.count(1.0), scores.count(0.5)
    losses = n - wins - draws
    mean = sum(scores) / n if n else 0.5
    # Pentanomial variance: games are paired by opening, colours reversed.
    pairs = {}
    for row, score in zip(rows, scores):
        pairs.setdefault(row["pair"], []).append(score)
    pair_scores = [sum(v) / len(v) for v in pairs.values() if len(v) == 2]
    if len(pair_scores) >= 2:
        pm = sum(pair_scores) / len(pair_scores)
        var = sum((s - pm) ** 2 for s in pair_scores) / (len(pair_scores) - 1)
        stderr = math.sqrt(var / len(pair_scores))
    else:
        var = sum((s - mean) ** 2 for s in scores) / max(1, n - 1)
        stderr = math.sqrt(var / max(1, n))

    def to_elo(p: float) -> float:
        p = min(max(p, 1e-3), 1 - 1e-3)
        return -400.0 * math.log10(1.0 / p - 1.0)

    return {
        "games": n, "wins": wins, "draws": draws, "losses": losses, "score": round(mean, 4),
        "elo": round(to_elo(mean), 1),
        "elo_95": [round(to_elo(mean - 1.96 * stderr), 1), round(to_elo(mean + 1.96 * stderr), 1)],
        "complete_pairs": len(pair_scores),
    }


def run_match(args) -> None:
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    openings = load_openings(args.openings, (args.games + 1) // 2)
    jobs = []
    for index, opening in enumerate(openings):
        jobs.append((index, opening, "A", "B"))
        jobs.append((index, opening, "B", "A"))
    jobs = jobs[: args.games]
    results_path = out / "results.jsonl"
    done = set()
    rows = []
    if results_path.exists() and not args.fresh:
        for line in results_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            rows.append(row)
            done.add((row["pair"], row["white"]))
    pending = [job for job in jobs if (job[0], job[2]) not in done]
    lock = threading.Lock()
    queue = list(pending)

    def worker(slot: int) -> None:
        a = EngineProcess("A", Path(args.a_root), args.a_config, args.a_set, args.a_model or args.model, args.threads,
                          out / f"engine_A_{slot}.log", args.budget)
        b = EngineProcess("B", Path(args.b_root), args.b_config, args.b_set, args.b_model or args.model, args.threads,
                          out / f"engine_B_{slot}.log", args.budget)
        try:
            while True:
                with lock:
                    if not queue:
                        return
                    pair, opening, white_name, _ = queue.pop(0)
                white, black = (a, b) if white_name == "A" else (b, a)
                row = play_game(white, black, opening, args)
                row["pair"] = pair
                row["opening"] = " ".join(opening)
                with lock:
                    rows.append(row)
                    with results_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(row) + "\n")
                    summary = elo_summary(rows, "B")
                    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
                    print(f"[{len(rows)}/{len(jobs)}] pair {pair} {row['white']}-{row['black']} {row['result']} "
                          f"{row['termination']} {row['plies']} plies | B: +{summary['wins']} ={summary['draws']} "
                          f"-{summary['losses']} score {summary['score']:.3f} elo {summary['elo']:+.0f} "
                          f"{summary['elo_95']}", flush=True)
        finally:
            a.close()
            b.close()

    threads = [threading.Thread(target=worker, args=(slot,), daemon=True) for slot in range(max(1, args.parallel))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    summary = elo_summary(rows, "B")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (out / "games.pgn").open("w", encoding="utf-8") as handle:
        for row in sorted(rows, key=lambda r: (r["pair"], r["white"])):
            handle.write(row["pgn"] + "\n\n")
    print("SUMMARY " + json.dumps(summary), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engine-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", help=argparse.SUPPRESS)
    parser.add_argument("--config", help=argparse.SUPPRESS)
    parser.add_argument("--a-root", default=str(BACKEND), help="backend directory of engine A (default: this one)")
    parser.add_argument("--b-root", default=str(BACKEND), help="backend directory of engine B (default: this one)")
    parser.add_argument("--a-config", default="config/default.yaml")
    parser.add_argument("--b-config", default="config/default.yaml")
    parser.add_argument("--a-set", action="append", default=[], help="override for A, e.g. mcts.penalty_mode=offset")
    parser.add_argument("--b-set", action="append", default=[], help="override for B")
    parser.add_argument("--model", default=str((BACKEND / "models" / "best_model.pth").resolve()))
    parser.add_argument("--a-model", default=None, help="checkpoint for A only (default: --model), to compare weights")
    parser.add_argument("--b-model", default=None, help="checkpoint for B only (default: --model)")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--time-limit", type=float, default=None, help="optional per-move deadline in seconds")
    parser.add_argument("--budget", choices=["topup", "extra"], default="topup",
                        help="topup: each move ends with --simulations root visits including retained ones "
                             "(equal effort); extra: --simulations new rollouts on top of the retained tree")
    parser.add_argument("--max-plies", type=int, default=260)
    parser.add_argument("--parallel", type=int, default=2, help="games played at the same time")
    parser.add_argument("--threads", type=int, default=3, help="torch threads per engine process")
    parser.add_argument("--openings", default=None, help="PGN or JSON list of UCI lines (default: local GM games)")
    parser.add_argument("--output", default="data/evaluations/engine_match")
    parser.add_argument("--fresh", action="store_true", help="ignore results already in --output")
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    if arguments.engine_worker:
        engine_worker(arguments)
    else:
        run_match(arguments)
