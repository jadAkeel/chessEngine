"""Offline, reproducible checkpoint/search probes; never submits online moves."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import chess
import numpy as np
import torch

from app.core.engine import Engine
from app.infra.config import load_config
from app.mcts.search import MCTS
from app.api.main import _move_allows_forced_mate_in_two, _move_allows_valuable_piece_capture


POSITIONS = {
    "opening": chess.STARTING_FEN,
    "reported_mate_two": "8/5pk1/3p1n2/3Br3/5Qp1/6K1/8/7R b - - 5 36",
    "free_queen": "4k3/8/8/8/3q4/8/3R4/4K3 w - - 0 1",
    "mate_one": "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq g3 0 2",
    "live_13": "r1b2rk1/pp2nppp/4p3/3pP3/1q6/1P1B1Q2/P2N1PPP/R3K2R w KQ - 3 13",
    "live_19": "r1b2rk1/pp3ppp/3q2n1/3pp3/P7/1P1BR2Q/5PPP/R5K1 w - - 0 19",
    "live_25": "5rk1/pp3ppp/6q1/3p4/P3pn2/1PR5/5PPP/1B2R1K1 w - - 1 25",
}


class ObservedMCTS(MCTS):
    def _expand_node_from_prediction(self, node, board, policy_logits, nn_value, add_noise):
        self.leaf_depths.append(board.ply() - self.root_ply)
        return super()._expand_node_from_prediction(node, board, policy_logits, nn_value, add_noise)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--model-path", default="models/best_model.pth")
    parser.add_argument("--simulations", type=int, default=128)
    parser.add_argument("--variants", nargs="+", choices=["default", "no_principles", "batch_one"], default=["default"])
    parser.add_argument("--positions", nargs="+", choices=list(POSITIONS), default=list(POSITIONS))
    parser.add_argument("--time-limit", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    engine = Engine(model_path=args.model_path, cfg=load_config(args.config), device="cpu", cache_size=0)
    original_cfg = engine.cfg
    rows = []
    for variant in args.variants:
        cfg = original_cfg
        if variant == "no_principles":
            cfg = replace(cfg, principle_penalties=replace(cfg.principle_penalties, enabled=False))
        elif variant == "batch_one":
            cfg = replace(cfg, mcts=replace(cfg.mcts, inference_batch_size=1))
        engine.cfg = cfg
        engine.mcts = ObservedMCTS(engine.model, cfg=cfg, device="cpu")
        for name in args.positions:
            board = chess.Board(POSITIONS[name])
            engine.mcts.root_ply = board.ply()
            engine.mcts.leaf_depths = []
            kwargs = {} if args.time_limit is None else {"time_limit_sec": args.time_limit}
            start = time.perf_counter()
            result = engine.analyze(board, num_simulations=args.simulations, temperature=0.05, **kwargs)
            row = {
                "variant": variant, "position": name, "fen": board.fen(),
                "seconds": round(time.perf_counter() - start, 4),
                "move": result.best_move.uci() if result.best_move else None,
                "score": result.score, "visits": sum(result.visit_counts.values()),
                "max_evaluated_depth": max(engine.mcts.leaf_depths, default=0),
                "mean_evaluated_depth": float(np.mean(engine.mcts.leaf_depths)) if engine.mcts.leaf_depths else 0.0,
                "top_visits": sorted(result.visit_counts.items(), key=lambda item: -item[1])[:5],
                "allows_short_mate": _move_allows_forced_mate_in_two(board, result.best_move.uci()) if result.best_move else False,
                "allows_valuable_capture": _move_allows_valuable_piece_capture(board, result.best_move.uci()) if result.best_move else False,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    with open(args.model_path, "rb") as handle:
        checkpoint_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    report = {"checkpoint_sha256": checkpoint_hash, "torch": torch.__version__, "device": "cpu",
              "threads": torch.get_num_threads(), "simulations": args.simulations,
              "time_limit": args.time_limit, "results": rows}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
