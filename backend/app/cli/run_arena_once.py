from __future__ import annotations

import argparse
from pathlib import Path

import torch

from app.cli.common import add_common_runtime_args, configure_runtime
from app.evaluation.arena import play_match
from app.model.network import ChessNet


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        return (
            ckpt.get("model_state_dict")
            or ckpt.get("state_dict")
            or ckpt.get("model")
            or ckpt
        )
    return ckpt


def _load_model(model_path: str, cfg, device: str):
    model = ChessNet(cfg)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = _extract_state_dict(ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.cfg = cfg
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one arena match between two checkpoints")
    add_common_runtime_args(parser)
    parser.add_argument("--best-model-path", required=True, help="Path to champion/best model checkpoint")
    parser.add_argument("--candidate-model-path", required=True, help="Path to candidate/other model checkpoint")
    args = parser.parse_args()

    cfg, logger, device = configure_runtime(args, "cli.run_arena_once")

    if not Path(args.best_model_path).exists():
        raise FileNotFoundError(f"Best model not found: {args.best_model_path}")
    if not Path(args.candidate_model_path).exists():
        raise FileNotFoundError(f"Candidate model not found: {args.candidate_model_path}")

    logger.info("Loading best model from %s", args.best_model_path)
    best_model = _load_model(args.best_model_path, cfg, device)

    logger.info("Loading candidate model from %s", args.candidate_model_path)
    candidate_model = _load_model(args.candidate_model_path, cfg, device)

    result = play_match(best_model, candidate_model, device=device, cfg=cfg)

    print("\n=== ARENA RESULT ===")
    print(result)


if __name__ == "__main__":
    main()