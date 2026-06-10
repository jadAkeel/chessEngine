from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import chess

from app.infra.config import AppConfig, get_current_config, validate_config
from app.infra.device import get_default_device
from app.infra.logging import setup_logging
from app.infra.runtime import configure_torch_runtime
from app.mcts.search import MCTS
from app.model.checkpoint import CheckpointLoadError, load_checkpoint, load_compatible_weights
from app.model.network import ChessNet

logger = setup_logging("core.engine")


@dataclass(frozen=True)
class AnalysisResult:
    best_move: chess.Move | None
    score: float
    visit_counts: dict[str, int]
    policy: dict[str, float]
    penalty_diagnostics: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "best_move": self.best_move.uci() if self.best_move is not None else None,
            "score": self.score,
            "visit_counts": dict(self.visit_counts),
            "policy": dict(self.policy),
        }
        if self.penalty_diagnostics is not None:
            payload["penalty_diagnostics"] = dict(self.penalty_diagnostics)
        return payload


class Engine:
    def __init__(
        self,
        model_path: str | None = None,
        model=None,
        cfg: AppConfig | None = None,
        device: str | None = None,
        *,
        mcts_factory: Callable[..., MCTS] = MCTS,
        allow_partial_weights: bool = False,
        cache_size: int = 128,
    ):
        self.device = get_default_device(device)
        self.cfg = self._resolve_config(model=model, cfg=cfg)
        validate_config(self.cfg)
        configure_torch_runtime(self.cfg, device=str(self.device), role='engine', worker_count=1)
        self.model = model or ChessNet(self.cfg)
        self.cache_size = max(0, int(cache_size))
        self._analysis_cache: OrderedDict[tuple[str, int, float], AnalysisResult] = OrderedDict()

        if model_path:
            self._load_model_weights(model_path=model_path, allow_partial_weights=allow_partial_weights)

        self.model.to(self.device)
        self.model.eval()
        self.mcts = mcts_factory(self.model, cfg=self.cfg, device=self.device)

    @staticmethod
    def _resolve_config(model=None, cfg: AppConfig | None = None) -> AppConfig:
        model_cfg = getattr(model, "cfg", None)
        resolved = cfg or model_cfg or get_current_config()
        if cfg is not None and model_cfg is not None and cfg != model_cfg:
            raise ValueError("Provided cfg does not match model.cfg")
        return resolved

    def _load_model_weights(self, model_path: str, allow_partial_weights: bool) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        try:
            data = load_checkpoint(path, self.model, device=self.device)
            checkpoint_cfg = data.get("config")
            if checkpoint_cfg is not None:
                self.cfg = checkpoint_cfg
                setattr(self.model, "cfg", checkpoint_cfg)
            return
        except CheckpointLoadError as exc:
            if not allow_partial_weights:
                logger.exception("Failed to load checkpoint strictly from %s", path)
                raise RuntimeError(f"Model loading failed for {path}") from exc
            logger.warning("Strict checkpoint load failed for %s, trying partial weights: %s", path, exc)

        loaded = load_compatible_weights(
            self.model,
            path,
            self.device,
            min_match_ratio=0.95,
            raise_on_mismatch=True,
        )
        if not loaded:
            raise RuntimeError(f"Partial checkpoint load failed for {path}")

    def new_board(self) -> chess.Board:
        return chess.Board()

    def _validate_board(self, board: chess.Board) -> chess.Board:
        if not isinstance(board, chess.Board):
            raise TypeError("Expected board to be an instance of chess.Board")
        return board

    def _resolve_simulations(self, num_simulations: int | None) -> int:
        resolved = int(num_simulations if num_simulations is not None else self.cfg.mcts.num_simulations)
        if resolved <= 0:
            raise ValueError("num_simulations must be greater than 0")
        return resolved

    @staticmethod
    def _cache_key(board: chess.Board, num_simulations: int, temperature: float) -> tuple[str, int, float]:
        return (board.fen(), int(num_simulations), float(temperature))

    def _cache_get(self, key: tuple[str, int, float]) -> AnalysisResult | None:
        if self.cache_size <= 0:
            return None
        result = self._analysis_cache.get(key)
        if result is not None:
            self._analysis_cache.move_to_end(key)
        return result

    def _cache_put(self, key: tuple[str, int, float], value: AnalysisResult) -> None:
        if self.cache_size <= 0:
            return
        self._analysis_cache[key] = value
        self._analysis_cache.move_to_end(key)
        while len(self._analysis_cache) > self.cache_size:
            self._analysis_cache.popitem(last=False)

    def analyze(
        self,
        board,
        add_noise: bool = False,
        num_simulations: int | None = None,
        temperature: float = 1.0,
    ) -> AnalysisResult:
        board = self._validate_board(board)
        sims = self._resolve_simulations(num_simulations)
        cache_key = None if add_noise else self._cache_key(board, sims, temperature)

        if cache_key is not None:
            cached = self._cache_get(cache_key)
            if cached is not None:
                return cached

        result = self.mcts.search(board=board, add_noise=add_noise, num_simulations=sims, temperature=temperature)
        effective_policy = result.get("adjusted_policy_target") or result["policy_target"]
        analysis = AnalysisResult(
            best_move=result["best_move"],
            score=float(result["root_value"]),
            visit_counts={move.uci(): int(count) for move, count in result["visit_counts"].items()},
            policy={move.uci(): float(prob) for move, prob in effective_policy.items()},
            penalty_diagnostics=result.get("penalty_diagnostics"),
        )

        if cache_key is not None:
            self._cache_put(cache_key, analysis)
        return analysis

    def get_best_move(
        self,
        board,
        add_noise: bool = False,
        num_simulations: int | None = None,
        temperature: float = 1.0,
        return_value: bool = False,
    ):
        result = self.analyze(board, add_noise=add_noise, num_simulations=num_simulations, temperature=temperature)
        move = result.best_move
        if return_value:
            return move, result.score
        return move
