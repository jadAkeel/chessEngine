from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from app.infra.logging import setup_logging

logger = setup_logging("model.checkpoint")


class CheckpointLoadError(RuntimeError):
    """Raised when a checkpoint exists but cannot be safely loaded."""


@dataclass(frozen=True)
class CompatibleWeightReport:
    matched_keys: tuple[str, ...]
    skipped_keys: tuple[str, ...]
    missing_model_keys: tuple[str, ...]

    @property
    def matched_ratio(self) -> float:
        total = len(self.matched_keys) + len(self.missing_model_keys)
        if total <= 0:
            return 0.0
        return len(self.matched_keys) / total

    @property
    def can_load(self) -> bool:
        return bool(self.matched_keys)

    def summary(self) -> str:
        return (
            f"matched={len(self.matched_keys)} "
            f"missing={len(self.missing_model_keys)} "
            f"skipped={len(self.skipped_keys)} "
            f"ratio={self.matched_ratio:.2%}"
        )



def save_model(model, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)



def save_checkpoint(
    path: str | Path,
    model,
    cfg,
    optimizer=None,
    scheduler=None,
    scaler=None,
    global_step: int = 0,
    meta: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_state_dict": model.state_dict(),
        "config": cfg,
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "global_step": int(global_step),
        "meta": meta or {},
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }

    torch.save(payload, path)



def _extract_state_dict(payload):
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"]
    return payload



def inspect_compatible_weights(model, path: str | Path, device: str = "cpu") -> CompatibleWeightReport:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    state = _extract_state_dict(torch.load(path, map_location=device, weights_only=False))
    if not isinstance(state, dict):
        raise CheckpointLoadError(f"Checkpoint payload is not a state dict: {path}")

    current = model.state_dict()
    matched_keys: list[str] = []
    skipped_keys: list[str] = []

    for key, value in state.items():
        if key not in current:
            skipped_keys.append(key)
            continue
        if not hasattr(value, "shape") or current[key].shape != value.shape:
            skipped_keys.append(key)
            continue
        matched_keys.append(key)

    missing_model_keys = [key for key in current.keys() if key not in matched_keys]
    return CompatibleWeightReport(
        matched_keys=tuple(sorted(matched_keys)),
        skipped_keys=tuple(sorted(skipped_keys)),
        missing_model_keys=tuple(sorted(missing_model_keys)),
    )



def load_compatible_weights(
    model,
    path: str | Path,
    device: str = "cpu",
    *,
    min_match_ratio: float = 0.0,
    raise_on_mismatch: bool = False,
) -> bool:
    path = Path(path)
    if not path.exists():
        if raise_on_mismatch:
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return False

    report = inspect_compatible_weights(model, path, device=device)
    if not report.can_load or report.matched_ratio < float(min_match_ratio):
        message = f"Incompatible checkpoint for partial load: {path} ({report.summary()})"
        if raise_on_mismatch:
            raise CheckpointLoadError(message)
        logger.warning(message)
        return False

    state = _extract_state_dict(torch.load(path, map_location=device, weights_only=False))
    current = model.state_dict()
    filtered = {key: state[key] for key in report.matched_keys}
    current.update(filtered)
    model.load_state_dict(current, strict=False)
    logger.info("Loaded compatible weights from %s (%s)", path, report.summary())
    return True



def load_checkpoint(
    path: str | Path,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    device: str = "cpu",
    restore_rng: bool = False,
):
    path = Path(path)

    if not path.exists():
        return {
            "loaded": False,
            "global_step": 0,
            "meta": {},
            "config": None,
        }

    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise CheckpointLoadError(f"Checkpoint payload must be a dict: {path}")

    state = _extract_state_dict(payload)
    if not isinstance(state, dict):
        raise CheckpointLoadError(f"Checkpoint state dict missing or invalid: {path}")

    try:
        model.load_state_dict(state, strict=False)
    except RuntimeError as exc:
        raise CheckpointLoadError(f"Checkpoint incompatible with model for {path}: {exc}") from exc

    if optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])

    if scheduler is not None and payload.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])

    if scaler is not None and payload.get("scaler_state_dict") is not None:
        scaler.load_state_dict(payload["scaler_state_dict"])

    if restore_rng:
        rng_state = payload.get("rng_state") or {}

        if rng_state.get("python") is not None:
            random.setstate(rng_state["python"])

        if rng_state.get("numpy") is not None:
            np.random.set_state(rng_state["numpy"])

        if rng_state.get("torch") is not None:
            torch.set_rng_state(rng_state["torch"])

        if torch.cuda.is_available() and rng_state.get("cuda") is not None:
            torch.cuda.set_rng_state_all(rng_state["cuda"])

    logger.info("Loaded checkpoint from %s", path)
    return {
        "loaded": True,
        "global_step": int(payload.get("global_step", 0)),
        "meta": payload.get("meta") or {},
        "config": payload.get("config"),
    }
