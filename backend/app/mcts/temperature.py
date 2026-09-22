from __future__ import annotations

import chess
import numpy as np


def _stable_policy_from_visits(visits: np.ndarray, temperature: float) -> np.ndarray:
    visits = np.asarray(visits, dtype=np.float64)
    if visits.size == 0:
        return visits

    if temperature <= 1e-6:
        probs = np.zeros_like(visits, dtype=np.float64)
        probs[int(np.argmax(visits))] = 1.0
        return probs

    safe_visits = np.maximum(visits, 1e-12)
    logits = np.log(safe_visits) / max(float(temperature), 1e-6)
    logits = logits - float(np.max(logits))
    probs = np.exp(logits)
    total = float(np.sum(probs))
    if total <= 0.0 or not np.isfinite(total):
        return np.full_like(safe_visits, 1.0 / len(safe_visits), dtype=np.float64)
    return probs / total


def apply_temperature(visit_counts: dict[chess.Move, int], temperature: float) -> dict[chess.Move, float]:
    """Convert visit counts into a probability distribution using a numerically stable temperature transform."""
    if not visit_counts:
        return {}

    moves = list(visit_counts.keys())
    visits = np.array([float(visit_counts[m]) for m in moves], dtype=np.float64)
    probs = _stable_policy_from_visits(visits, temperature)
    return {move: float(p) for move, p in zip(moves, probs)}
