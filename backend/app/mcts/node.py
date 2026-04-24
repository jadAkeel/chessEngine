from __future__ import annotations

from typing import Dict, Optional

import chess


class Node:
    def __init__(self, prior: float, parent: Optional["Node"] = None):
        self.prior: float = float(prior)
        self.parent: Optional["Node"] = parent
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.virtual_visits: int = 0
        self.children: Dict[chess.Move, "Node"] = {}

    def expanded(self) -> bool:
        return len(self.children) > 0

    @property
    def total_visit_count(self) -> int:
        return int(self.visit_count + self.virtual_visits)

    @property
    def q(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def add_virtual_visit(self, count: int = 1) -> None:
        self.virtual_visits = max(0, int(self.virtual_visits + count))

    def remove_virtual_visit(self, count: int = 1) -> None:
        self.virtual_visits = max(0, int(self.virtual_visits - count))

    def __repr__(self) -> str:
        return (
            f"Node(visits={self.visit_count}, virtual={self.virtual_visits}, "
            f"q={self.q:.3f}, prior={self.prior:.3f}, children={len(self.children)})"
        )
