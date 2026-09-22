from __future__ import annotations

from typing import Dict, Optional

import chess


class Node:
    """One search-tree node.

    ``value_sum``/``q`` are from the perspective of the side to move *at this
    node*. A parent therefore reads ``-child.q``. The edge penalty leading into
    this node (``penalty``) is static for the node's unique path from the game
    start, so it is computed once by the parent and cached here.
    """

    __slots__ = (
        "prior",
        "parent",
        "visit_count",
        "value_sum",
        "virtual_visits",
        "children",
        "value_estimate",
        "penalty",
        "penalty_components",
        "penalties_ready",
        "penalty_tier",
        "terminal_checked",
        "terminal_value",
    )

    def __init__(self, prior: float, parent: Optional["Node"] = None):
        self.prior: float = float(prior)
        self.parent: Optional["Node"] = parent
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.virtual_visits: int = 0
        self.children: Dict[chess.Move, "Node"] = {}
        # Network/blended value at expansion (side-to-move perspective); None until expanded.
        self.value_estimate: Optional[float] = None
        # Static selection penalty of the edge parent -> this node, cached by the parent.
        self.penalty: float = 0.0
        self.penalty_components: Optional[dict] = None
        # True once every child's edge penalty has been computed, and how rich that
        # computation was (0 = root tier, 1 = principle tier, 2 = tactical only).
        self.penalties_ready: bool = False
        self.penalty_tier: int = 2
        # Cached terminal state of this node's position (None = not terminal).
        self.terminal_checked: bool = False
        self.terminal_value: Optional[float] = None

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
