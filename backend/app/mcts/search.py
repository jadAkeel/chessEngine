from __future__ import annotations

import logging
import math
import time
from collections.abc import Iterable, Mapping

import chess
import numpy as np

from app.evaluation.metrics import evaluate_board
from app.game.move_encoding import move_to_index
from app.game.principles import principle_penalty_components
from app.game.repetition import PositionKey, build_seen_positions, current_repetition_count, filter_repetition_moves, position_key
from app.game.tactics import TacticalSearchExpired, find_mate_in_one, select_safe_move
from app.infra.config import AppConfig, get_current_config
from app.mcts.node import Node
from app.mcts.temperature import apply_temperature
from app.model.inference import predict_boards

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

PIECE_CONFIG_NAMES = {
    chess.PAWN: "PAWN",
    chess.KNIGHT: "KNIGHT",
    chess.BISHOP: "BISHOP",
    chess.ROOK: "ROOK",
    chess.QUEEN: "QUEEN",
}

# Below this root value the mover is worse; a draw by repetition is then
# acceptable and the root anti-repetition filter is skipped (play mode only).
REPETITION_FILTER_MIN_ROOT_VALUE = -0.05


class MCTS:
    def __init__(self, model, cfg: AppConfig | None = None, device: str = "cpu", c_puct: float | None = None):
        self.model = model
        self.cfg = cfg or getattr(model, "cfg", None) or get_current_config()
        self.device = device
        self.c_puct = float(c_puct if c_puct is not None else self.cfg.mcts.c_puct)
        self.virtual_loss = max(0.0, float(getattr(self.cfg.mcts, "virtual_loss", 1.0)))
        self.fpu_reduction = max(0.0, float(getattr(self.cfg.mcts, "fpu_reduction", 0.25)))
        self.reuse_tree = bool(getattr(self.cfg.mcts, "reuse_tree", True))
        principles_cfg = getattr(self.cfg, "principle_penalties", None)
        self.principle_max_depth = int(getattr(principles_cfg, "max_tree_depth", 2))
        self.logger = logging.getLogger(__name__)
        # Per-search compute caches (cleared every search). Node-level caches persist with the tree.
        self._tactical_penalty_cache: dict[tuple[PositionKey, str, int], float] = {}
        self._position_tactical_cache: dict[tuple[PositionKey, str], float] = {}
        self._principle_penalty_cache: dict[tuple, dict[str, float]] = {}
        # Retained tree for reuse between consecutive searches.
        self._root: Node | None = None
        self._root_stack: tuple[chess.Move, ...] | None = None
        self._root_board: chess.Board | None = None

    # ------------------------------------------------------------------ config helpers

    def _get_piece_penalty_cfg(self, piece_type: chess.PieceType) -> dict[str, float] | None:
        if piece_type == chess.KING:
            return None

        piece_value = float(PIECE_VALUES.get(piece_type, 0))
        queen_value = float(PIECE_VALUES[chess.QUEEN])
        if piece_value <= 0.0 or queen_value <= 0.0:
            return None

        scale = piece_value / queen_value
        fallback = {
            "blunder_penalty": float(getattr(self.cfg.mcts, "queen_blunder_penalty", 0.0)) * scale,
            "hanging_penalty": float(getattr(self.cfg.mcts, "queen_hanging_penalty", 0.0)) * scale,
            "sac_compensation_threshold": float(getattr(self.cfg.mcts, "queen_sac_compensation_threshold", 0.0)) * scale,
            "check_discount": float(getattr(self.cfg.mcts, "queen_check_discount", 1.0)),
        }
        configured = getattr(self.cfg.mcts, "piece_penalties", {}) or {}
        piece_name = PIECE_CONFIG_NAMES.get(piece_type, "")
        custom = configured.get(piece_name) or configured.get(piece_name.lower()) or {}
        if not isinstance(custom, Mapping):
            return fallback
        return {
            "blunder_penalty": float(custom.get("blunder_penalty", fallback["blunder_penalty"])),
            "hanging_penalty": float(custom.get("hanging_penalty", fallback["hanging_penalty"])),
            "sac_compensation_threshold": float(
                custom.get("sac_compensation_threshold", fallback["sac_compensation_threshold"])
            ),
            "check_discount": float(custom.get("check_discount", fallback["check_discount"])),
        }

    # ------------------------------------------------------------------ exchange helpers

    def _capture_replies_to_square(self, board: chess.Board, target_square: int) -> list[chess.Move]:
        return list(board.generate_legal_captures(chess.BB_ALL, chess.BB_SQUARES[target_square]))

    def _best_recapture_delta(
        self,
        board: chess.Board,
        *,
        mover: chess.Color,
        before_material: int,
        target_square: int,
    ) -> int:
        best_delta = int(self._material_balance(board, mover) - before_material)
        for reply in self._capture_replies_to_square(board, target_square):
            board.push(reply)
            try:
                delta = int(self._material_balance(board, mover) - before_material)
                if delta > best_delta:
                    best_delta = delta
            finally:
                board.pop()
        return best_delta

    def _worst_exchange_delta_on_square(
        self,
        board: chess.Board,
        *,
        mover: chess.Color,
        before_material: int,
        target_square: int,
    ) -> int | None:
        worst_delta: int | None = None
        capture_replies = self._capture_replies_to_square(board, target_square)
        if not capture_replies:
            return None

        for reply in capture_replies:
            board.push(reply)
            try:
                exchange_delta = self._best_recapture_delta(
                    board,
                    mover=mover,
                    before_material=before_material,
                    target_square=target_square,
                )
                if worst_delta is None or exchange_delta < worst_delta:
                    worst_delta = exchange_delta
            finally:
                board.pop()
        return worst_delta

    # ------------------------------------------------------------------ tree reuse

    def reset_tree(self) -> None:
        self._root = None
        self._root_stack = None
        self._root_board = None

    def _find_reusable_root(self, board: chess.Board) -> Node | None:
        """Return the retained node matching ``board`` (same game history), or None."""
        if not self.reuse_tree or self._root is None or self._root_stack is None or self._root_board is None:
            return None
        new_stack = tuple(board.move_stack)
        old_stack = self._root_stack
        if len(new_stack) < len(old_stack) or new_stack[: len(old_stack)] != old_stack:
            return None
        node = self._root
        probe = self._root_board.copy(stack=False)
        for move in new_stack[len(old_stack):]:
            node = node.children.get(move)
            if node is None:
                return None
            probe.push(move)
        if probe.fen(en_passant="fen") != board.fen(en_passant="fen"):
            return None
        return node

    def retained_visits(self, board: chess.Board) -> int:
        """Root visits a search on ``board`` would start from (0 when nothing is reusable)."""
        node = self._find_reusable_root(board)
        if node is None or not node.expanded():
            return 0
        return int(node.visit_count)

    def _renoise_root(self, root: Node) -> None:
        priors = {move: child.prior for move, child in root.children.items()}
        if not priors:
            return
        for move, prior in self._apply_dirichlet_noise(priors).items():
            root.children[move].prior = float(prior)

    # ------------------------------------------------------------------ search

    def search(
        self,
        board: chess.Board,
        add_noise: bool = False,
        num_simulations: int | None = None,
        temperature: float | None = None,
        time_limit_sec: float | None = None,
    ) -> dict:
        if not isinstance(board, chess.Board):
            raise TypeError("Expected board to be chess.Board")

        if time_limit_sec is not None and (not math.isfinite(time_limit_sec) or time_limit_sec <= 0):
            raise ValueError("time_limit_sec must be finite and greater than zero")
        deadline = None if time_limit_sec is None else time.monotonic() + time_limit_sec
        self._tactical_penalty_cache.clear()
        self._position_tactical_cache.clear()
        self._principle_penalty_cache.clear()

        root_seen_positions = build_seen_positions(board)
        if self._is_terminal_board(board, root_seen_positions):
            self.reset_tree()
            return {"best_move": None, "root_value": self._terminal_value(board, root_seen_positions), "visit_counts": {},
                    "policy_target": {}, "adjusted_policy_target": {}, "root_repetition_counts": {},
                    "root_diagnostics": [], "retained_visits": 0, "completed_simulations": 0}
        try:
            mate = find_mate_in_one(board, deadline)
        except TacticalSearchExpired:
            mate = None
        if mate is not None:
            self.reset_tree()
            return {"best_move": mate, "root_value": 1.0, "visit_counts": {mate: 1},
                    "policy_target": {mate: 1.0}, "adjusted_policy_target": {mate: 1.0},
                    "root_repetition_counts": {}, "root_diagnostics": [], "retained_visits": 0,
                    "completed_simulations": 0}
        sims = max(1, int(num_simulations if num_simulations is not None else self.cfg.mcts.num_simulations))
        temperature = float(self.cfg.mcts.temperature if temperature is None else temperature)
        batch_limit = max(1, int(self.cfg.mcts.inference_batch_size))

        self.logger.debug(
            "mcts start fen=%s sims=%s temperature=%.3f add_noise=%s",
            board.fen(),
            sims,
            temperature,
            add_noise,
        )

        root = self._find_reusable_root(board)
        retained_visits = 0
        if root is not None and root.expanded():
            root.parent = None
            root.virtual_visits = 0
            retained_visits = int(root.visit_count)
            initial_root_value = float(root.value_estimate if root.value_estimate is not None else root.q)
            if add_noise:
                self._renoise_root(root)
        else:
            root = Node(prior=0.0)
            initial_root_value = self._expand_node(root, board, add_noise=add_noise)
        self._root = root
        self._root_stack = tuple(board.move_stack)
        self._root_board = board.copy(stack=False)

        pending_simulations = sims
        completed = 0
        # Leave some of the move budget for the root tactical check.
        search_deadline = None if deadline is None else deadline - min(0.25, time_limit_sec * 0.2)
        diagnostics = self._new_penalty_diagnostics() if self._penalty_diagnostics_enabled() else None

        while pending_simulations > 0:
            if search_deadline is not None and time.monotonic() >= search_deadline:
                break
            rollout_batch = min(batch_limit, pending_simulations)
            pending_simulations -= rollout_batch
            pending_groups: dict[str, list[tuple[Node, list[Node], chess.Board]]] = {}

            for _ in range(rollout_batch):
                if search_deadline is not None and time.monotonic() >= search_deadline:
                    break
                node = root
                sim_board = board.copy(stack=True)
                sim_seen_positions = dict(root_seen_positions)
                search_path = [node]
                depth = 0

                while True:
                    terminal_value = self._node_terminal_value(node, sim_board, sim_seen_positions)
                    if terminal_value is not None:
                        self._backpropagate(search_path, terminal_value)
                        completed += 1
                        break
                    if not node.expanded():
                        self._reserve_virtual_path(search_path)
                        leaf_key = self._prediction_key(sim_board)
                        pending_groups.setdefault(leaf_key, []).append((node, search_path, sim_board))
                        break

                    move, next_node = self._select_child(
                        node, sim_board, sim_seen_positions, diagnostics=diagnostics, depth=depth
                    )
                    if move is None or next_node is None:
                        break
                    sim_board.push(move)
                    key = position_key(sim_board)
                    sim_seen_positions[key] = sim_seen_positions.get(key, 0) + 1
                    node = next_node
                    search_path.append(node)
                    depth += 1

            if pending_groups:
                grouped_boards = [entries[0][2] for entries in pending_groups.values()]
                policy_logits_batch, values_batch = predict_boards(
                    self.model,
                    grouped_boards,
                    cfg=self.cfg,
                    device=self.device,
                )
                for entries, policy_logits, nn_value in zip(pending_groups.values(), policy_logits_batch, values_batch):
                    for node, search_path, sim_board in entries:
                        self._release_virtual_path(search_path)
                        leaf_value = self._expand_node_from_prediction(
                            node=node,
                            board=sim_board,
                            policy_logits=policy_logits,
                            nn_value=float(nn_value),
                            add_noise=False,
                        )
                        self._backpropagate(search_path, leaf_value)
                        completed += 1

        visit_counts = {move: child.visit_count for move, child in root.children.items()}
        raw_policy_target = self._visit_policy(root, temperature=temperature)
        if not any(visit_counts.values()):
            raw_policy_target = {move: child.prior for move, child in root.children.items()}
        root_value = float(root.q) if root.visit_count > 0 else float(initial_root_value)
        adjusted_policy_target, root_repetition_counts = self._adjust_root_policy(
            board, raw_policy_target, root_seen_positions, root_value=root_value, training_mode=add_noise
        )
        best_move = self._select_root_move(board, root, adjusted_policy_target or raw_policy_target, root_seen_positions)
        ranked_moves = sorted(root.children, key=lambda move: (
            move == best_move, float(adjusted_policy_target.get(move, 0.0)),
            root.children[move].visit_count, -root.children[move].q,
        ), reverse=True)
        best_move, rejected = select_safe_move(board, ranked_moves, deadline)
        if rejected:
            adjusted_policy_target = {move: prob for move, prob in adjusted_policy_target.items() if move not in rejected}
            total = sum(adjusted_policy_target.values())
            adjusted_policy_target = ({move: prob / total for move, prob in adjusted_policy_target.items()}
                                      if total > 1e-12 else {best_move: 1.0})
        root_diagnostics = self._root_move_diagnostics(
            board,
            root,
            raw_policy_target,
            adjusted_policy_target,
            root_seen_positions,
        )

        self.logger.debug(
            "mcts done sims=%s completed=%s retained=%s expanded_children=%s root_visits=%s best_move=%s root_value=%.4f",
            sims,
            completed,
            retained_visits,
            len(root.children),
            root.visit_count,
            best_move.uci() if best_move else None,
            root_value,
        )
        result = {
            "best_move": best_move,
            "visit_counts": visit_counts,
            "policy_target": raw_policy_target,
            "adjusted_policy_target": adjusted_policy_target,
            "root_repetition_counts": root_repetition_counts,
            "root_diagnostics": root_diagnostics,
            "root_value": root_value,
            "retained_visits": retained_visits,
            "completed_simulations": completed,
        }
        if diagnostics is not None:
            result["penalty_diagnostics"] = self._finalize_penalty_diagnostics(diagnostics)
        return result

    def _prediction_key(self, board: chess.Board) -> str:
        return board.fen(en_passant="fen")

    def _reserve_virtual_path(self, search_path: list[Node]) -> None:
        for node in search_path:
            node.add_virtual_visit(1)

    def _release_virtual_path(self, search_path: list[Node]) -> None:
        for node in search_path:
            node.remove_virtual_visit(1)

    # ------------------------------------------------------------------ expansion / evaluation

    def _expand_node(self, node: Node, board: chess.Board, add_noise: bool) -> float:
        policy_logits_batch, value_batch = predict_boards(self.model, [board], cfg=self.cfg, device=self.device)
        return self._expand_node_from_prediction(
            node=node,
            board=board,
            policy_logits=policy_logits_batch[0],
            nn_value=float(value_batch[0]),
            add_noise=add_noise,
        )

    def _expand_node_from_prediction(self, node: Node, board: chess.Board, policy_logits, nn_value: float, add_noise: bool) -> float:
        legal_moves = list(board.legal_moves)
        priors = self._legal_priors(policy_logits, legal_moves, board)
        if add_noise and priors:
            priors = self._apply_dirichlet_noise(priors)
        for move, prior in priors.items():
            if move not in node.children:
                node.children[move] = Node(prior=prior, parent=node)
        value = self._blend_value(board, nn_value)
        node.value_estimate = float(value)
        return value

    def _blend_value(self, board: chess.Board, nn_value: float) -> float:
        classical_alpha = min(max(float(self.cfg.mcts.classical_value_alpha), 0.0), 1.0)
        classical_eval = float(evaluate_board(board))
        if board.turn == chess.BLACK:
            classical_eval *= -1.0
        classical_value = float(np.tanh(classical_eval / 600.0))
        blended = classical_alpha * classical_value + (1.0 - classical_alpha) * float(nn_value)
        # A stagnant halfmove clock drifts toward the fifty-move draw: shrink the
        # value toward 0 for both sides rather than charging the side to move at
        # the leaf (which would alternate sign with search depth).
        blended *= max(0.0, 1.0 - 2.0 * self._progress_penalty(board))
        return float(np.clip(blended, -1.0, 1.0))

    def _progress_penalty(self, board: chess.Board) -> float:
        halfmove_clock = int(getattr(board, "halfmove_clock", 0))
        if halfmove_clock >= 80:
            return 0.20
        if halfmove_clock >= 60:
            return 0.12
        if halfmove_clock >= 40:
            return 0.06
        if halfmove_clock >= 20:
            return 0.03
        return 0.0

    def _apply_dirichlet_noise(self, priors: dict[chess.Move, float]) -> dict[chess.Move, float]:
        moves = list(priors.keys())
        if not moves:
            return priors
        alpha = float(self.cfg.mcts.dirichlet_alpha)
        epsilon = min(max(float(self.cfg.mcts.dirichlet_eps), 0.0), 1.0)
        noise = np.random.dirichlet([alpha] * len(moves))
        mixed = {move: (1.0 - epsilon) * float(priors[move]) + epsilon * float(noise[i]) for i, move in enumerate(moves)}
        total = float(sum(mixed.values()))
        if total <= 0.0 or not np.isfinite(total):
            uniform = 1.0 / len(moves)
            return {move: uniform for move in moves}
        return {move: value / total for move, value in mixed.items()}

    def _legal_priors(self, policy_logits, legal_moves: Iterable[chess.Move], board: chess.Board) -> dict[chess.Move, float]:
        legal_moves = list(legal_moves)
        if not legal_moves:
            return {}
        logits = np.asarray(policy_logits, dtype=np.float32)
        legal_indices = np.array([move_to_index(move, board) for move in legal_moves], dtype=np.int64)
        legal_logits = logits[legal_indices]
        max_logit = float(np.max(legal_logits))
        probs = np.exp(legal_logits - max_logit)
        total = float(np.sum(probs))
        if total <= 0.0 or not np.isfinite(total):
            uniform = 1.0 / len(legal_moves)
            return {move: uniform for move in legal_moves}
        probs = probs / total
        return {move: float(prob) for move, prob in zip(legal_moves, probs)}

    # ------------------------------------------------------------------ diagnostics

    def _penalty_diagnostics_enabled(self) -> bool:
        return bool(getattr(getattr(self.cfg, "penalty_diagnostics", None), "enabled", False))

    def _new_penalty_diagnostics(self) -> dict:
        return {
            "components": {},
            "total": {"count": 0, "sum": 0.0, "max": 0.0},
            "thresholds": {"gt_0.25": 0, "gt_0.5": 0, "gt_0.75": 0, "gt_1.0": 0},
            "ranking": {"comparisons": 0, "changed": 0},
        }

    def _record_penalty_diagnostics(self, diagnostics: dict, components: Mapping[str, float]) -> None:
        total = float(sum(float(value) for value in components.values()))
        total_stats = diagnostics["total"]
        total_stats["count"] += 1
        total_stats["sum"] += total
        total_stats["max"] = max(float(total_stats["max"]), total)

        for threshold in (0.25, 0.5, 0.75, 1.0):
            if total > threshold:
                diagnostics["thresholds"][f"gt_{threshold}"] += 1

        for name, value in components.items():
            value = float(value)
            if value <= 0.0:
                continue
            stats = diagnostics["components"].setdefault(name, {"count": 0, "sum": 0.0, "max": 0.0})
            stats["count"] += 1
            stats["sum"] += value
            stats["max"] = max(float(stats["max"]), value)

    def _finalize_penalty_diagnostics(self, diagnostics: dict) -> dict:
        components = {}
        for name, stats in diagnostics["components"].items():
            count = int(stats["count"])
            components[name] = {
                "count": count,
                "avg": float(stats["sum"]) / count if count else 0.0,
                "max": float(stats["max"]),
            }

        total_count = int(diagnostics["total"]["count"])
        return {
            "components": components,
            "total_move_penalty": {
                "count": total_count,
                "sum": float(diagnostics["total"]["sum"]),
                "avg": float(diagnostics["total"]["sum"]) / total_count if total_count else 0.0,
                "max": float(diagnostics["total"]["max"]),
            },
            "thresholds": dict(diagnostics["thresholds"]),
            "ranking_changed": int(diagnostics["ranking"]["changed"]),
            "ranking_comparisons": int(diagnostics["ranking"]["comparisons"]),
        }

    # ------------------------------------------------------------------ selection

    def _ensure_child_penalties(
        self,
        node: Node,
        board: chess.Board,
        seen_positions: Mapping[PositionKey, int] | None,
        depth: int = 0,
    ) -> None:
        """Compute every child's static edge penalty once and cache it on the child.

        Deeper nodes get a cheaper penalty set; when tree reuse later surfaces a node
        to a shallower depth, its children are recomputed at the richer tier.
        """
        include_principles = self.principle_max_depth < 0 or int(depth) <= self.principle_max_depth
        # The opponent-reply mate/fork scan is the one expensive principle; the root
        # safety screen already covers short mates for the move actually played.
        include_tactics = int(depth) == 0
        tier = 0 if include_tactics else (1 if include_principles else 2)
        if node.penalties_ready and node.penalty_tier <= tier:
            return
        for move, child in node.children.items():
            components = self._move_penalty_components(
                board, move, seen_positions,
                include_principles=include_principles, include_tactics=include_tactics,
            )
            child.penalty_components = components
            child.penalty = float(sum(components.values()))
        node.penalties_ready = True
        node.penalty_tier = tier

    def _select_child(
        self,
        node: Node,
        board: chess.Board,
        seen_positions: Mapping[PositionKey, int] | None = None,
        diagnostics: dict | None = None,
        depth: int = 0,
    ):
        if not node.children:
            return None, None

        self._ensure_child_penalties(node, board, seen_positions, depth)

        best_score = -float("inf")
        best_raw_score = -float("inf")
        best_move = None
        best_raw_move = None
        best_child = None
        parent_visits = max(1, node.total_visit_count)
        exploration = self.c_puct * math.sqrt(parent_visits)
        # First-play urgency: an unvisited child inherits the parent's estimate
        # minus a reduction instead of a neutral 0, so a losing parent does not
        # spray visits over every untried move.
        parent_value = float(node.q) if node.visit_count > 0 else float(node.value_estimate or 0.0)
        fpu_value = parent_value - self.fpu_reduction
        virtual_loss = self.virtual_loss

        for move, child in node.children.items():
            visits = child.visit_count
            virtual = child.virtual_visits
            if visits + virtual == 0:
                q_value = fpu_value
            else:
                # Pending (virtual) visits count as losses for the mover, diluted by the
                # child's real visits, so in-flight rollouts spread without erasing Q.
                q_value = (-child.value_sum - virtual_loss * virtual) / (visits + virtual)
            u_value = exploration * child.prior / (1 + visits + virtual)
            raw_score = q_value + u_value
            move_penalty = child.penalty
            if diagnostics is not None:
                self._record_penalty_diagnostics(diagnostics, child.penalty_components or {})
            score = raw_score - move_penalty
            if raw_score > best_raw_score:
                best_raw_score = raw_score
                best_raw_move = move
            if score > best_score:
                best_score = score
                best_move = move
                best_child = child

        if diagnostics is not None and best_move is not None and best_raw_move is not None:
            diagnostics["ranking"]["comparisons"] += 1
            if best_move != best_raw_move:
                diagnostics["ranking"]["changed"] += 1

        return best_move, best_child

    # ------------------------------------------------------------------ move penalties

    def _move_penalty(self, board: chess.Board, move: chess.Move, seen_positions: Mapping[PositionKey, int] | None = None) -> float:
        return float(sum(self._move_penalty_components(board, move, seen_positions).values()))

    def _move_penalty_components(
        self,
        board: chess.Board,
        move: chess.Move,
        seen_positions: Mapping[PositionKey, int] | None = None,
        *,
        include_principles: bool = True,
        include_tactics: bool = True,
    ) -> dict[str, float]:
        oscillation_penalty = self._oscillation_penalty(board, move)
        before_halfmove = int(getattr(board, "halfmove_clock", 0))
        was_capture = bool(board.is_capture(move))
        mover = board.turn
        before_material = self._material_balance(board, mover)
        before_position_key = position_key(board)

        # Evaluate on a stackless copy: the original board (with history) stays the
        # untouched "before" position for the principle heuristics.
        after = board.copy(stack=False)
        after.push(move)

        tactical_key = (before_position_key, move.uci())
        tactical_penalty = self._position_tactical_cache.get(tactical_key)
        if tactical_penalty is None:
            tactical_penalty = 0.0
            # A quiet move can abandon another piece (e.g. Rae1 leaving Qh3
            # to ...Bxh3). Inspect every legally capturable friendly piece.
            targets = {}
            for reply in after.generate_legal_captures():
                if after.is_en_passant(reply):
                    targets[reply.to_square] = chess.PAWN
                else:
                    piece = after.piece_at(reply.to_square)
                    if piece is not None and piece.color == mover:
                        targets[reply.to_square] = piece.piece_type
            for target, piece_type in targets.items():
                tactical_penalty = max(tactical_penalty, self._piece_tactical_penalty_after_push(
                    after, mover=mover, piece_type=piece_type,
                    before_material=before_material, target_square=target,
                    before_position_key=before_position_key, move_uci=move.uci(),
                ))
            self._position_tactical_cache[tactical_key] = tactical_penalty

        principle_components: dict[str, float] = {}
        if include_principles:
            principle_key = (
                before_position_key,
                int(board.halfmove_clock),
                int(board.fullmove_number),
                tuple(board.move_stack),
                move,
                bool(include_tactics),
            )
            cached = self._principle_penalty_cache.get(principle_key)
            if cached is None:
                cached = principle_penalty_components(
                    before=board,
                    after=after,
                    move=move,
                    cfg=self.cfg.principle_penalties,
                    include_tactics=include_tactics,
                ).components
                self._principle_penalty_cache[principle_key] = cached
            principle_components = cached

        return {
            "oscillation": float(oscillation_penalty),
            "repetition": float(self._repetition_penalty_in_position(after, seen_positions)),
            "progress": float(self._forward_progress_penalty_after_push(after, before_halfmove, was_capture)),
            "tactical": float(tactical_penalty),
            **{f"principle.{name}": float(value) for name, value in principle_components.items()},
        }

    def _piece_tactical_penalty(self, board: chess.Board, move: chess.Move) -> float:
        moving_piece = board.piece_at(move.from_square)
        if moving_piece is None:
            return 0.0

        mover = bool(moving_piece.color)
        before_material = self._material_balance(board, mover)
        before_position_key = position_key(board)

        board.push(move)
        try:
            moved_piece = board.piece_at(move.to_square)
            if moved_piece is None or moved_piece.color != mover:
                return 0.0
            return float(
                self._piece_tactical_penalty_after_push(
                    board,
                    mover=mover,
                    piece_type=moved_piece.piece_type,
                    before_material=before_material,
                    target_square=move.to_square,
                    before_position_key=before_position_key,
                    move_uci=move.uci(),
                )
            )
        finally:
            board.pop()

    def _queen_tactical_penalty(self, board: chess.Board, move: chess.Move) -> float:
        piece = board.piece_at(move.from_square)
        if piece is None or piece.piece_type != chess.QUEEN:
            return 0.0
        return float(self._piece_tactical_penalty(board, move))

    def _bit_count(self, mask: int) -> int:
        return int(mask).bit_count()

    def _material_balance(self, board: chess.Board, perspective: chess.Color) -> int:
        own = perspective
        opp = not perspective
        score = 0
        for piece_type, value in PIECE_VALUES.items():
            if value <= 0:
                continue
            score += value * (self._bit_count(board.pieces_mask(piece_type, own)) - self._bit_count(board.pieces_mask(piece_type, opp)))
        return int(score)

    def _piece_tactical_penalty_after_push(
        self,
        board: chess.Board,
        *,
        mover: chess.Color,
        piece_type: chess.PieceType,
        before_material: int,
        target_square: int,
        before_position_key: PositionKey,
        move_uci: str,
    ) -> float:
        cfg = self._get_piece_penalty_cfg(piece_type)
        if cfg is None or board.is_checkmate():
            return 0.0

        cache_key = (before_position_key, move_uci, target_square)
        cached = self._tactical_penalty_cache.get(cache_key)
        if cached is not None:
            return float(cached)

        worst_exchange_delta = self._worst_exchange_delta_on_square(
            board,
            mover=mover,
            before_material=before_material,
            target_square=target_square,
        )
        if worst_exchange_delta is None:
            penalty = 0.0
        else:
            threshold = float(cfg["sac_compensation_threshold"])
            if worst_exchange_delta >= 0 or worst_exchange_delta >= threshold:
                penalty = 0.0
            else:
                piece_value = float(PIECE_VALUES.get(piece_type, 0))
                net_loss = float(-worst_exchange_delta)
                if net_loss >= max(piece_value * 0.75, 100.0):
                    penalty = float(cfg["blunder_penalty"])
                else:
                    penalty = float(cfg["hanging_penalty"])

                if penalty > 0.0 and board.is_check():
                    penalty *= float(cfg.get("check_discount", 1.0))

        self._tactical_penalty_cache[cache_key] = float(penalty)
        return float(penalty)

    def _repetition_penalty_in_position(self, board: chess.Board, seen_positions: Mapping[PositionKey, int] | None = None) -> float:
        repetition_penalty = max(0.0, float(self.cfg.selfplay.repetition_penalty))
        if repetition_penalty <= 0.0:
            return 0.0

        if seen_positions is not None:
            next_key = position_key(board)
            repeat_count = int(seen_positions.get(next_key, 0) + 1)
        else:
            repeat_count = current_repetition_count(board)

        penalty = 0.0
        if repeat_count > 1:
            penalty += repetition_penalty * float(repeat_count - 1)
        can_claim_draw = repeat_count >= 3 or int(getattr(board, "halfmove_clock", 0)) >= 100
        if can_claim_draw:
            penalty += repetition_penalty * 2.0
        return float(penalty)

    def _oscillation_penalty(self, board: chess.Board, move: chess.Move) -> float:
        # The last ply is the opponent's; the mover's own previous move is two plies back.
        if len(board.move_stack) < 2:
            return 0.0
        previous = board.move_stack[-2]
        if move.from_square == previous.to_square and move.to_square == previous.from_square:
            # Returning with a capture or a check is purposeful, not shuffling.
            if board.is_capture(move) or board.gives_check(move):
                return 0.0
            return max(0.0, float(self.cfg.selfplay.repetition_penalty)) * 0.25
        return 0.0

    def _forward_progress_penalty_after_push(self, board: chess.Board, before_halfmove: int, was_capture: bool) -> float:
        after_halfmove = int(getattr(board, "halfmove_clock", 0))
        penalty = self._progress_penalty(board)
        if after_halfmove > before_halfmove and not was_capture:
            penalty += 0.02
        return float(penalty)

    # ------------------------------------------------------------------ root policy / move choice

    def _adjust_root_policy(
        self,
        board: chess.Board,
        policy_target: dict[chess.Move, float],
        seen_positions: Mapping[PositionKey, int],
        *,
        root_value: float = 0.0,
        training_mode: bool = False,
    ) -> tuple[dict[chess.Move, float], dict[str, int]]:
        adjusted_policy, repetition_counts = filter_repetition_moves(
            policy_target,
            board,
            seen_positions,
            repeat_break_count=int(self.cfg.selfplay.repetition_break_count),
            repeat_weight=float(self.cfg.selfplay.repetition_move_weight),
        )
        if not adjusted_policy:
            return adjusted_policy, repetition_counts
        # In play, a mover who is worse should be allowed to hold a draw by
        # repetition; keep the anti-repetition filter unconditional only for
        # self-play (noisy) searches, where game diversity is the goal.
        if not training_mode and float(root_value) < REPETITION_FILTER_MIN_ROOT_VALUE:
            adjusted_policy = dict(policy_target)

        if self._root is not None:
            self._ensure_child_penalties(self._root, board, seen_positions, 0)
        reweighted: dict[chess.Move, float] = {}
        for move, prob in adjusted_policy.items():
            reweighted[move] = max(0.0, float(prob) - self._root_edge_penalty(board, move, seen_positions) * 0.05)

        total = float(sum(reweighted.values()))
        if total <= 1e-12:
            return adjusted_policy, repetition_counts
        return ({move: value / total for move, value in reweighted.items()}, repetition_counts)

    def _root_edge_penalty(self, board: chess.Board, move: chess.Move, seen_positions: Mapping[PositionKey, int] | None) -> float:
        root = self._root
        if root is not None and root.penalties_ready:
            child = root.children.get(move)
            if child is not None:
                return float(child.penalty)
        return self._move_penalty(board, move, seen_positions)

    def _select_root_move(self, board: chess.Board, root: Node, policy_target: dict[chess.Move, float], seen_positions: Mapping[PositionKey, int] | None = None) -> chess.Move | None:
        if not policy_target:
            return None

        ranked_moves = sorted(
            policy_target.items(),
            key=lambda item: (
                float(item[1]),
                float(root.children[item[0]].visit_count if item[0] in root.children else 0),
                float(-root.children[item[0]].q if item[0] in root.children else -1e9),
            ),
            reverse=True,
        )
        if not ranked_moves:
            return None
        if len(ranked_moves) == 1:
            return ranked_moves[0][0]

        top_move, top_prob = ranked_moves[0]
        second_move, second_prob = ranked_moves[1]
        if abs(float(top_prob) - float(second_prob)) < 0.03:
            top_penalty = self._move_penalty(board, top_move, seen_positions)
            second_penalty = self._move_penalty(board, second_move, seen_positions)
            return second_move if second_penalty + 1e-9 < top_penalty else top_move
        return top_move

    def _root_move_diagnostics(
        self,
        board: chess.Board,
        root: Node,
        raw_policy_target: dict[chess.Move, float],
        adjusted_policy_target: dict[chess.Move, float],
        seen_positions: Mapping[PositionKey, int] | None = None,
        limit: int = 8,
    ) -> list[dict]:
        parent_visits = max(1, root.total_visit_count)
        diagnostics: list[dict] = []
        self._ensure_child_penalties(root, board, seen_positions, 0)

        for move, child in root.children.items():
            components = child.penalty_components or {}
            penalty = float(child.penalty)
            q_value = float(-child.q)
            u_value = float(self.c_puct * child.prior * math.sqrt(parent_visits) / (1 + child.total_visit_count))
            raw_score = float(q_value + u_value)
            nonzero_components = {
                name: round(float(value), 6)
                for name, value in components.items()
                if float(value) > 0.0
            }
            diagnostics.append(
                {
                    "uci": move.uci(),
                    "san": board.san(move),
                    "prior": round(float(child.prior), 6),
                    "visits": int(child.visit_count),
                    "q": round(float(child.q), 6),
                    "policy": round(float(raw_policy_target.get(move, 0.0)), 6),
                    "adjusted_policy": round(float(adjusted_policy_target.get(move, 0.0)), 6),
                    "raw_score": round(raw_score, 6),
                    "penalty": round(penalty, 6),
                    "final_score": round(raw_score - penalty, 6),
                    "penalty_components": nonzero_components,
                }
            )

        diagnostics.sort(
            key=lambda item: (
                float(item["adjusted_policy"]),
                int(item["visits"]),
                float(item["prior"]),
            ),
            reverse=True,
        )
        return diagnostics[: max(1, int(limit))]

    # ------------------------------------------------------------------ backup / terminal

    def _backpropagate(self, search_path: list[Node], value: float) -> None:
        current_value = float(value)
        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += current_value
            current_value = -current_value

    def _node_terminal_value(
        self,
        node: Node,
        board: chess.Board,
        seen_positions: Mapping[PositionKey, int],
    ) -> float | None:
        if not node.terminal_checked:
            node.terminal_value = self._terminal_state(board, seen_positions)
            node.terminal_checked = True
        return node.terminal_value

    def _terminal_state(self, board: chess.Board, seen_positions: Mapping[PositionKey, int] | None = None) -> float | None:
        """Terminal value for the side to move, or None. Mirrors ``is_game_over(claim_draw=True)``
        for the current position but uses the incremental repetition map instead of
        replaying the move stack at every node."""
        if not any(board.generate_legal_moves()):
            return -1.0 if board.is_check() else 0.0
        if board.is_insufficient_material():
            return 0.0
        if int(board.halfmove_clock) >= 100:
            return 0.0
        if seen_positions is not None:
            repeat_count = int(seen_positions.get(position_key(board), 0))
        else:
            repeat_count = current_repetition_count(board)
        if repeat_count >= 3:
            return 0.0
        return None

    def _is_terminal_board(self, board: chess.Board, seen_positions: Mapping[PositionKey, int] | None = None) -> bool:
        return self._terminal_state(board, seen_positions) is not None

    def _terminal_value(self, board: chess.Board, seen_positions: Mapping[PositionKey, int] | None = None) -> float:
        value = self._terminal_state(board, seen_positions)
        return 0.0 if value is None else float(value)

    def _visit_policy(self, root: Node, temperature: float = 1.0) -> dict[chess.Move, float]:
        visit_counts = {move: int(child.visit_count) for move, child in root.children.items()}
        return apply_temperature(visit_counts, temperature)
