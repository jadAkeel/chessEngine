from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping

import chess
import numpy as np

from app.evaluation.metrics import evaluate_board
from app.game.move_encoding import move_to_index
from app.game.principles import principle_penalty_components
from app.game.repetition import PositionKey, build_seen_positions, current_repetition_count, filter_repetition_moves, position_key
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


class MCTS:
    def __init__(self, model, cfg: AppConfig | None = None, device: str = "cpu", c_puct: float | None = None):
        self.model = model
        self.cfg = cfg or getattr(model, "cfg", None) or get_current_config()
        self.device = device
        self.c_puct = float(c_puct if c_puct is not None else self.cfg.mcts.c_puct)
        self.virtual_loss = max(0.0, float(getattr(self.cfg.mcts, "virtual_loss", 1.0)))
        self.logger = logging.getLogger(__name__)
        self._tactical_penalty_cache: dict[tuple[PositionKey, str], float] = {}

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

    def _capture_replies_to_square(self, board: chess.Board, target_square: int) -> list[chess.Move]:
        replies: list[chess.Move] = []
        for reply in board.legal_moves:
            if reply.to_square != target_square or not board.is_capture(reply):
                continue
            replies.append(reply)
        return replies

    def _best_recapture_delta(
        self,
        board: chess.Board,
        *,
        mover: chess.Color,
        before_material: int,
        target_square: int,
    ) -> int:
        best_delta = int(self._material_balance(board, mover) - before_material)
        for reply in board.legal_moves:
            if reply.to_square != target_square or not board.is_capture(reply):
                continue
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



    def search(
        self,
        board: chess.Board,
        add_noise: bool = False,
        num_simulations: int | None = None,
        temperature: float | None = None,
    ) -> dict:
        if not isinstance(board, chess.Board):
            raise TypeError("Expected board to be chess.Board")

        self._tactical_penalty_cache.clear()
        sims = max(1, int(num_simulations if num_simulations is not None else self.cfg.mcts.num_simulations))
        temperature = float(self.cfg.mcts.temperature if temperature is None else temperature)
        batch_limit = max(1, int(self.cfg.mcts.inference_batch_size))
        root_seen_positions = build_seen_positions(board)

        self.logger.debug(
            "mcts start fen=%s sims=%s temperature=%.3f add_noise=%s",
            board.fen(),
            sims,
            temperature,
            add_noise,
        )

        root = Node(prior=0.0)
        initial_root_value = self._expand_node(root, board, add_noise=add_noise)
        pending_simulations = sims
        diagnostics = self._new_penalty_diagnostics() if self._penalty_diagnostics_enabled() else None

        while pending_simulations > 0:
            rollout_batch = min(batch_limit, pending_simulations)
            pending_simulations -= rollout_batch
            pending_groups: dict[str, list[tuple[Node, list[Node], chess.Board]]] = {}

            for _ in range(rollout_batch):
                node = root
                sim_board = board.copy(stack=True)
                sim_seen_positions = dict(root_seen_positions)
                search_path = [node]

                while node.expanded() and not self._is_terminal_board(sim_board):
                    move, next_node = self._select_child(node, sim_board, sim_seen_positions, diagnostics=diagnostics)
                    if move is None or next_node is None:
                        break

                    sim_board.push(move)
                    key = position_key(sim_board)
                    sim_seen_positions[key] = sim_seen_positions.get(key, 0) + 1

                    node = next_node
                    search_path.append(node)

                if self._is_terminal_board(sim_board):
                    self._backpropagate(search_path, self._terminal_value(sim_board))
                    continue

                self._reserve_virtual_path(search_path)
                leaf_key = self._prediction_key(sim_board)
                pending_groups.setdefault(leaf_key, []).append((node, search_path, sim_board))

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

        visit_counts = {move: child.visit_count for move, child in root.children.items()}
        raw_policy_target = self._visit_policy(root, temperature=temperature)
        adjusted_policy_target, root_repetition_counts = self._adjust_root_policy(board, raw_policy_target, root_seen_positions)
        best_move = self._select_root_move(board, root, adjusted_policy_target or raw_policy_target, root_seen_positions)
        root_diagnostics = self._root_move_diagnostics(
            board,
            root,
            raw_policy_target,
            adjusted_policy_target,
            root_seen_positions,
        )
        root_value = float(root.q) if root.visit_count > 0 else float(initial_root_value)

        self.logger.debug(
            "mcts done sims=%s expanded_children=%s root_visits=%s best_move=%s root_value=%.4f",
            sims,
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
        return self._blend_value(board, nn_value)

    def _blend_value(self, board: chess.Board, nn_value: float) -> float:
        classical_alpha = min(max(float(self.cfg.mcts.classical_value_alpha), 0.0), 1.0)
        classical_eval = float(evaluate_board(board))
        if board.turn == chess.BLACK:
            classical_eval *= -1.0
        classical_value = float(np.tanh(classical_eval / 600.0))
        blended = classical_alpha * classical_value + (1.0 - classical_alpha) * float(nn_value)
        blended -= self._progress_penalty(board)
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

    def _select_child(
        self,
        node: Node,
        board: chess.Board,
        seen_positions: Mapping[PositionKey, int] | None = None,
        diagnostics: dict | None = None,
    ):
        if not node.children:
            return None, None

        best_score = -float("inf")
        best_raw_score = -float("inf")
        best_move = None
        best_raw_move = None
        best_child = None
        parent_visits = max(1, node.total_visit_count)

        for move, child in node.children.items():
            q_value = -child.q
            u_value = self.c_puct * child.prior * math.sqrt(parent_visits) / (1 + child.total_visit_count)
            virtual_penalty = self.virtual_loss * float(child.virtual_visits)
            raw_score = q_value + u_value - virtual_penalty
            if diagnostics is None:
                move_penalty = self._move_penalty(board, move, seen_positions)
            else:
                components = self._move_penalty_components(board, move, seen_positions)
                self._record_penalty_diagnostics(diagnostics, components)
                move_penalty = float(sum(components.values()))
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

    def _move_penalty(self, board: chess.Board, move: chess.Move, seen_positions: Mapping[PositionKey, int] | None = None) -> float:
        return float(sum(self._move_penalty_components(board, move, seen_positions).values()))

    def _move_penalty_components(self, board: chess.Board, move: chess.Move, seen_positions: Mapping[PositionKey, int] | None = None) -> dict[str, float]:
        oscillation_penalty = self._oscillation_penalty(board, move)
        before_halfmove = int(getattr(board, "halfmove_clock", 0))
        was_capture = bool(board.is_capture(move))
        moving_piece = board.piece_at(move.from_square)
        mover = board.turn
        before_material = self._material_balance(board, mover)
        before_position_key = position_key(board)
        before_board = board.copy(stack=True)

        board.push(move)
        try:
            tactical_penalty = 0.0
            moved_piece = board.piece_at(move.to_square)
            if moved_piece is not None and moved_piece.color == mover:
                tactical_penalty = self._piece_tactical_penalty_after_push(
                    board,
                    mover=mover,
                    piece_type=moved_piece.piece_type,
                    before_material=before_material,
                    target_square=move.to_square,
                    before_position_key=before_position_key,
                    move_uci=move.uci(),
                )

            principle_components = principle_penalty_components(
                before=before_board,
                after=board,
                move=move,
                cfg=self.cfg.principle_penalties,
            ).components

            return {
                "oscillation": float(oscillation_penalty),
                "repetition": float(self._repetition_penalty_in_position(board, seen_positions)),
                "progress": float(self._forward_progress_penalty_after_push(board, before_halfmove, was_capture)),
                "tactical": float(tactical_penalty),
                **{f"principle.{name}": float(value) for name, value in principle_components.items()},
            }
        finally:
            board.pop()

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

        cache_key = (before_position_key, move_uci)
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
            if worst_exchange_delta >= threshold:
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
        if not board.move_stack:
            return 0.0
        last_move = board.move_stack[-1]
        if move.from_square == last_move.to_square and move.to_square == last_move.from_square:
            return max(0.0, float(self.cfg.selfplay.repetition_penalty)) * 0.75
        return 0.0

    def _forward_progress_penalty_after_push(self, board: chess.Board, before_halfmove: int, was_capture: bool) -> float:
        after_halfmove = int(getattr(board, "halfmove_clock", 0))
        penalty = self._progress_penalty(board)
        if after_halfmove > before_halfmove and not was_capture:
            penalty += 0.02
        return float(penalty)

    def _adjust_root_policy(self, board: chess.Board, policy_target: dict[chess.Move, float], seen_positions: Mapping[PositionKey, int]) -> tuple[dict[chess.Move, float], dict[str, int]]:
        adjusted_policy, repetition_counts = filter_repetition_moves(
            policy_target,
            board,
            seen_positions,
            repeat_break_count=int(self.cfg.selfplay.repetition_break_count),
            repeat_weight=float(self.cfg.selfplay.repetition_move_weight),
        )
        if not adjusted_policy:
            return adjusted_policy, repetition_counts

        reweighted: dict[chess.Move, float] = {}
        for move, prob in adjusted_policy.items():
            reweighted[move] = max(0.0, float(prob) - self._move_penalty(board, move, seen_positions) * 0.05)

        total = float(sum(reweighted.values()))
        if total <= 1e-12:
            return adjusted_policy, repetition_counts
        return ({move: value / total for move, value in reweighted.items()}, repetition_counts)

    def _select_root_move(self, board: chess.Board, root: Node, policy_target: dict[chess.Move, float], seen_positions: Mapping[PositionKey, int] | None = None) -> chess.Move | None:
        if not policy_target:
            return None

        ranked_moves = sorted(
            policy_target.items(),
            key=lambda item: (
                float(item[1]),
                float(root.children[item[0]].visit_count if item[0] in root.children else 0),
                float(root.children[item[0]].q if item[0] in root.children else -1e9),
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

        for move, child in root.children.items():
            components = self._move_penalty_components(board, move, seen_positions)
            penalty = float(sum(components.values()))
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

    def _backpropagate(self, search_path: list[Node], value: float) -> None:
        current_value = float(value)
        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += current_value
            current_value = -current_value

    def _is_terminal_board(self, board: chess.Board) -> bool:
        return bool(board.is_game_over(claim_draw=True))

    def _terminal_value(self, board: chess.Board) -> float:
        outcome = board.outcome(claim_draw=True)
        if outcome is None or outcome.winner is None:
            return 0.0
        return 1.0 if outcome.winner == board.turn else -1.0

    def _visit_policy(self, root: Node, temperature: float = 1.0) -> dict[chess.Move, float]:
        visit_counts = {move: int(child.visit_count) for move, child in root.children.items()}
        return apply_temperature(visit_counts, temperature)
