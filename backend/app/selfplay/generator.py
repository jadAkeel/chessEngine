from __future__ import annotations

import multiprocessing as mp
import random
import time
from collections import Counter

import chess
import numpy as np

from app.core.engine import Engine
from app.game.board_encoding import encode_board
from app.game.move_encoding import move_to_index
from app.game.repetition import PositionKey, filter_repetition_moves, position_key, position_key_token
from app.infra.config import AppConfig, apply_overrides, config_to_dict, get_current_config
from app.infra.logging import setup_logging
from app.infra.runtime import configure_torch_runtime
from app.model.network import ChessNet
from app.training.replay_buffer import PackedPolicy

_WORKER_ENGINE: Engine | None = None
_WORKER_CFG: AppConfig | None = None


def _init_selfplay_worker(model_state_dict: dict, device: str, cfg_dict: dict) -> None:
    global _WORKER_ENGINE, _WORKER_CFG
    worker_cfg = apply_overrides(cfg_dict)
    configure_torch_runtime(worker_cfg, device=str(device), role='selfplay_worker', worker_count=max(1, int(worker_cfg.selfplay.num_workers)))
    model = ChessNet(worker_cfg)
    model.load_state_dict(model_state_dict, strict=False)
    model.eval()
    _WORKER_CFG = worker_cfg
    _WORKER_ENGINE = Engine(model=model, cfg=worker_cfg, device=device)


def _temperature_for_ply(ply: int, cfg: AppConfig) -> float:
    if ply < cfg.selfplay.temperature_high_moves:
        return cfg.selfplay.temperature_high
    if ply < cfg.selfplay.temperature_mid_moves:
        return cfg.selfplay.temperature_mid
    return cfg.selfplay.temperature_low


def _policy_dict_to_sparse(policy_dict: dict[chess.Move, float], board: chess.Board) -> PackedPolicy:
    if not policy_dict:
        return PackedPolicy(np.zeros((0,), dtype=np.uint16), np.zeros((0,), dtype=np.float16))

    indexed_moves = []
    for move, prob in policy_dict.items():
        if not np.isfinite(prob) or prob <= 0.0:
            continue
        indexed_moves.append((move_to_index(move, board), float(prob)))

    if not indexed_moves:
        return PackedPolicy(np.zeros((0,), dtype=np.uint16), np.zeros((0,), dtype=np.float16))

    indexed_moves.sort(key=lambda item: item[0])
    indices = np.fromiter((idx for idx, _ in indexed_moves), dtype=np.uint16, count=len(indexed_moves))
    probs = np.fromiter((prob for _, prob in indexed_moves), dtype=np.float32, count=len(indexed_moves))
    total = float(probs.sum())
    if total > 1e-12:
        probs = probs / total
    else:
        probs = np.zeros_like(probs, dtype=np.float32)
    return PackedPolicy(indices=indices, probs=probs.astype(np.float16, copy=False))


def _sample_move(policy_dict: dict[chess.Move, float], fallback: chess.Move | None):
    if not policy_dict:
        return fallback
    moves = list(policy_dict.keys())
    probs = np.array(list(policy_dict.values()), dtype=np.float64)
    total = float(probs.sum())
    if total <= 1e-12 or not np.isfinite(total):
        return fallback
    probs = probs / total
    idx = np.random.choice(len(moves), p=probs)
    return moves[int(idx)]


_position_key = position_key


def _describe_board_state(board: chess.Board) -> dict:
    return {
        "fen": board.fen(),
        "fullmove_number": int(board.fullmove_number),
        "halfmove_clock": int(board.halfmove_clock),
        "turn": "white" if board.turn == chess.WHITE else "black",
        "legal_moves": int(board.legal_moves.count()),
        "is_check": bool(board.is_check()),
        "can_claim_draw": bool(board.can_claim_draw()),
        "can_claim_threefold_repetition": bool(board.can_claim_threefold_repetition()),
        "can_claim_fifty_moves": bool(board.can_claim_fifty_moves()),
        "is_fivefold_repetition": bool(board.is_fivefold_repetition()),
        "is_seventyfive_moves": bool(board.is_seventyfive_moves()),
        "is_insufficient_material": bool(board.is_insufficient_material()),
        "is_stalemate": bool(board.is_stalemate()),
        "is_checkmate": bool(board.is_checkmate()),
    }


def _top_policy_moves(policy_dict: dict[chess.Move, float] | None, top_k: int = 3) -> list[tuple[str, float]]:
    if not policy_dict:
        return []
    top_moves = sorted(policy_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [(move.uci(), float(round(prob, 4))) for move, prob in top_moves]


def _empty_penalty_diagnostics() -> dict:
    return {
        "components": {},
        "total_move_penalty": {"count": 0, "sum": 0.0, "max": 0.0},
        "thresholds": {"gt_0.25": 0, "gt_0.5": 0, "gt_0.75": 0, "gt_1.0": 0},
        "ranking_changed": 0,
        "ranking_comparisons": 0,
    }


def _merge_penalty_diagnostics(target: dict, update: dict | None) -> None:
    if not update:
        return

    total = update.get("total_move_penalty", {})
    target_total = target["total_move_penalty"]
    target_total["count"] += int(total.get("count", 0))
    target_total["sum"] += float(total.get("sum", 0.0))
    target_total["max"] = max(float(target_total["max"]), float(total.get("max", 0.0)))

    for name, stats in (update.get("components") or {}).items():
        count = int(stats.get("count", 0))
        avg = float(stats.get("avg", 0.0))
        component = target["components"].setdefault(name, {"count": 0, "sum": 0.0, "max": 0.0})
        component["count"] += count
        component["sum"] += avg * count
        component["max"] = max(float(component["max"]), float(stats.get("max", 0.0)))

    for name, count in (update.get("thresholds") or {}).items():
        target["thresholds"][name] = int(target["thresholds"].get(name, 0)) + int(count)

    target["ranking_changed"] += int(update.get("ranking_changed", 0))
    target["ranking_comparisons"] += int(update.get("ranking_comparisons", 0))


def _finalize_penalty_diagnostics(stats: dict, *, games: int | None = None) -> dict:
    components = {}
    for name, component in stats["components"].items():
        count = int(component["count"])
        components[name] = {
            "count": count,
            "avg": float(component["sum"]) / count if count else 0.0,
            "max": float(component["max"]),
        }

    total = stats["total_move_penalty"]
    total_count = int(total["count"])
    result = {
        "components": components,
        "total_move_penalty": {
            "count": total_count,
            "sum": float(total["sum"]),
            "avg": float(total["sum"]) / total_count if total_count else 0.0,
            "max": float(total["max"]),
        },
        "thresholds": dict(stats["thresholds"]),
        "ranking_changed": int(stats["ranking_changed"]),
        "ranking_comparisons": int(stats["ranking_comparisons"]),
    }
    if games is not None:
        result["avg_total_penalty_per_game"] = float(total["sum"]) / max(1, int(games))
    return result


def _terminal_value_from_result(winner: bool | None) -> float:
    if winner is None:
        return 0.0
    return 1.0 if winner == chess.WHITE else -1.0


def _official_outcome_name(outcome: chess.Outcome | None) -> str | None:
    if outcome is None:
        return None
    termination = getattr(outcome, "termination", None)
    if termination is None:
        return None
    return str(termination)


def _play_single_game(engine: Engine, cfg: AppConfig, logger=None, game_label: str | None = None):
    board = chess.Board()
    samples = []
    seen_positions: dict[PositionKey, int] = {}
    opening_keys = []
    repeated_positions = 0
    move_history: list[str] = []
    value_history: list[float] = []
    policy_top3_history: list[list[tuple[str, float]]] = []
    penalty_diagnostics = _empty_penalty_diagnostics() if cfg.penalty_diagnostics.enabled else None
    random_opening_plies = random.randint(cfg.selfplay.opening_random_plies_min, cfg.selfplay.opening_random_plies_max)
    winner = None
    custom_terminal_value: float | None = None
    termination_reason = None
    termination_detail = {}
    started_at = time.perf_counter()

    if logger is not None:
        logger.info(
            "self-play game start label=%s max_len=%s sims=%s opening_random_plies=%s",
            game_label or "single",
            cfg.selfplay.max_game_length,
            cfg.mcts.num_simulations,
            random_opening_plies,
        )

    for ply in range(cfg.selfplay.max_game_length):
        if logger is not None and (ply == 0 or (ply + 1) % 10 == 0):
            max_repeat_so_far = max(seen_positions.values(), default=0)
            logger.info(
                "self-play progress label=%s ply=%s legal_moves=%s halfmove=%s repeats=%s max_repeat=%s unique_positions=%s fen=%s",
                game_label or "single",
                ply + 1,
                board.legal_moves.count(),
                board.halfmove_clock,
                repeated_positions,
                max_repeat_so_far,
                len(seen_positions),
                board.fen(),
            )

        if board.is_game_over(claim_draw=False):
            outcome = board.outcome(claim_draw=False)
            termination_reason = f"official_{str(outcome.termination).lower()}" if outcome else "official_game_over"
            termination_detail = {
                **_describe_board_state(board),
                "last_moves": move_history[-10:],
                "last_values": [round(v, 4) for v in value_history[-10:]],
                "last_policy_top3": policy_top3_history[-5:],
                "max_repeat": int(max(seen_positions.values(), default=0)),
                "unique_positions": int(len(seen_positions)),
            }
            break

        key = _position_key(board)
        seen_positions[key] = seen_positions.get(key, 0) + 1

        if seen_positions[key] >= cfg.selfplay.repetition_break_count:
            termination_reason = "manual_threefold_repetition_break"
            repetition_culprit = not board.turn
            custom_terminal_value = float(cfg.selfplay.repetition_draw_value)
            if repetition_culprit == chess.WHITE:
                custom_terminal_value *= -1.0
            termination_detail = {
                "position_key": key,
                "count": int(seen_positions[key]),
                "repetition_culprit": "white" if repetition_culprit == chess.WHITE else "black",
                "custom_terminal_value_white": float(custom_terminal_value),
                "last_moves": move_history[-10:],
                "last_values": [round(v, 4) for v in value_history[-10:]],
                "last_policy_top3": policy_top3_history[-5:],
                "max_repeat": int(max(seen_positions.values(), default=0)),
                "unique_positions": int(len(seen_positions)),
                **_describe_board_state(board),
            }
            break

        if seen_positions[key] > 1:
            repeated_positions += 1

        if ply < 8:
            opening_keys.append(position_key_token(key))

        temperature = _temperature_for_ply(ply, cfg)
        search = engine.mcts.search(
            board=board,
            add_noise=True,
            num_simulations=cfg.mcts.num_simulations,
            temperature=temperature,
        )
        base_best_move = search["best_move"]
        raw_policy_dict = search["policy_target"]
        adjusted_policy_dict = search.get("adjusted_policy_target")
        if adjusted_policy_dict is not None:
            filtered_policy_dict = adjusted_policy_dict
            repetition_candidates = search.get("root_repetition_counts", {})
        else:
            filtered_policy_dict, repetition_candidates = filter_repetition_moves(
                raw_policy_dict,
                board,
                seen_positions,
                repeat_break_count=int(cfg.selfplay.repetition_break_count),
                repeat_weight=float(cfg.selfplay.repetition_move_weight),
            )
        best_move = max(filtered_policy_dict, key=filtered_policy_dict.get) if filtered_policy_dict else base_best_move
        top3 = _top_policy_moves(filtered_policy_dict, top_k=3)
        move = _sample_move(filtered_policy_dict, best_move) if ply < random_opening_plies else best_move
        root_value = float(search.get("root_value", 0.0))
        if penalty_diagnostics is not None:
            _merge_penalty_diagnostics(penalty_diagnostics, search.get("penalty_diagnostics"))

        if move is None:
            termination_reason = "no_move_from_search"
            termination_detail = {
                "root_value": root_value,
                "policy_size": len(filtered_policy_dict) if filtered_policy_dict is not None else 0,
                "policy_top3": top3,
                "repetition_candidates": repetition_candidates,
                "last_moves": move_history[-10:],
                "last_values": [round(v, 4) for v in value_history[-10:]],
                "last_policy_top3": policy_top3_history[-5:],
                "max_repeat": int(max(seen_positions.values(), default=0)),
                "unique_positions": int(len(seen_positions)),
                **_describe_board_state(board),
            }
            break

        if move not in board.legal_moves:
            termination_reason = "illegal_move_from_search"
            termination_detail = {
                "move": move.uci(),
                "root_value": root_value,
                "policy_size": len(filtered_policy_dict) if filtered_policy_dict is not None else 0,
                "policy_top3": top3,
                "repetition_candidates": repetition_candidates,
                "last_moves": move_history[-10:],
                "last_values": [round(v, 4) for v in value_history[-10:]],
                "last_policy_top3": policy_top3_history[-5:],
                "max_repeat": int(max(seen_positions.values(), default=0)),
                "unique_positions": int(len(seen_positions)),
                **_describe_board_state(board),
            }
            break

        samples.append(
            {
                "state": encode_board(board, engine.cfg),
                "policy": _policy_dict_to_sparse(filtered_policy_dict, board),
                "player": board.turn,
            }
        )

        played_move_uci = move.uci()
        best_move_uci = None if best_move is None else best_move.uci()

        move_history.append(played_move_uci)
        value_history.append(root_value)
        policy_top3_history.append(top3)

        if logger is not None and (ply < 5 or (ply + 1) % 10 == 0):
            logger.info(
                "self-play decision label=%s ply=%s turn=%s best_move=%s played_move=%s root_value=%.4f top3=%s repeat_count=%s future_repeat=%s",
                game_label or "single",
                ply + 1,
                "white" if board.turn == chess.WHITE else "black",
                best_move_uci,
                played_move_uci,
                root_value,
                top3,
                seen_positions.get(key, 0),
                repetition_candidates.get(played_move_uci),
            )

        if ply >= cfg.mcts.min_resign_plies and root_value < cfg.mcts.resign_threshold:
            winner = not board.turn
            termination_reason = "resignation_by_root_value"
            termination_detail = {
                "root_value": root_value,
                "threshold": float(cfg.mcts.resign_threshold),
                "side_to_move": "white" if board.turn == chess.WHITE else "black",
                "best_move": best_move_uci,
                "played_move": played_move_uci,
                "policy_top3": top3,
                "repetition_candidates": repetition_candidates,
                "last_moves": move_history[-10:],
                "last_values": [round(v, 4) for v in value_history[-10:]],
                "last_policy_top3": policy_top3_history[-5:],
                "max_repeat": int(max(seen_positions.values(), default=0)),
                "unique_positions": int(len(seen_positions)),
                **_describe_board_state(board),
            }
            break

        board.push(move)

    if termination_reason is None:
        termination_reason = "max_game_length_reached"
        custom_terminal_value = float(cfg.selfplay.max_length_draw_value)
        termination_detail = {
            "max_game_length": int(cfg.selfplay.max_game_length),
            "custom_terminal_value_white": float(custom_terminal_value),
            "last_moves": move_history[-10:],
            "last_values": [round(v, 4) for v in value_history[-10:]],
            "last_policy_top3": policy_top3_history[-5:],
            "max_repeat": int(max(seen_positions.values(), default=0)),
            "unique_positions": int(len(seen_positions)),
            **_describe_board_state(board),
        }

    outcome = None
    official_outcome = None

    if winner is None:
        outcome = board.outcome(claim_draw=False)
        winner = None if outcome is None else outcome.winner
        official_outcome = _official_outcome_name(outcome)
    else:
        # resignation is custom engine logic, not a python-chess Termination enum
        official_outcome = termination_reason

    if custom_terminal_value is None:
        value_for_white = _terminal_value_from_result(winner)
    else:
        value_for_white = float(custom_terminal_value)

    game_data = []
    for sample in samples:
        value = value_for_white if sample["player"] == chess.WHITE else -value_for_white
        game_data.append((sample["state"], sample["policy"], float(value)))

    elapsed = time.perf_counter() - started_at
    max_repeat = max(seen_positions.values(), default=0)
    unique_positions = len(seen_positions)

    meta = {
        "plies": len(samples),
        "opening_key": "|".join(opening_keys[:6]),
        "repeated_positions": repeated_positions,
        "elapsed_sec": float(elapsed),
        "termination_reason": termination_reason,
        "termination_detail": termination_detail,
        "final_fen": board.fen(),
        "winner": None if winner is None else ("white" if winner == chess.WHITE else "black"),
        "official_outcome": official_outcome,
        "last_moves": move_history[-10:],
        "last_values": [round(v, 4) for v in value_history[-10:]],
        "last_policy_top3": policy_top3_history[-5:],
        "max_repeat": int(max_repeat),
        "unique_positions": int(unique_positions),
        "custom_terminal_value_white": None if custom_terminal_value is None else float(custom_terminal_value),
    }
    if penalty_diagnostics is not None:
        meta["penalty_diagnostics"] = _finalize_penalty_diagnostics(penalty_diagnostics, games=1)

    if logger is not None:
        logger.info(
            "self-play game end label=%s plies=%s repeats=%s max_repeat=%s unique_positions=%s elapsed=%.2fs reason=%s detail=%s winner=%s outcome=%s last_moves=%s last_values=%s final_fen=%s",
            game_label or "single",
            meta["plies"],
            meta["repeated_positions"],
            meta["max_repeat"],
            meta["unique_positions"],
            meta["elapsed_sec"],
            meta["termination_reason"],
            meta["termination_detail"],
            meta["winner"],
            meta["official_outcome"],
            meta["last_moves"],
            meta["last_values"],
            meta["final_fen"],
        )

    return game_data, meta


def _worker_play_one_game(_: int):
    if _WORKER_ENGINE is None or _WORKER_CFG is None:
        raise RuntimeError("Worker not initialized")
    return _play_single_game(_WORKER_ENGINE, _WORKER_CFG)


def generate_self_play_data(model, device="cpu", num_workers=None, games_per_worker=None, cfg: AppConfig | None = None):
    logger = setup_logging("selfplay.generator")
    cfg = cfg or getattr(model, "cfg", None) or get_current_config()
    workers = max(1, int(num_workers or cfg.selfplay.num_workers))
    games_each = max(1, int(games_per_worker or cfg.selfplay.games_per_worker))
    total_games = workers * games_each
    all_samples = []
    game_lengths = []
    repeated = []
    openings = []
    max_repeats = []
    unique_positions_list = []
    reason_counts = Counter()
    penalty_diagnostics = _empty_penalty_diagnostics() if cfg.penalty_diagnostics.enabled else None
    use_mp = workers > 1

    configure_torch_runtime(cfg, device=str(device), role='selfplay_main', worker_count=workers)

    logger.info(
        "selfplay start workers=%s games_per_worker=%s total_games=%s device=%s sims=%s",
        workers,
        games_each,
        total_games,
        device,
        cfg.mcts.num_simulations,
    )

    if use_mp:
        model_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        runtime_cfg = config_to_dict(cfg)
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=workers,
            initializer=_init_selfplay_worker,
            initargs=(model_state, str(device), runtime_cfg),
        ) as pool:
            completed_games = 0
            for game_data, meta in pool.imap_unordered(_worker_play_one_game, range(total_games), chunksize=1):
                completed_games += 1
                all_samples.extend(game_data)
                game_lengths.append(meta["plies"])
                repeated.append(meta["repeated_positions"])
                openings.append(meta["opening_key"])
                max_repeats.append(meta.get("max_repeat", 0))
                unique_positions_list.append(meta.get("unique_positions", 0))
                reason_counts[meta.get("termination_reason", "unknown")] += 1
                if penalty_diagnostics is not None:
                    _merge_penalty_diagnostics(penalty_diagnostics, meta.get("penalty_diagnostics"))
                logger.info(
                    "selfplay completed=%s/%s samples=%s last_plies=%s last_elapsed=%.2fs last_reason=%s last_max_repeat=%s last_unique_positions=%s last_moves=%s last_values=%s",
                    completed_games,
                    total_games,
                    len(all_samples),
                    meta["plies"],
                    meta.get("elapsed_sec", 0.0),
                    meta.get("termination_reason"),
                    meta.get("max_repeat", 0),
                    meta.get("unique_positions", 0),
                    meta.get("last_moves", []),
                    meta.get("last_values", []),
                )
    else:
        engine = Engine(model=model, cfg=cfg, device=device)
        for game_index in range(total_games):
            logger.info("selfplay dispatch game=%s/%s", game_index + 1, total_games)
            game_data, meta = _play_single_game(engine, cfg, logger=logger, game_label=f"{game_index + 1}/{total_games}")
            all_samples.extend(game_data)
            game_lengths.append(meta["plies"])
            repeated.append(meta["repeated_positions"])
            openings.append(meta["opening_key"])
            max_repeats.append(meta.get("max_repeat", 0))
            unique_positions_list.append(meta.get("unique_positions", 0))
            reason_counts[meta.get("termination_reason", "unknown")] += 1
            if penalty_diagnostics is not None:
                _merge_penalty_diagnostics(penalty_diagnostics, meta.get("penalty_diagnostics"))

    unique_openings = len(set(openings)) if openings else 0
    opening_diversity = unique_openings / len(openings) if openings else 0.0
    draw_reasons = {
        "manual_threefold_repetition_break",
        "official_termination.threefold_repetition",
        "official_termination.fivefold_repetition",
    }
    draw_count = sum(
        count
        for reason, count in reason_counts.items()
        if "draw" in str(reason) or reason in draw_reasons or str(reason).endswith("stalemate")
    )
    repetition_draw_count = sum(
        count
        for reason, count in reason_counts.items()
        if "repetition" in str(reason)
    )
    stats = {
        "games": total_games,
        "workers": workers,
        "games_per_worker": games_each,
        "avg_game_length": float(np.mean(game_lengths)) if game_lengths else 0.0,
        "opening_diversity": float(opening_diversity),
        "unique_openings": int(unique_openings),
        "avg_repeated_positions": float(np.mean(repeated)) if repeated else 0.0,
        "avg_max_repeat": float(np.mean(max_repeats)) if max_repeats else 0.0,
        "avg_unique_positions": float(np.mean(unique_positions_list)) if unique_positions_list else 0.0,
        "draw_rate": float(draw_count / max(1, total_games)),
        "repetition_draw_frequency": float(repetition_draw_count / max(1, total_games)),
        "termination_reasons": dict(reason_counts),
    }
    if penalty_diagnostics is not None:
        stats["penalty_diagnostics"] = _finalize_penalty_diagnostics(penalty_diagnostics, games=total_games)
    logger.info(
        "selfplay done games=%s workers=%s samples=%s avg_len=%.2f avg_repeats=%.2f avg_max_repeat=%.2f avg_unique_positions=%.2f reasons=%s",
        stats["games"],
        stats["workers"],
        len(all_samples),
        stats["avg_game_length"],
        stats["avg_repeated_positions"],
        stats["avg_max_repeat"],
        stats["avg_unique_positions"],
        stats["termination_reasons"],
    )
    if cfg.penalty_diagnostics.enabled:
        logger.info("penalty diagnostics=%s", stats.get("penalty_diagnostics", {}))
    return all_samples, stats
