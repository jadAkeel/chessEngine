from __future__ import annotations

import math
from collections import Counter

import chess

from app.core.engine import Engine
from app.game.repetition import PositionKey, count_repetition_after_move, filter_repetition_moves, position_key
from app.infra.config import AppConfig, get_current_config
from app.infra.logging import setup_logging


def _winner_label(winner: bool | None) -> str | None:
    if winner is None:
        return None
    return 'white' if winner == chess.WHITE else 'black'


_position_key = position_key


def _describe_board_state(board: chess.Board) -> dict:
    return {
        'fen': board.fen(),
        'fullmove_number': int(board.fullmove_number),
        'halfmove_clock': int(board.halfmove_clock),
        'side_to_move': 'white' if board.turn == chess.WHITE else 'black',
        'legal_moves': int(board.legal_moves.count()),
        'is_check': bool(board.is_check()),
        'can_claim_draw': bool(board.can_claim_draw()),
        'can_claim_threefold_repetition': bool(board.can_claim_threefold_repetition()),
        'can_claim_fifty_moves': bool(board.can_claim_fifty_moves()),
        'is_fivefold_repetition': bool(board.is_fivefold_repetition()),
        'is_seventyfive_moves': bool(board.is_seventyfive_moves()),
        'is_insufficient_material': bool(board.is_insufficient_material()),
        'is_stalemate': bool(board.is_stalemate()),
        'is_checkmate': bool(board.is_checkmate()),
    }


def _make_outcome_info(
    board: chess.Board,
    outcome: chess.Outcome | None = None,
    *,
    termination: str | None = None,
    winner: bool | None = None,
    result: str | None = None,
    extra: dict | None = None,
) -> dict:
    if outcome is not None:
        payload = {
            'termination': str(outcome.termination),
            'winner': _winner_label(outcome.winner),
            'result': board.result(claim_draw=True),
            **_describe_board_state(board),
        }
    else:
        payload = {
            'termination': termination,
            'winner': _winner_label(winner),
            'result': result or ('1-0' if winner == chess.WHITE else '0-1' if winner == chess.BLACK else '1/2-1/2'),
            **_describe_board_state(board),
        }

    if extra:
        payload.update(extra)
    return payload


def _build_arena_opening_positions() -> list[chess.Board]:
    opening_fens = [
        chess.STARTING_FEN,
        # Italian Game, calm but developed
        'r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQ1RK1 w kq - 2 6',
        # Sicilian with early development
        'r1bqkbnr/pp2pppp/2np4/2p5/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 w kq - 4 5',
        # Queen pawn structure
        'rnbqkb1r/pp2pppp/5n2/2pp4/3P4/2N1PN2/PPP2PPP/R1BQKB1R w KQkq - 0 4',
        # Italian with kingside castling
        'r1bqkb1r/pppp1ppp/2n2n2/4p3/2BPP3/5N2/PPP2PPP/RNBQ1RK1 b kq - 4 5',
        # Queen's Gambit Declined style structure
        'rnbqk2r/ppp2ppp/4pn2/3p4/3P4/2PBPN2/PP3PPP/RNBQ1RK1 b kq - 0 6',
        # Two Knights / Italian mix
        'r1bqkbnr/pppp1ppp/2n5/4p3/2BPP3/5N2/PPP2PPP/RNBQK2R b KQkq - 2 4',
        # Slav / semi-slav type development
        'rnbqkb1r/pp2pppp/5n2/2pp4/3P4/2N1PN2/PPP1BPPP/R1BQK2R b KQkq - 3 4',
    ]

    boards: list[chess.Board] = []
    for fen in opening_fens:
        try:
            board = chess.Board(fen)
        except ValueError:
            continue
        if board.is_valid():
            boards.append(board)

    return boards or [chess.Board()]


def _get_start_board_for_arena(game_idx: int, cfg: AppConfig, logger=None) -> chess.Board:
    use_randomized = bool(getattr(cfg.arena, 'randomize_openings', True))
    if not use_randomized:
        return chess.Board()

    opening_boards = _build_arena_opening_positions()
    if not opening_boards:
        return chess.Board()

    board = opening_boards[game_idx % len(opening_boards)].copy(stack=False)

    if logger is not None:
        logger.info(
            'arena opening game=%d start_fen=%s',
            game_idx + 1,
            board.fen(),
        )

    return board


def _candidate_sort_key(
    move_uci: str,
    visit_count: int,
    policy: dict[str, float],
    repetition_counts: dict[str, int],
    seen_positions: dict[PositionKey, int],
    board: chess.Board,
    cfg: AppConfig,
    is_candidate: bool,
) -> tuple[float, float, int, int, str]:
    move = chess.Move.from_uci(move_uci)
    repeat_count = int(repetition_counts.get(move_uci, count_repetition_after_move(board, move, seen_positions)))
    score = float(policy.get(move_uci, 0.0))

    repetition_break_count = int(getattr(cfg.arena, 'repetition_break_count', 3))
    repetition_move_weight = float(getattr(cfg.arena, 'repetition_move_weight', 0.3))

    if repeat_count >= repetition_break_count:
        if bool(getattr(cfg.arena, 'hard_block_repetition', False)):
            score = -1.0
        else:
            score *= repetition_move_weight
    elif repeat_count == repetition_break_count - 1:
        score *= max(repetition_move_weight, 0.35)

    if is_candidate:
        score += float(getattr(cfg.arena, 'contempt_factor', 0.0))

    board.push(move)
    try:
        legal_reply_count = int(board.legal_moves.count())
    finally:
        board.pop()

    avoids_repeat = 1 if repeat_count < repetition_break_count else 0
    return (score, float(visit_count), avoids_repeat, legal_reply_count, move_uci)


def _select_move_with_fallback(
    engine: Engine,
    board: chess.Board,
    cfg: AppConfig,
    seen_positions: dict[PositionKey, int],
    *,
    is_candidate: bool = False,
):
    analysis = engine.analyze(
        board,
        add_noise=False,
        num_simulations=cfg.mcts.num_simulations,
        temperature=float(cfg.arena.search_temperature),
    )

    visit_counts = analysis.visit_counts
    if not visit_counts:
        return None, analysis.score, {}

    policy_dict = {
        chess.Move.from_uci(move_uci): float(analysis.policy.get(move_uci, 0.0))
        for move_uci in visit_counts
        if chess.Move.from_uci(move_uci) in board.legal_moves
    }
    adjusted_policy, repetition_counts = filter_repetition_moves(
        policy_dict,
        board,
        seen_positions,
        repeat_break_count=int(getattr(cfg.arena, 'repetition_break_count', 3)),
        repeat_weight=float(getattr(cfg.arena, 'repetition_move_weight', 0.3)),
    )
    adjusted_policy_uci = {move.uci(): float(prob) for move, prob in adjusted_policy.items()}

    ranked_moves = sorted(
        visit_counts.items(),
        key=lambda item: _candidate_sort_key(
            item[0],
            int(item[1]),
            adjusted_policy_uci,
            repetition_counts,
            seen_positions,
            board,
            cfg,
            is_candidate,
        ),
        reverse=True,
    )
    if not ranked_moves:
        return None, analysis.score, repetition_counts

    top_k = max(1, int(getattr(cfg.arena, 'fallback_top_k', 3)))
    candidates = ranked_moves[:top_k]
    repetition_break_count = int(getattr(cfg.arena, 'repetition_break_count', 3))
    hard_block_repetition = bool(getattr(cfg.arena, 'hard_block_repetition', False))

    preferred: list[str] = []
    fallback: list[str] = []

    for move_uci, _ in candidates:
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            continue
        next_repeat = int(repetition_counts.get(move_uci, count_repetition_after_move(board, move, seen_positions)))
        if next_repeat >= repetition_break_count:
            if hard_block_repetition:
                continue
            fallback.append(move_uci)
        else:
            preferred.append(move_uci)

    chosen_pool = preferred or fallback or [ranked_moves[0][0]]
    chosen_uci = chosen_pool[0]
    return chess.Move.from_uci(chosen_uci), analysis.score, repetition_counts


def _play_engine_game(
    white_engine,
    black_engine,
    cfg: AppConfig,
    logger=None,
    game_idx: int | None = None,
    candidate_engine=None,
):
    board = _get_start_board_for_arena(game_idx or 0, cfg, logger=logger)
    plies = 0
    last_value = None
    move_history: list[str] = []

    seen_positions: dict[PositionKey, int] = {}
    repetition_break_count = int(getattr(cfg.arena, 'repetition_break_count', 3))
    repetition_soft_limit_plies = int(getattr(cfg.arena, 'repetition_soft_limit_plies', 16))

    while not board.is_game_over(claim_draw=True):
        key = _position_key(board)
        seen_positions[key] = seen_positions.get(key, 0) + 1

        if seen_positions[key] >= repetition_break_count and plies >= repetition_soft_limit_plies:
            return board, _make_outcome_info(
                board,
                termination='forced_repetition_draw',
                winner=None,
                result='1/2-1/2',
                extra={'plies': plies, 'last_value': last_value, 'last_moves': move_history[-12:]},
            ), 'forced_repetition_draw', {}

        if plies >= cfg.selfplay.max_game_length:
            return board, _make_outcome_info(
                board,
                termination='max_game_length',
                winner=None,
                result='1/2-1/2',
                extra={'plies': plies, 'last_value': last_value, 'last_moves': move_history[-12:]},
            ), 'max_game_length', {}

        engine = white_engine if board.turn == chess.WHITE else black_engine
        is_candidate_turn = engine is candidate_engine

        move, value, repetition_counts = _select_move_with_fallback(
            engine,
            board,
            cfg,
            seen_positions,
            is_candidate=is_candidate_turn,
        )
        last_value = value

        if logger is not None and repetition_counts:
            logger.debug(
                'arena candidate repetition_counts=%s',
                {k: int(v) for k, v in sorted(repetition_counts.items(), key=lambda item: (-item[1], item[0]))[:8]},
            )

        if value is not None and value < float(cfg.arena.resign_threshold):
            winner = not board.turn
            outcome_info = _make_outcome_info(
                board,
                termination='resignation_by_eval',
                winner=winner,
                extra={'plies': plies, 'last_value': value, 'last_moves': move_history[-12:]},
            )
            return board, outcome_info, 'resignation_by_eval', {}

        if move is None:
            return board, _make_outcome_info(
                board,
                termination='no_move_returned',
                winner=None,
                result='1/2-1/2',
                extra={'plies': plies, 'last_value': last_value, 'last_moves': move_history[-12:]},
            ), 'no_move_returned', {}

        if move not in board.legal_moves:
            return board, _make_outcome_info(
                board,
                termination='illegal_move',
                winner=not board.turn,
                extra={'plies': plies, 'last_value': last_value, 'last_moves': move_history[-12:]},
            ), 'illegal_move', {}

        move_history.append(move.uci())
        board.push(move)
        plies += 1

    outcome = board.outcome(claim_draw=True)
    if logger is not None and outcome is not None and outcome.termination == chess.Termination.THREEFOLD_REPETITION:
        logger.info('REPETITION DETECTED last_moves=%s', move_history[-12:])
        ranked_candidates = []
        for legal_move in board.legal_moves:
            ranked_candidates.append((legal_move.uci(), int(count_repetition_after_move(board, legal_move, seen_positions))))
        logger.info('REPETITION CANDIDATES counts=%s', ranked_candidates[:12])

    return board, _make_outcome_info(
        board,
        outcome=outcome,
        extra={'plies': plies, 'last_value': last_value, 'last_moves': move_history[-12:]},
    ), str(outcome.termination), {}


def play_match(best_model, candidate_model, device='cpu', cfg: AppConfig | None = None):
    logger = setup_logging('evaluation.arena')

    cfg = cfg or getattr(candidate_model, 'cfg', None) or getattr(best_model, 'cfg', None) or get_current_config()

    wins_new = wins_best = draws = 0
    total_games = int(cfg.arena.games)
    target = float(cfg.arena.update_threshold)

    min_games_before_stop = max(
        2,
        int(math.ceil(total_games * float(cfg.arena.early_stop_margin)))
    )

    cand_engine = Engine(model=candidate_model, cfg=cfg, device=device)
    best_engine = Engine(model=best_model, cfg=cfg, device=device)

    decision = None
    decision_reason = None
    reason_counts = Counter()

    logger.info(
        'arena start games=%d target=%.3f min_games_before_stop=%d '
        'search_temperature=%.3f resign_threshold=%.3f sims=%d '
        'repetition_break_count=%d repetition_soft_limit_plies=%d repetition_move_weight=%.3f '
        'hard_block_repetition=%s contempt_factor=%.3f randomize_openings=%s fallback_top_k=%d',
        total_games,
        target,
        min_games_before_stop,
        float(cfg.arena.search_temperature),
        float(cfg.arena.resign_threshold),
        int(cfg.mcts.num_simulations),
        int(getattr(cfg.arena, 'repetition_break_count', 3)),
        int(getattr(cfg.arena, 'repetition_soft_limit_plies', 16)),
        float(getattr(cfg.arena, 'repetition_move_weight', 0.3)),
        bool(getattr(cfg.arena, 'hard_block_repetition', False)),
        float(getattr(cfg.arena, 'contempt_factor', 0.0)),
        bool(getattr(cfg.arena, 'randomize_openings', True)),
        int(getattr(cfg.arena, 'fallback_top_k', 3)),
    )

    for game_idx in range(total_games):
        candidate_is_white = game_idx % 2 == 0

        _, outcome_info, termination_reason, _ = (
            _play_engine_game(cand_engine, best_engine, cfg, logger=logger, game_idx=game_idx, candidate_engine=cand_engine)
            if candidate_is_white
            else _play_engine_game(best_engine, cand_engine, cfg, logger=logger, game_idx=game_idx, candidate_engine=cand_engine)
        )

        reason_counts[termination_reason] += 1
        winner = outcome_info.get('winner')
        result = outcome_info.get('result')
        plies = outcome_info.get('plies')
        last_value = outcome_info.get('last_value')

        if winner is None:
            draws += 1
            game_outcome = 'draw'
        else:
            candidate_won = (winner == 'white' and candidate_is_white) or (
                winner == 'black' and not candidate_is_white
            )
            if candidate_won:
                wins_new += 1
                game_outcome = 'candidate_win'
            else:
                wins_best += 1
                game_outcome = 'champion_win'

        games_played = wins_new + wins_best + draws
        score = wins_new + 0.5 * draws

        max_possible = (score + (total_games - games_played)) / total_games
        min_possible = score / total_games
        current_win_rate = score / max(1, games_played)
        draw_rate = draws / max(1, games_played)

        logger.info(
            'arena game=%d/%d candidate_color=%s outcome=%s result=%s winner=%s '
            'termination=%s plies=%s last_value=%s score=%.1f wins_new=%d losses_new=%d draws=%d '
            'current_win_rate=%.3f draw_rate=%.3f min_possible=%.3f max_possible=%.3f',
            game_idx + 1,
            total_games,
            'white' if candidate_is_white else 'black',
            game_outcome,
            result,
            winner,
            termination_reason,
            plies,
            None if last_value is None else round(float(last_value), 4),
            score,
            wins_new,
            wins_best,
            draws,
            current_win_rate,
            draw_rate,
            min_possible,
            max_possible,
        )

        if games_played >= min_games_before_stop:
            if max_possible < target:
                decision = 'rejected'
                decision_reason = 'early_stop_max_possible_below_target'
                logger.info(
                    'arena early stop -> rejected after %d games '
                    '(max_possible=%.3f < target=%.3f)',
                    games_played,
                    max_possible,
                    target,
                )
                break
            if min_possible >= target:
                decision = 'accepted'
                decision_reason = 'early_stop_min_possible_above_target'
                logger.info(
                    'arena early stop -> accepted after %d games '
                    '(min_possible=%.3f >= target=%.3f)',
                    games_played,
                    min_possible,
                    target,
                )
                break

    games_played = wins_new + wins_best + draws
    score = wins_new + 0.5 * draws
    win_rate = score / max(1, games_played)
    draw_rate = draws / max(1, games_played)

    if draw_rate > float(cfg.arena.max_repetition_draw_rate):
        if wins_new == 0 and wins_best == 0:
            decision = 'rejected'
            decision_reason = 'inconclusive_all_draws'
        else:
            decision = 'rejected'
            decision_reason = 'draw_rate_above_limit'

        logger.info(
            'arena draw-rate guard fired: draw_rate=%.3f limit=%.3f reason=%s',
            draw_rate,
            float(cfg.arena.max_repetition_draw_rate),
            decision_reason,
        )

    if decision is None:
        decision = 'accepted' if win_rate >= target else 'rejected'
        decision_reason = 'final_win_rate_check'

    result = {
        'win_rate': float(win_rate),
        'wins_new': wins_new,
        'losses_new': wins_best,
        'draws': draws,
        'games_played': games_played,
        'accepted': decision == 'accepted',
        'decision': decision,
        'decision_reason': decision_reason,
        'termination_reasons': dict(reason_counts),
    }

    logger.info(
        'arena complete games=%d score=%.1f win_rate=%.3f wins_new=%d losses_new=%d '
        'draws=%d decision=%s reason=%s termination_reasons=%s',
        games_played,
        score,
        win_rate,
        wins_new,
        wins_best,
        draws,
        decision,
        decision_reason,
        dict(reason_counts),
    )
    return result
