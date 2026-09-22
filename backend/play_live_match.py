"""
Live full match demonstration:
Dynamic Engine (White) vs Tactical Benchmark Engine (Black)
Time control: 3 minutes + 2 seconds increment (Blitz 3+2)
"""

import time
import chess
import chess.pgn
from app.core.engine import Engine
from app.evaluation.benchmark import find_best_move
from app.cli.lichess_bot import (
    calculate_dynamic_thinking,
    _fast_policy_move,
    _fastmove_complexity,
    _is_decisive_fast_choice,
    _should_use_adaptive_search,
    _is_light_adaptive_search,
    _adaptive_simulation_steps,
    _best_move_with_mcts,
    _move_safety_flags,
    _has_safety_risk,
    _safe_candidate_fallback,
    _legal_moves_with_probs,
    COMPLEX_TOPK_BOOST_THRESHOLD,
    COMPLEX_TOPK_MAX,
)

def compute_dynamic_bot_move(engine: Engine, board: chess.Board, time_left: float, inc: float):
    sims, allocated_time, urgency = calculate_dynamic_thinking(
        board=board,
        time_left_sec=time_left,
        increment_sec=inc,
        max_sims=256,
    )
    
    start_time = time.monotonic()
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None, 0.0, "NONE", 0, urgency
    if len(legal_moves) == 1:
        elapsed = time.monotonic() - start_time
        return legal_moves[0].uci(), elapsed, "FORCED", 1, urgency

    import torch
    with torch.no_grad():
        logits, _ = engine.model.predict(board, device=engine.device)

    raw_policy_top = _legal_moves_with_probs(board, logits, 8)
    fast_move, fast_san, candidates = _fast_policy_move(board, logits, 8, raw_policy_top)
    complexity, adaptive_reasons = _fastmove_complexity(board, candidates)

    if complexity >= COMPLEX_TOPK_BOOST_THRESHOLD:
        raw_policy_top = _legal_moves_with_probs(board, logits, COMPLEX_TOPK_MAX)
        fast_move, fast_san, candidates = _fast_policy_move(board, logits, COMPLEX_TOPK_MAX, raw_policy_top)

    decisive_fast = _is_decisive_fast_choice(board, candidates, adaptive_reasons)
    if urgency == "critical" and sims >= 32:
        decisive_fast = False
        complexity = max(complexity, 4)
        adaptive_reasons.append("critical_position_urgency")

    if sims >= 180:
        depth = 8
    elif sims >= 120:
        depth = 7
    elif sims >= 64:
        depth = 5
    elif sims >= 32:
        depth = 4
    else:
        depth = 2

    full_adaptive = bool(not decisive_fast and _should_use_adaptive_search(complexity, adaptive_reasons, depth))
    light_adaptive = bool(not full_adaptive and not decisive_fast and _is_light_adaptive_search(complexity, adaptive_reasons, depth))
    use_adaptive = bool((full_adaptive or light_adaptive) and sims > 16)
    light_budget = bool(not full_adaptive)

    if not use_adaptive:
        elapsed = time.monotonic() - start_time
        return fast_move, elapsed, "FAST_MOVE", 0, urgency

    simulation_steps = _adaptive_simulation_steps(depth, complexity, sims, light=light_budget)
    if not simulation_steps:
        elapsed = time.monotonic() - start_time
        return fast_move, elapsed, "FAST_MOVE", 0, urgency

    chosen_move = fast_move
    actual_sims = 0
    for step_sims in simulation_steps:
        actual_sims = step_sims
        if time.monotonic() - start_time >= allocated_time * 0.95:
            break

        mcts_move, mcts_san = _best_move_with_mcts(
            engine.model, board, engine.device, step_sims, include_diagnostics=False
        )
        mcts_safety = _move_safety_flags(board, mcts_move)
        safe_fallback = _safe_candidate_fallback(board, candidates) if _has_safety_risk(mcts_safety) else None
        if safe_fallback is not None and safe_fallback[0] != mcts_move:
            if any(s > step_sims for s in simulation_steps):
                continue
            chosen_move = safe_fallback[0]
            break

        chosen_move = mcts_move
        if step_sims == simulation_steps[-1]:
            break

    final_safety = _move_safety_flags(board, chosen_move)
    if _has_safety_risk(final_safety):
        safe_fallback = _safe_candidate_fallback(board, candidates)
        if safe_fallback is not None and safe_fallback[0] != chosen_move:
            chosen_move = safe_fallback[0]

    elapsed = time.monotonic() - start_time
    return chosen_move, elapsed, "ADAPTIVE_MCTS", actual_sims, urgency

def main():
    print("=" * 65)
    print("      LIVE MATCH DEMO: DYNAMIC ENGINE vs TACTICAL BENCHMARK      ")
    print("           Time Control: 3 min + 2s increment (Blitz)            ")
    print("=" * 65)

    engine = Engine(model_path="models/best_model.pth")
    board = chess.Board()

    white_clock = 180.0
    black_clock = 180.0
    increment = 2.0

    fast_move_count = 0
    adaptive_mcts_count = 0

    game = chess.pgn.Game()
    game.headers["Event"] = "Antigravity Live Engine Match"
    game.headers["White"] = "Dynamic FastMove Engine"
    game.headers["Black"] = "Tactical Benchmark (Depth 2)"
    node = game

    ply = 0
    max_plies = 80  # Limit to 40 full moves for demonstration

    while not board.is_game_over(claim_draw=True) and ply < max_plies:
        ply += 1
        move_num = (ply + 1) // 2

        if board.turn == chess.WHITE:
            move_uci, elapsed, mode, sims, urg = compute_dynamic_bot_move(
                engine, board, white_clock, increment
            )
            white_clock = max(0.1, white_clock - elapsed + increment)
            
            if mode == "FAST_MOVE":
                fast_move_count += 1
            elif mode == "ADAPTIVE_MCTS":
                adaptive_mcts_count += 1

            move = chess.Move.from_uci(move_uci)
            san = board.san(move)
            print(f"Move {move_num:2d}. [WHITE] {san:6s} | Mode: {mode:13s} (Sims: {sims:2d}) | Time: {elapsed:5.3f}s | Clock: {white_clock:5.1f}s | Urg: {urg}")
            board.push(move)
            node = node.add_variation(move)

        else:
            t0 = time.monotonic()
            opp_uci, score = find_best_move(board.fen(), depth=2)
            elapsed = time.monotonic() - t0
            black_clock = max(0.1, black_clock - elapsed + increment)

            move = chess.Move.from_uci(opp_uci)
            san = board.san(move)
            print(f"     ...  [BLACK] {san:6s} | Tactical Minimax          | Time: {elapsed:5.3f}s | Clock: {black_clock:5.1f}s")
            board.push(move)
            node = node.add_variation(move)

        if white_clock <= 0 or black_clock <= 0:
            print("\n>> Match ended on time!")
            break

    print("\n" + "=" * 65)
    print("                     MATCH FINISHED                      ")
    print("=" * 65)
    outcome = board.outcome(claim_draw=True)
    if outcome:
        print(f"Outcome: {outcome.termination.name} | Winner: {outcome.winner}")
        game.headers["Result"] = outcome.result()
    else:
        print(f"Finished at move {ply // 2} (Reached move limit)")
        game.headers["Result"] = "*"

    print(f"Final White Clock: {white_clock:.1f}s (Remaining time conserved!)")
    print(f"Total White Fast Moves: {fast_move_count} (Lightning fast moves)")
    print(f"Total White Adaptive MCTS: {adaptive_mcts_count} (Deep tactical searches)")
    print(f"\nPGN:\n{game}\n")

if __name__ == "__main__":
    main()
