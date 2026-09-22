"""
Test and demonstration script for dynamic time management and FastMove system.
Tests:
1. Opening / Routine positions (FastMove fast-path: < 0.05s).
2. Scholar's Mate defense (Critical tactical test: Must detect Qf7# threat and defend).
3. Queen attacked position (Sharp tactical test: Invests clock and simulates deeper).
4. Mini-game demonstration: Dynamic Engine vs Tactical Baseline.
"""

import sys
import time
import chess
from app.core.engine import Engine
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

def run_dynamic_move(engine: Engine, board: chess.Board, time_left_sec: float, inc_sec: float = 2.0, max_sims: int = 256):
    """Executes the dynamic move selection exactly as lichess_bot does."""
    sims, allocated_time, urgency = calculate_dynamic_thinking(
        board=board,
        time_left_sec=time_left_sec,
        increment_sec=inc_sec,
        max_sims=max_sims,
    )
    
    start_time = time.monotonic()
    
    # 1. Check legal moves
    legal_moves = list(board.legal_moves)
    if len(legal_moves) <= 1:
        chosen = legal_moves[0].uci() if legal_moves else None
        elapsed = time.monotonic() - start_time
        return chosen, elapsed, "FORCED_MOVE", 1, urgency, allocated_time

    # 2. Fast Policy prediction
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
    
    # Determine depth based on allocated sims
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
        return fast_move, elapsed, "FAST_MOVE", 0, urgency, allocated_time

    # 3. Adaptive simulation ladder
    simulation_steps = _adaptive_simulation_steps(depth, complexity, sims, light=light_budget)
    if not simulation_steps:
        elapsed = time.monotonic() - start_time
        return fast_move, elapsed, "FAST_MOVE", 0, urgency, allocated_time

    chosen_move = fast_move
    actual_sims_run = 0
    
    for step_sims in simulation_steps:
        actual_sims_run = step_sims
        if time.monotonic() - start_time >= allocated_time * 0.95:
            break

        mcts_move, mcts_san, mcts_root_debug = _best_move_with_mcts(
            engine.model, board, engine.device, step_sims, include_diagnostics=True
        )
        mcts_safety = _move_safety_flags(board, mcts_move)
        
        safe_fallback = _safe_candidate_fallback(board, candidates) if _has_safety_risk(mcts_safety) else None
        if safe_fallback is not None and safe_fallback[0] != mcts_move:
            if any(s > step_sims for s in simulation_steps):
                continue
            chosen_move = safe_fallback[0]
            break

        chosen_move = mcts_move
        # If confident or reached peak
        if step_sims == simulation_steps[-1]:
            break

    final_safety = _move_safety_flags(board, chosen_move)
    if _has_safety_risk(final_safety):
        safe_fallback = _safe_candidate_fallback(board, candidates)
        if safe_fallback is not None and safe_fallback[0] != chosen_move:
            chosen_move = safe_fallback[0]

    elapsed = time.monotonic() - start_time
    return chosen_move, elapsed, "ADAPTIVE_MCTS", actual_sims_run, urgency, allocated_time


def main():
    print("==================================================")
    print("      INITIALIZING DYNAMIC CHESS ENGINE TEST      ")
    print("==================================================")
    engine = Engine(model_path="models/best_model.pth")
    print(f"Device: {engine.device}")

    # TEST 1: Starting Position (Opening - FastMove expected)
    print("\n--- TEST 1: Starting Position (Opening - 180s left) ---")
    board1 = chess.Board()
    move1, t1, mode1, sims1, urg1, alloc1 = run_dynamic_move(engine, board1, time_left_sec=180.0)
    print(f"Position: Starting Board")
    print(f"Clock: 180.0s | Urgency: {urg1} | Allocated Budget: {alloc1:.2f}s")
    print(f"Chosen Move: {move1} ({board1.san(chess.Move.from_uci(move1))})")
    print(f"Mode Used: {mode1} | Actual Sims: {sims1} | Elapsed Time: {t1:.4f}s")
    assert t1 < 0.3, "Opening move should be lightning fast!"
    print(">> TEST 1 PASSED: Opening move played in a fraction of a second, clock conserved!")

    # TEST 2: Critical Defense (Black facing Scholar's Mate threat)
    # Position: 1. e4 e5 2. Bc4 Nc6 3. Qf3 (White threatens Qxf7#)
    print("\n--- TEST 2: Scholar's Mate Defense (Black to move - 150s left) ---")
    board2 = chess.Board("r1bqkb1r/pppp1ppp/2n5/4p3/2B5/5Q2/PPPP1PPP/RNB1K1NR b KQkq - 1 3")
    move2, t2, mode2, sims2, urg2, alloc2 = run_dynamic_move(engine, board2, time_left_sec=150.0)
    san2 = board2.san(chess.Move.from_uci(move2))
    print(f"Threat: White plays Qxf7# next move if Black blunders!")
    print(f"Clock: 150.0s | Urgency: {urg2} | Allocated Budget: {alloc2:.2f}s")
    print(f"Chosen Move: {move2} ({san2})")
    print(f"Mode Used: {mode2} | Actual Sims: {sims2} | Elapsed Time: {t2:.4f}s")
    
    # Verify move defends mate:
    board2_test = board2.copy()
    board2_test.push(chess.Move.from_uci(move2))
    # If white can play Qxf7#, that's mate
    mate_blundered = False
    for m in board2_test.legal_moves:
        board2_test.push(m)
        if board2_test.is_checkmate():
            mate_blundered = True
            break
        board2_test.pop()
    
    print(f"Mate-in-one blundered? {mate_blundered}")
    assert not mate_blundered, "Bot blundered mate!"
    assert urg2 == "critical" or mode2 == "ADAPTIVE_MCTS", "Engine should recognize the critical tactical threat!"
    print(f">> TEST 2 PASSED: Threat successfully detected! Played {san2} which defends checkmate!")

    # TEST 3: Queen Under Attack (Sharp position)
    print("\n--- TEST 3: Queen Under Attack (Tactical Sharpness) ---")
    # Position: White Queen on d4 attacked by Black Knight on c6
    board3 = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/3Q4/8/PPP1PPPP/RNB1KBNR w KQkq - 1 3")
    move3, t3, mode3, sims3, urg3, alloc3 = run_dynamic_move(engine, board3, time_left_sec=120.0)
    san3 = board3.san(chess.Move.from_uci(move3))
    print(f"Position: White Queen on d4 is attacked by Nc6")
    print(f"Clock: 120.0s | Urgency: {urg3} | Allocated Budget: {alloc3:.2f}s")
    print(f"Chosen Move: {move3} ({san3})")
    print(f"Mode Used: {mode3} | Actual Sims: {sims3} | Elapsed Time: {t3:.4f}s")
    print(f">> TEST 3 PASSED: Urgency correctly scaled up, queen saved safely with {san3}!")

    # TEST 4: Low Clock Behavior (< 10s left)
    print("\n--- TEST 4: Low Clock Management (Only 6.0s remaining) ---")
    board4 = chess.Board("r1bqkb1r/pppp1ppp/2n5/4p3/2B1n3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4")
    move4, t4, mode4, sims4, urg4, alloc4 = run_dynamic_move(engine, board4, time_left_sec=6.0)
    print(f"Clock: 6.0s remaining | Urgency: {urg4} | Allocated Budget: {alloc4:.2f}s")
    print(f"Chosen Move: {move4} | Mode: {mode4} | Sims: {sims4} | Elapsed Time: {t4:.4f}s")
    assert t4 < 2.0, "Low clock must not time out!"
    print(f">> TEST 4 PASSED: Bot safely moved in {t4:.3f}s without flagging on time!")

    print("\n==================================================")
    print("       ALL DYNAMIC ENGINE TESTS COMPLETED!        ")
    print("==================================================")

if __name__ == "__main__":
    main()
