"""/fastmove as the web UI calls it: one time-bounded search, game history, tree reuse."""
import asyncio
import time
from unittest.mock import patch

import chess
import pytest

import app.api.main as api
from app.api.main import FastMoveRequest, _board_from_request, _fastmove_budget, fastmove
from app.infra.config import AppConfig, MCTSConfig, ModelConfig
from app.model.network import ChessNet


@pytest.fixture()
def tiny_engine(monkeypatch):
    cfg = AppConfig(
        model=ModelConfig(channels=8, res_blocks=1, value_dropout=0.0),
        mcts=MCTSConfig(inference_batch_size=8, smart_pruning=False),
    )
    model = ChessNet(cfg).eval()
    monkeypatch.setattr(api, "MODEL", model)
    monkeypatch.setattr(api, "DEVICE", "cpu")
    monkeypatch.setattr(api, "_MCTS_BY_MODEL", {})
    return model


def _history(*ucis):
    board = chess.Board()
    for uci in ucis:
        board.push_uci(uci)
    return board


def test_stage_budget_gives_opening_moves_less_time_than_endgames():
    opening = chess.Board()
    middlegame = chess.Board(chess.STARTING_FEN.replace(" 0 1", " 0 15"))
    endgame = chess.Board("4k3/8/8/8/8/8/8/4K2R w K - 0 40")
    assert _fastmove_budget(opening, 6) == (5.0, "opening", False)
    assert _fastmove_budget(middlegame, 6) == (9.0, "middlegame", False)
    assert _fastmove_budget(endgame, 6) == (12.0, "endgame", False)
    assert _fastmove_budget(opening, 3)[0] < _fastmove_budget(opening, 6)[0]


def test_cpu_lifespan_warms_inference_before_accepting_requests(monkeypatch):
    model = object()
    warmed = []
    monkeypatch.setattr(api, "_load_model", lambda: (model, "cpu"))
    monkeypatch.setattr(api, "predict_boards", lambda loaded, boards, device: warmed.append((loaded, boards[0].fen(), device)))
    monkeypatch.setattr(api, "MODEL", None)
    monkeypatch.setattr(api, "DEVICE", None)

    async def run_lifespan():
        async with api.lifespan(api.app):
            assert warmed == [(model, chess.STARTING_FEN, "cpu")]

    asyncio.run(run_lifespan())


def test_history_that_reproduces_the_fen_is_used():
    board = _history("e2e4", "e7e5", "g1f3")
    replayed, used = _board_from_request(FastMoveRequest(fen=board.fen(), moves=["e2e4", "e7e5", "g1f3"]))
    assert used and replayed.move_stack == board.move_stack


def test_history_that_does_not_reach_the_fen_falls_back_to_the_fen():
    board = _history("e2e4", "e7e5", "g1f3")
    replayed, used = _board_from_request(FastMoveRequest(fen=board.fen(), moves=["d2d4", "d7d5", "c2c4"]))
    assert not used and replayed.move_stack == [] and replayed.fen() == board.fen()
    replayed, used = _board_from_request(FastMoveRequest(fen=board.fen(), moves=["e2e5"]))
    assert not used


def test_history_from_a_custom_start_position():
    start = "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"
    board = chess.Board(start)
    board.push_uci("e2e4")
    replayed, used = _board_from_request(FastMoveRequest(fen=board.fen(), start_fen=start, moves=["e2e4"]))
    assert used and len(replayed.move_stack) == 1


def test_fastmove_searches_and_reuses_its_tree_across_the_game(tiny_engine):
    first = fastmove(FastMoveRequest(fen=chess.STARTING_FEN, depth=3, max_simulations=48, moves=[]))
    assert first["source"] == "mcts"
    assert chess.Move.from_uci(first["move"]) in chess.Board().legal_moves
    # Opponent answers with the reply the search explored most under our move.
    tree = api._MCTS_BY_MODEL[(id(tiny_engine), "cpu")]._root
    our = chess.Move.from_uci(first["move"])
    reply = max(tree.children[our].children.items(), key=lambda kv: kv[1].visit_count)[0]
    board = chess.Board()
    board.push(our)
    board.push(reply)
    second = fastmove(FastMoveRequest(fen=board.fen(), depth=3, max_simulations=48, moves=[our.uci(), reply.uci()]))
    assert second["adaptive"]["history"] is True
    assert second["adaptive"]["retained_visits"] > 0
    assert second["adaptive"]["root_visits"] <= 48 + 16  # the cap counts retained visits
    assert chess.Move.from_uci(second["move"]) in board.legal_moves


def test_fastmove_respects_its_time_budget(tiny_engine):
    started = time.perf_counter()
    result = fastmove(FastMoveRequest(fen=chess.STARTING_FEN, time_budget_sec=0.3, max_simulations=1024))
    assert time.perf_counter() - started < 1.5
    assert result["adaptive"]["simulations"] < 1024


def test_fastmove_logs_unsearched_decision_with_game_and_position(tiny_engine, monkeypatch):
    monkeypatch.setattr(api, "_run_mcts", lambda *args, **kwargs: {
        "best_move": chess.Move.from_uci("e2e4"),
        "root_value": 0.2,
        "visit_counts": {},
        "completed_simulations": 0,
        "retained_visits": 0,
        "root_eval_ms": 2500.0,
        "presearch_ms": 2600.0,
    })
    with patch.object(api.logger, "warning") as warning:
        result = fastmove(FastMoveRequest(fen=chess.STARTING_FEN, game_id="game-123", time_budget_sec=2))
    log = warning.call_args.args[0] % warning.call_args.args[1:]
    assert result["source"] == "mcts"
    assert "game=game-123" in log
    assert f"fen={chess.STARTING_FEN}" in log
    assert "mode=policy_fallback reason=deadline_before_rollout" in log
    assert "root_eval_ms=2500.00 presearch_ms=2600.00" in log


def test_fastmove_without_adaptive_is_an_instant_policy_move(tiny_engine):
    result = fastmove(FastMoveRequest(fen=chess.STARTING_FEN, adaptive=False))
    assert result["source"] == "fast_policy"
    assert result["adaptive"]["simulations"] == 0
    assert chess.Move.from_uci(result["move"]) in chess.Board().legal_moves


def test_fastmove_plays_a_proven_mate(tiny_engine):
    board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2")
    result = fastmove(FastMoveRequest(fen=board.fen()))
    assert result["move"] == "d8h4" and result["source"] == "mate_proof"


def test_fastmove_rejects_a_finished_game(tiny_engine):
    board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    with pytest.raises(api.HTTPException):
        fastmove(FastMoveRequest(fen=board.fen()))
