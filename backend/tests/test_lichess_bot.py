from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import chess
import pytest

from app.cli.lichess_bot import (
    BotConfig,
    LichessBot,
    _is_endgame_position,
    calculate_adaptive_simulations,
    compute_time_allocation,
    get_api_token,
    should_accept_challenge,
)
from app.api.main import (
    _is_king_exposed,
    _move_safety_flags,
    _safe_candidate_fallback,
    _should_use_adaptive_search,
)
from app.core.engine import AnalysisResult
from app.evaluation.elo_tracker import EloTracker, MatchRecord


# ==========================================
# 1. Challenge Acceptance & Rejection Tests
# ==========================================

def test_challenge_acceptance_standard_blitz():
    challenge = {
        "variant": {"key": "standard"},
        "rated": True,
        "speed": "blitz",
        "timeControl": {"type": "clock", "limit": 180, "increment": 2},
        "challenger": {"rating": 1800},
    }
    accept, reason = should_accept_challenge(challenge)
    assert accept is True
    assert reason == ""


def test_challenge_acceptance_standard_rapid():
    challenge = {
        "variant": {"key": "standard"},
        "rated": True,
        "speed": "rapid",
        "timeControl": {"type": "clock", "limit": 600, "increment": 0},
        "challenger": {"rating": 1950},
    }
    accept, reason = should_accept_challenge(challenge)
    assert accept is True
    assert reason == ""


def test_challenge_rejection_non_standard_variant():
    challenge = {
        "variant": {"key": "chess960"},
        "rated": True,
        "speed": "blitz",
        "timeControl": {"type": "clock", "limit": 180, "increment": 0},
        "challenger": {"rating": 1800},
    }
    accept, reason = should_accept_challenge(challenge)
    assert accept is False
    assert reason == "variant"


def test_challenge_rejection_unrated():
    challenge = {
        "variant": {"key": "standard"},
        "rated": False,
        "speed": "blitz",
        "timeControl": {"type": "clock", "limit": 180, "increment": 0},
        "challenger": {"rating": 1800},
    }
    accept, reason = should_accept_challenge(challenge)
    assert accept is False
    assert reason == "casual"


def test_challenge_rejection_ultra_bullet():
    challenge = {
        "variant": {"key": "standard"},
        "rated": True,
        "speed": "ultraBullet",
        "timeControl": {"type": "clock", "limit": 30, "increment": 0},
        "challenger": {"rating": 1800},
    }
    accept, reason = should_accept_challenge(challenge)
    assert accept is False
    assert reason == "tooFast"


def test_challenge_rejection_unlimited():
    challenge = {
        "variant": {"key": "standard"},
        "rated": True,
        "speed": "correspondence",
        "timeControl": {"type": "unlimited"},
        "challenger": {"rating": 1800},
    }
    accept, reason = should_accept_challenge(challenge)
    assert accept is False
    assert reason == "tooSlow"


def test_challenge_rating_filters():
    challenge = {
        "variant": {"key": "standard"},
        "rated": True,
        "speed": "blitz",
        "timeControl": {"type": "clock", "limit": 180, "increment": 0},
        "challenger": {"rating": 1200},
    }
    # Min rating 1500 -> reject
    accept, reason = should_accept_challenge(challenge, min_rating=1500)
    assert accept is False
    assert reason == "generic"

    # Max rating 1100 -> reject
    accept, reason = should_accept_challenge(challenge, max_rating=1100)
    assert accept is False
    assert reason == "generic"

    # In bounds [1000, 1500] -> accept
    accept, reason = should_accept_challenge(challenge, min_rating=1000, max_rating=1500)
    assert accept is True
    assert reason == ""


# ==========================================
# 2. Time Allocation & Simulation Tests
# ==========================================

def test_compute_time_allocation():
    # 180s left, 2s inc, 25 moves -> 180/25 + 2*0.8 = 7.2 + 1.6 = 8.8s
    allocated = compute_time_allocation(180.0, 2.0, estimated_remaining_moves=25)
    assert pytest.approx(allocated, 0.01) == 8.8

    # Low time: 10s left, 0s inc -> 10/25 = 0.4s
    allocated_low = compute_time_allocation(10.0, 0.0, estimated_remaining_moves=25)
    assert pytest.approx(allocated_low, 0.01) == 0.4

    # Zero or negative time: returns min_time
    allocated_zero = compute_time_allocation(0.0, 0.0)
    assert allocated_zero >= 0.05


def test_calculate_adaptive_simulations():
    # Very low clock (<5s): fast tactical mode (16 sims)
    sims_panic = calculate_adaptive_simulations(time_left_sec=3.5, increment_sec=0.0)
    assert sims_panic == 16

    # Low clock (<15s): fast tactical mode (24 sims)
    sims_low = calculate_adaptive_simulations(time_left_sec=12.0, increment_sec=1.0)
    assert sims_low == 24

    # Normal clock (180s left, 2s inc -> allocated 8.8s >= 5.0s -> max sims: 256)
    sims_plenty = calculate_adaptive_simulations(time_left_sec=180.0, increment_sec=2.0)
    assert sims_plenty == 256

    # Medium clock (60s left, 0s inc -> allocated 2.4s -> 128 sims)
    sims_mid = calculate_adaptive_simulations(time_left_sec=60.0, increment_sec=0.0)
    assert sims_mid == 128


# ==========================================
# 3. EloTracker Tests
# ==========================================

def test_elo_tracker_record_and_summary(tmp_path: Path):
    log_file = tmp_path / "matches.jsonl"
    tracker = EloTracker(log_path=log_file)
    tracker.set_initial_elo(1800)

    # Record first game (win)
    rec1 = tracker.record_match(
        game_id="game1",
        opponent="PlayerA",
        opponent_elo=1750,
        result="win",
        engine_elo_after=1815,
        time_control="3+2",
        moves_count=35,
    )
    assert rec1.result == "win"
    assert rec1.engine_elo_after == 1815

    # Record second game (draw)
    rec2 = tracker.record_match(
        game_id="game2",
        opponent="PlayerB",
        opponent_elo=1850,
        result="draw",
        engine_elo_after=1817,
        time_control="5+0",
        moves_count=50,
    )
    assert rec2.result == "draw"

    # Verify summary
    summary = tracker.get_summary()
    assert summary["total_games"] == 2
    assert summary["wins"] == 1
    assert summary["losses"] == 0
    assert summary["draws"] == 1
    assert summary["win_rate"] == 0.75
    assert summary["initial_elo"] == 1800
    assert summary["current_elo"] == 1817
    assert summary["elo_delta"] == 17

    # Verify JSONL persistence
    assert log_file.exists()
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    parsed = json.loads(lines[0])
    assert parsed["game_id"] == "game1"
    assert parsed["result"] == "win"

    # Test loading existing tracker from file
    reloaded_tracker = EloTracker(log_path=log_file)
    assert len(reloaded_tracker.matches) == 2
    assert reloaded_tracker.latest_elo == 1817


def test_parse_rating_from_account():
    account_data = {
        "id": "test_bot",
        "perfs": {
            "blitz": {"rating": 1920, "games": 42},
            "rapid": {"rating": 1980, "games": 18},
        },
    }
    assert EloTracker.parse_rating_from_account(account_data, "blitz") == 1920
    assert EloTracker.parse_rating_from_account(account_data, "rapid") == 1980
    assert EloTracker.parse_rating_from_account(account_data, "bullet") == 1920  # fallback to blitz


# ==========================================
# 4. Async Lichess Bot Event & Stream Tests
# ==========================================

class _MockEngine:
    def __init__(self, best_move: chess.Move):
        self._best_move = best_move

    def analyze(self, board, add_noise=False, num_simulations=32, temperature=0.1, time_limit_sec=None):
        return AnalysisResult(
            best_move=self._best_move,
            score=0.25,
            visit_counts={self._best_move.uci(): num_simulations},
            policy={self._best_move.uci(): 1.0},
        )


class _MockStreamContent:
    def __init__(self, lines: list[dict | str]):
        self.lines = lines

    async def __aiter__(self):
        for line in self.lines:
            if isinstance(line, dict):
                yield (json.dumps(line) + "\n").encode("utf-8")
            else:
                yield (str(line) + "\n").encode("utf-8")


@pytest.mark.asyncio
async def test_bot_accept_and_decline_challenge():
    bot_cfg = BotConfig(token="test_token", min_rating=1500)
    bot = LichessBot(bot_cfg=bot_cfg)

    session = MagicMock()

    # Mock accept response
    mock_accept_resp = MagicMock()
    mock_accept_resp.status = 200
    mock_accept_resp.__aenter__ = AsyncMock(return_value=mock_accept_resp)
    mock_accept_resp.__aexit__ = AsyncMock(return_value=None)

    # Mock decline response
    mock_decline_resp = MagicMock()
    mock_decline_resp.status = 200
    mock_decline_resp.__aenter__ = AsyncMock(return_value=mock_decline_resp)
    mock_decline_resp.__aexit__ = AsyncMock(return_value=None)

    session.post.return_value = mock_accept_resp

    valid_challenge = {
        "id": "chal_1",
        "variant": {"key": "standard"},
        "rated": True,
        "speed": "blitz",
        "timeControl": {"type": "clock", "limit": 180, "increment": 2},
        "challenger": {"rating": 1600},
    }
    await bot.handle_challenge_event(session, valid_challenge)
    session.post.assert_called_with(
        f"{bot.api_base}/api/challenge/chal_1/accept",
        headers=bot.get_headers(),
    )

    invalid_challenge = {
        "id": "chal_2",
        "variant": {"key": "chess960"},
        "rated": True,
        "speed": "blitz",
        "timeControl": {"type": "clock", "limit": 180, "increment": 0},
        "challenger": {"rating": 1600},
    }
    await bot.handle_challenge_event(session, invalid_challenge)


@pytest.mark.asyncio
async def test_auto_seek_only_challenges_bots_inside_rating_window():
    bot = LichessBot(bot_cfg=BotConfig(token="test_token", min_rating=1550, max_rating=2100))
    bot.bot_id = "me"

    online = [
        {"id": "weak", "perfs": {"blitz": {"rating": 1200}}},
        {"id": "unrated", "perfs": {}},
        {"id": "toostrong", "perfs": {"blitz": {"rating": 2400}}},
        {"id": "target", "perfs": {"blitz": {"rating": 1750}}},
    ]
    list_resp = MagicMock()
    list_resp.status = 200
    list_resp.text = AsyncMock(return_value="\n".join(json.dumps(b) for b in online))
    list_resp.__aenter__ = AsyncMock(return_value=list_resp)
    list_resp.__aexit__ = AsyncMock(return_value=None)

    chal_resp = MagicMock()
    chal_resp.status = 200
    chal_resp.__aenter__ = AsyncMock(return_value=chal_resp)
    chal_resp.__aexit__ = AsyncMock(return_value=None)

    session = MagicMock()
    session.get.return_value = list_resp
    session.post.return_value = chal_resp

    assert await bot._challenge_random_online_bot(session) is True
    challenged = [call.args[0].rsplit("/", 1)[-1] for call in session.post.call_args_list]
    assert challenged == ["target"]


@pytest.mark.asyncio
async def test_bot_game_stream_and_move_submission(tmp_path: Path):
    bot_cfg = BotConfig(token="test_token")
    mock_engine = _MockEngine(best_move=chess.Move.from_uci("e2e4"))
    bot = LichessBot(bot_cfg=bot_cfg, engine=mock_engine)
    bot.bot_id = "testbot"
    bot.bot_username = "TestBot"
    bot.elo_tracker = EloTracker(log_path=tmp_path / "matches.jsonl")

    session = MagicMock()

    # Mock move POST response
    mock_post_resp = MagicMock()
    mock_post_resp.status = 200
    mock_post_resp.__aenter__ = AsyncMock(return_value=mock_post_resp)
    mock_post_resp.__aexit__ = AsyncMock(return_value=None)
    session.post.return_value = mock_post_resp

    # Simulate gameFull + move + gameState finished
    stream_events = [
        {
            "type": "gameFull",
            "id": "game123",
            "white": {"id": "testbot", "name": "TestBot", "rating": 1850},
            "black": {"id": "opponent", "name": "Opponent", "rating": 1820},
            "speed": "blitz",
            "clock": {"initial": 180, "increment": 0},
            "state": {
                "moves": "",
                "wtime": 180000,
                "btime": 180000,
                "winc": 0,
                "binc": 0,
                "status": "started",
            },
        },
        {
            "type": "gameState",
            "moves": "e2e4 e7e5",
            "wtime": 178000,
            "btime": 179000,
            "winc": 0,
            "binc": 0,
            "status": "started",
        },
        {
            "type": "gameState",
            "moves": "e2e4 e7e5 g1f3",
            "wtime": 175000,
            "btime": 170000,
            "winc": 0,
            "binc": 0,
            "status": "mate",
            "winner": "white",
        },
    ]

    mock_stream_resp = MagicMock()
    mock_stream_resp.status = 200
    mock_stream_resp.content = _MockStreamContent(stream_events)
    mock_stream_resp.__aenter__ = AsyncMock(return_value=mock_stream_resp)
    mock_stream_resp.__aexit__ = AsyncMock(return_value=None)

    def get_side_effect(url, **kwargs):
        if "stream" in url:
            return mock_stream_resp
        acc_resp = MagicMock()
        acc_resp.status = 200
        acc_resp.json = AsyncMock(return_value={"id": "testbot", "perfs": {"blitz": {"rating": 1865}}})
        acc_resp.__aenter__ = AsyncMock(return_value=acc_resp)
        acc_resp.__aexit__ = AsyncMock(return_value=None)
        return acc_resp

    session.get.side_effect = get_side_effect

    await bot.play_game(session, "game123")

    # Verify move e2e4 was posted
    session.post.assert_any_call(
        f"{bot.api_base}/api/bot/game/game123/move/e2e4",
        headers=bot.get_headers(),
    )

    # Verify match recorded
    assert bot.games_completed == 1
    summary = bot.elo_tracker.get_summary()
    assert summary["total_games"] == 1
    assert summary["wins"] == 1
    assert summary["current_elo"] == 1865


@pytest.mark.asyncio
async def test_bot_game_stream_as_black_and_loss(tmp_path: Path):
    bot_cfg = BotConfig(token="test_token")
    mock_engine = _MockEngine(best_move=chess.Move.from_uci("c7c5"))
    bot = LichessBot(bot_cfg=bot_cfg, engine=mock_engine)
    bot.bot_id = "testbot"
    bot.bot_username = "TestBot"
    bot.elo_tracker = EloTracker(log_path=tmp_path / "matches.jsonl")

    session = MagicMock()

    mock_post_resp = MagicMock()
    mock_post_resp.status = 200
    mock_post_resp.__aenter__ = AsyncMock(return_value=mock_post_resp)
    mock_post_resp.__aexit__ = AsyncMock(return_value=None)
    session.post.return_value = mock_post_resp

    # Bot is Black, opponent is White
    stream_events = [
        {
            "type": "gameFull",
            "id": "game_black_1",
            "white": {"id": "grandmaster", "name": "Grandmaster", "rating": 2300},
            "black": {"id": "testbot", "name": "TestBot", "rating": 1850},
            "speed": "rapid",
            "clock": {"initial": 600, "increment": 0},
            "state": {
                "moves": "e2e4",
                "wtime": 599000,
                "btime": 600000,
                "winc": 0,
                "binc": 0,
                "status": "started",
            },
        },
        {
            "type": "gameState",
            "moves": "e2e4 c7c5 d2d4 c5d4",
            "wtime": 580000,
            "btime": 570000,
            "winc": 0,
            "binc": 0,
            "status": "resign",
            "winner": "white",
        },
    ]

    mock_stream_resp = MagicMock()
    mock_stream_resp.status = 200
    mock_stream_resp.content = _MockStreamContent(stream_events)
    mock_stream_resp.__aenter__ = AsyncMock(return_value=mock_stream_resp)
    mock_stream_resp.__aexit__ = AsyncMock(return_value=None)

    def get_side_effect(url, **kwargs):
        if "stream" in url:
            return mock_stream_resp
        acc_resp = MagicMock()
        acc_resp.status = 200
        acc_resp.json = AsyncMock(return_value={"id": "testbot", "perfs": {"rapid": {"rating": 1845}}})
        acc_resp.__aenter__ = AsyncMock(return_value=acc_resp)
        acc_resp.__aexit__ = AsyncMock(return_value=None)
        return acc_resp

    session.get.side_effect = get_side_effect

    await bot.play_game(session, "game_black_1")

    # Verify move c7c5 was posted
    session.post.assert_any_call(
        f"{bot.api_base}/api/bot/game/game_black_1/move/c7c5",
        headers=bot.get_headers(),
    )

    # Verify loss recorded
    assert bot.games_completed == 1
    summary = bot.elo_tracker.get_summary()
    assert summary["total_games"] == 1
    assert summary["losses"] == 1
    assert summary["current_elo"] == 1845


@pytest.mark.asyncio
async def test_post_move_with_retry():
    bot = LichessBot(bot_cfg=BotConfig(token="token123"))
    session = MagicMock()

    # Case 1: Fails once (status 500) then succeeds (status 200)
    fail_resp = MagicMock()
    fail_resp.status = 500
    fail_resp.text = AsyncMock(return_value="Server error")
    fail_resp.__aenter__ = AsyncMock(return_value=fail_resp)
    fail_resp.__aexit__ = AsyncMock(return_value=None)

    success_resp = MagicMock()
    success_resp.status = 200
    success_resp.__aenter__ = AsyncMock(return_value=success_resp)
    success_resp.__aexit__ = AsyncMock(return_value=None)

    session.post.side_effect = [fail_resp, success_resp]

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        success = await bot.post_move_with_retry(session, "g1", "e2e4", max_retries=2)
        assert success is True
        assert session.post.call_count == 2
        mock_sleep.assert_called_once()

    # Case 2: 400 Bad Request (illegal move / not our turn) -> returns False immediately without retries
    session.reset_mock()
    bad_req_resp = MagicMock()
    bad_req_resp.status = 400
    bad_req_resp.text = AsyncMock(return_value="Not your turn")
    bad_req_resp.__aenter__ = AsyncMock(return_value=bad_req_resp)
    bad_req_resp.__aexit__ = AsyncMock(return_value=None)
    session.post.side_effect = [bad_req_resp]

    success = await bot.post_move_with_retry(session, "g1", "e2e4", max_retries=3)
    assert success is False
    assert session.post.call_count == 1


def test_format_time_control():
    from app.cli.lichess_bot import format_time_control
    # Standard blitz 3+2 in seconds
    assert format_time_control({"initial": 180, "increment": 2}, "blitz") == "3+2"
    # Standard blitz 3+2 in milliseconds
    assert format_time_control({"initial": 180000, "increment": 2000}, "blitz") == "3+2"
    # Rapid 10+0 in seconds
    assert format_time_control({"initial": 600, "increment": 0}, "rapid") == "10+0"
    # Fallback to speed string when clock is None
    assert format_time_control(None, "rapid") == "rapid"


def test_challenge_concurrency_limit():
    challenge = {
        "variant": {"key": "standard"},
        "rated": True,
        "speed": "blitz",
        "timeControl": {"type": "clock", "limit": 180, "increment": 2},
        "challenger": {"rating": 1800},
    }
    # When active games reaches max_concurrent_games -> decline with "later"
    accept, reason = should_accept_challenge(challenge, active_games_count=1, max_concurrent_games=1)
    assert accept is False
    assert reason == "later"


def test_load_production_engine_missing_checkpoint():
    from app.cli.lichess_bot import load_production_engine
    from app.infra.config import AppConfig

    cfg = AppConfig()
    with pytest.raises(FileNotFoundError, match="Model checkpoint not found"):
        load_production_engine(cfg, "cpu", model_path_arg="non_existent_model_checkpoint.pth")


@pytest.mark.asyncio
async def test_compute_best_move_returns_completed_search_with_budget():
    class _BudgetEngine:
        budget = None

        def analyze(self, board, add_noise=False, num_simulations=32, temperature=0.1, time_limit_sec=None):
            self.budget = time_limit_sec
            return AnalysisResult(chess.Move.from_uci('e2e4'), 0.0, {'e2e4': 3}, {'e2e4': 1.0})

    engine = _BudgetEngine()
    bot = LichessBot(bot_cfg=BotConfig(token="test_token"), engine=engine)
    board = chess.Board()
    move_uci = await bot._compute_best_move(board, num_simulations=32, allocated_time_sec=0.05)
    assert move_uci == 'e2e4'
    assert engine.budget == pytest.approx(0.045)


@pytest.mark.asyncio
async def test_failed_search_still_returns_a_legal_emergency_move():
    class BrokenEngine:
        def analyze(self, **kwargs):
            raise RuntimeError('simulated inference failure')
    bot = LichessBot(bot_cfg=BotConfig(token='test'), engine=BrokenEngine())
    board = chess.Board()
    move = await bot._compute_best_move(board, 32, 0.05)
    assert chess.Move.from_uci(move) in board.legal_moves


@pytest.mark.asyncio
async def test_cancelled_search_keeps_engine_locked_until_worker_finishes():
    import threading
    started, release = threading.Event(), threading.Event()
    class BlockingEngine:
        def analyze(self, **kwargs):
            started.set()
            release.wait(timeout=3)
            return AnalysisResult(chess.Move.from_uci('e2e4'), 0.0, {'e2e4': 1}, {'e2e4': 1.0})
    bot = LichessBot(bot_cfg=BotConfig(token='test'), engine=BlockingEngine())
    task = asyncio.create_task(bot._compute_best_move(chess.Board(), 32, 0.05))
    try:
        assert await asyncio.to_thread(started.wait, 1)
        task.cancel()
        await asyncio.sleep(0)
        assert bot._engine_lock.locked()
        assert not task.done()
        task.cancel()
        await asyncio.sleep(0)
        assert bot._engine_lock.locked()
        assert not task.done()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not bot._engine_lock.locked()


def test_reported_turkjs_rook_sacrifice_is_marked_unsafe():
    board = chess.Board("rn5r/2q5/2pk4/1p2n1Bp/3b3P/p2P2p1/P1R2PBN/2Q1R1K1 w - - 0 30")

    flags = _move_safety_flags(board, "c2c6")

    assert flags["valuable_piece_capture"] is True


def test_reported_turochamp_exchange_sacrifice_is_marked_unsafe():
    board = chess.Board("r1bqr1k1/1p3ppp/p7/2Q5/3n3P/8/PPP1BPP1/3RK1NR b K - 0 16")

    flags = _move_safety_flags(board, "e8e2")

    assert flags["valuable_piece_capture"] is True


def test_reported_jibbby_king_position_forces_adaptive_search():
    board = chess.Board("r4r1k/ppp2p2/2n2NpB/8/3p4/6R1/1K2PP1P/R4B2 b - - 3 27")

    assert _is_king_exposed(board, chess.BLACK) is True
    assert _should_use_adaptive_search(1, ["king_exposed"], depth=5) is True


def test_safe_fallback_scans_beyond_policy_shortlist():
    board = chess.Board()
    board.push_san("f3")
    board.push_san("e5")
    candidates = [{"uci": "g2g4", "score": 1000.0}]

    fallback = _safe_candidate_fallback(board, candidates)

    assert fallback is not None
    assert fallback[0] != "g2g4"
    assert _move_safety_flags(board, fallback[0])["mate_one"] is False


def test_queenless_low_material_position_is_endgame():
    assert _is_endgame_position(chess.Board("8/5p2/p1R3p1/1p3k1p/1P3P1P/6P1/P2KN3/8 b - - 0 28")) is True
    assert _is_endgame_position(chess.Board()) is False


class _RecordingEngine:
    """Engine stand-in: records what the bot asked the search for."""

    device = "cpu"

    def __init__(self, move="d2d4"):
        self.calls = []
        self.move = move

        class _Model:
            def predict(inner, board, device=None):
                return object(), 0.0

        self.model = _Model()

    def analyze(self, board, add_noise=False, num_simulations=32, temperature=0.1, time_limit_sec=None):
        from app.core.engine import AnalysisResult
        self.calls.append({"num_simulations": num_simulations, "time_limit_sec": time_limit_sec})
        return AnalysisResult(chess.Move.from_uci(self.move), 0.1, {self.move: num_simulations}, {self.move: 1.0})


def test_critical_position_gets_one_search_with_the_full_budget():
    engine = _RecordingEngine("d2d4")
    bot = LichessBot(bot_cfg=BotConfig(token="test"), engine=engine)
    move = bot._compute_move_sync(chess.Board(), 128, 10.0, "critical")
    assert move == "d2d4"
    assert len(engine.calls) == 1
    assert engine.calls[0]["num_simulations"] == 128
    assert engine.calls[0]["time_limit_sec"] == pytest.approx(9.5)


def test_cli_arg_parser():
    from app.cli.lichess_bot import build_arg_parser
    parser = build_arg_parser()
    args = parser.parse_args([
        "--config", "config/custom.yaml",
        "--model-path", "models/my_model.pth",
        "--device", "cpu",
        "--max-games", "10",
        "--min-rating", "1600",
        "--max-rating", "2200",
        "--max-sims", "384",
        "--auto-seek",
    ])
    assert args.config == "config/custom.yaml"
    assert args.model_path == "models/my_model.pth"
    assert args.device == "cpu"
    assert args.max_games == 10
    assert args.min_rating == 1600
    assert args.max_rating == 2200
    assert args.max_sims == 384
    assert args.auto_seek is True


def test_get_api_token():
    with patch.dict("os.environ", {"LICHESS_BOT_TOKEN": "lip_test123"}):
        assert get_api_token() == "lip_test123"

    with patch.dict("os.environ", {}, clear=True), patch("app.cli.lichess_bot.load_env_file"):
        with pytest.raises(ValueError, match="LICHESS_BOT_TOKEN"):
            get_api_token()


@pytest.mark.asyncio
async def test_bot_still_moves_when_it_could_claim_a_threefold():
    """is_game_over(claim_draw=True) is already true when a repeating move exists; the bot
    returned without moving there and lost on time."""
    bot = LichessBot(bot_cfg=BotConfig(token="test_token"), engine=_MockEngine(best_move=chess.Move.from_uci("e7e5")))
    posted = []

    async def fake_post(session, game_id, move_uci):
        posted.append(move_uci)
        return True

    bot.post_move_with_retry = fake_post
    shuffle = "g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1"  # ...Ng8 would repeat the start a third time
    board = chess.Board()
    for uci in shuffle.split():
        board.push_uci(uci)
    assert board.is_game_over(claim_draw=True) and not board.is_repetition(3)
    state = {"moves": shuffle, "wtime": 60000, "btime": 60000, "winc": 0, "binc": 0}
    await bot._process_game_turn(MagicMock(), "g1", state, chess.BLACK, last_moved_ply=-1)
    assert posted == ["e7e5"]


@pytest.mark.asyncio
async def test_bot_does_not_move_after_an_actual_threefold():
    bot = LichessBot(bot_cfg=BotConfig(token="test_token"), engine=_MockEngine(best_move=chess.Move.from_uci("e2e4")))
    posted = []

    async def fake_post(session, game_id, move_uci):
        posted.append(move_uci)
        return True

    bot.post_move_with_retry = fake_post
    state = {"moves": "g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1 f6g8", "wtime": 60000, "btime": 60000, "winc": 0, "binc": 0}
    await bot._process_game_turn(MagicMock(), "g1", state, chess.WHITE, last_moved_ply=-1)
    assert posted == []



def test_auto_seek_backs_off_when_challenges_go_unanswered():
    from app.cli.lichess_bot import auto_seek_backoff_sec

    waits = [auto_seek_backoff_sec(n) for n in range(0, 8)]
    assert waits[:6] == [30.0, 30.0, 60.0, 120.0, 240.0, 480.0]
    assert waits[6] == waits[7] == 600.0, "capped at 10 minutes"


@pytest.mark.asyncio
async def test_auto_seek_rests_a_bot_that_ignored_the_challenge():
    bot = LichessBot(bot_cfg=BotConfig(token="test_token", min_rating=1500, max_rating=2100))
    bot.bot_id = "me"

    online = [{"id": "sleepy", "perfs": {"blitz": {"rating": 1700}}}]
    list_resp = MagicMock()
    list_resp.status = 200
    list_resp.text = AsyncMock(return_value="\n".join(json.dumps(b) for b in online))
    list_resp.__aenter__ = AsyncMock(return_value=list_resp)
    list_resp.__aexit__ = AsyncMock(return_value=None)

    chal_resp = MagicMock()
    chal_resp.status = 200
    chal_resp.json = AsyncMock(return_value={"challenge": {"id": "c123"}})
    chal_resp.__aenter__ = AsyncMock(return_value=chal_resp)
    chal_resp.__aexit__ = AsyncMock(return_value=None)

    session = MagicMock()
    session.get.return_value = list_resp
    session.post.return_value = chal_resp

    assert await bot._challenge_random_online_bot(session) is True
    await bot._drop_ignored_challenge(session)

    assert session.post.call_args_list[-1].args[0].endswith("/api/challenge/c123/cancel")
    assert bot._failed_targets["sleepy"] > time.time() + 3000
    session.post.reset_mock()
    assert await bot._challenge_random_online_bot(session) is False, "the ignoring bot is skipped"
    session.post.assert_not_called()
