from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any

import aiohttp
import chess

from app.cli.common import add_common_runtime_args, configure_runtime
from app.core.engine import Engine
from app.evaluation.elo_tracker import EloTracker
from app.game.tactics import select_safe_move
from app.infra.config import load_config
from app.infra.device import select_device
from app.infra.logging import setup_logging
from app.api.main import (
    _fast_policy_move,
    _legal_moves_with_probs,
)

logger = setup_logging("cli.lichess_bot")

LICHESS_API_BASE = "https://lichess.org"


def load_env_file(env_path: Path | str | None = None) -> None:
    try:
        import dotenv
        if env_path:
            dotenv.load_dotenv(dotenv_path=env_path)
        else:
            dotenv.load_dotenv()
    except Exception:
        candidates = [
            Path(".env"),
            Path(__file__).resolve().parents[2] / ".env",
            Path(__file__).resolve().parents[3] / ".env",
        ]
        for candidate in candidates:
            if candidate.exists():
                try:
                    with open(candidate, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line or line.startswith("#") or "=" not in line:
                                continue
                            key, val = line.split("=", 1)
                            key = key.strip()
                            val = val.strip().strip("'\"")
                            if key and key not in os.environ:
                                os.environ[key] = val
                except Exception as exc:
                    logger.debug("Failed to manually load %s: %s", candidate, exc)


def get_api_token() -> str:
    load_env_file()
    token = os.environ.get("LICHESS_BOT_TOKEN")
    if not token or not token.strip():
        raise ValueError(
            "LICHESS_BOT_TOKEN environment variable not set. "
            "Please set LICHESS_BOT_TOKEN in your environment or provide it in a .env file."
        )
    return token.strip()


# Below this per-move budget a search cannot finish its first batch; play the policy move.
MIN_CLOCK_SEARCH_SEC = 0.35
# Clock held back from the allocation for network latency to the Lichess server.
LATENCY_RESERVE_SEC = 0.3
# Share of the move budget given to the search deadline; the rest covers the final
# screen and posting the move.
SEARCH_BUDGET_FRACTION = 0.95


# Increments at least this large pay for the fixed per-move cost (inference + latency).
COVERING_INCREMENT_SEC = 0.7

# With an increment, ordinary moves leave this many increments on the clock so
# critical positions late in long games still get a real search.
INCREMENT_RESERVE_MOVES = 10


def auto_seek_backoff_sec(unanswered: int) -> float:
    """Seconds between challenges after ``unanswered`` ignored ones in a row.

    Re-challenging every 30 s while no bot in range was accepting sent 102
    challenges in two hours and got the account rate-limited (429) for the
    rest of the night, so the wait doubles up to 10 minutes.
    """
    if unanswered <= 0:
        return 30.0
    return float(min(600.0, 30.0 * 2 ** (unanswered - 1)))


def _estimated_moves_to_go(board: chess.Board, increment_sec: float = 0.0) -> int:
    """Moves the remaining clock must cover.

    With a real increment every move refills part of the clock, but games
    against stronger opponents run past move 100; a 15-move floor spent the
    clock by move 60 (3+2, lost a 157-move game on the increment alone), so the
    floor is 30. Without an increment the fixed cost of each move (inference and
    network latency) is never refunded, so keep a longer horizon (measured: no
    flag up to ~100 moves in 5+0 with 0.3 s overruns).
    """
    moves_played = max(0, int(board.fullmove_number) - 1)
    if float(increment_sec) >= COVERING_INCREMENT_SEC:
        return max(30, 45 - moves_played)
    return max(30, 50 - moves_played)


def _clock_reserve(increment_sec: float, urgency: str) -> float:
    """Clock that normal/quiet moves must not spend; sharp and critical ones may."""
    if float(increment_sec) < COVERING_INCREMENT_SEC or urgency not in ("normal", "quiet"):
        return 0.0
    return INCREMENT_RESERVE_MOVES * float(increment_sec)


def calculate_dynamic_thinking(
    board: chess.Board,
    time_left_sec: float,
    increment_sec: float = 0.0,
    max_sims: int = 256,
) -> tuple[int, float, str]:
    """Return ``(simulation cap, time budget, urgency)`` for the next move.

    The budget spreads the usable clock over the expected remaining moves plus
    most of the increment, scaled by how sharp the position is. The search
    treats it as a deadline and stops earlier by itself once its choice is
    settled, so the simulation cap stays at ``max_sims``; the time spent is what
    the position needs, and unspent time returns to the clock.
    """
    legal_moves = list(board.legal_moves)
    num_legal = len(legal_moves)

    # 1. Forced move: only 1 legal move -> play instantly!
    if num_legal <= 1:
        return 1, 0.05, "forced_move"

    # 2. Criticality analysis
    is_check = board.is_check()
    captures_count = sum(1 for m in legal_moves if board.is_capture(m))

    queen_under_attack = any(
        board.is_attacked_by(not board.turn, q_sq) for q_sq in board.pieces(chess.QUEEN, board.turn)
    )

    # Does the opponent threaten mate in one? (The side to move is not in check here,
    # so passing the move is legal for python-chess.)
    opponent_threatens_mate = False
    if not is_check:
        opp_board = board.copy(stack=False)
        opp_board.turn = not board.turn
        opp_board.ep_square = None
        for opp_move in opp_board.legal_moves:
            if not opp_board.gives_check(opp_move):
                continue
            opp_board.push(opp_move)
            mated = opp_board.is_checkmate()
            opp_board.pop()
            if mated:
                opponent_threatens_mate = True
                break

    if is_check or queen_under_attack or opponent_threatens_mate:
        urgency, urgency_multiplier = "critical", 1.8
    elif captures_count >= 3:
        urgency, urgency_multiplier = "sharp", 1.3
    elif captures_count == 0 and num_legal <= 8:
        urgency, urgency_multiplier = "quiet", 0.75
    else:
        urgency, urgency_multiplier = "normal", 1.0

    # 3. Budget
    usable = max(0.0, float(time_left_sec) - LATENCY_RESERVE_SEC - _clock_reserve(increment_sec, urgency))
    base = usable / _estimated_moves_to_go(board, increment_sec) + max(0.0, float(increment_sec)) * 0.8
    allocated = base * urgency_multiplier
    # Never more than a quarter of the clock on one move, and far less once it is low.
    cap_fraction = 0.25 if time_left_sec > 5.0 else 0.15
    allocated = min(allocated, max(0.05, float(time_left_sec) * cap_fraction))

    return max(1, int(max_sims)), max(0.05, allocated), urgency


def compute_time_allocation(
    time_left_sec: float,
    increment_sec: float,
    estimated_remaining_moves: int = 25,
    min_time_sec: float = 0.05,
) -> float:
    """Compatibility helper for callers that only need a clock-based budget."""
    remaining_moves = max(1, int(estimated_remaining_moves))
    allocation = max(0.0, float(time_left_sec)) / remaining_moves
    allocation += max(0.0, float(increment_sec)) * 0.8
    return max(float(min_time_sec), allocation)


def calculate_adaptive_simulations(
    time_left_sec: float,
    increment_sec: float = 0.0,
    max_sims: int = 256,
) -> int:
    """Compatibility helper for the legacy clock-only simulation policy."""
    if time_left_sec < 5.0:
        return min(max_sims, 16)
    if time_left_sec < 15.0:
        return min(max_sims, 24)

    allocation = compute_time_allocation(time_left_sec, increment_sec)
    if allocation >= 5.0:
        return min(max_sims, 256)
    if allocation >= 2.0:
        return min(max_sims, 128)
    if allocation >= 1.0:
        return min(max_sims, 64)
    return min(max_sims, 32)


def _is_endgame_position(board: chess.Board) -> bool:
    if board.queens:
        return False
    non_pawn_material = sum(
        len(board.pieces(piece_type, color))
        for color in (chess.WHITE, chess.BLACK)
        for piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK)
    )
    return non_pawn_material <= 6


def format_time_control(clock_data: dict[str, Any] | None, speed_str: str) -> str:
    if not clock_data:
        return speed_str
    try:
        initial = int(clock_data.get("initial", 0))
        increment = int(clock_data.get("increment", 0))
        init_min = initial // 60000 if initial >= 1000 else initial // 60
        inc_sec = increment // 1000 if increment >= 1000 else increment
        return f"{init_min}+{inc_sec}"
    except Exception:
        return speed_str


def should_accept_challenge(
    challenge: dict[str, Any],
    min_rating: int | None = None,
    max_rating: int | None = None,
    active_games_count: int = 0,
    max_concurrent_games: int = 1,
    allow_casual: bool = False,
) -> tuple[bool, str]:
    if active_games_count >= max_concurrent_games:
        return False, "later"

    variant = challenge.get("variant", {})
    if variant.get("key") != "standard":
        return False, "variant"

    if not challenge.get("rated", False) and not allow_casual:
        return False, "casual"

    time_control = challenge.get("timeControl", {})
    tc_type = time_control.get("type")
    if tc_type != "clock":
        return False, "tooSlow"

    limit = time_control.get("limit", 0)
    increment = time_control.get("increment", 0)
    speed = challenge.get("speed", "")

    if limit < 60:
        return False, "tooFast"

    supported_speeds = {"blitz", "rapid"}
    if speed not in supported_speeds and not (180 <= limit <= 900 and increment <= 10):
        return False, "tooSlow" if limit > 900 else "tooFast"

    challenger = challenge.get("challenger", {})
    rating = challenger.get("rating")
    if isinstance(rating, (int, float)):
        if min_rating is not None and rating < min_rating:
            return False, "generic"
        if max_rating is not None and rating > max_rating:
            return False, "generic"

    return True, ""


@dataclass
class BotConfig:
    token: str
    config_path: str = "config/default.yaml"
    model_path: str = "models/best_model.pth"
    device: str | None = None
    max_games: int | None = None
    min_rating: int | None = None
    max_rating: int | None = None
    max_concurrent_games: int = 1
    max_sims: int = 256
    auto_seek: bool = False
    allow_casual: bool = True
    api_base: str = LICHESS_API_BASE


class LichessBot:
    def __init__(self, bot_cfg: BotConfig, engine: Engine | None = None) -> None:
        self.bot_cfg = bot_cfg
        self.api_base = bot_cfg.api_base
        self.token = bot_cfg.token
        self.engine = engine
        self.elo_tracker = EloTracker()
        self.bot_id: str = ""
        self.bot_username: str = ""
        self.games_completed: int = 0
        self.active_games: set[str] = set()
        self._shutdown_event = asyncio.Event()
        self._engine_lock = asyncio.Lock()
        self._failed_targets: dict[str, float] = {}

    def get_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "User-Agent": "LichessBot-ChessNet/1.0 (PyTorch-MCTS engine)",
        }

    async def fetch_account(self, session: aiohttp.ClientSession) -> dict[str, Any]:
        url = f"{self.api_base}/api/account"
        async with session.get(url, headers=self.get_headers()) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"Failed to fetch account (status {resp.status}): {text}")
            return await resp.json()

    async def accept_challenge(self, session: aiohttp.ClientSession, challenge_id: str) -> bool:
        url = f"{self.api_base}/api/challenge/{challenge_id}/accept"
        try:
            async with session.post(url, headers=self.get_headers()) as resp:
                if resp.status == 200:
                    logger.info("Accepted challenge %s", challenge_id)
                    return True
                text = await resp.text()
                logger.warning("Failed to accept challenge %s (status %d): %s", challenge_id, resp.status, text)
        except Exception as exc:
            logger.exception("Error accepting challenge %s: %s", challenge_id, exc)
        return False

    async def decline_challenge(self, session: aiohttp.ClientSession, challenge_id: str, reason: str = "generic") -> bool:
        url = f"{self.api_base}/api/challenge/{challenge_id}/decline"
        try:
            data = aiohttp.FormData()
            data.add_field("reason", reason)
            async with session.post(url, headers=self.get_headers(), data=data) as resp:
                if resp.status == 200:
                    logger.info("Declined challenge %s (reason: %s)", challenge_id, reason)
                    return True
                text = await resp.text()
                logger.warning("Failed to decline challenge %s (status %d): %s", challenge_id, resp.status, text)
        except Exception as exc:
            logger.exception("Error declining challenge %s: %s", challenge_id, exc)
        return False

    async def post_move_with_retry(
        self,
        session: aiohttp.ClientSession,
        game_id: str,
        move_uci: str,
        max_retries: int = 3,
    ) -> bool:
        url = f"{self.api_base}/api/bot/game/{game_id}/move/{move_uci}"
        for attempt in range(1, max_retries + 1):
            try:
                async with session.post(url, headers=self.get_headers()) as resp:
                    if resp.status == 200:
                        logger.info("Game %s: Submitted move %s", game_id, move_uci)
                        return True
                    text = await resp.text()
                    logger.warning("Game %s: Move %s attempt %d returned status %d: %s", game_id, move_uci, attempt, resp.status, text)
                    if resp.status == 400:
                        return False
            except Exception as exc:
                logger.warning("Game %s: Move %s attempt %d error: %s", game_id, move_uci, attempt, exc)
            if attempt < max_retries:
                await asyncio.sleep(0.5 * (2 ** (attempt - 1)))
        return False

    async def play_game(self, session: aiohttp.ClientSession, game_id: str) -> None:
        if game_id in self.active_games:
            return
        self.active_games.add(game_id)
        url = f"{self.api_base}/api/bot/game/stream/{game_id}"
        logger.info("Starting game session for %s", game_id)

        bot_color: chess.Color = chess.WHITE
        opponent_name: str = "opponent"
        opponent_elo: int | None = None
        speed: str = "blitz"
        time_control_str: str = "3+0"
        last_moved_ply: int = -1
        game_finished: bool = False
        reconnect_attempts: int = 0
        max_reconnects: int = 5

        try:
            while not game_finished and reconnect_attempts < max_reconnects:
                try:
                    async with session.get(url, headers=self.get_headers()) as resp:
                        if resp.status != 200:
                            text = await resp.text()
                            logger.error("Failed to open game stream %s (status %d): %s", game_id, resp.status, text)
                            reconnect_attempts += 1
                            await asyncio.sleep(1.0 * (2 ** reconnect_attempts))
                            continue

                        reconnect_attempts = 0
                        async for raw_line in resp.content:
                            line = raw_line.decode("utf-8").strip()
                            if not line:
                                continue

                            try:
                                data = json.loads(line)
                            except json.JSONDecodeError:
                                continue

                            event_type = data.get("type")

                            if event_type == "gameFull":
                                white_info = data.get("white", {})
                                black_info = data.get("black", {})
                                white_id = str(white_info.get("id", "")).lower()
                                white_name = str(white_info.get("name", "")).lower()

                                if white_id == self.bot_id.lower() or white_name == self.bot_username.lower():
                                    bot_color = chess.WHITE
                                    opponent_info = black_info
                                else:
                                    bot_color = chess.BLACK
                                    opponent_info = white_info

                                opponent_name = opponent_info.get("name") or opponent_info.get("id") or "opponent"
                                opponent_elo = opponent_info.get("rating")
                                speed = data.get("speed", "blitz")
                                clock = data.get("clock")
                                time_control_str = format_time_control(clock, speed)

                                logger.info(
                                    "Game %s initialized | Color: %s | Opponent: %s (%s) | TC: %s",
                                    game_id,
                                    "White" if bot_color == chess.WHITE else "Black",
                                    opponent_name,
                                    opponent_elo if opponent_elo is not None else "?",
                                    time_control_str,
                                )

                                state = data.get("state", {})
                                last_moved_ply = await self._process_game_turn(
                                    session=session,
                                    game_id=game_id,
                                    state=state,
                                    bot_color=bot_color,
                                    last_moved_ply=last_moved_ply,
                                )

                                status = state.get("status")
                                if status and status not in {"started", "created"}:
                                    game_finished = True
                                    await self._handle_game_end(
                                        session=session,
                                        game_id=game_id,
                                        state=state,
                                        bot_color=bot_color,
                                        opponent_name=opponent_name,
                                        opponent_elo=opponent_elo,
                                        time_control_str=time_control_str,
                                        speed=speed,
                                    )
                                    break

                            elif event_type == "gameState":
                                state = data
                                status = state.get("status")

                                if status and status not in {"started", "created"}:
                                    game_finished = True
                                    await self._handle_game_end(
                                        session=session,
                                        game_id=game_id,
                                        state=state,
                                        bot_color=bot_color,
                                        opponent_name=opponent_name,
                                        opponent_elo=opponent_elo,
                                        time_control_str=time_control_str,
                                        speed=speed,
                                    )
                                    break

                                last_moved_ply = await self._process_game_turn(
                                    session=session,
                                    game_id=game_id,
                                    state=state,
                                    bot_color=bot_color,
                                    last_moved_ply=last_moved_ply,
                                )

                except asyncio.CancelledError:
                    break
                except Exception as exc:
                    if game_finished:
                        break
                    reconnect_attempts += 1
                    logger.warning("Game stream %s dropped (%s). Reconnecting attempt %d/%d...", game_id, exc, reconnect_attempts, max_reconnects)
                    await asyncio.sleep(1.0 * (2 ** (reconnect_attempts - 1)))

        except Exception as exc:
            logger.exception("Exception in game loop for %s: %s", game_id, exc)
        finally:
            self.active_games.discard(game_id)

    async def _process_game_turn(
        self,
        session: aiohttp.ClientSession,
        game_id: str,
        state: dict[str, Any],
        bot_color: chess.Color,
        last_moved_ply: int,
    ) -> int:
        moves_str = state.get("moves", "")
        board = chess.Board()
        if moves_str.strip():
            for uci_move in moves_str.split():
                try:
                    board.push_uci(uci_move)
                except ValueError:
                    logger.error("Game %s: Invalid move %s in move string", game_id, uci_move)

        current_ply = len(board.move_stack)
        # Only a position that has actually ended stops us. is_game_over(claim_draw=True)
        # is also true when a repeating move merely exists; the bot then never moved and
        # lost on time. Whether to repeat is the search's decision.
        if board.turn != bot_color or board.is_game_over(claim_draw=False) or board.is_repetition(3):
            return last_moved_ply

        if current_ply <= last_moved_ply:
            return last_moved_ply

        wtime_ms = state.get("wtime", 180000)
        btime_ms = state.get("btime", 180000)
        winc_ms = state.get("winc", 0)
        binc_ms = state.get("binc", 0)

        my_time_ms = wtime_ms if bot_color == chess.WHITE else btime_ms
        my_inc_ms = winc_ms if bot_color == chess.WHITE else binc_ms

        time_left_sec = max(0.1, my_time_ms / 1000.0)
        inc_sec = max(0.0, my_inc_ms / 1000.0)

        adaptive_sims, allocated_time_sec, urgency = calculate_dynamic_thinking(
            board=board,
            time_left_sec=time_left_sec,
            increment_sec=inc_sec,
            max_sims=self.bot_cfg.max_sims,
        )
        logger.info(
            "Game %s [Ply %d]: Clock=%.1fs (+%.1fs) | [%s] Allocated=%.2fs -> MCTS Sims=%d",
            game_id,
            current_ply,
            time_left_sec,
            inc_sec,
            urgency.upper(),
            allocated_time_sec,
            adaptive_sims,
        )

        best_move_uci = await self._compute_best_move(board, adaptive_sims, allocated_time_sec, urgency)
        if best_move_uci:
            success = await self.post_move_with_retry(session, game_id, best_move_uci)
            if success:
                return current_ply
        return last_moved_ply

    async def _compute_best_move(
        self,
        board: chess.Board,
        num_simulations: int,
        allocated_time_sec: float = 3.0,
        urgency: str = "normal",
    ) -> str | None:
        if not self.engine:
            legal = next(iter(board.legal_moves), None)
            return legal.uci() if legal else None

        loop = asyncio.get_running_loop()
        # The search observes its own deadline and returns completed visits.
        # Cancelling an executor Future does not stop its Python worker.
        async with self._engine_lock:
            future = loop.run_in_executor(
                None,
                self._compute_move_sync,
                board,
                num_simulations,
                allocated_time_sec,
                urgency,
            )
            try:
                return await asyncio.shield(future)
            except asyncio.CancelledError:
                # Keep ownership of the engine until the worker actually stops.
                while not future.done():
                    try:
                        await asyncio.shield(future)
                    except asyncio.CancelledError:
                        continue
                    except Exception:
                        break
                raise

    def _compute_move_sync(
        self,
        board: chess.Board,
        num_simulations: int,
        allocated_time_sec: float,
        urgency: str = "normal",
    ) -> str | None:
        """Pick a move within ``allocated_time_sec``.

        One time-bounded search. The engine keeps its tree between moves, so the
        visits already spent on the opponent's actual reply carry over, and the
        search stops by itself once the remaining budget cannot change its
        choice. Its result is final apart from the proven-mate / dropped-material
        screen; heuristic flags never overrule it. The policy move (with the same
        screen) is used only when the clock leaves no room for a search.
        """
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        if len(legal_moves) == 1:
            logger.info("Only 1 legal move available; playing %s instantly", legal_moves[0].uci())
            return legal_moves[0].uci()

        started = time.monotonic()
        budget = max(0.05, float(allocated_time_sec))
        try:
            if budget < MIN_CLOCK_SEARCH_SEC or int(num_simulations) <= 1:
                return self._policy_move(board, budget, started)

            analysis = self.engine.analyze(
                board=board,
                num_simulations=max(1, int(num_simulations)),
                temperature=0.0,
                time_limit_sec=budget * SEARCH_BUDGET_FRACTION,
            )
            move = analysis.best_move
            if move is None or move not in board.legal_moves:
                raise RuntimeError("search returned no legal move")

            # The search already screened its choice; repeat the screen without a
            # deadline in case the in-search screen ran out of time.
            by_visits = sorted(analysis.visit_counts.items(), key=lambda item: -int(item[1]))
            ranked = [move] + [
                candidate for candidate in (chess.Move.from_uci(uci) for uci, _ in by_visits)
                if candidate != move and candidate in board.legal_moves
            ]
            screened, _rejected = select_safe_move(board, ranked, root_value=float(analysis.score))
            if screened is not None and screened != move:
                logger.warning("Post-search screen replaced %s with %s", move.uci(), screened.uci())
                move = screened
            logger.info(
                "Search chose %s | visits=%d | value=%+.3f | %.2fs of %.2fs budget (%s)",
                move.uci(),
                sum(int(v) for v in analysis.visit_counts.values()),
                float(analysis.score),
                time.monotonic() - started,
                budget,
                urgency,
            )
            return move.uci()
        except Exception as exc:
            logger.exception("Move search failed; falling back to a budgeted engine analysis: %s", exc)

        try:
            analysis = self.engine.analyze(
                board=board,
                num_simulations=min(num_simulations, 64),
                temperature=0.05,
                time_limit_sec=max(0.01, allocated_time_sec * 0.9),
            )
            if analysis and analysis.best_move:
                return analysis.best_move.uci()
        except Exception:
            pass

        fallback = next(iter(board.legal_moves), None)
        return fallback.uci() if fallback else None

    def _policy_move(self, board: chess.Board, budget: float, started: float) -> str:
        """Low-clock move: the fast policy ranking, screened for short mates and dropped material."""
        import torch

        with torch.no_grad():
            logits, value = self.engine.model.predict(board, device=self.engine.device)
        raw_policy_top = _legal_moves_with_probs(board, logits, 16)
        _move, _san, candidates = _fast_policy_move(board, logits, 16, raw_policy_top)
        ranked = [chess.Move.from_uci(item["uci"]) for item in candidates]
        ranked += [move for move in board.legal_moves if move not in ranked]
        remaining = max(0.02, budget - (time.monotonic() - started))
        root_value = float(value) if isinstance(value, (int, float)) else None
        chosen, _rejected = select_safe_move(board, ranked, time.monotonic() + remaining * 0.8, root_value=root_value)
        chosen = chosen or ranked[0]
        logger.info("Low clock: policy move %s (%.2fs budget)", chosen.uci(), budget)
        return chosen.uci()

    async def auto_seek_loop(self, session: aiohttp.ClientSession) -> None:
        logger.info("Auto-seek active: Will search for rated Blitz matches whenever idle.")
        await asyncio.sleep(5.0)
        unanswered = 0
        while not self._shutdown_event.is_set():
            try:
                if not self.active_games and self.games_completed < (self.bot_cfg.max_games or float("inf")):
                    logger.info("Auto-seek: Bot is idle, seeking opponent...")
                    games_before = self.games_completed
                    sent = await self._challenge_random_online_bot(session)
                    if sent:
                        await asyncio.sleep(30.0)
                        if self.active_games or self.games_completed != games_before:
                            unanswered = 0
                        else:
                            unanswered += 1
                            wait = auto_seek_backoff_sec(unanswered) - 30.0
                            if wait > 0:
                                logger.info("Auto-seek: %s challenges unanswered, waiting %.0fs", unanswered, wait)
                                await asyncio.sleep(wait)
                        continue
            except Exception as exc:
                logger.debug("Auto-seek loop error: %s", exc)

            await asyncio.sleep(25.0)

    async def _challenge_random_online_bot(self, session: aiohttp.ClientSession) -> bool:
        preferred_targets = ["maia1", "maia5", "maia9"]
        now = time.time()
        try:
            url = f"{self.api_base}/api/bot/online?nb=50"
            async with session.get(url, headers=self.get_headers()) as resp:
                if resp.status == 200:
                    lines = (await resp.text()).strip().splitlines()
                    online_ids = []
                    for line in lines:
                        if not line.strip():
                            continue
                        try:
                            data = json.loads(line)
                            bid = str(data.get("id", "")).lower()
                            blitz_rating = data.get("perfs", {}).get("blitz", {}).get("rating")
                            max_auto_rating = self.bot_cfg.max_rating or 1850
                            if isinstance(blitz_rating, (int, float)) and blitz_rating > max_auto_rating:
                                continue
                            min_auto_rating = self.bot_cfg.min_rating
                            if min_auto_rating is not None and (
                                not isinstance(blitz_rating, (int, float)) or blitz_rating < min_auto_rating
                            ):
                                continue
                            if bid and bid != self.bot_id.lower() and not bid.startswith("leela"):
                                if self._failed_targets.get(bid, 0) < now:
                                    online_ids.append(bid)
                        except Exception:
                            continue

                    import random
                    # Prioritize varied bots in rating range rather than single crowded targets
                    random.shuffle(online_ids)
                    candidates = online_ids

                    for target_id in candidates[:15]:
                        chal_url = f"{self.api_base}/api/challenge/{target_id}"
                        data = aiohttp.FormData()
                        data.add_field("rated", "true")
                        data.add_field("clock.limit", "180")
                        data.add_field("clock.increment", "2")
                        try:
                            async with session.post(chal_url, headers=self.get_headers(), data=data) as chal_resp:
                                if chal_resp.status == 200:
                                    logger.info("Auto-seek: Sent challenge to online bot %s", target_id)
                                    return True
                                elif chal_resp.status == 429:
                                    resp_text = await chal_resp.text()
                                    logger.warning(
                                        "Lichess outgoing challenge endpoint is rate-limited (429: %s). Pausing auto-seek for 10 minutes to let cooldown expire...",
                                        resp_text[:80],
                                    )
                                    await asyncio.sleep(600.0)
                                    return False
                                else:
                                    text = await chal_resp.text()
                                    self._failed_targets[target_id] = now + 900.0  # 15 min cooldown
                                    logger.info("Challenge to %s unavailable (%s), trying next candidate...", target_id, text[:60])
                        except Exception as exc:
                            logger.debug("Challenge to %s error: %s", target_id, exc)
        except Exception as exc:
            logger.debug("Error seeking online bots: %s", exc)
        return False

    async def _handle_game_end(
        self,
        session: aiohttp.ClientSession,
        game_id: str,
        state: dict[str, Any],
        bot_color: chess.Color,
        opponent_name: str,
        opponent_elo: int | None,
        time_control_str: str,
        speed: str,
    ) -> None:
        winner = state.get("winner")
        status = state.get("status", "unknown")
        moves_str = state.get("moves", "")
        moves_count = (len(moves_str.split()) + 1) // 2 if moves_str.strip() else 0

        if winner == "white":
            result = "win" if bot_color == chess.WHITE else "loss"
        elif winner == "black":
            result = "win" if bot_color == chess.BLACK else "loss"
        else:
            result = "draw"

        logger.info("Game %s ended (%s) | Result: %s | Status: %s", game_id, time_control_str, result, status)

        engine_elo_after = None
        try:
            account_data = await self.fetch_account(session)
            engine_elo_after = EloTracker.parse_rating_from_account(account_data, speed=speed)
        except Exception as exc:
            logger.warning("Could not refresh account Elo: %s", exc)

        self.elo_tracker.record_match(
            game_id=game_id,
            opponent=opponent_name,
            opponent_elo=opponent_elo,
            result=result,
            engine_elo_after=engine_elo_after,
            time_control=time_control_str,
            moves_count=moves_count,
        )

        self.games_completed += 1
        if self.bot_cfg.max_games is not None and self.games_completed >= self.bot_cfg.max_games:
            logger.info("Reached maximum games limit (%d). Shutting down bot.", self.bot_cfg.max_games)
            self._shutdown_event.set()

    async def handle_challenge_event(self, session: aiohttp.ClientSession, challenge: dict[str, Any]) -> None:
        challenge_id = challenge.get("id", "")
        if not challenge_id:
            return

        direction = challenge.get("direction")
        challenger_id = str(challenge.get("challenger", {}).get("id", "")).lower()
        if direction == "out" or (challenger_id and challenger_id == self.bot_id.lower()):
            logger.debug("Ignoring outgoing challenge %s", challenge_id)
            return

        accept, decline_reason = should_accept_challenge(
            challenge,
            min_rating=self.bot_cfg.min_rating,
            max_rating=self.bot_cfg.max_rating,
            active_games_count=len(self.active_games),
            max_concurrent_games=self.bot_cfg.max_concurrent_games,
            allow_casual=self.bot_cfg.allow_casual,
        )

        if accept:
            self.active_games.add(challenge_id)
            ok = await self.accept_challenge(session, challenge_id)
            if not ok:
                self.active_games.discard(challenge_id)
        else:
            await self.decline_challenge(session, challenge_id, reason=decline_reason)

    async def run(self) -> None:
        async with aiohttp.ClientSession() as session:
            account = await self.fetch_account(session)
            self.bot_id = str(account.get("id", "")).lower()
            self.bot_username = str(account.get("username", ""))
            blitz_elo = EloTracker.parse_rating_from_account(account, "blitz")
            rapid_elo = EloTracker.parse_rating_from_account(account, "rapid")
            self.elo_tracker.set_initial_elo(blitz_elo or rapid_elo)

            print("=" * 65)
            print(f"[BOT] Connected to Lichess as: {self.bot_username} (ID: {self.bot_id})")
            print(f"      Live Blitz Elo: {blitz_elo if blitz_elo is not None else 'N/A'}")
            print(f"      Live Rapid Elo: {rapid_elo if rapid_elo is not None else 'N/A'}")
            print(f"      Listening for rated Blitz/Rapid challenges (Max concurrent: {self.bot_cfg.max_concurrent_games})...")
            print("=" * 65)

            if self.bot_cfg.auto_seek:
                asyncio.create_task(self.auto_seek_loop(session))

            retry_delay = 2.0
            event_stream_url = f"{self.api_base}/api/stream/event"

            while not self._shutdown_event.is_set():
                try:
                    logger.info("Opening event stream...")
                    async with session.get(event_stream_url, headers=self.get_headers()) as resp:
                        if resp.status != 200:
                            text = await resp.text()
                            logger.error("Event stream error (status %d): %s", resp.status, text)
                            await asyncio.sleep(retry_delay)
                            retry_delay = min(60.0, retry_delay * 2)
                            continue

                        retry_delay = 2.0
                        async for raw_line in resp.content:
                            if self._shutdown_event.is_set():
                                break

                            line = raw_line.decode("utf-8").strip()
                            if not line:
                                continue

                            try:
                                event = json.loads(line)
                            except json.JSONDecodeError:
                                continue

                            event_type = event.get("type")

                            if event_type == "challenge":
                                challenge_data = event.get("challenge", {})
                                asyncio.create_task(self.handle_challenge_event(session, challenge_data))

                            elif event_type == "gameStart":
                                game_data = event.get("game", {})
                                game_id = game_data.get("gameId") or game_data.get("id")
                                if game_id:
                                    asyncio.create_task(self.play_game(session, game_id))

                except asyncio.CancelledError:
                    break
                except Exception as exc:
                    logger.warning("Event stream disconnected: %s. Reconnecting...", exc)
                    await asyncio.sleep(retry_delay)
                    retry_delay = min(60.0, retry_delay * 2)

        summary = self.elo_tracker.get_summary()
        print("\n" + "=" * 65)
        print("[SESSION SUMMARY] LICHESS BOT RESULTS")
        print(f"   Total Games: {summary['total_games']}")
        print(f"   Score: W={summary['wins']} L={summary['losses']} D={summary['draws']} ({summary['win_rate']*100:.1f}%)")
        print(f"   Starting Elo: {summary['initial_elo']} | Current Elo: {summary['current_elo']} (Delta: {summary['elo_delta']})")
        print("=" * 65 + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lichess Bot CLI for Chess Engine")
    add_common_runtime_args(parser)
    parser.add_argument("--max-games", type=int, default=None, help="Maximum number of games to play before exiting")
    parser.add_argument("--min-rating", type=int, default=None, help="Minimum opponent rating to accept")
    parser.add_argument("--max-rating", type=int, default=None, help="Maximum opponent rating to accept")
    parser.add_argument("--max-sims", type=int, default=256, help="Maximum MCTS simulations per move (default: 256)")
    parser.add_argument("--auto-seek", action="store_true", help="Actively seek rated Blitz games whenever idle")
    parser.add_argument("--disallow-casual", action="store_true", help="Disallow unrated (casual) games")
    return parser


def load_production_engine(cfg, device: str, model_path_arg: str | None) -> Engine:
    model_path = model_path_arg or getattr(cfg.system, "checkpoint_path", "models/best_model.pth")
    p = Path(model_path)
    if not p.is_absolute():
        candidates = [
            p,
            Path.cwd() / p,
            Path(__file__).resolve().parents[2] / p,
        ]
        resolved = None
        for candidate in candidates:
            if candidate.exists():
                resolved = candidate
                break
        if resolved is None:
            raise FileNotFoundError(
                f"Model checkpoint not found at '{model_path}'. "
                "A valid trained model checkpoint is strictly required to run the Lichess Bot."
            )
        model_path = str(resolved)
    elif not p.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at '{model_path}'. "
            "A valid trained model checkpoint is strictly required to run the Lichess Bot."
        )

    logger.info("Loading production model checkpoint from %s onto %s", model_path, device)
    return Engine(
        model_path=model_path,
        cfg=cfg,
        device=device,
        allow_partial_weights=False,
    )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        token = get_api_token()
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    cfg, _, device = configure_runtime(args, "cli.lichess_bot")

    try:
        engine = load_production_engine(cfg=cfg, device=device, model_path_arg=args.model_path)
        logger.info("ChessNet Engine successfully loaded and validated on %s", device)
    except Exception as exc:
        logger.error("Engine initialization error: %s", exc)
        print(f"Error: Could not load trained engine model: {exc}", file=sys.stderr)
        sys.exit(1)

    bot_cfg = BotConfig(
        token=token,
        config_path=args.config,
        model_path=args.model_path or "models/best_model.pth",
        device=device,
        max_games=args.max_games,
        min_rating=args.min_rating,
        max_rating=args.max_rating,
        max_sims=args.max_sims,
        auto_seek=args.auto_seek,
        allow_casual=not getattr(args, "disallow_casual", False),
    )

    bot = LichessBot(bot_cfg=bot_cfg, engine=engine)

    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        print("\nBot interrupted by user. Exiting gracefully...")


if __name__ == "__main__":
    main()
