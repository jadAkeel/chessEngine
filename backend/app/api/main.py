from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
import time

import chess
import torch
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from app.infra.config import get_current_config, load_config, validate_config
from app.infra.device import select_device
from app.infra.logging import setup_logging
from app.infra.runtime import configure_torch_runtime
from app.game.principles import principle_penalty_components
from app.mcts.search import MCTS
from app.model.checkpoint import CheckpointLoadError, load_checkpoint, load_compatible_weights
from app.model.network import ChessNet


# ================= INIT =================
load_config()
logger = setup_logging('api.main')

MODEL: ChessNet | None = None
DEVICE: str | None = None


# ================= CONNECTION MANAGER =================
class ConnectionManager:
    def __init__(self) -> None:
        self.rooms: dict[str, set[WebSocket]] = {}
        self.room_fens: dict[str, str] = {}
        self._lock = asyncio.Lock()

    async def connect(self, room_id: str, websocket: WebSocket) -> str:
        await websocket.accept()
        async with self._lock:
            self.rooms.setdefault(room_id, set()).add(websocket)
            fen = self.room_fens.setdefault(room_id, chess.STARTING_FEN)

        logger.info(f"🔌 WS CONNECTED | room={room_id} | clients={len(self.rooms[room_id])}")
        return fen

    async def disconnect(self, room_id: str, websocket: WebSocket) -> None:
        async with self._lock:
            room = self.rooms.get(room_id)
            if room is not None:
                room.discard(websocket)
                if not room:
                    self.rooms.pop(room_id, None)
                    self.room_fens.pop(room_id, None)

        logger.info(f"❌ WS DISCONNECTED | room={room_id}")

    async def update_room_fen(self, room_id: str, fen: str) -> None:
        async with self._lock:
            self.room_fens[room_id] = fen

    async def get_room_fen(self, room_id: str) -> str:
        async with self._lock:
            return self.room_fens.setdefault(room_id, chess.STARTING_FEN)

    async def broadcast(self, room_id: str, payload: dict) -> None:
        async with self._lock:
            sockets = list(self.rooms.get(room_id, set()))

        stale: list[WebSocket] = []
        for websocket in sockets:
            try:
                await websocket.send_json(payload)
            except Exception:
                stale.append(websocket)

        for websocket in stale:
            await self.disconnect(room_id, websocket)


manager = ConnectionManager()


# ================= SCHEMAS =================
class FenRequest(BaseModel):
    fen: str


class BestMoveRequest(FenRequest):
    simulations: int | None = Field(default=None, ge=1, le=80)


class PredictRequest(FenRequest):
    topk: int | None = Field(default=5, ge=1, le=50)


class FastMoveRequest(FenRequest):
    topk: int | None = Field(default=16, ge=1, le=36)
    depth: int | None = Field(default=6, ge=1, le=10)
    max_simulations: int | None = Field(default=None, ge=0, le=180)
    adaptive: bool = True


class MoveRequest(FenRequest):
    move: str

    @field_validator('move')
    @classmethod
    def validate_move_uci(cls, value: str) -> str:
        chess.Move.from_uci(value)
        return value


# ================= HELPERS =================
def _validate_fen(fen: str) -> chess.Board:
    try:
        return chess.Board(fen.strip())
    except Exception as exc:
        logger.error(f"❌ Invalid FEN: {fen}")
        raise HTTPException(400, f'Invalid FEN: {exc}') from exc


def _get_model() -> ChessNet:
    if MODEL is None:
        logger.error("❌ MODEL not loaded")
        raise HTTPException(503, 'Model not loaded')
    return MODEL


def _get_device() -> str:
    if DEVICE is None:
        logger.error("❌ DEVICE not initialized")
        raise HTTPException(503, 'Device not initialized')
    return DEVICE


def _load_model():
    logger.info("🔄 Loading model...")

    cfg = load_config()
    validate_config(cfg)

    device = select_device(cfg.system.device)
    configure_torch_runtime(cfg, device=str(device), role='api', worker_count=1)

    model = ChessNet(cfg).to(device)

    path = Path(cfg.system.checkpoint_path)
    logger.info(f"📦 Checkpoint path: {path}")

    if path.exists():
        try:
            state = load_checkpoint(path, model, device=str(device))
            if not state.get('loaded', False):
                raise CheckpointLoadError(f'Invalid checkpoint: {path}')
            logger.info("✅ Loaded FULL checkpoint")
        except CheckpointLoadError:
            load_compatible_weights(
                model,
                path,
                device=str(device),
                min_match_ratio=0.95,
                raise_on_mismatch=True
            )
            logger.warning("⚠️ Loaded PARTIAL weights")
    else:
        logger.warning("⚠️ NO CHECKPOINT → random weights (BAD)")

    model.eval()
    return model, str(device)


def _legal_moves_with_probs(board, logits, topk):
    from app.game.move_encoding import move_to_index

    probs = torch.softmax(torch.tensor(logits), dim=0)
    moves = []

    for move in board.legal_moves:
        try:
            idx = move_to_index(move, board)
            moves.append({
                'uci': move.uci(),
                'san': board.san(move),
                'prob': float(probs[idx])
            })
        except Exception:
            continue

    moves.sort(key=lambda x: x['prob'], reverse=True)
    return moves[:topk]


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


def _piece_value(piece_type: chess.PieceType | None) -> int:
    values = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 0,
    }
    return values.get(piece_type, 0)


def _is_opening_minor_development(board: chess.Board, move: chess.Move) -> bool:
    piece = board.piece_at(move.from_square)
    if piece is None or piece.piece_type not in {chess.KNIGHT, chess.BISHOP}:
        return False
    home_rank = 0 if piece.color == chess.WHITE else 7
    return chess.square_rank(move.from_square) == home_rank and chess.square_rank(move.to_square) != home_rank


def _fast_principle_penalty(board: chess.Board, move: chess.Move, candidate: chess.Board) -> float:
    if board.is_capture(move) or move.promotion or candidate.is_check():
        return 0.0
    return float(principle_penalty_components(
        board,
        candidate,
        move,
        get_current_config().principle_penalties,
    ).total)


def _score_fast_candidate(board: chess.Board, move: chess.Move, policy_prob: float) -> float:
    score = float(policy_prob) * 1000.0
    moving_piece = board.piece_at(move.from_square)
    material_gain = _material_gain_for_move(board, move)
    score += material_gain

    candidate = board.copy(stack=False)
    candidate.push(move)
    score -= _fast_principle_penalty(board, move, candidate) * 1400.0
    if _is_opening_minor_development(board, move):
        score += 45.0

    if candidate.is_checkmate():
        return score + 100000.0
    if _find_mate_in_one(candidate) is not None:
        score -= 50000.0
    if _side_can_create_promotion_threat(candidate):
        score -= 1200.0
    if _side_can_capture_queen(candidate):
        score -= 3000.0
    hanging_value = _max_valuable_capture_value(candidate, min_value=_piece_value(chess.KNIGHT))
    net_hanging_value = max(0, hanging_value - material_gain)
    if net_hanging_value >= _piece_value(chess.ROOK):
        score -= net_hanging_value * 2.0
    elif net_hanging_value >= _piece_value(chess.KNIGHT):
        score -= net_hanging_value * 1.25
    if candidate.is_check():
        score += 60.0

    mover = not candidate.turn
    destination = move.to_square
    is_attacked = candidate.is_attacked_by(candidate.turn, destination)
    is_defended = candidate.is_attacked_by(mover, destination)
    if is_attacked and moving_piece is not None:
        moved_value = _piece_value(move.promotion or moving_piece.piece_type)
        score -= moved_value * (0.15 if is_defended else 0.45)

    file_index = chess.square_file(destination)
    rank_index = chess.square_rank(destination)
    if file_index in (3, 4) and rank_index in (3, 4):
        score += 15.0

    return score


def _fast_score_diagnostics(board: chess.Board, candidates: list[dict], limit: int = 5) -> list[dict]:
    diagnostics: list[dict] = []
    for item in candidates[: max(1, int(limit))]:
        move_uci = item.get('uci')
        if not move_uci:
            continue
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            continue

        moving_piece = board.piece_at(move.from_square)
        captured_value = _captured_value_for_move(board, move)
        promotion_value = _piece_value(move.promotion) if move.promotion else 0
        candidate = _board_after_move(board, move)
        destination_attacked = candidate.is_attacked_by(candidate.turn, move.to_square)
        destination_defended = candidate.is_attacked_by(not candidate.turn, move.to_square)
        moved_value = _piece_value(move.promotion or (moving_piece.piece_type if moving_piece else None))
        destination_risk_penalty = 0.0
        if destination_attacked and moving_piece is not None:
            destination_risk_penalty = moved_value * (0.15 if destination_defended else 0.45)

        diagnostics.append({
            'uci': move.uci(),
            'san': board.san(move),
            'policy_prob': round(float(item.get('prob', 0.0)), 6),
            'fast_score': round(float(item.get('score', 0.0)), 3),
            'captured_value': captured_value,
            'promotion_value': promotion_value,
            'gives_check': candidate.is_check(),
            'allows_mate_one': _find_mate_in_one(candidate) is not None,
            'allows_promotion_threat': _side_can_create_promotion_threat(candidate),
            'allows_queen_capture': _side_can_capture_queen(candidate),
            'allows_valuable_capture_value': _max_valuable_capture_value(
                candidate,
                min_value=_piece_value(chess.KNIGHT),
            ),
            'destination_attacked': destination_attacked,
            'destination_defended': destination_defended,
            'destination_risk_penalty': round(float(destination_risk_penalty), 3),
        })
    return diagnostics


def _captured_value_for_move(board: chess.Board, move: chess.Move) -> int:
    if board.is_en_passant(move):
        return _piece_value(chess.PAWN)

    captured_piece = board.piece_at(move.to_square)
    if captured_piece is None:
        return 0
    return _piece_value(captured_piece.piece_type)


def _material_gain_for_move(board: chess.Board, move: chess.Move) -> int:
    material_gain = _captured_value_for_move(board, move)
    if move.promotion:
        material_gain += _piece_value(move.promotion)
    return material_gain


def _is_decisive_fast_choice(board: chess.Board, candidates: list[dict], reasons: list[str]) -> bool:
    reason_set = set(reasons)
    if {
        'king_in_check',
        'king_exposed',
        'best_fast_move_allows_mate',
        'best_fast_move_allows_mate_two',
        'best_fast_move_allows_promotion_threat',
        'best_fast_move_allows_queen_capture',
    }.intersection(reason_set):
        return False
    if len(candidates) < 2:
        return False

    move = chess.Move.from_uci(candidates[0]['uci'])
    if move not in board.legal_moves or _move_allows_mate_in_one(board, move.uci()):
        return False

    material_gain = _captured_value_for_move(board, move)
    if move.promotion:
        material_gain += _piece_value(move.promotion)
    if material_gain < _piece_value(chess.BISHOP):
        return False

    moving_piece = board.piece_at(move.from_square)
    moved_value = _piece_value(move.promotion or (moving_piece.piece_type if moving_piece else None))
    candidate = board.copy(stack=False)
    candidate.push(move)

    destination_attacked = candidate.is_attacked_by(candidate.turn, move.to_square)
    destination_defended = candidate.is_attacked_by(not candidate.turn, move.to_square)
    clearly_profitable = (
        not destination_attacked
        or material_gain >= moved_value
        or (destination_defended and material_gain >= moved_value + 200)
    )
    if not clearly_profitable:
        return False

    top_gap = float(candidates[0]['score']) - float(candidates[1]['score'])
    if material_gain >= _piece_value(chess.QUEEN):
        return top_gap >= 35.0
    if material_gain >= _piece_value(chess.ROOK):
        return top_gap >= 80.0
    return top_gap >= 140.0


def _find_mate_in_one(board: chess.Board) -> chess.Move | None:
    for move in board.legal_moves:
        candidate = board.copy(stack=False)
        candidate.push(move)
        if candidate.is_checkmate():
            return move
    return None


def _move_allows_mate_in_one(board: chess.Board, move_uci: str) -> bool:
    move = chess.Move.from_uci(move_uci)
    if move not in board.legal_moves:
        return True

    candidate = board.copy(stack=False)
    candidate.push(move)
    return _find_mate_in_one(candidate) is not None


def _side_has_forced_mate_in_two(board: chess.Board) -> bool:
    for attack_move in board.legal_moves:
        candidate = board.copy(stack=False)
        candidate.push(attack_move)
        if candidate.is_checkmate():
            return True
        if not candidate.is_check():
            continue

        replies = list(candidate.legal_moves)
        if replies and all(
            _find_mate_in_one(_board_after_move(candidate, reply)) is not None
            for reply in replies
        ):
            return True
    return False


def _move_allows_forced_mate_in_two(board: chess.Board, move_uci: str) -> bool:
    move = chess.Move.from_uci(move_uci)
    if move not in board.legal_moves:
        return True

    candidate = board.copy(stack=False)
    candidate.push(move)
    return _side_has_forced_mate_in_two(candidate)


def _promotion_distance(color: chess.Color, square: chess.Square) -> int:
    rank = chess.square_rank(square)
    return 7 - rank if color == chess.WHITE else rank


def _is_passed_pawn(board: chess.Board, square: chess.Square, color: chess.Color) -> bool:
    direction = 1 if color == chess.WHITE else -1
    file_index = chess.square_file(square)
    rank_index = chess.square_rank(square)

    for adjacent_file in range(max(0, file_index - 1), min(7, file_index + 1) + 1):
        check_rank = rank_index + direction
        while 0 <= check_rank <= 7:
            piece = board.piece_at(chess.square(adjacent_file, check_rank))
            if piece is not None and piece.color != color and piece.piece_type == chess.PAWN:
                return False
            check_rank += direction
    return True


def _is_dangerous_passed_pawn(board: chess.Board, square: chess.Square, color: chess.Color) -> bool:
    piece = board.piece_at(square)
    if piece is None or piece.color != color or piece.piece_type != chess.PAWN:
        return False
    return _promotion_distance(color, square) <= 2 and _is_passed_pawn(board, square, color)


def _side_can_create_promotion_threat(board: chess.Board) -> bool:
    color = board.turn
    for move in board.legal_moves:
        moving_piece = board.piece_at(move.from_square)
        if moving_piece is None or moving_piece.color != color or moving_piece.piece_type != chess.PAWN:
            continue
        if move.promotion:
            return True

        candidate = _board_after_move(board, move)
        if _is_dangerous_passed_pawn(candidate, move.to_square, color):
            return True
    return False


def _move_allows_promotion_threat(board: chess.Board, move_uci: str) -> bool:
    move = chess.Move.from_uci(move_uci)
    if move not in board.legal_moves:
        return True

    candidate = _board_after_move(board, move)
    return _side_can_create_promotion_threat(candidate)


def _captured_piece_for_move(board: chess.Board, move: chess.Move) -> chess.Piece | None:
    if board.is_en_passant(move):
        offset = -8 if board.turn == chess.WHITE else 8
        return board.piece_at(move.to_square + offset)
    return board.piece_at(move.to_square)


def _max_valuable_capture_value(board: chess.Board, min_value: int = 0) -> int:
    defender = not board.turn
    max_value = 0

    for move in board.legal_moves:
        if not board.is_capture(move):
            continue

        captured_piece = _captured_piece_for_move(board, move)
        if captured_piece is None or captured_piece.color != defender or captured_piece.piece_type == chess.KING:
            continue

        captured_value = _piece_value(captured_piece.piece_type)
        if captured_value < min_value:
            continue

        attacker_piece = board.piece_at(move.from_square)
        attacker_value = _piece_value(attacker_piece.piece_type if attacker_piece else None)
        target_defended = board.is_attacked_by(defender, move.to_square)
        bad_exchange = not target_defended or captured_value >= attacker_value + 200
        if bad_exchange:
            max_value = max(max_value, captured_value)

    return max_value


def _move_allows_valuable_piece_capture(board: chess.Board, move_uci: str, min_value: int | None = None) -> bool:
    move = chess.Move.from_uci(move_uci)
    if move not in board.legal_moves:
        return True

    candidate = _board_after_move(board, move)
    threshold = _piece_value(chess.KNIGHT) if min_value is None else int(min_value)
    hanging_value = _max_valuable_capture_value(candidate, min_value=threshold)
    net_hanging_value = max(0, hanging_value - _material_gain_for_move(board, move))
    return net_hanging_value >= threshold


def _side_can_capture_queen(board: chess.Board) -> bool:
    return _max_valuable_capture_value(board, min_value=_piece_value(chess.QUEEN)) >= _piece_value(chess.QUEEN)


def _move_allows_queen_capture(board: chess.Board, move_uci: str) -> bool:
    return _move_allows_valuable_piece_capture(board, move_uci, min_value=_piece_value(chess.QUEEN))


def _king_exposure_score(board: chess.Board, color: chess.Color) -> int:
    king_square = board.king(color)
    if king_square is None:
        return 0

    enemy = not color
    score = len(board.attackers(enemy, king_square)) * 2
    safe_squares = 0

    for square in chess.SquareSet(chess.BB_KING_ATTACKS[king_square]):
        piece = board.piece_at(square)
        if piece is not None and piece.color == color:
            continue
        if not board.is_attacked_by(enemy, square):
            safe_squares += 1

    enemy_queens = board.pieces(chess.QUEEN, enemy)
    queen_near_king = any(chess.square_distance(king_square, queen_square) <= 3 for queen_square in enemy_queens)
    if queen_near_king:
        score += 2
        if safe_squares <= 4:
            score += 1

    return score + max(0, 2 - safe_squares)


def _is_king_exposed(board: chess.Board, color: chess.Color) -> bool:
    return _king_exposure_score(board, color) >= 3


def _move_safety_flags(board: chess.Board, move_uci: str) -> dict[str, bool]:
    return {
        'mate_one': _move_allows_mate_in_one(board, move_uci),
        'mate_two': _move_allows_forced_mate_in_two(board, move_uci),
        'promotion_threat': _move_allows_promotion_threat(board, move_uci),
        'queen_capture': _move_allows_queen_capture(board, move_uci),
        'valuable_piece_capture': _move_allows_valuable_piece_capture(board, move_uci),
    }


def _has_safety_risk(flags: dict[str, bool]) -> bool:
    return any(flags.values())


def _has_critical_safety_risk(flags: dict[str, bool]) -> bool:
    return any(value for key, value in flags.items() if key != 'valuable_piece_capture')


def _safe_candidate_fallback(board: chess.Board, candidates: list[dict]) -> tuple[str, str, dict[str, bool]] | None:
    soft_fallback: tuple[str, str, dict[str, bool]] | None = None
    for candidate in candidates:
        move_uci = candidate.get('uci')
        if not move_uci:
            continue
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            continue
        flags = _move_safety_flags(board, move_uci)
        if not _has_safety_risk(flags):
            return move_uci, board.san(move), flags
        if soft_fallback is None and not _has_critical_safety_risk(flags):
            soft_fallback = (move_uci, board.san(move), flags)
    return soft_fallback


def _board_after_move(board: chess.Board, move: chess.Move) -> chess.Board:
    candidate = board.copy(stack=False)
    candidate.push(move)
    return candidate


def _is_forcing_move(board: chess.Board, move: chess.Move) -> bool:
    if board.is_capture(move) or move.promotion:
        return True

    candidate = board.copy(stack=False)
    candidate.push(move)
    return candidate.is_check()


def _fastmove_complexity(board: chess.Board, candidates: list[dict]) -> tuple[int, list[str]]:
    legal_moves = list(board.legal_moves)
    complexity = 0
    reasons: list[str] = []

    if board.is_check():
        complexity += 3
        reasons.append('king_in_check')
    elif _is_king_exposed(board, board.turn):
        complexity += 1
        reasons.append('king_exposed')

    if len(legal_moves) <= 8:
        complexity += 2
        reasons.append('few_legal_moves')
    elif len(legal_moves) <= 14:
        complexity += 1
        reasons.append('limited_legal_moves')

    forcing_count = sum(1 for move in legal_moves if _is_forcing_move(board, move))
    if forcing_count >= 5:
        complexity += 2
        reasons.append('many_forcing_moves')
    elif forcing_count >= 2:
        complexity += 1
        reasons.append('forcing_moves_available')

    if len(candidates) >= 2:
        top_gap = float(candidates[0]['score']) - float(candidates[1]['score'])
        if top_gap <= 35.0:
            complexity += 3
            reasons.append('top_moves_very_close')
        elif top_gap <= 80.0:
            complexity += 2
            reasons.append('top_moves_close')
        elif top_gap <= 160.0:
            complexity += 1
            reasons.append('top_moves_competitive')

    if candidates:
        best_move = chess.Move.from_uci(candidates[0]['uci'])
        candidate = board.copy(stack=False)
        candidate.push(best_move)
        if _find_mate_in_one(candidate) is not None:
            complexity += 4
            reasons.append('best_fast_move_allows_mate')
        elif _side_has_forced_mate_in_two(candidate):
            complexity += 4
            reasons.append('best_fast_move_allows_mate_two')
        elif _side_can_create_promotion_threat(candidate):
            complexity += 3
            reasons.append('best_fast_move_allows_promotion_threat')

        if _side_can_capture_queen(candidate):
            complexity += 4
            reasons.append('best_fast_move_allows_queen_capture')
        else:
            hanging_value = _max_valuable_capture_value(candidate, min_value=_piece_value(chess.KNIGHT))
            if hanging_value >= _piece_value(chess.ROOK):
                complexity += 3
                reasons.append('best_fast_move_allows_rook_capture')
            elif hanging_value >= _piece_value(chess.KNIGHT):
                complexity += 2
                reasons.append('best_fast_move_allows_minor_capture')

    high_value_capture = False
    for move in legal_moves:
        captured_piece = board.piece_at(move.to_square)
        if captured_piece is not None and _piece_value(captured_piece.piece_type) >= _piece_value(chess.ROOK):
            high_value_capture = True
            break
    if high_value_capture:
        complexity += 1
        reasons.append('high_value_capture')

    return complexity, reasons


def _is_light_adaptive_search(complexity: int, reasons: list[str], depth: int | None) -> bool:
    reason_set = set(reasons)
    if int(depth or 6) > 7:
        return False
    if complexity > 4:
        return False
    if reason_set.intersection({
        'king_in_check',
        'king_exposed',
        'best_fast_move_allows_mate',
        'best_fast_move_allows_mate_two',
        'best_fast_move_allows_promotion_threat',
        'best_fast_move_allows_queen_capture',
        'best_fast_move_allows_rook_capture',
        'few_legal_moves',
    }):
        return False
    return bool(reason_set.intersection({
        'forcing_moves_available',
        'many_forcing_moves',
        'top_moves_competitive',
        'top_moves_close',
        'top_moves_very_close',
        'high_value_capture',
        'best_fast_move_allows_minor_capture',
    }))


def _adaptive_simulations(
    depth: int | None,
    complexity: int,
    max_simulations: int | None = None,
    light: bool = False,
) -> int:
    depth = max(1, min(10, int(depth or 6)))

    if light:
        base = 12
    elif complexity >= 8:
        base = 96
    elif complexity >= 6:
        base = 76
    elif complexity >= 4:
        base = 64
    elif complexity >= 3:
        base = 48
    else:
        base = 12

    depth_factor = {
        1: 0.25,
        2: 0.35,
        3: 0.50,
        4: 0.65,
        5: 0.80,
        6: 1.00,
        7: 1.15,
        8: 1.30,
        9: 1.45,
        10: 1.60,
    }[depth]
    simulations = int(round(base * depth_factor))

    cap = 120 if max_simulations is None else max(12, min(180, int(max_simulations)))
    if light:
        cap = min(cap, 48)
    return min(cap, max(12, simulations))


def _adaptive_simulation_steps(
    depth: int | None,
    complexity: int,
    max_simulations: int | None = None,
    light: bool = False,
) -> list[int]:
    target = _adaptive_simulations(depth, complexity, max_simulations, light=light)
    if target <= 0:
        return []
    return [int(target)]


def _has_later_simulation_step(current_simulations: int, simulation_steps: list[int]) -> bool:
    current = int(current_simulations)
    return any(int(step) > current for step in simulation_steps)


def _mcts_confident_enough(
    *,
    fast_move: str,
    mcts_move: str,
    root_debug: list[dict],
    safety_flags: dict[str, bool],
    light: bool,
    current_simulations: int,
) -> bool:
    if _has_safety_risk(safety_flags):
        return False
    if light and mcts_move == fast_move:
        return True
    if not light and current_simulations < 32:
        return False
    if not root_debug:
        return mcts_move == fast_move and current_simulations >= 48

    by_visits = sorted(root_debug, key=lambda item: int(item.get('visits', 0)), reverse=True)
    top = by_visits[0]
    if top.get('uci') != mcts_move:
        return False

    top_visits = int(top.get('visits', 0))
    second_visits = int(by_visits[1].get('visits', 0)) if len(by_visits) > 1 else 0
    total_visits = max(1, sum(int(item.get('visits', 0)) for item in by_visits))
    visit_share = top_visits / total_visits
    visit_gap = top_visits - second_visits

    if current_simulations >= 48 and mcts_move == fast_move and visit_gap >= 2:
        return True
    if current_simulations >= 48 and visit_share >= 0.55 and visit_gap >= 3:
        return True
    if current_simulations >= 76 and visit_share >= 0.45 and visit_gap >= 4:
        return True
    return False


def _should_use_adaptive_search(complexity: int, reasons: list[str], depth: int | None) -> bool:
    reason_set = set(reasons)

    if (
        'king_in_check' in reason_set
        or 'best_fast_move_allows_mate' in reason_set
        or 'best_fast_move_allows_mate_two' in reason_set
        or 'best_fast_move_allows_promotion_threat' in reason_set
        or 'best_fast_move_allows_queen_capture' in reason_set
        or 'best_fast_move_allows_rook_capture' in reason_set
    ):
        return True
    if complexity >= 6:
        return True
    if 'few_legal_moves' in reason_set and reason_set.intersection({
        'forcing_moves_available',
        'many_forcing_moves',
        'top_moves_close',
        'top_moves_very_close',
        'high_value_capture',
    }):
        return True
    if 'top_moves_very_close' in reason_set and reason_set.intersection({
        'many_forcing_moves',
        'high_value_capture',
    }):
        return True
    if complexity >= 4 and 'many_forcing_moves' in reason_set and reason_set.intersection({
        'top_moves_close',
        'top_moves_very_close',
        'high_value_capture',
    }):
        return True
    if complexity >= 4 and 'high_value_capture' in reason_set and reason_set.intersection({
        'many_forcing_moves',
        'top_moves_competitive',
        'top_moves_close',
        'top_moves_very_close',
    }):
        return True
    if complexity >= 4 and 'top_moves_very_close' in reason_set and 'forcing_moves_available' in reason_set:
        return True
    if int(depth or 6) >= 8 and complexity >= 4:
        return True
    return False


def _fast_policy_move(
    board: chess.Board,
    logits,
    topk: int,
    legal_policy_moves: list[dict] | None = None,
) -> tuple[str, str, list[dict]]:
    scored: list[dict] = []

    policy_moves = legal_policy_moves if legal_policy_moves is not None else _legal_moves_with_probs(board, logits, topk)
    for item in policy_moves:
        move = chess.Move.from_uci(item['uci'])
        if move not in board.legal_moves:
            continue
        scored.append({
            **item,
            'score': _score_fast_candidate(board, move, item['prob']),
        })

    if not scored:
        legal_move = next(iter(board.legal_moves), None)
        if legal_move is None:
            raise HTTPException(400, 'Game over')
        return legal_move.uci(), board.san(legal_move), []

    scored.sort(key=lambda item: item['score'], reverse=True)
    best = chess.Move.from_uci(scored[0]['uci'])
    return best.uci(), board.san(best), scored


def _best_move_with_mcts(model, board, device, simulations, include_diagnostics: bool = False):
    start = time.perf_counter()
    logger.info(f"🧠 MCTS START | sims={simulations}")

    mcts = MCTS(model=model, cfg=getattr(model, 'cfg', None), device=device)
    result = mcts.search(board, num_simulations=int(simulations))

    move = result.get('best_move')

    if move is None or move not in board.legal_moves:
        logger.error("❌ MCTS returned invalid move")
        raise HTTPException(500, 'MCTS did not return a legal move')

    logger.info(f"✅ MCTS BEST MOVE: {move.uci()}")
    logger.info(f"MCTS DONE | move={move.uci()} | search_ms={_elapsed_ms(start)}")
    if include_diagnostics:
        return move.uci(), board.san(move), result.get('root_diagnostics', [])
    return move.uci(), board.san(move)


# ================= LIFESPAN =================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, DEVICE

    logger.info("🚀 STARTING CHESS API...")

    try:
        MODEL, DEVICE = _load_model()
        logger.info(f"✅ MODEL READY on {DEVICE}")
    except Exception:
        logger.exception("❌ FAILED TO LOAD MODEL")
        raise

    logger.info("🔥 SERVER READY")

    yield

    logger.info("🛑 SHUTDOWN")


# ================= APP =================
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*']
)


# ================= GLOBAL REQUEST LOGGER =================
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()

    logger.info(f"➡️ {request.method} {request.url}")

    response = await call_next(request)

    duration = round((time.time() - start) * 1000, 2)
    logger.info(f"⬅️ {response.status_code} ({duration} ms)")

    return response


# ================= ROUTES =================
@app.get('/')
def root():
    return {'ok': True}


@app.get('/health')
def health():
    return {'ok': True, 'model': MODEL is not None, 'device': str(DEVICE)}


@app.post('/predict')
def predict(req: PredictRequest):
    total_start = time.perf_counter()
    logger.info("📥 /predict")

    model = _get_model()
    validate_start = time.perf_counter()
    board = _validate_fen(req.fen)
    validate_ms = _elapsed_ms(validate_start)

    predict_start = time.perf_counter()
    with torch.no_grad():
        logits, value = model.predict(board, device=_get_device())
    predict_ms = _elapsed_ms(predict_start)

    rank_start = time.perf_counter()
    moves = _legal_moves_with_probs(board, logits, req.topk)
    rank_ms = _elapsed_ms(rank_start)
    total_ms = _elapsed_ms(total_start)
    logger.info(
        f"/predict TIMING | validate_ms={validate_ms} | predict_ms={predict_ms} | "
        f"rank_ms={rank_ms} | total_ms={total_ms} | topk={req.topk}"
    )

    logger.info("📤 prediction done")

    return {
        'value': float(value),
        'moves': moves,
        'timing_ms': {
            'validate': validate_ms,
            'predict': predict_ms,
            'rank': rank_ms,
            'total': total_ms,
        },
    }


@app.post('/fastmove')
def fastmove(req: FastMoveRequest):
    total_start = time.perf_counter()
    logger.info(
        f"/fastmove START | topk={req.topk} | depth={req.depth} | "
        f"max_sims={req.max_simulations} | adaptive={req.adaptive}"
    )

    model = _get_model()
    validate_start = time.perf_counter()
    board = _validate_fen(req.fen)
    validate_ms = _elapsed_ms(validate_start)

    if board.is_game_over():
        logger.warning("/fastmove game over")
        raise HTTPException(400, 'Game over')

    mate_move = _find_mate_in_one(board)
    if mate_move is not None:
        total_ms = _elapsed_ms(total_start)
        logger.info(
            f"/fastmove MATE_IN_ONE | move={mate_move.uci()} | "
            f"validate_ms={validate_ms} | total_ms={total_ms}"
        )
        return {
            'move': mate_move.uci(),
            'san': board.san(mate_move),
            'source': 'mate_in_one',
            'value': None,
            'candidates': [],
            'adaptive': {
                'depth': int(req.depth or 6),
                'complexity': 999,
                'reasons': ['mate_in_one'],
                'simulations': 0,
                'max_simulations': req.max_simulations,
            },
            'timing_ms': {
                'validate': validate_ms,
                'predict': 0.0,
                'rank': 0.0,
                'search': 0.0,
                'total': total_ms,
            },
        }

    predict_start = time.perf_counter()
    with torch.no_grad():
        logits, value = model.predict(board, device=_get_device())
    predict_ms = _elapsed_ms(predict_start)

    rank_start = time.perf_counter()
    raw_policy_top = _legal_moves_with_probs(board, logits, int(req.topk or 8))
    move, san, candidates = _fast_policy_move(board, logits, int(req.topk or 8), raw_policy_top)
    rank_ms = _elapsed_ms(rank_start)
    fast_move, fast_san = move, san
    fast_score_debug = _fast_score_diagnostics(board, candidates)
    complexity, adaptive_reasons = _fastmove_complexity(board, candidates)
    decisive_fast_choice = _is_decisive_fast_choice(board, candidates, adaptive_reasons)
    full_adaptive_search = bool(
        req.adaptive
        and not decisive_fast_choice
        and _should_use_adaptive_search(complexity, adaptive_reasons, req.depth)
    )
    light_adaptive_search = bool(
        req.adaptive
        and not full_adaptive_search
        and not decisive_fast_choice
        and _is_light_adaptive_search(complexity, adaptive_reasons, req.depth)
    )
    use_adaptive_search = bool(full_adaptive_search or light_adaptive_search)
    baseline_adaptive_probe = bool(use_adaptive_search and not full_adaptive_search and not light_adaptive_search)
    light_budget = bool(not full_adaptive_search)
    simulation_steps = (
        _adaptive_simulation_steps(req.depth, complexity, req.max_simulations, light=light_budget)
        if use_adaptive_search
        else []
    )
    simulations = simulation_steps[-1] if simulation_steps else 0
    simulation_total = 0
    confidence_stop = False
    search_ms = 0.0
    source = 'fast_policy'
    rejected_safety_reasons: list[str] = []
    mcts_root_debug: list[dict] = []
    mcts_attempts: list[dict] = []

    if simulation_steps:
        search_start = time.perf_counter()
        try:
            for step_simulations in simulation_steps:
                mcts_move, mcts_san, mcts_root_debug = _best_move_with_mcts(
                    model,
                    board,
                    _get_device(),
                    step_simulations,
                    include_diagnostics=True,
                )
                mcts_safety = _move_safety_flags(board, mcts_move)
                step_confident = _mcts_confident_enough(
                    fast_move=fast_move,
                    mcts_move=mcts_move,
                    root_debug=mcts_root_debug,
                    safety_flags=mcts_safety,
                    light=light_budget,
                    current_simulations=step_simulations,
                )
                mcts_attempts.append({
                    'sims': step_simulations,
                    'move': mcts_move,
                    'san': mcts_san,
                    'safety': mcts_safety,
                    'confident': step_confident,
                    'root': mcts_root_debug[:5],
                })
                safe_fallback = _safe_candidate_fallback(board, candidates) if _has_safety_risk(mcts_safety) else None
                if safe_fallback is not None and safe_fallback[0] != mcts_move:
                    rejected_safety_reasons = [key for key, value in mcts_safety.items() if value]
                    if _has_later_simulation_step(step_simulations, simulation_steps):
                        logger.info(
                            f"/fastmove MCTS safety risk move={mcts_move}; "
                            f"retrying with higher budget | risks={rejected_safety_reasons}"
                        )
                        continue

                    move, san, _fallback_safety = safe_fallback
                    logger.warning(
                        f"/fastmove MCTS rejected safety risk move={mcts_move}; "
                        f"fallback={move} | risks={rejected_safety_reasons}"
                    )
                    source = 'adaptive_mcts_rejected_safety'
                    confidence_stop = True
                    break

                if baseline_adaptive_probe and not step_confident:
                    move, san = fast_move, fast_san
                    source = 'adaptive_probe_kept_fast'
                else:
                    move, san = mcts_move, mcts_san
                    source = 'adaptive_mcts_confident' if step_confident else 'adaptive_mcts'
                if step_confident or step_simulations == simulation_steps[-1]:
                    confidence_stop = step_confident
                    break
        except Exception:
            logger.exception("/fastmove adaptive MCTS failed; falling back to fast policy")
        search_ms = _elapsed_ms(search_start)
        simulation_total = sum(int(attempt.get('sims', 0)) for attempt in mcts_attempts)

    final_safety = _move_safety_flags(board, move)
    if _has_safety_risk(final_safety):
        safe_fallback = _safe_candidate_fallback(board, candidates)
        if safe_fallback is not None and safe_fallback[0] != move:
            rejected_safety_reasons = [key for key, value in final_safety.items() if value]
            logger.warning(
                f"/fastmove final move rejected safety risk move={move}; "
                f"fallback={safe_fallback[0]} | risks={rejected_safety_reasons}"
            )
            move, san, final_safety = safe_fallback
            source = f"{source}_safety_fallback"

    total_ms = _elapsed_ms(total_start)

    decision_debug = {
        'raw_policy_top': raw_policy_top[:5],
        'fast_scores': fast_score_debug,
        'mcts_root': mcts_root_debug[:5],
        'mcts_attempts': mcts_attempts,
    }
    logger.info(
        f"/fastmove DECISION_DEBUG | raw_policy_top={decision_debug['raw_policy_top']} | "
        f"fast_scores={decision_debug['fast_scores']} | mcts_root={decision_debug['mcts_root']}"
    )

    logger.info(
        f"/fastmove TIMING | move={move} | source={source} | validate_ms={validate_ms} | "
        f"predict_ms={predict_ms} | rank_ms={rank_ms} | search_ms={search_ms} | "
        f"total_ms={total_ms} | candidates={len(candidates)} | complexity={complexity} | "
        f"adaptive_search={use_adaptive_search} | decisive_fast={decisive_fast_choice} | "
        f"light_adaptive={light_adaptive_search} | sims={simulations} | steps={simulation_steps} | "
        f"sim_total={simulation_total} | confidence_stop={confidence_stop} | reasons={adaptive_reasons} | "
        f"final_safety={final_safety} | rejected_safety={rejected_safety_reasons}"
    )

    return {
        'move': move,
        'san': san,
        'source': source,
        'value': float(value),
        'candidates': candidates[:5],
        'adaptive': {
            'depth': int(req.depth or 6),
            'complexity': complexity,
            'reasons': adaptive_reasons,
            'search': use_adaptive_search,
            'decisive_fast': decisive_fast_choice,
            'light_adaptive': light_adaptive_search,
            'simulations': simulations,
            'simulation_steps': simulation_steps,
            'simulation_total': simulation_total,
            'confidence_stop': confidence_stop,
            'max_simulations': req.max_simulations,
            'final_safety': final_safety,
            'rejected_safety': rejected_safety_reasons,
        },
        'diagnostics': decision_debug,
        'timing_ms': {
            'validate': validate_ms,
            'predict': predict_ms,
            'rank': rank_ms,
            'search': search_ms,
            'total': total_ms,
        },
    }


@app.post('/bestmove')
def bestmove(req: BestMoveRequest):
    total_start = time.perf_counter()
    logger.info("♟️ /bestmove")

    validate_start = time.perf_counter()
    board = _validate_fen(req.fen)
    validate_ms = _elapsed_ms(validate_start)

    if board.is_game_over():
        logger.warning("⚠️ game over")
        raise HTTPException(400, 'Game over')

    sims = req.simulations or int(get_current_config().system.default_bestmove_simulations)

    move, san = _best_move_with_mcts(_get_model(), board, _get_device(), sims)
    total_ms = _elapsed_ms(total_start)
    logger.info(
        f"/bestmove TIMING | move={move} | validate_ms={validate_ms} | "
        f"total_ms={total_ms} | sims={sims}"
    )

    return {
        'move': move,
        'san': san,
        'simulations': sims,
        'timing_ms': {
            'validate': validate_ms,
            'total': total_ms,
        },
    }


# ================= WEBSOCKET =================
@app.websocket('/ws/{room_id}')
async def websocket_endpoint(websocket: WebSocket, room_id: str):
    initial_fen = await manager.connect(room_id, websocket)

    await websocket.send_json({
        'type': 'init',
        'room_id': room_id,
        'fen': initial_fen
    })

    try:
        while True:
            data = await websocket.receive_json()
            message_type = data.get('type')

            logger.info(f"📡 WS message: {message_type}")

            if message_type == 'analyze':
                board = _validate_fen(data['fen'])
                logits, value = _get_model().predict(board, device=_get_device())

                await websocket.send_json({
                    'type': 'analyze',
                    'value': float(value),
                    'moves': _legal_moves_with_probs(board, logits, topk=5),
                    'room_id': room_id,
                })

            elif message_type == 'move':
                move_text = data.get('move')
                if not move_text:
                    await websocket.send_json({
                        'type': 'error',
                        'error': 'Move message requires a UCI move',
                        'room_id': room_id
                    })
                    continue

                board = chess.Board(await manager.get_room_fen(room_id))
                try:
                    move = chess.Move.from_uci(str(move_text))
                except ValueError:
                    await websocket.send_json({
                        'type': 'error',
                        'error': 'Invalid UCI move',
                        'room_id': room_id
                    })
                    continue

                if move not in board.legal_moves:
                    await websocket.send_json({
                        'type': 'error',
                        'error': 'Illegal move',
                        'room_id': room_id,
                        'fen': board.fen()
                    })
                    continue

                board.push(move)

                await manager.update_room_fen(room_id, board.fen())
                await manager.broadcast(room_id, {
                    'type': 'move',
                    'room_id': room_id,
                    'move': move.uci(),
                    'fen': board.fen()
                })

            else:
                logger.warning("⚠️ unknown WS message")

                await websocket.send_json({
                    'type': 'error',
                    'error': 'Unsupported message type',
                    'room_id': room_id
                })

    except WebSocketDisconnect:
        await manager.disconnect(room_id, websocket)
