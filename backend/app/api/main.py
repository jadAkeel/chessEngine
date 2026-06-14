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


def _score_fast_candidate(board: chess.Board, move: chess.Move, policy_prob: float) -> float:
    score = float(policy_prob) * 1000.0
    moving_piece = board.piece_at(move.from_square)
    captured_piece = board.piece_at(move.to_square)

    if captured_piece is not None:
        score += _piece_value(captured_piece.piece_type)
    if move.promotion:
        score += _piece_value(move.promotion)

    candidate = board.copy(stack=False)
    candidate.push(move)

    if candidate.is_checkmate():
        return score + 100000.0
    if _find_mate_in_one(candidate) is not None:
        score -= 50000.0
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


def _adaptive_simulations(depth: int | None, complexity: int, max_simulations: int | None = None) -> int:
    depth = max(1, min(10, int(depth or 6)))

    if complexity >= 8:
        base = 72
    elif complexity >= 6:
        base = 56
    elif complexity >= 4:
        base = 40
    elif complexity >= 3:
        base = 28
    elif complexity >= 2 and depth >= 7:
        base = 18
    else:
        return 0

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

    cap = 120 if max_simulations is None else max(0, min(180, int(max_simulations)))
    return min(cap, max(1, simulations))


def _fast_policy_move(board: chess.Board, logits, topk: int) -> tuple[str, str, list[dict]]:
    scored: list[dict] = []

    for item in _legal_moves_with_probs(board, logits, topk):
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


def _best_move_with_mcts(model, board, device, simulations):
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
    move, san, candidates = _fast_policy_move(board, logits, int(req.topk or 8))
    rank_ms = _elapsed_ms(rank_start)
    fast_move, fast_san = move, san
    complexity, adaptive_reasons = _fastmove_complexity(board, candidates)
    simulations = _adaptive_simulations(req.depth, complexity, req.max_simulations) if req.adaptive else 0
    search_ms = 0.0
    source = 'fast_policy'

    if simulations > 0:
        search_start = time.perf_counter()
        try:
            mcts_move, mcts_san = _best_move_with_mcts(model, board, _get_device(), simulations)
            if _move_allows_mate_in_one(board, mcts_move) and not _move_allows_mate_in_one(board, fast_move):
                logger.warning(
                    f"/fastmove MCTS rejected unsafe move={mcts_move}; "
                    f"fallback={fast_move}"
                )
                source = 'adaptive_mcts_rejected'
            else:
                move, san = mcts_move, mcts_san
                source = 'adaptive_mcts'
        except Exception:
            logger.exception("/fastmove adaptive MCTS failed; falling back to fast policy")
        search_ms = _elapsed_ms(search_start)

    total_ms = _elapsed_ms(total_start)

    logger.info(
        f"/fastmove TIMING | move={move} | source={source} | validate_ms={validate_ms} | "
        f"predict_ms={predict_ms} | rank_ms={rank_ms} | search_ms={search_ms} | "
        f"total_ms={total_ms} | candidates={len(candidates)} | complexity={complexity} | "
        f"sims={simulations} | reasons={adaptive_reasons}"
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
            'simulations': simulations,
            'max_simulations': req.max_simulations,
        },
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
