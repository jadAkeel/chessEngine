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


def _best_move_with_mcts(model, board, device, simulations):
    logger.info(f"🧠 MCTS START | sims={simulations}")

    mcts = MCTS(model=model, cfg=getattr(model, 'cfg', None), device=device)
    result = mcts.search(board, num_simulations=int(simulations))

    move = result.get('best_move')

    if move is None or move not in board.legal_moves:
        logger.error("❌ MCTS returned invalid move")
        raise HTTPException(500, 'MCTS did not return a legal move')

    logger.info(f"✅ MCTS BEST MOVE: {move.uci()}")
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
    logger.info("📥 /predict")

    model = _get_model()
    board = _validate_fen(req.fen)

    with torch.no_grad():
        logits, value = model.predict(board, device=_get_device())

    logger.info("📤 prediction done")

    return {
        'value': float(value),
        'moves': _legal_moves_with_probs(board, logits, req.topk)
    }


@app.post('/bestmove')
def bestmove(req: BestMoveRequest):
    logger.info("♟️ /bestmove")

    board = _validate_fen(req.fen)

    if board.is_game_over():
        logger.warning("⚠️ game over")
        raise HTTPException(400, 'Game over')

    sims = req.simulations or int(get_current_config().system.default_bestmove_simulations)

    move, san = _best_move_with_mcts(_get_model(), board, _get_device(), sims)

    return {'move': move, 'san': san, 'simulations': sims}


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
