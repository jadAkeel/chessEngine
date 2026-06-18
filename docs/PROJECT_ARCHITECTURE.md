# Project Architecture

This project is a full-stack chess engine built around a Python backend and a React/Vite frontend. The backend contains the chess model, move encoding, Monte Carlo Tree Search, self-play generation, training loops, evaluation tools, CLI entry points, and FastAPI service. The frontend provides a playable chess UI that talks to the backend for engine moves.

## High-Level Flow

The main runtime flow is:

1. A chess position is represented by a `python-chess` board or a FEN string.
2. The backend encodes the board into tensor planes.
3. `ChessNet` predicts two outputs:
   - policy logits over the move space
   - a value estimate for the side to move
4. MCTS uses the model output as guidance, searches legal continuations, applies safety and repetition penalties, and returns a move.
5. The FastAPI backend exposes this through HTTP endpoints.
6. The React frontend sends the current FEN to the backend and applies the returned move on the board.

The engine is therefore not only a legal move generator. It combines learned evaluation, search, chess rules, safety heuristics, and training data generation.

## Backend Structure

Important backend areas:

- `backend/app/game/`: board encoding, move encoding, chess rule helpers, repetition detection, and principle penalties.
- `backend/app/model/`: the PyTorch neural network, batched inference helpers, and checkpoint loading/saving.
- `backend/app/mcts/`: Monte Carlo Tree Search nodes, search logic, visit policy, temperature handling, and move penalties.
- `backend/app/core/`: the `Engine` wrapper that combines a model, config, device, cache, and MCTS.
- `backend/app/selfplay/`: self-play game generation for reinforcement learning data.
- `backend/app/training/`: replay buffer, supervised/external sample loading, trainer, and training loops.
- `backend/app/evaluation/`: arena matches, metrics, and benchmark helpers.
- `backend/app/api/`: FastAPI routes and websocket support.
- `backend/app/cli/`: command-line entry points for training, self-play, play, evaluation, and external training.

## Frontend Structure

The frontend is a React/Vite app. The main chess screen is implemented in `frontend/src/pages/ChessHybridApp.jsx`, with supporting board UI in `frontend/src/components/ChessBoardPanel.jsx`.

The frontend uses:

- `chess.js` for client-side legal move handling.
- `react-chessboard` for rendering the board.
- Fetch requests to the backend for engine decisions.
- WebSockets for the multiplayer room flow.

In local development, the frontend defaults to:

```powershell
http://localhost:8000
```

as the backend API base URL. In production it uses the configured hosted API URL unless overridden with `VITE_API_BASE_URL`.

## Board Encoding

The neural network does not receive a visual chessboard. It receives a tensor with 20 input planes by default:

- 12 planes for piece placement: 6 piece types for White and 6 for Black.
- 1 plane for side to move.
- 4 planes for castling rights.
- 1 plane for en passant square.
- 1 plane for the halfmove clock.
- 1 plane for the fullmove number.

The board is oriented from the side to move where needed, so the model can learn from a consistent perspective.

Current config:

```yaml
model:
  input_planes: 20
```

## Move Encoding

The policy head predicts over an AlphaZero-style flat move space:

```text
8 * 8 * 73 = 4672 possible policy indexes
```

Each index represents a source square and one of 73 movement planes:

- sliding move directions and distances
- knight moves
- underpromotion planes

Queen promotions are encoded through the normal directional move planes and restored when decoding against the real board position.

Only legal moves are considered when ranking model policy output for an actual board.

## Neural Network

The model is implemented as `ChessNet`. It has:

- an input convolution block
- a residual tower
- a Policy Head for move probabilities
- a Value Head for position strength

Current model config:

```yaml
model:
  input_planes: 20
  channels: 160
  res_blocks: 24
  value_dropout: 0.05
```

The current configured model has `11,757,287` trainable parameters.

## Engine Wrapper

The `Engine` class is the main backend wrapper around the model and search. It:

- loads a checkpoint when a model path is provided
- validates config
- moves the model to the selected device
- creates the MCTS instance
- exposes `analyze()` and `get_best_move()`
- caches analysis results when search noise is disabled

`analyze()` returns:

- best move
- root score
- visit counts
- effective policy
- optional penalty diagnostics

## API Layer

The FastAPI backend exposes the engine through these main endpoints:

- `GET /health`: checks service readiness and whether the model is loaded.
- `POST /predict`: returns raw model value and top policy moves among legal moves.
- `POST /bestmove`: runs MCTS with a requested or default simulation count.
- `POST /fastmove`: uses policy ranking plus adaptive MCTS and safety checks for a faster UI move.
- `WS /ws/{room_id}`: synchronizes room state for multiplayer.

The API loads the configured checkpoint from:

```yaml
system:
  checkpoint_path: models/best_model.pth
```

If no checkpoint exists, API startup requires `ALLOW_RANDOM_WEIGHTS=1` for testing only.

## CLI Tools

The backend can also run without the frontend:

```powershell
cd backend
$env:PYTHONPATH="."
python -m app.cli.play
```

Other useful commands:

```powershell
python -m app.cli.train
python -m app.cli.train_external --config config/external_training.yaml
python -m app.cli.selfplay
python -m app.cli.evaluate
```

## How Components Connect

The complete system can be summarized as:

```text
Frontend board
  -> sends FEN to FastAPI
  -> backend validates FEN with python-chess
  -> board encoder creates 20x8x8 tensor
  -> ChessNet predicts policy logits and value
  -> MCTS searches legal continuations
  -> penalties and repetition filters adjust move quality
  -> API returns UCI/SAN move
  -> frontend applies the move and updates the UI
```

Training uses the same core pieces in a different loop:

```text
Self-play or external data
  -> encoded states, policy targets, value targets
  -> replay buffer or external sample stream
  -> ChessNet optimization
  -> checkpoint
  -> stronger engine for search and future data generation
```
