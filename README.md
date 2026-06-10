# Chess Engine Project

This version has been cleaned up so the backend modules use one consistent configuration system and can run together again.

## What was fixed

- Unified the config layer and restored backward compatibility for the old `Config` style constants.
- Fixed constructor/signature mismatches across `ChessNet`, `encode_board`, `predict_boards`, and `MCTS`.
- Completed flat move indexing with `NUM_MOVES`, `move_to_index`, and `index_to_move`.
- Repaired engine, self-play, replay buffer, training loop, arena evaluation, API startup, and CLI entry points.
- Updated backend startup so it loads `backend/config/default.yaml` automatically.

## Project structure

- `frontend/` – React + Vite chess UI
- `backend/app/api/` – FastAPI endpoints and websocket
- `backend/app/core/` – engine wrapper
- `backend/app/model/` – neural network + checkpoint helpers
- `backend/app/mcts/` – MCTS search
- `backend/app/selfplay/` – self-play data generation
- `backend/app/training/` – replay buffer + trainer + loop
- `backend/config/default.yaml` – backend config

## Backend setup

From the project root:

```bash
cd backend
pip install -r requirements.txt
```

Run the API:

```bash
PYTHONPATH=. uvicorn app.api.main:app --reload
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

## CLI commands

Run training:

```bash
cd backend
PYTHONPATH=. python -m app.cli.train --config config/default.yaml --iterations 5 --device cpu
```

Generate self-play data:

```bash
cd backend
PYTHONPATH=. python -m app.cli.selfplay --config config/default.yaml --workers 1 --games-per-worker 1 --device cpu
```

Play in terminal:

```bash
cd backend
PYTHONPATH=. python -m app.cli.play --config config/default.yaml --device cpu
```

Evaluate a checkpoint:

```bash
cd backend
PYTHONPATH=. python -m app.cli.evaluate --config config/default.yaml --model-path models/best_model.pth --device cpu
```

## Frontend setup

```bash
cd frontend
npm install
npm run dev
```

The frontend expects the backend to be running locally.

## Notes

- If no checkpoint exists, the backend starts with random weights.
- The network output size is `4672` (`8 * 8 * 73`).
- Training and gameplay are now internally consistent, but engine strength still depends on real training data and checkpoints.
