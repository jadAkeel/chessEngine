# AGENTS.md

Guidance for Codex and other coding agents working in this repository.

## Project Overview

This is a chess engine project with a Python backend and a React/Vite frontend.

- `backend/` contains the chess engine, model, MCTS, training, evaluation, and CLI code.
- `frontend/` contains the web UI for interacting with the chess app.
- Large data/model artifacts may exist under `backend/data/` and `backend/models/`.

## Repository Notes

- This workspace may not be initialized as a Git repository.
- Avoid deleting or rewriting large data/model files unless the user explicitly asks.
- Prefer small, targeted changes that follow the existing module structure.
- The README mentions `backend/app/api/main.py`, but the API folder may be missing. Verify the backend API entrypoint before assuming it exists.

## Backend

### Setup

```powershell
cd C:\Users\10User\Desktop\ai\chesEngineWithData\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH="."
```

### Run API

The README suggests:

```powershell
uvicorn app.api.main:app --reload
```

Before using this command, confirm that `backend/app/api/main.py` exists. If it does not, inspect the backend package and either restore/create the API entrypoint or use the CLI tools instead.

### CLI Commands

Run backend CLI modules from inside `backend/` with `PYTHONPATH` set:

```powershell
python -m app.cli.train
python -m app.cli.selfplay
python -m app.cli.play
python -m app.cli.evaluate
```

### Tests

From `backend/`:

```powershell
pytest
```

If dependencies are missing:

```powershell
pip install -r requirements.txt
pip install pytest
```

## Frontend

### Setup

```powershell
cd C:\Users\10User\Desktop\ai\chesEngineWithData\frontend
npm install
```

### Run

```powershell
npm run dev
```

Vite usually serves the app at:

```text
http://localhost:5173
```

### Build

```powershell
npm run build
```

## Important Frontend Files

- `frontend/src/App.jsx`
- `frontend/src/pages/ChessHybridApp.jsx`
- `frontend/src/components/ChessBoardPanel.jsx`

Use existing React component patterns and styling conventions. For UI work, run the dev server and visually verify the result in the browser when possible.

## Important Backend Areas

- `backend/app/core/engine.py`
- `backend/app/model/`
- `backend/app/mcts/`
- `backend/app/game/`
- `backend/app/training/`
- `backend/app/evaluation/`
- `backend/app/cli/`
- `backend/config/`

## Agent Workflow

1. Read the relevant files before editing.
2. Prefer `rg` / `rg --files` for searching.
3. Keep changes focused on the user request.
4. Do not revert user changes unless explicitly requested.
5. Use structured parsers/APIs where available instead of fragile string manipulation.
6. After code changes, run the smallest useful verification command.
7. For frontend changes, also verify the rendered UI when feasible.

## Editing Rules

- Use `apply_patch` for manual file edits.
- Do not use destructive commands like `git reset --hard` or broad recursive deletes.
- Do not modify generated data, checkpoints, or model artifacts unless necessary for the task.
- Keep comments concise and only add them when they clarify non-obvious logic.

## Common Troubleshooting

### Backend API Import Fails

If this fails:

```powershell
uvicorn app.api.main:app --reload
```

Check whether `backend/app/api/main.py` exists. If missing, inspect the frontend API calls and backend CLI/engine modules to decide whether to create a FastAPI entrypoint or adjust the run instructions.

### Frontend Cannot Connect To Backend

Check:

- Backend server is actually running.
- Frontend API base URL configuration.
- Browser console and network tab.
- CORS settings if a FastAPI API exists or is added.

### Tests Cannot Import `app`

Run tests from `backend/` and set:

```powershell
$env:PYTHONPATH="."
```

Then run:

```powershell
pytest
```
