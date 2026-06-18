# MCTS Move Selection And Demo

The engine chooses moves by combining neural-network intuition with Monte Carlo Tree Search. The network gives an initial policy and value. MCTS improves that initial guess by searching legal continuations and updating tree statistics.

## Neural Network Intuition

For a given board, `ChessNet` predicts:

- policy logits over `4672` possible move indexes
- a value estimate in `[-1, 1]`

The policy logits are converted into probabilities only for legal moves. This gives MCTS a prior probability for each legal child move.

The value estimate gives MCTS a first evaluation of leaf positions. The search also blends this neural value with a classical board evaluation, controlled by `mcts.classical_value_alpha`.

## MCTS Search Loop

The search process is:

1. Expand the root using model policy and value.
2. Repeat for the configured number of simulations.
3. Select child moves using a PUCT-style score.
4. Apply virtual visits while batched inference is pending.
5. Expand leaf nodes with model predictions.
6. Backpropagate the evaluated value through the search path.
7. Build a policy target from child visit counts.
8. Adjust the root policy for repetition and penalties.
9. Select the final root move.

Default config includes:

```yaml
mcts:
  num_simulations: 256
  c_puct: 1.8
  temperature: 1.0
  inference_batch_size: 24
  classical_value_alpha: 0.35
```

The API can use smaller or adaptive simulation budgets for interactive speed.

## Selection

During selection, each child receives a score based on:

- `Q`: the average value from previous simulations
- `U`: exploration bonus from the policy prior
- virtual-loss penalty for in-flight batched simulations
- move penalties for risky or unproductive moves

The core idea is:

```text
score = value term + exploration term - virtual loss - move penalties
```

The search therefore balances moves that already look good with moves that the policy prior says deserve exploration.

## Expansion And Evaluation

When MCTS reaches a leaf node, it asks the neural network for:

- policy logits for legal child priors
- value for the leaf position

The engine expands children only for legal moves. If root noise is enabled, it can mix Dirichlet noise into root priors to encourage exploration during self-play.

The value used by MCTS is blended:

```text
blended_value = classical_alpha * classical_value + (1 - classical_alpha) * neural_value
```

Progress penalties can reduce the value when the halfmove clock suggests the game is becoming stagnant.

## MCTS Backpropagation

MCTS backpropagation updates search-tree statistics, not neural-network weights.

For each node in the reversed search path:

- increment `visit_count`
- add the current value to `value_sum`
- flip the sign of the value for the opponent perspective

The node quality is:

```text
q = value_sum / visit_count
```

This tells future simulations which branches have performed well in search.

## MCTS Backpropagation Vs Neural Network Backpropagation

These two processes share the word "backpropagation", but they are different:

- MCTS backpropagation updates tree statistics such as visits and average value.
- Neural-network backpropagation computes gradients and updates model weights.

MCTS backpropagation happens during move search. It affects only the current search tree.

Neural-network backpropagation happens during training. It changes `ChessNet` parameters through the optimizer.

## Penalties And Repetition Handling

The search includes penalties that guide the engine away from weak behavior:

- oscillating a piece back and forth
- repeating positions without useful progress
- reaching draw-claim conditions
- increasing the halfmove clock without capture or pawn progress
- leaving valuable pieces tactically exposed
- violating chess principles such as king safety, development, center control, and piece activity

At the root, the policy can also be adjusted to reduce repeated positions. This helps self-play generate more useful games and helps the engine avoid unnecessary repetition in play.

## Final Move Selection

After simulations finish, the engine builds a policy target from root child visit counts. Temperature controls how sharp or broad this distribution is.

The final move is selected from the adjusted root policy. If the top two moves are close, the engine compares penalties and can prefer the safer move.

The API returns:

- `best_move`
- visit counts
- policy target
- adjusted policy target
- root value
- optional diagnostics

## Fast Move Path

The `/fastmove` endpoint is optimized for interactive frontend play.

It first ranks legal moves from the neural policy, then scores candidates with lightweight tactical and safety checks. If the position is complex or risky, it escalates to adaptive MCTS.

The response includes:

- selected move
- SAN notation
- source of decision, such as `fast_policy` or `adaptive_mcts`
- candidate list
- adaptive-search details
- timing information
- diagnostics

This path is useful for UI responsiveness. `/bestmove` is the simpler fixed-budget MCTS endpoint.

## Run The Backend API

From the backend folder:

```powershell
cd backend
$env:PYTHONPATH="."
uvicorn app.api.main:app --reload
```

If no checkpoint exists and this is only a local smoke demo, allow random weights explicitly:

```powershell
$env:ALLOW_RANDOM_WEIGHTS="1"
uvicorn app.api.main:app --reload
```

For real engine play, use a trained checkpoint at the configured path:

```text
backend/models/best_model.pth
```

## Health Check

```powershell
curl http://127.0.0.1:8000/health
```

Expected shape:

```json
{
  "ok": true,
  "model": true,
  "device": "cpu"
}
```

The device value can differ depending on local hardware and config.

## Predict Demo

`/predict` returns the model value and top legal policy moves without full MCTS.

```powershell
curl -X POST http://127.0.0.1:8000/predict `
  -H "Content-Type: application/json" `
  -d "{\"fen\":\"rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1\",\"topk\":5}"
```

Use this to inspect raw model intuition.

## Best Move Demo

`/bestmove` runs fixed-budget MCTS.

```powershell
curl -X POST http://127.0.0.1:8000/bestmove `
  -H "Content-Type: application/json" `
  -d "{\"fen\":\"rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1\",\"simulations\":32}"
```

Expected response shape:

```json
{
  "move": "e7e5",
  "san": "e5",
  "simulations": 32,
  "timing_ms": {
    "validate": 0.0,
    "total": 0.0
  }
}
```

The exact move and timing depend on the checkpoint, device, and simulation count.

## Fast Move Demo

`/fastmove` uses fast policy scoring and adaptive MCTS when needed.

```powershell
curl -X POST http://127.0.0.1:8000/fastmove `
  -H "Content-Type: application/json" `
  -d "{\"fen\":\"rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1\",\"topk\":16,\"depth\":6,\"max_simulations\":96,\"adaptive\":true}"
```

Use this endpoint for the frontend-style engine move.

## Frontend Demo

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the Vite URL, usually:

```text
http://localhost:5173
```

The frontend sends the current FEN to `/fastmove` when the AI needs to move.

## CLI Play Demo

You can play in the terminal without the frontend:

```powershell
cd backend
$env:PYTHONPATH="."
python -m app.cli.play --config config/default.yaml --device cpu
```

The CLI starts a game where the user plays White and enters UCI moves such as:

```text
e2e4
```

The engine replies with a move selected by MCTS.

## Self-Play Demo

Generate a tiny self-play batch:

```powershell
cd backend
$env:PYTHONPATH="."
python -m app.cli.selfplay --config config/default.yaml --workers 1 --games-per-worker 1 --device cpu
```

Self-play stores positions with MCTS-improved policy targets and final game values.

## External Training Demo

Run external supervised training:

```powershell
cd backend
$env:PYTHONPATH="."
python -m app.cli.train_external --config config/external_training.yaml
```

For a small smoke run:

```powershell
python -m app.cli.train_external --config config/external_training.yaml --iterations 1 --max-val-samples 2 --max-train-samples 4 --train-steps 1 --no-save
```

This verifies the external data path without writing checkpoints.
