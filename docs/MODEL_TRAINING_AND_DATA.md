# Model Training And Data

The model is trained to imitate strong move choices and evaluate chess positions. It follows an AlphaZero-style shape: a shared residual network produces a Policy Head and a Value Head.

## Model Inputs And Outputs

Each chess position is encoded as a tensor:

```text
20 x 8 x 8
```

The 20 planes contain piece placement and game-state features such as side to move, castling rights, en passant, halfmove clock, and fullmove number.

The network outputs:

- Policy Head: logits over `4672` possible move indexes.
- Value Head: a scalar in `[-1, 1]` estimating the position from the side-to-move perspective.

The policy output is not used blindly. At inference time, the engine filters the policy through legal moves for the current board.

## Current Model Size

Current configuration:

```yaml
model:
  input_planes: 20
  channels: 160
  res_blocks: 24
  value_dropout: 0.05
```

This means:

- 20 input planes
- 160 trunk channels
- 24 residual blocks
- dropout in the value head
- `11,757,287` trainable parameters

The parameters are distributed across the input convolution, residual tower, Policy Head, Value Head, batch normalization, and layer normalization layers.

## Training Targets

For each training sample, the model learns from:

- a board state tensor
- a policy target
- a value target

The policy target tells the model which move or move distribution should be preferred. In self-play, this target comes from MCTS visit counts. In external supervised data, the target is loaded from the external sample files.

The value target represents the final outcome or evaluation target for the position. During self-play, it is assigned after the game finishes, from the perspective of the player who made the move.

## Loss And Weight Updates

Training combines:

- policy loss: makes predicted move probabilities closer to the target policy
- value loss: makes predicted value closer to the target result
- entropy term: encourages useful policy spread

The trainer uses PyTorch backpropagation:

1. Run a forward pass through `ChessNet`.
2. Compute policy and value losses.
3. Backpropagate gradients through the neural network.
4. Clip gradients.
5. Update model weights with `AdamW`.
6. Step the learning-rate scheduler when configured.

This is the point where the neural network actually changes.

## External Data Training

External training is a separate path from the normal self-play loop. It is configured by:

```powershell
backend/config/external_training.yaml
```

Run it from `backend/`:

```powershell
$env:PYTHONPATH="."
python -m app.cli.train_external --config config/external_training.yaml
```

The external loader expects `.npz` shard files containing:

- `states`: encoded board tensors with shape `(N, 20, 8, 8)`
- `policy_indices`: move indexes in the `4672`-move policy space
- `values`: value targets

Optional metadata can include:

- `input_planes`
- `policy_size`

The loader validates these against the current config.

## External Data Filtering

The external data path can:

- shuffle shard files
- shuffle samples inside shards
- limit the maximum number of samples
- validate policy indexes
- reject non-finite value targets
- reject states containing `NaN` or `Inf`
- drop all-zero states when configured
- deduplicate samples
- optionally filter by decoded fullmove number

The current deduplication key is based on:

```text
state + move_index + value
```

Accepted samples are converted into sparse policy targets and streamed into a replay buffer for training.

## External Checkpoints

External training writes separate checkpoint names by default:

```text
models/external/external_latest_checkpoint.pth
models/external/external_best_model.pth
models/external/external_history.json
```

The latest checkpoint tracks the newest trained model. The best checkpoint is updated when validation loss improves.

## Self-Play Training

The standard training loop is AlphaZero-style:

```text
current model
  -> generate self-play games
  -> run MCTS for each position
  -> store state, MCTS policy target, and final game value
  -> train from replay buffer
  -> evaluate candidate against champion
  -> save latest and best checkpoints
```

Run self-play data generation:

```powershell
cd backend
$env:PYTHONPATH="."
python -m app.cli.selfplay --config config/default.yaml --workers 1 --games-per-worker 1 --device cpu
```

Run the full training loop:

```powershell
cd backend
$env:PYTHONPATH="."
python -m app.cli.train --config config/default.yaml --iterations 5 --device cpu
```

## Replay Buffer

Self-play samples are saved into the replay buffer. Each sample contains:

- encoded board state
- sparse policy target from MCTS visit distribution
- final value target

The replay buffer can be saved in shards and reused when training resumes. The training loop can also prefill the replay buffer from external samples before generating new self-play data.

## Arena Acceptance

After training steps run, the loop can evaluate a candidate model against the current best model in arena games. If the candidate reaches the configured acceptance threshold, it becomes the new best model.

This keeps training from automatically replacing the champion with every newly optimized checkpoint. The model has to prove that it improved in match play.

## Kaggle And Colab Training

Kaggle and Colab use the same backend training code but different runtime setup. The repo includes:

- `backend/KAGGLE_TRAINING.md`
- `backend/requirements2_kaggle.txt`
- `backend/scripts/kaggle_train_external.py`

The Kaggle runner can auto-discover external samples, copy checkpoint datasets, run external training iterations, and update a checkpoint dataset after each iteration.

## External Data Vs Self-Play

External data and self-play serve different purposes:

- External data gives the model a supervised starting point from existing positions and move labels.
- Self-play lets the model generate new training targets from its own MCTS-improved games.

The strongest pipeline uses both:

```text
external supervised data
  -> stronger initial model
  -> better self-play games
  -> better MCTS targets
  -> better future model checkpoints
```
