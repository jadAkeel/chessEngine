# ♟️ Chess Engine (Scalable ML Pipeline)

## Overview

This project is a chess engine built around a full machine learning pipeline.
The focus was not just training a model, but designing a system that can scale to large datasets without running into memory limitations.

The system takes raw chess games and turns them into structured training data, then trains a neural network to evaluate positions and predict moves.

---

## Data Pipeline (Sharding + Metadata)

The dataset is built from PGN chess games and converted into three main components:

* **State** → encoded board (tensor)
* **Policy** → move index
* **Value** → game outcome

Instead of storing everything in a single large file, the dataset is split into **multiple shard files**:

```text id="d1a9e2"
shard_0.npz
shard_1.npz
shard_2.npz
...
```

Each shard contains:

* states
* policy indices
* values
* metadata (input planes, policy size)

### Why sharding?

* Avoid loading huge datasets into RAM
* Enable streaming during training
* Allow scaling to millions of samples

During training, shards are loaded one by one and processed incrementally.

---

## Neural Network

The model is implemented in PyTorch and follows a **dual-head architecture**:

* **Policy head** → predicts the next move
* **Value head** → evaluates the position

Input:

* Encoded chess board (multi-channel tensor)

Output:

* Move probabilities over all possible moves
* Scalar evaluation of the position

This structure is inspired by modern chess engines that combine move prediction and position evaluation.

---

## Training Pipeline

The training system is built to work with large datasets efficiently:

1. Data is streamed from shard files
2. Samples are pushed into a **Replay Buffer**
3. The model is trained on mini-batches
4. Loss is computed:

   * Policy loss (classification)
   * Value loss (regression)
5. Model checkpoints are saved (latest + best)

Validation is done on a separate subset of data to track performance.

---

## Data Quality Handling

To ensure training stability:

* Duplicate samples are removed using hashing
* Invalid board states are filtered
* Illegal or out-of-range moves are discarded

This prevents noisy data from degrading the model.

---

## Testing & Evaluation

The model is evaluated using:

* Validation loss on unseen samples
* Comparison between training iterations
* (Optional) model vs model matches

This helps verify that training is actually improving performance.

---

## Project Structure

```text id="ab83c1"
backend/
 ├── training/
 │    ├── train_external.py
 │    ├── external_samples.py
 │    ├── replay_buffer.py
 │    └── trainer.py
 │
 ├── model/
 │    └── network.py
 │
 ├── game/
 │    ├── board_encoding.py
 │    └── move_encoding.py
 │
 └── scripts/
      └── import_lichess_samples.py
```

---

## Example Workflow

```text id="7c1f0d"
PGN games
   ↓
Data extraction
   ↓
Shard generation (+ metadata)
   ↓
Streaming loader
   ↓
Replay buffer
   ↓
Neural network training
   ↓
Validation / evaluation
```

---

## Tech Stack

* Python
* PyTorch
* NumPy
* python-chess

---

## What I focused on

* Building a scalable data pipeline
* Handling large datasets efficiently
* Designing a clean training architecture
* Integrating data processing with model training

---

## Future Improvements

* Self-play training loop
* MCTS integration
* Better evaluation (Elo rating)
* Training optimizations
