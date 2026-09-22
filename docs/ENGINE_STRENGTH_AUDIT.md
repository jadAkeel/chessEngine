# Engine strength investigation — 2026-09-05

> Follow-up: [Search strength repair — 2026-09-22](SEARCH_STRENGTH_2026-09-22.md) (virtual loss, FPU, thread plan, penalty caching, tree reuse, clock management).

تمّت إعادة إنتاج ثلاث نقلات خاطئة من مباراة فعلية بنفس الموديل، ثم إصلاح أسباب مباشرة في البحث وحماية القطع وإدارة الوقت. التحسّن المثبت هنا تكتيكي وتشغيلي؛ لم يُقَس ارتفاع Elo بعد، ولم تُعدَّل الأوزان أو ملفات الداتا الأصلية.

## Scope and evidence

- Python/PyTorch residual policy/value network, MCTS, FastAPI/React UI, and a separate Lichess CLI.
- `models/best_model.pth`: 11,757,287 parameters, 24 residual blocks, 160 channels. All state keys matched the current network in the initial audit. No missing weights were found in this checkpoint.
- Local match history, excluding the two demo records: 20 games, 6 wins, 13 losses, 1 draw; last locally recorded Blitz rating 1337. This is not an independent Elo estimate.
- Actual public game examined: [chessEngineboot–maia5, TNXeAh13](https://lichess.org/TNXeAh13). The bot was mated with 2:09 remaining. Running out of clock was **not** the cause of that loss.
- A reproducible sample of 256 positions from `shard_0.npz` had 256 legal move labels and zero discrepancies after decoding and re-encoding with the current encoder. Policy top-1 matched 106/256 labels. This was an existing-data sanity check, **not held-out strength evaluation**.
- Checkpoint value outputs varied (standard deviation approximately 0.461 across that sample); the value head was not a constant-output model. Its MSE against final game outcomes was approximately 0.811. These noisy targets do not measure tactical accuracy.
- SHA-256 inspection of all 752 external shards found 702 unique files and 50 redundant exact copies. No original shard was removed.

## Confirmed causes and repairs

### 1. Ignoring an attacked piece when moving a different piece

At move 19 of the public game, `Rae1` left `Qh3` to `...Bxh3`. The old tactical component returned **0.0** for the move because it only examined the destination of the rook. At 128 simulations, the old checkpoint/search combination reproduced `Rae1`.

`MCTS._move_penalty_components` now considers legally capturable friendly pieces throughout the resulting position, accounting for the candidate's material gain and legal recaptures. The existing penalty scale remains in use; penalties are not summed across mutually exclusive opponent captures. Target squares are part of the tactical cache key.

Root selection also screens quiet moves that drop at least 300 centipawns to an immediate capture/recapture sequence when a safer alternative is available. This prevents the reproduced move-13 castle from abandoning `Nd2` to `...Qxd2`. This material screen is deliberately limited: checking sacrifices are left to search, and deeper or positional sacrifices can still be underestimated.

### 2. Root value sign was wrong when visit counts tied

Child values describe the **opponent's** perspective. Tree selection correctly used `-child.q`, but root tie-breaking sorted by `+child.q`. With equal visits/probability, it could prefer the worse result for the mover. The regression test failed before the sign was corrected. Shallow batched searches produce many such ties.

### 3. A known short-mate guard was absent from the actual bot path

The API had a short-mate check, while Lichess used `Engine.analyze` directly. An existing reported position reproduced `...Rxd5`, allowing `Qh6+ Kg8 Qh8#`, at both 32 and 256 simulations in the initial audit.

Shared search now:

- returns an available mate in one without neural inference;
- checks candidate moves for an opponent's mate in one or checking mate in two;
- preserves legal draw outcomes and board history during tactical probes;
- rejects a proved short mate when an alternative is available;
- handles terminal/claimable-draw roots without producing a move or evaluating the network.

This is a bounded tactical screen, **not a complete mate solver**. Quiet first moves in a mating combination and deeper tactics remain outside its proof horizon.

### 4. Even trades were classified as hanging pieces

For the legal sequence `d3 cxd3 exd3` in the regression fixture, net material change is zero. The old code nevertheless assigned a hanging-pawn penalty because zero did not reach a positive sacrifice-compensation threshold. Non-losing exchanges are now exempt; genuinely losing exchanges retain their penalties.

### 5. Repeated heuristic work consumed the search budget

The initial cProfile probe of 32 opening simulations spent about 1.158 of 1.462 cumulative seconds inside move-penalty computation, versus about 0.205 seconds in batched inference. Disabling principle penalties alone sped up some positions but still reproduced the short-mate blunder: performance overhead and tactical correctness were distinct problems.

Principle results are now cached per search using the full position and move history. Repetition penalties remain dynamic and are explicitly tested with different repetition maps. Caches clear at each search. The engine's result cache also distinguishes the same FEN with different histories.

### 6. Timeout discarded completed search and could leave work running

The old Lichess code cancelled an executor Future after a timeout and then ran policy inference. Cancelling that Future does not stop its worker; it could discard completed search, overlap use of the engine, and run fallback inference on the event loop.

`Engine.analyze`/`MCTS.search` accept an optional `time_limit_sec`. The bot uses 90% of its clock allocation, and search returns completed visits when its cooperative deadline is reached. It reserves a small part of the budget for root safety checks. Budgeted results do not enter the full-search result cache. The bot retains the engine lock until an executor worker actually finishes, including repeated task cancellation. Emergency inference remains off the event loop.

This is a cooperative budget. One in-flight inference batch or operation can overrun it; there is no hard real-time guarantee. Actual online timeout frequency could not be established from the available log.

### 7. Checkpoint metadata replaced explicit runtime configuration

The saved checkpoint contained `num_simulations=64` and old training/runtime settings. Loading it replaced the explicitly supplied configuration. Explicit engine configuration now takes precedence; the test checks search settings after checkpoint load. Lichess's explicit per-move simulation argument already overrode the saved simulation count, so **this alone did not explain its weak Elo**.

## Training integrity repairs

The following defects are confirmed in the current code. Their individual contribution to the existing checkpoint's strength cannot be quantified without a controlled retraining run.

- **Validation leakage:** train and validation independently sampled the same source. The external trainer now partitions by a deterministic hash of position features, independent of labels, clocks, shard ordering, and horizontal mirror equivalents. A given position group cannot enter both partitions. Existing shards lack game IDs, so related positions from the same game can still cross partitions; this is a position split, not a game split.
- **Duplicate weighting:** exact duplicate shard files and duplicate samples across shards are skipped during loading when deduplication is enabled. This does not rewrite the dataset. Global sample-hash storage scales with the number of accepted samples; account for that memory when increasing buffer limits.
- **Invalid castling augmentation:** horizontal reflection moves the king's start file from e to d. Positions with castling rights are no longer horizontally augmented. Legal no-rights augmentation remains enabled.
- **Optimizer reset:** model/optimizer/scheduler were recreated every iteration and the global step reset; even `--no-save` could lose progress between iterations. The trainer now retains them between iterations and saves/restores optimizer, scheduler, scaler, and cumulative steps. Old checkpoints without optimizer state cannot recover state that was never saved.
- **Unstable validation comparisons:** validation content is reproducible, fingerprinted together with loss coefficients, and an old minimum from a different validation set/protocol no longer blocks selecting a new best. Previous history entries are retained, with a comparison start index.
- **Resume sampling:** train shuffle changes with cumulative optimization steps, including resumed runs; validation ordering stays fixed.
- Empty train/validation sets now fail rather than producing a misleading zero validation loss.

The importer still imitates PGN moves and labels positions with the final outcome; it does not validate move quality with a stronger engine or filter player Elo. The external trainer still selects a **validation-loss winner**, not an Elo-proven champion. Stronger targets and a fixed opponent/puzzle evaluation gate remain the next training work, rather than blindly extending the old training run.

## Reproduction and validation

The final automated run passed **146 tests in 9.13 seconds**; `compileall` also passed. The initial three regression tests (equal exchange, repeated heuristic evaluation, and terminal-root inference) failed before the fixes. Separate root-value-sign and abandoned-queen tests also demonstrated failures before their repairs.

Same-checkpoint, 128-simulation results (single local CPU runs; timings vary with load):

| Position | Before | After | Before time | After time |
| --- | --- | --- | ---: | ---: |
| Opening | d4 | d4 | 3.30 s | 1.66 s |
| Reported mate in two | Rxd5, allows forced mate | Rh5, avoids that short mate | 4.73 s | 2.11 s |
| Actual move 13 | O-O, abandons Nd2 | Qh5, does not abandon Nd2 | 7.91 s | 5.41 s |
| Actual move 19 | Rae1, abandons Qh3 | Qf3 | 8.18 s | 4.92 s |
| Actual move 25 | Rc7, allows Qxg2# | g3, avoids immediate mate | 4.74 s | 2.43 s |

The existing API checks flagged four dangerous choices in the seven-position baseline and none in the repaired choices. These checks are limited tactical heuristics, not an independent strong-engine analysis. In particular, avoiding the final mate does **not** establish that an already losing position can be saved.

With a two-second search budget and 256 requested simulations, the six nonterminal/non-mate-in-one probes completed in approximately **1.76–1.94 s**, returning 55–168 completed visits. The mate-in-one probe returned immediately. See the JSON artifacts in `docs/diagnostics/` for checkpoint hashes, settings, and individual results.

From `backend/` in PowerShell:

```powershell
$env:PYTHONPATH = "."
python -m pytest -q
python -m scripts.diagnose_engine --simulations 128 --output ../docs/diagnostics/search_after_verified.json
python -m scripts.diagnose_engine --simulations 256 --time-limit 2 --output ../docs/diagnostics/search_budget_2s.json
```

`scripts/diagnose_engine.py` records checkpoint SHA-256, device, thread count, move, completed visits, evaluated leaf depth, time, and independent checks using the existing API safety helpers. It supports `--variants default no_principles batch_one`. These are diagnostic positions, not an Elo benchmark.

Baseline measurements were taken from the pre-edit source snapshot with the same checkpoint and configuration. The snapshot is at `C:/Users/10User/AppData/Local/Temp/chess-engine-audit-20260905-013151`; it contains source/tests/config/scripts, not copied model/data artifacts. The repository has no Git history in this workspace.

Automated coverage includes losing versus equal exchanges, an abandoned queen, a castled-away knight defender, root value sign, mate selection/avoidance, terminal draws, board restoration, repeated cancellation/engine ownership, deadline results, configuration precedence, history-aware cache keys, disjoint/reproducible dataset partitions, duplicate shards, castling augmentation, validation fingerprint changes, and a real tiny two-iteration training run followed by resume with optimizer steps 1 → 2 → 3.

No production checkpoint was retrained or replaced. No online move, challenge, or game was submitted during this investigation. Changes take effect in a bot process after that process reloads the code. General playing strength, long tactical combinations, and an Elo gain remain unverified until a sufficiently large controlled match evaluation is run.

Continuation context and changed-file inventory: [handoff](ENGINE_STRENGTH_HANDOFF.md).
