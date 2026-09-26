# Engine strength handoff

> **Update 2026-09-23:** second pass and corrections to the 09-22 evidence — see [SEARCH_STRENGTH_2026-09-23.md](SEARCH_STRENGTH_2026-09-23.md).
>
> **Update 2026-09-22:** the search itself was found to be flattening root visits (virtual loss), running 8 torch threads on a hybrid CPU, and the Lichess bot was not spending its clock. See [SEARCH_STRENGTH_2026-09-22.md](SEARCH_STRENGTH_2026-09-22.md) for evidence, fixes, paired-match results and the new config keys. The notes below describe the 2026-09-05 state.

## 1. Repository context

Root: `C:/Users/10User/Desktop/ai/chesEngineWithData`. Python/PyTorch chess engine, MCTS, FastAPI, React/Vite, and Lichess bot CLI. No Git repository exists in this workspace. Follow `AGENTS.md`; use `apply_patch`, preserve model/data artifacts, and avoid unrelated changes. Delegation requires an explicit request.

## 2. Current goal

The user asked in Arabic for deeper investigation of weak Elo and to start repairing actual causes. This pass reproduced real-game mistakes and applied tested search/runtime/training-integrity repairs. An Elo improvement is not yet measured.

## 3. Completed work

See [the full audit](ENGINE_STRENGTH_AUDIT.md) for evidence, explanations, timing tables, limitations, and reproduction commands.

Changed runtime files:

- `backend/app/mcts/search.py`: root tactical screening, even-exchange fix, all-piece capture penalties, root value sign, deadline, terminal handling, per-search static heuristic caches.
- `backend/app/game/tactics.py` (new): bounded checking-mate detection, legal capture/recapture loss estimate, ranked safety selection.
- `backend/app/core/engine.py`: optional time budget, explicit config precedence, history-aware cache, no caching partial-budget results.
- `backend/app/cli/lichess_bot.py`: cooperative search budget; off-event-loop fallback; retains engine ownership during cancellation until actual worker completion.
- `backend/app/training/external_samples.py`: deterministic position split; mirrored/counter-independent position grouping; duplicate files/samples skipped across the stream; reproducible shuffling; supports individual NPZ files.
- `backend/app/training/train_external.py`: separate partitions; optimizer/step continuity and resume; validation fingerprint/comparison boundary; empty-data failure.
- `backend/app/training/trainer.py`: no horizontal augmentation while castling rights remain.

Tests: new `test_search_regressions.py`, new `test_training_integrity.py`, updated `test_lichess_bot.py`. Diagnostics: new `backend/scripts/diagnose_engine.py`. Docs: audit, handoff, and `backend/EXTERNAL_TRAINING.md`.

## 4. Repository state

No commit/branch was created. No dependency or deployment configuration changed. Production weights/data were not modified; final checkpoint SHA-256 matched baseline. There was no running `app.cli.lichess_bot` Python process found in the final process check. No bot/game was launched.

Pre-edit source/test/script/config snapshot: `C:/Users/10User/AppData/Local/Temp/chess-engine-audit-20260905-013151`. Do not restore it over later user changes. It does not contain model/data copies.

Generated evidence is in `docs/diagnostics/`: verified before/after JSON, earlier principle ablation, two-second-budget run, and actual-game FENs with their public source URL.

## 5. Commands run

From `backend/` with `$env:PYTHONPATH = "."`:

| Command | Result |
| --- | --- |
| `python -m pytest -q` | 146 passed in 9.13 s |
| `python -m compileall -q app scripts/diagnose_engine.py` | Passed |
| `python -m scripts.diagnose_engine --simulations 128 ...` | Verified before/after with same checkpoint |
| `python -m scripts.diagnose_engine --simulations 256 --time-limit 2 ...` | Completed partial searches within ~1.76–1.94 s in measured non-mate fixtures |

Regression-first failures were observed before fixes for even exchanges, repeated principle evaluation, terminal inference, root value sign, and ignoring an attacked queen. Isolated temporary-data training verified optimizer/global-step continuity across two iterations and a resumed third step.

## 6. Verified results

The original engine reproduced `O-O`, `Rae1`, and `Rc7` at moves 13, 19, and 25 of public game `TNXeAh13`. The repaired version at 128 simulations chose `Qh5`, `Qf3`, and `g3`, avoiding the identified immediate piece losses/mate. Independent existing API checks also cleared those repaired choices. The previous reported `...Rxd5` mate-in-two blunder changed to `...Rh5`. This does not prove the resulting positions are winning or drawable.

752 dataset shards contain 50 exact redundant copies. A 256-position sanity sample had no illegal labels or encoding mismatches. Existing weights are not shown to be untrained or structurally broken.

## 7. Remaining risks

- Short-mate proof only covers mate in one/checking mate in two. Root material screen is a short-horizon heuristic; deeper/positional sacrifices can be undervalued.
- Deadline is cooperative: an in-flight batch/operation can overrun. It is not a hard real-time limit.
- Dataset split is by position, not by game: shards have no game IDs. Global dedup hash memory grows with accepted samples.
- Old external checkpoints lacked optimizer state; previously discarded state cannot be restored.
- Training data still imitates PGN moves/final outcomes, without player-quality filtering or engine labels. Validation-loss selection is not a strength gate.
- The existing general checkpoint helper still uses `strict=False`; this audit verified all keys of the actual production checkpoint, but did not redesign that helper.
- No broader match/Elo test or production retraining was performed. No tests were skipped.

## 8. Next work

1. Review current files and rerun targeted checks if changed. Preserve the confirmed regressions.
2. Build/run a fixed opponent and held-out tactical evaluation, alternating colours and matching hardware/time budgets. Do not advertise an Elo gain from seven diagnostic positions or a handful of games.
3. Improve data provenance (game IDs/player ratings), then evaluate target quality and plan a controlled retraining experiment using the repaired partition/resume path.
4. Evaluate batch size/search allocation and critic quality against strength results, not just speed or training loss.

## 9. Constraints

Keep original model/data files intact. Do not start online matches/challenges or a long production retraining run as an incidental validation step. The user prefers autonomous progress on authorized fixes; do not ask permission again for ordinary local tests or repairs.

## 10. Next-session prompt

```text
Continue the engine-strength work from docs/ENGINE_STRENGTH_HANDOFF.md and docs/ENGINE_STRENGTH_AUDIT.md. Verify the current files first. The user wants the actual reasons for weak Elo treated, not cosmetic refactoring or an unsupported Elo promise. Preserve the successful regressions and existing checkpoint/data. Next establish a controlled strength benchmark, then assess training target quality and retraining needs. Report what is proven versus still unmeasured.
```
