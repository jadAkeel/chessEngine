# Search strength repair — 2026-09-22

مراجعة كاملة لكل ما يؤثّر على قوّة اللعب في البحث (MCTS)، ونظام العقوبات (penalties)، ومسار النقلة السريعة (fast move) وبوت Lichess — **بدون أي تغيير في أوزان الشبكة العصبية** (`models/best_model.pth` لم يُمَسّ). النتيجة المثبتة: البحث نفسه كان معطوباً (توزيع الزيارات على نقلات الجذر شبه منتظم) وبطيئاً جداً (6–11 محاكاة/ثانية)، وبعد الإصلاح صار البحث يركّز فعلاً على أفضل النقلات ويصل لعمق 8 أنصاف نقلات، وربح مباراة مقارنة مباشرة بنفس الأوزان ونفس عدد المحاكاة.

Same checkpoint, same config values except the new keys listed below. Everything here is reproducible from `backend/` with `$env:PYTHONPATH = "."`.

## 1. What was actually wrong

### 1.1 Virtual loss flattened the root (the dominant bug)

`_select_child` scored children as `Q + U - virtual_loss * pending_visits` with `virtual_loss = 1.0` and values in `[-1, 1]`. One in-flight rollout therefore subtracted a full point from a child's score — larger than any possible Q difference — so inside each inference batch of 24 rollouts the search was *forced* to visit 24 distinct root children. Over a 128-simulation search that made the root visit distribution almost uniform, and the "best move" was decided by penalty tie-breaks rather than by search.

Measured with the production checkpoint, 128 simulations (`probe_root` in the session scratchpad; the numbers are reproducible with `scripts/diagnose_engine.py --simulations 128`):

| Position | Before: top-1 visit share | Before: root entropy / max | After: top-1 share | After: entropy / max | Leaf depth mean / max (before → after) |
| --- | ---: | ---: | ---: | ---: | --- |
| Start position (20 legal) | 8.6 % | 2.94 / 3.00 | 24.2 % | 2.37 / 3.00 | 1.9 / 3 → 2.6 / 6 |
| `live_19` (42 legal) | 4.7 % (top five: 6 visits each) | 3.58 / 3.74 | 32.8 % | 1.66 / 3.74 | 2.2 / 5 → 3.4 / 8 |
| Quiet middlegame (41 legal) | 5.5 % | 3.44 / 3.71 | 21.1 % | 2.34 / 3.71 | 2.2 / 5 → 3.3 / 8 |
| Rook endgame (17 legal) | 14.8 % | 2.72 / 2.83 | 89.1 % | 0.59 / 2.83 | 2.4 / 5 → 3.6 / 7 |

Fix: pending visits are now treated as losses *diluted by the child's real visits*, `Q_eff = (W − vl·n_pending) / (N + n_pending)`, which is the standard formulation. Regression: `test_virtual_loss_is_diluted_by_real_visits_instead_of_flattening_the_root`.

### 1.2 Unvisited children looked "neutral" (no first-play urgency)

An unvisited child had `Q = 0`. In a losing position (parent Q ≈ −0.6) every untried move looked better than the moves already known to be best, so the search sprayed visits over all children. Unvisited children now inherit the parent's value minus `mcts.fpu_reduction` (default 0.25). Regression: `test_unvisited_children_inherit_parent_value_minus_fpu_reduction`.

### 1.3 Torch threads: 8 threads on a hybrid CPU

The development machine is an i5-1235U (2 P-cores + 8 E-cores). The auto thread plan picked 8 intra-op threads. Measured single-board inference: **8 threads = 3.6 s, 4 threads = 79 ms**; batch of 24: 8 threads 0.5–7.9 s (unstable), 4 threads 0.73 s; 12 threads = 37 s. Every search starts with a single-board root expansion and `/fastmove` and the bot call `model.predict` on one board, so each move paid seconds before any search began. `system.cpu_threads: 4` is now set in `config/default.yaml`, and the engine/API role cap in `app/infra/runtime.py` is 4.

### 1.4 Heuristic overhead per selection step

Every selection step recomputed every child's penalty: `board.fen()`, `tuple(move_stack)`, a push/pop, repetition key, and the tactical exchange scan — even on cache hits. Profile at 48 simulations: `_select_child` ≈ 4.3 s. In addition `is_game_over(claim_draw=True)` was evaluated at every node of every rollout; python-chess implements the repetition checks by replaying the whole move stack.

Fixes:

- Edge penalties are computed **once per expanded node** and cached on the child (`Node.penalty`, `Node.penalty_components`); they are static for a node's path from the game start, so this is exact.
- Terminal detection uses the incremental `seen_positions` map the search already maintained (`MCTS._terminal_state`).
- The mate-in-one scan inside the principle heuristics uses `gives_check` as a prefilter and only runs for root children; principle heuristics as a whole are applied down to `principle_penalties.max_tree_depth` (default 2) — tactical hanging-piece penalties still apply at every depth.
- Exchange helpers use `generate_legal_captures(to_mask=…)` instead of scanning all legal moves.

Measured after the changes (64 simulations, 4 threads): inference 65–75 % of the time, penalties 25–30 %, terminal checks ≈ 0. Throughput went from 6–11 to roughly 25–60 simulations/second depending on the laptop's power state.

### 1.5 Terminal semantics were wrong for "could repeat"

`is_game_over(claim_draw=True)` is true when the side to move *could* claim a draw by making a repeating move, not only when the position has repeated three times. The search treated such positions as terminal draws worth 0 — including winning positions where a repetition merely happened to be available. Only an actual third occurrence (or the fifty-move clock) is terminal now. Regression: `test_only_an_actual_third_occurrence_is_a_terminal_draw`.

### 1.6 Value noise that alternated sign with depth

- The stagnation ("progress") penalty was subtracted from the leaf value *for the side to move at the leaf*. The halfmove clock is a property of the position, so this charged alternating sides at alternating depths (up to ±0.20). The value is now shrunk toward 0 (drawish) for both sides.
- The classical mobility term counted only the side to move's legal moves, handing every leaf ≈ +0.04 for whoever was to move there. It is now own-minus-opponent mobility.

### 1.7 Dead and over-eager penalties

- The oscillation penalty compared the candidate with `move_stack[-1]` — always the *opponent's* last move — so it could never fire. It now looks at the mover's own previous move, skips captures and checks, and is weighted 0.25× the repetition penalty.
- The root anti-repetition filter (`repetition_move_weight`, designed for self-play diversity) also ran in play, which stops a losing engine from holding a draw by repetition. In play mode it is skipped when the root value is below −0.05; self-play (noisy) searches keep it unconditional.

## 2. Tree reuse and a cumulative ladder

`MCTS` now retains its tree between searches (`mcts.reuse_tree`, default on). A search on the same position continues the tree; a search after the played move and the reply starts from that subtree; anything else starts fresh (verified by move history *and* FEN). `Engine` already kept one `MCTS`, and the API now keeps one per loaded model (`_get_mcts`, serialized by a lock), so:

- the `/fastmove` ladder 30 → 64 → 96 costs 96 rollouts instead of 190 (`_best_move_with_mcts(..., cumulative=True)` runs only the missing visits);
- consecutive bot moves in a game start from the visits already spent on the opponent's actual reply.

Regressions: `test_same_position_search_continues_the_retained_tree`, `test_tree_is_reused_after_the_played_moves_and_dropped_otherwise`, `test_api_search_is_cumulative_across_ladder_rungs`.

## 3. Lichess bot: spend the clock

The bot returned the raw policy move without any search whenever the fast-move complexity heuristics rated the position "not complex" — with two minutes on the clock. The search also received no deadline; only the ladder steps were time-checked.

Now (`app/cli/lichess_bot.py`):

- `calculate_dynamic_thinking` spreads the clock over `max(20, 50 − fullmove)` moves plus 80 % of the increment, caps a move at 25 % of the remaining clock, and returns `max_sims` as a cap whenever the clock is above 30 s — the search deadline governs the actual number of rollouts.
- `_compute_move_sync` always searches when it has ≥ 0.35 s; the policy move is played only when the clock is nearly gone (or the move is forced). Each rung passes `time_limit_sec` into MCTS and is cumulative; a 30-visit agreement is not accepted as "confident" while clock remains (it must reach the 64 rung). A "decisive" capture still gets a cheap 30-visit sanity check.

Regressions: `test_bot_spends_its_clock_on_quiet_positions`, `test_bot_plays_policy_move_only_when_the_clock_is_nearly_gone`, `test_clock_allocation_spreads_time_and_lets_the_deadline_govern_simulations`. The existing critical-position test (`128` sims in one rung) still holds.

## 4. Paired match, same weights, 64 simulations per move

`scripts/play_revision_match.py --before <snapshot> --before-config <snapshot>/config/default_t4.yaml --simulations 64 --games 6` (colours alternate; the snapshot's config only had `cpu_threads: 4` added so both sides ran on the same thread setting).

| Game | White | Black | Result | Termination | Plies | Time before / after |
| ---: | --- | --- | :---: | --- | ---: | --- |
| 1 | after | before | 1-0 | checkmate | 51 | 35 s / 25 s |
| 2 | before | after | 0-1 | checkmate | 130 | 103 s / 70 s |
| 3 | after | before | ½-½ | insufficient material | 98 | 77 s / 53 s |
| 4 | before | after | 0-1 | checkmate | 134 | 138 s / 88 s |
| 5 | after | before | ½-½ | threefold repetition | 112 | 112 s / 68 s |
| 6 | before | after | ½-½ | stalemate | 126 | 130 s / 84 s |

**New search: 3 wins, 3 draws, 0 losses (4.5 / 6)** — both wins with Black were by checkmate; the old search never won a game. The new side also used ~35 % less wall time at the same simulation count. PGNs and `results.json`: `docs/diagnostics/revision_match_2026-09-22/`.

Six games is a smoke-level sample, not an Elo measurement; the qualitative change (peaked root distributions, depth 6–8) is the stronger evidence. The match ran with the code state before the small oscillation-penalty and clock-confidence tweaks of the same day.

## 5. What did not help

- Folding BatchNorm into the convolutions for inference is numerically exact (max |Δlogit| 8e-5) but gave **no** speedup on CPU — oneDNN already fuses it — so it was not added.

## 6. New configuration keys

```yaml
mcts:
  inference_batch_size: 16   # was 24; smaller batches give short searches more sequential decisions
  fpu_reduction: 0.25        # first-play urgency reduction from the parent's value
  reuse_tree: true           # keep the subtree of the played moves between searches
principle_penalties:
  max_tree_depth: 2          # principle heuristics bias selection at the root and the next two plies only
system:
  cpu_threads: 4             # hybrid P/E-core CPUs collapse above this
```

## 7. Verify

```powershell
cd backend
$env:PYTHONPATH = "."
python -m pytest -q ..\backend\tests                       # 221 tests
python -m scripts.diagnose_engine --simulations 128        # root visits, depth, time per position
python scripts/play_revision_match.py --before <old-tree> --before-config <old-yaml> --simulations 64 --games 6
```

## 8. Still open (not code)

- The network's own strength: policy top-1 matches the human label on ~41 % of sampled positions and the value head is noisy (see the 2026-09-05 audit). Search now extracts what the network knows; a stronger checkpoint remains the main lever and is out of scope for this pass.
- Short-mate proof is still mate-in-one / checking mate-in-two; the root exchange screen is a 2-ply material estimate.
- No online games were played; an Elo change on Lichess is unmeasured until the bot runs with this code.
