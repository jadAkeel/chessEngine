# Render bot search diagnosis and recovery plan

## What the live logs showed

On 2026-09-28 between 18:25 and 18:39 UTC, the Render backend logged 92 `/fastmove` decisions. The current checkpoint loaded successfully on CPU. In 89 decisions, `root_visits=0`, `new=0`, and `retained=0`; two mate proofs used no search, and only one decision completed four new simulations. The default UI budget at depth 6 is two seconds. Search calls commonly took two to five seconds, and the first took about 44 seconds.

When a search has zero root visits, the engine ranks the root moves by policy priors and applies the tactical screen. This explains why play can weaken in a particular position without proving that the model overfit or that the checkpoint failed to load. The old logs lack a game identifier and FEN, so those 92 requests cannot be assigned reliably to the two reported completed matches.

## Read the new logs

Each `/fastmove decision` line now includes `game`, `ply`, `fen`, `mode`, `reason`, `root_visits`, `new`, `retained`, `budget_ms`, `root_eval_ms`, `presearch_ms`, `search_ms`, and `total_ms`. The UI sends a new `game_id` for each reset game. `mode=policy_fallback` is a warning. `reason=deadline_before_rollout` means pre-search work exhausted the budget; `no_completed_rollout` means pre-search ended within the budget but no simulation completed. `root_eval_ms` is zero when a root is reused or a mate proof bypasses evaluation.

## Recovery plan

Live probes after adding diagnostics confirmed the budget problem: three previously unseen positions at the two-second default completed zero visits; root evaluation took about 2.8 seconds after warmup. The first inference after startup took about 43 seconds because it prepared the frozen model. A fresh position with a 16-second budget completed eight visits, and another with a 30-second budget completed 16 visits in about 26 seconds. These positions differ, so the counts show feasibility rather than a controlled playing-strength comparison.

The first change is to warm CPU inference before the API reports ready and use a 30-second default for the web game's strongest level (6). Levels 1–5 have budgets of 3, 5, 7, 10, and 16 seconds. The API still accepts an explicit budget up to 30 seconds. This trades move latency for actual search on the free instance; it does not establish an Elo gain by itself.

1. Collect several complete games with the new logs. Group decisions by `game` and compare positions where play weakened. Track the share of non-mate decisions with `root_visits=0` and the median and 95th percentile of `root_eval_ms` and `total_ms`.
2. Benchmark representative opening, middlegame, and endgame positions on the deployed Render CPU. Measure model inference and search at different budgets and simulation caps. Tune the new budget using actual zero-visit rate and response latency. If the current model still cannot search enough on the free CPU, test a smaller or optimized checkpoint before changing hardware.
3. Replay the reported games from PGN if available and compare candidate moves with a reference engine at fixed search limits. Separate mistakes from model policy, tactical screening, and missing MCTS visits. Do not start another training run solely because of the observed weakness.
4. After selecting a search configuration, test it over a fixed position set and several full games. Compare zero-visit rate, move quality, latency, and outcomes against the current deployment before changing the live bot.
