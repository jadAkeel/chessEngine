# Search strength, second pass — 2026-09-23

مراجعة ثانية عدائية (adversarial) لإصلاحات 2026-09-22: هل كانت فعلاً أقوى طريقة؟ الجواب: لا. وجدنا أخطاء حقيقية بقيت في البحث ونظام العقوبات وشاشة الأمان التكتيكية وبوت Lichess ومسار `/fastmove`، وأيضاً أن نتيجة مباراة الأمس لم تكن دليلاً كافياً. كل ما يلي بنفس الأوزان (`models/best_model.pth` لم يُمَسّ).

Same checkpoint throughout. Reproduce from `backend/` with `$env:PYTHONPATH = "."`.

## 0. Corrections to the 2026-09-22 write-up

| Claim on 09-22 | What is actually true |
| --- | --- |
| "Won a paired match at equal simulations" (+3 =3 −0) | Not equal effort: tree reuse gave the new side ≈ 28 % more root visits per move. Not significant (p ≈ 0.05–0.09). Deterministic engines + 3 opening pairs = 6 games that are really 3 samples. |
| Game 6 draw | Stalemate of a won K+Q vs K — a search bug (terminal draws absorbed inside a batch), fixed below. |
| Game 3 draw | Unconverted winning ending (no mop-up signal from the net). |
| Game 5 draw | Harness artifact: `is_game_over(claim_draw=True)` is true when the side to move merely *can* claim, so the harness ended games that were not over. `play_revision_match.py` fixed; superseded by `scripts/engine_match.py`. |

## 1. Bugs found and fixed

### Search (`app/mcts/search.py`)

1. **The engine refused threefold draws even when losing.** The repetition penalty (up to 1.76 in value units) was applied in play, so a losing engine avoided a draw it should have taken. Repetitions are now scored as draws (Stockfish rule: a twofold inside the search path, or a threefold overall, is a draw), and the repetition/progress penalty components are dropped in play mode (`mcts.twofold_draw`).
2. **Stalemate / terminal draws absorbed inside a batch.** When a rollout reached a terminal node, its value was backed up immediately while other rollouts of the same batch were still pending, and the terminal was not known when its siblings were scored. Terminal children (mate, stalemate, repetition) are now marked at expansion, an unvisited child with a known terminal value is scored by its exact value, and terminal backups are deferred to the end of the batch.
3. **Duplicate backups on collisions.** Two rollouts of one batch reaching the same unexpanded leaf expanded it twice (12/64 duplicate expansions on a test tree). Collisions are now detected and counted (`collisions` in the result), and every reserved path's virtual visits are released in a `finally` (there was a leak on exceptions).
4. **Virtual loss** is now relative: `q -= vl·v/(N+v)`.
5. **Root move choice** uses a KataGo-style lower confidence bound (`value − lcb_scale/√N`, among moves with ≥ 15 % of the top move's visits), plus smart pruning (stop when the top move cannot be overtaken; requires the LCB top move to equal the visit top move).
6. **Batch size.** A fixed inference batch of 16 made small searches (64–128 sims) too flat. The batch now grows with the search: `clamp(visits // 16, 4, 16)` (`mcts.batch_growth`).
7. **Tree reuse without history.** The API sent only a FEN, so reuse never worked there. Reuse now also finds the new position by FEN up to 2 plies below the old root, with penalties reset for position-matched subtrees. Self-play never reuses.
8. **Won bare-king endings.** K+Q / K+R vs K get a graded value (king to the edge, kings close) instead of the network's flat ≈ 0.85, so the search has a direction (`mcts.mopup_endgames`). Measured effect in §2: faster K+Q mates, and all K+R test positions mated (one is a fifty-move draw without it).

### Penalty

9. **Tactical penalty was a step function.** It is now proportional: `blunder_penalty · min(1, net_loss / piece_value)`, and zero when the exchange loses ≤ 50 cp (even trades).
10. **Checking sacrifices were suppressed.** `queen_check_discount: 0.0` — checks are exempt from the tactical penalty (the search sees the follow-up).
11. `mcts.penalty_mode`: `offset` (default), `progressive` (`penalty / (1 + visits)`), `off`. Below the root nothing is computed with `off`.

### Tactical safety screen (`app/game/tactics.py`)

12. **Accepted any game-ending move.** `select_safe_move` now refuses draw-ending moves when the root value is > 0.15.
13. **Counted material that was already hanging** as lost by the candidate move. A null-move baseline is now subtracted.
14. **Overrode the search on weak evidence.** The screen only replaces the search's move with another the search itself rates at least 0.2 better with ≥ 8 visits, or on a material veto of ≥ 300 cp.
15. Root proofs: mate-in-1, and checking mate-in-2 (`find_forcing_mate_in_two`).

### Lichess bot (`app/cli/lichess_bot.py`)

16. **Lost on time when it could claim a threefold** (pre-existing): the turn loop treated "can claim" as "game over" and never moved. Fixed (`claim_draw=False` + actual threefold).
17. **The confidence ladder stopped complex positions at 64 visits**, and a `promotion_threat` fallback overrode the search. Replaced by one time-bounded search per move, then the screen.
18. Clock: moves-to-go horizon depends on the increment, 0.3 s latency reserve, 95 % of the budget to the search.

### `/fastmove` (`app/api/main.py`) and the web UI

19. The UI's engine move was a policy ladder capped at 96 sims with no history. It is now one time-bounded search (depth slider → 0.3–5 s, cap 400 root visits), and the request carries the game history (`moves`, `start_fen`), so the search sees repetitions and keeps its tree. `source` in the response: `mcts`, `mate_proof`, or `fast_policy` (with `adaptive: false`).
20. Frontend (`ChessHybridApp.jsx`, `ChessBoardPanel.jsx`) sends the history; the hard-coded `max_simulations: 96` is removed.

### Inference (`app/model/inference.py`)

21. Frozen TorchScript copy (trace → freeze → optimize_for_inference), rebuilt when any weight changes, checked against eager on first use and abandoned on any failure; `torch.inference_mode`. `system.fast_inference` (default true).

## 2. Measurements

### Proven tactical suite (27 positions from the engine's own games, forced answers verified)

| Variant | 64 sims | 128 sims |
| --- | ---: | ---: |
| 09-21 baseline | 16 | 19 |
| 09-22 (round 1) | 15 | 18 |
| New, fixed batch 16 | 20 | 20 |
| New, fixed batch 8 | 20 | 23 |
| New, fixed batch 4 | 22 | 23 |
| **New, batch growth (default)** | **23** | **23** |
| growth + penalties off | 23 | 23 |
| batch 4 + progressive penalty | 23 | — |
| batch 4 + penalties off | 23 | 23 |
| batch 4, LCB off (choose by visits) | 21 | 23 |

The penalty mode makes no difference on this suite; `offset` costs ≈ 18 % time per search (2.95 vs 2.41 s per 64 sims, interleaved). Kept as the default because it is what the net was tuned with; `off` is a reasonable faster alternative to test in games.

### Inference (production net, 4 threads)

| | eager | frozen |
| --- | ---: | ---: |
| batch 16, per board | 15.2 ms | 11.1 ms |
| single board | 28 ms | 21 ms |

Max output difference 8e-5.

### Clock simulation (bot)

No flag in 3+2 or 1+1 up to 120 moves. 5+0 flags only past move 107 if every move spends its full budget.

### Endgame conversion (real net, 64 sims, both sides the same engine)

Plies to mate from the start position; a game ends at mate, stalemate, threefold or the fifty-move rule.

| Start (strong side to move) | Mop-up on | Mop-up off |
| --- | ---: | ---: |
| K+Q `8/8/8/4k3/8/8/8/KQ6 w` | 23 | 41 |
| K+Q `8/8/3k4/8/8/2K5/8/6Q1 w` | 27 | 47 |
| K+Q `4k3/8/8/8/8/8/8/4K2Q w` | 19 | 19 |
| K+Q `8/8/8/3k4/8/8/8/Q3K3 w` | 21 | 21 |
| K+Q `8/2k5/8/8/8/8/5K2/7Q w` | 25 | 25 |
| K+Q `q6k/8/8/8/4K3/8/8/8 b` | 27 | 43 |
| K+R `8/8/8/4k3/8/8/8/KR6 w` | 83 | 59 |
| K+R `8/8/3k4/8/8/3K4/8/7R w` | 79 | 67 |
| K+R `8/8/8/8/3k4/8/8/R3K3 w` | 77 | 59 |
| K+R `8/8/2k5/8/8/8/5K2/7R w` | 39 | 69 |
| K+R `7R/8/8/4k3/8/8/1K6/8 w` | 39 | **fifty-move draw** |
| K+R `8/3k4/8/8/8/8/R7/6K1 w` | 45 | 81 |
| K+R `8/8/8/8/4k3/8/8/K6R w` | 73 | 55 |
| K+R `r6k/8/8/8/4K3/8/8/8 b` | 33 | 71 |

Mop-up is never slower in K+Q and mates in about half the plies in 3 of 6. In K+R it is mixed per position (slower in 4 of 8) but it mates in all 8, averaging 58.5 plies, while without it one K+R position is a fifty-move draw and the 7 mates average 66 plies. A first run on only the first two K+R positions suggested restricting mop-up to K+Q; the wider test shows that would lose the drawn position, so mop-up stays on for both. The net alone does convert most basic endings; the 09-22 game 3 non-conversion was a more complex ending.

### Match: new search (B) vs the 09-22 commit (A)

`scripts/engine_match.py`, 64 sims, `--budget topup` (a reused tree is topped up to the budget, not given it on top, so neither side gets extra visits), openings from the GM PGN, colours swapped per pair, resign adjudication only when both engines agree (|v| ≥ 0.92 for 3 moves).

Stopped after 17 of the planned 60 games, when master was fast-forwarded to this code (side A ran from the master checkout, so later games would no longer have been the 09-22 code):

| | Games | B wins | Draws | B losses | Mean root visits / move (A · B) |
| --- | ---: | ---: | ---: | ---: | --- |
| B (new) vs A (09-22) | 17 | 17 (14 mate, 3 adjudicated) | 0 | 0 | 62.9 · 49.2 |

All 8 complete opening pairs were won 2–0 (plus one game of a ninth pair); counting each pair as one coin flip, p ≈ 0.004. B wins while using fewer visits (smart pruning stops early). The losing side's value drops gradually in these games, with no single-move collapse, so this is being outplayed, not a crash or harness fault. The Elo point estimate is unbounded at a 100 % score and is not meaningful; what this shows is that the new search is clearly stronger at 64 sims, not by how much.

### Mutation check

The test suite kills all 7 search mutants tried: backup sign flipped, virtual loss disabled / flipped, reuse picks the wrong child, final move by prior, immediate terminal backup, no terminal marking.

## 3. New config keys (`config/default.yaml`, `MCTSConfig`)

| Key | Default | Meaning |
| --- | --- | --- |
| `mcts.penalty_mode` | `offset` | `offset` / `progressive` / `off` (quote `"off"` in YAML) |
| `mcts.twofold_draw` | true | repetition in the search path = draw |
| `mcts.smart_pruning` | true | stop when the best move is settled |
| `mcts.lcb_scale` | 1.0 | 0 = choose by visits |
| `mcts.batch_growth` | true | batch = `clamp(visits//16, 4, inference_batch_size)` |
| `mcts.mopup_endgames` | true | graded value for K+Q/K+R vs K |
| `system.fast_inference` | true | frozen TorchScript inference |
| `mcts.queen_check_discount` | 0.0 | checks exempt from the tactical penalty |

## 4. Not done / open

- Lichess rating not measured (no online games were played).
- Quiescence in the classical evaluation term: −0.17 bias on 13 % of leaves but changed 0 of 14 test moves; not worth the cost yet.
- Phase-dependent value blend, FPU from the value estimate, penalties only at even depths: not tested.
- int8 quantization: the x86 engine is unavailable in this torch build.

## 5. Verify

```powershell
cd backend; $env:PYTHONPATH = "."
pytest -q                                   # 279 passed
python scripts/engine_match.py --help        # match harness
```

After updating, restart the API (`uvicorn app.api.main:app`) and the Lichess bot, and rebuild/reload the frontend so it sends the game history.
