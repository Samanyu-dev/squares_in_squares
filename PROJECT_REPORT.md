# Squares in Squares: project report

Competition: Kaggle "Squares in Squares", n = 1..1000 unit squares per puzzle, 500,500 rows.
Score = sum over n of (L / sqrt(n) - 1), where L is the side of the smallest axis-aligned box around the rotated squares. Lower is better.
Account: samanyu1808.

## Current values

| Item | Value |
|---|---|
| Best file | `watch/final_enforced4.csv` |
| Kaggle score (submitted 2026-11-06) | **22.2362359319**; certified local best 22.2318355162 (7 Nov, unsubmitted) |
| Previous submission | 22.2498820302 (`final_enforced4.csv`) |
| Starting point (grid baseline) | 28.9377 |
| Leaderboard top (2026-11-06) | 22.0544, 22.0871, 22.1217 |
| Gap to 22.20 / 22.10 / 22.00 | 0.050 / 0.150 / 0.250 |

Per-n quality (`quality_by_n.csv`: n, L, excess, tier, tilted squares):

| Tier | Excess over sqrt(n) | n count | Share of score |
|---|---|---|---|
| Perfect | 0 (n = K^2, 31 values) | 31 | 0 |
| Near perfect | under 0.4% | 69 | 0.16 |
| Middle | 0.4 to 1.2% | 162 | 1.29 |
| Weak | 1.2 to 2.5% | 482 | 8.68 |
| Bad | over 2.5% (all n <= 411) | 256 | 12.12 |

Selected side lengths: n=2: 2.0, n=5: 2.7071, n=10: 3.7071, n=17: 4.6755, n=26: 5.6213, n=37: 6.5986, n=52: 7.7071.
Small-n values are the proven or published optima.

Reference files: `best_fit_summary.csv` (one row per n, to n=2000, rows above 1000 are outside the competition), `best_fit.csv` (n=1..100 per square), `best_fit_101_500.csv`, `best_fit_501_1000.csv`.

## What decides the score

- Only L matters. Square id is a label 0..n-1, not a quality factor.
- x, y place each centre. angle tilts a square, which costs |cos|+|sin| of width unless many tilted squares lock together.
- Gap: Kaggle needs at least about 1e-11 SAT clearance between squares. We enforce 2e-11 (`watch/enforce.py`), costing about 1e-9 per puzzle.
- Floor: L >= sqrt(n). Real packings waste about 1 to 3% of the area for large n.

## Score history (certified totals)

28.3553 -> 27.32 -> 26.79 -> 22.4776 -> 22.3747 -> 22.3611 -> 22.2616 -> 22.2604 -> 22.2603 (submitted) -> **22.2499** (submitted).
Every step came from merging public work with per-puzzle certification, never from our own search.

## Methods tried

| Method | Result |
|---|---|
| Grid baseline and monotone closure | 28.94, then 28.36 |
| Border extension (L-shaped border on a good packing) | Small gains, already used by public notebooks |
| Soft-to-rigid annealing, CPU (numba) | Matches public values for n up to about 70, jams above |
| CUDA annealing kernel (Kaggle GPU) | Same plateau, no gain beyond public |
| Own hop search on AWS SageMaker (about 1200 searches, about $10.6 of $50) | 0 improvements |
| Gadget search and fault-line DP rebuild (Deotte-style) | Matched public values, did not beat them |
| Tilted-stack (Erdos-Graham) probe | Negative: needs f of about 0.87 at K=31, our table has 0.52 |
| Contact LP unjam on a Kaggle GPU (leaders' open notebook, our baseline) | 21 minutes, gain 0.000014 (puzzles 797, 875) |
| Merging public notebooks per puzzle with certification (98 notebooks) | All gains. Now at the public frontier |
| Merging public datasets (about 12, includes best-of-1105, private-packings, l-band) | Equal to ours, no further gain |
| Reading competition discussions | Metric bug fixes and papers already tried; no packings posted |

## Lessons and rules

- Always certify each puzzle in a fresh process with the current metric (`runner/isolated_cert.py`, metric copy `scoretest_n100/`). Whole-file scoring is order dependent for touching squares.
- Always apply `watch/enforce.py` before submitting. A file with about 1e-12 gaps scored 22.3564 on Kaggle against 22.2604 locally.
- Merge by certified per-puzzle L, never by raw smallest L, because the metric tightened (literal length cap 100, coordinates relative to square 0).
- Never run two `watch/check2.py` at once. The 3-hourly scheduled check only runs while the Claude app is open.
- Do not submit without the user's go-ahead. Submissions went through when asked for directly in chat.

## Key scripts

- `merge.py` per-n minimum merge; `watch/check2.py` certification-aware public check; `watch/enforce.py` clearance; `runner/isolated_cert.py`.
- `apply_improvement.py N packing.csv` updates one row; `update_summary_refresh.py` refreshes the summary.
- `make_long.py` builds the Kaggle GPU kernel; `fault/` and `probe/` hold the construction attempts.

## Where a gain could come from

1. A leader publishing a notebook or dataset (the 3-hourly check picks it up).
2. New structure for the weak and bad tiers, especially n just above a perfect square (962 to 967, 901 to 906, 842 to 847): richer gadget catalogs, mixed-angle seams, rectangular MILP refills.
3. Reaching 22.0 would need layouts nobody has published; none of our own methods has produced one.

## 7 Nov update
- Kaggle 22.2246648506 (final_enforced11, per-puzzle merge incl. jazivxt Boxed In, certified with isolated_cert).
- Per-puzzle leader sides form lift chains L(n)=L(n')+m (23 of 407 behind-puzzles, e.g. 259->445->488->681->734->967); our deficit is identical along a chain, so only improving the root helps. No lift gain exists today (checked).
- Hot kick+quench on roots 259,202,228,292,328,442,444: no gain. seedband.py (corner-to-corner band seeds): no gain. Chains on n>=100 top-gap puzzles and AWS n>=500 chains still running.
