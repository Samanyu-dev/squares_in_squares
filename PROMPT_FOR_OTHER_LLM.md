# Task: lower the score of a Kaggle "Squares in Squares" submission

## Problem
For each n = 1..1000, place n unit squares (side 1, any rotation, no overlap) inside the smallest axis-aligned square of side L.
Score = sum over n of (L_n / sqrt(n) - 1). Lower is better. Perfect squares n = K^2 give 0.
Submission CSV columns: `id,n,square_id,x,y,angle` (500,500 rows, `angle` in radians, x,y = square centre).
Only the bounding-box side L matters; `square_id` is just a label.

## Current state (6 Nov 2026)
- Our best certified total: **22.2382112** (file `final_enforced5.csv`, clearance-enforced, 0 of 1000 puzzles rejected in isolation).
- Last Kaggle-scored submission: 22.2498820; `final_enforced5.csv` was submitted afterwards and its score may still be pending.
- Target: 22.1199 (about 0.118 lower). Leaderboard top is about 22.05-22.12; their layouts are unpublished.
- Where the score sits (excess L/sqrt(n) - 1): perfect 31 puzzles (0), near <0.4% 69 puzzles, middle 162, weak 482 puzzles (1.2-2.5%, 8.68 of score), bad 256 puzzles (>2.5%, n <= 411, 12.12 of score). Small n are proven or published optima, so gains must come from n >= ~100.

## Hard rules (a violation makes a submission worthless)
1. The grader needs about 1e-11 SAT clearance between squares and between squares and the box. We enforce >= 2e-11 by a tiny dilation about the box centre. Never deliver a packing with touching or overlapping squares.
2. Certify each puzzle on its own, in a fresh process, with the current official metric; whole-file scoring is order dependent for touching squares.
3. Deliver per-puzzle improvements only where certified L is smaller than the current best L for that n; we merge per puzzle.
4. Do not submit to Kaggle. Deliver files only.
5. Report only numbers you computed. No fabricated or "illustrative" results.

## What has already been tried (do not repeat)
- Annealing, CUDA/GPU annealing, SageMaker hop search: matched public values for n up to about 70, nothing beyond.
- Fault-line gadget DP (Deotte style), contact-LP unjam, boundary-band LNS, delete-one-from-(n+1)-and-squeeze, neighbour transfer, SLP polish: no gains over public values.
- Discrete seam solver (pose library + CP-SAT cell cover): the cover is feasible, but the continuous polish never produced a valid smaller packing.
- Tilted-stack (Erdos-Graham) probe: negative at these sizes.
- Merged all ~100 public notebooks and ~25 datasets per puzzle with certification: all current gains came from there.
- **Stretch + fq (in progress):** cut a packing with tilted arms along walls, copy one arm column and one row, shift by the arm pitch, relax with Evan Daniel's `fq` quench. Public notebook gained 0.0071 this way.

## Ideas worth trying
1. Stretch/shrink generalisations: stretch with different arm pitches, three-cut stretches, stretching specimens from other tilt families (45 degrees and about 18.4 degrees dominate).
2. Longer kick-and-quench chains (sigma 0.002-0.02, random loosen) on near-miss stretches; the public notebook found extra wins there.
3. Seam solver with a better continuous repair: sequential LP from the CP selection instead of annealing.
4. n just above a perfect square (842-847, 901-906, 962-967): richer gadget catalogues, mixed-angle seams.
5. Weak/bad tiers: n in 100-411 with excess above 1.2%, using structure (arms, L-bands, diamond gadgets), not plain annealing.

## Inputs you will be given
- `final_enforced5.csv`: current best, all 1000 puzzles.
- `best_L.json`: current certified L per n; a packing is useful only if it beats this.
- Official metric code (`sis_metric.py`) and `enforce.py` (`enforce(n, P, target=2e-11)`).

## Deliverables
1. `improved.csv` in submission format containing ONLY the puzzles you improved (or the full file with improved rows replaced).
2. `report.csv` with columns `n, old_L, new_L, delta_score, method`.
3. A short note: what worked, what did not, and a runnable script for each method.
Every improved puzzle must pass the official metric after a pandas round-trip read and keep >= 2e-11 clearance. Small gains are welcome; a gain of 1e-5 total is still a gain.
