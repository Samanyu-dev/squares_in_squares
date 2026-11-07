# Progress report, 8 Nov 2026

Everything we did on Kaggle "Squares in Squares" (n = 1..1000, score = Σ(L/√n − 1), lower is better), what worked, what did not, and where we stand.

## 1. Where we stand

| Item | Value |
|---|---|
| Kaggle score (verified) | **22.2228889219** (notebook `samanyu1808/sis-submit`, submission 58101752, 7 Nov) |
| Certified locally, not yet submitted | **22.2226173288** (`watch/best_current.csv`, isolated per-puzzle check 0 rejected) |
| Previous Kaggle scores | 22.2246648506 → 22.2241274836 → 22.2234504853 → 22.2228889219 |
| Start of the project | 28.9377 (plain grid) |
| Leaderboard top (6 Nov) | 22.0544 / 22.0871 / 22.1217 / 22.1224 |
| Gap to top | about 0.17 |
| Per-puzzle leaderboard (473 puzzles with a known leader) | **1 strictly ahead (n=109), 61 tied, 411 behind** |
| Total gap to per-puzzle leaders on those 411 | 0.157 score points (0.019 for n<400, 0.139 for n≥400) |
| Clearance we enforce | 2e-11 (costs about 1e-9 per puzzle) |

Score history (certified): 28.94 → 28.36 → 27.32 → 26.79 → 22.48 → 22.37 → 22.36 → 22.26 → 22.2603 → 22.2499 → 22.2382 → 22.2362 → 22.2318 → 22.2247 → 22.2241 → 22.2235 → 22.2229 → **22.2226**.

## 2. How the n values look

Excess = L/√n − 1 per puzzle, from the current certified best.

| Tier | Excess | n count | Share of the total |
|---|---|--:|--:|
| Perfect | 0 | 31 | 0.00 |
| Near perfect | < 0.4% | 69 | 0.16 |
| Middle | 0.4–1.2% | 162 | 1.29 |
| Weak | 1.2–2.5% | 483 | 8.69 |
| Bad | > 2.5% | 255 | 12.09 |

| n range | Score sum | Mean per puzzle |
|---|--:|--:|
| 1–100 | 6.3012 | 0.0630 |
| 101–300 | 5.4873 | 0.0274 |
| 301–600 | 5.2937 | 0.0176 |
| 601–1000 | 5.1404 | 0.0129 |

- Worst by excess: n = 2, 6, 5, 10, 11, 3, 12, 18 (proven or published optima, nothing to gain).
- Largest remaining gaps to a per-puzzle leader: n = 971, 498, 593, 544, 972, 911, 592, 367, 545, 791 (each about 0.001 of score, gap in side 0.02–0.04).
- Where the 411 behind-puzzles sit by D = n − K²: D=1: 12 puzzles, 0.0052; D=2: 6, 0.0016; D=3: 13, 0.0030; D=4: 12, 0.0046; D≥5: 368, 0.143.
- The 7 Nov submissions closed about 0.002 of the gap; the rest needs layouts we cannot yet produce.

## 3. What worked

| Method | Gain |
|---|---|
| Per-puzzle merge of public notebooks and datasets, each puzzle certified alone with the current official metric | almost all of 28.9 → 22.22 |
| Stretch + fq quench + kick chains on mid and high n (`stretch/chain.py`) | 1e-5 to 3e-4 per win, about 0.002 in total |
| Same chains on AWS SageMaker (ml.c5.4xlarge, 5 h cap, code via S3, `aws_job/cloud_chain.py`) | 18 wins n=650–990 (0.0004); latest pair of jobs gave 19 win files on 15 puzzles, 13 certified wins incl. a local n=690 win (0.00013) |
| Merging every public improvement even if tiny (kmmmvs, kirilltym n=991, malyshevdanil atlas, jazivxt updates) | 0.0003–0.0006 |
| Lower-n chains (n=417, 419, 460, 595, 690, 339, 109) | n=109 beats the per-puzzle leader by 1.5e-5 |
| Notebook submission pipeline (`nbsub/submit_nb.sh`) with a detailed dataset and notebook | repeatable submissions |

## 4. What did not work (do not repeat)

- Annealing (CPU numba and CUDA), basin hopping, LNS, hop search on AWS (about 1200 searches): matches public values up to n≈70 only.
- Fault-line DP gadgets, tilted-stack (Erdős–Graham) probe, contact-LP unjam, seam CP-SAT with continuous repair.
- Jitter-and-quench on near-ties (n=308, 409, 332, …): 0 gain.
- Lifting our own packings along the leaders' lift chains L(n)=L(n')+m: no gain, our deficit is identical along a chain.
- Rigid 45° corner-to-corner band and diamond seeds plus fq (`stretch/seedband.py`, `evo/band*.py`, angles 33–57°): worse than ours.
- Overlap-energy squeeze (`stretch/ovl.py`), cut-and-reinsert (`ovl2.py`), mirrored arms (`mirror.py`), global rotation (`rot.py`), LLM-style program-search harness (`evo/harness.py`): all 0 gains.
- Exhaustive {0° corner-anchored grids + free 45° lattice} search by exact independent-set ILP (`evo/lattice.py`): finds the known 5×5 diamond at K=20, S=20.5356 (401 squares) but nothing below. At S=20.530 and 20.526 the best is 400, over 64 lattice offsets.
- Lower clearance (1e-11, 3e-12) test: not run to completion; 2e-11 kept.

## 5. What the per-puzzle leaderboard tells us

- Leaders' K²+1 sides (K=20..31, plus n=678, 403) are **not** in Q(√2) and satisfy no small polynomial: their packings are numerically relaxed, not closed-form 0°/45° contact geometry. Controls n=325, 362 come out exactly as (30+5√2)/2 and (32+5√2)/2.
- The offset L−K is constant in runs of K (0.513435 for K=25..28, 0.524648 for K=21,22, 0.506591 for K=30,31): one fixed local gadget dropped into a K×K grid. Each larger gadget needs a larger minimum K (20, 21, 23, 24, 25, 29, 30) and gives a smaller offset.
- Our K≥25 value 0.520815 is the same kind of gadget, but less relaxed and with a smaller footprint than theirs.
- A leader (rank 45 in the competition) confirmed in the discussion that the angles are exactly 0° and 45° with the 0° squares anchored at a corner. Our simple 0°/45° model does not reproduce their K=20 value, so there is a degree of freedom we do not yet model (several 45° lattices, or grids offset by multiples of √2/2).
- 527 puzzles have no leader row in the snapshot; the 473 that do are the basis for the numbers above.

## 6. Next steps

1. Small-box gadget search for K²+1 (K=20, 24, 25) using the leader's offset as the target; a wider 0°/45° family.
2. Keep the local and AWS chains running on the 411 behind-puzzles, merge with `watch/check2.py`, certify, submit with `nbsub/submit_nb.sh` (only with the owner's go-ahead).
3. Submit the certified 22.2226173288 file when authorised (6 submissions were left on 7 Nov).

## 7. Reproduce

```bash
python watch/check2.py                 # download public outputs, per-puzzle merge, isolated certification
python watch/make_enforced.py watch/best_current.csv final.csv      # enforce 2e-11 clearance
python runner/isolated_cert.py <scoretest dir> final.csv iso.json   # certify every puzzle alone
python watch/wins2csv.py out.csv win_*.npy                          # chain wins -> partial submission csv
sh nbsub/submit_nb.sh final.csv "message"                           # notebook submission (needs go-ahead)
```
