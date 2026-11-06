import json
def md(s): return {"cell_type":"markdown","metadata":{},"source":s.strip().splitlines(True)}
def code(s): return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":s.strip().splitlines(True)}
cells=[
md("""# Squares in Squares: certified packings via stretch + quench

**What this notebook does.** It takes our best per-puzzle packings (dataset `samanyu1808/sis-final-packings`), independently re-verifies every puzzle (bounds, overlap, clearance), recomputes the competition score, explains where the score comes from, and writes `submission.csv`.

**Score** = sum over n=1..1000 of `L/sqrt(n) - 1`, where L is the side of the smallest axis-aligned square containing the n unit squares. Lower is better. Perfect squares n=K^2 contribute 0.

## How the packings were produced
1. **Per-puzzle merge** of ~100 public notebooks and ~25 public datasets, keeping the smallest *certified* side for each n (never the raw smallest: the official metric needs about 1e-11 clearance, so each puzzle is certified in a fresh process).
2. **Stretch + quench.** A packing with tilted arms along two walls is cut by one vertical and one horizontal line. One arm column and one row are copied in, everything past the cut moves by the measured arm pitch, and Evan Daniel's `fq` quench relaxes the overlaps. Kick-and-quench chains (random jitter, random loosen, keep if smaller) refine near-misses.
3. **Clearance enforcement.** Every non-grid puzzle is dilated slightly about the box centre so every pair keeps at least 2e-11 separating-axis clearance.

Credits: public packings from many Kaggle authors; `fq` is Evan Daniel's square-packing (MIT); the stretch idea follows az05192000gmailcom's notebook *22.2427 | Stretching good packings to nearby n*."""),
code("""import glob, math, time
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from scipy.spatial import cKDTree
SRC = glob.glob('/kaggle/input/**/submission.csv', recursive=True)[0]
df = pd.read_csv(SRC)
print(SRC, df.shape, list(df.columns))
assert list(df.columns) == ['id','n','square_id','x','y','angle'] and len(df) == 500500"""),
md("""## 1. Independent verification
For every puzzle we rebuild the four corners of each rotated unit square, then check (a) the bounding side L, (b) all pairs of nearby squares with the separating-axis test and report the smallest clearance, and (c) that every n has exactly n squares with ids 0..n-1."""),
code("""OFF = np.array([(-.5,-.5),(.5,-.5),(.5,.5),(-.5,.5)])
def corners(s):
    c, si = np.cos(s[:,2]), np.sin(s[:,2])
    return np.stack([s[:,None,0]+OFF[None,:,0]*c[:,None]-OFF[None,:,1]*si[:,None],
                     s[:,None,1]+OFF[None,:,0]*si[:,None]+OFF[None,:,1]*c[:,None]], -1)
def side(s):
    p = corners(s).reshape(-1,2); return float(max(np.ptp(p[:,0]), np.ptp(p[:,1])))
def min_clearance(s):
    # smallest separating-axis gap over all nearby pairs (negative = overlap)
    if len(s) < 2: return np.inf
    cs = corners(s); pr = cKDTree(s[:,:2]).query_pairs(math.sqrt(2)+1e-6, output_type='ndarray')
    if len(pr) == 0: return np.inf
    I, J = pr[:,0], pr[:,1]
    ci, si, cj, sj = np.cos(s[I,2]), np.sin(s[I,2]), np.cos(s[J,2]), np.sin(s[J,2])
    ax = np.stack([np.stack([ci,si],-1), np.stack([-si,ci],-1), np.stack([cj,sj],-1), np.stack([-sj,cj],-1)], 1)
    pi, pj = np.einsum('mkd,mad->mak', cs[I], ax), np.einsum('mkd,mad->mak', cs[J], ax)
    return float((np.maximum(pi.min(-1), pj.min(-1)) - np.minimum(pi.max(-1), pj.max(-1))).max(-1).min())
rows = []
t0 = time.time()
for n, g in df.groupby('n'):
    g = g.sort_values('square_id'); assert len(g) == n and list(g.square_id) == list(range(n))
    s = g[['x','y','angle']].to_numpy(float); L = side(s)
    rows.append((n, L, L/math.sqrt(n)-1, min_clearance(s), float(np.degrees(s[:,2]).round(3).__abs__().__mod__(90).__gt__(0.5).sum()) if n > 1 else 0.0))
R = pd.DataFrame(rows, columns=['n','L','excess','min_clearance','tilted_squares']).set_index('n')
print(f'verified {len(R)} puzzles in {time.time()-t0:.0f}s')
print('smallest clearance anywhere:', R.min_clearance.replace(np.inf, np.nan).min())
grid = R.tilted_squares == 0\nprint(f'axis-aligned grid puzzles (touching exactly, accepted by the metric): {int(grid.sum())}')\nprint('tilted puzzles with clearance < 1e-11 (would be rejected):', int(((R.min_clearance < 1e-11) & ~grid).sum()))
score = R.excess.sum(); print(f'SCORE = {score:.10f}')"""),
md("""## 2. Where the score comes from
Excess `L/sqrt(n) - 1` per puzzle, split into quality tiers. Weak and bad puzzles (more than 1.2% over the area bound) carry almost all of the score, and the leaders' remaining advantage lives there too."""),
code("""bins = [-1e-9, 1e-9, 0.004, 0.012, 0.025, 10]
R['tier'] = pd.cut(R.excess, bins, labels=['perfect','near (<0.4%)','middle (<1.2%)','weak (<2.5%)','bad (>=2.5%)'])
T = R.groupby('tier', observed=True).agg(puzzles=('L','size'), score=('excess','sum'), mean_excess=('excess','mean'))
T['share_%'] = 100*T.score/T.score.sum(); display(T.round(4))
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].scatter(R.index, 100*R.excess, s=4); ax[0].set_xlabel('n'); ax[0].set_ylabel('excess over sqrt(n), %'); ax[0].set_title('Per-puzzle excess')
K = np.floor(np.sqrt(R.index)).astype(int); D = R.index - K*K
m = (K >= 20) & (D > 0); ax[1].scatter((D[m]+4)/K[m], (R.L - K)[m], s=5)
ax[1].set_xlabel('g = (D+4)/K'); ax[1].set_ylabel('f = L - K'); ax[1].set_title('Overhang curve, K >= 20'); plt.tight_layout(); plt.savefig('/kaggle/working/excess.png', dpi=110)
display(R.sort_values('excess', ascending=False).head(10).round(6))"""),
md("""## 3. A few packings
Grey squares are axis-aligned, coloured squares are tilted (colour = tilt angle), red is empty space. The large-n layouts use two tilted arms of different thickness along two walls, ending against an upright block in the corner."""),
code("""def draw(ax, n):
    g = df[df.n == n].sort_values('square_id'); s = g[['x','y','angle']].to_numpy(float)
    s[:,:2] -= corners(s).reshape(-1,2).min(0); L = side(s)
    ax.add_patch(plt.Rectangle((0,0), L, L, color='#d9534f'))
    tilt = (np.degrees(s[:,2]) % 90 + 45) % 90 - 45
    cols = ['#d4d4d4' if abs(v) < 0.05 else plt.cm.viridis((v+45)/90) for v in tilt]
    ax.add_collection(PolyCollection(corners(s), facecolors=cols, edgecolors='#333', linewidths=0.3))
    ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f'n={n}  L={L:.5f}  excess {100*(L/math.sqrt(n)-1):.2f}%', fontsize=9)
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
for a, n in zip(axs.ravel(), [17, 37, 150, 377, 599, 993]): draw(a, n)
plt.tight_layout(); plt.savefig('/kaggle/working/packings.png', dpi=100)"""),
md("""## 4. Write the submission
The certified file is copied byte for byte (no re-formatting of floats, which could change the last digit), and the table of per-puzzle sides is saved next to it."""),
code("""import shutil; shutil.copy(SRC, '/kaggle/working/submission.csv')  # byte-identical copy of the certified file (no float re-formatting)
R[['L','excess','min_clearance','tilted_squares']].to_csv('/kaggle/working/per_puzzle_L.csv')
print('wrote submission.csv', df.shape, f'| local score {score:.10f}')"""),
md("""## What did not work
Own annealing/GPU search beyond n~70, gadget DP rebuilds, contact-LP unjam, boundary LNS, delete-one-and-squeeze, plain random restarts below the grid ceiling for n=12..57. All matched public values or found nothing. The only real gains came from per-puzzle merging of public work and from stretch + quench.

## Remaining gap
The leaders' per-puzzle sides (29 Oct bank) are ~0.03 to 0.04 smaller than ours on about 410 puzzles with n = K^2 + D, K = 20..31, worth about 0.17 of score. Closing that needs a better construction for the tapered arm ends.""")]
json.dump({"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"}},"nbformat":4,"nbformat_minor":5}, open('kernel/nb.ipynb','w'))
print('ok')
