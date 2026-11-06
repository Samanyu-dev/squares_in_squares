import csv, glob, math, os, shutil, subprocess, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from scipy.spatial import cKDTree

DATA = Path(glob.glob('/kaggle/input/**/stretch_wins.csv', recursive=True)[0]).parent
BANK = glob.glob('/kaggle/input/**/perpuzzle_best.csv', recursive=True)[0]
WORK = Path('/kaggle/working')
FQ = '/tmp/fq'                     # outside /kaggle/working, so the binary is not part of the notebook's output
shutil.copy(DATA / 'fq', FQ)
os.chmod(FQ, 0o755)
sys.path.insert(0, str(DATA))
sys.path.insert(0, '/tmp')

OFF = np.array([(-.5, -.5), (.5, -.5), (.5, .5), (-.5, .5)])


def corners(s):
    s = np.asarray(s, float)
    c, si = np.cos(s[:, 2]), np.sin(s[:, 2])
    x = s[:, None, 0] + OFF[None, :, 0] * c[:, None] - OFF[None, :, 1] * si[:, None]
    y = s[:, None, 1] + OFF[None, :, 0] * si[:, None] + OFF[None, :, 1] * c[:, None]
    return np.stack([x, y], -1)


def normalise(s):
    s = np.array(s, float)
    s[:, :2] -= corners(s).reshape(-1, 2).min(0)
    return s


def side(s):
    p = corners(s).reshape(-1, 2)
    return float(max(np.ptp(p[:, 0]), np.ptp(p[:, 1])))


def tilt_deg(t):
    return (np.degrees(t) % 90 + 45) % 90 - 45


def overlaps(s, tol=1e-9):
    """Number of pairs overlapping by more than tol (separating-axis test on all close pairs)."""
    s = np.asarray(s, float)
    cs = corners(s)
    p = cKDTree(s[:, :2]).query_pairs(math.sqrt(2) + 1e-6, output_type='ndarray')
    if len(p) == 0:
        return 0
    I, J = p[:, 0], p[:, 1]
    ci, si, cj, sj = np.cos(s[I, 2]), np.sin(s[I, 2]), np.cos(s[J, 2]), np.sin(s[J, 2])
    axes = np.stack([np.stack([ci, si], -1), np.stack([-si, ci], -1), np.stack([cj, sj], -1), np.stack([-sj, cj], -1)], 1)
    pi, pj = np.einsum('mkd,mad->mak', cs[I], axes), np.einsum('mkd,mad->mak', cs[J], axes)
    sep = (np.maximum(pi.min(-1), pj.min(-1)) - np.minimum(pi.max(-1), pj.max(-1))).max(-1)
    return int((sep < -tol).sum())


def draw(ax, s, title='', mark=None):
    """Red = empty space, grey = axis-aligned squares, colour = tilt angle; `mark` (bool mask) gets a black outline."""
    s = normalise(s)
    L = side(s)
    ax.add_patch(plt.Rectangle((0, 0), L, L, color='#d9534f'))
    a = tilt_deg(s[:, 2])
    cols = ['#d4d4d4' if abs(v) < 0.05 else plt.cm.viridis((v + 45) / 90) for v in a]
    ax.add_collection(PolyCollection(corners(s), facecolors=cols, edgecolors='#333333', linewidths=0.3))
    if mark is not None and mark.any():
        ax.add_collection(PolyCollection(corners(s[mark]), facecolors='none', edgecolors='black', linewidths=1.4))
    ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10)


def load_packings(path):
    g = defaultdict(list)
    for r in csv.DictReader(open(path)):
        g[int(r['n'])].append((int(r['square_id']), float(r['x']), float(r['y']), float(r['angle'])))
    return {n: normalise([v[1:] for v in sorted(rows)]) for n, rows in g.items()}


def write_cfg(path, s):
    s = normalise(s)
    with open(path, 'w') as f:
        f.write(f'{len(s)} {side(s) + 1e-9:.17f}\n')
        for x, y, t in s:
            f.write(f'{x:.17f} {y:.17f} {math.degrees(t) % 90:.15f}\n')


def read_cfg(path):
    rows = [l.split() for l in open(path) if l.strip() and not l.startswith('#')]
    n = int(rows[0][0])
    return normalise([[float(a), float(b), math.radians(float(c))] for a, b, c, *_ in rows[1:1 + n]])


def quench(s, seconds=180, loosen=1.0, tag='q'):
    """One `fq quench --finish`. None if fq gives nothing back or the result overlaps."""
    a, b = str(WORK / f'{tag}_in.cfg'), str(WORK / f'{tag}_out.cfg')
    write_cfg(a, s)
    if os.path.exists(b):
        os.remove(b)
    subprocess.run([FQ, 'quench', '--in', a, '--out', b, '--finish', '--deadline', str(seconds), '--loosen', str(loosen)],
                   capture_output=True, timeout=seconds + 120)
    if not os.path.exists(b):
        return None
    q = read_cfg(b)
    return q if len(q) == len(s) and overlaps(q) == 0 else None


demo = load_packings(DATA / 'demo_packings.csv')      # the packings as they were when I stretched them
ours = pd.read_csv(DATA / 'ours_L.csv', index_col='n')  # my side per puzzle: previous notebook (L_v10) and this file

start = read_cfg(DATA / 'stretch_930_to_993_start.cfg')
after = read_cfg(DATA / 'stretch_930_to_993_fq.cfg')
fig, axs = plt.subplots(1, 3, figsize=(16, 5.6))
draw(axs[0], demo[930], f'n = 930, side {side(demo[930]):.6f}')
draw(axs[1], start, f'one column + one row copied in: {len(start)} squares, {overlaps(start)} overlapping pairs',
     mark=np.arange(len(start)) >= len(demo[930]))
draw(axs[2], after, f'993 after fq: {side(after):.6f}  (I had {ours.L_v10[993]:.6f})')
plt.tight_layout(); plt.show()