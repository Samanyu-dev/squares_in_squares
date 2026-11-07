"""cut-and-reinsert: crop the packing to a smaller box (random corner), put the squares that fall outside back at the least-overlapping
random poses, minimise overlap energy (ovl.relax), quench if E~0.  usage: ovl2.py best.csv out tries n1 n2 ...  env NPROC DLO DHI"""
import sys, os, math, json, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run, ovl
from scipy.spatial import cKDTree
def job(a):
    n, P0, L, tries, out = a
    rng = random.Random(n * 7 + 1); gen = np.random.default_rng(n); wd = f'{out}/w{n}'; os.makedirs(wd, exist_ok=True); bestE = 1e9
    for t in range(tries):
        d = rng.uniform(float(os.environ.get('DLO', 0.001)), float(os.environ.get('DHI', 0.02))); S = L - d
        ox, oy = rng.choice([0, d]), rng.choice([0, d]); P = P0.copy(); P[:, 0] -= ox; P[:, 1] -= oy
        c = run.corners(P); out_ = (c[:, :, 0].min(1) < -1e-12) | (c[:, :, 0].max(1) > S + 1e-12) | (c[:, :, 1].min(1) < -1e-12) | (c[:, :, 1].max(1) > S + 1e-12)
        m = int(out_.sum())
        if m == 0 or m > 30: continue
        idx = np.where(out_)[0]; keep = P[~out_]; tree = cKDTree(keep[:, :2])
        for i in idx:                              # least-crowded random pose
            cand = np.column_stack([gen.uniform(.5, S - .5, 300), gen.uniform(.5, S - .5, 300), gen.uniform(0, math.pi / 2, 300)])
            dist, _ = tree.query(cand[:, :2]); j = int(np.argmax(dist)); P[i] = cand[j]
            keep = np.vstack([keep, cand[j:j + 1]]); tree = cKDTree(keep[:, :2])
        Q, E = ovl.relax(P, S); bestE = min(bestE, E)
        if E < 1e-14:
            q = run.quench(Q, 40, f'c{n}_{t}', wd)
            if q is not None and run.side(q) < L - 1e-9 and run.overlaps(q) == 0:
                np.save(f'{out}/win_{n}_c.npy', q); return (n, 'WIN', run.side(q), L)
    return (n, 'minE', bestE)
if __name__ == '__main__':
    best, out, tries = sys.argv[1], sys.argv[2], int(sys.argv[3]); ns = [int(x) for x in sys.argv[4:]]; os.makedirs(out, exist_ok=True)
    PK = run.load(best); from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(int(os.environ.get('NPROC', 4))) as ex:
        for r in ex.map(job, [(n, run.normalise(PK[n]), run.side(PK[n]), tries, out) for n in ns]): print(r, flush=True)
