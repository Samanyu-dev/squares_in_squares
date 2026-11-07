"""opposite-handed arms: reflect the part of a packing beyond a vertical (or horizontal) cut about the middle of that part (angles flip sign),
repair the seam with overlap-energy relax in a slightly larger box, then fq quench.  usage: mirror.py best.csv out tries n1 n2 ...  env NPROC"""
import sys, os, math, json, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run, ovl
def job(a):
    n, P0, L, tries, out = a; rng = random.Random(n); wd = f'{out}/w{n}'; os.makedirs(wd, exist_ok=True); bestL = 1e9; cnt = [0, 0, 0]
    for t in range(tries):
        P = P0.copy(); ax = rng.choice([0, 1]); cut = rng.uniform(.25, .75) * L
        if rng.random() < .5: P[:, ax] = L - P[:, ax]
        right = P[:, ax] > cut
        if right.sum() < 3: continue
        lo = (P[right, ax] - 1).min() if False else cut; Q = P.copy()
        Q[right, ax] = cut + L - P[right, ax]          # reflect the far part about the middle of [cut, L]
        Q[right, 2] = -Q[right, 2]
        Q[right, ax] = Q[right, ax] + 0.0
        S = L + rng.uniform(.01, .06); cnt[0] += 1
        Z, E = ovl.relax(Q, S)
        if E > 1e-6: continue
        cnt[1] += 1
        q = run.quench(Z, 50, f'm{n}_{t}', wd)
        if q is not None and run.overlaps(q) == 0:
            s = run.side(q); bestL = min(bestL, s)
            if s < L - 1e-9: np.save(f'{out}/win_{n}_m.npy', q); return (n, 'WIN', s, L)
    return (n, 'tries/relaxed/quenched', cnt[0], cnt[1], round(bestL, 5), round(L, 5))
if __name__ == '__main__':
    best, out, tries = sys.argv[1], sys.argv[2], int(sys.argv[3]); ns = [int(x) for x in sys.argv[4:]]; os.makedirs(out, exist_ok=True)
    PK = run.load(best); from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(int(os.environ.get('NPROC', 4))) as ex:
        for r in ex.map(job, [(n, run.normalise(PK[n]), run.side(PK[n]), tries, out) for n in ns]): print(r, flush=True)
