import numpy as np, math, sys, time, pickle, os
from multiprocessing import Pool
from sq import *

def compress(P, L, rounds=40, iters=200000, seed=0, d0=0.01):
    """Greedily shrink container; P must be feasible at L. Returns (P, L)."""
    rng = np.random.default_rng(seed); d = d0; fails = 0
    P = P.copy()
    while d > 2e-5 and rounds > 0:
        rounds -= 1
        L2 = L * (1 - d); Q = P.copy(); Q[:, :2] *= (1 - d)
        e = anneal(Q, L2, S, iters, 0.15 * d * L / 0.01 if d < 0.01 else 0.15, 0.0, int(rng.integers(1 << 30)))
        if e < 1e-22: P, L = Q, L2
        else: d *= 0.5
    return P, L

def work(args):
    n, P0, L0, seed, mode = args
    rng = np.random.default_rng(seed)
    best, bL = None, 1e9
    if mode == 'rand':
        Lg = math.ceil(math.sqrt(n))
        P, L = shrink(n, float(Lg), math.sqrt(n), tries=6, iters=300000, seed=seed)
        if P is not None: best, bL = P, L
    elif P0 is not None:
        P = P0
        if len(P) > n:   # drop random squares
            P = P[np.sort(rng.choice(len(P), n, replace=False))]
        elif len(P) < n: # add squares at random spots, then repair at slightly larger L
            L = L0 * 1.03; P = np.vstack([P, np.c_[rng.uniform(.5, L-.5, (n-len(P),2)), rng.uniform(0, 1.57, n-len(P))]])
            P[:, :2] *= 1.03
            for _ in range(4):
                e = anneal(P, L, S, 400000, 0.2, 1e-3, int(rng.integers(1<<30)))
                if e < 1e-22: break
            if e >= 1e-22: return n, None, 1e9
            P, L0 = P, L
        if len(P) == n:
            ok = total(P, L0, S) < 1e-22
            if ok:
                best, bL = compress(P, L0, seed=seed)
    return n, best, bL

if __name__ == '__main__':
    N = int(sys.argv[1]); out = sys.argv[2]
    db = pickle.load(open(out, 'rb')) if os.path.exists(out) else {}
    for n in range(1, N + 2):
        if n not in db:
            P, L = grid(n); db[n] = (P, L)
    t = time.time()
    with Pool(10) as pool:
        for sweep in range(3):
            jobs = []
            for n in range(2, N + 1):
                jobs.append((n, None, 0, sweep * 1000 + n, 'rand')) if sweep == 0 else None
                for m in (n - 1, n + 1):
                    if m in db: jobs.append((n, db[m][0], db[m][1], sweep * 1000 + n * 7 + m, 'warm'))
            for n, P, L in pool.imap_unordered(work, jobs):
                if P is not None and L < db[n][1] - 1e-9:
                    db[n] = (P, L)
            pickle.dump(db, open(out, 'wb'))
            sc = sum(db[n][1] / math.sqrt(n) - 1 for n in range(1, N + 1))
            gs = sum(math.ceil(math.sqrt(n)) / math.sqrt(n) - 1 for n in range(1, N + 1))
            print(f'sweep {sweep}: score {sc:.4f} vs grid {gs:.4f}  t={time.time()-t:.0f}s', flush=True)
