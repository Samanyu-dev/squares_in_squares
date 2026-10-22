import numpy as np, math, sys, time, pickle, os
from multiprocessing import Pool
from sq2 import *

def compress(P, L, iters, seed, d0=0.02, dmin=3e-4):
    rng = np.random.default_rng(seed); d = d0; P = P.copy()
    while d > dmin:
        L2 = L * (1 - d); Q = P.copy(); Q[:, :2] *= (1 - d)
        e = anneal_fast(Q, L2, S, iters, min(0.2, 8 * d), 0.0, int(rng.integers(1 << 30)))
        if e < 1e-22: P, L = Q, L2
        else: d *= 0.6
    return P, L

def run_chain(args):
    k, mult, seed, start = args   # chain from (k+1)^2 grid down to k^2+1
    rng = np.random.default_rng(seed)
    out = {}
    hi = (k + 1) ** 2
    P, L = start if start is not None else grid(hi)
    if start is None: P = P.copy()
    n = len(P)
    while n > k * k:
        n -= 1
        best = None
        for c in range(2):
            idx = rng.integers(len(P)); Q = np.delete(P, idx, axis=0)
            Q2, L2 = compress(Q, L, mult * n, int(rng.integers(1 << 30)))
            if best is None or L2 < best[1]: best = (Q2, L2)
        P, L = best
        out[n] = (P.copy(), L)
    return k, out

if __name__ == '__main__':
    kmin, kmax, mult, dbf = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    db = pickle.load(open(dbf, 'rb')) if os.path.exists(dbf) else {}
    t = time.time()
    jobs = [(k, mult, 1000 + k, None) for k in range(kmax, kmin - 1, -1)]
    with Pool(10) as pool:
        for k, out in pool.imap_unordered(run_chain, jobs):
            for n, (P, L) in out.items():
                if n > 1000: continue
                if n not in db or L < db[n][1] - 1e-9: db[n] = (P, L)
            pickle.dump(db, open(dbf, 'wb'))
            g = sum(math.ceil(math.sqrt(n)) / math.sqrt(n) - 1 - (out[n][1] / math.sqrt(n) - 1) for n in out)
            print(f'k={k} gain vs grid {g:.4f} t={time.time()-t:.0f}s', flush=True)
