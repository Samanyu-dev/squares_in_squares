"""Per-n optimiser: all cores work on one n at a time, master.pkl holds the best packing known for every n."""
import numpy as np, math, sys, time, pickle, os
from multiprocessing import Pool
from sq2 import *
from chain import compress
import assemble

def job(a):
    n, P0, L0, seed, secs, mult = a
    rng = np.random.default_rng(seed); t0 = time.time()
    best = (P0, L0); r = math.sqrt(n)
    while time.time() - t0 < secs:
        L = best[1]
        if rng.random() < 0.5:   # warm: kick some squares of the best packing, squeeze a little
            Q = best[0].copy(); k = max(1, int(rng.integers(1, 1 + max(2, n // 6))))
            idx = rng.choice(n, min(k, n), replace=False)
            Q[idx] = np.c_[rng.uniform(.5, L - .5, len(idx)), rng.uniform(.5, L - .5, len(idx)), rng.uniform(0, 1.57, len(idx))]
            Q[:, :2] += rng.normal(0, 0.03, (n, 2))
            L1 = L * (1 - rng.uniform(0.0, 0.01))
            Q[:, :2] *= L1 / L
            e = anneal_fast(Q, L1, S, mult * n, 0.25, 2e-3, int(rng.integers(1 << 30)))
        else:                     # fresh random start somewhat below current best
            L1 = L - rng.uniform(0.003, 0.06) * (L - r) / max(L - r, 1e-9) * min(1.0, L - r) if L > r else L
            L1 = max(L1, r * 1.0001)
            Q = np.c_[rng.uniform(.5, L1 - .5, n), rng.uniform(.5, L1 - .5, n), rng.uniform(0, 1.57, n)]
            e = anneal_fast(Q, L1, S, mult * n, 0.3, 2e-3, int(rng.integers(1 << 30)))
        if e < 1e-22:
            Q, L2 = compress(Q, L1, max(200, mult * n // 8), int(rng.integers(1 << 30)), d0=0.01, dmin=1e-4)
            if L2 < best[1] - 1e-9: best = (Q, L2)
    return best

def refresh(master):
    """extension + monotone closure over the whole table"""
    ext = assemble.assemble({n: v for n, v in master.items() if n <= 400})
    for n in ext:
        if ext[n][1] < master[n][1] - 1e-9: master[n] = ext[n]
    return master

if __name__ == '__main__':
    n0, n1, per, dbf = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
    mult = int(sys.argv[5]) if len(sys.argv) > 5 else 20000
    master = pickle.load(open(dbf, 'rb')) if os.path.exists(dbf) else {n: grid(n) for n in range(1, 1001)}
    for n in range(1, 1001):
        master.setdefault(n, grid(n))
    srcs = [f for f in ('db100.pkl', 'db2.pkl') if os.path.exists(f)]
    for f in srcs:
        for n, (P, L) in pickle.load(open(f, 'rb')).items():
            if n <= 1000 and L < master[n][1] - 1e-9: master[n] = (P, L)
    master = refresh(master)
    sc = lambda: sum(master[n][1] / math.sqrt(n) - 1 for n in range(1, 1001))
    print(f'start score {sc():.4f}', flush=True)
    t = time.time()
    with Pool(10) as pool:
        for n in range(n0, n1 + 1):
            P0, L0 = master[n]
            if math.ceil(math.sqrt(n)) - math.sqrt(n) < 1e-9: continue
            secs = per * (1 + n / 50)
            res = pool.map(job, [(n, normalize(P0), side(normalize(P0)) + 1e-9, int(time.time() * 1000) % (1 << 30) + s, secs, mult) for s in range(10)])
            b = min(res, key=lambda r: r[1])
            if b[1] < L0 - 1e-9:
                master[n] = (b[0], b[1])
            if n % 5 == 0 or n == n1:
                master = refresh(master)
            pickle.dump(master, open(dbf, 'wb'))
            print(f'n={n} {L0:.5f} -> {master[n][1]:.5f}  grid {math.ceil(math.sqrt(n))}  score {sc():.4f}  t={time.time()-t:.0f}s', flush=True)
