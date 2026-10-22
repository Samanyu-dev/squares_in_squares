"""Per-n optimiser using soft-to-rigid hardening at progressively smaller containers."""
import numpy as np, math, sys, time, pickle, os
from multiprocessing import Pool
from hard import *
from chain import compress
import assemble
from sq import grid, side, normalize

def hjob(a):
    n, L, iters, seed = a
    P = harden(n, L, iters, seed)
    if P is None: return None
    P, L2 = compress(P, L, max(300, iters // 20), seed + 1, d0=0.005, dmin=5e-5)
    return P, L2

def refresh(master):
    ext = assemble.assemble({n: v for n, v in master.items() if n <= 400})
    for n in ext:
        if ext[n][1] < master[n][1] - 1e-9: master[n] = ext[n]
    return master

if __name__ == '__main__':
    n0, n1, secs, dbf, mult = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), sys.argv[4], int(sys.argv[5])
    master = pickle.load(open(dbf, 'rb'))
    master = refresh(master)
    sc = lambda: sum(master[n][1] / math.sqrt(n) - 1 for n in range(1, 1001))
    print(f'start {sc():.4f}', flush=True); t = time.time()
    with Pool(10) as pool:
        for n in range(n0, n1 + 1):
            r = math.sqrt(n)
            if math.ceil(r) - r < 1e-9: continue
            L0 = master[n][1]; t0 = time.time(); step = 0.04 * (L0 - r) + 0.004
            seed = int(time.time() * 1000) % (1 << 28)
            while time.time() - t0 < secs * (1 + n / 100) and step > 1e-4:
                Lb = master[n][1]; Lt = Lb - step
                if Lt <= r: step *= 0.5; continue
                res = pool.map(hjob, [(n, Lt, mult * n, seed + s) for s in range(10)]); seed += 10
                ok = [x for x in res if x is not None]
                if ok:
                    P, L2 = min(ok, key=lambda x: x[1])
                    if L2 < Lb - 1e-9: master[n] = (P, L2)
                    step *= 1.3
                else:
                    step *= 0.55
            if n % 5 == 0 or n == n1: master = refresh(master)
            pickle.dump(master, open(dbf, 'wb'))
            print(f'n={n} {L0:.5f} -> {master[n][1]:.5f} (grid {math.ceil(r)})  score {sc():.4f}  t={time.time()-t:.0f}s', flush=True)
