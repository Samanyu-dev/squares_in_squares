import numpy as np, math, sys, time, pickle, os
from multiprocessing import Pool
from sq2 import *
from chain import compress

def attempt(n, L, iters, rng):
    P = np.c_[rng.uniform(.5, L-.5, n), rng.uniform(.5, L-.5, n), rng.uniform(0, 1.57, n)]
    e = anneal_fast(P, L, S, iters, 0.3, 2e-3, int(rng.integers(1 << 30)))
    return P if e < 1e-22 else None

def work(a):
    n, mult, seed, tries, P0, L0 = a
    rng = np.random.default_rng(seed)
    g = math.ceil(math.sqrt(n)); r = math.sqrt(n)
    best = (P0, L0) if P0 is not None else None
    if g - r < 1e-9: return n, best
    # frac of the gap closed: try ambitious first
    for frac in (0.55, 0.7, 0.82, 0.92, 0.97):
        L = r + frac * (g - r)
        if best is not None and best[1] <= L: break
        for _ in range(tries):
            P = attempt(n, L, mult * n, rng)
            if P is not None:
                P, L2 = compress(P, L, max(200, mult * n // 10), int(rng.integers(1 << 30)), d0=0.01, dmin=2e-4)
                if best is None or L2 < best[1]: best = (P, L2)
                break
        if best is not None and best[1] <= L + 1e-9: break
    return n, best

if __name__ == '__main__':
    n0, n1, mult, tries, dbf = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    db = pickle.load(open(dbf, 'rb')) if os.path.exists(dbf) else {}
    t = time.time(); seedbase = int(time.time()) % 100000
    jobs = [(n, mult, seedbase + n, tries, db[n][0] if n in db else None, db[n][1] if n in db else 1e9) for n in range(n1, n0 - 1, -1)]
    done = 0
    with Pool(10) as pool:
        for n, best in pool.imap_unordered(work, jobs):
            done += 1
            if best is not None and best[0] is not None and (n not in db or best[1] < db[n][1] - 1e-9):
                db[n] = best
            if done % 10 == 0 or done == len(jobs):
                pickle.dump(db, open(dbf, 'wb'))
                print(done, len(jobs), f't={time.time()-t:.0f}s', flush=True)
    pickle.dump(db, open(dbf, 'wb'))
    sc = sum((db[n][1] if n in db else math.ceil(math.sqrt(n))) / math.sqrt(n) - 1 for n in range(n0, n1 + 1))
    gs = sum(math.ceil(math.sqrt(n)) / math.sqrt(n) - 1 for n in range(n0, n1 + 1))
    print(f'range {n0}-{n1}: {sc:.4f} vs grid {gs:.4f}')
