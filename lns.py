"""Seam/boundary LNS: shrink container by delta, free only squares near the violated boundary, repair by annealing."""
import sys, math, numpy as np, csv, collections, time
from numba import njit
import sq
from sq import pair, wall, e_one, total, side, normalize

@njit(cache=True)
def anneal_mask(P, mov, L, s, iters, sig0, T0, seed):
    np.random.seed(seed)
    idx = np.where(mov)[0]; m = len(idx)
    E = total(P, L, s)
    for it in range(iters):
        if E < 1e-22: break
        f = 1.0 - it / iters
        sig = sig0 * (0.02 + f); T = T0 * f * f
        i = idx[np.random.randint(m)]
        x = P[i, 0] + sig * np.random.randn(); y = P[i, 1] + sig * np.random.randn()
        t = P[i, 2] + sig * 2.0 * np.random.randn() if np.random.rand() < 0.5 else P[i, 2]
        old = e_one(P, i, P[i, 0], P[i, 1], P[i, 2], L, s); new = e_one(P, i, x, y, t, L, s)
        d = new - old
        if d < 0 or (T > 0 and np.random.rand() < math.exp(-d / T)):
            P[i, 0] = x; P[i, 1] = y; P[i, 2] = t; E += d
    return total(P, L, s)

def load(path, n):
    A = []
    for r in csv.reader(open(path)):
        if r[0] != 'id' and int(r[1]) == n: A.append([float(v) for v in r[3:6]])
    return np.array(A)

def try_n(n, path, deltas, radius, iters, seeds):
    P0 = normalize(load(path, n)); L0 = side(P0); best = None
    for dl in deltas:
        Lt = L0 - dl
        for sd in range(seeds):
            P = P0.copy(); h = .5 * (abs(np.cos(P[:, 2])) + abs(np.sin(P[:, 2])))
            viol = (P[:, 0] + h > Lt) | (P[:, 1] + h > Lt)
            # movable: violators and anything within radius of them (or tilted squares near them)
            c = P[:, :2]; mov = viol.copy()
            for i in np.where(viol)[0]: mov |= (np.hypot(*(c - c[i]).T) < radius)
            if mov.sum() < 2: continue
            e = anneal_mask(P, mov, Lt, sq.S, iters, 0.15, 3e-4, 1000 * sd + n)
            if e < 1e-22 and side(P) <= Lt + 1e-9: best = (Lt, P.copy()); break
        if best is None: break
    return L0, best

if __name__ == '__main__':
    n = int(sys.argv[1]); path = sys.argv[2]
    t = time.time(); L0, b = try_n(n, path, [2e-4, 1e-3, 3e-3], radius=float(sys.argv[3]), iters=int(sys.argv[4]), seeds=int(sys.argv[5]))
    print(n, 'L0', round(L0, 6), 'improved' if b else 'no', (round(L0 - b[0], 6) if b else 0), round(time.time() - t), 's', flush=True)
    if b: np.save(f'lns_out_{n}.npy', b[1])
