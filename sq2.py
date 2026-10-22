import numpy as np, math
from numba import njit
from sq import pair, wall, S, grid, side, normalize, valid, total

CS = 1.5
CAP = 12

@njit(cache=True)
def build(P, L, nc):
    cnt = np.zeros((nc, nc), np.int32); mem = np.zeros((nc, nc, CAP), np.int32)
    for i in range(P.shape[0]):
        cx = min(nc - 1, max(0, int(P[i, 0] / CS))); cy = min(nc - 1, max(0, int(P[i, 1] / CS)))
        if cnt[cx, cy] < CAP:
            mem[cx, cy, cnt[cx, cy]] = i
        cnt[cx, cy] += 1
    return cnt, mem

@njit(cache=True)
def e_cell(P, i, x, y, t, L, s, cnt, mem, nc):
    e = wall(x, y, t, L, s)
    cx = min(nc - 1, max(0, int(x / CS))); cy = min(nc - 1, max(0, int(y / CS)))
    for a in range(max(0, cx - 1), min(nc, cx + 2)):
        for b in range(max(0, cy - 1), min(nc, cy + 2)):
            for k in range(min(cnt[a, b], CAP)):
                j = mem[a, b, k]
                if j != i:
                    e += pair(x, y, t, P[j, 0], P[j, 1], P[j, 2], s)
    return e

@njit(cache=True)
def anneal_fast(P, L, s, iters, sig0, T0, seed):
    np.random.seed(seed)
    n = P.shape[0]; nc = int(L / CS) + 2
    cnt, mem = build(P, L, nc)
    E = total(P, L, s)
    for it in range(iters):
        if E < 1e-22: break
        f = 1.0 - it / iters; sig = sig0 * (0.02 + f); T = T0 * f * f
        i = np.random.randint(n)
        x = P[i, 0] + sig * np.random.randn(); y = P[i, 1] + sig * np.random.randn()
        t = P[i, 2] + 2.0 * sig * np.random.randn() if np.random.rand() < 0.5 else P[i, 2]
        old = e_cell(P, i, P[i, 0], P[i, 1], P[i, 2], L, s, cnt, mem, nc)
        new = e_cell(P, i, x, y, t, L, s, cnt, mem, nc)
        d = new - old
        if d < 0 or (T > 0 and np.random.rand() < math.exp(-d / T)):
            ocx = min(nc - 1, max(0, int(P[i, 0] / CS))); ocy = min(nc - 1, max(0, int(P[i, 1] / CS)))
            ncx = min(nc - 1, max(0, int(x / CS))); ncy = min(nc - 1, max(0, int(y / CS)))
            if ocx != ncx or ocy != ncy:
                if cnt[ncx, ncy] >= CAP: continue
                for k in range(cnt[ocx, ocy]):
                    if mem[ocx, ocy, k] == i:
                        mem[ocx, ocy, k] = mem[ocx, ocy, cnt[ocx, ocy] - 1]; break
                cnt[ocx, ocy] -= 1
                mem[ncx, ncy, cnt[ncx, ncy]] = i; cnt[ncx, ncy] += 1
            P[i, 0] = x; P[i, 1] = y; P[i, 2] = t
            E += d
    return total(P, L, s)
