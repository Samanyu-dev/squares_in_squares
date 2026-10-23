"""Rectangular-container hardening: pack n unit squares into Lx x Ly (gadget search)."""
import numpy as np, math
from numba import njit
from sq import S
from sq2 import CS, CAP, build
from hard import hpair
from chain import compress as _unused

@njit(cache=True)
def hwallR(x, y, t, Lx, Ly, rho, s):
    h = (0.5 * s - rho) * (abs(math.cos(t)) + abs(math.sin(t))) + rho
    e = 0.0
    v = h - x
    if v > 0: e += v * v
    v = x + h - Lx
    if v > 0: e += v * v
    v = h - y
    if v > 0: e += v * v
    v = y + h - Ly
    if v > 0: e += v * v
    return e

@njit(cache=True)
def e_cellR(P, i, x, y, t, Lx, Ly, rho, s, cnt, mem, nc):
    e = hwallR(x, y, t, Lx, Ly, rho, s)
    cx = min(nc - 1, max(0, int(x / CS))); cy = min(nc - 1, max(0, int(y / CS)))
    for a in range(max(0, cx - 1), min(nc, cx + 2)):
        for b in range(max(0, cy - 1), min(nc, cy + 2)):
            for k in range(min(cnt[a, b], CAP)):
                j = mem[a, b, k]
                if j != i: e += hpair(x, y, t, P[j, 0], P[j, 1], P[j, 2], rho, s)
    return e

@njit(cache=True)
def totalR(P, Lx, Ly, rho, s):
    n = P.shape[0]; e = 0.0
    for i in range(n):
        e += hwallR(P[i, 0], P[i, 1], P[i, 2], Lx, Ly, rho, s)
        for j in range(i + 1, n): e += hpair(P[i, 0], P[i, 1], P[i, 2], P[j, 0], P[j, 1], P[j, 2], rho, s)
    return e

@njit(cache=True)
def annealR(P, Lx, Ly, rho, s, iters, sig0, T0, seed):
    np.random.seed(seed)
    n = P.shape[0]; nc = int(max(Lx, Ly) / CS) + 2
    cnt, mem = build(P, max(Lx, Ly), nc)
    E = totalR(P, Lx, Ly, rho, s)
    for it in range(iters):
        if E < 1e-22: break
        f = 1.0 - it / iters; sig = sig0 * (0.02 + f); T = T0 * f * f
        i = np.random.randint(n)
        x = P[i, 0] + sig * np.random.randn(); y = P[i, 1] + sig * np.random.randn()
        t = P[i, 2] + 2.0 * sig * np.random.randn() if np.random.rand() < 0.5 else P[i, 2]
        old = e_cellR(P, i, P[i, 0], P[i, 1], P[i, 2], Lx, Ly, rho, s, cnt, mem, nc)
        new = e_cellR(P, i, x, y, t, Lx, Ly, rho, s, cnt, mem, nc)
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
    return totalR(P, Lx, Ly, rho, s)

def hardenR(n, Lx, Ly, iters, seed, stages=10, sig0=0.3, T0=2e-3):
    rng = np.random.default_rng(seed)
    P = np.c_[rng.uniform(.5, Lx - .5, n), rng.uniform(.5, Ly - .5, n), rng.uniform(0, 1.57, n)]
    for rho in np.linspace(0.5 * S, 0.0, stages + 1)[:-1]:
        annealR(P, Lx, Ly, float(rho), S, iters // stages, sig0, T0, int(rng.integers(1 << 30)))
    e = annealR(P, Lx, Ly, 0.0, S, iters // stages, 0.1, T0 * 0.3, int(rng.integers(1 << 30)))
    return P if e < 1e-22 else None

def compressR(P, w, h, f, iters, seed, d0=0.02, dmin=2e-4):
    """shrink the overhang f (box (w+f)x(h+f)) while staying feasible"""
    rng = np.random.default_rng(seed); d = d0; P = P.copy()
    while d > dmin:
        f2 = f * (1 - d); Q = P.copy(); Q[:, 0] *= (w + f2) / (w + f); Q[:, 1] *= (h + f2) / (h + f)
        e = annealR(Q, w + f2, h + f2, 0.0, S, iters, min(0.2, 6 * d), 0.0, int(rng.integers(1 << 30)))
        if e < 1e-22: P, f = Q, f2
        else: d *= 0.6
    return P, f
