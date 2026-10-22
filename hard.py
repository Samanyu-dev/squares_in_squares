import numpy as np, math
from numba import njit
from sq import pair, S
from sq2 import CS, CAP, build

@njit(cache=True)
def _pseg(px, py, ax, ay, bx, by):
    dx = bx - ax; dy = by - ay
    u = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy + 1e-300)
    u = min(1.0, max(0.0, u))
    qx = ax + u * dx - px; qy = ay + u * dy - py
    return qx * qx + qy * qy

@njit(cache=True)
def _verts(x, y, t, a):
    c = math.cos(t); s = math.sin(t)
    vx = np.empty(4); vy = np.empty(4)
    for k in range(4):
        sx = -a if (k == 0 or k == 3) else a
        sy = -a if k < 2 else a
        vx[k] = x + sx * c - sy * s; vy[k] = y + sx * s + sy * c
    return vx, vy

@njit(cache=True)
def hpair(xi, yi, ti, xj, yj, tj, rho, s):
    """rounded-square overlap depth^2 (rho=0.5*s disk ... rho=0 square); s scales the unit side."""
    a = 0.5 * s - rho
    dx = xj - xi; dy = yj - yi
    reach = 2.0 * (a * 1.4142136 + rho)
    if dx * dx + dy * dy >= reach * reach: return 0.0
    ci = math.cos(ti); si = math.sin(ti); cj = math.cos(tj); sj = math.sin(tj)
    g = -1e18
    for k in range(4):
        if k == 0: ax, ay = ci, si
        elif k == 1: ax, ay = -si, ci
        elif k == 2: ax, ay = cj, sj
        else: ax, ay = -sj, cj
        ri = a * (abs(ax * ci + ay * si) + abs(-ax * si + ay * ci))
        rj = a * (abs(ax * cj + ay * sj) + abs(-ax * sj + ay * cj))
        gg = abs(dx * ax + dy * ay) - ri - rj
        if gg > g: g = gg
    if g <= 0.0:
        d = 2.0 * rho - g
        return d * d
    if g >= 2.0 * rho: return 0.0
    if a <= 1e-12:
        dist = math.sqrt(dx * dx + dy * dy)
    else:
        vix, viy = _verts(xi, yi, ti, a); vjx, vjy = _verts(xj, yj, tj, a)
        m = 1e18
        for p in range(4):
            for e in range(4):
                e2 = (e + 1) % 4
                m = min(m, _pseg(vix[p], viy[p], vjx[e], vjy[e], vjx[e2], vjy[e2]))
                m = min(m, _pseg(vjx[p], vjy[p], vix[e], viy[e], vix[e2], viy[e2]))
        dist = math.sqrt(m)
    d = 2.0 * rho - dist
    return d * d if d > 0 else 0.0

@njit(cache=True)
def hwall(x, y, t, L, rho, s):
    h = (0.5 * s - rho) * (abs(math.cos(t)) + abs(math.sin(t))) + rho
    e = 0.0
    v = h - x
    if v > 0: e += v * v
    v = x + h - L
    if v > 0: e += v * v
    v = h - y
    if v > 0: e += v * v
    v = y + h - L
    if v > 0: e += v * v
    return e

@njit(cache=True)
def e_hcell(P, i, x, y, t, L, rho, s, cnt, mem, nc):
    e = hwall(x, y, t, L, rho, s)
    cx = min(nc - 1, max(0, int(x / CS))); cy = min(nc - 1, max(0, int(y / CS)))
    for a in range(max(0, cx - 1), min(nc, cx + 2)):
        for b in range(max(0, cy - 1), min(nc, cy + 2)):
            for k in range(min(cnt[a, b], CAP)):
                j = mem[a, b, k]
                if j != i: e += hpair(x, y, t, P[j, 0], P[j, 1], P[j, 2], rho, s)
    return e

@njit(cache=True)
def htotal(P, L, rho, s):
    n = P.shape[0]; e = 0.0
    for i in range(n):
        e += hwall(P[i, 0], P[i, 1], P[i, 2], L, rho, s)
        for j in range(i + 1, n): e += hpair(P[i, 0], P[i, 1], P[i, 2], P[j, 0], P[j, 1], P[j, 2], rho, s)
    return e

@njit(cache=True)
def hanneal(P, L, rho, s, iters, sig0, T0, seed):
    np.random.seed(seed)
    n = P.shape[0]; nc = int(L / CS) + 2
    cnt, mem = build(P, L, nc)
    E = htotal(P, L, rho, s)
    for it in range(iters):
        if E < 1e-22: break
        f = 1.0 - it / iters; sig = sig0 * (0.02 + f); T = T0 * f * f
        i = np.random.randint(n)
        x = P[i, 0] + sig * np.random.randn(); y = P[i, 1] + sig * np.random.randn()
        t = P[i, 2] + 2.0 * sig * np.random.randn() if np.random.rand() < 0.5 else P[i, 2]
        old = e_hcell(P, i, P[i, 0], P[i, 1], P[i, 2], L, rho, s, cnt, mem, nc)
        new = e_hcell(P, i, x, y, t, L, rho, s, cnt, mem, nc)
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
    return htotal(P, L, rho, s)

def harden(n, L, iters, seed, stages=10, sig0=0.3, T0=2e-3):
    """Disk-start, gradually harden to square at fixed L. Returns P or None."""
    rng = np.random.default_rng(seed)
    P = np.c_[rng.uniform(.5, L - .5, n), rng.uniform(.5, L - .5, n), rng.uniform(0, 1.57, n)]
    for rho in np.linspace(0.5 * S, 0.0, stages + 1)[:-1]:
        hanneal(P, L, float(rho), S, iters // stages, sig0, T0, int(rng.integers(1 << 30)))
    e = hanneal(P, L, 0.0, S, iters // stages, 0.1, T0 * 0.3, int(rng.integers(1 << 30)))
    return P if e < 1e-22 else None
