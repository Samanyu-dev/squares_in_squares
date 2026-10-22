"""GPU hardening annealer (numba.cuda). One thread block = one independent chain."""
import math, numpy as np
from numba import cuda, float32

import os
BD = int(os.environ.get("SQ_BD", 64))
BD2 = 2 * BD
S_GPU = 1.0 + 1.5e-4     # inflated side used on GPU (float32 safe margin)
S_CHK = 1.0 + 5e-5       # float64 feasibility check afterwards

@cuda.jit(device=True)
def rnd(st, b):
    """xorshift32 uniform in (0,1); state kept in int64 array with explicit 32-bit masking (identical in simulator and on GPU)"""
    x = st[b]
    x = x ^ ((x << 13) & 4294967295)
    x = x ^ (x >> 17)
    x = x ^ ((x << 5) & 4294967295)
    st[b] = x
    return (x + 0.5) / 4294967296.0

@cuda.jit(device=True)
def rndn(st, b):
    u1 = rnd(st, b); u2 = rnd(st, b)
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(6.283185307179586 * u2)

@cuda.jit(device=True)
def _dpseg(px, py, ax, ay, bx, by):
    dx = bx - ax; dy = by - ay
    u = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy + 1e-20)
    u = min(1.0, max(0.0, u))
    qx = ax + u * dx - px; qy = ay + u * dy - py
    return qx * qx + qy * qy

@cuda.jit(device=True)
def dpair(xi, yi, ti, xj, yj, tj, rho, s):
    a = 0.5 * s - rho
    dx = xj - xi; dy = yj - yi
    reach = 2.0 * (a * 1.4142136 + rho)
    if dx * dx + dy * dy >= reach * reach:
        return 0.0
    ci = math.cos(ti); si = math.sin(ti); cj = math.cos(tj); sj = math.sin(tj)
    g = -1e18
    for k in range(4):
        if k == 0: ax = ci; ay = si
        elif k == 1: ax = -si; ay = ci
        elif k == 2: ax = cj; ay = sj
        else: ax = -sj; ay = cj
        ri = a * (abs(ax * ci + ay * si) + abs(-ax * si + ay * ci))
        rj = a * (abs(ax * cj + ay * sj) + abs(-ax * sj + ay * cj))
        gg = abs(dx * ax + dy * ay) - ri - rj
        if gg > g: g = gg
    if g <= 0.0:
        d = 2.0 * rho - g
        return d * d
    if g >= 2.0 * rho:
        return 0.0
    if a <= 1e-6:
        dist = math.sqrt(dx * dx + dy * dy)
    else:
        vix = cuda.local.array(4, float32); viy = cuda.local.array(4, float32)
        vjx = cuda.local.array(4, float32); vjy = cuda.local.array(4, float32)
        for k in range(4):
            sx = -a if (k == 0 or k == 3) else a
            sy = -a if k < 2 else a
            vix[k] = xi + sx * ci - sy * si; viy[k] = yi + sx * si + sy * ci
            vjx[k] = xj + sx * cj - sy * sj; vjy[k] = yj + sx * sj + sy * cj
        m = 1e18
        for p in range(4):
            for e in range(4):
                e2 = (e + 1) % 4
                m = min(m, _dpseg(vix[p], viy[p], vjx[e], vjy[e], vjx[e2], vjy[e2]))
                m = min(m, _dpseg(vjx[p], vjy[p], vix[e], viy[e], vix[e2], viy[e2]))
        dist = math.sqrt(m)
    d = 2.0 * rho - dist
    return d * d if d > 0 else 0.0

@cuda.jit(device=True)
def dwall(x, y, t, L, rho, s):
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

@cuda.jit
def kern(P, n, L, rho, s, iters, sig0, T0, states):
    b = cuda.blockIdx.x; tid = cuda.threadIdx.x
    sh = cuda.shared.array(8, float32)
    red = cuda.shared.array(BD2, float32)   # one buffer: [0,BD) old-energy partials, [BD,2BD) new-energy partials
    for it in range(iters):
        if tid == 0:
            f = 1.0 - it / iters; sig = sig0 * (0.02 + f); T = T0 * f * f
            i = int(rnd(states, b) * n)
            if i >= n: i = n - 1
            x = P[b, i, 0] + sig * rndn(states, b)
            y = P[b, i, 1] + sig * rndn(states, b)
            t = P[b, i, 2]
            if rnd(states, b) < 0.5:
                t = t + 2.0 * sig * rndn(states, b)
            sh[0] = i; sh[1] = x; sh[2] = y; sh[3] = t; sh[4] = T
            sh[5] = rnd(states, b)
        cuda.syncthreads()
        i = int(sh[0]); x = sh[1]; y = sh[2]; t = sh[3]
        xo = P[b, i, 0]; yo = P[b, i, 1]; to = P[b, i, 2]
        eo = 0.0; en = 0.0
        for j in range(tid, n, BD):
            if j != i:
                xj = P[b, j, 0]; yj = P[b, j, 1]; tj = P[b, j, 2]
                eo += dpair(xo, yo, to, xj, yj, tj, rho, s)
                en += dpair(x, y, t, xj, yj, tj, rho, s)
        if tid == 0:
            eo += dwall(xo, yo, to, L, rho, s); en += dwall(x, y, t, L, rho, s)
        red[tid] = eo; red[BD + tid] = en
        cuda.syncthreads()
        st = BD // 2
        while st > 0:
            if tid < st:
                red[tid] += red[tid + st]; red[BD + tid] += red[BD + tid + st]
            cuda.syncthreads()
            st //= 2
        if tid == 0:
            d = red[BD] - red[0]; T = sh[4]
            if d < 0 or (T > 0 and sh[5] < math.exp(-d / T)):
                P[b, i, 0] = x; P[b, i, 1] = y; P[b, i, 2] = t
        cuda.syncthreads()

def harden_gpu(n, L, chains, iters, seed, stages=10, sig0=0.3, T0=2e-3):
    """Run `chains` independent hardening chains at container L. Returns float64 array (chains,n,3)."""
    rng = np.random.default_rng(seed)
    P0 = np.stack([np.c_[rng.uniform(.5, L - .5, n), rng.uniform(.5, L - .5, n), rng.uniform(0, 1.57, n)] for _ in range(chains)]).astype(np.float32)
    dP = cuda.to_device(P0)
    states = cuda.to_device(((np.arange(chains, dtype=np.int64) + 1) * 2654435761 + int(seed) * 40503 + 12345) % 4294967295 + 1)
    for _ in range(3): pass
    per = max(1, iters // stages)
    for rho in list(np.linspace(0.5 * S_GPU, 0.0, stages + 1)[:-1]) + [0.0]:
        kern[chains, BD](dP, n, np.float32(L), np.float32(rho), np.float32(S_GPU), per, np.float32(sig0 if rho > 0 else 0.1), np.float32(T0 if rho > 0 else T0 * 0.3), states)
    cuda.synchronize()
    return dP.copy_to_host().astype(np.float64)
