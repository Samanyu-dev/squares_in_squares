import numpy as np, math
from numba import njit

S = 1.0 + 1e-7  # inflated side used while optimizing -> real unit squares keep a tiny gap


@njit(cache=True)
def pair(xi, yi, ti, xj, yj, tj, s):
    dx = xj - xi; dy = yj - yi
    if dx * dx + dy * dy >= 2.0 * s * s:
        return 0.0
    ci = math.cos(ti); si = math.sin(ti); cj = math.cos(tj); sj = math.sin(tj)
    best = 1e18
    for k in range(4):
        if k == 0: ax, ay = ci, si
        elif k == 1: ax, ay = -si, ci
        elif k == 2: ax, ay = cj, sj
        else: ax, ay = -sj, cj
        ri = 0.5 * s * (abs(ax * ci + ay * si) + abs(-ax * si + ay * ci))
        rj = 0.5 * s * (abs(ax * cj + ay * sj) + abs(-ax * sj + ay * cj))
        o = ri + rj - abs(dx * ax + dy * ay)
        if o <= 0.0:
            return 0.0
        if o < best: best = o
    return best * best


@njit(cache=True)
def wall(x, y, t, L, s):
    h = 0.5 * s * (abs(math.cos(t)) + abs(math.sin(t)))
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
def e_one(P, i, x, y, t, L, s):
    e = wall(x, y, t, L, s)
    for j in range(P.shape[0]):
        if j != i:
            e += pair(x, y, t, P[j, 0], P[j, 1], P[j, 2], s)
    return e


@njit(cache=True)
def total(P, L, s):
    n = P.shape[0]
    e = 0.0
    for i in range(n):
        e += wall(P[i, 0], P[i, 1], P[i, 2], L, s)
        for j in range(i + 1, n):
            e += pair(P[i, 0], P[i, 1], P[i, 2], P[j, 0], P[j, 1], P[j, 2], s)
    return e


@njit(cache=True)
def anneal(P, L, s, iters, sig0, T0, seed):
    """Minimise overlap energy by single-square random moves with annealing. Returns final energy."""
    np.random.seed(seed)
    n = P.shape[0]
    E = total(P, L, s)
    for it in range(iters):
        if E < 1e-22:
            break
        f = 1.0 - it / iters
        sig = sig0 * (0.02 + f)
        T = T0 * f * f
        i = np.random.randint(n)
        x = P[i, 0] + sig * np.random.randn()
        y = P[i, 1] + sig * np.random.randn()
        t = P[i, 2] + sig * 2.0 * np.random.randn() if np.random.rand() < 0.5 else P[i, 2]
        old = e_one(P, i, P[i, 0], P[i, 1], P[i, 2], L, s)
        new = e_one(P, i, x, y, t, L, s)
        d = new - old
        if d < 0 or (T > 0 and np.random.rand() < math.exp(-d / T)):
            P[i, 0] = x; P[i, 1] = y; P[i, 2] = t
            E += d
    return total(P, L, s)


def grid(n):
    k = math.ceil(math.sqrt(n))
    P = np.zeros((n, 3))
    for i in range(n):
        P[i, 0] = 0.5 + i % k; P[i, 1] = 0.5 + i // k
    return P, float(k)


def fit(n, L, tries=20, iters=400000, seed=0, P0=None):
    """Try to pack n squares in side L. Returns P or None."""
    rng = np.random.default_rng(seed)
    for r in range(tries):
        if P0 is not None and r == 0:
            P = P0 * np.array([L / P0_L(P0), L / P0_L(P0), 1.0]) if False else P0.copy()
        else:
            P = np.c_[rng.uniform(0.5, L - 0.5, n), rng.uniform(0.5, L - 0.5, n), rng.uniform(0, math.pi / 2, n)]
        e = anneal(P, L, S, iters, 0.3, 1e-3, int(rng.integers(1 << 30)))
        if e < 1e-22:
            return P
    return None


def side(P):
    c = np.cos(P[:, 2]); s = np.sin(P[:, 2]); h = 0.5 * (abs(c) + abs(s))
    return max((P[:, 0] + h).max(), (P[:, 1] + h).max()) - min((P[:, 0] - h).min(), (P[:, 1] - h).min())


def normalize(P):
    c = np.cos(P[:, 2]); s = np.sin(P[:, 2]); h = 0.5 * (abs(c) + abs(s))
    P = P.copy()
    P[:, 0] -= (P[:, 0] - h).min(); P[:, 1] -= (P[:, 1] - h).min()
    return P


def valid(P, tol=0.0):
    """Exact-ish SAT check with real unit squares (s=1)."""
    n = len(P)
    return total(P, side(P) + 1e-9, 1.0 - tol) == 0.0


def shrink(n, L_hi, L_lo, **kw):
    """Bisect smallest L found feasible."""
    best = None
    for _ in range(12):
        L = 0.5 * (L_hi + L_lo)
        P = fit(n, L, **kw)
        if P is None:
            L_lo = L
        else:
            best, L_hi = P, L
    return best, L_hi
