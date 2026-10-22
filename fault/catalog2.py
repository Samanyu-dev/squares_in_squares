"""Gadget catalog with tip flags. Variants v: bit0 = top-left tip free, bit1 = bottom-right tip free.
F[v,w,h,dm] = min overhang f among gadgets (donor, symmetry, crop) having at least those free tips; provenance stored."""
import numpy as np, pandas as pd, sys, time
from numba import njit, prange
W = 31; DM = 63; TOL = 1e-9
d = pd.read_csv('merged2.csv'); donors = []
for n, g in d.groupby('n'):
    n = int(n)
    if n < 5: continue
    P = g.sort_values('square_id')[['x', 'y', 'angle']].to_numpy(float)
    c, s = np.cos(P[:, 2]), np.sin(P[:, 2]); h = .5 * (abs(c) + abs(s))
    lo = min((P[:, 0] - h).min(), (P[:, 1] - h).min())
    donors.append((n, P[:, 0] - lo, P[:, 1] - lo, h))
nd = len(donors)
off = np.zeros(nd + 1, np.int64); off[1:] = np.cumsum([len(x[1]) for x in donors])
X = np.concatenate([x[1] for x in donors]); Y = np.concatenate([x[2] for x in donors]); H = np.concatenate([x[3] for x in donors])
@njit(parallel=True, cache=True)
def batch(X, Y, H, off, d0, d1, W, DM, TOL):
    nb = d1 - d0
    F = np.full((nb, 4, W + 1, W + 1, DM + 1), 1e9); S = np.zeros((nb, 4, W + 1, W + 1, DM + 1), np.int8)
    for bi in prange(nb):
        di = d0 + bi; a, b = off[di], off[di + 1]; n = b - a
        x = X[a:b]; y = Y[a:b]; h = H[a:b]; L = max((x + h).max(), (y + h).max())
        for sym in range(8):
            xs = L - x if (sym & 1) else x
            ys = L - y if (sym & 2) else y
            if sym & 4: xs, ys = ys, xs
            xhi = xs + h; yhi = ys + h; xlo = xs - h; ylo = ys - h
            for w in range(1, W + 1):
                for hh in range(1, W + 1):
                    wh = w * hh
                    if wh >= n: continue
                    r = np.maximum(0.0, np.maximum(xhi - w, yhi - hh))
                    idx = np.argsort(r)
                    mbr = 1e9; mtl = 1e9
                    kmax = min(n, wh + DM)
                    for k in range(1, kmax + 1):
                        i = idx[k - 1]
                        if xhi[i] > w + TOL and ylo[i] < mbr: mbr = ylo[i]
                        if yhi[i] > hh + TOL and xlo[i] < mtl: mtl = xlo[i]
                        dm = k - wh
                        if dm < 1: continue
                        f = r[i]
                        br = mbr >= f - TOL; tl = mtl >= f - TOL
                        for v in range(4):
                            if ((v & 1) and not tl) or ((v & 2) and not br): continue
                            if f < F[bi, v, w, hh, dm]:
                                F[bi, v, w, hh, dm] = f; S[bi, v, w, hh, dm] = sym
    return F, S
Fb = np.full((4, W + 1, W + 1, DM + 1), 1e9); Db = np.zeros((4, W + 1, W + 1, DM + 1), np.int16); Sb = np.zeros((4, W + 1, W + 1, DM + 1), np.int8)
t = time.time()
for d0 in range(0, nd, 40):
    d1 = min(nd, d0 + 40)
    F, S = batch(X, Y, H, off, d0, d1, W, DM, TOL)
    am = F.argmin(axis=0); mn = F.min(axis=0)
    better = mn < Fb
    Db[better] = (am + d0)[better]; Sb[better] = np.take_along_axis(S, am[None], axis=0)[0][better]; Fb[better] = mn[better]
    print(f'donors {d1}/{nd} t={time.time() - t:.0f}s', flush=True)
np.save('fault/F4.npy', Fb); np.save('fault/D4.npy', Db); np.save('fault/S4.npy', Sb); np.save('fault/donor_n.npy', np.array([x[0] for x in donors]))
for v in range(4): print('variant', v, 'finite gadgets', int((Fb[v] < 1e8).sum()), ' f(5,5,1)=', round(Fb[v, 5, 5, 1], 6))
