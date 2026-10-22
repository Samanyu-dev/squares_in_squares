"""Stage A: for every donor packing, 8 symmetries, every (w,h): min overhang f for each surplus dm (order statistic)."""
import numpy as np, pandas as pd, math, sys, time
from numba import njit, prange
W = 31; DM = 63
d = pd.read_csv('merged2.csv')
donors = []
for n, g in d.groupby('n'):
    n = int(n)
    if n < 5: continue
    P = g.sort_values('square_id')[['x', 'y', 'angle']].to_numpy(float)
    c, s = np.cos(P[:, 2]), np.sin(P[:, 2]); h = .5 * (abs(c) + abs(s))
    lo = min((P[:, 0] - h).min(), (P[:, 1] - h).min())
    donors.append((n, P[:, 0] - lo, P[:, 1] - lo, h))
Lmax = 1001
nd = len(donors)
off = np.zeros(nd + 1, np.int64); off[1:] = np.cumsum([len(x[1]) for x in donors])
X = np.concatenate([x[1] for x in donors]); Y = np.concatenate([x[2] for x in donors]); H = np.concatenate([x[3] for x in donors])
@njit(parallel=True, cache=True)
def run(X, Y, H, off, W, DM):
    nd = len(off) - 1
    F = np.full((nd, W + 1, W + 1, DM + 1), 1e9)
    for di in prange(nd):
        a, b = off[di], off[di + 1]; n = b - a
        x = X[a:b]; y = Y[a:b]; h = H[a:b]
        L = max((x + h).max(), (y + h).max())
        for sym in range(8):
            # symmetry: flip x, flip y, transpose
            xs = L - x if (sym & 1) else x
            ys = L - y if (sym & 2) else y
            if sym & 4: xs, ys = ys, xs
            xhi = xs + h; yhi = ys + h
            for w in range(1, W + 1):
                for hh in range(1, W + 1):
                    wh = w * hh
                    if wh >= n: continue
                    r = np.maximum(0.0, np.maximum(xhi - w, yhi - hh))
                    r.sort()
                    for dm in range(1, DM + 1):
                        k = wh + dm
                        if k > n: break
                        if r[k - 1] < F[di, w, hh, dm]: F[di, w, hh, dm] = r[k - 1]
    return F
t = time.time(); Fd = run(X, Y, H, off, W, DM); print('per-donor catalog', Fd.shape, round(time.time() - t), 's', flush=True)
F = Fd.min(axis=0); arg = Fd.argmin(axis=0).astype(np.int16)
np.save('fault/F.npy', F); np.save('fault/argdonor.npy', arg); np.save('fault/donor_n.npy', np.array([x[0] for x in donors]))
print('finite gadgets:', int((F < 1e8).sum()), '| smallest f for (w,h,dm)=(2,2,1):', F[2, 2, 1], ' (5,5,1):', F[5, 5, 1])
