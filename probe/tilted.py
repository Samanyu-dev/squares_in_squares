"""Feasibility probe: one block of r parallel tilted rods (K+1 unit squares each) + axis-aligned squares anchored to the left/right walls."""
import numpy as np, math, sys, time

def theta_for(K, L):
    # smallest theta with (K+1)cos(theta)+sin(theta) <= L   (rod of K+1 squares must fit in height L)
    lo, hi = 0.0, 1.2
    if (K + 1) <= L: return 0.0
    for _ in range(80):
        m = 0.5 * (lo + hi)
        if (K + 1) * math.cos(m) + math.sin(m) > L: lo = m
        else: hi = m
    return hi

def hull(P):
    P = sorted(map(tuple, P)); 
    def half(pts):
        h = []
        for p in pts:
            while len(h) >= 2 and (h[-1][0]-h[-2][0])*(p[1]-h[-2][1]) - (h[-1][1]-h[-2][1])*(p[0]-h[-2][0]) <= 0: h.pop()
            h.append(p)
        return h
    lo = half(P); up = half(P[::-1]); return np.array(lo[:-1] + up[:-1])

def build(K, f, r, x0):
    """returns (centres array (N,3)) or None if the block does not fit in the box"""
    L = K + f; th = theta_for(K, L)
    if th <= 0: return None
    c, s = math.cos(th), math.sin(th)
    if x0 < (K + 1) * s - 1e-12 or x0 + (r - 1) / c + c > L + 1e-12: return None
    sq = []
    for i in range(r):
        ox, oy = x0 + i / c, 0.0
        for j in range(K + 1):
            sq.append((ox + 0.5 * c - (j + 0.5) * s, oy + 0.5 * s + (j + 0.5) * c, th))
    sq = np.array(sq)
    # polygon hull of all rod corners
    cr = []
    for (x, y, t) in sq:
        for a, b in ((.5, .5), (-.5, .5), (-.5, -.5), (.5, -.5)):
            cr.append((x + a * c - b * s, y + a * s + b * c))
    H = hull(cr)
    if H[:, 0].min() < -1e-9 or H[:, 0].max() > L + 1e-9 or H[:, 1].min() < -1e-9 or H[:, 1].max() > L + 1e-9: return None
    edges = H[np.r_[1:len(H), 0]] - H; normals = np.c_[-edges[:, 1], edges[:, 0]]; normals /= np.linalg.norm(normals, axis=1)[:, None]
    def free(cx, cy):
        # axis square centred (cx,cy) side 1 vs convex hull H: separated along some axis?
        sep = np.zeros(len(cx), bool)
        for ax in (np.array([1.0, 0]), np.array([0, 1.0])):
            ph = H @ ax; sep |= (cx * ax[0] + cy * ax[1] - 0.5 >= ph.max() - 1e-9) | (cx * ax[0] + cy * ax[1] + 0.5 <= ph.min() + 1e-9)
        for nrm in normals:
            ph = H @ nrm; cc = cx * nrm[0] + cy * nrm[1]; rad = 0.5 * (abs(nrm[0]) + abs(nrm[1]))
            sep |= (cc - rad >= ph.max() - 1e-9) | (cc + rad <= ph.min() + 1e-9)
        return sep
    nx = int(L) + 1; ny = int(L)
    I, J = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij'); I = I.ravel(); J = J.ravel()
    lx = I + 0.5; ly = J + 0.5; ok = (lx + 0.5 <= L + 1e-9) & (ly + 0.5 <= L + 1e-9) & free(lx, ly)
    left = np.c_[lx[ok], ly[ok]]
    rx = L - I - 0.5; ry = J + 0.5; ok = (rx - 0.5 >= -1e-9) & (ry + 0.5 <= L + 1e-9) & free(rx, ry)
    right = np.c_[rx[ok], ry[ok]]
    keep = []
    for p in right:   # drop right squares overlapping a left one
        if len(left) and np.any((np.abs(left[:, 0] - p[0]) < 1 - 1e-9) & (np.abs(left[:, 1] - p[1]) < 1 - 1e-9)): continue
        keep.append(p)
    right = np.array(keep) if keep else np.zeros((0, 2))
    ax = np.vstack([np.c_[left, np.zeros(len(left))], np.c_[right, np.zeros(len(right))]])
    return np.vstack([sq, ax])

def best_for(K, f, rs=None, nx0=24):
    L = K + f; th = theta_for(K, L); best = (0, None)
    if th <= 0: return best
    c, s = math.cos(th), math.sin(th); lo = (K + 1) * s
    for r in (rs or range(1, int((L - lo) * c) + 1)):
        hi = L - (r - 1) / c - c
        if hi < lo: continue
        for x0 in np.linspace(lo, hi, nx0 if hi > lo else 1):
            P = build(K, f, r, x0)
            if P is not None and len(P) > best[0]: best = (len(P), (r, x0, th))
    return best

if __name__ == '__main__':
    K = int(sys.argv[1]); t0 = time.time()
    print(f'K={K}: need n=K^2+D; table f for D=1 is about 0.52')
    for f in [0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9]:
        N, par = best_for(K, f, nx0=10)
        print(f'  f={f:.2f}  L={K+f:.2f}  best squares found: {N}  (= K^2 {N - K*K:+d})  params={None if par is None else (par[0], round(par[1],2), round(par[2],3))}  [{time.time()-t0:.0f}s]', flush=True)
