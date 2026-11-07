"""exact search over {0deg corner-anchored grids} + {45deg lattice} packings via max independent set (ILP).
usage: lattice.py K S ng   -> scans ng*ng lattice offsets, prints best count (target K*K+1).  env NPROC, TL"""
import sys, os, math, itertools
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree
R2 = math.sqrt(2); H = 0.5 + R2 / 2; TOL = 1e-9

def cands(S, s, t):
    ups = set()
    for cx in (0, S):
        for cy in (0, S):
            sx = 1 if cx == 0 else -1; sy = 1 if cy == 0 else -1
            for i in range(int(S)):
                for j in range(int(S)):
                    x, y = cx + sx * (i + .5), cy + sy * (j + .5)
                    if .5 - 1e-12 <= x <= S - .5 + 1e-12 and .5 - 1e-12 <= y <= S - .5 + 1e-12: ups.add((round(x, 12), round(y, 12)))
    ups = np.array(sorted(ups)); u = np.array([1, 1]) / R2; v = np.array([-1, 1]) / R2; o = np.array([S / 2, S / 2]) + s * u + t * v
    d = []; m = int(S * R2) + 2
    for i in range(-m, m + 1):
        for j in range(-m, m + 1):
            p = o + i * u + j * v
            if R2 / 2 <= p[0] <= S - R2 / 2 and R2 / 2 <= p[1] <= S - R2 / 2: d.append(p)
    return ups, np.array(d)

def solve(S, s, t, tl=30):
    U, D = cands(S, s, t); nu, nd = len(U), len(D); N = nu + nd; E = []
    tu = cKDTree(U); 
    for i, j in tu.query_pairs(1 - TOL, p=np.inf, output_type='ndarray'): E.append((i, j))
    if nd:
        td = cKDTree(D)
        for i, lst in enumerate(td.query_ball_point(U, 2.0, p=np.inf)):
            for j in lst:
                dx, dy = U[i, 0] - D[j, 0], U[i, 1] - D[j, 1]
                if abs(dx) < H - TOL and abs(dy) < H - TOL and abs(dx + dy) / R2 < 0.5 + R2 / 2 * 1.0 + 0.0 - TOL + (0.5 * R2 - R2 / 2 + 0) and abs(dy - dx) / R2 < 1.2071067811865475 - TOL:
                    E.append((i, nu + j))
    E = np.array(E); r = np.repeat(np.arange(len(E)), 2); c = E.ravel()
    # clique rows: every sample point p lies in at most one chosen square
    h = float(os.environ.get('GRID', 0.1)); g = np.arange(h / 2, S, h); G = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2)
    rr, cc = [], []
    for k in range(N):
        if k < nu:
            cx, cy = U[k]; m = (np.abs(G[:, 0] - cx) < .5 - 1e-6) & (np.abs(G[:, 1] - cy) < .5 - 1e-6)
        else:
            cx, cy = D[k - nu]; dx, dy = G[:, 0] - cx, G[:, 1] - cy; m = (np.abs(dx + dy) / R2 < .5 - 1e-6) & (np.abs(dy - dx) / R2 < .5 - 1e-6)
        idx = np.nonzero(m)[0]; rr.append(idx); cc.append(np.full(len(idx), k))
    rr = np.concatenate(rr); cc = np.concatenate(cc); ne = len(E)
    A = coo_matrix((np.ones(len(r) + len(rr)), (np.concatenate([r, ne + rr]), np.concatenate([c, cc]))), shape=(ne + len(G), N)).tocsr()
    res = milp(-np.ones(N), constraints=LinearConstraint(A, -np.inf, 1), integrality=np.ones(N), bounds=Bounds(0, 1), options={'time_limit': tl})
    if res.x is None: return 0, None
    x = res.x > .5; P = [(*U[i], 0.0) for i in range(nu) if x[i]] + [(*D[j], math.pi / 4) for j in range(nd) if x[nu + j]]
    return int(x.sum()), np.array(P)

def job(a):
    S, s, t, tl = a; k, P = solve(S, s, t, tl); return k, s, t, P

if __name__ == '__main__':
    K, S, ng = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]); tl = int(os.environ.get('TL', 30))
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(S, a / ng, b / ng, tl) for a in range(ng) for b in range(ng)]; best = (0,)
    with ProcessPoolExecutor(int(os.environ.get('NPROC', 8))) as ex:
        for k, s, t, P in ex.map(job, jobs):
            print(k, round(s, 3), round(t, 3), flush=True)
            if k > best[0]: best = (k, s, t, P)
    print('BEST', best[:3], 'target', K * K + 1)
    if best[0] >= K * K + 1: np.save(f'lat_{K}_{S}.npy', best[3])
