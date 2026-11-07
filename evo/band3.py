"""band seeds at angle th (31..59) in a slightly larger box, capacity >= n, then fq quench; reports best side vs ours.  usage: band3.py n S"""
import sys, os, math, json, numpy as np
sys.path.insert(0, 'progs'); sys.path.insert(0, '../stretch')
import band2, run
from concurrent.futures import ProcessPoolExecutor
def seed(S, a, b, c, th):
    t = math.radians(th); ctr = S / 2; d = np.array([math.cos(t), math.sin(t)]); dv = np.array([-math.sin(t), math.cos(t)]); pl = []
    from common import free
    for i in range(a):
        for j in range(b):
            p = np.array([ctr, ctr]) + (i - (a - 1) / 2 + c) * dv + (j - (b - 1) / 2) * d; pl.append((p[0], p[1], t))
    h = .5 * (abs(math.cos(t)) + abs(math.sin(t)))
    if any(p[0] < h - 1e-9 or p[0] > S - h + 1e-9 or p[1] < h - 1e-9 or p[1] > S - h + 1e-9 for p in pl): return None
    K = int(S); cand = []
    for cx in (0, 1):
        for cy in (0, 1):
            for i in range(K):
                for j in range(K): cand.append((.5 + i if cx == 0 else S - .5 - i, .5 + j if cy == 0 else S - .5 - j, 0.0))
    cand.sort(key=lambda q: -(abs(q[0] - ctr) + abs(q[1] - ctr)))
    for q in cand:
        if .5 <= q[0] <= S - .5 and .5 <= q[1] <= S - .5 and free(q, pl): pl.append(q)
    return np.array(pl)
def job(a):
    n, S, aa, b, c, th = a; P = seed(S, aa, b, c, th)
    if P is None or len(P) < n: return None
    P = P[:n]; wd = f'w3/{n}_{aa}_{b}_{c}_{th}'; os.makedirs(wd, exist_ok=True)
    q = run.quench(P, 80, 'q', wd)
    if q is None or run.overlaps(q): return (aa, b, c, th, 'fail')
    L = run.side(q); 
    if L < bl[n] - 1e-9: np.save(f'win_{n}_band3.npy', q)
    return (aa, b, c, th, round(L, 5), round(bl[n], 5))
bl = {int(k): v for k, v in json.load(open('../watch/best_L.json')).items()}
if __name__ == '__main__':
    n, S = int(sys.argv[1]), float(sys.argv[2])
    jobs = [(n, S, a, b, c, th) for th in range(33, 58, 2) for a in range(4, 9) for b in range(8, 26) for c in (0, 0.5)]
    with ProcessPoolExecutor(8) as ex: res = [r for r in ex.map(job, jobs, chunksize=2) if r]
    ok = sorted([r for r in res if r[4] != 'fail'], key=lambda r: r[4]); print(len(res), 'seeds >= n;', 'best', ok[:6])
