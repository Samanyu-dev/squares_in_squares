"""K^2+1 band search: a x b rectangle of 45-degree squares along the box diagonal (centre shifted by c along the other diagonal) +
upright squares greedily anchored at the 4 corners.  usage: band.py K S1 S2 ...  -> max capacity per (a,b,c), flags capacity >= K^2+1"""
import sys, math, numpy as np
sys.path.insert(0, 'progs'); from common import free
from concurrent.futures import ProcessPoolExecutor
def cap(args):
    S, a, b, c = args; t = math.pi / 4; ctr = S / 2; d = np.array([1., 1.]) / math.sqrt(2); dv = np.array([-1., 1.]) / math.sqrt(2); pl = []
    for i in range(a):
        for j in range(b):
            p = np.array([ctr, ctr]) + (i - (a - 1) / 2 + c) * dv + (j - (b - 1) / 2) * d; pl.append((p[0], p[1], t))
    h = .5 * math.sqrt(2)
    if any(p[0] < h - 1e-9 or p[0] > S - h + 1e-9 or p[1] < h - 1e-9 or p[1] > S - h + 1e-9 for p in pl): return (S, a, b, c, -1)
    K = int(S); cand = []
    for cx in (0, 1):
        for cy in (0, 1):
            for i in range(K):
                for j in range(K): cand.append((.5 + i if cx == 0 else S - .5 - i, .5 + j if cy == 0 else S - .5 - j, 0.0))
    cand.sort(key=lambda q: -(abs(q[0] - ctr) + abs(q[1] - ctr)))
    for q in cand:
        if .5 <= q[0] <= S - .5 and .5 <= q[1] <= S - .5 and free(q, pl): pl.append(q)
    return (S, a, b, c, len(pl))
if __name__ == '__main__':
    K = int(sys.argv[1]); Ss = [float(x) for x in sys.argv[2:]]; need = K * K + 1
    jobs = [(S, a, b, c) for S in Ss for a in range(2, 11) for b in range(a, int(K * 1.5)) for c in (0, 0.5, 0.25)]
    with ProcessPoolExecutor(6) as ex: res = list(ex.map(cap, jobs, chunksize=4))
    for S in Ss:
        r = [x for x in res if x[0] == S and x[4] > 0]; r.sort(key=lambda x: -x[4]); print('S', S, 'need', need, 'best', r[:4])
