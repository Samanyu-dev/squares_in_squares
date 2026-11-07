"""central a x b diamond (45 deg, optional extra tilt th) + four upright corner-anchored grids, greedy; prints capacity vs n_needed."""
import sys, math, numpy as np
sys.path.insert(0, 'progs'); from common import free
def cap(S, a, b, th=45.0):
    t = math.radians(th); c, s = math.cos(t), math.sin(t); ctr = S / 2; pl = []
    for i in range(a):
        for j in range(b):
            u, v = i - (a - 1) / 2, j - (b - 1) / 2
            pl.append((ctr + u * c - v * s, ctr + u * s + v * c, t))
    h = .5 * (abs(c) + abs(s))
    if any(p[0] < h - 1e-9 or p[0] > S - h + 1e-9 or p[1] < h - 1e-9 or p[1] > S - h + 1e-9 for p in pl): return None
    K = int(S); cand = []
    for cx in (0, 1):
        for cy in (0, 1):
            for i in range(K):
                for j in range(K):
                    x = .5 + i if cx == 0 else S - .5 - i; y = .5 + j if cy == 0 else S - .5 - j
                    cand.append((x, y, 0.0))
    cand.sort(key=lambda q: -(abs(q[0] - ctr) + abs(q[1] - ctr)))
    for q in cand:
        if .5 <= q[0] <= S - .5 and .5 <= q[1] <= S - .5 and free(q, pl): pl.append(q)
    return pl
if __name__ == '__main__':
    for k in range(21, 33):
        nfirst = {21: 424, 22: 466, 23: 510, 24: 557, 25: 606, 26: 656, 27: 708, 28: 763, 29: 819, 30: 878, 31: 938, 32: 1000}[k]; best = (0, None)
        S = k - 0.03
        for a in range(2, 12):
            for b in range(a, 12):
                pl = cap(S, a, b)
                if pl and len(pl) > best[0]: best = (len(pl), (a, b))
        print(k, 'S', S, 'capacity', best[0], 'need', nfirst, 'diamond', best[1], flush=True)
