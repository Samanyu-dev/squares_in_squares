"""Delete-and-squeeze: take the n+1 packing, delete one low-contact square, shrink below our incumbent L(n), repair near boundary + hole."""
import sys, math, numpy as np, time, csv
import sq
from lns import anneal_mask, load
from sq import normalize, side
def main(n, path, Linc, per=6, iters=1500000):
    P1 = normalize(load(path, n + 1)); L1 = side(P1)
    if L1 - Linc > 0.03: return None
    Lt = Linc - 1e-4
    c = P1[:, :2]
    d = np.hypot(c[:, None, 0] - c[None, :, 0], c[:, None, 1] - c[None, :, 1]); np.fill_diagonal(d, 9)
    deg = (d < 1.12).sum(1)
    order = np.argsort(deg + 0.001 * np.random.rand(len(deg)))
    h = .5 * (abs(np.cos(P1[:, 2])) + abs(np.sin(P1[:, 2])))
    cand = list(order[:per])
    for i in cand:
        P = np.delete(P1, i, axis=0); cc = P[:, :2]; hh = np.delete(h, i)
        viol = (P[:, 0] + hh > Lt) | (P[:, 1] + hh > Lt)
        mov = viol.copy()
        for j in np.where(viol)[0]: mov |= np.hypot(*(cc - cc[j]).T) < 3.0
        mov |= np.hypot(*(cc - c[i]).T) < 3.5
        for sd in range(2):
            Q = P.copy()
            e = anneal_mask(Q, mov, Lt, sq.S, iters, 0.15, 3e-4, 7 * sd + int(i) + n)
            if e < 1e-22 and side(Q) <= Lt + 1e-9: return Lt, Q
    return None
if __name__ == '__main__':
    n = int(sys.argv[1]); Linc = float(sys.argv[2]); t = time.time()
    r = main(n, sys.argv[3], Linc)
    print(n, 'Linc', round(Linc, 6), 'IMPROVED' if r else 'no', round(time.time() - t), 's', flush=True)
    if r: np.save(f'lns2_out_{n}.npy', r[1])
