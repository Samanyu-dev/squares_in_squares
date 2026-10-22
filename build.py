"""Single pipeline: packings pickle -> post-processed, strictly verified submission.csv"""
import pickle, math, sys, numpy as np
from sq import grid, side, normalize

DELTA = 1e-6      # relative spread applied to non-grid packings -> guaranteed real gap
MIN_GAP = 5e-7    # required SAT gap between any two squares in non-grid puzzles

def bbox_side(P):
    c, s = np.cos(P[:, 2]), np.sin(P[:, 2]); h = .5 * (abs(c) + abs(s))
    return max((P[:, 0] + h).max() - (P[:, 0] - h).min(), (P[:, 1] + h).max() - (P[:, 1] - h).min())

def min_gap(P):
    """Smallest SAT separation over all pairs (negative = overlap). Independent of the optimiser code."""
    n = len(P)
    if n < 2: return 1e9
    x, y, t = P.T; i, j = np.triu_indices(n, 1)
    dx, dy = x[j] - x[i], y[j] - y[i]
    m = dx * dx + dy * dy < 2.0 + 1e-6
    i, j, dx, dy = i[m], j[m], dx[m], dy[m]
    if len(i) == 0: return 1e9
    mn = np.full(len(i), 1e18)
    for k in range(4):
        a = (t[i] if k < 2 else t[j]) + (math.pi / 2 if k % 2 else 0); ax, ay = np.cos(a), np.sin(a); r = 0
        for q in (i, j):
            r = r + .5 * (abs(ax * np.cos(t[q]) + ay * np.sin(t[q])) + abs(-ax * np.sin(t[q]) + ay * np.cos(t[q])))
        mn = np.minimum(mn, r - abs(dx * ax + dy * ay))
    return -mn.max()

def finalize(n, P, L):
    """Return (P, L) ready to write. Grid-equal packings stay exact; others get spread + verified."""
    g = grid(n)
    if L >= math.ceil(math.sqrt(n)) - 1e-9 or n == 1:
        return g
    P = normalize(P.copy()); P[:, 2] = np.mod(P[:, 2], math.pi / 2)
    P[:, :2] *= (1 + DELTA); P = normalize(P); P = np.round(P, 13)
    L2 = bbox_side(P)
    if len(P) != n or not np.isfinite(P).all() or min_gap(P) < MIN_GAP or L2 >= math.ceil(math.sqrt(n)) - 1e-9:
        return g
    return P, L2

def write(best, out):
    rows = []; sc = 0; fall = 0
    for n in range(1, 1001):
        P, L = finalize(n, *best[n])
        fall += (L >= math.ceil(math.sqrt(n)) - 1e-9)
        sc += bbox_side(P) / math.sqrt(n) - 1
        for i, (x, y, t) in enumerate(P):
            rows.append(f'{len(rows)},{n},{i},{float(x)!r},{float(y)!r},{float(t)!r}')
    open(out, 'w').write('id,n,square_id,x,y,angle\n' + '\n'.join(rows) + '\n')
    print(f'wrote {out}: {len(rows)} rows, score {sc:.5f}, grid-equal puzzles {fall}')

if __name__ == '__main__':
    write(pickle.load(open(sys.argv[1], 'rb')), sys.argv[2])
