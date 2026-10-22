"""Rebuild actual packings for the puzzles where the chain DP beats merged2, and write them as CSV."""
import numpy as np, pandas as pd, math, pickle, sys
import build
W = 31; DM = 63
G_, out = None, None
st = pickle.load(open('fault/dp2b.pkl', 'rb')); G = st['G']; dp, pc, pr, pd_, pb, pg, pm = st['out']
F4 = np.load('fault/F4b.npy'); D4 = np.load('fault/D4b.npy'); S4 = np.load('fault/S4b.npy')
# donors in the same order as catalog2b
d = pd.read_csv('merged2.csv'); donors = []
for n, g in d.groupby('n'):
    n = int(n)
    if n < 5: continue
    P = g.sort_values('square_id')[['x', 'y', 'angle']].to_numpy(float); donors.append((n, P))
big = np.load('fault/big_donors.npz')
for k in sorted(big.files, key=int): donors.append((int(k), big[k]))
def norm(P):
    c, s = np.cos(P[:, 2]), np.sin(P[:, 2]); h = .5 * (abs(c) + abs(s)); lo = min((P[:, 0] - h).min(), (P[:, 1] - h).min())
    return P[:, 0] - lo, P[:, 1] - lo, P[:, 2], h
def gadget_squares(di, sym, w, hh, dm):
    n, P = donors[di]; x, y, a, h = norm(P); L = max((x + h).max(), (y + h).max())
    xs = L - x if (sym & 1) else x; ys = L - y if (sym & 2) else y
    if sym & 4: xs, ys = ys, xs
    if bin(sym).count('1') % 2: a = -a
    r = np.maximum(0.0, np.maximum(xs + h - w, ys + h - hh)); idx = np.argsort(r, kind='stable')[:w * hh + dm]
    return np.c_[xs[idx], ys[idx], a[idx]], float(r[idx].max())
def backtrack(state):
    c, r, dmv, b = (int(v) for v in state); chain = []
    while True:
        g = int(pg[c, r, dmv, b]); mode = int(pm[c, r, dmv, b]); chain.append((g, mode))
        if mode == 9: break
        c, r, dmv, b = int(pc[c, r, dmv, b]), int(pr[c, r, dmv, b]), int(pd_[c, r, dmv, b]), int(pb[c, r, dmv, b])
    return chain[::-1]
def construct(n, K, state):
    chain = backtrack(state); gad = [(int(G[g, 0]), int(G[g, 1]), int(G[g, 2]), int(G[g, 3])) for g, _ in chain]; modes = [m for _, m in chain]
    parts = []; fs = []
    for (w, hh, dm, v) in gad:
        di = int(D4[v, w, hh, dm]); sym = int(S4[v, w, hh, dm]); sq, f = gadget_squares(di, sym, w, hh, dm); parts.append(sq); fs.append(f)
    f = max(fs); M = len(gad); ox = [0] * M; oy = [0] * M; oy[0] = K - gad[0][1]
    for m in range(1, M):
        w0, h0 = gad[m - 1][0], gad[m - 1][1]; hm = gad[m][1]
        if modes[m] == 0: ox[m] = ox[m - 1] + w0; oy[m] = oy[m - 1] - hm - 1
        elif modes[m] == 1: ox[m] = ox[m - 1] + w0 + 1; oy[m] = oy[m - 1] - hm
        else: ox[m] = ox[m - 1] + w0; oy[m] = oy[m - 1] - hm
    if min(oy) < 0 or max(ox[m] + gad[m][0] for m in range(M)) > K: return None
    fault = [None] * K
    for m in range(M):
        for i in range(ox[m], ox[m] + gad[m][0]): fault[i] = oy[m]
        if m >= 1 and modes[m] == 1: fault[ox[m] - 1] = oy[m - 1]
    last = oy[-1]
    for i in range(K):
        if fault[i] is None: fault[i] = last
    occ = np.zeros((K, K), bool)
    for m in range(M): occ[ox[m]:ox[m] + gad[m][0], oy[m]:oy[m] + gad[m][1]] = True
    sq = []
    for i in range(K):
        for j in range(K):
            if occ[i, j]: continue
            sq.append((i + .5, j + .5, 0.0) if j < fault[i] else (i + .5 + f, j + .5 + f, 0.0))
    for m in range(M):
        p = parts[m].copy(); p[:, 0] += ox[m]; p[:, 1] += oy[m]; sq.extend(map(tuple, p))
    P = np.array(sq)
    if len(P) < n: return None
    return P[:n], K + f
tab = pd.read_pickle('fault/dp2b_vs_merged2.pkl'); todo = tab[tab.adv > 1e-7].sort_values('adv', ascending=False)
rows = []; built = 0; failed = 0; SC = 1 + 1e-12
for _, r in todo.iterrows():
    n = int(r.n)
    try: res = construct(n, int(r.K), r.state)
    except Exception as e: res = None
    if res is None: failed += 1; continue
    P, Lt = res; P = P.copy()
    cx = (build.bbox_side(P)) / 2; P[:, :2] = cx + (P[:, :2] - cx) * SC           # tiny dilation about centre so contacts are strictly non-touching
    P = build.normalize(P) if hasattr(build, 'normalize') else P
    gap = build.min_gap(P); L = build.bbox_side(P)
    if len(P) != n or gap < -1e-12 or L >= r.merged2_L - 1e-9: failed += 1; print(f'  n={n}: rejected (gap {gap:.2e}, L {L:.6f} vs merged2 {r.merged2_L:.6f})'); continue
    built += 1; print(f'  n={n}: built L={L:.8f} merged2 {r.merged2_L:.8f} gain {(r.merged2_L - L) / math.sqrt(n):.2e} min_gap {gap:.2e}')
    for i, (x, y, t) in enumerate(P): rows.append(f'{len(rows)},{n},{i},{float(x)!r},{float(y)!r},{float(t)!r}')
open('fault_cands.csv', 'w').write('id,n,square_id,x,y,angle\n' + '\n'.join(rows) + '\n')
print('built', built, 'failed/rejected', failed, 'of', len(todo))
