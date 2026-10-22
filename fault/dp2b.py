"""Chain DP with exact tip-aware seams (modes Y, X always; mode 0 if prev br-free or next tl-free). State (cu,ru,dm,b), b = last gadget br-free."""
import numpy as np, pandas as pd, math, time, pickle
from numba import njit
W = 31; DM = 63
F4 = np.load('fault/F4b.npy')
gl = []
for v in range(4):
    for w in range(1, W + 1):
        for h in range(1, W + 1):
            for dm in range(1, DM + 1):
                if F4[v, w, h, dm] < 0.9: gl.append((w, h, dm, v, F4[v, w, h, dm]))
G = np.array(gl); print('gadget variants with f<0.9:', len(G), flush=True)
@njit(cache=True)
def solve(G, W, DM):
    INF = 1e9; ng = G.shape[0]
    dp = np.full((W + 2, W + 2, DM + 1, 2), INF)
    pc = np.full((W + 2, W + 2, DM + 1, 2), -1, np.int16); pr = np.full((W + 2, W + 2, DM + 1, 2), -1, np.int16)
    pd_ = np.full((W + 2, W + 2, DM + 1, 2), -1, np.int16); pb = np.full((W + 2, W + 2, DM + 1, 2), -1, np.int8)
    pg = np.full((W + 2, W + 2, DM + 1, 2), -1, np.int32); pm = np.full((W + 2, W + 2, DM + 1, 2), -1, np.int8)
    for g in range(ng):
        w, h, dm, v, f = int(G[g, 0]), int(G[g, 1]), int(G[g, 2]), int(G[g, 3]), G[g, 4]
        if v & 1: continue
        b2 = (v >> 1) & 1
        if f < dp[w, h, dm, b2]:
            dp[w, h, dm, b2] = f; pg[w, h, dm, b2] = g; pm[w, h, dm, b2] = 9
    for s in range(2, 2 * W + 2):
        for cu in range(0, W + 1):
            ru = s - cu
            if ru < 0 or ru > W: continue
            for d0 in range(1, DM + 1):
                for b in range(2):
                    val = dp[cu, ru, d0, b]
                    if val >= INF: continue
                    for g in range(ng):
                        w = int(G[g, 0]); h = int(G[g, 1]); dm = int(G[g, 2]); v = int(G[g, 3]); f = G[g, 4]
                        if d0 + dm > DM: continue
                        nf = f if f > val else val
                        tl = v & 1; b2 = (v >> 1) & 1; dn = d0 + dm
                        for mode in range(3):
                            if mode == 0:
                                if tl: continue
                                c1 = cu + w; r1 = ru + h + 1            # Y
                            elif mode == 1:
                                if tl: continue
                                c1 = cu + w + 1; r1 = ru + h            # X
                            else:
                                if b == 0 and tl == 0: continue         # zero buffer: need prev br-free or next tl-free
                                c1 = cu + w; r1 = ru + h
                            if c1 <= W and r1 <= W and nf < dp[c1, r1, dn, b2]:
                                dp[c1, r1, dn, b2] = nf; pc[c1, r1, dn, b2] = cu; pr[c1, r1, dn, b2] = ru
                                pd_[c1, r1, dn, b2] = d0; pb[c1, r1, dn, b2] = b; pg[c1, r1, dn, b2] = g; pm[c1, r1, dn, b2] = mode
    return dp, pc, pr, pd_, pb, pg, pm
t = time.time(); out = solve(G, W, DM); print('dp done', round(time.time() - t), 's', flush=True)
dp, pc, pr, pd_, pb, pg, pm = out
pickle.dump({'G': G, 'out': out}, open('fault/dp2b.pkl', 'wb'))
tab = pd.read_pickle('table_merged2.pkl').set_index('n'); rows = []
for n in range(5, 1001):
    best = (1e9, None)
    for K in range(max(1, math.isqrt(n) - 2), math.isqrt(n) + 1):
        D = n - K * K
        if D < 1 or D > DM: continue
        sub = dp[:K + 1, :K + 1, D:, :]
        v = sub.min()
        if K + v < best[0]:
            ci, ri, di, bi = np.unravel_index(sub.argmin(), sub.shape); best = (K + v, K, (ci, ri, di + D, bi))
    if best[1] is not None: rows.append((n, best[1], best[0], tab.loc[n, 'L'], tab.loc[n, 'L'] - best[0], best[2]))
r = pd.DataFrame(rows, columns=['n', 'K', 'dp_L', 'merged2_L', 'adv', 'state']); r.to_pickle('fault/dp2b_vs_merged2.pkl')
print('exact tip-aware DP beats merged2 at', int((r.adv > 1e-6).sum()), 'puzzles | total score gain', round(float((r.adv.clip(lower=1e-6) / np.sqrt(r.n))[r.adv > 1e-6].sum()), 6))
print('equal (<=1e-6):', int((r.adv.abs() <= 1e-6).sum()), '| merged2 better:', int((r.adv < -1e-6).sum()))
print(r[r.adv > 1e-6].sort_values('adv', ascending=False).head(12).drop(columns='state').to_string(index=False))
