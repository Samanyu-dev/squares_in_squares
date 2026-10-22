"""Stage B: staircase chain DP (modes X/Y only -> conservative). dp[cu][ru][dm] = min over chains of max overhang f."""
import numpy as np, pandas as pd, math, time
from numba import njit
W = 31; DM = 63
F = np.load('fault/F.npy')
gl = [(w, h, dm, F[w, h, dm]) for w in range(1, W + 1) for h in range(1, W + 1) for dm in range(1, DM + 1) if F[w, h, dm] < 0.9]
G = np.array(gl); print('gadgets with f<0.9:', len(G), flush=True)
@njit(cache=True)
def solve(G, W, DM):
    INF = 1e9
    dp = np.full((W + 2, W + 2, DM + 1), INF)
    ng = G.shape[0]
    for g in range(ng):
        w, h, dm, f = int(G[g, 0]), int(G[g, 1]), int(G[g, 2]), G[g, 3]
        if f < dp[w, h, dm]: dp[w, h, dm] = f
    for s in range(2, 2 * W + 2):
        for cu in range(0, W + 1):
            ru = s - cu
            if ru < 0 or ru > W: continue
            for d0 in range(1, DM + 1):
                v = dp[cu, ru, d0]
                if v >= INF: continue
                for g in range(ng):
                    f = G[g, 3]
                    nf = f if f > v else v
                    dm = int(G[g, 2])
                    if d0 + dm > DM: continue
                    w = int(G[g, 0]); h = int(G[g, 1])
                    c1 = cu + w; r1 = ru + h + 1          # mode Y
                    if c1 <= W and r1 <= W and nf < dp[c1, r1, d0 + dm]: dp[c1, r1, d0 + dm] = nf
                    c0 = cu + w; r0 = ru + h              # mode 0 (OPTIMISTIC: tips assumed free)
                    if c0 <= W and r0 <= W and nf < dp[c0, r0, d0 + dm]: dp[c0, r0, d0 + dm] = nf
                    c2 = cu + w + 1; r2 = ru + h          # mode X
                    if c2 <= W and r2 <= W and nf < dp[c2, r2, d0 + dm]: dp[c2, r2, d0 + dm] = nf
    return dp
t = time.time(); dp = solve(G, W, DM); print('dp done', round(time.time() - t), 's', flush=True)
np.save('fault/dp_opt.npy', dp)
tab = pd.read_pickle('table_merged2.pkl').set_index('n')
rows = []
for n in range(5, 1001):
    best = (1e9, None)
    for K in range(max(1, math.isqrt(n) - 2), math.isqrt(n) + 1):
        D = n - K * K
        if D < 1 or D > DM: continue
        v = dp[:K + 1, :K + 1, D:].min()
        if K + v < best[0]: best = (K + v, K)
    if best[1] is not None: rows.append((n, best[1], best[0], tab.loc[n, 'L'], tab.loc[n, 'L'] - best[0]))
r = pd.DataFrame(rows, columns=['n', 'K', 'dp_L', 'merged2_L', 'adv'])
r.to_pickle('fault/dp_opt_vs_merged2.pkl')
print('puzzles where the chain DP (no tips) would beat merged2:', int((r.adv > 1e-9).sum()), '| total score gain', round(float((r.adv.clip(lower=0) / np.sqrt(r.n)).sum()), 6))
print('equal within 1e-9:', int((r.adv.abs() <= 1e-9).sum()), '| merged2 better:', int((r.adv < -1e-9).sum()))
print(r.sort_values('adv', ascending=False).head(10).to_string(index=False))
