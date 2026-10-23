import numpy as np, pandas as pd, math, pickle, collections
st = pickle.load(open('fault/dp2b.pkl', 'rb')); G = st['G']; dp, pc, pr, pd_, pb, pg, pm = st['out']
r = pd.read_pickle('fault/dp2b_vs_merged2.pkl')
def chain(state):
    c, rr, dmv, b = (int(v) for v in state); ch = []
    while True:
        g = int(pg[c, rr, dmv, b]); mode = int(pm[c, rr, dmv, b]); ch.append(g)
        if mode == 9: break
        c, rr, dmv, b = int(pc[c, rr, dmv, b]), int(pr[c, rr, dmv, b]), int(pd_[c, rr, dmv, b]), int(pb[c, rr, dmv, b])
    return ch
bind = collections.defaultdict(list)
for _, x in r[r.n > 324].iterrows():
    ch = chain(x.state); gs = [(int(G[g, 0]), int(G[g, 1]), int(G[g, 2]), G[g, 4]) for g in ch]
    fmax = max(g[3] for g in gs)
    for g in gs:
        if abs(g[3] - fmax) < 1e-9: bind[(g[0], g[1], g[2])].append((int(x.n), fmax))
rows = sorted(((k, len(v), sum(1 / math.sqrt(n) for n, _ in v), v[0][1]) for k, v in bind.items()), key=lambda z: -z[2])
print('distinct binding gadgets for n>324:', len(rows))
print(' (w,h,dm)   puzzles bound   sum 1/sqrt(n)   f')
for k, c, s, f in rows[:20]: print(f'  {k}   {c:4d}   {s:.3f}   {f:.4f}')
tot = sum(s for _, _, s, _ in rows); print('if every binding gadget improved by 0.01 in f, score gain would be about', round(0.01 * tot, 3))
print('if every binding gadget improved by 0.05 in f:', round(0.05 * tot, 3))
