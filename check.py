import sys, math, csv, numpy as np
f = sys.argv[1]
lines = open(f).read().split('\n')
assert lines[0] == 'id,n,square_id,x,y,angle', lines[0]
assert lines[-1] == '' and lines[-2] != '', 'trailing newline'
rows = [l.split(',') for l in lines[1:-1]]
assert all(len(r) == 6 for r in rows), 'column count'
ids = [int(r[0]) for r in rows]; assert ids == list(range(len(rows))), 'ids'
ns = np.array([int(r[1]) for r in rows]); sid = np.array([int(r[2]) for r in rows])
A = np.array([[float(v) for v in r[3:]] for r in rows])   # float() rejects np.float64(..) strings
assert np.isfinite(A).all()
assert len(rows) == 500500
tot = 0; worst = 1e9; viol = []; gridc = 0
for n in range(1, 1001):
    m = ns == n; assert m.sum() == n, n
    assert sorted(sid[m]) == list(range(n)), ('square_id', n)
    P = A[m]; x, y, t = P.T
    c, s = np.cos(t), np.sin(t); h = .5 * (abs(c) + abs(s))
    L = max((x + h).max() - (x - h).min(), (y + h).max() - (y - h).min())
    tot += L / math.sqrt(n) - 1
    if abs(L - math.ceil(math.sqrt(n))) < 1e-9: gridc += 1
    # independent exact SAT on all pairs (vectorised), real unit squares
    i, j = np.triu_indices(n, 1)
    dx, dy = x[j] - x[i], y[j] - y[i]
    near = dx * dx + dy * dy < 2.0
    i, j, dx, dy = i[near], j[near], dx[near], dy[near]
    mn = np.full(len(i), 1e9)
    for k in range(4):
        a = t[i] if k < 2 else t[j]; a = a + (math.pi / 2 if k % 2 else 0)
        ax, ay = np.cos(a), np.sin(a)
        r = 0
        for q in (i, j):
            r = r + .5 * (abs(ax * np.cos(t[q]) + ay * np.sin(t[q])) + abs(-ax * np.sin(t[q]) + ay * np.cos(t[q])))
        mn = np.minimum(mn, r - abs(dx * ax + dy * ay))
    if len(mn):
        worst = min(worst, -mn.max()) if mn.max() > -1e9 else worst
        if (mn > 0).any(): viol.append((n, mn.max()))
print('score', round(tot, 6), '| non-improved(grid) puzzles', gridc, '| overlap violations', viol[:5], '| min separation (neg=gap) worst', worst)
# gap check on improved puzzles only
import itertools
g = 1e9
for n in range(2, 101):
    m = ns == n; P = A[m]; x, y, t = P.T
    if abs((lambda c,s:max((x+.5*(abs(c)+abs(s))).max()-(x-.5*(abs(c)+abs(s))).min(),(y+.5*(abs(c)+abs(s))).max()-(y-.5*(abs(c)+abs(s))).min()))(np.cos(t),np.sin(t)) - math.ceil(math.sqrt(n))) < 1e-9: continue
    i, j = np.triu_indices(n, 1); dx, dy = x[j]-x[i], y[j]-y[i]
    mn = np.full(len(i), 1e9)
    for k in range(4):
        a = (t[i] if k < 2 else t[j]) + (math.pi/2 if k % 2 else 0); ax, ay = np.cos(a), np.sin(a); r = 0
        for q in (i, j): r = r + .5*(abs(ax*np.cos(t[q])+ay*np.sin(t[q]))+abs(-ax*np.sin(t[q])+ay*np.cos(t[q])))
        mn = np.minimum(mn, r - abs(dx*ax+dy*ay))
    g = min(g, -mn.max())
print('min gap between any two squares in improved puzzles:', g)
