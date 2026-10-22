"""Per-n best across sources, copying the ORIGINAL text of each chosen packing. Usage: merge.py out.csv src1.csv src2.csv ..."""
import sys, math, numpy as np
from collections import defaultdict
def load(path):
    g = defaultdict(list)
    with open(path) as f:
        next(f)
        for line in f:
            p = line.rstrip('\n').split(',')
            if len(p) < 6: continue
            g[int(p[1])].append(p)
    return g
def side(rows):
    A = np.array([[float(v) for v in r[3:6]] for r in rows])
    c, s = np.cos(A[:, 2]), np.sin(A[:, 2]); h = .5 * (abs(c) + abs(s))
    return max((A[:, 0] + h).max() - (A[:, 0] - h).min(), (A[:, 1] + h).max() - (A[:, 1] - h).min())
out = sys.argv[1]; srcs = [(p, load(p)) for p in sys.argv[2:]]
rows = []; total = 0; won = defaultdict(int)
for n in range(1, 1001):
    cands = []
    for name, g in srcs:
        if n in g and len(g[n]) == n and sorted(int(r[2]) for r in g[n]) == list(range(n)):
            try: cands.append((side(g[n]), name, g[n]))
            except ValueError: pass
    L, name, R = min(cands, key=lambda c: c[0])
    won[name] += 1; total += L / math.sqrt(n) - 1
    for r in sorted(R, key=lambda r: int(r[2])): rows.append(f'{len(rows)},{n},{r[2]},{r[3]},{r[4]},{r[5]}')
open(out, 'w').write('id,n,square_id,x,y,angle\n' + '\n'.join(rows) + '\n')
print(f'merged {len(rows)} rows, local score {total:.5f}'); [print(' ', k.split('/')[-1][:60], v) for k, v in won.items()]
