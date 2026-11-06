import sys, numpy as np
sys.path.insert(0,'/Users/apple/Desktop/squares_in_squares'); sys.path.insert(0,'/Users/apple/Desktop/squares_in_squares/watch')
from enforce import enforce
from collections import defaultdict
src, dst = sys.argv[1:3]
g = defaultdict(list)
for l in open(src).read().split('\n')[1:]:
    p = l.split(',')
    if len(p) >= 6: g[int(p[1])].append(p)
out = []; mx = 0
for n in sorted(g):
    rows = sorted(g[n], key=lambda r: int(r[2])); P = np.array([[float(v) for v in r[3:6]] for r in rows])
    P2, gap, d = enforce(n, P); mx = max(mx, d)
    for r, q in zip(rows, P2): out.append(f'{len(out)},{n},{r[2]},{float(q[0])!r},{float(q[1])!r},{float(q[2])!r}')
open(dst, 'w').write('id,n,square_id,x,y,angle\n' + '\n'.join(out) + '\n'); print('max delta', mx)
