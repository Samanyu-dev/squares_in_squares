import pickle, math, sys, numpy as np
from sq import *
db = pickle.load(open(sys.argv[1], 'rb')); out = sys.argv[2]
rows = []; sc = 0; gs = 0; bad = 0
for n in range(1, 1001):
    P, L = db.get(n, grid(n))
    P = normalize(P)
    if n > 1 and not valid(P):
        P, L = grid(n); bad += 1
    L = side(P)
    sc += L / math.sqrt(n) - 1; gs += math.ceil(math.sqrt(n)) / math.sqrt(n) - 1
    for i, (x, y, t) in enumerate(P):
        rows.append(f'{len(rows)},{n},{i},{float(x)!r},{float(y)!r},{float(t)!r}')
open(out, 'w').write('id,n,square_id,x,y,angle\n' + '\n'.join(rows) + '\n')
print('rows', len(rows), 'score', round(sc, 4), 'grid', round(gs, 4), 'invalid->grid', bad)
