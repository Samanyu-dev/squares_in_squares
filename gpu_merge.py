"""Convert GPU-found packings (P_n.npy) to submission-format text, verified by build.finalize. Usage: gpu_merge.py results_dir out.csv"""
import sys, glob, re, math, numpy as np
import build
d, out = sys.argv[1], sys.argv[2]; rows = []; kept = []
for f in sorted(glob.glob(f'{d}/P_*.npy'), key=lambda p: int(re.findall(r'P_(\d+)', p)[0])):
    n = int(re.findall(r'P_(\d+)', f)[0]); P = np.load(f)
    if P.shape != (n, 3) or not np.isfinite(P).all(): print('skip bad shape', n); continue
    Pf, L = build.finalize(n, P, build.bbox_side(P))
    if L >= math.ceil(math.sqrt(n)) - 1e-9: print('n=%d failed verification, dropped' % n); continue
    kept.append((n, L))
    for i, (x, y, t) in enumerate(Pf): rows.append(f'{len(rows)},{n},{i},{float(x)!r},{float(y)!r},{float(t)!r}')
open(out, 'w').write('id,n,square_id,x,y,angle\n' + '\n'.join(rows) + '\n')
print('GPU candidates written:', len(kept), 'puzzles'); [print(f'  n={n} L={L:.8f}') for n, L in kept]
