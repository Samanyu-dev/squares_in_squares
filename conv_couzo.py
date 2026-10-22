"""Read-only conversion of exact certificates (data) into submission-format rows. Usage: conv_couzo.py certs_dir table_pkl out.csv"""
import sys, glob, re, math, numpy as np, pandas as pd
from fractions import Fraction
SCALE = Fraction(10**12 + 1, 10**12)
d, tab, out = sys.argv[1], sys.argv[2], sys.argv[3]
t = pd.read_pickle(tab).set_index('n')
rows = []; stats = []
for f in sorted(glob.glob(f'{d}/n*.cert'), key=lambda p: int(re.findall(r'n(\d+)\.cert', p)[0])):
    n = int(re.findall(r'n(\d+)\.cert', f)[0])
    if n > 1000 or n < 2: continue
    lines = [l for l in open(f).read().splitlines() if l.strip() and not l.startswith('#')]
    hdr = lines[0].split(); assert int(hdr[0]) == n and len(lines) == n + 1, f
    S = Fraction(hdr[1]); P = []
    for l in lines[1:]:
        x, y, tt = (Fraction(v) for v in l.split())
        x = S / 2 + (x - S / 2) * SCALE; y = S / 2 + (y - S / 2) * SCALE
        P.append((float(x), float(y), 2 * math.atan(float(tt))))
    P = np.array(P); c, s = np.cos(P[:, 2]), np.sin(P[:, 2]); h = .5 * (abs(c) + abs(s))
    L = max((P[:, 0] + h).max() - (P[:, 0] - h).min(), (P[:, 1] + h).max() - (P[:, 1] - h).min())
    if L < t.loc[n, 'L'] - 1e-9:
        stats.append((n, t.loc[n, 'L'], L, (t.loc[n, 'L'] - L) / math.sqrt(n)))
        for i, (x, y, a) in enumerate(P): rows.append(f'{len(rows)},{n},{i},{x!r},{y!r},{a!r}')
open(out, 'w').write('id,n,square_id,x,y,angle\n' + '\n'.join(rows) + '\n')
print('puzzles where the register beats merged2:', len(stats), ' total score gain', round(sum(s[3] for s in stats), 6))
for n, a, b, g in sorted(stats, key=lambda z: -z[3])[:15]: print(f'  n={n}  merged2 {a:.6f} -> register {b:.6f}  gain {g:.6f}')
