import glob, re, math, numpy as np
from fractions import Fraction
out = {}
for f in glob.glob('ext/couzo/certificates/best/n*.cert') + glob.glob('ext/couzo/certificates/extended/n*.cert') + glob.glob('ext/couzo/certificates/n*.cert'):
    m = re.findall(r'n(\d+)\.cert$', f)
    if not m: continue
    n = int(m[0])
    if n <= 1000 or n > 3600: continue          # donors the public authors used went to n=3500
    lines = [l for l in open(f).read().splitlines() if l.strip() and not l.startswith('#')]
    P = np.array([[float(Fraction(v)) for v in l.split()] for l in lines[1:]]); P[:, 2] = 2 * np.arctan(P[:, 2])
    out[str(n)] = P
np.savez_compressed('fault/big_donors.npz', **out); print('big donors:', len(out), 'n range', min(map(int, out)), '-', max(map(int, out)))
