import subprocess, sys, numpy as np, math, tempfile, os
from build import *
from sq import grid
# n=1, small grids
assert finalize(1, *grid(1))[1] == 1.0
for n in (2, 3, 4, 9, 16): assert abs(finalize(n, *grid(n))[1] - math.ceil(math.sqrt(n))) < 1e-12
# overlap detector must flag overlaps and accept touching
P = np.array([[.5, .5, 0], [1.5, .5, 0]]); assert abs(min_gap(P)) < 1e-12
P = np.array([[.5, .5, 0], [1.4, .5, 0]]); assert min_gap(P) < -.05
P = np.array([[.5, .5, 0], [1.2, 1.2, math.pi / 4]]); assert min_gap(P) < 0           # rotated overlap
P = np.array([[.5, .5, 0], [2.0, 2.0, math.pi / 4]]); assert min_gap(P) > 0           # rotated apart
# angle wrap equivalence: pi/2 rotation is identical square
P = np.array([[.5, .5, 0], [1.5, .5, math.pi / 2]]); assert abs(min_gap(P)) < 1e-9
# bogus/overlapping packing is rejected -> falls back to grid
bad = np.array([[.5 + .1 * i, .5, 0.3] for i in range(5)])
assert finalize(5, bad, 2.5)[1] == 3.0
# spread keeps validity and a real gap
import pickle
best = pickle.load(open('best_a.pkl', 'rb'))
for n in (5, 10, 26, 101, 500, 999):
    P, L = finalize(n, *best[n]); assert len(P) == n and min_gap(P) >= (0 if L >= math.ceil(math.sqrt(n)) - 1e-9 else MIN_GAP), n
# float format: no numpy reprs in the csv
write({n: best[n] for n in range(1, 1001)}, '/tmp/_t.csv')
txt = open('/tmp/_t.csv').read(); assert 'np.' not in txt and 'nan' not in txt.lower() and 'inf' not in txt.lower()
print('all edge tests passed')
