"""Usage: apply_improvement.py N packing.csv   (packing.csv holds ONLY puzzle N's rows: id,n,square_id,x,y,angle)
Certifies puzzle N with the CURRENT official metric; if accepted and smaller than the current certified L, updates ONLY that row of
best_fit_summary.csv, swaps that puzzle's rows in best_fit.csv (n<=100) and watch/best_current.csv, and updates watch/best_L.json."""
import sys, os, csv, json, math, subprocess, shutil, re
import pandas as pd
ROOT = '/Users/apple/Desktop/squares_in_squares'; PY = f'{ROOT}/.venv/bin/python'; SCT = f'{ROOT}/scoretest_n100'
n = int(sys.argv[1]); path = sys.argv[2]
d = pd.read_csv(path); d = d[d.n == n].sort_values('square_id').reset_index(drop=True)
assert len(d) == n and list(d.square_id) == list(range(n)), 'packing must contain exactly n squares'
tmp = f'{ROOT}/tmp_apply_{n}.csv'; d2 = d.copy(); d2['id'] = range(len(d2)); d2[['id', 'n', 'square_id', 'x', 'y', 'angle']].to_csv(tmp, index=False)
det = f'{SCT}/submission_details.csv'
if os.path.exists(det): os.remove(det)
subprocess.run([PY, f'{ROOT}/runner/run_official.py', tmp, SCT], capture_output=True, text=True, timeout=3000)
L = {int(a): float(b) for a, b in list(csv.reader(open(det)))[1:]}.get(n, 0.0); os.remove(tmp)
summ = pd.read_csv(f'{ROOT}/best_fit_summary.csv'); i = summ.index[summ.n == n]
cur = float(summ.loc[i[0], 'L_certified']) if len(i) else None
if L <= 0: print(f'n={n}: REJECTED by the official metric; nothing changed'); sys.exit(1)
if cur is not None and L >= cur - 1e-9: print(f'n={n}: certified L={L:.10f} is not better than current {cur:.10f}; nothing changed'); sys.exit(0)
g = math.ceil(math.sqrt(n)); row = [round(L, 12), round(L / math.sqrt(n) - 1, 12), 'grid' if abs(L - g) < 1e-9 else 'improved', round((g - L) / math.sqrt(n), 8)]
if len(i): summ.loc[i[0], ['L_certified', 'score', 'type', 'gain_vs_grid']] = row
else: summ = pd.concat([summ, pd.DataFrame([{'n': n, 'L_certified': row[0], 'score': row[1], 'grid_L': g, 'type': row[2], 'gain_vs_grid': row[3]}])]).sort_values('n')
summ.to_csv(f'{ROOT}/best_fit_summary.csv', index=False)
def swap(csvpath):
    a = pd.read_csv(csvpath); a = a[a.n != n]; b = pd.concat([a, d[['id', 'n', 'square_id', 'x', 'y', 'angle']]]).sort_values(['n', 'square_id']).reset_index(drop=True)
    b['id'] = range(len(b)); b[['id', 'n', 'square_id', 'x', 'y', 'angle']].to_csv(csvpath, index=False)
if n <= 100: swap(f'{ROOT}/best_fit.csv')
swap(f'{ROOT}/watch/best_current.csv')
bl = {int(k): v for k, v in json.load(open(f'{ROOT}/watch/best_L.json')).items()}; bl[n] = L; json.dump(bl, open(f'{ROOT}/watch/best_L.json', 'w'))
print(f'n={n}: UPDATED row. certified L {cur} -> {L:.12f} (score gain {((cur or g) - L) / math.sqrt(n):.8f})')

import subprocess as _sp
_sp.run(['/Users/apple/Desktop/squares_in_squares/.venv/bin/python', '/Users/apple/Desktop/squares_in_squares/export_ranges.py'], capture_output=True)
