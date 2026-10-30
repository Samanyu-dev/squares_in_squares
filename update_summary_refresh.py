"""Lower ONLY the rows of best_fit_summary.csv where watch/best_L.json holds a smaller certified L; for n<=100 also swap that puzzle's rows in best_fit.csv."""
import json, math, pandas as pd
R = '/Users/apple/Desktop/squares_in_squares'
L = {int(k): v for k, v in json.load(open(f'{R}/watch/best_L.json')).items()}
s = pd.read_csv(f'{R}/best_fit_summary.csv'); changed = []
for i, r in s.iterrows():
    n = int(r.n); new = L.get(n)
    if n > 1000 or new is None or new >= r.L_certified - 1e-9: continue
    g = math.ceil(math.sqrt(n)); changed.append((n, r.L_certified, new))
    s.loc[i, ['L_certified', 'score', 'type', 'gain_vs_grid']] = [round(new, 12), round(new / math.sqrt(n) - 1, 12), 'grid' if abs(new - g) < 1e-9 else 'improved', round((g - new) / math.sqrt(n), 8)]
s.to_csv(f'{R}/best_fit_summary.csv', index=False)
low = [c[0] for c in changed if c[0] <= 100]
if low:
    cur = pd.read_csv(f'{R}/watch/best_current.csv'); bf = pd.read_csv(f'{R}/best_fit.csv'); bf = bf[~bf.n.isin(low)]
    bf = pd.concat([bf, cur[cur.n.isin(low)]]).sort_values(['n', 'square_id']).reset_index(drop=True); bf['id'] = range(len(bf)); bf[['id','n','square_id','x','y','angle']].to_csv(f'{R}/best_fit.csv', index=False)
print('rows changed:', len(changed)); [print(f'  n={n}: {a:.10f} -> {b:.10f}') for n, a, b in changed[:30]]
print('whole total:', round(s.score.sum(), 8))

import subprocess as _sp
_sp.run(['/Users/apple/Desktop/squares_in_squares/.venv/bin/python', '/Users/apple/Desktop/squares_in_squares/export_ranges.py'], capture_output=True)
