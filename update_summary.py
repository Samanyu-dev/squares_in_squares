"""Fill/extend best_fit_summary.csv from watch/best_L.json for n up to MAXN (does not lower any existing row's L unless certified smaller)."""
import json, math, sys, pandas as pd
maxn = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
L = {int(k): v for k, v in json.load(open('/Users/apple/Desktop/squares_in_squares/watch/best_L.json')).items()}
df = pd.read_csv('/Users/apple/Desktop/squares_in_squares/best_fit_summary.csv'); have = set(df.n); rows = []
for n in range(1, maxn + 1):
    if n in have: continue
    g = math.ceil(math.sqrt(n)); l = L[n]
    rows.append({'n': n, 'L_certified': round(l, 12), 'score': round(l / math.sqrt(n) - 1, 12), 'grid_L': g, 'type': 'grid' if abs(l - g) < 1e-9 else 'improved', 'gain_vs_grid': round((g - l) / math.sqrt(n), 8)})
df = pd.concat([df, pd.DataFrame(rows)]).sort_values('n').reset_index(drop=True); df.to_csv('/Users/apple/Desktop/squares_in_squares/best_fit_summary.csv', index=False)
for lo, hi in [(1, 100), (101, 500), (501, 1000)]:
    s = df[(df.n >= lo) & (df.n <= hi)]
    if len(s): print(f'n={lo}-{hi}: {len(s)} puzzles | improved {int((s.type == "improved").sum())} | grid {int((s.type == "grid").sum())} | total score {s.score.sum():.6f}')
print('summary rows:', len(df), '| whole total', round(df.score.sum(), 8))
