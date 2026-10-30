"""Regenerate best_fit_101_500.csv and best_fit_501_1000.csv (per-square rows: id,n,square_id,x,y,angle) from watch/best_current.csv."""
import pandas as pd
R = '/Users/apple/Desktop/squares_in_squares'
cur = pd.read_csv(f'{R}/watch/best_current.csv')
for lo, hi, name in [(101, 500, 'best_fit_101_500.csv'), (501, 1000, 'best_fit_501_1000.csv')]:
    s = cur[(cur.n >= lo) & (cur.n <= hi)].sort_values(['n', 'square_id']).reset_index(drop=True); s['id'] = range(len(s))
    s[['id', 'n', 'square_id', 'x', 'y', 'angle']].to_csv(f'{R}/{name}', index=False)
    c = s.groupby('n').size(); print(f'{name}: {len(s)} rows, {s.n.nunique()} puzzles (n {s.n.min()}-{s.n.max()}), each n has n rows: {bool((c.index == c.values).all())}')
