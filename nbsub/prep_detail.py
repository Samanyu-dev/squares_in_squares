"""build the extra dataset files: per_puzzle_table.csv, lift_chains.csv, score_history.csv, METHODS.md.  usage: prep_detail.py final.csv"""
import sys, math, json, numpy as np, pandas as pd
sys.path.insert(0, '../stretch'); import run
R = '/Users/apple/Desktop/squares_in_squares'
df = pd.read_csv(sys.argv[1]); rows = []
for n, g in df.groupby('n'):
    s = g.sort_values('square_id')[['x', 'y', 'angle']].to_numpy(float); L = run.side(s)
    t = (np.degrees(s[:, 2]) % 90 + 45) % 90 - 45; tilted = int((abs(t) > 0.05).sum())
    cls = 'grid' if tilted == 0 else ('diag45' if (abs(abs(t[abs(t) > .05]) - 45) < 0.5).mean() > 0.5 else 'tilted-arms')
    K = int(math.isqrt(n)); rows.append((n, K, n - K * K, L, L / math.sqrt(n) - 1, tilted, cls))
T = pd.DataFrame(rows, columns=['n', 'K', 'D', 'L', 'excess', 'tilted_squares', 'structure']).set_index('n')
old = pd.read_csv(f'{R}/targets_vs_leaders.csv').set_index('n')
ld = {int(a): float(b) for a, b in (x.split(':') for x in open(f'{R}/pubd/lb2/leaders_lt400.txt').read().split())}
T['leader_L'] = np.nan
T.loc[old.index, 'leader_L'] = old.L_leader
for n, v in ld.items(): T.loc[n, 'leader_L'] = v
T['leader_team'] = old.team.reindex(T.index)
T['gap_L'] = T.L - T.leader_L; T['gap_score'] = T.gap_L / np.sqrt(T.index)
T['status'] = np.where(T.leader_L.isna(), 'no leader row', np.where(T.gap_L < -1e-9, 'AHEAD', np.where(T.gap_L < 1e-9, 'tied', 'behind')))
T.round(12).to_csv('data/per_puzzle_table.csv')
# lift chains: leader side L(n)=L(n')+m
Ls = {n: T.leader_L[n] for n in T.index if not math.isnan(T.leader_L[n])}; ch = []
for n, v in Ls.items():
    for p, u in Ls.items():
        if p < n and 0 < v - u < 12 and abs((v - u) - round(v - u)) < 2e-8 and T.K[n] - T.K[p] == round(v - u): ch.append((p, n, round(v - u), u, v)); break
pd.DataFrame(ch, columns=['base_n', 'lifted_n', 'rows_added', 'base_L', 'lifted_L']).to_csv('data/lift_chains.csv', index=False)
pd.DataFrame([(0, 28.9377, 'Grid baseline'), (1, 28.3553, 'Own packings for small n'), (2, 27.32, 'More own packings plus border extension'),
 (3, 26.79, 'Public notebook outputs, per-puzzle merge'), (4, 22.4776, 'Fault-line packings from public notebooks'), (5, 22.3747, 'Newer public notebooks'),
 (6, 22.3611, 'First Kaggle submission of a merged file'), (7, 22.2616, 'Merge under the tightened metric'), (8, 22.2603, 'Clearance fix, re-merge'),
 (9, 22.2499, 'All public notebooks, ~12 datasets, 2e-11 clearance'), (10, 22.2382, 'New public notebooks'), (11, 22.2362, 'Stretch+quench chain wins'),
 (12, 22.2318, 'penguin069 portfolio, own AWS/local chains'), (13, 22.2247, 'jazivxt Boxed In per-puzzle merge'), (14, 22.2241, 'Own chain wins n=417, 595 (Kaggle 22.2241274836)'),
 (15, float(T.excess.sum()), 'This version: kmmmvs merge, AWS chain wins, n=109 (beats a leader), n=339')], columns=['step', 'score', 'what']).to_csv('data/score_history.csv', index=False)
print(T.status.value_counts().to_dict(), len(ch), 'lift pairs', f'score {T.excess.sum():.10f}')
