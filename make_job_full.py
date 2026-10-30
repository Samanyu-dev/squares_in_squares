"""Full remaining search over every puzzle: remaining small-n tasks, all improved 101..1000 (10 min), grid-valued 101..1000 started from the grid (5 min)."""
import math, json, os, shutil, tarfile, numpy as np, pandas as pd
summ = pd.read_csv('best_fit_summary.csv'); summ = summ[summ.n <= 1000]
cur = pd.concat([pd.read_csv('best_fit.csv'), pd.read_csv('best_fit_101_500.csv'), pd.read_csv('best_fit_501_1000.csv')]).sort_values(['n', 'square_id']).reset_index(drop=True)
cur['id'] = range(len(cur)); cur[['id', 'n', 'square_id', 'x', 'y', 'angle']].to_csv('best_all_500500.csv', index=False)
print('full file rows:', len(cur), '| puzzles:', cur.n.nunique())
shutil.rmtree('aws_job/starts', ignore_errors=True); os.makedirs('aws_job/starts')
def start_file(n):
    g = cur[cur.n == n]; P = g[['x', 'y', 'angle']].to_numpy(float); c, s = np.cos(P[:, 2]), np.sin(P[:, 2]); h = .5 * (abs(c) + abs(s))
    L = max((P[:, 0] + h).max() - (P[:, 0] - h).min(), (P[:, 1] + h).max() - (P[:, 1] - h).min()); P[:, 0] -= (P[:, 0] - h).min(); P[:, 1] -= (P[:, 1] - h).min()
    open(f'aws_job/starts/start_{n}.txt', 'w').write(f'{n} {float(L)!r}\n' + '\n'.join(f'{float(x)!r} {float(y)!r} {float(t)!r}' for x, y, t in P) + '\n')
t100 = []
for n in summ[(summ.n <= 100) & (summ.type == 'improved')].n: t100 += [['start', int(n), 2000 + k, 0.0] for k in range(4)]; start_file(int(n))
for n in summ[(summ.n <= 100) & (summ.type == 'grid')].n:
    r = math.sqrt(n); g = math.ceil(r)
    if g - r < 1e-9: continue
    S = g - 0.02 if (g - r) > 0.05 else g - 0.5 * (g - r)
    t100 += [['rand', int(n), 3000 + k, round(S, 6), 600] for k in range(3)]
tasks = [t for i, t in enumerate(t100) if i % 4 in (2, 3)]
big = summ[(summ.n > 100) & (summ.type == 'improved')].sort_values('score', ascending=False)
for n in big.n: start_file(int(n)); tasks.append(['start', int(n), 5000, 0.0, 600])
gr = summ[(summ.n > 100) & (summ.type == 'grid')]
ng = 0
for n in gr.n:
    if math.isqrt(int(n)) ** 2 == int(n): continue
    start_file(int(n)); tasks.append(['start', int(n), 6000, 0.0, 300]); ng += 1
json.dump(tasks, open('aws_job/starts/tasks.json', 'w'))
secs = sum((t[4] if len(t) > 4 else 600) for t in tasks)
with tarfile.open('full.tar.gz', 'w:gz') as t: t.add('aws_job', arcname='.')
print(len(tasks), 'tasks | grid-start tasks:', ng, '| core-hours:', round(secs / 3600, 1), '| package', os.path.getsize('full.tar.gz') // 1024, 'KB')
