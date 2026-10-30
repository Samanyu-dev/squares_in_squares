"""Everything not yet covered: n<=100 shards 2,3 + all improved puzzles 101..1000, 2 shards."""
import math, json, os, shutil, tarfile, numpy as np, pandas as pd
summ = pd.read_csv('best_fit_summary.csv'); summ = summ[summ.n <= 1000]
cur = pd.concat([pd.read_csv('best_fit.csv'), pd.read_csv('best_fit_101_500.csv'), pd.read_csv('best_fit_501_1000.csv')])
shutil.rmtree('aws_job/starts', ignore_errors=True); os.makedirs('aws_job/starts')
def start_file(n):
    g = cur[cur.n == n].sort_values('square_id'); P = g[['x', 'y', 'angle']].to_numpy(float); c, s = np.cos(P[:, 2]), np.sin(P[:, 2]); h = .5 * (abs(c) + abs(s))
    L = max((P[:, 0] + h).max() - (P[:, 0] - h).min(), (P[:, 1] + h).max() - (P[:, 1] - h).min()); P[:, 0] -= (P[:, 0] - h).min(); P[:, 1] -= (P[:, 1] - h).min()
    open(f'aws_job/starts/start_{n}.txt', 'w').write(f'{n} {float(L)!r}\n' + '\n'.join(f'{float(x)!r} {float(y)!r} {float(t)!r}' for x, y, t in P) + '\n')
# the original n<=100 task list (same construction as small100); shards 0,1 (index%4 in {0,1}) already ran
t100 = []
for n in summ[(summ.n <= 100) & (summ.type == 'improved')].n: t100 += [['start', int(n), 2000 + k, 0.0] for k in range(4)]
for n in summ[(summ.n <= 100) & (summ.type == 'grid')].n:
    r = math.sqrt(n); g = math.ceil(r)
    if g - r < 1e-9: continue
    S = g - 0.02 if (g - r) > 0.05 else g - 0.5 * (g - r)
    t100 += [['rand', int(n), 3000 + k, round(S, 6)] for k in range(3)]
rest100 = [t for i, t in enumerate(t100) if i % 4 in (2, 3)]
tasks = list(rest100)
for n in summ[(summ.n <= 100) & (summ.type == 'improved')].n: start_file(int(n))
big = summ[(summ.n > 100) & (summ.type == 'improved')].sort_values('score', ascending=False)
for n in big.n: start_file(int(n)); tasks.append(['start', int(n), 5000, 0.0])
json.dump(tasks, open('aws_job/starts/tasks.json', 'w'))
with tarfile.open('all2.tar.gz', 'w:gz') as t: t.add('aws_job', arcname='.')
print(len(tasks), 'tasks (', len(rest100), 'small-n +', len(big), 'big-n ); package', os.path.getsize('all2.tar.gz') // 1024, 'KB')
