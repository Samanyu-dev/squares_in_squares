"""small-n job: improved puzzles start from best_fit.csv packings (4 seeds); grid-valued non-squares start random below the grid (3 seeds)"""
import math, json, os, shutil, tarfile, numpy as np, pandas as pd
summ = pd.read_csv('best_fit_summary.csv'); d = pd.read_csv('best_fit.csv')
shutil.rmtree('aws_job/starts', ignore_errors=True); os.makedirs('aws_job/starts')
tasks = []
for n in summ[summ.type == 'improved'].n:
    g = d[d.n == n].sort_values('square_id'); P = g[['x', 'y', 'angle']].to_numpy(float); c, s = np.cos(P[:, 2]), np.sin(P[:, 2]); h = .5 * (abs(c) + abs(s))
    L = max((P[:, 0] + h).max() - (P[:, 0] - h).min(), (P[:, 1] + h).max() - (P[:, 1] - h).min()); P[:, 0] -= (P[:, 0] - h).min(); P[:, 1] -= (P[:, 1] - h).min()
    open(f'aws_job/starts/start_{n}.txt', 'w').write(f'{n} {float(L)!r}\n' + '\n'.join(f'{float(x)!r} {float(y)!r} {float(t)!r}' for x, y, t in P) + '\n')
    tasks += [['start', int(n), 2000 + k, 0.0] for k in range(4)]
for n in summ[(summ.type == 'grid')].n:
    r = math.sqrt(n); g = math.ceil(r)
    if g - r < 1e-9: continue
    S = g - 0.02 if (g - r) > 0.05 else g - 0.5 * (g - r)
    tasks += [['rand', int(n), 3000 + k, round(S, 6)] for k in range(3)]
json.dump(tasks, open('aws_job/starts/tasks.json', 'w'))
with tarfile.open('small100.tar.gz', 'w:gz') as t: t.add('aws_job', arcname='.')
print(len(tasks), 'tasks;', sum(1 for t in tasks if t[0] == 'start'), 'from packings,', sum(1 for t in tasks if t[0] == 'rand'), 'random; package', os.path.getsize('small100.tar.gz') // 1024, 'KB')
