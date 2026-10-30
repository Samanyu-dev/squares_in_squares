import math, json, os, shutil, tarfile, numpy as np, pandas as pd
summ = pd.read_csv('best_fit_summary.csv'); d = pd.read_csv('watch/best_current.csv')
sel = summ[(summ.n >= 501) & (summ.type == 'improved')].sort_values('score', ascending=False)
shutil.rmtree('aws_job/starts', ignore_errors=True); os.makedirs('aws_job/starts'); tasks = []
for n in sel.n:
    g = d[d.n == n].sort_values('square_id'); P = g[['x', 'y', 'angle']].to_numpy(float); c, s = np.cos(P[:, 2]), np.sin(P[:, 2]); h = .5 * (abs(c) + abs(s))
    L = max((P[:, 0] + h).max() - (P[:, 0] - h).min(), (P[:, 1] + h).max() - (P[:, 1] - h).min()); P[:, 0] -= (P[:, 0] - h).min(); P[:, 1] -= (P[:, 1] - h).min()
    open(f'aws_job/starts/start_{n}.txt', 'w').write(f'{n} {float(L)!r}\n' + '\n'.join(f'{float(x)!r} {float(y)!r} {float(t)!r}' for x, y, t in P) + '\n')
    tasks.append(['start', int(n), 4000, 0.0])
json.dump(tasks, open('aws_job/starts/tasks.json', 'w'))
with tarfile.open('big501.tar.gz', 'w:gz') as t: t.add('aws_job', arcname='.')
print(len(tasks), 'puzzles queued (highest room first); package', os.path.getsize('big501.tar.gz') // 1024, 'KB')
