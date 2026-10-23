"""make_job.py out.tar.gz n1,n2,... : write start configs for those puzzles from merged3.csv and tar everything"""
import sys, tarfile, os, shutil, pandas as pd, numpy as np
tar, ns = sys.argv[1], [int(v) for v in sys.argv[2].split(',')]
d = pd.read_csv(os.environ.get('JOB_CSV', 'merged3.csv')); d = d[d.n.isin(ns)]
shutil.rmtree('aws_job/starts', ignore_errors=True); os.makedirs('aws_job/starts')
for n, g in d.groupby('n'):
    P = g.sort_values('square_id')[['x', 'y', 'angle']].to_numpy(float); c, s = np.cos(P[:, 2]), np.sin(P[:, 2]); h = .5 * (abs(c) + abs(s))
    L = max((P[:, 0] + h).max() - (P[:, 0] - h).min(), (P[:, 1] + h).max() - (P[:, 1] - h).min()); P[:, 0] -= (P[:, 0] - h).min(); P[:, 1] -= (P[:, 1] - h).min()
    open(f'aws_job/starts/start_{n}.txt', 'w').write(f'{n} {float(L)!r}\n' + '\n'.join(f'{float(x)!r} {float(y)!r} {float(t)!r}' for x, y, t in P) + '\n')
open('aws_job/starts/order.txt', 'w').write(' '.join(map(str, ns)))
with tarfile.open(tar, 'w:gz') as t: t.add('aws_job', arcname='.')
print('package', tar, os.path.getsize(tar) // 1024, 'KB;', len(ns), 'puzzles')
