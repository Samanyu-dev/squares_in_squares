import sys, os, time, pandas as pd
work = os.path.abspath(sys.argv[1]); csvp = os.path.abspath(sys.argv[2]); ns = [int(v) for v in sys.argv[3].split(',')]
sys.path.insert(0, work); os.chdir(work)
import official_metric as m
d = pd.read_csv(csvp); d = d[d.n.isin(ns)]
puzzles = m._parse_submission_csv(d[['n', 'square_id', 'x', 'y', 'angle']].to_csv(index=False))
for n in ns:
    t = time.time(); r = m.evaluate_single_puzzle(n, puzzles[n])
    print(f'n={n}: score={r[1]:.8f}  L_text={"None" if r[2] is None else r[2][:22]}  msg={str(r[3])[:160]}  [{time.time()-t:.0f}s]', flush=True)
