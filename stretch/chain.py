"""stretch + fq + kick chains, only for targets in targets_vs_leaders.csv.  usage: chain.py best.csv out_dir steps_per_chain topN"""
import sys, os, json, random, math
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run, stretch
from concurrent.futures import ProcessPoolExecutor
ROOT = run.ROOT
def job(a):
    n, st, out, bestL, K = a
    if not run.PK: run.PK = run.load(os.environ['SQ_BEST'])
    if not TG: TG.update(pd.read_csv(f'{ROOT}/targets_vs_leaders.csv').head(int(os.environ['SQ_TOP'])).n.tolist())
    wd = f'{out}/w{n}_{st}'; os.makedirs(wd, exist_ok=True)
    try:
        b, _ = stretch.stretch(run.PK[n], st)
        if b is None: return None
        m = len(b); tgt = bestL.get(m)
        if tgt is None or m not in TG: return None
        q = run.quench(b, int(os.environ.get('QSEC', 60)), f's{n}_{st}', wd)
        if q is None: return (n, st, m, None)
        cur, curL = q, run.side(q); rng = random.Random(n * 10 + st + int(os.environ.get('SEED', 0)))
        for k in range(K):
            if curL - tgt > float(os.environ.get('SQ_GATE', 0.06)): break
            sig = rng.uniform(0.002, 0.02); p = cur.copy()
            p[:, :2] += np.random.default_rng(rng.randrange(1 << 30)).normal(0, sig, (len(p), 2)); p[:, 2] += np.random.default_rng(rng.randrange(1 << 30)).normal(0, sig, len(p))
            c = run.quench(p, int(os.environ.get('KSEC', 45)), f'k{n}_{st}', wd, loosen=rng.uniform(0.5, 2.0))
            if c is not None and run.side(c) < curL: cur, curL = c, run.side(c)
        if curL < tgt - 1e-9: np.save(f'{out}/win_{m}_{n}_{st}.npy', cur)
        return (n, st, m, curL, tgt)
    except Exception as e: return (n, st, None, repr(e)[:60])
TG = set()
if __name__ == '__main__':
    best, out, K, top = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]); os.makedirs(out, exist_ok=True)
    os.environ['SQ_BEST'] = os.path.abspath(best); os.environ['SQ_TOP'] = str(top); run.PK = run.load(best)
    bestL = {int(k): v for k, v in json.load(open(f'{ROOT}/watch/best_L.json')).items()}
    t = pd.read_csv(f'{ROOT}/targets_vs_leaders.csv').head(top); TG.update(t.n.tolist())
    specs = [n for n in sorted(run.PK) if n >= int(os.environ.get('MINN', 100)) and len(stretch.patches(run.PK[n]))]
    jobs = [(n, st, out, bestL, K) for n in specs for st in (1, -1, 2, -2, 3, -3, 4, -4)]
    jobs = jobs[int(os.environ.get('SHARD', 0))::int(os.environ.get('SHARDS', 1))]
    print(len(jobs), 'jobs', flush=True)
    with ProcessPoolExecutor(int(os.environ.get('NPROC', 9))) as ex:
        for r in ex.map(job, jobs):
            if r: print(r, flush=True)
