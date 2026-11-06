"""rattle: kick+fq quench directly on our packings. usage: rattle.py best.csv out_dir steps secs n1 n2 ..."""
import sys, os, json, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run
from concurrent.futures import ProcessPoolExecutor
def job(a):
    n, K, secs, out, bestL = a
    if not run.PK: run.PK = run.load(os.environ['SQ_BEST'])
    wd = f'{out}/r{n}'; os.makedirs(wd, exist_ok=True); rng = random.Random(n)
    cur = run.PK[n]; curL = run.side(cur); L0 = curL; hist = []
    for k in range(K):
        sig = rng.uniform(float(os.environ.get('SIGLO',0.002)), float(os.environ.get('SIGHI',0.02))); p = cur.copy(); g = np.random.default_rng(rng.randrange(1 << 30))
        p[:, :2] += g.normal(0, sig, (len(p), 2)); p[:, 2] += g.normal(0, sig, len(p))
        c = run.quench(p, secs, f'r{n}', wd, loosen=rng.uniform(float(os.environ.get('LOLO',0.5)), float(os.environ.get('LOHI',2.0))))
        if c is not None and run.side(c) < curL: cur, curL = c, run.side(c)
    if curL < bestL[n] - 1e-9: np.save(f'{out}/win_{n}_r.npy', cur)
    return (n, round(L0, 9), round(curL, 9), round(L0 - curL, 9))
if __name__ == '__main__':
    best, out, K, secs = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]); ns = [int(x) for x in sys.argv[5:]]
    os.makedirs(out, exist_ok=True); os.environ['SQ_BEST'] = os.path.abspath(best)
    bestL = {int(k): v for k, v in json.load(open(f'{run.ROOT}/watch/best_L.json')).items()}
    with ProcessPoolExecutor(int(os.environ.get('NPROC',4))) as ex:
        for r in ex.map(job, [(n, K, secs, out, bestL) for n in ns]): print(r, flush=True)
