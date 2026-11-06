"""neighbour transfer via fq: n from our (n+d) packing minus d squares, quench. usage: transfer.py best.csv out secs n1 n2 ..."""
import sys, os, json, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run
from concurrent.futures import ProcessPoolExecutor
def job(a):
    n, secs, out, bestL = a
    if not run.PK: run.PK = run.load(os.environ['SQ_BEST'])
    wd = f'{out}/t{n}'; os.makedirs(wd, exist_ok=True); rng = random.Random(n); bestc = None; res = []
    for d in (1, 2):
        if n + d not in run.PK: continue
        P = run.PK[n + d]; L = run.side(P); c = run.corners(P).reshape(len(P), 4, 2)
        edge = np.minimum.reduce([c[:, :, 0].min(1), c[:, :, 1].min(1), L - c[:, :, 0].max(1), L - c[:, :, 1].max(1)])
        cand = list(np.argsort(edge)[:14]); rng.shuffle(cand)
        for k in range(6):
            drop = set(cand[k * d:(k + 1) * d]) if d == 1 else set(rng.sample(cand, 2))
            Q = np.array([p for i, p in enumerate(P) if i not in drop])
            try: q = run.quench(Q, secs, f't{n}', wd)
            except Exception: q = None
            if q is not None:
                s = run.side(q); res.append(s)
                if s < bestL[n] - 1e-9 and (bestc is None or s < bestc[0]): bestc = (s, q)
    if bestc: np.save(f'{out}/win_{n}_t.npy', bestc[1])
    return (n, round(bestL[n], 9), round(min(res), 9) if res else None, len(res))
if __name__ == '__main__':
    best, out, secs = sys.argv[1], sys.argv[2], int(sys.argv[3]); ns = [int(x) for x in sys.argv[4:]]
    os.makedirs(out, exist_ok=True); os.environ['SQ_BEST'] = os.path.abspath(best)
    bestL = {int(k): v for k, v in json.load(open(f'{run.ROOT}/watch/best_L.json')).items()}
    with ProcessPoolExecutor(5) as ex:
        for r in ex.map(job, [(n, secs, out, bestL) for n in ns]): print(r, flush=True)
