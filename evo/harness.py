"""program-search harness: every progs/*.py defines gen(n, S, rng) -> (m,3) feasible poses in [0,S]^2 with m>=n or None.
Each program is run on target puzzles at a few box sides; seeds are trimmed to n, fq-quenched (60 s); result is logged with a counter.
usage: harness.py out.csv n1 n2 ...   env NPROC"""
import sys, os, glob, importlib.util, json, math, random, csv
import numpy as np
sys.path.insert(0, '../stretch'); import run
sys.path.insert(0, 'progs')
bl = {int(k): v for k, v in json.load(open('../watch/best_L.json')).items()}
def load(p):
    sp = importlib.util.spec_from_file_location(os.path.basename(p)[:-3], p); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
def job(a):
    prog, n, ds, seed = a
    m = load(prog); rng = random.Random(seed); S = math.ceil((bl[n] - ds) * 1000) / 1000; wd = f'w/{os.path.basename(prog)[:-3]}_{n}_{ds}_{seed}'; os.makedirs(wd, exist_ok=True)
    try: P = m.gen(n, S, rng)
    except Exception as e: return (os.path.basename(prog), n, ds, seed, 'error', str(e)[:60])
    if P is None or len(P) < n: return (os.path.basename(prog), n, ds, seed, 'short', 0 if P is None else len(P))
    P = np.asarray(P)[:n]
    if run.overlaps(P): return (os.path.basename(prog), n, ds, seed, 'overlap', 0)
    q = run.quench(P, 60, 't', wd)
    if q is None: return (os.path.basename(prog), n, ds, seed, 'fqfail', 0)
    L = run.side(q)
    if L < bl[n] - 1e-9: np.save(f'win_{n}_{os.path.basename(prog)[:-3]}.npy', q)
    return (os.path.basename(prog), n, ds, seed, 'ok', round(L, 6), round(bl[n], 6), round(bl[n] - L, 6))
if __name__ == '__main__':
    out = sys.argv[1]; ns = [int(x) for x in sys.argv[2:]]; os.makedirs('w', exist_ok=True)
    jobs = [(p, n, ds, s) for p in sorted(glob.glob('progs/*.py')) for n in ns for ds in (0.05, 0.2, 0.5) for s in range(3)]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(int(os.environ.get('NPROC', 4))) as ex, open(out, 'w', newline='') as f:
        w = csv.writer(f)
        for r in ex.map(job, jobs): w.writerow(r); f.flush()
