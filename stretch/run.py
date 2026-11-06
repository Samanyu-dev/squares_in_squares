"""stretch + fq driver. usage: run.py best.csv out_dir seconds [ns...]  -> out_dir/win_<n>.csv for packings beating bestL"""
import sys, os, math, json, subprocess, csv
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stretch
from scipy.spatial import cKDTree
FQ = os.environ.get('FQ', '/Users/apple/Desktop/squares_in_squares/tools/evand/search/packer/target/release/fq')
ROOT = os.environ.get('SQ_ROOT', '/Users/apple/Desktop/squares_in_squares')
OFF = np.array([(-.5, -.5), (.5, -.5), (.5, .5), (-.5, .5)])
def corners(s):
    c, si = np.cos(s[:, 2]), np.sin(s[:, 2])
    return np.stack([s[:, None, 0] + OFF[None, :, 0] * c[:, None] - OFF[None, :, 1] * si[:, None], s[:, None, 1] + OFF[None, :, 0] * si[:, None] + OFF[None, :, 1] * c[:, None]], -1)
def normalise(s):
    s = np.array(s, float); s[:, :2] -= corners(s).reshape(-1, 2).min(0); return s
def side(s):
    p = corners(s).reshape(-1, 2); return float(max(np.ptp(p[:, 0]), np.ptp(p[:, 1])))
def overlaps(s, tol=1e-9):
    cs = corners(s); p = cKDTree(s[:, :2]).query_pairs(math.sqrt(2) + 1e-6, output_type='ndarray')
    if len(p) == 0: return 0
    I, J = p[:, 0], p[:, 1]
    ci, si, cj, sj = np.cos(s[I, 2]), np.sin(s[I, 2]), np.cos(s[J, 2]), np.sin(s[J, 2])
    axes = np.stack([np.stack([ci, si], -1), np.stack([-si, ci], -1), np.stack([cj, sj], -1), np.stack([-sj, cj], -1)], 1)
    pi, pj = np.einsum('mkd,mad->mak', cs[I], axes), np.einsum('mkd,mad->mak', cs[J], axes)
    return int(((np.maximum(pi.min(-1), pj.min(-1)) - np.minimum(pi.max(-1), pj.max(-1))).max(-1) < -tol).sum())
def write_cfg(path, s):
    s = normalise(s)
    with open(path, 'w') as f:
        f.write(f'{len(s)} {side(s) + 1e-9:.17f}\n')
        for x, y, t in s: f.write(f'{x:.17f} {y:.17f} {math.degrees(t) % 90:.15f}\n')
def read_cfg(path):
    rows = [l.split() for l in open(path) if l.strip() and not l.startswith('#')]; n = int(rows[0][0])
    return normalise([[float(a), float(b), math.radians(float(c))] for a, b, c, *_ in rows[1:1 + n]])
def quench(s, secs, tag, wd, loosen=1.0):
    a, b = f'{wd}/{tag}_in.cfg', f'{wd}/{tag}_out.cfg'; write_cfg(a, s)
    if os.path.exists(b): os.remove(b)
    try: subprocess.run([FQ, 'quench', '--in', a, '--out', b, '--finish', '--deadline', str(secs), '--loosen', str(loosen)], capture_output=True, timeout=secs + 400)
    except subprocess.TimeoutExpired: return None
    if not os.path.exists(b): return None
    q = read_cfg(b); return q if len(q) == len(s) and overlaps(q) == 0 else None
def load(path):
    g = defaultdict(list)
    for r in csv.DictReader(open(path)): g[int(r['n'])].append((int(r['square_id']), float(r['x']), float(r['y']), float(r['angle'])))
    return {n: normalise([v[1:] for v in sorted(rows)]) for n, rows in g.items()}
PK = {}
def job(args):
    n, steps, secs, out, bestL = args
    global PK
    if not PK: PK = load(os.environ['SQ_BEST'])
    wd = f'{out}/w{n}_{steps}'; os.makedirs(wd, exist_ok=True)
    try:
        b, notes = stretch.stretch(PK[n], steps)
        if b is None or len(b) < 2: return (n, steps, None, 'nostretch')
        m = len(b)
        if m > 1000 or m < 2: return (n, steps, None, 'range')
        q = quench(b, secs, f's{n}_{steps}', wd)
        if q is None: return (n, steps, m, 'fqfail')
        L = side(q); res = (n, steps, m, L, bestL.get(m, 1e9))
        if L < bestL.get(m, 1e9) - 1e-9:
            np.save(f'{out}/win_{m}_{n}_{steps}.npy', q)
        return res
    except Exception as e: return (n, steps, None, repr(e)[:80])
if __name__ == '__main__':
    best, out, secs = sys.argv[1], sys.argv[2], int(sys.argv[3]); ns = [int(x) for x in sys.argv[4:]]
    os.makedirs(out, exist_ok=True)
    os.environ['SQ_BEST'] = os.path.abspath(best); PK = load(best); bestL = {int(k): v for k, v in json.load(open(f'{ROOT}/watch/best_L.json')).items()}
    ns = ns or [n for n in sorted(PK) if n >= 100 and len(stretch.patches(PK[n]))]
    print(len(ns), 'specimens', flush=True)
    jobs = [(n, st, secs, out, bestL) for n in ns for st in (1, -1, 2, -2)]
    with ProcessPoolExecutor(9) as ex:
        for r in ex.map(job, jobs):
            print(r, flush=True)
