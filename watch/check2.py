"""Certification-aware check. For each public source: certify (with the CURRENT official metric) only the puzzles where it beats our
current certified L; keep a packing only if certified. Never submits.  Usage: check2.py [--no-download]"""
import csv, io, json, os, subprocess, sys, time, shutil, math, re
from collections import defaultdict
ROOT = '/Users/apple/Desktop/squares_in_squares'; W = f'{ROOT}/watch'; PY = f'{ROOT}/.venv/bin/python'; SCT = f'{ROOT}/scoretest_new'
STATE = f'{W}/state.json'; BEST = f'{W}/best_current.csv'; BEST_SCORE = f'{W}/best_score.txt'; BEST_L = f'{W}/best_L.json'
os.makedirs(f'{W}/dl', exist_ok=True); os.makedirs(f'{W}/reports', exist_ok=True); os.makedirs(f'{W}/subsets', exist_ok=True)
logl = []
def log(m):
    l = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {m}'; print(l, flush=True); logl.append(l)
def run(cmd, **kw): return subprocess.run(cmd, capture_output=True, text=True, **kw)
def load(path):
    g = defaultdict(list)
    with open(path) as f:
        next(f)
        for line in f:
            p = line.rstrip('\n').split(',')
            if len(p) >= 6: g[int(p[1])].append(p)
    return g
def fside(rows):
    import numpy as np
    A = np.array([[float(v) for v in r[3:6]] for r in rows]); c, s = np.cos(A[:, 2]), np.sin(A[:, 2]); h = .5 * (abs(c) + abs(s))
    return max((A[:, 0] + h).max() - (A[:, 0] - h).min(), (A[:, 1] + h).max() - (A[:, 1] - h).min())
def certify(path):
    det = f'{SCT}/submission_details.csv'
    if os.path.exists(det): os.remove(det)
    r = run([PY, f'{ROOT}/runner/run_official.py', path, SCT], timeout=6000)
    m = re.search(r'OFFICIAL SCORE ([0-9.]+)', r.stdout); score = float(m.group(1)) if m else None
    L = {}
    if os.path.exists(det):
        for n, v in list(csv.reader(open(det)))[1:]: L[int(n)] = float(v)
    return score, L
def write_csv(path, rows_by_n, ns):
    out = []
    for n in ns:
        for r in sorted(rows_by_n[n], key=lambda r: int(r[2])): out.append(f'{len(out)},{n},{r[2]},{r[3]},{r[4]},{r[5]}')
    open(path, 'w').write('id,n,square_id,x,y,angle\n' + '\n'.join(out) + '\n')
# ---- download new/updated notebooks
if '--no-download' not in sys.argv:
    out = run(['kaggle', 'kernels', 'list', '--competition', 'squares-in-squares', '--sort-by', 'dateRun', '--page-size', '60', '--csv']).stdout
    now = {r['ref']: r['lastRunTime'] for r in csv.DictReader(io.StringIO(out[out.find('ref,'):]))}
    state = json.load(open(STATE)) if os.path.exists(STATE) else {}
    for ref in [r for r, t in now.items() if state.get(r) != t]:
        d = f'{W}/dl/' + ref.replace('/', '_'); shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
        run(['kaggle', 'kernels', 'output', ref, '-p', d, '--file-pattern', 'submission.*csv'], timeout=900)
        f = f'{d}/submission.csv'
        if os.path.exists(f) and os.path.getsize(f) > 1_000_000: log(f'downloaded {ref} ({os.path.getsize(f) // 1_000_000} MB)')
        else: shutil.rmtree(d, ignore_errors=True)
        state[ref] = now[ref]; json.dump(state, open(STATE, 'w'), indent=1)
# ---- baseline certified L per puzzle
best_rows = load(BEST); ns = sorted(best_rows)
if os.path.exists(BEST_L): bestL = {int(k): v for k, v in json.load(open(BEST_L)).items()}
else:
    score, bestL = certify(BEST); log(f'baseline certified: {score}, rejected {sum(1 for v in bestL.values() if v == 0)}')
    for n in ns:
        if bestL.get(n, 0) == 0: bestL[n] = float(math.ceil(math.sqrt(n)))        # rejected -> grid ceiling
    json.dump(bestL, open(BEST_L, 'w'))
base_score = sum(bestL[n] / math.sqrt(n) - 1 for n in ns); log(f'current certified total (from per-puzzle L): {base_score:.10f}')
# ---- sources ordered by potential
srcs = []
for d in sorted(os.listdir(f'{W}/dl')):
    p = f'{W}/dl/{d}/submission.csv'
    if os.path.exists(p): srcs.append(p)
pot = []
for p in srcs:
    g = load(p); gain = 0.0
    for n, rows in g.items():
        if n in bestL and len(rows) == n:
            try: L = fside(rows)
            except Exception: continue
            if L < bestL[n] - 1e-9: gain += (bestL[n] - L) / math.sqrt(n)
    pot.append((gain, p, g))
pot.sort(key=lambda z: -z[0]); log('sources by uncertified potential: ' + ', '.join(f'{os.path.basename(os.path.dirname(p))[:28]}={g:.4f}' for g, p, _ in pot[:8]))
improved = 0
for gain, p, g in pot:
    if gain < 1e-7: continue
    cands = [n for n, rows in g.items() if n in bestL and len(rows) == n and fside(rows) < bestL[n] - 1e-9]
    if not cands: continue
    sub = f'{W}/subsets/subset.csv'; write_csv(sub, g, sorted(cands))
    score, L = certify(sub); won = 0; gsum = 0.0
    for n in cands:
        c = L.get(n, 0.0)
        if c > 0 and c < bestL[n] - 1e-9: gsum += (bestL[n] - c) / math.sqrt(n); bestL[n] = c; best_rows[n] = g[n]; won += 1
    improved += won; json.dump(bestL, open(BEST_L, 'w'))
    log(f'{os.path.basename(os.path.dirname(p))[:40]}: {len(cands)} candidates, {won} certified wins, score gain {gsum:.6f}')
new_score = sum(bestL[n] / math.sqrt(n) - 1 for n in ns)
log(f'after merging: {new_score:.10f} (was {base_score:.10f}); certified wins total {improved}')
if new_score < base_score - 1e-9:
    cand = f'{W}/candidate.csv'; write_csv(cand, best_rows, ns)
    score, L = certify(cand); bad = [n for n in ns if L.get(n, 0) == 0]
    iso_bad = None
    if score is not None and not bad:
        subprocess.run([PY, f'{ROOT}/runner/isolated_cert.py', f'{ROOT}/scoretest_n100', cand, f'{W}/iso_cand.json'], capture_output=True, text=True, timeout=6000)
        iso = json.load(open(f'{W}/iso_cand.json')); iso_bad = [int(n) for n, v in iso.items() if v['L'] <= 0]
        log(f'isolated per-puzzle check: rejected {len(iso_bad)} {iso_bad[:10]}')
    if score is not None and not bad and not iso_bad and abs(score - new_score) < 1e-7:
        ts = time.strftime('%Y%m%d_%H%M'); shutil.copy(cand, BEST); shutil.copy(cand, f'{W}/reports/best_{ts}_{score:.8f}.csv'); open(BEST_SCORE, 'w').write(repr(score))
        log(f'*** NEW CERTIFIED BEST {score:.10f} (gain {base_score - score:.6f}); saved to watch/best_current.csv. NOT submitted - ask the user.')
    else:
        log(f'final whole-file check FAILED (score={score}, rejected={len(bad)}); not promoted'); os.remove(BEST_L)
else: log('no certified improvement')
open(f'{W}/reports/last_run.txt', 'w').write('\n'.join(logl) + '\n')
