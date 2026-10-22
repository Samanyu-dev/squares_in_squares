"""Deep, cross-verified solve of ONE n at a time. Usage: onebyone.py N0 N1 SECS_PER_N"""
import numpy as np, math, sys, time, pickle, json, os
from multiprocessing import Pool
from hard import *
from chain import compress
import assemble, build
from sq import grid, side, normalize

LIVE = 'results/live.log'
# values I recall from the literature (Friedman survey etc.); used only as a sanity reference, NOT as proof
LIT = {1: 1.0, 2: 2.0, 3: 2.0, 4: 2.0, 5: 2 + 1 / math.sqrt(2), 6: 3.0, 7: 3.0, 8: 3.0, 9: 3.0, 10: 3 + 1 / math.sqrt(2),
       11: 3.8771, 17: 4.6756, 18: 4.8229, 19: 4.8856, 26: 5.6214, 27: 5.7071, 28: 5.8284}
PROVEN_GRID = {2, 3, 4, 5 * 0 + 6, 7, 8}   # s(n)=ceil(sqrt n) established in the literature (Kearney-Shiu etc.)

def log(msg):
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    open(LIVE, 'a').write(line + '\n')

def hjob(a):
    n, L, iters, seed = a
    P = harden(n, L, iters, seed)
    if P is None: return None
    P, L2 = compress(P, L, max(300, iters // 20), seed + 1, d0=0.005, dmin=2e-6)
    return P, L2

def verify(n, P, L):
    """independent cross-checks; returns dict"""
    out = {}
    Pf, Lf = build.finalize(n, P, L)
    out['count_ok'] = len(Pf) == n
    out['finite_ok'] = bool(np.isfinite(Pf).all())
    out['bbox_L'] = build.bbox_side(Pf)
    out['min_gap'] = float(build.min_gap(Pf))
    out['gap_ok'] = out['min_gap'] >= (0 if abs(out['bbox_L'] - math.ceil(math.sqrt(n))) < 1e-9 else build.MIN_GAP) - 1e-15
    # CSV round-trip: serialise exactly as submitted, parse back, re-validate
    lines = [f'{i},{n},{i},{float(x)!r},{float(y)!r},{float(t)!r}' for i, (x, y, t) in enumerate(Pf)]
    Q = np.array([[float(v) for v in l.split(',')[3:]] for l in lines])
    out['roundtrip_L'] = build.bbox_side(Q); out['roundtrip_gap'] = float(build.min_gap(Q))
    out['roundtrip_ok'] = abs(out['roundtrip_L'] - out['bbox_L']) < 1e-12 and out['roundtrip_gap'] >= out['min_gap'] - 1e-12
    # second, separately written overlap test: shapely-free polygon clipping via exact area of intersection samples is overkill;
    # use brute force vertex-in-polygon test as an independent geometric witness (touching allowed)
    c, s = np.cos(Pf[:, 2]), np.sin(Pf[:, 2]); bad = 0
    V = []
    for k in range(len(Pf)):
        R = np.array([[c[k], -s[k]], [s[k], c[k]]])
        V.append((Pf[k, :2] + (R @ np.array([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]]).T).T, R, Pf[k, :2]))
    for i in range(len(Pf)):
        for j in range(len(Pf)):
            if i == j: continue
            vi = V[i][0]; R = V[j][1]; cj = V[j][2]
            loc = (R.T @ (vi - cj).T).T
            if (np.abs(loc) < 0.5 - 1e-9).all(axis=1).any(): bad += 1   # a corner strictly inside another square
    out['vertex_inside_count'] = bad
    out['witness_ok'] = bad == 0
    out['ok'] = all(out[k] for k in ('count_ok', 'finite_ok', 'gap_ok', 'roundtrip_ok', 'witness_ok'))
    return out, Pf, Lf

def solve_one(n, secs, pool, master):
    r = math.sqrt(n); g = math.ceil(r); t0 = time.time()
    log(f'===== n={n}  area lower bound {r:.6f}  grid {g}  literature ref {LIT.get(n, "n/a")}')
    if g - r < 1e-9:
        log(f'n={n} is a perfect square: grid L={g} equals the area bound -> OPTIMAL (proof: L>=sqrt(n))')
        best = grid(n); status = 'optimal (area bound)'
    elif n in PROVEN_GRID and False:
        pass
    else:
        best = master[n]
        log(f'start from master: L={best[1]:.6f} (grid {g})')
        # sanity probe that grid-optimal cases really resist: always try below the best for the allotted time
        seed = int(time.time() * 1000) % (1 << 28); step = 0.04 * (best[1] - r) + 0.004; rounds = 0
        while time.time() - t0 < secs and step > 5e-5:
            Lt = best[1] - step
            if Lt <= r: step *= 0.5; continue
            iters = 60000 * n
            res = pool.map(hjob, [(n, Lt, iters, seed + s) for s in range(10)]); seed += 10; rounds += 1
            ok = [x for x in res if x is not None]
            if ok:
                P, L2 = min(ok, key=lambda x: x[1])
                if L2 < best[1] - 1e-9:
                    best = (P, L2); log(f'  round {rounds}: {len(ok)}/10 feasible at {Lt:.6f} -> new best L={L2:.6f}')
                step *= 1.3
            else:
                log(f'  round {rounds}: 0/10 feasible at {Lt:.6f}  (step {step:.5f})'); step *= 0.55
        status = 'searched'
    P, L = best
    chk, Pf, Lf = verify(n, P, L)
    log(f'VERIFY n={n}: ' + ', '.join(f'{k}={v}' for k, v in chk.items()))
    lit = LIT.get(n)
    cmp_ = f'vs literature {lit:.4f}: {Lf - lit:+.4f}' if lit else 'no literature ref in table'
    log(f'RESULT n={n}: L={Lf:.8f}  score(n)={Lf / r - 1:.6f}  gap to area bound {Lf - r:.6f}  {cmp_}  [{status}]  verified={chk["ok"]}  time {time.time() - t0:.0f}s')
    return (Pf, Lf) if chk['ok'] else grid(n), chk, status

if __name__ == '__main__':
    n0, n1, secs = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
    master = pickle.load(open('master.pkl', 'rb'))
    summ = json.load(open('results/summary.json')) if os.path.exists('results/summary.json') else {}
    with Pool(10) as pool:
        for n in range(n0, n1 + 1):
            res, chk, status = solve_one(n, secs, pool, master)
            if res[1] < master[n][1] - 1e-9 or n not in master: master[n] = res
            pickle.dump(master, open('master.pkl', 'wb'))
            np.save(f'results/n{n:04d}.npy', res[0])
            summ[str(n)] = {'L': res[1], 'score': res[1] / math.sqrt(n) - 1, 'status': status, 'verified': chk['ok'], 'min_gap': chk['min_gap']}
            json.dump(summ, open('results/summary.json', 'w'), indent=1)
            tot = sum(v['score'] for v in summ.values())
            log(f'SAVED n={n}. completed {len(summ)} puzzles, their score sum {tot:.5f}')
