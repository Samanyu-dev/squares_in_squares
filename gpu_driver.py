import os, time, math, json
import numpy as np
CH = int(os.environ.get('SQ_CHAINS', 320)); MULT = int(os.environ.get('SQ_MULT', 30000))
TOTAL = float(os.environ.get('SQ_TOTAL', 1500)); PER = float(os.environ.get('SQ_PER', 150))
NLIST = [int(v) for v in os.environ.get('SQ_NLIST', '').split(',') if v] or list(WEAK) + [n for n in range(51, 161) if n not in WEAK]
OUT = os.environ.get('SQ_OUT', 'gpu_results'); os.makedirs(OUT, exist_ok=True)
LOG = open(f'{OUT}/log.txt', 'a')
def log(m):
    line = f'[{time.strftime("%H:%M:%S")}] {m}'; print(line, flush=True); LOG.write(line + '\n'); LOG.flush()
log(f'GPU driver: chains={CH} mult={MULT} total={TOTAL}s per_n={PER}s, {len(NLIST)} puzzles queued')
t_all = time.time(); found = {}
for n in NLIST:
    if time.time() - t_all > TOTAL: log('total time budget reached'); break
    r = math.sqrt(n); L0 = float(math.ceil(r)) if os.environ.get('SQ_NOSTART') else START.get(n, float(math.ceil(r)))
    if math.ceil(r) - r < 1e-9: continue
    best_L, best_P = L0, None; step = 0.04 * (L0 - r) + 0.004; t0 = time.time(); seed = n * 1000; rounds = 0
    while time.time() - t0 < PER and step > 5e-5 and time.time() - t_all < TOTAL:
        Lt = best_L - step
        if Lt <= r: step *= 0.5; continue
        R = harden_gpu(n, Lt, CH, MULT * n, seed); seed += 1; rounds += 1
        ok = [c for c in range(CH) if htotal(R[c], Lt, 0.0, S_CHK) < 1e-18]
        if ok:
            res = []
            for c in ok[:3]:
                P, L2 = compress(R[c], Lt, max(300, MULT * n // 20), seed + c, d0=0.005, dmin=2e-6); res.append((L2, P))
            L2, P = min(res, key=lambda z: z[0])
            if L2 < best_L - 1e-9: best_L, best_P = L2, P
            log(f'n={n} round {rounds}: {len(ok)}/{CH} feasible at {Lt:.6f} -> best {best_L:.6f}'); step *= 1.3
        else:
            log(f'n={n} round {rounds}: 0/{CH} feasible at {Lt:.6f}'); step *= 0.55
    if best_P is not None:
        found[n] = (best_P, best_L); np.save(f'{OUT}/P_{n}.npy', best_P)
        log(f'RESULT n={n}: L={best_L:.6f} (start {L0:.6f}) improvement {L0 - best_L:.6f}')
        json.dump({str(k): v[1] for k, v in found.items()}, open(f'{OUT}/summary.json', 'w'))
    else:
        log(f'RESULT n={n}: no improvement over {L0:.6f}')
log(f'done. improved {len(found)} puzzles')
