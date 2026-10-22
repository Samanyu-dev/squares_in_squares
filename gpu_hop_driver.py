import os, time, math, json, io, base64
import numpy as np
CH = int(os.environ.get('SQ_CHAINS', 160)); MULT = int(os.environ.get('SQ_MULT', 2000))
TOTAL = float(os.environ.get('SQ_TOTAL', 4800)); PER = float(os.environ.get('SQ_PER', 300))
OUT = os.environ.get('SQ_OUT', 'gpu_results'); os.makedirs(OUT, exist_ok=True)
LOG = open(f'{OUT}/log.txt', 'a')
def log(m):
    line = f'[{time.strftime("%H:%M:%S")}] {m}'; print(line, flush=True); LOG.write(line + '\n'); LOG.flush()
CANDS = {int(k): v for k, v in np.load(io.BytesIO(base64.b64decode(BLOB))).items()}
ORDER = [int(v) for v in os.environ.get('SQ_NLIST', '').split(',') if v] or ORDER_DEFAULT
log(f'GPU hop driver: chains={CH} mult={MULT} total={TOTAL}s per_n={PER}s, {len(ORDER)} puzzles queued')
def bbox(P):
    c, s = np.cos(P[:, 2]), np.sin(P[:, 2]); h = .5 * (abs(c) + abs(s))
    return max((P[:, 0] + h).max() - (P[:, 0] - h).min(), (P[:, 1] + h).max() - (P[:, 1] - h).min())
t_all = time.time(); found = {}
for n in ORDER:
    if time.time() - t_all > TOTAL: log('total time budget reached'); break
    Pb = normalize(CANDS[n].copy()); L0 = bbox(Pb)
    Lb = L0; step = 0.015; t0 = time.time(); seed = n * 1000; rounds = 0; improved = False
    while time.time() - t0 < PER and step > 4e-4 and time.time() - t_all < TOTAL:
        Lt = Lb - step
        R = hop_gpu(Pb, Lb, Lt, CH, MULT * n, seed); seed += 1; rounds += 1
        ok = [c for c in range(CH) if htotal(R[c], Lt, 0.0, S_HOPCHK) < 1e-18]
        if ok:
            res = []
            for c in ok[:2]:
                P, L2 = compress(R[c], Lt, max(300, 100 * n), seed + c, d0=0.002, dmin=2e-6); res.append((L2, P))
            L2, P = min(res, key=lambda z: z[0])
            if L2 < Lb - 1e-9: Lb, Pb, improved = L2, P, True
            log(f'n={n} round {rounds}: {len(ok)}/{CH} feasible at {Lt:.6f} -> best {Lb:.6f}'); step *= 1.3
        else:
            log(f'n={n} round {rounds}: 0/{CH} feasible at {Lt:.6f}'); step *= 0.6
    if improved:
        found[n] = (Pb, Lb); np.save(f'{OUT}/P_{n}.npy', Pb)
        json.dump({str(k): v[1] for k, v in found.items()}, open(f'{OUT}/summary.json', 'w'))
        log(f'RESULT n={n}: L={Lb:.6f} (start {L0:.6f}) improvement {L0 - Lb:.6f}')
    else:
        log(f'RESULT n={n}: no improvement over {L0:.6f}')
log(f'done. improved {len(found)} puzzles')
