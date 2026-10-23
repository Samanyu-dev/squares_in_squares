"""Search better rectangular gadgets. Usage: gsearch.py w,h,dm [w,h,dm ...]  (writes gadgets/gadget_w_h_dm.npy when improved)"""
import numpy as np, math, sys, os, time
from multiprocessing import Pool
from hardr import *
F4 = np.load('fault/F4b.npy')
def job(a):
    w, h, dm, f, iters, seed = a
    P = hardenR(w * h + dm, w + f, h + f, iters, seed)
    if P is None: return None
    P, f2 = compressR(P, w, h, f, max(300, iters // 40), seed + 1)
    return P, f2
if __name__ == '__main__':
    os.makedirs('gadgets', exist_ok=True); t0 = time.time()
    with Pool(10) as pool:
        for spec in sys.argv[2:] if sys.argv[1] == 'x' else sys.argv[1:]:
            w, h, dm = (int(v) for v in spec.split(','))
            f0 = float(F4[0, w, h, dm]); best = f0; bestP = None; n = w * h + dm
            step = 0.03; seed = int(time.time()) % 100000; rounds = 0; secs = float(os.environ.get('G_SECS', 120))
            t = time.time()
            while time.time() - t < secs and step > 5e-4:
                ft = best - step
                if ft <= 0.05: step *= 0.5; continue
                res = pool.map(job, [(w, h, dm, ft, 40000 * n, seed + s) for s in range(10)]); seed += 10; rounds += 1
                ok = [r for r in res if r is not None]
                if ok:
                    P, f2 = min(ok, key=lambda r: r[1])
                    if f2 < best - 1e-7: best, bestP = f2, P
                    step *= 1.3
                else: step *= 0.6
            tag = 'IMPROVED' if bestP is not None else 'no gain'
            print(f'gadget ({w},{h},{dm}) n={n}: catalog f={f0:.5f} -> {best:.5f}  [{tag}]  rounds={rounds}  {time.time() - t:.0f}s', flush=True)
            if bestP is not None: np.save(f'gadgets/gadget_{w}_{h}_{dm}.npy', np.c_[bestP, np.full(len(bestP), best)])
