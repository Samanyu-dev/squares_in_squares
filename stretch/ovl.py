"""overlap-energy shrink: squeeze our packing into side S<L (squares overlap), minimise sum penetration^2 + wall^2 with L-BFGS
(pair gradients by vectorised central differences), basin-hop with kicks; E~0 -> hand to fq quench --finish.
usage: ovl.py best.csv out_dir eps tries n1 n2 ...   env NPROC, KSIG"""
import sys, os, math, json, random
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run

def pen(a, b):
    """a,b (m,3) -> penetration depth (m,) = min over the 4 SAT axes of interval overlap (0 if separated)."""
    ca, cb = run.corners(a), run.corners(b)
    ax = np.stack([np.stack([np.cos(a[:, 2]), np.sin(a[:, 2])], -1), np.stack([-np.sin(a[:, 2]), np.cos(a[:, 2])], -1),
                   np.stack([np.cos(b[:, 2]), np.sin(b[:, 2])], -1), np.stack([-np.sin(b[:, 2]), np.cos(b[:, 2])], -1)], 1)
    pa, pb = np.einsum('mkd,mad->mak', ca, ax), np.einsum('mkd,mad->mak', cb, ax)
    ov = np.minimum(pa.max(-1), pb.max(-1)) - np.maximum(pa.min(-1), pb.min(-1))
    return np.maximum(ov.min(-1), 0.0)

def energy_grad(x, S, pairs):
    P = x.reshape(-1, 3); n = len(P); g = np.zeros_like(P); E = 0.0
    c = run.corners(P); h = 1e-6
    for k in range(2):                       # walls: corners outside [0,S]
        lo = np.maximum(-c[:, :, k], 0); hi = np.maximum(c[:, :, k] - S, 0)
        E += (lo ** 2).sum() + (hi ** 2).sum()
    wall_fd = np.zeros_like(P)
    if E > 0:
        for v in range(3):
            for sg in (1, -1):
                Q = P.copy(); Q[:, v] += sg * h; cq = run.corners(Q); e = 0.0
                e = sum((np.maximum(-cq[:, :, k], 0) ** 2).sum(1) + (np.maximum(cq[:, :, k] - S, 0) ** 2).sum(1) for k in range(2))
                wall_fd[:, v] += sg * e / (2 * h)
    g += wall_fd
    if len(pairs):
        I, J = pairs[:, 0], pairs[:, 1]; A, B = P[I], P[J]; d0 = pen(A, B); E += (d0 ** 2).sum()
        act = d0 > 0
        if act.any():
            Ia, Ja, Aa, Ba = I[act], J[act], A[act], B[act]
            for v in range(3):
                for who in (0, 1):
                    for sg in (1, -1):
                        A2, B2 = Aa.copy(), Ba.copy()
                        (A2 if who == 0 else B2)[:, v] += sg * h
                        d = pen(A2, B2) ** 2
                        np.add.at(g[:, v], Ia if who == 0 else Ja, sg * d / (2 * h))
    return E, g.ravel()

def relax(P, S, iters=400):
    for _ in range(3):
        tr = cKDTree(P[:, :2]); pairs = tr.query_pairs(math.sqrt(2) + 0.05, output_type='ndarray')
        r = minimize(energy_grad, P.ravel(), args=(S, pairs), jac=True, method='L-BFGS-B', options={'maxiter': iters, 'ftol': 1e-30, 'gtol': 1e-14})
        P = r.x.reshape(-1, 3)
    tr = cKDTree(P[:, :2]); pairs = tr.query_pairs(math.sqrt(2) + 0.05, output_type='ndarray')
    return P, energy_grad(P.ravel(), S, pairs)[0]

def job(a):
    n, P0, L, eps, tries, out = a
    rng = random.Random(n); gen = np.random.default_rng(n); S = L * (1 - eps); best = None
    wd = f'{out}/w{n}'; os.makedirs(wd, exist_ok=True)
    P = P0.copy(); P[:, :2] *= S / L; Es = []
    for t in range(tries):
        Q = P.copy()
        if t: Q[:, :2] += gen.normal(0, float(os.environ.get('KSIG', 0.01)) * rng.random(), (n, 2)); Q[:, 2] += gen.normal(0, 0.01 * rng.random(), n)
        Q, E = relax(Q, S); Es.append(E)
        if best is None or E < best[0]: best = (E, Q.copy())
        if E < 1e-14:
            q = run.quench(Q, 40, f'o{n}_{t}', wd)
            if q is not None and run.side(q) < L - 1e-9 and run.overlaps(q) == 0:
                np.save(f'{out}/win_{n}_o.npy', q); return (n, 'WIN', run.side(q), L)
        P = best[1]                              # continue from lowest-energy state
    return (n, 'minE', float(min(Es)), round(S, 6))

if __name__ == '__main__':
    best, out, eps, tries = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]); ns = [int(x) for x in sys.argv[5:]]
    os.makedirs(out, exist_ok=True); PK = run.load(best); bl = json.load(open(f'{run.ROOT}/watch/best_L.json'))
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(n, run.normalise(PK[n]), run.side(PK[n]), eps, tries, out) for n in ns]
    with ProcessPoolExecutor(int(os.environ.get('NPROC', 2))) as ex:
        for r in ex.map(job, jobs): print(r, flush=True)
