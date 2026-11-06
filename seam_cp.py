"""Discrete seam refill: remove a band of squares, enumerate candidate poses, choose a non-overlapping set with CP-SAT (cell-cover model), then polish continuously."""
import sys, math, time, numpy as np
from ortools.sat.python import cp_model
import sq
from lns import load, anneal_mask
from sq import normalize, side, pair, total
ANG = [0, 45, 18.43, 24.1, 36.87, 11.3, 30.0, 14.0]
def run(n, path, delta, w=2.2, g=0.1, tl=120, tol=0.04, workers=8):
    P0 = normalize(load(path, n)); L0 = side(P0); Lt = L0 - delta
    h = .5 * (abs(np.cos(P0[:, 2])) + abs(np.sin(P0[:, 2])))
    inband = (P0[:, 0] + h > Lt - w) | (P0[:, 1] + h > Lt - w)
    fro = P0[~inband]; m = int(inband.sum())
    angs = sorted({round(math.radians(a), 6) for a in ANG} | {round(-math.radians(a), 6) for a in ANG[2:]})
    poses = []
    xs = np.arange(0.5, Lt, g)
    for a in angs:
        hh = .5 * (abs(math.cos(a)) + abs(math.sin(a)))
        for x in xs[(xs >= hh - 1e-9) & (xs <= Lt - hh + 1e-9)]:
            for y in xs[(xs >= hh - 1e-9) & (xs <= Lt - hh + 1e-9)]:
                if not (x + hh > Lt - w - 0.6 or y + hh > Lt - w - 0.6): continue
                poses.append((x, y, a))
    # drop poses overlapping frozen squares (with tolerance)
    ok = []
    for (x, y, a) in poses:
        near = fro[(abs(fro[:, 0] - x) < 1.5) & (abs(fro[:, 1] - y) < 1.5)]
        bad = False
        for q in near:
            if pair(x, y, a, q[0], q[1], q[2], 1.0 - tol) > 0: bad = True; break
        if not bad: ok.append((x, y, a))
    cells = {}
    cov = []
    for k, (x, y, a) in enumerate(ok):
        c, s = math.cos(a), math.sin(a); cs = set()
        r = int(0.75 / g) + 1
        ix, iy = int(x / g), int(y / g)
        for i in range(ix - r, ix + r + 1):
            for j in range(iy - r, iy + r + 1):
                dx, dy = (i + .5) * g - x, (j + .5) * g - y
                u, v = dx * c + dy * s, -dx * s + dy * c
                if abs(u) <= 0.5 - 0.5 * g and abs(v) <= 0.5 - 0.5 * g: cs.add((i, j))
        cov.append(cs)
        for cc in cs: cells.setdefault(cc, []).append(k)
    model = cp_model.CpModel(); X = [model.NewBoolVar(f'p{k}') for k in range(len(ok))]
    for cc, ks in cells.items():
        if len(ks) > 1: model.AddAtMostOne([X[k] for k in ks])
    model.Add(sum(X) >= m)
    solver = cp_model.CpSolver(); solver.parameters.max_time_in_seconds = tl; solver.parameters.num_workers = workers
    st = solver.Solve(model)
    res = solver.StatusName(st)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE): return L0, Lt, m, len(ok), res, None
    sel = np.array([ok[k] for k in range(len(ok)) if solver.Value(X[k])])
    P = np.vstack([fro, sel[:m]])
    # continuous polish
    mov = np.zeros(len(P), bool); mov[len(fro):] = True
    for i in np.where(~mov)[0]:
        if np.min(np.hypot(*(P[mov, :2] - P[i, :2]).T)) < 2.0: mov[i] = True
    best = None
    for sd in range(3):
        Q = P.copy(); e = anneal_mask(Q, mov, Lt, sq.S, 1500000, 0.05, 1e-4, sd)
        if e < 1e-22 and side(Q) <= Lt + 1e-9: best = Q; break
    return L0, Lt, m, len(ok), res, best
if __name__ == '__main__':
    n = int(sys.argv[1]); d = float(sys.argv[2]); t = time.time()
    L0, Lt, m, np_, res, b = run(n, sys.argv[3], d, tl=int(sys.argv[4]))
    print(f'n={n} L0={L0:.6f} target={Lt:.6f} band_m={m} poses={np_} cp={res} polished={"YES" if b is not None else "no"} {time.time()-t:.0f}s', flush=True)
    if b is not None: np.save(f'seam_out_{n}.npy', b)
