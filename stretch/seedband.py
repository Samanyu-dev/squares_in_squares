"""rigid corner-to-corner band seeds + fq relax.
usage: seedband.py out_dir secs n1 n2 ...   env NPROC, DS (box-side offsets above sqrt(n)*1.0, csv), KS (band rows), CS (band offsets), TH (tilt deg csv)
A seed = tilted band (k rows at angle th along the main diagonal, centre offset c) + upright squares on two grids anchored to
the lower-right and upper-left corners, dropped when they hit the band or each other.  Seeds with >= n squares lose extras, then fq quench shrinks the box."""
import sys, os, math, json, random
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run

def sat_overlap(a, b, tol=1e-9):
    ca, cb = run.corners(np.array([a]))[0], run.corners(np.array([b]))[0]
    for t in (a[2], a[2] + math.pi / 2, b[2], b[2] + math.pi / 2):
        ax = np.array([math.cos(t), math.sin(t)]); pa, pb = ca @ ax, cb @ ax
        if pa.max() <= pb.min() + tol or pb.max() <= pa.min() + tol: return False
    return True

def seed(n, S, k, c, th):
    """returns array (n,3) feasible inside [0,S]^2 or None"""
    th = math.radians(th); u = np.array([math.cos(th), math.sin(th)]); v = np.array([-math.sin(th), math.cos(th)])
    d = np.array([1., 1.]) / math.sqrt(2); dv = np.array([-1., 1.]) / math.sqrt(2)   # box diagonal frame
    ctr = np.array([S / 2, S / 2]); placed = []
    h = 0.5 * (abs(math.cos(th)) + abs(math.sin(th)))
    for r in range(k):
        off = (r - (k - 1) / 2 + c) * dv
        for t in np.arange(-S, S, 1.0):
            p = ctr + off + t * d
            if h <= p[0] <= S - h and h <= p[1] <= S - h: placed.append((p[0], p[1], th))
    # th is the tilt of band squares; rows follow the diagonal so squares only fit flush at 45 deg, others leave slack handled by quench
    def free(q, tree_pts):
        for o in tree_pts:
            if (q[0] - o[0]) ** 2 + (q[1] - o[1]) ** 2 < 2.0 and sat_overlap(q, o): return False
        return True
    ups = []
    for corner in ((S, 0), (0, S)):
        sx = -1 if corner[0] else 1; sy = -1 if corner[1] else 1
        K = int(S)
        for i in range(K):
            for j in range(K):
                ups.append((corner[0] + sx * (i + .5), corner[1] + sy * (j + .5), 0.0))
    ups.sort(key=lambda q: -(abs(q[0] - S / 2) + abs(q[1] - S / 2)))
    for q in ups:
        if q[0] < .5 - 1e-9 or q[0] > S - .5 + 1e-9 or q[1] < .5 - 1e-9 or q[1] > S - .5 + 1e-9: continue
        if free(q, placed): placed.append(q)
    seed.last = len(placed)
    if len(placed) < n: return None
    P = np.array(placed)
    # drop extras: those farthest from the band centreline first (keeps the band), but always feasible
    if len(P) > n:
        dist = np.abs((P[:, 0] - S / 2) - (P[:, 1] - S / 2)) / math.sqrt(2)
        keep = np.argsort(dist)[:n]; P = P[np.sort(keep)]
    return P

def job(a):
    n, S, k, c, th, secs, out = a
    P = seed(n, S, k, c, th)
    if P is None: return (n, S, k, c, th, 'none')
    tag = f'b{n}_{S:.2f}_{k}_{c}_{th}'; wd = f'{out}/w{n}'; os.makedirs(wd, exist_ok=True)
    L0 = run.side(P)
    q = run.quench(P, secs, tag, wd)
    if q is None: return (n, S, k, c, th, round(L0, 4), 'fqfail')
    L = run.side(q); rng = random.Random(hash(tag) & 0xffff); cur = q
    for _ in range(int(os.environ.get('KICKS', 12))):
        sg = rng.uniform(float(os.environ.get('SIGLO', 0.002)), float(os.environ.get('SIGHI', 0.02))); p = cur.copy(); g = np.random.default_rng(rng.randrange(1 << 30))
        p[:, :2] += g.normal(0, sg, (len(p), 2)); p[:, 2] += g.normal(0, sg, len(p))
        c2 = run.quench(p, int(os.environ.get('KSEC', 45)), 'k' + tag, wd, loosen=rng.uniform(0.5, 2.0))
        if c2 is not None and run.side(c2) < L: cur, L = c2, run.side(c2)
    q = cur
    best = json.load(open(f'{run.ROOT}/watch/best_L.json'))
    if L < best[str(n)] - 1e-9: np.save(f'{out}/win_{n}_{tag}.npy', q)
    return (n, S, k, c, th, round(L0, 4), round(L, 6), round(best[str(n)], 6))

if __name__ == '__main__':
    out, secs = sys.argv[1], int(sys.argv[2]); ns = [int(x) for x in sys.argv[3:]]; os.makedirs(out, exist_ok=True)
    DS = [float(x) for x in os.environ.get('DS', '0.1,0.25,0.4').split(',')]
    KS = [int(x) for x in os.environ.get('KS', '3,4,5,6,7').split(',')]
    CS = [float(x) for x in os.environ.get('CS', '0,0.5').split(',')]
    TH = [float(x) for x in os.environ.get('TH', '45').split(',')]
    jobs = []
    for n in ns:
        base = math.sqrt(n)
        for ds in DS:
            S = math.ceil((base + ds) * 100) / 100
            for k in KS:
                for c in CS:
                    for th in TH: jobs.append((n, S, k, c, th, secs, out))
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(int(os.environ.get('NPROC', 4))) as ex:
        for r in ex.map(job, jobs): print(r, flush=True)
