import pickle, math, sys, numpy as np, glob
from sq import grid, side, normalize, valid
EPS = 1e-8

def add_border(P, L):
    k = math.floor(L); Ln = L + 1 + EPS
    col = [(L + EPS + .5, .5 + i, 0.0) for i in range(math.floor(Ln))]
    top = [(.5 + i, L + EPS + .5, 0.0) for i in range(k)]
    return np.vstack([P, np.array(col + top)]), Ln

def load(files):
    db = {}
    for f in files:
        for n, (P, L) in pickle.load(open(f, 'rb')).items():
            if n <= 1000 and (n not in db or L < db[n][1] - 1e-12): db[n] = (P, L)
    return db

def assemble(db, core_max=400):
    best = {n: grid(n) for n in range(1, 1001)}
    for n, (P, L) in db.items():
        if L < best[n][1]: best[n] = (P, L)
    for n0 in sorted(db):
        if n0 > core_max: continue
        P, L = normalize(db[n0][0]), side(normalize(db[n0][0])); n = n0
        while True:
            P, L = add_border(P, L); n += math.floor(L - EPS - 1 + 1e-9) * 2 + 1 if False else (math.floor(L) + math.floor(L - 1 - EPS))
            if n > 1000: break
            if L < best[n][1] - 1e-12: best[n] = (P, L)
    for n in range(999, 0, -1):   # monotone: drop a square from n+1
        if best[n + 1][1] < best[n][1] - 1e-12:
            best[n] = (best[n + 1][0][:n], best[n + 1][1])
    return best

if __name__ == '__main__':
    db = load(sys.argv[2:]); best = assemble(db)
    for n in range(1, 1001):
        P, L = best[n]
        if len(P) != n: raise SystemExit(f'count mismatch {n} {len(P)}')
    sc = sum(best[n][1] / math.sqrt(n) - 1 for n in best); gs = sum(math.ceil(math.sqrt(n)) / math.sqrt(n) - 1 for n in best)
    print('assembled score', round(sc, 4), 'grid', round(gs, 4))
    pickle.dump(best, open(sys.argv[1], 'wb'))
