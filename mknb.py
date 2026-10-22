import json, pickle, base64, zlib, io, sys, numpy as np
db = pickle.load(open(sys.argv[1], 'rb'))
import math
blob = {n: np.round(P, 12) for n, (P, L) in db.items() if L < math.ceil(math.sqrt(n)) - 1e-6}
buf = io.BytesIO(); np.savez_compressed(buf, **{str(n): P for n, P in blob.items()})
b64 = base64.b64encode(buf.getvalue()).decode()
sq = open('sq.py').read(); sv = open('solve.py').read().split("if __name__")[0]
sv = sv.replace("from sq import *", "")
cells = [
("markdown", "# Squares in Squares: own solver\nOverlap-energy annealing (SAT penetration depth) + greedy container shrinking, warm-started from neighbouring n. Everything not improved falls back to the trivial grid. Output: `submission.csv`."),
("code", sq),
("code", sv),
("code", f"import io, base64, numpy as np\nBLOB = '{b64}'\nSOL = {{int(k): v for k, v in np.load(io.BytesIO(base64.b64decode(BLOB))).items()}}\nprint(len(SOL), 'precomputed packings')"),
("markdown", "Optional: re-run the search on this machine to improve further (set `RUN=True`, takes a while)."),
("code", "RUN = False\nif RUN:\n    import math, time\n    t0 = time.time()\n    for n in sorted(SOL):\n        P0 = SOL[n]; L0 = side(P0)\n        P, L = compress(normalize(P0), L0 + 1e-6, seed=int(time.time()) % 1000 + n)\n        if total(P, L, S) < 1e-22 and L < L0 - 1e-9: SOL[n] = P\n        if time.time() - t0 > 3000: break"),
("code", "import math\nrows = []; score = 0; bad = 0\nfor n in range(1, 1001):\n    P = normalize(SOL[n]) if n in SOL else grid(n)[0]\n    if n > 1 and not valid(P):\n        P = grid(n)[0]; bad += 1\n    score += side(P) / math.sqrt(n) - 1\n    for i, (x, y, t) in enumerate(P):\n        rows.append(f'{len(rows)},{n},{i},{float(x)!r},{float(y)!r},{float(t)!r}')\nopen('submission.csv', 'w').write('id,n,square_id,x,y,angle\\n' + '\\n'.join(rows) + '\\n')\nprint(len(rows), 'rows; local score', round(score, 4), '; fallbacks', bad)"),
]
nb = {"cells": [{"cell_type": t, "metadata": {}, "source": s.splitlines(True), **({"outputs": [], "execution_count": None} if t == "code" else {})} for t, s in cells],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
import os; os.makedirs('kernel', exist_ok=True)
json.dump(nb, open('kernel/squares-own-solver.ipynb', 'w'))
json.dump({"id": "REPLACE/squares-own-solver", "title": "Squares own solver", "code_file": "squares-own-solver.ipynb", "language": "python", "kernel_type": "notebook", "is_private": "true", "enable_gpu": "false", "enable_internet": "false", "dataset_sources": [], "competition_sources": ["squares-in-squares"], "kernel_sources": []}, open('kernel/kernel-metadata.json', 'w'), indent=1)
print('ok', len(b64))
