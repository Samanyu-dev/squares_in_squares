import json, pickle, base64, io, sys, math, numpy as np, os, re
from assemble import load
files = sys.argv[1:]
db = load(files)
cores = {n: np.round(P, 13) for n, (P, L) in db.items() if n <= 400 and L < math.ceil(math.sqrt(n)) - 1e-9}
buf = io.BytesIO(); np.savez_compressed(buf, **{str(n): P for n, P in cores.items()})
b64 = base64.b64encode(buf.getvalue()).decode()
def src(f, drop=()):
    t = open(f).read().split("if __name__")[0]
    return "\n".join(l for l in t.splitlines() if not any(l.startswith(d) for d in drop))
sq = open('sq.py').read()
asm = src('assemble.py', ('from sq import', 'import pickle'))
bld = src('build.py', ('from sq import', 'import pickle'))
cells = [
("markdown", "# Squares in Squares - own solver\nSmall-n packings found by overlap-energy annealing (embedded below, n<=400), grown to every n<=1000 by exact L-shaped border extension, monotone closure, then strict independent verification. Anything that fails verification falls back to the trivial grid."),
("code", sq), ("code", "import pickle, math, numpy as np\n" + asm), ("code", "import pickle\n" + bld),
("code", f"import io, base64, numpy as np\nBLOB = '{b64}'\nCORES = {{int(k): v for k, v in np.load(io.BytesIO(base64.b64decode(BLOB))).items()}}\nprint(len(CORES), 'embedded core packings')"),
("code", "db = {n: (P, side(normalize(P))) for n, P in CORES.items()}\nbest = assemble(db)\nwrite(best, 'submission.csv')"),
]
nb = {"cells": [{"cell_type": t, "metadata": {}, "source": s.splitlines(True), **({"outputs": [], "execution_count": None} if t == "code" else {})} for t, s in cells],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open('kernel/squares-own-solver.ipynb', 'w')); print('notebook ok, cores', len(cores), 'b64 KB', len(b64) // 1024)
