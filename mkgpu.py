import json, pickle, math, os, re
master = pickle.load(open('master.pkl', 'rb'))
START = {n: round(master[n][1], 9) for n in range(2, 301) if master[n][1] < math.ceil(math.sqrt(n)) - 1e-9}
WEAK = [11, 17, 18, 19, 26, 27, 28, 37, 38, 39, 50]
def src(f, drop=()):
    t = open(f).read()
    return "\n".join(l for l in t.splitlines() if not any(l.startswith(d) for d in drop))
parts = [
 src('sq.py'),
 src('sq2.py', ('from sq import',)),
 src('hard.py', ('from sq import', 'from sq2 import')),
 src('chain.py', ('from sq2 import',)).split("def run_chain")[0],
 src('gpu.py'),
 f"START = {START!r}\nWEAK = {WEAK!r}",
 src('gpu_driver.py'),
]
code = "import numpy as np, math, os\n" + "\n\n".join(parts)
code = code.replace("from multiprocessing import Pool\n", "")
os.makedirs('kernel_gpu', exist_ok=True)
nb = {"cells": [{"cell_type": "code", "metadata": {}, "source": code.splitlines(True), "outputs": [], "execution_count": None}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open('kernel_gpu/squares-gpu-search.ipynb', 'w'))
open('kernel_gpu/run_local.py', 'w').write(code)
json.dump({"id": "samanyu1808/squares-gpu-search", "title": "Squares GPU search", "code_file": "squares-gpu-search.ipynb", "language": "python", "kernel_type": "notebook", "is_private": "true", "enable_gpu": "true", "enable_internet": "false", "dataset_sources": [], "competition_sources": [], "kernel_sources": []}, open('kernel_gpu/kernel-metadata.json', 'w'), indent=1)
print('ok', len(code), 'chars; start Ls for', len(START), 'puzzles')
