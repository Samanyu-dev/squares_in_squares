import json, io, base64, math, os
import numpy as np, pandas as pd
t = pd.read_pickle('table_merged2.pkl')
cand = t[(t.n >= 325) & (t.n <= 700) & (t.type == '0/45')].sort_values('excess', ascending=False).head(45)
d = pd.read_csv('merged2.csv'); sel = set(cand.n)
arrs = {}
for n, g in d[d.n.isin(sel)].groupby('n'):
    arrs[str(int(n))] = g.sort_values('square_id')[['x', 'y', 'angle']].to_numpy(float)
buf = io.BytesIO(); np.savez_compressed(buf, **arrs); b64 = base64.b64encode(buf.getvalue()).decode()
order = [int(n) for n in cand.n]
def src(f, drop=()):
    return "\n".join(l for l in open(f).read().splitlines() if not any(l.startswith(x) for x in drop))
parts = [src('sq.py'), src('sq2.py', ('from sq import',)), src('hard.py', ('from sq import', 'from sq2 import')),
         src('chain.py', ('from sq2 import',)).split("def run_chain")[0], src('gpu.py'), src('gpu_hop.py'),
         f"BLOB = '{b64}'\nORDER_DEFAULT = {order!r}", src('gpu_hop_driver.py')]
code = "import numpy as np, math, os\n" + "\n\n".join(parts).replace("from multiprocessing import Pool\n", "")
os.makedirs('kernel_gpu', exist_ok=True)
nb = {"cells": [{"cell_type": "code", "metadata": {}, "source": code.splitlines(True), "outputs": [], "execution_count": None}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open('kernel_gpu/squares-gpu-search.ipynb', 'w')); open('kernel_gpu/run_local.py', 'w').write(code)
print('ok', len(code) // 1024, 'KB code;', len(order), 'puzzles queued; n range', min(order), '-', max(order), '; sum excess of queued', round(cand.excess.sum(), 3))
