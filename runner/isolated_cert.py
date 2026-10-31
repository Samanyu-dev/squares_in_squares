"""Certify every puzzle of a CSV in its OWN fresh process (no leftover precision state). Usage: isolated_cert.py work_dir in.csv out.json [n1,n2,...]"""
import sys, os, json, time, multiprocessing as mp
import pandas as pd
work = os.path.abspath(sys.argv[1]); csvp = os.path.abspath(sys.argv[2]); outp = os.path.abspath(sys.argv[3])
only = [int(v) for v in sys.argv[4].split(',')] if len(sys.argv) > 4 else None
def job(a):
    n, text = a
    sys.path.insert(0, work); os.chdir(work)
    import official_metric as m
    puzzles = m._parse_submission_csv(text)
    r = m.evaluate_single_puzzle(n, puzzles[n])
    return n, (float(r[2]) if r[2] is not None else 0.0), str(r[3])[:80]
if __name__ == '__main__':
    d = pd.read_csv(csvp); ns = only or sorted(d.n.unique())
    tasks = [(int(n), d[d.n == n][['n', 'square_id', 'x', 'y', 'angle']].to_csv(index=False)) for n in ns]
    t = time.time(); res = {}
    with mp.get_context('spawn').Pool(10, maxtasksperchild=1) as pool:
        for n, L, msg in pool.imap_unordered(job, tasks):
            res[n] = {'L': L, 'msg': msg}
    json.dump(res, open(outp, 'w'))
    bad = sorted(n for n, v in res.items() if v['L'] == 0.0)
    print(f'isolated certification of {len(res)} puzzles in {time.time() - t:.0f}s | rejected in isolation: {len(bad)} {bad[:40]}')
