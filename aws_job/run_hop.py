import os, sys, json, subprocess, time, glob, shutil, multiprocessing as mp
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = '/opt/ml/model'; os.makedirs(OUT + '/best', exist_ok=True)
TASK_SECS = float(os.environ.get('TASK_SECS', 300)); SEEDS = int(os.environ.get('SEEDS', 1)); TOTAL_SECS = float(os.environ.get('TOTAL_SECS', 1500))
def sh(cmd, **kw): return subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)
print('cpus', os.cpu_count(), flush=True)
t0 = time.time()
r = sh('curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal'); print('rustup rc', r.returncode, r.stderr[-300:], flush=True)
r = sh(f'cd {HERE}/packer && $HOME/.cargo/bin/cargo build --release --bin packer 2>&1 | tail -3'); print(r.stdout, flush=True)
PACKER = f'{HERE}/packer/target/release/packer'; assert os.path.exists(PACKER), 'build failed'
print('built in', round(time.time() - t0), 's', flush=True)
def first_side(path):
    try: return float(open(path).readline().split()[1])
    except Exception: return None
def task(a):
    kind, n, seed, S = a[:4]; secs = a[4] if len(a) > 4 else TASK_SECS
    out = f'/tmp/best_{kind}_{n}_{seed}.txt'
    if kind == 'start':
        start = f'{HERE}/starts/start_{n}.txt'; s0 = first_side(start)
        cmd = [PACKER, 'hop', '--in', start, '--seed', str(seed), '--iters', '100000000', '--out', out, '--quiet']
    else:
        s0 = float(__import__('math').ceil(n ** 0.5))
        cmd = [PACKER, 'hop', '--n', str(n), '--s', repr(S), '--seed', str(seed), '--iters', '100000000', '--out', out, '--quiet']
    try: subprocess.run(cmd, capture_output=True, timeout=secs)
    except subprocess.TimeoutExpired: pass
    s1 = first_side(out) if os.path.exists(out) else None
    if s1 is not None and s1 < s0 - 1e-9:
        shutil.copy(out, f'{OUT}/best/best_{kind}_{n}_{seed}.txt')
        import zlib, base64
        print('IMPROVED_B64', n, kind, seed, repr(s0), repr(s1), base64.b64encode(zlib.compress(open(out, 'rb').read(), 9)).decode(), flush=True)
    return kind, n, seed, s0, s1
tasks = json.load(open(f'{HERE}/starts/tasks.json'))
SHARD = int(os.environ.get('SHARD', 0)); SHARDS = int(os.environ.get('SHARDS', 1))
jobs = [tuple(t) for t in tasks][SHARD::SHARDS]
print('shard', SHARD, 'of', SHARDS, ':', len(jobs), 'tasks', flush=True)
res = []; t1 = time.time()
with mp.Pool(os.cpu_count()) as pool:
    for kind, n, seed, s0, s1 in pool.imap_unordered(task, jobs):
        res.append((kind, n, seed, s0, s1)); gain = None if s1 is None else s0 - s1
        print(f'n={n} {kind} seed={seed} start={s0:.9f} best={s1} gain={gain}  [{len(res)}/{len(jobs)} t={time.time() - t1:.0f}s]', flush=True)
        json.dump(res, open(f'{OUT}/results.json', 'w'))
print('DONE', round(time.time() - t0), 's total; improved', sum(1 for r in res if r[4] is not None and r[4] < r[3] - 1e-9), 'of', len(res), flush=True)
