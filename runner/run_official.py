import sys, pandas as pd, time, os
csv_path = os.path.abspath(sys.argv[1]); work = os.path.abspath(sys.argv[2])
sys.path.insert(0, work); os.chdir(work)
import official_metric as m
if __name__ == '__main__':
    sub = pd.read_csv(csv_path)                      # default dtypes, as Kaggle's backend does
    sol = pd.DataFrame({'id': range(1000), 'n': range(1, 1001)})
    t = time.time(); s = m.score(sol, sub, 'id')
    print(f'OFFICIAL SCORE {s:.10f}  ({time.time()-t:.0f}s)')
