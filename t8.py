import os, time
os.environ['NUMBA_ENABLE_CUDASIM'] = '1'; os.environ['SQ_BD'] = '4'
import numpy as np
from gpu import *
from hard import htotal, harden
IT = 2000
print('CPU success at', IT, 'iters:', sum(harden(5, 2.965443, IT, s) is not None for s in range(40)), '/40', flush=True)
t = time.time()
R = harden_gpu(5, 2.965443, chains=8, iters=IT, seed=11)
e = [htotal(R[c], 2.965443, 0.0, S_CHK) for c in range(8)]
print('GPU energies', [round(x, 6) for x in e], 'feasible', sum(x < 1e-18 for x in e), '/8', round(time.time() - t), 's')
