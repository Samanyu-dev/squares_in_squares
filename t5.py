import os, sys, time
os.environ['NUMBA_ENABLE_CUDASIM'] = '1'
import numpy as np
from gpu import *
from hard import htotal
# correctness in the simulator: hardening n=5 at L=2.9 (feasible, known optimum 2.707) with a few chains
t = time.time()
R = harden_gpu(5, 2.9, chains=2, iters=400, seed=1, stages=4)
e = [htotal(R[c], 2.9, 0.0, S_CHK) for c in range(2)]
print('energies', e, 'time', round(time.time() - t, 1))
