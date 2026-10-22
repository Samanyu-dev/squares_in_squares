import os
os.environ['NUMBA_ENABLE_CUDASIM'] = '1'
import numpy as np, math
from numba import cuda
from gpu import *
from hard import htotal
rng = np.random.default_rng(0); n, L, rho = 5, 2.9, 0.3
P0 = np.c_[rng.uniform(.5, L-.5, n), rng.uniform(.5, L-.5, n), rng.uniform(0, 1.57, n)].astype(np.float32)
dP = cuda.to_device(P0[None].copy()); st = cuda.to_device(np.array([123456789], dtype=np.int64))
prev = htotal(P0.astype(np.float64), L, rho, 1.0)
for k in range(12):
    before = dP.copy_to_host()[0].copy()
    kern[1, BD](dP, n, np.float32(L), np.float32(rho), np.float32(1.0), 1, np.float32(0.3), np.float32(2e-3), st)
    after = dP.copy_to_host()[0]
    ch = np.where(np.abs(after - before).max(axis=1) > 0)[0]
    e = htotal(after.astype(np.float64), L, rho, 1.0)
    print(k, 'moved', list(ch), 'dE(cpu)=%.5f' % (e - prev)); prev = e
