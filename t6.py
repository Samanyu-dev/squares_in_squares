import os, time
os.environ['NUMBA_ENABLE_CUDASIM'] = '1'
import numpy as np, math
from numba import cuda
import gpu
from gpu import *
from hard import hpair, hwall, htotal

@cuda.jit
def kpair(A, out, rho, s):
    k = cuda.grid(1)
    if k < A.shape[0]:
        out[k] = dpair(A[k,0],A[k,1],A[k,2],A[k,3],A[k,4],A[k,5],rho,s)

rng = np.random.default_rng(0)
A = np.c_[rng.uniform(0,1.6,(300,2)), rng.uniform(0,1.57,300), rng.uniform(0,1.6,(300,2)), rng.uniform(0,1.57,300)].astype(np.float32)
bad = 0
for rho in (0.5, 0.3, 0.1, 0.0):
    out = np.zeros(300, np.float32); kpair[3,128](A, out, np.float32(rho), np.float32(1.0))
    ref = np.array([hpair(*map(float,a), rho, 1.0) for a in A])
    err = np.abs(out - ref).max(); bad += err > 1e-4
    print('rho', rho, 'max abs diff dpair vs hpair', err, 'nonzero pairs', int((ref>0).sum()))
print('pair functions agree' if not bad else 'MISMATCH')

# energy trend over a short run
n, L = 5, 2.9
P0 = np.c_[rng.uniform(.5, L-.5, n), rng.uniform(.5, L-.5, n), rng.uniform(0, 1.57, n)]
print('start energy', htotal(P0.astype(np.float64), L, 0.3, 1.0))
dP = cuda.to_device(P0[None].astype(np.float32))
states = cuda.to_device(np.array([123456789], dtype=np.int64))
for step in range(4):
    kern[1, BD](dP, n, np.float32(L), np.float32(0.3), np.float32(1.0), 60, np.float32(0.3), np.float32(2e-3), states)
    print('after', (step+1)*60, 'moves energy', htotal(dP.copy_to_host()[0].astype(np.float64), L, 0.3, 1.0))

