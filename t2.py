import time, numpy as np, math
from sq2 import *
n=500; P,L=grid(n); print(L)
# time: shrink 1% from grid k=23 -> should be feasible-ish
for iters in (200000,):
    Q=P.copy(); Q[:,:2]*=0.99; t=time.time(); e=anneal_fast(Q,L*0.99,S,iters,0.1,0.0,1); print(e,time.time()-t)
