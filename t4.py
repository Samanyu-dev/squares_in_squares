import time, numpy as np, math
from multiprocessing import Pool
from hard import *
from sq2 import anneal_fast
def job(a):
    n, L, iters, seed, mode = a
    if mode == 'hard': return harden(n, L, iters, seed) is not None
    rng=np.random.default_rng(seed); P=np.c_[rng.uniform(.5,L-.5,n),rng.uniform(.5,L-.5,n),rng.uniform(0,1.57,n)]
    return anneal_fast(P,L,S,iters,0.3,2e-3,seed)<1e-22
if __name__=='__main__':
    job((5,2.8,1000,1,'hard'))
    for n,L in [(17,4.71),(18,4.85),(26,5.70),(37,6.70),(50,7.70),(65,8.85)]:
        for mode in ('hard','plain'):
            t=time.time()
            with Pool(10) as p: r=p.map(job,[(n,L,n*60000,s,mode) for s in range(20)])
            print(n,L,mode,'success',sum(r),'/20',round(time.time()-t),'s',flush=True)
