import time, numpy as np, math, sys
from multiprocessing import Pool
from sq2 import *
def job(a):
    n, L, iters, seed = a
    rng=np.random.default_rng(seed)
    P=np.c_[rng.uniform(.5,L-.5,n),rng.uniform(.5,L-.5,n),rng.uniform(0,1.57,n)]
    e=anneal_fast(P,L,S,iters,0.3,2e-3,seed)
    return e
if __name__=='__main__':
    for n,L in [(50,7.6),(50,7.5),(101,10.8),(101,10.7),(150,12.6)]:
        t=time.time()
        with Pool(10) as p: r=p.map(job,[(n,L,n*20000,s) for s in range(10)])
        print(n,L,'best E',min(r),'ok',sum(e<1e-22 for e in r),round(time.time()-t))
