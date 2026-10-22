import time, math
from sq import *
for n in [5,10,17]:
    t=time.time()
    P,Lg=grid(n)
    best,L=shrink(n,Lg,math.sqrt(n),tries=10,iters=300000)
    print(n,Lg,L, None if best is None else (side(best), valid(normalize(best))), round(time.time()-t,1))
