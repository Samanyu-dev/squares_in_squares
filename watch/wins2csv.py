import sys,glob,re,numpy as np
sys.path.insert(0,"../stretch"); from run import side
out,files=sys.argv[1],sys.argv[2:]
best={}
for f in files:
    n=int(re.search(r'win_(\d+)_',f).group(1)); P=np.load(f)
    import math
    s=lambda P:max(np.abs(P[:,0])+0,0)
    best.setdefault(n,[]).append(f)
rows=[];i=0
for n,fs in sorted(best.items()):
    for f in sorted(fs,key=lambda f:side(np.load(f)))[:1]:
        P=np.load(f)
        for k,(x,y,a) in enumerate(P): rows.append(f'{i},{n},{k},{float(x)!r},{float(y)!r},{float(a)!r}');i+=1
open(out,'w').write('id,n,square_id,x,y,angle\n'+'\n'.join(rows)+'\n')
