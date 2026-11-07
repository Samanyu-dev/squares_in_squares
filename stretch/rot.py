import sys,json,math,numpy as np
import run
PK=run.load('../watch/best_current.csv'); bl={int(k):v for k,v in json.load(open('../watch/best_L.json')).items()}
res=[]
for n in sorted(PK):
    P=np.array(PK[n],float)
    if n<2 or (np.abs(((np.degrees(P[:,2])+45)%90)-45)<0.05).all(): continue
    c=P[:,:2].mean(0); best=(run.side(P),0)
    for ph in np.linspace(-math.pi/4,math.pi/4,3601):
        co,si=math.cos(ph),math.sin(ph); Q=P.copy(); d=P[:,:2]-c
        Q[:,0]=c[0]+co*d[:,0]-si*d[:,1]; Q[:,1]=c[1]+si*d[:,0]+co*d[:,1]; Q[:,2]=P[:,2]+ph
        s=run.side(Q)
        if s<best[0]-1e-9: best=(s,ph)
    if best[1]!=0: res.append((bl[n]-best[0],n))
print(len(res),sorted(res,reverse=True)[:5])
