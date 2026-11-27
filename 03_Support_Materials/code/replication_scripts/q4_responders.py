import sys, numpy as np, pandas as pd
DATA=sys.argv[1]
df=pd.read_csv(DATA+"/Data02_AfricanPenguins_playbacks_data.csv")
g=df.groupby("expID").Npenguins.first()
vals,cnts=np.unique(g.values,return_counts=True); p=cnts/cnts.sum()
print("group size dist n=",len(g),"mean",round(g.mean(),2),"median",g.median())
# GEE baseline coefficients
b0=1.4172; bct={'flat':2.6207,'mod':2.7896,'tv':3.1081}; bcomm=-0.1818; blogD=-0.7111; bN=-0.0633
def P(ct,d,n,ctx='commuting'):
    eta=b0+bct[ct]+bcomm+blogD*np.log(d)+bN*n
    return 1/(1+np.exp(-eta))
print("\nP(any) by n at 50m (flat/mod/tv):")
for n in [1,3,5,8,10,15,20,30,40]:
    print(n, [round(P(ct,50,n),3) for ct in ['flat','mod','tv']])
print("\nE[N_resp] over empirical group distribution, rho=1 (all individuals respond if group responds):")
for d in [50,100,150,200]:
    row=[]
    for ct in ['flat','mod','tv']:
        e=sum(pp*n*P(ct,d,n) for n,pp in zip(vals,p))
        row.append(round(float(e),2))
    print(d,"m:",row)
print("\nE[N_resp] over empirical group distribution, rho=0.5:")
for d in [50,100,150,200]:
    row=[]
    for ct in ['flat','mod','tv']:
        e=0.5*sum(pp*n*P(ct,d,n) for n,pp in zip(vals,p))
        row.append(round(float(e),2))
    print(d,"m:",row)
print("\nE[N_resp] at median n=8 and mean n=10.75, rho=1:")
for n in [8,10.75]:
    print("n=",n,{ct: round(n*P(ct,100,n),2) for ct in ['flat','mod','tv']})
