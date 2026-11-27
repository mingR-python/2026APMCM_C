import sys, numpy as np, pandas as pd
from scipy import stats
df = pd.read_csv(sys.argv[1] + "/Data02_AfricanPenguins_playbacks_data.csv")
df["combo"] = df[["HeadsUpYesNo","VocalYesNo","ApproachYesNo"]].astype(int).astype(str).agg("".join, axis=1)
codes = {c:i for i,c in enumerate(sorted(df.combo.unique()))}
df["code"] = df.combo.map(codes)
piv = df.pivot_table(index="expID", columns="Type", values="code", aggfunc="first")
C = piv["call"].values.astype(int); K = piv["control"].values.astype(int)
print("codes", codes)
print("配对表:\n", pd.crosstab(pd.Series(C,name='call'), pd.Series(K,name='control')).to_string())

def stat(call, ctrl):
    # 8x2 marginal table
    tab = np.zeros((8,2))
    for v in call: tab[v,0]+=1
    for v in ctrl: tab[v,1]+=1
    tab = tab[tab.sum(1)>0]
    if tab.shape[0] < 2: return 0.0
    return stats.chi2_contingency(tab, correction=False)[0]
obs = stat(C,K)
rng = np.random.default_rng(7)
B = 5000
cnt = 0
for _ in range(B):
    sw = rng.random(len(C)) < 0.5
    cc = np.where(sw, K, C); kk = np.where(sw, C, K)
    if stat(cc,kk) >= obs: cnt += 1
print(f"配对置换检验 p={(cnt+1)/(B+1):.4f}, obs chi2={obs:.2f}")
