import sys, numpy as np, pandas as pd
from scipy import stats
df = pd.read_csv(sys.argv[1] + "/Data02_AfricanPenguins_playbacks_data.csv")
for col in ["ResponseYesNo","HeadsUpYesNo","VocalYesNo","ApproachYesNo"]:
    p = df.pivot_table(index="expID", columns="Type", values=col, aggfunc="first")
    b = int(((p["call"]==1)&(p["control"]==0)).sum()); c = int(((p["call"]==0)&(p["control"]==1)).sum())
    lo, hi = stats.beta.ppf([0.025,0.975], b+0.5, c+0.5)
    # beta.ppf gives lower/upper for Beta(b+.5,c+.5); OR CI = (lo,hi) reversed? For OR=b/c, use beta on b+0.5,c+0.5 directly
    print(f"{col}: b={b} c={c} OR={b/c if c else float('inf'):.1f} p={stats.binomtest(b,b+c,0.5).pvalue:.2e} OR95%CI=[{lo:.2f},{hi:.2f}]")
