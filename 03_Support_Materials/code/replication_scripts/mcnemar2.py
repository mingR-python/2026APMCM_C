import sys, numpy as np, pandas as pd
from scipy import stats
df = pd.read_csv(sys.argv[1] + "/Data02_AfricanPenguins_playbacks_data.csv")
for col in ["ResponseYesNo","HeadsUpYesNo","VocalYesNo","ApproachYesNo"]:
    p = df.pivot_table(index="expID", columns="Type", values=col, aggfunc="first")
    b = int(((p["call"]==1)&(p["control"]==0)).sum()); c = int(((p["call"]==0)&(p["control"]==1)).sum())
    n = b + c
    if n == 0: continue
    lo_p = stats.beta.ppf(0.025, b, c+1) if b>0 else 0.0
    hi_p = stats.beta.ppf(0.975, b+1, c) if c>0 else 1.0
    or_lo = lo_p/(1-lo_p) if lo_p < 1 else np.inf
    or_hi = hi_p/(1-hi_p) if hi_p < 1 else np.inf
    print(f"{col}: b={b} c={c} OR={(b/c if c else float('inf')):.1f} exact p={stats.binomtest(b,n,0.5).pvalue:.2e} OR95%CI=[{or_lo:.1f},{or_hi:.1f}]")
