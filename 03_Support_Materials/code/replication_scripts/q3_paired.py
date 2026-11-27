# -*- coding: utf-8 -*-
import sys, numpy as np, pandas as pd
from scipy import stats
DATA = sys.argv[1]
df = pd.read_csv(DATA + "/Data02_AfricanPenguins_playbacks_data.csv")
df["combo"] = df[["HeadsUpYesNo","VocalYesNo","ApproachYesNo"]].astype(int).astype(str).agg("".join, axis=1)
piv = df.pivot_table(index="expID", columns="Type", values="combo", aggfunc="first")
print("配对组合表:")
print(pd.crosstab(piv["call"], piv["control"]).to_string())

# Stuart-Maxwell / 配对卡方（8类）
def stuart_maxwell(tab):
    tab = np.asarray(tab, float)
    k = tab.shape[0]
    n = tab.sum()
    p = tab.sum(1)/n
    q = tab.sum(0)/n
    # 只用不一致对
    T = 0.0
    # 构造 (k-1)x(k-1) 协方差
    # 简化: 采用 McNemar-Bowker 的置换检验替代
    return None

# 置换检验：在每对内部随机交换 call/control 标签
rng = np.random.default_rng(42)
def combo_table(d):
    call = d[d.Type=="call"].set_index("expID").combo
    ctrl = d[d.Type=="control"].set_index("expID").combo
    return pd.crosstab(call, ctrl).reindex(index=sorted(df.combo.unique()), columns=sorted(df.combo.unique()), fill_value=0)
obs = combo_table(df)
# 统计量：8x8 列联表的卡方（call vs control 边际分布差异）
def stat_from_table(tab):
    marg = np.column_stack([tab.sum(1).values, tab.sum(0).values])
    return stats.chi2_contingency(marg)[0]
obs_stat = stat_from_table(obs)
cnt = 0; B = 5000
base = df.copy()
for _ in range(B):
    swap = rng.random(len(piv)) < 0.5
    d = base.copy()
    # 交换 expID 的 call/control 的 combo 标签
    idx = d.index
    for eid, sw in zip(piv.index, swap):
        if not sw: continue
        m = d.expID == eid
        types = d.loc[m, "Type"].values
        combos = d.loc[m, "combo"].values
        d.loc[m, "combo"] = combos[::-1]
    if stat_from_table(combo_table(d)) >= obs_stat:
        cnt += 1
print(f"配对置换检验（8类组合分布）p = {(cnt+1)/(B+1):.4f}, 统计量={obs_stat:.2f}")

# McNemar for each behavior
for col in ["ResponseYesNo","HeadsUpYesNo","VocalYesNo","ApproachYesNo"]:
    p = df.pivot_table(index="expID", columns="Type", values=col, aggfunc="first")
    b = int(((p["call"]==1)&(p["control"]==0)).sum()); c = int(((p["call"]==0)&(p["control"]==1)).sum())
    # exact CI for OR = b/c using binomial
    lo, hi = stats.beta.ppf([0.025,0.975], b+0.5, c+0.5)[::-1] if b>0 and c>0 else (np.nan,np.nan)
    orr = b/c if c else np.inf
    print(f"{col}: b={b} c={c} OR={orr:.1f} McNemar p={stats.binomtest(b,b+c,0.5).pvalue:.2e}  approx95%CI=[{lo:.1f},{hi:.1f}]")

# call 组内 按 context / distance / Npenguins
s = df[df.Type=="call"]
print("\ncall 组内 Context:")
print(s.groupby("Context").ResponseYesNo.agg(["size","sum","mean"]).round(3).to_string())
print("\ncall 组内 Distance 档:")
s2 = s.assign(dband=pd.cut(s.Distance,[0,30,50,80,100,200],labels=["5-30","31-50","51-80","81-100","101-200"]))
print(s2.groupby("dband",observed=True).ResponseYesNo.agg(["size","sum","mean"]).round(3).to_string())
print("\ncall 组内 Npenguins 档:")
s3 = s.assign(nband=pd.cut(s.Npenguins,[0,3,8,15,40],labels=["1-3","4-8","9-15","16-40"]))
print(s3.groupby("nband",observed=True).ResponseYesNo.agg(["size","sum","mean"]).round(3).to_string())

# 期望响应个体数（示意）：典型群体 N=10.73，P(any) 与响应比例
print("\n群体规模分布:")
g = df.groupby("expID").Npenguins.first()
print("mean",round(g.mean(),2),"median",g.median(),"sd",round(g.std(),2))
print("P(any) overall by N band")
print(s3.groupby("nband",observed=True).ResponseYesNo.mean().round(3).to_dict())
