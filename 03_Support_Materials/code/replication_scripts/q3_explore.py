# -*- coding: utf-8 -*-
"""Q3 可执行性预演：配对结构、McNemar、四因素、组合、潜伏期删失。"""
import sys, os
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

C = sys.argv[1]
c2 = pd.read_csv(os.path.join(C, "Data02_AfricanPenguins_playbacks_data.csv"))

def hr(t):
    print("\n" + "=" * 18, t, "=" * 18)

hr("配对完整性：同一 expID 内 Distance/Npenguins/Context 是否相同")
g = c2.groupby("expID")[["Distance", "Npenguins", "Context"]].nunique()
print((g > 1).sum().to_dict())

hr("McNemar：call vs control 响应")
piv = c2.pivot_table(index="expID", columns="Type", values="ResponseYesNo", aggfunc="first")
b = int(((piv["call"] == 1) & (piv["control"] == 0)).sum())
c = int(((piv["call"] == 0) & (piv["control"] == 1)).sum())
print(f"call=1&ctrl=0: {b}, call=0&ctrl=1: {c}")
print("精确二项检验 p:", stats.binomtest(b, b + c, 0.5).pvalue)

hr("四因素：call 子集响应率的粗表")
s = c2[c2.Type == "call"].copy()
for f in ["Context", "CallType"]:
    print(f"\n-- {f} --")
    print(s.groupby(f).ResponseYesNo.agg(["size", "sum", "mean"]).round(3).to_string())
print("\n-- Distance 分档 --")
s["dband"] = pd.cut(s.Distance, [0, 30, 60, 100, 150, 200], labels=["5-30", "31-60", "61-100", "101-150", "151-200"])
print(s.groupby("dband", observed=True).ResponseYesNo.agg(["size", "sum", "mean"]).round(3).to_string())
print("\n-- Npenguins 分档 --")
s["nband"] = pd.cut(s.Npenguins, [0, 3, 8, 15, 40], labels=["1-3", "4-8", "9-15", "16-40"])
print(s.groupby("nband", observed=True).ResponseYesNo.agg(["size", "sum", "mean"]).round(3).to_string())
print("\n-- CallType × Context --")
print(s.pivot_table(index="CallType", columns="Context", values="ResponseYesNo", aggfunc=["size", "mean"]).round(3).to_string())
print("\n-- CallType × dband --")
print(s.pivot_table(index="CallType", columns="dband", values="ResponseYesNo", aggfunc=["size", "mean"], observed=True).round(3).to_string())
print("\n-- 各因素与 Distance/Npenguins 的相关 --")
print("CallType vs Distance 中位:", s.groupby("CallType").Distance.median().to_dict())
print("Context vs Distance 中位:", s.groupby("Context").Distance.median().to_dict())
print("CallType vs Npenguins 中位:", s.groupby("CallType").Npenguins.median().to_dict())
print("Context vs Npenguins 中位:", s.groupby("Context").Npenguins.median().to_dict())
print("Distance 与 Npenguins Spearman:", stats.spearmanr(s.Distance, s.Npenguins))

hr("logistic 回归（call 子集）：逐步加因素")
s["logD"] = np.log(s.Distance)
formulas = {
    "仅截距": "ResponseYesNo ~ 1",
    "距离": "ResponseYesNo ~ logD",
    "距离+规模": "ResponseYesNo ~ logD + Npenguins",
    "距离+规模+类型": "ResponseYesNo ~ logD + Npenguins + C(CallType)",
    "距离+规模+类型+情境": "ResponseYesNo ~ logD + Npenguins + C(CallType) + C(Context)",
    "全交互": "ResponseYesNo ~ logD * Npenguins + C(CallType) + C(Context)",
}
for name, f in formulas.items():
    try:
        m = smf.logit(f, data=s).fit(disp=0, maxiter=200)
        print(f"{name:18s} AIC={m.aic:7.2f} pseudoR2={m.prsquared:.3f}  llf={m.llf:.2f}")
        if name in ("距离+规模+类型+情境",):
            print(m.summary2().tables[1].round(4).to_string())
    except Exception as e:
        print(name, "ERR", e)

hr("配对条件 logistic（call vs control，按 expID 分层）")
# 用 GLM with cluster-robust 近似 + 条件 logit 手工
long = c2.copy()
long["is_call"] = (long.Type == "call").astype(int)
try:
    m = smf.logit("ResponseYesNo ~ is_call + C(Context) + log(Distance) + Npenguins", data=long.assign(**{"log(Distance)": np.log(long.Distance)})).fit(disp=0)
    print("普通 logistic（未配对）is_call 系数:", round(m.params["is_call"], 3), "OR:", round(np.exp(m.params["is_call"]), 2), "p:", round(m.pvalues["is_call"], 6))
except Exception as e:
    print("ERR", e)
# 条件 logistic 用
try:
    from statsmodels.discrete.conditional_models import ConditionalLogit
    y = long.ResponseYesNo.values
    x = np.column_stack([long.is_call.values, long.Npenguins.values, np.log(long.Distance.values),
                         (long.Context == "searching").astype(float), (long.Context == "after").astype(float)])
    groups = long.expID.values
    cm = ConditionalLogit(y, x, groups=groups).fit(disp=0)
    print("ConditionalLogit 系数:", np.round(cm.params, 4), "p:", np.round(cm.pvalues, 6))
except Exception as e:
    print("ConditionalLogit ERR", repr(e))

hr("8 组合：call vs control 配对差异")
comb = c2[["HeadsUpYesNo", "VocalYesNo", "ApproachYesNo"]].astype(int).astype(str).agg("".join, axis=1)
c2["combo"] = comb
piv2 = c2.pivot_table(index="expID", columns="Type", values="combo", aggfunc="first")
print("call 组合 ∩ control 组合 一致比例:", round(float((piv2["call"] == piv2["control"]).mean()), 3))
print(pd.crosstab(piv2["call"], piv2["control"]).to_string())
print("\n对照中非 000 的组合:", piv2[piv2["control"] != "000"]["control"].value_counts().to_dict())

hr("潜伏期删失结构")
for b_ in ["HeadsUp", "Vocal", "Approach"]:
    y = c2[f"{b_}YesNo"]
    t = c2[f"{b_}Latency"]
    print(f"{b_}: 发生 {int(y.sum())}/{len(y)}, 潜伏期非空 {int(t.notna().sum())}, 发生但潜伏期空 {int(((y==1)&t.isna()).sum())}, 未发生但潜伏期非空 {int(((y==0)&t.notna()).sum())}")
    if b_ == "Vocal":
        print("  发生组潜伏期分位:", c2.loc[y == 1, "VocalLatency"].quantile([0,.25,.5,.75,1]).round(2).to_dict())
        print("  未发生组 VocalNcalls 非零:", int(((y==0)&(c2.VocalNcalls>0)).sum()))

hr("响应组合与 Npenguins/Distance 的关系（call）")
print(s.groupby("nband", observed=True).apply(lambda d: pd.Series({
    "n": len(d), "头": d.HeadsUpYesNo.mean(), "叫": d.VocalYesNo.mean(), "近": d.ApproachYesNo.mean(),
    "组合数": d.combo.nunique() if "combo" in d else np.nan}), include_groups=False).round(3).to_string())
