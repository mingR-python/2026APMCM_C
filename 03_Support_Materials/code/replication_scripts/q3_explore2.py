# -*- coding: utf-8 -*-
import sys, numpy as np, pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
DATA = sys.argv[1]
c2 = pd.read_csv(DATA + "/Data02_AfricanPenguins_playbacks_data.csv")
def hr(t): print("\n" + "="*16, t, "="*16)

hr("call vs control 的距离分配")
print(c2.groupby("Type").Distance.describe().round(1).to_string())
piv = c2.pivot_table(index="expID", columns="Type", values="Distance", aggfunc="first").dropna()
piv["diff"] = piv["call"] - piv["control"]
print("\ncall 更近 / 相同 / 更远:", int((piv["diff"]<0).sum()), int((piv["diff"]==0).sum()), int((piv["diff"]>0).sum()))
print("距离差分布:", piv["diff"].value_counts().sort_index().to_dict())
print("\n示例 expID:")
print(c2[c2.expID.isin([1,2,3,4,5])][["expID","Type","Context","CallType","Distance","Npenguins","ResponseYesNo"]].to_string(index=False))
print("\n每个 expID 内 CallType 一致性(control 无类型):")
print(c2.groupby("expID").Context.nunique().value_counts().to_dict())

hr("Context × Npenguins 交互表(call)")
s = c2[c2.Type=="call"].copy()
s["nband"] = pd.cut(s.Npenguins,[0,3,8,15,40],labels=["1-3","4-8","9-15","16-40"])
print(pd.crosstab(s.Context, s.nband).to_string())
print("\nContext × 响应率:")
print(s.groupby("Context").ResponseYesNo.agg(["size","mean"]).round(3).to_string())

hr("logistic 模型比较(call 子集)")
s["logD"] = np.log(s.Distance)
forms = {
 "M0 截距": "ResponseYesNo ~ 1",
 "M1 距离": "ResponseYesNo ~ logD",
 "M2 规模": "ResponseYesNo ~ Npenguins",
 "M3 情境": "ResponseYesNo ~ C(Context)",
 "M4 规模+情境": "ResponseYesNo ~ Npenguins + C(Context)",
 "M5 距离+规模+情境": "ResponseYesNo ~ logD + Npenguins + C(Context)",
 "M6 +类型": "ResponseYesNo ~ logD + Npenguins + C(Context) + C(CallType)",
 "M7 +类型+交互": "ResponseYesNo ~ logD * Npenguins + C(Context) + C(CallType)",
 "M8 类型单独": "ResponseYesNo ~ C(CallType)",
}
for k,f in forms.items():
    try:
        m = smf.logit(f, data=s).fit(disp=0, maxiter=300)
        print(f"{k:16s} AIC={m.aic:7.2f} BIC={m.bic:7.2f} pseudoR2={m.prsquared:.3f}")
    except Exception as e:
        print(k, "ERR", e)
m = smf.logit("ResponseYesNo ~ logD + Npenguins + C(Context)", data=s).fit(disp=0, maxiter=300)
print("\nM5 系数:\n", m.summary2().tables[1].round(4).to_string())
print("Odds ratios:", np.round(np.exp(m.params),3).to_dict())

hr("配对条件 logistic：call vs control")
long = c2.copy()
long["is_call"] = (long.Type=="call").astype(int)
long["logD"] = np.log(long.Distance)
try:
    from statsmodels.discrete.conditional_models import ConditionalLogit
    y = long.ResponseYesNo.values
    X = np.column_stack([long.is_call.values, long.logD.values, long.Npenguins.values])
    cm = ConditionalLogit(y, X, groups=long.expID.values).fit(disp=0, maxiter=500)
    print("coef is_call/logD/N:", np.round(cm.params,4), "p:", np.round(cm.pvalues,6))
except Exception as e:
    print("ConditionalLogit ERR", repr(e))
# 普通 logistic + cluster robust
m2 = smf.logit("ResponseYesNo ~ is_call + logD + Npenguins + C(Context)", data=long).fit(disp=0, cov_type="cluster", cov_kwds={"groups": long.expID})
print("\n普通 logistic + expID 聚类稳健 SE:")
print(m2.summary2().tables[1].round(4).to_string())
print("is_call OR:", round(float(np.exp(m2.params['is_call'])),2))

hr("8 组合 call 组内 × CallType（Fisher 精确）")
comb = c2[["HeadsUpYesNo","VocalYesNo","ApproachYesNo"]].astype(int).astype(str).agg("".join, axis=1)
c2["combo"] = comb
s2 = c2[c2.Type=="call"]
tab = pd.crosstab(s2.combo, s2.CallType)
print(tab.to_string())
print("\n各 CallType 组合熵 / 非000率:")
for ct in ["flat","mod","tv"]:
    p = tab[ct]/tab[ct].sum()
    ent = -float((p[p>0]*np.log(p[p>0])).sum())
    print(f"  {ct}: 非000={1-p.get('000',0):.3f}, 熵={ent:.3f}, 有接近={float(tab.loc[[i for i in tab.index if i[2]=='1'], ct].sum()/tab[ct].sum()):.3f}")

hr("潜伏期：仅响应者，按类型")
for ct in ["flat","mod","tv"]:
    for b in ["HeadsUp","Vocal","Approach"]:
        v = s2.loc[s2.CallType==ct, f"{b}Latency"].dropna()
        if len(v)>=3:
            print(f"{ct:4s} {b:8s} n={len(v):2d} 中位={v.median():5.1f}s 均值={v.mean():5.1f}")
print("\n按 Context 的声响应潜伏期:")
print(c2.groupby(["Type","Context"]).VocalLatency.agg(["count","median","mean"]).round(2).to_string())
