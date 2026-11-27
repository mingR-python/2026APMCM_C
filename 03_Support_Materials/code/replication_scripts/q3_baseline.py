# -*- coding: utf-8 -*-
"""Q3 基线模型（统计上站得住的版本）：GEE logistic + 对数正态 AFT + 首响应时间。
仅用 statsmodels，输出可写进论文的数值表。"""
import sys, numpy as np, pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

DATA = sys.argv[1]
df = pd.read_csv(DATA + "/Data02_AfricanPenguins_playbacks_data.csv")
df["CallType"] = pd.Categorical(df.CallType, ["control","flat","mod","tv"])
df["Context"] = pd.Categorical(df.Context, ["after","commuting","searching"])
df["logD"] = np.log(df.Distance)
df["is_call"] = (df.Type=="call").astype(int)

# 首响应时间 = 三个 latency 的最小非空值；无响应 = 删失
lat_cols = ["HeadsUpLatency","VocalLatency","ApproachLatency"]
df["first_lat"] = df[lat_cols].min(axis=1)
df["responded"] = df.ResponseYesNo.astype(int)
print("首响应时间非空:", int(df.first_lat.notna().sum()), " 响应=1:", int((df.responded==1).sum()))
print("响应但首响应为空:", int(((df.responded==1)&df.first_lat.isna()).sum()))
print("未响应但有首响应:", int(((df.responded==0)&df.first_lat.notna()).sum()))
print("首响应分位:", df.loc[df.responded==1,"first_lat"].quantile([0,.25,.5,.75,.9,1]).round(2).to_dict())

# ---- 第一段 GEE logistic
print("\n=== GEE logistic: ResponseYesNo ~ is_call + logD + Npenguins + Context ===")
m1 = smf.gee("responded ~ is_call + logD + Npenguins + C(Context)",
             groups="expID", data=df, family=sm.families.Binomial(),
             cov_struct=sm.cov_struct.Exchangeable()).fit()
print(m1.summary())
print("OR is_call:", round(float(np.exp(m1.params["is_call"])),2))

print("\n=== GEE logistic: + CallType (4 levels) ===")
m2 = smf.gee("responded ~ C(CallType) + logD + Npenguins + C(Context)",
             groups="expID", data=df, family=sm.families.Binomial(),
             cov_struct=sm.cov_struct.Exchangeable()).fit()
print(m2.summary())

# ---- 第二段 lognormal AFT（对数正态 = log(y) 的正态 GLM）
print("\n=== lognormal AFT: log(first_lat) ~ C(CallType) + logD + Npenguins + C(Context) ===")
pos = df[(df.responded==1) & df.first_lat.notna() & (df.first_lat>0)].copy()
pos["logT"] = np.log(pos.first_lat)
m3 = smf.glm("logT ~ C(CallType) + logD + Npenguins + C(Context)",
             data=pos, family=sm.families.Gaussian()).fit(
             cov_type="cluster", cov_kwds={"groups": pos.expID})
print(m3.summary())
print("残差 sigma:", round(float(np.sqrt(m3.scale)),3))

# ---- 预测表：Npenguins=8, Distance 50/100/150, CallType flat/mod/tv, Context=commuting
NMED = 8
CTX = "commuting"
rows = []
for ct in ["flat","mod","tv"]:
    for d in [50,100,150]:
        new = pd.DataFrame({"CallType":pd.Categorical([ct],["control","flat","mod","tv"]),
                            "logD":[np.log(d)], "Npenguins":[NMED],
                            "Context":pd.Categorical([CTX],["after","commuting","searching"])})
        p = float(m1.predict(new, linear=True).pipe(lambda x: 1/(1+np.exp(-x))).iloc[0])
        # 用 m3 预测条件中位潜伏期
        new2 = pd.DataFrame({"CallType":pd.Categorical([ct],["control","flat","mod","tv"]),
                             "logD":[np.log(d)], "Npenguins":[NMED],
                             "Context":pd.Categorical([CTX],["after","commuting","searching"])})
        logmu = float(m3.predict(new2).iloc[0]); med = float(np.exp(logmu))
        # delta method CI for median
        X = m3.model.exog[-1:] if False else None
        rows.append([ct,d,NMED,p,med])
print("\n基线预测表（Context=commuting）:")
print(pd.DataFrame(rows, columns=["CallType","Distance","Npenguins","P(response)","median first latency (s)"]).round(4).to_string(index=False))

# ---- CEI 与基于指数的半径
print("\n=== 用 F(d) 等权复合 + P(response) 构造 CEI ===")
def load_F(ct):
    m = {"flat":"flat","mod":"modulated","tv":"twovoice"}[ct]
    files = []
    for tag in ["spectrogram","amplitude_spectrum","envelope"]:
        if ct=="flat": f=f"Data03_correlation1m_flat_call_{tag}.csv" if tag!="amplitude_spectrum" else "Data06_correlation1m_flat_call_amplitude_spectrum.csv"
        elif ct=="mod": f=f"Data04_correlation1m_modulated_call_{tag}.csv" if tag!="amplitude_spectrum" else "Data07_correlation1m_modulated_call_amplitude_spectrum.csv"
        else: f=f"Data05_correlation1m_twovoice_call_{tag}.csv" if tag!="amplitude_spectrum" else "Data08_correlation1m_twovoice_call_amplitude_spectrum.csv"
        files.append(pd.read_csv(DATA+"/"+f).values)
    A = np.stack(files)
    return np.nanmean(A.mean(0), axis=1)  # 逐距离均值再对三指标平均
DIST = np.array([4,8,16,32,64,128,256],float)
for ct in ["flat","mod","tv"]:
    print(ct, "F:", np.round(load_F(ct),3))

def P_any(ct,d,ctx=CTX):
    new = pd.DataFrame({"CallType":pd.Categorical([ct],["control","flat","mod","tv"]),
                        "logD":[np.log(d)], "Npenguins":[NMED],
                        "Context":pd.Categorical([ctx],["after","commuting","searching"])})
    eta = float(m1.predict(new, linear=True).iloc[0]); return 1/(1+np.exp(-eta))

def cei(ct,d):
    F = np.interp(np.log(d), np.log(DIST), load_F(ct))
    return float(F * P_any(ct,d))

print("\nCEI 曲线:")
for ct in ["flat","mod","tv"]:
    vals = [cei(ct,d) for d in [4,8,16,32,64,128,256]]
    print(ct, np.round(vals,3))
# 以 50m 为参考，定义 CEI 降到 50% 的距离
def radius_cei(ct, frac=0.5):
    c50 = cei(ct,50); target = frac*c50
    ds = np.exp(np.linspace(np.log(4), np.log(256), 2000))
    vals = np.array([cei(ct,d) for d in ds])
    idx = np.where(vals >= target)[0]
    return float(ds[idx[-1]]) if len(idx) else np.nan
for ct in ["flat","mod","tv"]:
    print(ct, "CEI 半衰半径(50m 基准):", round(radius_cei(ct),1), "m")
