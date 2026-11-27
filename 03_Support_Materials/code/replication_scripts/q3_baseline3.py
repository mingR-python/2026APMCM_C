# -*- coding: utf-8 -*-
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
df["first_lat"] = df[["HeadsUpLatency","VocalLatency","ApproachLatency"]].min(axis=1)
df["responded"] = df.ResponseYesNo.astype(int)

def tab(m, name):
    d = pd.DataFrame({"coef":m.params, "se":m.bse, "z":m.tvalues, "p":m.pvalues})
    ci = m.conf_int(); d["lo"]=ci[0]; d["hi"]=ci[1]
    print(f"\n--- {name} ---")
    print(d.round(4).to_string())
    return d

m2 = smf.gee("responded ~ C(CallType) + logD + Npenguins + C(Context)",
             groups="expID", data=df, family=sm.families.Binomial(),
             cov_struct=sm.cov_struct.Exchangeable()).fit()
d2 = tab(m2, "GEE logistic: CallType 4 levels")
print("OR:", {k: round(float(np.exp(v)),2) for k,v in m2.params.items() if k!="Intercept"})
print("OR CI:", {k: [round(float(np.exp(lo)),2), round(float(np.exp(hi)),2)] for k,(lo,hi) in m2.conf_int().iterrows() if k!="Intercept"})

m1 = smf.gee("responded ~ is_call + logD + Npenguins + C(Context)",
             groups="expID", data=df, family=sm.families.Binomial(),
             cov_struct=sm.cov_struct.Exchangeable()).fit()
tab(m1, "GEE logistic: is_call")
print("OR is_call:", round(float(np.exp(m1.params["is_call"])),2), "CI:", [round(float(np.exp(x)),2) for x in m1.conf_int().loc["is_call"]])

pos = df[(df.responded==1) & df.first_lat.notna() & (df.first_lat>0)].copy()
pos["logT"] = np.log(pos.first_lat)
m3 = smf.glm("logT ~ C(CallType) + logD + Npenguins + C(Context)", data=pos,
             family=sm.families.Gaussian()).fit(cov_type="cluster", cov_kwds={"groups": pos.expID})
tab(m3, f"lognormal AFT (n={len(pos)}, sigma={np.sqrt(m3.scale):.3f})")
print("乘性倍数:", {k: round(float(np.exp(v)),3) for k,v in m3.params.items() if k!="Intercept"})

def pred_p(ct,d,ctx="commuting"):
    new = pd.DataFrame({"CallType":pd.Categorical([ct],["control","flat","mod","tv"]),
                        "logD":[np.log(d)],"Npenguins":[8],
                        "Context":pd.Categorical([ctx],["after","commuting","searching"])})
    return float(m2.predict(new).iloc[0])
def pred_med(ct,d,ctx="commuting"):
    new = pd.DataFrame({"CallType":pd.Categorical([ct],["control","flat","mod","tv"]),
                        "logD":[np.log(d)],"Npenguins":[8],
                        "Context":pd.Categorical([ctx],["after","commuting","searching"])})
    return float(np.exp(m3.predict(new).iloc[0]))
print("\n=== 预测表 (Npenguins=8, Context=commuting) ===")
rows=[]
for ct in ["flat","mod","tv"]:
    for d in [50,100,150]:
        rows.append([ct,d,8,round(pred_p(ct,d),3),round(pred_med(ct,d),2)])
print(pd.DataFrame(rows,columns=["CallType","Distance","Npenguins","P(response)","median latency|response (s)"]).to_string(index=False))

print("\n=== CEI = F(d)*P(response) ===")
FILE = {"flat":{"spec":"Data03_correlation1m_flat_call_spectrogram.csv","amp":"Data06_correlation1m_flat_call_amplitude_spectrum.csv","env":"Data09_correlation1m_flat_call_envelope.csv"},
        "mod":{"spec":"Data04_correlation1m_modulated_call_spectrogram.csv","amp":"Data07_correlation1m_modulated_call_amplitude_spectrum.csv","env":"Data10_correlation1m_modulated_call_envelope.csv"},
        "tv":{"spec":"Data05_correlation1m_twovoice_call_spectrogram.csv","amp":"Data08_correlation1m_twovoice_call_amplitude_spectrum.csv","env":"Data11_correlation1m_twovoice_call_envelope.csv"}}
DIST=np.array([4,8,16,32,64,128,256],float)
def Fcurve(ct, w=(1/3,1/3,1/3)):
    A=np.stack([pd.read_csv(DATA+"/"+FILE[ct][k]).values for k in ("spec","amp","env")])
    return np.nanmean((A*np.array(w)[:,None,None]).sum(0),axis=0)
def cei(ct,d,w=(1/3,1/3,1/3)):
    F=np.interp(np.log(d),np.log(DIST),Fcurve(ct,w))
    return float(F*pred_p(ct,d))
for ct in ["flat","mod","tv"]:
    print(f"  {ct}: "+" ".join(f"{d}m={cei(ct,d):.3f}" for d in [4,8,16,32,64,128,256]))
def radius_cei(ct,frac=0.5):
    target=frac*cei(ct,50); ds=np.exp(np.linspace(np.log(4),np.log(256),3000))
    v=np.array([cei(ct,d) for d in ds]); ok=np.where(v>=target)[0]
    return float(ds[ok[-1]]) if len(ok) else np.nan
print("\nCEI 半衰半径(相对 50m):")
for ct in ["flat","mod","tv"]: print(f"  {ct}: {radius_cei(ct):.1f} m")
print("\nCEI 绝对阈值半径:")
for thr in [0.1,0.15,0.2,0.25]:
    out=[]
    for ct in ["flat","mod","tv"]:
        ds=np.exp(np.linspace(np.log(4),np.log(256),3000)); v=np.array([cei(ct,d) for d in ds]); ok=np.where(v>=thr)[0]
        out.append(f"{ct}={ds[ok[-1]]:.0f}" if len(ok) else f"{ct}=<4")
    print(f"  >={thr}: "+", ".join(out))
