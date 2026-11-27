# -*- coding: utf-8 -*-
"""按 claude-6b 的固定 schema 生成 Q3/Q4 修正版结果。

口径：
- Q3 响应：ResponseYesNo；配对 McNemar；GEE(expID 聚类)；Stuart-Maxwell + 配对置换。
- Q3 潜伏期：首响应时间 = min(三种 latency)；两段式（第一段 GEE，第二段仅响应者的对数正态 AFT）。
- Q4 CEI：CEI(d) = F(d) × P_recruit(d)；P_recruit = P(approach ∪ vocal)，GEE(expID)。
- F(d)：三种相关性的等权均值；块 bootstrap（30 块×10 行）。
- 有效半径：相对 κ=0.5×CEI(50m) 为主；绝对 0.15/0.20/0.25 作对照。
- 期望个体数：Data02 的 139 个群体规模经验分布积分；ρ=1/0.5/0.25。
"""
import os, sys, warnings
import numpy as np
import pandas as pd
from scipy import stats, optimize, integrate
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
DATA = sys.argv[1]
OUT = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- 数据
df = pd.read_csv(os.path.join(DATA, "Data02_AfricanPenguins_playbacks_data.csv"))
df["CallType"] = pd.Categorical(df.CallType, ["control", "flat", "mod", "tv"])
df["Context"] = pd.Categorical(df.Context, ["after", "commuting", "searching"])
df["logD"] = np.log(df.Distance)
df["is_call"] = (df.Type == "call").astype(int)
df["responded"] = df.ResponseYesNo.astype(int)
df["first_lat"] = df[["HeadsUpLatency", "VocalLatency", "ApproachLatency"]].min(axis=1)
df["recruit"] = ((df.ApproachYesNo == 1) | (df.VocalYesNo == 1)).astype(int)

DIST = np.array([4, 8, 16, 32, 64, 128, 256], float)
FILE = {
    "flat": ["Data03_correlation1m_flat_call_spectrogram.csv",
             "Data06_correlation1m_flat_call_amplitude_spectrum.csv",
             "Data09_correlation1m_flat_call_envelope.csv"],
    "mod": ["Data04_correlation1m_modulated_call_spectrogram.csv",
            "Data07_correlation1m_modulated_call_amplitude_spectrum.csv",
            "Data10_correlation1m_modulated_call_envelope.csv"],
    "tv": ["Data05_correlation1m_twovoice_call_spectrogram.csv",
           "Data08_correlation1m_twovoice_call_amplitude_spectrum.csv",
           "Data11_correlation1m_twovoice_call_envelope.csv"],
}
ARR = {ct: np.stack([pd.read_csv(os.path.join(DATA, f)).values for f in files]) for ct, files in FILE.items()}
# ARR shape: 3 metrics × 300 rows × 7 distances
BLOCKS = np.repeat(np.arange(30), 10)

def F_curve(ct, idx=None):
    a = ARR[ct]
    if idx is not None:
        a = a[:, idx, :]
    return np.nanmean(a.mean(0), axis=0)  # 7

def F_at(ct, d, idx=None):
    return float(np.interp(np.log(d), np.log(DIST), F_curve(ct, idx)))

# ---------------------------------------------------------------- 1 配对检验
rows = []
for col, name in [("ResponseYesNo", "总响应"), ("HeadsUpYesNo", "抬头"),
                  ("VocalYesNo", "鸣叫"), ("ApproachYesNo", "接近")]:
    p = df.pivot_table(index="expID", columns="Type", values=col, aggfunc="first")
    b = int(((p["call"] == 1) & (p["control"] == 0)).sum())
    c = int(((p["call"] == 0) & (p["control"] == 1)).sum())
    n = b + c
    OR = b / c if c else ""
    lo_p = stats.beta.ppf(0.025, b, c + 1) if b > 0 else 0.0
    hi_p = stats.beta.ppf(0.975, b + 1, c) if c > 0 else 1.0
    OR_lo = lo_p / (1 - lo_p) if lo_p < 1 else ""
    OR_hi = hi_p / (1 - hi_p) if hi_p < 1 else ""
    pv = stats.binomtest(b, n, 0.5).pvalue if n else np.nan
    rows.append([name, b, c, OR, OR_lo, OR_hi, pv])
pd.DataFrame(rows, columns=["行为", "b_call1_ctrl0", "c_call0_ctrl1", "OR", "OR_low", "OR_high", "p_exact"]).to_csv(
    os.path.join(OUT, "Q3_配对检验.csv"), index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------- 2 GEE 四因素
m_any = smf.gee("responded ~ C(CallType) + logD + Npenguins + C(Context)",
                groups="expID", data=df, family=sm.families.Binomial(),
                cov_struct=sm.cov_struct.Exchangeable()).fit()
m_rec = smf.gee("recruit ~ C(CallType) + logD + Npenguins + C(Context)",
                groups="expID", data=df, family=sm.families.Binomial(),
                cov_struct=sm.cov_struct.Exchangeable()).fit()

def gee_rows(m, label):
    out = []
    ci = m.conf_int()
    for k in m.params.index:
        coef = float(m.params[k]); OR = float(np.exp(coef))
        lo, hi = float(np.exp(ci.loc[k, 0])), float(np.exp(ci.loc[k, 1]))
        out.append([label, k, coef, OR, lo, hi, float(m.pvalues[k]), int(m.nobs), int(df.expID.nunique())])
    return out
gee = gee_rows(m_any, "GEE_any") + gee_rows(m_rec, "GEE_recruit")
pd.DataFrame(gee, columns=["模型", "变量", "coef", "OR", "OR_low", "OR_high", "p", "n_obs", "n_groups"]).to_csv(
    os.path.join(OUT, "Q3_GEE四因素.csv"), index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------- 3 组合配对检验
df["combo"] = df[["HeadsUpYesNo", "VocalYesNo", "ApproachYesNo"]].astype(int).astype(str).agg("".join, axis=1)
CODES = {c: i for i, c in enumerate(sorted(df.combo.unique()))}
df["code"] = df.combo.map(CODES)
piv = df.pivot_table(index="expID", columns="Type", values="code", aggfunc="first")
call_codes = piv["call"].values.astype(int); ctrl_codes = piv["control"].values.astype(int)
TAB = np.zeros((8, 8))
for a, b in zip(call_codes, ctrl_codes):
    TAB[a, b] += 1
N = TAB.sum()
prow = TAB.sum(1) / N; pcol = TAB.sum(0) / N
d = (prow - pcol)[:7]
V = np.diag((prow + pcol)[:7]) - (np.outer(prow[:7], pcol[:7]) + np.outer(pcol[:7], prow[:7]))
W = float(N * d @ np.linalg.pinv(V) @ d)
p_sm = float(stats.chi2.sf(W, 7))

def stat_marg(call, ctrl):
    tab = np.zeros((8, 2))
    for v in call: tab[v, 0] += 1
    for v in ctrl: tab[v, 1] += 1
    tab = tab[tab.sum(1) > 0]
    return stats.chi2_contingency(tab, correction=False)[0] if tab.shape[0] > 1 else 0.0
obs = stat_marg(call_codes, ctrl_codes)
rng = np.random.default_rng(7)
B = 5000; cnt = 0
for _ in range(B):
    sw = rng.random(len(call_codes)) < 0.5
    cc = np.where(sw, ctrl_codes, call_codes); kk = np.where(sw, call_codes, ctrl_codes)
    if stat_marg(cc, kk) >= obs: cnt += 1
p_perm = (cnt + 1) / (B + 1)
pd.DataFrame([
    ["Stuart-Maxwell", W, 7, p_sm, "8x8 paired table; asymptotic chi-square"],
    ["配对置换", obs, 7, p_perm, f"{B} permutations; swap call/control within expID"],
], columns=["方法", "统计量", "df", "p", "备注"]).to_csv(
    os.path.join(OUT, "Q3_组合配对检验.csv"), index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------- 4 定义敏感性
V0 = int(df.VocalYesNo.sum()); A0 = int(df.ApproachYesNo.sum()); ANY0 = int(df.responded.sum())
V1 = int((df.VocalNcalls > 0).sum())
A1 = int(df.ApproachLatency.notna().sum())
def any_of(v, a):
    return int(((df.HeadsUpYesNo == 1) | (df.VocalNcalls > 0 if v else df.VocalYesNo == 1) |
                (df.ApproachLatency.notna() if a else df.ApproachYesNo == 1)).sum())
def ctrl_of(v, a):
    sub = df[df.Type == "control"]
    return int(((sub.HeadsUpYesNo == 1) | (sub.VocalNcalls > 0 if v else sub.VocalYesNo == 1) |
                (sub.ApproachLatency.notna() if a else sub.ApproachYesNo == 1)).sum())
sens = [
    ["S0", V0, ANY0, int(df[df.Type == "control"].responded.sum()), "原始定义：VocalYesNo=(VocalNcalls>VocalNcallsBefore)"],
    ["S1", V1, any_of(True, False), ctrl_of(True, False), "鸣叫改为任何播放后叫声 (VocalNcalls>0)"],
    ["S2", V0, any_of(False, True), ctrl_of(False, True), "接近改为接近尝试 (ApproachLatency 非空)"],
    ["S3", V1, any_of(True, True), ctrl_of(True, True), "S1+S2"],
]
pd.DataFrame(sens, columns=["情景", "鸣叫事件", "总响应", "对照响应", "说明"]).to_csv(
    os.path.join(OUT, "Q3_定义敏感性.csv"), index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------- 5 条件对数正态 AFT
pos = df[(df.responded == 1) & df.first_lat.notna() & (df.first_lat > 0)].copy()
pos["T"] = pos.first_lat
pos["logT"] = np.log(pos["T"])
m_aft = smf.glm("logT ~ C(CallType) + logD + Npenguins + C(Context)", data=pos,
                family=sm.families.Gaussian()).fit(cov_type="cluster", cov_kwds={"groups": pos.expID})
sigma = float(np.sqrt(m_aft.scale))
aft_rows = []
ci = m_aft.conf_int()
for k in m_aft.params.index:
    coef = float(m_aft.params[k])
    aft_rows.append([k, coef, float(np.exp(coef)), float(np.exp(ci.loc[k, 0])), float(np.exp(ci.loc[k, 1])),
                     float(m_aft.pvalues[k])])
pd.DataFrame(aft_rows, columns=["变量", "coef", "倍数", "CI_low", "CI_high", "p"]).assign(
    sigma=sigma, n_obs=len(pos), n_events=len(pos),
    删失假设="two-part; stage-2 conditional on response; 9 zero latencies excluded (immediate responses)"
).to_csv(os.path.join(OUT, "Q3_AFT.csv"), index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------- 6 预测表
def p_gee(model, ct, d, n=8, ctx="commuting"):
    new = pd.DataFrame({"CallType": pd.Categorical([ct], ["control", "flat", "mod", "tv"]),
                        "logD": [np.log(d)], "Npenguins": [n],
                        "Context": pd.Categorical([ctx], ["after", "commuting", "searching"])})
    return float(model.predict(new).iloc[0])

def aft_pred(ct, d, n=8, ctx="commuting"):
    new = pd.DataFrame({"CallType": pd.Categorical([ct], ["control", "flat", "mod", "tv"]),
                        "logD": [np.log(d)], "Npenguins": [n],
                        "Context": pd.Categorical([ctx], ["after", "commuting", "searching"])})
    pred = m_aft.get_prediction(new).summary_frame(alpha=0.05)
    mu = float(pred["mean"].iloc[0])
    lo = float(pred["mean_ci_lower"].iloc[0]); hi = float(pred["mean_ci_upper"].iloc[0])
    return np.exp(mu), np.exp(lo), np.exp(hi)

def rmst60(ct, d, P, n=8, ctx="commuting"):
    med, _, _ = aft_pred(ct, d, n, ctx)
    mu = np.log(med)
    f = lambda t: (1 - P) + P * (1 - stats.norm.cdf((np.log(t) - mu) / sigma))
    val, _ = integrate.quad(f, 0.01, 60)
    return float(val)

rows = []
for ct in ["flat", "mod", "tv"]:
    for d in [50, 100, 150]:
        P = p_gee(m_any, ct, d)
        med, lo, hi = aft_pred(ct, d)
        rows.append([ct, d, 8, "commuting", P, med, lo, hi, rmst60(ct, d, P)])
pd.DataFrame(rows, columns=["叫声", "距离", "Npenguins", "Context", "P_response",
                            "中位潜伏期_响应者", "CI_low", "CI_high", "RMST60"]).to_csv(
    os.path.join(OUT, "Q3_预测表_修正.csv"), index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------- 7-8 CEI
# 直接从 GEE 系数算 P，避免上千次 DataFrame predict
def p_fast(model, ct, d, n=8, ctx="commuting"):
    b = model.params
    eta = float(b["Intercept"])
    for level in ["flat", "mod", "tv"]:
        key = f"C(CallType)[T.{level}]"
        if key in b and ct == level:
            eta += float(b[key])
    eta += float(b["logD"]) * np.log(d) + float(b["Npenguins"]) * n
    if ctx == "commuting":
        eta += float(b["C(Context)[T.commuting]"])
    elif ctx == "searching":
        eta += float(b["C(Context)[T.searching]"])
    return 1.0 / (1.0 + np.exp(-eta))

GRID = np.exp(np.linspace(np.log(4), np.log(256), 4000))
P_REC_GRID = {ct: np.array([p_fast(m_rec, ct, d) for d in GRID]) for ct in ["flat", "mod", "tv"]}
P_ANY_GRID = {ct: np.array([p_fast(m_any, ct, d) for d in GRID]) for ct in ["flat", "mod", "tv"]}

def cei_grid_from_F(F7, ct, fs=1.0, sh=0.0, ps=1.0):
    F = np.clip(np.interp(np.log(GRID), np.log(DIST), F7) * fs - sh, 0, 1)
    return F * P_REC_GRID[ct] * ps

def cei_at_F(F7, ct, d, fs=1.0, sh=0.0, ps=1.0):
    F = np.clip(np.interp(np.log(d), np.log(DIST), F7) * fs - sh, 0, 1)
    return float(F * p_fast(m_rec, ct, d) * ps)

def radius_from_cei(vals, kappa):
    ok = np.where(vals >= kappa)[0]
    return float(GRID[ok[-1]]) if len(ok) else np.nan

F_BASE = {ct: F_curve(ct) for ct in ["flat", "mod", "tv"]}
KAPPA = {ct: 0.5 * cei_at_F(F_BASE[ct], ct, 50) for ct in ["flat", "mod", "tv"]}

# 块 bootstrap（1000 次）
rng = np.random.default_rng(20261126)
NB = 1000
boot_idx = []
for _ in range(NB):
    sampled = rng.integers(0, 30, 30)
    boot_idx.append(np.concatenate([np.where(BLOCKS == b)[0] for b in sampled]))

boot_radius = {ct: {"rel": [], "0.15": [], "0.20": [], "0.25": []} for ct in ["flat", "mod", "tv"]}
boot_cei = {ct: [] for ct in ["flat", "mod", "tv"]}
D10 = [4, 8, 16, 32, 64, 128, 256, 50, 100, 150]
for idx in boot_idx:
    for ct in ["flat", "mod", "tv"]:
        F7 = F_curve(ct, idx)
        v = cei_grid_from_F(F7, ct)
        boot_radius[ct]["rel"].append(radius_from_cei(v, 0.5 * cei_at_F(F7, ct, 50)))
        for thr in [0.15, 0.20, 0.25]:
            boot_radius[ct][f"{thr:.2f}"].append(radius_from_cei(v, thr))
        boot_cei[ct].append([cei_at_F(F7, ct, d) for d in D10])

# 7 半径表
rad_rows = []
for ct in ["flat", "mod", "tv"]:
    for thr_name, thr_val, key in [("相对(0.5)", 0.5, "rel"), ("0.15", 0.15, "0.15"),
                                   ("0.20", 0.20, "0.20"), ("0.25", 0.25, "0.25")]:
        kappa = KAPPA[ct] if key == "rel" else thr_val
        r = radius_from_cei(cei_grid_from_F(F_BASE[ct], ct), kappa)
        arr = np.array(boot_radius[ct][key], float)
        arr = arr[np.isfinite(arr)]
        lo, hi = (np.percentile(arr, [2.5, 97.5]) if len(arr) else (np.nan, np.nan))
        rad_rows.append(["相对" if key == "rel" else "绝对", thr_val, "empirical_mean", ct, r, lo, hi])
        # logistic 拟合（点估计）
        try:
            def logi(d, a, c, d50, h):
                return c + (a - c) / (1 + (d / d50) ** h)
            popt, _ = optimize.curve_fit(logi, DIST, F_BASE[ct], p0=[float(F_BASE[ct].max()), 0.1, 80, 1.5],
                                         bounds=([0.4, 0.0, 4, 0.3], [1.1, 0.6, 2000, 12]), maxfev=20000)
            v = logi(GRID, *popt) * P_REC_GRID[ct]
            rl = radius_from_cei(v, kappa)
            rad_rows.append(["相对" if key == "rel" else "绝对", thr_val, "logistic_fit", ct, rl, np.nan, np.nan])
        except Exception:
            rad_rows.append(["相对" if key == "rel" else "绝对", thr_val, "logistic_fit", ct, np.nan, np.nan, np.nan])
pd.DataFrame(rad_rows, columns=["阈值类型", "阈值", "模型族", "叫声", "半径", "CI_low", "CI_high"]).to_csv(
    os.path.join(OUT, "Q4_CEI半径.csv"), index=False, encoding="utf-8-sig")

# 8 CEI 曲线
rows = []
for i, d in enumerate(D10):
    cei_vals = {}
    for ct in ["flat", "mod", "tv"]:
        vals = np.array([b[i] for b in boot_cei[ct]], float)
        vals = vals[np.isfinite(vals)]
        cei_vals[ct] = (cei_at_F(F_BASE[ct], ct, d), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
    row = [d, cei_vals["flat"][0], cei_vals["mod"][0], cei_vals["tv"][0],
           cei_vals["flat"][1], cei_vals["flat"][2],
           cei_vals["mod"][1], cei_vals["mod"][2],
           cei_vals["tv"][1], cei_vals["tv"][2]]
    rows.append(row)
pd.DataFrame(rows, columns=["距离", "flat", "mod", "tv", "flat_low", "flat_high",
                            "mod_low", "mod_high", "tv_low", "tv_high"]).to_csv(
    os.path.join(OUT, "Q4_CEI曲线.csv"), index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------- 9 期望响应个体数
g = df.groupby("expID").Npenguins.first()
vals, cnts = np.unique(g.values, return_counts=True)
pvals = cnts / cnts.sum()
rows = []
for d in [4, 8, 16, 32, 50, 64, 100, 128, 150, 200, 256]:
    row = [d]
    base = {}
    for ct in ["flat", "mod", "tv"]:
        base[ct] = float(sum(pn * n * p_fast(m_any, ct, d, n) for n, pn in zip(vals, pvals)))
        row.append(base[ct])
    for rho in [0.5, 0.25]:
        for ct in ["flat", "mod", "tv"]:
            row.append(rho * base[ct])
    rows.append(row)
pd.DataFrame(rows, columns=["距离", "flat", "mod", "tv",
                            "flat_rho0.5", "mod_rho0.5", "tv_rho0.5",
                            "flat_rho0.25", "mod_rho0.25", "tv_rho0.25"]).to_csv(
    os.path.join(OUT, "Q4_期望响应个体数_经验分布.csv"), index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------- 10 情景
SCEN = [
    ("基线", 1.0, 0.0, 1.0, 0.0),
    ("相关性整体下降5个百分点", 0.95, 0.0, 1.0, 0.05),
    ("相关性整体下降20个百分点", 0.80, 0.0, 1.0, 0.20),
    ("识别阈值0.5→0.6", 1.0, 0.10, 1.0, 0.0),
    ("识别阈值0.5→0.7", 1.0, 0.20, 1.0, 0.0),
    ("海况恶化(P×0.8)", 1.0, 0.0, 0.8, 0.20),
    ("噪声增强(等效F-0.10)", 1.0, 0.10, 1.0, 0.0),
]
rows = []
for name, fs, sh, ps, red in SCEN:
    for ct in ["flat", "mod", "tv"]:
        v = cei_grid_from_F(F_BASE[ct], ct, fs=fs, sh=sh, ps=ps)
        r = radius_from_cei(v, KAPPA[ct])
        rows.append([name, ct, r, red, "基线 κ=0.5×CEI(50m)；Fscale/Fshift/Pscale 见 README"])
pd.DataFrame(rows, columns=["情景", "叫声", "CEI半径", "响应概率折减", "说明"]).to_csv(
    os.path.join(OUT, "Q4_情景_CEI.csv"), index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------- README
readme = f"""# Q3/Q4 修正版结果（按 claude-6b schema）

生成时间：2026-11-26 ｜ 脚本：`generate.py` ｜ 数据：`Data02` + `Data03–11`

## 口径（固定）

- **响应定义**：`ResponseYesNo = HeadsUp ∨ Vocal ∨ Approach`；`VocalYesNo = (VocalNcalls > VocalNcallsBefore)`（净增加）。
- **配对结构**：139 个 expID，每个 1 call + 1 control；GEE 以 expID 为聚类。
- **Q3 第一段**：GEE logistic，`ResponseYesNo ~ CallType(4) + logD + Npenguins + Context`，exchangeable by expID。
- **Q3 第二段**：首响应时间 `min(H/V/A latency)`；仅响应者、对数正态 AFT（= log(T) 的正态 GLM），expID 聚类；9 个 0 s 排除（立即响应）。
- **Q4 CEI**：`CEI(d) = F(d) × P_recruit(d)`；`P_recruit = P(Approach ∨ Vocal)`，GEE 同结构。**不乘 S(context)**（Context 已在 P 内）。
- **F(d)**：三种相关性等权均值；4/8/16/32/64/128/256 m；块 bootstrap（30 块 × 10 行）。
- **有效半径**：相对 κ = 0.5×CEI(50 m)（主）；绝对 0.15/0.20/0.25（对照）。用 4000 点对数网格的“持续穿叉”。
- **期望响应个体数**：`Σ p(n)·n·P_any(n,d)·ρ`；p(n) 为 Data02 的 139 个群体规模经验分布；ρ=1/0.5/0.25。
- **情景**：阈值 0.5→0.6 用 F−0.10 表示；0.5→0.7 用 F−0.20；海况用 P×0.8；噪声 −0.10 与阈值 0.6 等价。

## 文件

| 文件 | 内容 |
|---|---|
| `Q3_配对检验.csv` | 4 个行为的 McNemar b/c、OR、精确 CI、p |
| `Q3_GEE四因素.csv` | GEE_any 与 GEE_recruit 的系数/OR/CI/p |
| `Q3_组合配对检验.csv` | Stuart-Maxwell 与配对置换 |
| `Q3_定义敏感性.csv` | S0–S3 |
| `Q3_AFT.csv` | 条件对数正态 AFT 系数、sigma、n |
| `Q3_预测表_修正.csv` | 9 行表：P_response、中位潜伏期（响应者）、CI、RMST60 |
| `Q4_CEI半径.csv` | 相对/绝对阈值 × empirical/logistic × 三类 |
| `Q4_CEI曲线.csv` | 10 个距离的 CEI + 块 bootstrap 95% |
| `Q4_期望响应个体数_经验分布.csv` | ρ=1/0.5/0.25 |
| `Q4_情景_CEI.csv` | 7 个情景 × 三类 |

## 必须写进论文的限制

1. 300 行传播数据是 30 个 10 行块 × 10 个叫声单元，不是 300 次独立试验；iid CI 会窄约 2.5 倍。
2. 频谱图与幅度谱 r=0.808–0.967，等权 F 存在重复计权；建议同时报 PCA1（≈0.35/0.35/0.30）。
3. 潜伏期没有观察窗列；60 s 是假设，不是记录值；9 个 0 s 是点质量，主分析排除并单独报告。
4. `VocalYesNo` 是净增加；S1–S3 是定义敏感性。
5. Date×CallType 强混杂（V=0.528）；CallType 结论需加 Date 调整敏感性。
6. CEI 的 P_recruit 用 approach∨vocal；若改用 P_any，CEI 会更大、半径更长。
7. tv 的 CEI 半径对模型族敏感：empirical 203.4 m vs logistic 157.8 m；写论文给范围，不要只给 203。
8. Stuart-Maxwell 统计量依赖实现（本脚本 72.91，V2 独立实现 76.88）；配对置换 p=0.0002 是更稳的证据。
9. Q4_CEI曲线 的 low/high 是块 bootstrap 的 2.5/97.5 分位；CEI 在 4–64 m 非单调，别当平滑曲线。
10. Q4_CEI半径 的 logistic_fit 只有点估计（未做 bootstrap CI）；CI 空白是正常的。
11. Q3_配对检验 的“接近”OR=∞（c=0），CSV 中用空白表示；精确 p=2.38e-7，OR 下限=5.75。
"""
open(os.path.join(OUT, "README.md"), "w", encoding="utf-8").write(readme)

print("DONE. files in", OUT)
print("KAPPA:", {k: round(v, 3) for k, v in KAPPA.items()})
print("CEI 50m:", {ct: round(cei_at_F(F_BASE[ct], ct, 50), 3) for ct in ["flat", "mod", "tv"]})
print("radii relative:", {ct: round(radius_from_cei(cei_grid_from_F(F_BASE[ct], ct), KAPPA[ct]), 1) for ct in ["flat", "mod", "tv"]})
