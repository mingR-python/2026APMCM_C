# -*- coding: utf-8 -*-
"""C 题第二轮回核：统计检验、缺失结构、行对齐、复合保真度、阈值穿叉。"""
import sys, os, glob
import numpy as np
import pandas as pd
from scipy import stats

C = sys.argv[1]

def hr(t):
    print("\n" + "=" * 18, t, "=" * 18)

c1 = pd.read_csv(os.path.join(C, "Data01_AfricanPenguins_seasurface_calls_acoustics.csv"))
c2 = pd.read_csv(os.path.join(C, "Data02_AfricanPenguins_playbacks_data.csv"))

hr("Data01 分组计数 / 可用性")
print(c1.groupby("CallType").agg(
    n=("Duration", "size"),
    F02_n=("F02", "count"),
    PR_n=("PulseRate", "count"),
    dur_min=("Duration", "min"), dur_max=("Duration", "max"),
).to_string())
print("\nDate 唯一值:", c1.Date.nunique(), sorted(c1.Date.unique())[:5])
print("Duration 分位:", c1.Duration.quantile([0, .05, .25, .5, .75, .95, 1]).round(3).to_dict())
print("\nPulseRate 按类型:")
print(c1.groupby("CallType").PulseRate.agg(["count", "median", "min", "max"]).round(2).to_string())

hr("Kruskal-Wallis 逐指标")
for col in ["Duration", "F01", "F02", "Fpeak", "Q25", "Q50", "Q75", "PulseRate"]:
    groups = [g[col].dropna().values for _, g in c1.groupby("CallType")]
    groups = [g for g in groups if len(g) >= 2]
    if len(groups) < 2:
        print(f"{col}: 可比较组数 {len(groups)}，跳过")
        continue
    H, p = stats.kruskal(*groups)
    print(f"{col:10s} groups={[len(g) for g in groups]} H={H:.3f} p={p:.4f}")

hr("Mann-Whitney 两两 + Bonferroni")
pairs = [("flat", "mod"), ("flat", "tv"), ("mod", "tv")]
for col in ["Duration", "F01", "Fpeak", "Q25", "Q50", "Q75", "PulseRate"]:
    out = []
    for a, b in pairs:
        xa = c1.loc[c1.CallType == a, col].dropna()
        xb = c1.loc[c1.CallType == b, col].dropna()
        if len(xa) < 2 or len(xb) < 2:
            out.append(f"{a}-{b}:na")
            continue
        U, p = stats.mannwhitneyu(xa, xb, alternative="two-sided")
        out.append(f"{a}-{b}: p={p:.4f} (p*3={min(p*3,1):.4f})")
    print(f"{col:10s} " + " | ".join(out))

hr("Data02 配对结构")
print("expID 唯一值:", c2.expID.nunique(), " 每 exp 行数分布:", c2.groupby("expID").size().value_counts().to_dict())
print("每个 exp 是否恰 1 call + 1 control:", c2.groupby("expID").Type.nunique().value_counts().to_dict())
print("PlaybackTrack 唯一值:", c2.PlaybackTrack.nunique())
print("重复行数:", int(c2.duplicated().sum()))

hr("Data02 行为-潜伏期/次数 矛盾")
print("HeadsUpLatency 非空但 HeadsUpYesNo=0:", int((c2.HeadsUpLatency.notna() & (c2.HeadsUpYesNo == 0)).sum()))
print("ApproachLatency 非空但 ApproachYesNo=0:", int((c2.ApproachLatency.notna() & (c2.ApproachYesNo == 0)).sum()))
print("VocalLatency 非空但 VocalYesNo=0:", int((c2.VocalLatency.notna() & (c2.VocalYesNo == 0)).sum()))
vr = [f"VR{i}" for i in range(1, 20)]
print("VR 有任何非空但 VocalYesNo=0:", int((c2[vr].notna().any(axis=1) & (c2.VocalYesNo == 0)).sum()))
print("VocalYesNo=1 但 VR 全空:", int(((c2.VocalYesNo == 1) & ~c2[vr].notna().any(axis=1)).sum()))
print("VocalNcalls != VR 非空个数:", int((c2.VocalNcalls.fillna(0) != c2[vr].notna().sum(axis=1)).sum()))
print("VocalNcalls>0 但 VocalYesNo=0:", int(((c2.VocalNcalls > 0) & (c2.VocalYesNo == 0)).sum()))
print("ResponseYesNo=0 但有任一子行为=1:", int(((c2.ResponseYesNo == 0) & (c2[["HeadsUpYesNo", "VocalYesNo", "ApproachYesNo"]].sum(axis=1) > 0)).sum()))
print("ResponseYesNo=1 但三子行为全 0:", int(((c2.ResponseYesNo == 1) & (c2[["HeadsUpYesNo", "VocalYesNo", "ApproachYesNo"]].sum(axis=1) == 0)).sum()))

hr("Data03-11 NaN 行结构")
files = {}
for ct, pat in [("flat", "*flat*"), ("mod", "*modulated*"), ("tv", "*twovoice*")]:
    for f in sorted(glob.glob(os.path.join(C, f"Data*{pat}*.csv"))):
        metric = "spec" if "spectrogram" in f else ("amp" if "amplitude_spectrum" in f else "env")
        files[(ct, metric)] = pd.read_csv(f)
for (ct, m), d in files.items():
    nanrow = d.isna().any(axis=1)
    print(f"{ct:5s} {m:4s}: NaN 总数 {int(d.isna().sum().sum()):4d} / {d.size}, 含 NaN 行数 {int(nanrow.sum())}, "
          f"全行 NaN {int(d.isna().all(axis=1).sum())}")
print("\n同类型三指标 NaN 掩码重合度 (Jaccard):")
for ct in ["flat", "mod", "tv"]:
    ms = {m: files[(ct, m)].isna().any(axis=1).values for m in ["spec", "amp", "env"]}
    print(f"  {ct}: spec∩amp={int((ms['spec'] & ms['amp']).sum())}/{(ms['spec'] | ms['amp']).sum()}, "
          f"spec∩env={int((ms['spec'] & ms['env']).sum())}/{(ms['spec'] | ms['env']).sum()}, "
          f"三交集 {int((ms['spec'] & ms['amp'] & ms['env']).sum())}")
print("\n完全无 NaN 的行数:", {ct: int((~files[(ct, 'spec')].isna().any(axis=1) & ~files[(ct, 'amp')].isna().any(axis=1) & ~files[(ct, 'env')].isna().any(axis=1)).sum()) for ct in ["flat", "mod", "tv"]})

hr("复合保真度 (按行, 权重 0.5/0.3/0.2) 与阈值穿叉")
def composite(ct, w):
    parts = []
    for m in ["spec", "amp", "env"]:
        d = files[(ct, m)]
        parts.append(d.values * w[m])
    S = np.nansum(np.stack(parts), axis=0)
    valid = ~np.stack([files[(ct, m)].isna().values for m in ["spec", "amp", "env"]]).all(axis=0)
    return S, valid

W = {"spec": 0.5, "amp": 0.3, "env": 0.2}
dists = np.array([4, 8, 16, 32, 64, 128, 256])
for ct in ["flat", "mod", "tv"]:
    S, valid = composite(ct, W)
    means = [np.nanmean(np.where(valid[:, j], S[:, j], np.nan)) for j in range(7)]
    print(f"\n{ct} 复合均值: " + " ".join(f"{d}m={v:.3f}" for d, v in zip(dists, means)))
    for thr in [0.5, 0.6, 0.7]:
        # 对数距离线性插值
        cross = None
        for j in range(6):
            a, b = means[j], means[j + 1]
            if (a >= thr > b) or (a > thr >= b):
                t = (a - thr) / (a - b)
                cross = 2 ** (np.log2(dists[j]) + t * (np.log2(dists[j + 1]) - np.log2(dists[j])))
                break
        print(f"  阈值 {thr}: 穿叉距离 ≈ {cross:.0f} m" if cross else f"  阈值 {thr}: 256 m 内未穿叉 (最小 {means[-1]:.3f})")

hr("等权复合 (1/3 each) 256m")
for ct in ["flat", "mod", "tv"]:
    vals = [files[(ct, m)]["256m"].mean() for m in ["spec", "amp", "env"]]
    print(f"{ct}: {[round(v,3) for v in vals]} 均值 {np.mean(vals):.3f}")

hr("响应组合分布 8 类 + 统计检验")
combo = c2[["HeadsUpYesNo", "VocalYesNo", "ApproachYesNo"]].astype(int).astype(str).agg("".join, axis=1)
c2 = c2.assign(combo=combo)
tab = pd.crosstab(c2.combo, c2.Type)
print(tab.to_string())
print("\ncall 组内按 CallType 的组合分布(行%)")
print((pd.crosstab(c2[c2.Type == "call"].combo, c2[c2.Type == "call"].CallType, normalize="columns") * 100).round(1).to_string())
print("\nType 总体 8 类卡方:", stats.chi2_contingency(tab).pvalue if tab.shape[0] > 1 else "na")
# 按 CallType 组内
tab2 = pd.crosstab(c2[c2.Type == "call"].combo, c2[c2.Type == "call"].CallType)
try:
    print("call 组内 8 类 × CallType 卡方 df/期望<5 比例:", stats.chi2_contingency(tab2)[:2],
          "| 期望<5 占比", round(float((stats.chi2_contingency(tab2)[3] < 5).mean()), 3))
except Exception as e:
    print("chi2 error", e)

hr("按 CallType 的关键响应率 + 置信区间")
for ct in ["flat", "mod", "tv"]:
    s = c2[(c2.Type == "call") & (c2.CallType == ct)]
    for col in ["ResponseYesNo", "HeadsUpYesNo", "VocalYesNo", "ApproachYesNo"]:
        k, n = int(s[col].sum()), len(s)
        lo, hi = stats.beta.ppf([0.025, 0.975], k + 0.5, n - k + 0.5)
        print(f"{ct:4s} {col:16s} {k:3d}/{n:3d} = {k/n:.3f}  95%CI [{lo:.3f},{hi:.3f}]")

hr("Distance × 响应 (call 组)")
s = c2[c2.Type == "call"]
print(s.groupby("Distance").agg(n=("ResponseYesNo", "size"), resp=("ResponseYesNo", "mean")).round(3).to_string())
print("\nNpenguins 分箱响应率:")
s2 = s.assign(bin=pd.cut(s.Npenguins, [0, 3, 8, 15, 40]))
print(s2.groupby("bin", observed=True).agg(n=("ResponseYesNo", "size"), resp=("ResponseYesNo", "mean")).round(3).to_string())
