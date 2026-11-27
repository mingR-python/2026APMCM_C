# -*- coding: utf-8 -*-
"""独立核验 C 题数据事实，与 输出/ 中其它 AI 的报告结论逐条对照。
用法: python -I verify_c.py <C数据目录>
"""
import sys, os, glob
import numpy as np
import pandas as pd

C = sys.argv[1]
pd.set_option("display.width", 200)

def hr(t):
    print("\n" + "=" * 20, t, "=" * 20)

# ---------------- Data01 ----------------
c1 = pd.read_csv(os.path.join(C, "Data01_AfricanPenguins_seasurface_calls_acoustics.csv"))
hr("Data01 shape / CallType")
print("shape:", c1.shape)
print(c1.CallType.value_counts(dropna=False).to_string())
print("\n缺失率(%):")
print((c1.isna().mean() * 100).round(1).to_string())
num1 = ["Duration", "F01", "F02", "Fpeak", "Q25", "Q50", "Q75", "PulseRate"]
print("\n按类型中位数:")
print(c1.groupby("CallType")[num1].median().round(3).to_string())
print("\nDuration 全表范围:", float(c1.Duration.min()), float(c1.Duration.max()))
print("Fpeak 全表范围:", float(c1.Fpeak.min()), float(c1.Fpeak.max()))
print("F01 全表范围:", float(c1.F01.min()), float(c1.F01.max()))
print("Q50 全表范围:", float(c1.Q50.min()), float(c1.Q50.max()))
print("PulseRate 范围/非空:", c1.PulseRate.min(), c1.PulseRate.max(), int(c1.PulseRate.notna().sum()))
print("Date 范围:", c1.Date.min(), "~", c1.Date.max())
print("Location:", c1.Location.value_counts().to_dict())

# ---------------- Data02 ----------------
c2 = pd.read_csv(os.path.join(C, "Data02_AfricanPenguins_playbacks_data.csv"))
hr("Data02 shape / Type / CallType / Context")
print("shape:", c2.shape)
print(pd.crosstab(c2.Type, c2.CallType, dropna=False).to_string())
print("\nContext x Type:")
print(pd.crosstab(c2.Context, c2.Type).to_string())
print("\nDistance 分布:", c2.Distance.value_counts().sort_index().to_dict())
print("Npenguins describe:", c2.Npenguins.describe().round(2).to_dict())

hr("Data02 响应率")
for col in ["ResponseYesNo", "HeadsUpYesNo", "ApproachYesNo", "VocalYesNo"]:
    g = c2.groupby("Type")[col].agg(["mean", "sum", "count"])
    print(f"\n{col}:\n{g.round(4).to_string()}")

hr("Data02 按 CallType 的响应率(call 子集)")
sub = c2[c2.Type == "call"]
print(sub.groupby("CallType")[["ResponseYesNo", "HeadsUpYesNo", "ApproachYesNo", "VocalYesNo"]].mean().round(3).to_string())

hr("Data02 响应行为组合")
behav = c2[["HeadsUpYesNo", "VocalYesNo", "ApproachYesNo"]].astype(int)
combo = behav.astype(str).agg("".join, axis=1)
c2 = c2.assign(combo=combo)
print(pd.crosstab(c2.combo, c2.Type, dropna=False).to_string())
print("\n按 Type 组合占比:")
print((pd.crosstab(c2.combo, c2.Type, normalize="columns") * 100).round(1).to_string())

hr("Data02 缺失结构检查")
for col in ["HeadsUpLatency", "ApproachLatency", "VocalLatency", "VocalNcalls", "HeadsUpDuration", "ApproachDuration"]:
    miss = c2[col].isna()
    cross = pd.crosstab(miss, c2.ResponseYesNo)
    print(f"{col}: 缺失 {int(miss.sum())}, 缺失时 ResponseYesNo 分布 {cross.to_dict()}")
print("\nVR1-VR19 非空计数:")
vr = [f"VR{i}" for i in range(1, 20)]
print(c2[vr].notna().sum().to_string())
print("\nVR 非空且 VocalYesNo=0 的行数:", int((c2[vr].notna().any(axis=1) & (c2.VocalYesNo == 0)).sum()))
print("VocalYesNo=1 的 VocalNcalls 分布:", c2.loc[c2.VocalYesNo == 1, "VocalNcalls"].describe().round(2).to_dict())
print("VocalNcalls 与 VR 非空数是否一致:")
vr_n = c2[vr].notna().sum(axis=1)
eq = (vr_n == c2.VocalNcalls.fillna(0)).mean()
print("  一致比例:", round(float(eq), 4))

hr("Data02 Latency 描述")
print(c2.groupby("Type")[["HeadsUpLatency", "ApproachLatency", "VocalLatency"]].describe().round(2).to_string())

# ---------------- Data03-11 ----------------
hr("Data03-11 相关性")
for f in sorted(glob.glob(os.path.join(C, "Data0[3-9]*.csv")) + glob.glob(os.path.join(C, "Data1[01]*.csv"))):
    d = pd.read_csv(f)
    name = os.path.basename(f)
    print(f"{name}")
    print("  shape", d.shape, "cols", list(d.columns))
    print("  均值", d.mean().round(3).to_dict())
    print("  中位", d.median().round(3).to_dict())
    print("  NaN", int(d.isna().sum().sum()), " min", round(float(d.min().min()), 3), " max", round(float(d.max().max()), 3))

# 综合三类相关性的总体
hr("三类叫声 × 三种相关性 × 距离 汇总(均值)")
rows = []
for ct, keys in {"flat": ["flat"], "mod": ["modulated"], "tv": ["twovoice"]}.items():
    for metric_file, metric_name in [("spectrogram", "频谱图"), ("amplitude_spectrum", "幅度谱"), ("envelope", "包络")]:
        f = os.path.join(C, f"Data0{'' }")
for ct, pat in [("flat", "*flat*"), ("mod", "*modulated*"), ("tv", "*twovoice*")]:
    for f in sorted(glob.glob(os.path.join(C, f"Data*{pat}*.csv"))):
        d = pd.read_csv(f)
        metric = "spectrogram" if "spectrogram" in f else ("amplitude_spectrum" if "amplitude_spectrum" in f else "envelope")
        row = {"call": ct, "metric": metric}
        for c in d.columns:
            row[c] = round(float(d[c].mean()), 3)
        rows.append(row)
tab = pd.DataFrame(rows)
print(tab.to_string(index=False))

hr("256m 处各类均值")
for ct, pat in [("flat", "*flat*"), ("mod", "*modulated*"), ("tv", "*twovoice*")]:
    vals = []
    for f in sorted(glob.glob(os.path.join(C, f"Data*{pat}*.csv"))):
        d = pd.read_csv(f)
        vals.append(float(d["256m"].mean()))
    print(ct, "三种指标均值:", [round(v, 4) for v in vals], " 平均:", round(float(np.mean(vals)), 4))
