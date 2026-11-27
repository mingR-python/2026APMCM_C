# -*- coding: utf-8 -*-
"""Generate all figures for the APMCM C paper (English labels, 300 dpi)."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pathlib
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = pathlib.Path(r"d:\02_开发\数模")
DATA = ROOT / "_paper/repo/date"
RES = ROOT / "输出/07_Q3Q4_修正版"
OUT = ROOT / "_paper/paper/figs"
OUT.mkdir(parents=True, exist_ok=True)

# Okabe-Ito colour-blind safe palette
C = {"flat": "#0072B2", "mod": "#D55E00", "tv": "#009E73",
     "call": "#0072B2", "control": "#8C8C8C", "grey": "#666666"}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10, "axes.labelsize": 9.5,
    "legend.fontsize": 8.5, "legend.frameon": False,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
})

# ---------------------------------------------------------------- load
d1 = pd.read_csv(DATA / "Data01_AfricanPenguins_seasurface_calls_acoustics.csv")
d2 = pd.read_csv(DATA / "Data02_AfricanPenguins_playbacks_data.csv")

# ---------------------------------------------------------------- Fig 1: Q1 features
feats = [("Duration", "Duration (s)"), ("F01", "F01 (Hz)"), ("Fpeak", "Fpeak (Hz)"),
         ("Q25", "Q25 (Hz)"), ("Q50", "Q50 (Hz)"), ("Q75", "Q75 (Hz)")]
fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.0))
for ax, (col, lab) in zip(axes.ravel(), feats):
    data = [d1.loc[d1.CallType == t, col].dropna().values for t in ["flat", "mod", "tv"]]
    bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False,
                    medianprops=dict(color="white", linewidth=1.3),
                    whiskerprops=dict(linewidth=0.9), capprops=dict(linewidth=0.9))
    for patch, t in zip(bp["boxes"], ["flat", "mod", "tv"]):
        patch.set_facecolor(C[t]); patch.set_alpha(0.9); patch.set_edgecolor("#333")
    ax.set_xticklabels(["flat", "mod", "tv"], fontsize=7.5)
    ax.set_title(lab, fontsize=9)
    ax.grid(axis="x", visible=False)
fig.suptitle("Acoustic features of the three call types (n = 44 flat, 40 mod, 24 tv)", y=1.01, fontsize=9.5)
fig.tight_layout()
fig.savefig(OUT / "fig1_features.png"); plt.close(fig)

# ---------------------------------------------------------------- Fig 2: Q2 fidelity curves
qfid = pd.read_csv(ROOT / "输出/_tmp_fid.csv") if (ROOT / "输出/_tmp_fid.csv").exists() else None
# compute from raw correlation files (equal-weight mean, censored at available rows)
dists = [4, 8, 16, 32, 64, 128, 256]
files = {
    "flat": ["Data03_correlation1m_flat_call_spectrogram.csv",
             "Data06_correlation1m_flat_call_amplitude_spectrum.csv",
             "Data09_correlation1m_flat_call_envelope.csv"],
    "mod":  ["Data04_correlation1m_modulated_call_spectrogram.csv",
             "Data07_correlation1m_modulated_call_amplitude_spectrum.csv",
             "Data10_correlation1m_modulated_call_envelope.csv"],
    "tv":   ["Data05_correlation1m_twovoice_call_spectrogram.csv",
             "Data08_correlation1m_twovoice_call_amplitude_spectrum.csv",
             "Data11_correlation1m_twovoice_call_envelope.csv"],
}
cols = ["4m", "8m", "16m", "32m", "64m", "128m", "256m"]
fid = {}
for t, fs in files.items():
    mats = []
    for f in fs:
        df = pd.read_csv(DATA / f)
        mats.append(df[cols].apply(pd.to_numeric, errors="coerce").values)
    stack = np.stack(mats)                      # 3 x 300 x 7
    with np.errstate(invalid="ignore"):
        mean_metric = np.nanmean(stack, axis=0)  # 300 x 7
        fid[t] = np.nanmean(mean_metric, axis=0)

fig, ax = plt.subplots(figsize=(4.6, 3.1))
for t in ["flat", "mod", "tv"]:
    ax.plot(dists, fid[t], "o-", color=C[t], label=t, lw=1.8, ms=4.5)
ax.axhline(0.5, ls="--", lw=1, color="#444")
ax.text(280, 0.492, "$\\tau=0.5$ potential", fontsize=7.4, color="#444", ha="right", va="top",
        bbox=dict(fc="white", ec="none", pad=1.8, alpha=0.92))
ax.axhline(0.7, ls=":", lw=1, color="#888")
ax.text(280, 0.708, "$\\tau=0.7$ reliable", fontsize=7.4, color="#777", ha="right", va="bottom",
        bbox=dict(fc="white", ec="none", pad=1.8, alpha=0.92))
ax.set_xscale("log", base=2)
ax.set_xticks(dists); ax.set_xticklabels([str(d) for d in dists])
ax.set_xlabel("Distance (m, log$_2$ scale)"); ax.set_ylabel("Composite fidelity $F(d)$")
ax.set_ylim(0, 1); ax.legend(loc="lower left", ncol=3, columnspacing=1.1, handlelength=1.6)
ax.grid(axis="x", alpha=0.15)
fig.tight_layout(); fig.savefig(OUT / "fig2_fidelity.png"); plt.close(fig)

# ---------------------------------------------------------------- Fig 3: Q3 response rates
behav = [("ResponseYesNo", "Any response"), ("HeadsUpYesNo", "Heads-up"),
         ("VocalYesNo", "Vocalisation"), ("ApproachYesNo", "Approach")]


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return p, max(0, c - h), min(1, c + h)


call = d2[d2.Type == "call"]; ctrl = d2[d2.Type == "control"]
fig, ax = plt.subplots(figsize=(4.4, 2.9))
x = np.arange(len(behav)); w = 0.36
for j, (grp, lab, col) in enumerate([(call, "Natural call", C["call"]),
                                    (ctrl, "Control", C["control"])]):
    vals = [wilson(int(grp[b].sum()), len(grp)) for b, _ in behav]
    p = [v[0] for v in vals]; lo = [v[0] - v[1] for v in vals]; hi = [v[2] - v[0] for v in vals]
    ax.bar(x + (j - 0.5) * w, p, w, color=col, label=lab, alpha=0.92, edgecolor="white", lw=.6)
    ax.errorbar(x + (j - 0.5) * w, p, yerr=[lo, hi], fmt="none", ecolor="#222", capsize=2.5, lw=1)
    for xi, pi in zip(x + (j - 0.5) * w, p):
        ax.text(xi, pi + 0.035, f"{pi*100:.1f}", ha="center", fontsize=7)
ax.set_xticks(x); ax.set_xticklabels([b[1] for b in behav], fontsize=8)
ax.set_ylabel("Response rate"); ax.set_ylim(0, 0.82)
ax.legend(loc="upper right"); ax.grid(axis="x", visible=False)
fig.tight_layout(); fig.savefig(OUT / "fig3_rates.png"); plt.close(fig)

# ---------------------------------------------------------------- Fig 4: 8 combinations
combos = []
for t in ["call", "control"]:
    g = d2[d2.Type == t]
    key = (g.HeadsUpYesNo.astype(int) * 4 + g.VocalYesNo.astype(int) * 2 + g.ApproachYesNo.astype(int))
    combos.append(key.value_counts().reindex(range(8), fill_value=0))
order = [0, 4, 2, 6, 1, 5, 3, 7]
labels = ["none", "H", "V", "HV", "A", "HA", "VA", "HVA"]
c_call = np.array([combos[0][i] for i in order]) / len(call) * 100
c_ctrl = np.array([combos[1][i] for i in order]) / len(ctrl) * 100
fig, ax = plt.subplots(figsize=(4.8, 2.9))
xx = np.arange(8); w = 0.4
ax.bar(xx - w/2, c_call, w, color=C["call"], label="Natural call", edgecolor="white", lw=.6)
ax.bar(xx + w/2, c_ctrl, w, color=C["control"], label="Control", edgecolor="white", lw=.6)
ax.set_xticks(xx); ax.set_xticklabels(labels, fontsize=8)
ax.set_xlabel("Response combination (H = heads-up, V = vocal, A = approach)")
ax.set_ylabel("Share of trials (%)"); ax.legend(loc="upper right"); ax.grid(axis="x", visible=False)
fig.tight_layout(); fig.savefig(OUT / "fig4_combos.png"); plt.close(fig)

# ---------------------------------------------------------------- Fig 5: GEE forest plot
gee = pd.read_csv(RES / "Q3_GEE四因素.csv")
gee = gee[gee["模型"] == "GEE_any"]
rows = [("C(CallType)[T.flat]", "flat vs control"), ("C(CallType)[T.mod]", "mod vs control"),
        ("C(CallType)[T.tv]", "tv vs control"), ("logD", "log distance"),
        ("Npenguins", "group size"), ("C(Context)[T.commuting]", "commuting vs after"),
        ("C(Context)[T.searching]", "searching vs after")]
fig, ax = plt.subplots(figsize=(4.5, 2.9))
ys = []
for i, (v, lab) in enumerate(rows):
    r = gee[gee["变量"] == v].iloc[0]
    y = len(rows) - 1 - i; ys.append(y)
    ax.plot([r.OR_low, r.OR_high], [y, y], color="#555", lw=1.4)
    ax.plot(r.OR, y, "o", color="#0072B2" if r.p < .05 else "#B0B0B0", ms=6,
            mfc="#0072B2" if r.p < .05 else "white", mew=1.2)
ax.axvline(1, ls="--", lw=1, color="#999")
ax.set_yticks(ys); ax.set_yticklabels([r[1] for r in rows], fontsize=8)
ax.set_xscale("log"); ax.set_xlabel("Odds ratio (95% CI, log scale)")
ax.set_xlim(0.25, 70); ax.grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig(OUT / "fig5_forest.png"); plt.close(fig)

# ---------------------------------------------------------------- Fig 6: KM curves
def km(times, events, grid):
    """Kaplan-Meier survival function evaluated on `grid`."""
    t = np.asarray(times, float); e = np.asarray(events, int)
    order = np.argsort(t); t, e = t[order], e[order]
    n = len(t)
    surv, out, i = 1.0, [], 0
    for g in grid:
        while i < n and t[i] <= g:
            if e[i] == 1:
                at_risk = n - i
                if at_risk > 0:
                    surv *= (1 - 1 / at_risk)
            i += 1
        out.append(surv)
    return np.array(out)


grid = np.arange(0, 61, 0.5)
fig, ax = plt.subplots(figsize=(4.4, 2.9))
lat_cols = ["HeadsUpLatency", "VocalLatency", "ApproachLatency"]
for t in ["flat", "mod", "tv"]:
    g = call[call.CallType == t]
    first = g[lat_cols].min(axis=1)
    responded = (g.ResponseYesNo == 1).to_numpy()
    times = np.where(responded, first.fillna(60.0), 60.0)
    events = responded.astype(int)
    s = km(times, events, grid)
    ax.step(grid, s, where="post", color=C[t], lw=1.8, label=f"{t} (n={len(g)})")
ax.set_xlabel("Time since playback (s)"); ax.set_ylabel("P(no response yet)")
ax.set_ylim(0, 1.02); ax.set_xlim(0, 60); ax.legend(loc="upper right")
fig.tight_layout(); fig.savefig(OUT / "fig6_km.png"); plt.close(fig)

# ---------------------------------------------------------------- Fig 7: CEI curves
cei = pd.read_csv(RES / "Q4_CEI曲线.csv").sort_values("距离")
fig, ax = plt.subplots(figsize=(4.5, 3.0))
main = cei[cei["距离"].isin(dists)]
for t in ["flat", "mod", "tv"]:
    ax.plot(main["距离"], main[t], "o-", color=C[t], lw=1.8, ms=4.5, label=t)
    ax.fill_between(main["距离"], main[f"{t}_low"], main[f"{t}_high"], color=C[t], alpha=0.13, lw=0)
ax.set_xscale("log", base=2); ax.set_xticks(dists); ax.set_xticklabels([str(d) for d in dists])
ax.set_xlabel("Distance (m)"); ax.set_ylabel("CEI $=F(d)\\times P_{recruit}(d)$")
ax.legend(loc="upper right"); ax.grid(axis="x", alpha=0.15)
fig.tight_layout(); fig.savefig(OUT / "fig7_cei.png"); plt.close(fig)

# ---------------------------------------------------------------- Fig 8: scenario radii
sc = pd.read_csv(RES / "Q4_情景_CEI.csv")
scen = ["基线", "识别阈值0.5→0.6", "识别阈值0.5→0.7", "相关性整体下降5个百分点",
        "相关性整体下降20个百分点", "噪声增强(等效F-0.10)", "海况恶化(P×0.8)"]
en = ["Baseline", "Threshold 0.6", "Threshold 0.7", "$\\rho_F-5\\%$",
      "$\\rho_F-20\\%$", "Noise +3 dB", "Sea state"]
fig, ax = plt.subplots(figsize=(5.4, 2.9))
xx = np.arange(len(scen)); w = 0.26
for j, t in enumerate(["flat", "mod", "tv"]):
    vals = [sc[(sc["情景"] == s) & (sc["叫声"] == t)]["CEI半径"].iloc[0] for s in scen]
    ax.bar(xx + (j - 1) * w, vals, w, color=C[t], label=t, edgecolor="white", lw=.6)
ax.set_xticks(xx); ax.set_xticklabels(en, rotation=18, ha="right", fontsize=7.4)
ax.set_ylabel("CEI half-effectiveness radius (m)")
ax.legend(loc="upper right", ncol=3, columnspacing=1.0, handlelength=1.4)
ax.grid(axis="x", visible=False)
fig.tight_layout(); fig.savefig(OUT / "fig8_scenarios.png"); plt.close(fig)

print("figures written:")
for p in sorted(OUT.glob("*.png")):
    print(" ", p.name, p.stat().st_size // 1024, "KB")
print("fidelity means:")
for t in ["flat", "mod", "tv"]:
    print(" ", t, np.round(fid[t], 3))
