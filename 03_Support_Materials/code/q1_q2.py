"""Reproducible analysis for 2026 APMCM Problem C, questions 1 and 2."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "date"
OUT = ROOT / "results" / "q1_q2"
TYPES = ["flat", "mod", "tv"]
SHARED = ["Duration", "F01", "Fpeak", "Q25", "Q50", "Q75"]
ALL = SHARED + ["F02", "PulseRate", "F02_present", "PulseRate_present"]
DISTANCES = np.array([4, 8, 16, 32, 64, 128, 256], dtype=float)
METRICS = ["spectrogram", "amplitude_spectrum", "envelope"]
FILE_TYPE = {"flat": "flat", "mod": "modulated", "tv": "twovoice"}
SEED = 2026


def save(df: pd.DataFrame, name: str) -> None:
    df.to_csv(OUT / name, index=False, encoding="utf-8-sig")


def md_table(df: pd.DataFrame, columns: list[str] | None = None) -> str:
    if columns is not None:
        df = df[columns]
    values = [[str(v) for v in row] for row in df.itertuples(index=False, name=None)]
    heads = [str(c) for c in df.columns]
    return "\n".join(
        ["| " + " | ".join(heads) + " |", "| " + " | ".join("---" for _ in heads) + " |"]
        + ["| " + " | ".join(row) + " |" for row in values]
    )


def holm(pvalues: list[float]) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    adjusted[order] = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    return np.minimum(adjusted, 1)


def read_acoustics() -> pd.DataFrame:
    data = pd.read_csv(DATA / "Data01_AfricanPenguins_seasurface_calls_acoustics.csv", na_values="NA")
    assert len(data) == 108 and set(data.CallType) == set(TYPES)
    for col in SHARED + ["F02", "PulseRate"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    assert data[SHARED].notna().all().all()
    assert data.loc[data.CallType == "tv", "F02"].notna().all()
    assert data.loc[data.CallType != "tv", "F02"].isna().all()
    assert data.loc[data.CallType == "flat", "PulseRate"].isna().all()
    data["F02_present"] = data.F02.notna().astype(int)
    data["PulseRate_present"] = data.PulseRate.notna().astype(int)
    return data


def q1_descriptives(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for call_type in TYPES:
        sub = data[data.CallType == call_type]
        for col in SHARED + ["F02", "PulseRate"]:
            x = sub[col].dropna()
            rows.append(
                dict(
                    CallType=call_type,
                    feature=col,
                    n=len(x),
                    median=x.median() if len(x) else np.nan,
                    q25=x.quantile(0.25) if len(x) else np.nan,
                    q75=x.quantile(0.75) if len(x) else np.nan,
                    mean=x.mean() if len(x) else np.nan,
                    sd=x.std() if len(x) > 1 else np.nan,
                )
            )
    desc = pd.DataFrame(rows)
    save(desc, "q1_descriptive_statistics.csv")

    omnibus = []
    pairwise = []
    for col in SHARED:
        samples = [data.loc[data.CallType == typ, col].to_numpy() for typ in TYPES]
        h, p = stats.kruskal(*samples)
        n = sum(len(x) for x in samples)
        omnibus.append(dict(feature=col, H=h, p=p, epsilon_squared=max(0, (h - 2) / (n - 3))))
    om = pd.DataFrame(omnibus)
    om["p_holm"] = holm(om.p.tolist())
    save(om, "q1_kruskal_wallis.csv")

    for col in SHARED:
        if om.loc[om.feature == col, "p_holm"].iloc[0] >= 0.05:
            continue
        for i, a in enumerate(TYPES):
            for b in TYPES[i + 1 :]:
                x = data.loc[data.CallType == a, col].to_numpy()
                y = data.loc[data.CallType == b, col].to_numpy()
                u, p = stats.mannwhitneyu(x, y, alternative="two-sided", method="asymptotic")
                pairwise.append(dict(feature=col, type_a=a, type_b=b, U=u, p=p,
                                     rank_biserial=2 * u / (len(x) * len(y)) - 1))
    x = data.loc[data.CallType == "mod", "PulseRate"].dropna().to_numpy()
    y = data.loc[data.CallType == "tv", "PulseRate"].dropna().to_numpy()
    u, p = stats.mannwhitneyu(x, y, alternative="two-sided", method="asymptotic")
    pairwise.append(dict(feature="PulseRate", type_a="mod", type_b="tv", U=u, p=p,
                         rank_biserial=2 * u / (len(x) * len(y)) - 1))
    pair = pd.DataFrame(pairwise)
    pair["p_holm"] = holm(pair.p.tolist())
    save(pair, "q1_pairwise_mann_whitney.csv")

    fig, axes = plt.subplots(2, 4, figsize=(15, 7), constrained_layout=True)
    for ax, col in zip(axes.flat, SHARED + ["PulseRate", "F02"]):
        groups = [data.loc[data.CallType == typ, col].dropna().to_numpy() for typ in TYPES]
        present = [(typ, v) for typ, v in zip(TYPES, groups) if len(v)]
        ax.boxplot([v for _, v in present], tick_labels=[t for t, _ in present], showfliers=False)
        for j, (_, values) in enumerate(present, 1):
            rng = np.random.default_rng(SEED + j)
            ax.scatter(rng.normal(j, 0.055, len(values)), values, s=8, alpha=0.35)
        ax.set_title(col)
    fig.savefig(OUT / "q1_feature_distributions.png", dpi=180)
    plt.close(fig)
    return desc, om, pair


def models() -> dict[str, tuple[Pipeline, dict]]:
    return {
        "logistic": (
            Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=False)),
                      ("scale", StandardScaler()),
                      ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
            {"model__C": [0.1, 1.0, 10.0]},
        ),
        "forest": (
            Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=False)),
                      ("model", RandomForestClassifier(n_estimators=120, class_weight="balanced",
                                                       random_state=SEED, n_jobs=1))]),
            {"model__min_samples_leaf": [2, 5]},
        ),
    }


def evaluate_cv(data: pd.DataFrame, columns: list[str], label: str, repeats: int) -> tuple[pd.DataFrame, str]:
    x = data[columns]
    y = data.CallType.to_numpy()
    summaries = []
    predictions = []
    configs = models()
    for name, (pipe, grid) in configs.items():
        for rep in range(repeats):
            outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + rep)
            pred = np.empty(len(data), dtype=object)
            for train, test in outer.split(x, y):
                inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED + rep)
                search = GridSearchCV(clone(pipe), grid, scoring="f1_macro", cv=inner, n_jobs=1)
                search.fit(x.iloc[train], y[train])
                pred[test] = search.predict(x.iloc[test])
            scores = dict(scope=label, model=name, repeat=rep,
                          accuracy=accuracy_score(y, pred), macro_f1=f1_score(y, pred, labels=TYPES, average="macro"))
            for typ, value in zip(TYPES, recall_score(y, pred, labels=TYPES, average=None)):
                scores[f"recall_{typ}"] = value
            summaries.append(scores)
            if rep == 0:
                for actual, estimated in zip(y, pred):
                    predictions.append(dict(scope=label, model=name, actual=actual, predicted=estimated))
                cm = confusion_matrix(y, pred, labels=TYPES)
                save(pd.DataFrame(cm, columns=[f"pred_{t}" for t in TYPES]).assign(actual=TYPES),
                     f"q1_confusion_{label}_{name}.csv")
    summary = pd.DataFrame(summaries)
    save(summary, f"q1_cv_{label}.csv")
    save(pd.DataFrame(predictions), f"q1_oof_predictions_{label}.csv")
    winner = summary.groupby("model").macro_f1.mean().sort_values(ascending=False).index[0]
    return summary, winner


def date_holdout(data: pd.DataFrame, columns: list[str], winner: str, scope: str) -> dict:
    x = data[columns]
    y = data.CallType.to_numpy()
    pred = np.empty(len(data), dtype=object)
    pipe, grid = models()[winner]
    # The tuning grid is evaluated only on training dates.
    for day in data.Date.unique():
        test = np.flatnonzero(data.Date.to_numpy() == day)
        train = np.flatnonzero(data.Date.to_numpy() != day)
        search = GridSearchCV(clone(pipe), grid, scoring="f1_macro",
                              cv=StratifiedKFold(3, shuffle=True, random_state=SEED), n_jobs=1)
        search.fit(x.iloc[train], y[train])
        pred[test] = search.predict(x.iloc[test])
    save(pd.DataFrame({"date": data.Date, "actual": y, "predicted": pred}),
         f"q1_date_holdout_predictions_{scope}.csv")
    return dict(scope=scope, model=winner, accuracy=accuracy_score(y, pred),
                macro_f1=f1_score(y, pred, labels=TYPES, average="macro"))


def feature_importance(data: pd.DataFrame, winner: str) -> pd.DataFrame:
    x = data[SHARED]
    y = data.CallType.to_numpy()
    pipe, grid = models()[winner]
    outer = StratifiedKFold(5, shuffle=True, random_state=SEED)
    rows = []
    for fold, (train, test) in enumerate(outer.split(x, y)):
        search = GridSearchCV(clone(pipe), grid, scoring="f1_macro",
                              cv=StratifiedKFold(3, shuffle=True, random_state=SEED), n_jobs=1)
        search.fit(x.iloc[train], y[train])
        importance = permutation_importance(search.best_estimator_, x.iloc[test], y[test],
                                            scoring="f1_macro", n_repeats=20,
                                            random_state=SEED + fold)
        for col, imp in zip(SHARED, importance.importances_mean):
            rows.append(dict(fold=fold, feature=col, importance=imp))
    raw = pd.DataFrame(rows)
    save(raw, "q1_importance_by_fold.csv")
    summary = raw.groupby("feature", as_index=False).agg(
        mean_importance=("importance", "mean"), sd_importance=("importance", "std"))
    save(summary, "q1_feature_importance.csv")
    return summary


def read_propagation() -> dict[str, np.ndarray]:
    result = {}
    for typ in TYPES:
        arrays = []
        for metric in METRICS:
            matches = sorted(DATA.glob(f"Data*_correlation1m_{FILE_TYPE[typ]}_call_{metric}.csv"))
            assert len(matches) == 1, (typ, metric, matches)
            df = pd.read_csv(matches[0], na_values="NA")
            assert list(df.columns) == [f"{int(d)}m" for d in DISTANCES]
            arrays.append(df.to_numpy(dtype=float))
        assert all(a.shape == (300, 7) for a in arrays)
        assert all(np.array_equal(np.isnan(arrays[0]), np.isnan(a)) for a in arrays[1:])
        stack = np.stack(arrays)
        assert np.nanmin(stack) >= 0 and np.nanmax(stack) <= 1
        result[typ] = stack
    return result


def monotone_curve(y: np.ndarray, n: np.ndarray) -> np.ndarray:
    return IsotonicRegression(increasing=False).fit_transform(np.log2(DISTANCES), y, sample_weight=n)


def crossing(curve: np.ndarray, threshold: float) -> tuple[float, str]:
    if curve[0] < threshold:
        return 4.0, "at_or_below_4m"
    if curve[-1] >= threshold:
        return 256.0, "at_or_above_256m"
    for i in range(1, len(curve)):
        if curve[i] <= threshold and curve[i - 1] > threshold:
            if curve[i] == curve[i - 1]:
                return DISTANCES[i], "within_range"
            fraction = (curve[i - 1] - threshold) / (curve[i - 1] - curve[i])
            return float(2 ** (np.log2(DISTANCES[i - 1]) + fraction)), "within_range"
    raise ValueError("No threshold crossing")


def q2_models(stacks: dict[str, np.ndarray], bootstraps: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    metric_rows, fidelity_rows, ranges, sensitivity = [], [], [], []
    bootstrap_curves = {}
    thresholds = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
    weights = {"equal": [1/3, 1/3, 1/3], "spectrogram_50pct": [0.5, 0.25, 0.25],
               "spectrum_50pct": [0.25, 0.5, 0.25], "envelope_50pct": [0.25, 0.25, 0.5]}
    for typ in TYPES:
        stack = stacks[typ]
        for k, metric in enumerate(METRICS):
            for j, distance in enumerate(DISTANCES):
                vals = stack[k, :, j]
                metric_rows.append(dict(CallType=typ, metric=metric, distance_m=int(distance),
                                        n=np.isfinite(vals).sum(), mean=np.nanmean(vals),
                                        sd=np.nanstd(vals, ddof=1)))
        combined = np.mean(stack, axis=0)
        n = np.isfinite(combined).sum(axis=0)
        mean = np.nanmean(combined, axis=0)
        fitted = monotone_curve(mean, n)
        samples = np.empty((bootstraps, 7))
        boot_ranges = {t: [] for t in [0.5, 0.7]}
        for b in range(bootstraps):
            indices = rng.integers(0, 300, 300)
            boot = combined[indices]
            bn = np.isfinite(boot).sum(axis=0)
            bc = monotone_curve(np.nanmean(boot, axis=0), bn)
            samples[b] = bc
            for threshold in boot_ranges:
                boot_ranges[threshold].append(crossing(bc, threshold)[0])
        bootstrap_curves[typ] = samples
        for j, distance in enumerate(DISTANCES):
            fidelity_rows.append(dict(CallType=typ, distance_m=int(distance), n=int(n[j]),
                                      raw_mean=mean[j], fitted=fitted[j],
                                      ci_low=np.quantile(samples[:, j], 0.025),
                                      ci_high=np.quantile(samples[:, j], 0.975)))
        for threshold, values in boot_ranges.items():
            estimate, status = crossing(fitted, threshold)
            ranges.append(dict(CallType=typ, threshold=threshold, range_m=estimate,
                               ci_low=np.quantile(values, 0.025), ci_high=np.quantile(values, 0.975),
                               boundary_status=status))
        for label, weight in weights.items():
            weighted = np.tensordot(np.asarray(weight), stack, axes=(0, 0))
            curve = monotone_curve(np.nanmean(weighted, axis=0), np.isfinite(weighted).sum(axis=0))
            for threshold in thresholds:
                value, status = crossing(curve, threshold)
                sensitivity.append(dict(CallType=typ, weight_scheme=label, threshold=threshold,
                                        range_m=value, boundary_status=status))
    save(pd.DataFrame(metric_rows), "q2_metric_correlations.csv")
    fidelity = pd.DataFrame(fidelity_rows)
    save(fidelity, "q2_fidelity_curves.csv")
    out_ranges = pd.DataFrame(ranges)
    save(out_ranges, "q2_effective_ranges.csv")
    save(pd.DataFrame(sensitivity), "q2_sensitivity.csv")
    # Long-distance area under the curve in log-distance, normalized to [0,1].
    robustness = []
    for typ in TYPES:
        curve = fidelity.loc[fidelity.CallType == typ, "fitted"].to_numpy()
        robustness.append(dict(CallType=typ, fidelity_64m=curve[4], fidelity_128m=curve[5],
                               fidelity_256m=curve[6], long_distance_mean=np.trapezoid(curve[4:], dx=1)/2))
    save(pd.DataFrame(robustness), "q2_long_distance_robustness.csv")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True, sharey=True, constrained_layout=True)
    for ax, typ in zip(axes, TYPES):
        x = fidelity[fidelity.CallType == typ]
        ax.plot(x.distance_m, x.raw_mean, "o", label="Observed mean")
        ax.plot(x.distance_m, x.fitted, "-", label="Monotone fit")
        ax.fill_between(x.distance_m, x.ci_low, x.ci_high, alpha=0.2, label="Bootstrap 95% CI")
        for threshold, style in [(0.5, "--"), (0.7, ":")]:
            ax.axhline(threshold, ls=style, lw=0.9, color="gray")
        ax.set_xscale("log", base=2)
        ax.set_xticks(DISTANCES, [str(int(d)) for d in DISTANCES])
        ax.set_title(typ)
        ax.set_xlabel("Distance (m)")
    axes[0].set_ylabel("Equal-weight acoustic fidelity")
    axes[0].legend(fontsize=8)
    fig.savefig(OUT / "q2_fidelity_and_thresholds.png", dpi=180)
    plt.close(fig)
    return fidelity, out_ranges


def playback_check() -> pd.DataFrame:
    data = pd.read_csv(DATA / "Data02_AfricanPenguins_playbacks_data.csv")
    assert len(data) == 278
    assert data.groupby("expID").size().eq(2).all()
    assert data.groupby("expID").Type.apply(lambda x: set(x) == {"call", "control"}).all()
    data["distance_band"] = pd.cut(data.Distance, [0, 50, 100, 256],
                                    labels=["5–50 m", "51–100 m", "101–200 m"])
    summary = data.groupby(["CallType", "distance_band"], observed=True).ResponseYesNo.agg(
        n="size", responses="sum", rate="mean").reset_index()
    save(summary, "q1_q2_playback_response_check.csv")
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    colors = {"control": "gray", "flat": "#2b8cbe", "mod": "#e6550d", "tv": "#31a354"}
    for typ in ["control"] + TYPES:
        sub = summary[summary.CallType == typ]
        positions = [{"5–50 m": 0, "51–100 m": 1, "101–200 m": 2}[s] for s in sub.distance_band.astype(str)]
        ax.plot(positions, sub.rate, "o-", label=typ, color=colors[typ])
    ax.set_xticks([0, 1, 2], ["5–50", "51–100", "101–200"])
    ax.set_xlabel("Playback distance band (m)")
    ax.set_ylabel("Observed response rate")
    ax.set_ylim(0, 1)
    ax.legend(title="CallType")
    fig.savefig(OUT / "q1_q2_playback_response_check.png", dpi=180)
    plt.close(fig)
    return summary


def report(data: pd.DataFrame, desc: pd.DataFrame, omnibus: pd.DataFrame, pair: pd.DataFrame,
           cv_shared: pd.DataFrame, cv_full: pd.DataFrame, winners: dict, dates: list[dict],
           importance: pd.DataFrame, ranges: pd.DataFrame, playback: pd.DataFrame,
           bootstraps: int, repeats: int) -> None:
    cv = pd.concat([cv_shared, cv_full]).groupby(["scope", "model"], as_index=False)[
        ["accuracy", "macro_f1", "recall_flat", "recall_mod", "recall_tv"]].mean().round(3)
    rank = importance.set_index("feature").mean_importance.rank(ascending=False).add(
        omnibus.set_index("feature").epsilon_squared.rank(ascending=False), fill_value=0).sort_values()
    selected = list(rank.index[:3])
    q1desc = desc[desc.feature.isin(selected)].copy()
    q1desc["median_IQR"] = q1desc.apply(
        lambda r: f"{r['median']:.3f} [{r['q25']:.3f}, {r['q75']:.3f}]", axis=1)
    threshold_table = ranges.copy()
    for col in ["range_m", "ci_low", "ci_high"]:
        threshold_table[col] = threshold_table[col].map(lambda x: f"{x:.1f}")
    om = omnibus.copy()
    for col in ["H", "p", "p_holm", "epsilon_squared"]:
        om[col] = om[col].map(lambda x: f"{x:.4g}")
    pws = pair.copy()
    for col in ["p", "p_holm", "rank_biserial"]:
        pws[col] = pws[col].map(lambda x: f"{x:.4g}")
    response = playback.copy()
    response["rate"] = response.rate.map(lambda x: f"{x:.3f}")
    robustness = pd.read_csv(OUT / "q2_long_distance_robustness.csv")
    sensitivity = pd.read_csv(OUT / "q2_sensitivity.csv")
    env_reliable = sensitivity[(sensitivity.weight_scheme == "envelope_50pct") &
                               np.isclose(sensitivity.threshold, 0.7)].set_index("CallType").range_m
    call_rows = pd.read_csv(DATA / "Data02_AfricanPenguins_playbacks_data.csv")
    call_rate = call_rows.loc[call_rows.Type == "call", "ResponseYesNo"].mean()
    control_rate = call_rows.loc[call_rows.Type == "control", "ResponseYesNo"].mean()
    lines = [
        "# 2026 APMCM C 题前两问：分析结果", "",
        "## 数据与方法", "",
        f"声学数据 108 条：flat {sum(data.CallType == 'flat')} 条、mod {sum(data.CallType == 'mod')} 条、tv {sum(data.CallType == 'tv')} 条。",
        "三类共有指标为 Duration、F01、Fpeak、Q25、Q50、Q75。F02 只在 tv 中有值；PulseRate 在 flat 中全部缺失。",
        f"分类采用 {repeats} 次重复分层五折交叉验证，每个训练折内用三折交叉验证调参；另做按采集日期留出验证。",
        f"传播分析使用三项相关性的等权均值和对数距离上的单调拟合。按类型同步重采样三项指标的对应行 {bootstraps} 次。", "",
        "## 问题 1：声学特征与分类", "",
        "### 关键指标的描述统计", "",
        "以下为主模型重要性和组间效应量综合排序最高的三项共有指标；完整统计见 CSV。", "",
        md_table(q1desc[["CallType", "feature", "n", "median_IQR"]]), "",
        "![各指标分布](q1_feature_distributions.png)", "",
        "### 非参数检验", "",
        "六项共有指标先做 Kruskal–Wallis 检验，并对六个总体检验的 p 值作 Holm 校正。仅对校正后显著的指标做两两 Mann–Whitney U 检验，再对这些两两检验及 PulseRate 的 mod–tv 比较作 Holm 校正。秩二列相关系数的正号表示前一类倾向取较大值。", "",
        md_table(om[["feature", "H", "p_holm", "epsilon_squared"]]), "",
        md_table(pws[["feature", "type_a", "type_b", "p_holm", "rank_biserial"]]) if len(pws) else "没有达到阈值的两两检验。", "",
        "### 分类结果", "",
        "下表为各次重复的平均结果。`shared` 只用共有指标；`full` 加入 F02、PulseRate 及其缺失标记。`full` 的高分可能主要来自类别特有的缺失模式，不能作为一般声学区分能力的证据。多数类基线准确率为 44/108 = 0.407。", "",
        md_table(cv), "",
        "按日期留出的验证结果：", "", md_table(pd.DataFrame(dates).round(3)), "",
        f"共有指标口径的优选分类器为 **{winners['shared']}**，结构特征口径为 **{winners['full']}**。关键共有指标：**{', '.join(selected)}**。",
        "混淆矩阵、逐条折外预测和置换重要性见同目录 CSV。按日期留出结果可能因某些日期仅含少数类别而不稳定。", "",
        "共有指标的最佳模型在普通验证中的宏平均 F1 仍不足 0.5，按日期留出后进一步下降；因此三类叫声在共有指标上的区分能力有限。Fpeak 的总体差异及 flat 与另外两类的两两差异达到校正后的显著水平；F01 虽有分类贡献，其总体检验在校正后未达到 0.05。", "",
        "## 问题 2：传播退化与有效距离", "",
        "定义综合保真度 $F_t(d)=\\{r_{\\mathrm{spectrogram},t}(d)+r_{\\mathrm{spectrum},t}(d)+r_{\\mathrm{envelope},t}(d)\\}/3$。对每个类型和距离的样本均值施加随距离不增的约束，在 $\\log_2 d$ 轴上线性插值。有效距离定义为拟合曲线首次降至指定阈值的位置。", "",
        "0.5 为可能识别，0.6 为中间识别层，0.7 为可靠识别。它们是建模情景阈值，现有数据不能证明其对应企鹅真实的感知边界。", "",
        md_table(threshold_table[["CallType", "threshold", "range_m", "ci_low", "ci_high", "boundary_status"]]), "",
        "![保真度与阈值](q2_fidelity_and_thresholds.png)", "",
        "### 传播比较与敏感性", "",
        "等权方案下，0.5 阈值的估计距离顺序为 tv > mod > flat；0.7 阈值下 tv 与 mod 接近，二者的自助法区间重叠。64–256 m 对数距离上的平均拟合保真度如下：", "",
        md_table(robustness[["CallType", "fidelity_64m", "fidelity_128m", "fidelity_256m", "long_distance_mean"]].round(3)), "",
        f"权重选择会影响严格阈值的结论：若幅度包络权重提高到 0.5，0.7 阈值下 tv 的估计距离为 {env_reliable['tv']:.1f} m，而 mod 为 {env_reliable['mod']:.1f} m。因此只能说 tv 在等权方案及 0.5 阈值下表现较强，不能无条件认定它在所有识别标准下都最稳健。", "",
        "权重与阈值敏感性结果见 `q2_sensitivity.csv`；各指标原始均值见 `q2_metric_correlations.csv`；64–256 m 的拟合保真度见 `q2_long_distance_robustness.csv`。区间是基于记录行可交换假设的自助法区间；数据没有独立个体或录音编号，不能将其解释为完整的种群层面不确定性。距离不外推到 4–256 m 以外。", "",
        (Path(__file__).with_name("model_comparison_addendum.md")).read_text(encoding="utf-8").strip(), "",
        "## 回放数据交叉核对", "",
        f"回放实验有 139 组自然叫声与对照声音的配对试验（每组各播放一次）。自然叫声总体响应率为 {call_rate:.3f}，对照声音为 {control_rate:.3f}。这支持叫声具有行为吸引作用，但不能由此确定具体信息内容。下表按叫声类型和回放距离分组：`n` 是该组的回放次数，`responses` 是其中 `ResponseYesNo=1` 的次数，`rate=responses/n`。这里的 `n` 不是企鹅只数；每次回放观察到的企鹅数另记为 `Npenguins`。不同距离段的样本量不等，远距离段尤其少。回放实验内部的自然叫声和对照声音可按 `expID` 配对，但传播实验与回放实验没有逐条对应的录音，不能据此确定听觉识别阈值。", "",
        md_table(response[["CallType", "distance_band", "n", "responses", "rate"]]), "",
        "![回放响应率](q1_q2_playback_response_check.png)", "",
        "## 文件说明", "",
        "运行：`analysis/.venv/bin/python analysis/q1_q2.py`。脚本从 `date/` 读取原始 CSV，结果写入 `results/q1_q2/`；随机种子固定为 2026。", "",
    ]
    (OUT / "结果说明.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--bootstraps", type=int, default=2000)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    data = read_acoustics()
    desc, omnibus, pair = q1_descriptives(data)
    shared, winner_shared = evaluate_cv(data, SHARED, "shared", args.repeats)
    full, winner_full = evaluate_cv(data, ALL, "full", args.repeats)
    dates = [date_holdout(data, SHARED, winner_shared, "shared"),
             date_holdout(data, ALL, winner_full, "full")]
    save(pd.DataFrame(dates), "q1_date_holdout_scores.csv")
    importance = feature_importance(data, winner_shared)
    stacks = read_propagation()
    _, ranges = q2_models(stacks, args.bootstraps)
    playback = playback_check()
    report(data, desc, omnibus, pair, shared, full,
           {"shared": winner_shared, "full": winner_full}, dates,
           importance, ranges, playback, args.bootstraps, args.repeats)
    print(f"Wrote analysis to {OUT}")


if __name__ == "__main__":
    main()
