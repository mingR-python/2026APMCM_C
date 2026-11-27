# 支撑材料（Support Materials）说明

> 官方：程序、数据（**不含赛题原始数据**）、结果；命名 `Capmcm<队号>fj.zip` 或 `.rar`。
> C 题未明确要求支撑材料，**可选**；但论文附录必须写软件名 + 全部源代码。

---

## 1. 内容

```
03_Support_Materials/
├── code/
│   ├── generate.py            ← 论文主生成脚本（Q1–Q4 表/图/数值）
│   ├── generate_en.py         ← 英文版生成
│   ├── make_figs.py           ← 9 张论文图
│   ├── q1_q2.py               ← Q1/Q2 信号侧
│   ├── generate_q3q4_cei.py   ← Q3/Q4：GEE + AFT + CEI（修正版）
│   └── replication_scripts/              ← 核验用脚本（McNemar/GEE/AFT/块 bootstrap 等）
├── results/
│   └── 10 个 CSV              ← Q3 配对/GEE/组合/定义敏感性/AFT/9行表；Q4 CEI 半径/曲线/期望个体数/情景
└── figures/
    └── 9 张 PNG               ← fig0–fig8
```

### 代码对应关系

| 文件 | 作用 | 对应论文 |
|---|---|---|
| `generate.py` / `generate_en.py` | 主流程：读原始 CSV → 表/图/数值 | 全文 |
| `make_figs.py` | 生成 fig0–fig8 | 全文图 |
| `q1_q2.py` | 分类器、非参数检验、保真度/衰减 | Q1/Q2 |
| `generate_q3q4_cei.py` | 配对 McNemar、GEE、8 组合、AFT、CEI、块 bootstrap | Q3/Q4 |
| `replication_scripts/verify_c*.py` | 独立复算数据事实 | 核验 |
| `replication_scripts/q3_*.py` | GEE/AFT/CEI 的基线与探索 | Q3/Q4 |
| `replication_scripts/q4_*.py` | 期望个体数/群体分布 | Q4 |

### 结果对应关系

| CSV | 内容 |
|---|---|
| `Q3_paired_test.csv` | 4 个行为的 McNemar OR/CI/p |
| `Q3_gee_four_factors.csv` | GEE_any / GEE_recruit 系数 |
| `Q3_combo_paired_test.csv` | Stuart-Maxwell + 配对置换 |
| `Q3_definition_sensitivity.csv` | S0–S3 |
| `Q3_aft.csv` | 条件对数正态 AFT |
| `Q3_prediction_table.csv` | 题面 9 行表（N=8，50/100/150） |
| `Q4_cei_radius.csv` | 相对/绝对阈值 × empirical/logistic × 三类 |
| `Q4_cei_curve.csv` | 10 个距离的 CEI + 块 bootstrap |
| `Q4_expected_responders.csv` | ρ=1/0.5/0.25 |
| `Q4_scenarios.csv` | 7 个情景 × 三类 |

---

## 2. 不包含什么（必须）

- **不包含赛题原始数据**：`Data01–Data11`、`Supplementary1–5`、`2026 APMCM Problem C.pdf`、原始压缩包。
- 不包含 62 MB 回放视频、961 MB 原始解包。
- 不包含中文内部文档（`输出/06`、`思路/`）；这些留在 `C题_成果汇总/`。

---

## 3. 打包（已打包）

已生成：`APMCM_C_Submission/Capmcm263XXXXXfj.zip`（832 KB），zip 内结构：

```
code/                  (含 replication_scripts/)
results/               (ASCII 文件名，如 Q3_paired_test.csv)
figures/               (fig0–fig8)
MANIFEST.txt           (中英文件名对照 + 说明)
README.md              (依赖/运行/限制)
```

- **提交前改名**：`Capmcm263XXXXXfj.zip` → `Capmcm<队号>fj.zip`（13 位队号）。
- 解压后应直接看到 `code/ results/ figures/` 三层，**不要**多套一层 `03_Support_Materials/`。
- 如需重打：用 7-Zip 选中 `code/ results/ figures/ MANIFEST.txt README.md` → 压缩；或 Python `zipfile`。
- 不含赛题原始数据；当前 832 KB。

---

## 4. 依赖与运行

- Python：`pandas / numpy / scipy / statsmodels / lifelines / matplotlib / openpyxl`。
- 论文排版：XeLaTeX（`00_Official_Rules_and_Templates/APMCM_LaTeX_Template/` 有 cls/cfg）。
- 运行顺序：`generate.py` → `make_figs.py` → LaTeX 编译；replication_scripts独立可跑。
- 原始数据路径：脚本默认指向 `输出/00_原始资料解包/…/Problem C/date/`；如换机器，改脚本顶部 `DATA` 变量。

---

## 5. 最后检查

- [ ] 不含赛题原始数据
- [ ] 解压后目录完整（code/results/figures）
- [ ] 有 README 说明依赖与运行
- [ ] zip 命名 `Capmcm<队号>fj.zip`
- [ ] 大小合理（几十 MB 以内）
