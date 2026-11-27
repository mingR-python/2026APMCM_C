# 2026 APMCM · Problem C — Submission Package

> English note: all folder/file names in this repository are ASCII/English; the paper is in English, the Chinese version is for internal reference only.


> 按官方《Essay format and submission specifications》整理 ｜ 生成：2026-11-27
> **本文件夹是复制副本，所有原文件保留在原位置未动。**
> 官方来源：`D:\2026 APMCM 论文提交规范、论文模板、承诺书 (1).zip`（已解包在 `00_Official_Rules_and_Templates/`）。

---

## 一、官方要交什么（三件套）

| # | 材料 | 官方命名 | 本包位置 | 状态 |
|---|---|---|---|---|
| 1 | 论文 PDF | `#***.pdf`（题号+队号） | `01_Paper/Capmcm263XXXXX.pdf` | ✅ 已生成，**队号待替换** |
| 2 | 控制页（承诺书） | `#***cns.pdf` 或 `.jpg` | `02_Control_Sheet/` | ⏳ 待填写、打印、签字、扫描 |
| 3 | 支撑材料（已附上） | `#***fj.zip` 或 `.rar` | `Capmcm263XXXXXfj.zip`（根目录） | ✅ **已打包**，提交前改名 |

> `#` = 题号（本题 **C**）；`***` = 13 位队号（如 `apmcm26300001`）。
> 例：论文 `Capmcm26300001.pdf`，控制页 `Capmcm26300001cns.pdf`，支撑材料 `Capmcm26300001fj.zip`。

**官方原话**：三件同时提交到 https://www.saikr.com/vse/apmcm26 ；
“if they are not expressly required in the problem, please do not submit”（支撑材料 C 题未明确要求，可选；但论文附录必须写软件名 + 全部源代码）。

---

## 二、目录结构

```
APMCM_C_Submission/
├── README.md              ← 本文件
├── 00_Official_Rules_and_Templates/               ← 原样保留官方 zip 内容
│   ├── 2026 APMCM Essay format and submission specifications.pdf
│   ├── 2026 APMCM Control Sheet.docx
│   ├── 2026 APMCM Summary Sheet.docx
│   └── APMCM_LaTeX_Template/            （cls / cfg / main.tex / ControlSheet.tex）
├── 01_Paper/
│   ├── Capmcm263XXXXX.pdf                     ← 英文论文（27 页，0 中文，提交用）
│   ├── 2026_APMCM_C_Paper_Chinese_Reference.pdf  ← 中文版（内部参考，不提交）
│   └── paper_source/                           ← main.tex + sections/ + figs/ + code/ + cls + cfg
├── 02_Control_Sheet/
│   ├── Capmcm263XXXXX_ControlSheet.docx   ← 官方模板（待填/签字）
│   └── Signing_and_Scanning_Guide.md
├── 03_Support_Materials/
│   ├── code/                        ← 论文脚本 + Q3/Q4 修正版 + replication_scripts
│   ├── results/                     ← 10 个 Q3/Q4 结果 CSV
│   ├── figures/                     ← 9 张论文图
│   └── README_Support_Materials.md
├── Capmcm263XXXXXfj.zip            ← 支撑材料打包（832 KB，提交前改名）
├── 04_Summary_Sheet/
│   └── Capmcm263XXXXX_SummarySheet.docx   ← 官方摘要模板（论文第1页已含摘要）
└── 05_Submission_Checklist.md                  ← 逐条对照官方规范，提交前打勾
```

---

## 三、论文当前状态（已核验）

| 项 | 状态 |
|---|---|
| 页数 | **27 页** = 摘要 1 页 + 目录 1 页 + 正文/附录 **25 页**（规范：正文 ≤25 页，附录不限） |
| 语言 | **全英文**，正文/图表 **0 个中文字符**（已用 pypdf 复算） |
| 第 1 页 | 2026 APMCM summary sheet（Team Number + Problem Chosen: C + 摘要） |
| 第 2 页 | Contents（目录） |
| 页眉 | `Team#apmcm263XXXXX  Page X of 25`（**队号占位符待替换**） |
| 附录 | A. Reproducibility / A.5 Software and source code（已含软件名 + 代码说明） |
| 图 | 9 张（fig0–fig8），英文图注 |
| 中文版 | `01_Paper/2026_APMCM_C_Paper_Chinese_Reference.pdf` 仅内部参考，**不要提交** |

> ⚠️ **待办 0（最重要）**：论文 PDF 摘要页与页眉里的 `apmcm263XXXXX` 是占位符。拿到 13 位队号后，改 `paper_source/main.tex` 的 `\baominghao{}` 并重新编译，或至少确保摘要页/页眉显示真实队号。规范要求页眉只含队号+页码；占位符会被视为不合规。

---

## 四、你现在要做的 5 件事

1. **填队号**：把 `apmcm263XXXXX` 替换为 13 位真实队号；重编译论文 → 覆盖 `01_Paper/Capmcm263XXXXX.pdf`（改名 `Capmcm<队号>.pdf`）。
2. **填控制页**：打开 `02_Control_Sheet/Capmcm263XXXXX_ControlSheet.docx`，填 Problem chosen = **C**、Team control number、School 全称（英文）、3 名队员英文签名、指导老师、日期。
3. **打印签字**：打印控制页 → 全体队员**手写英文签名** → 扫描为 PDF 或拍照为 JPG → 命名 `Capmcm<队号>cns.pdf` / `.jpg`。
4. **支撑材料已打包**：`APMCM_C_Submission/Capmcm263XXXXXfj.zip`（832 KB，zip 内 ASCII 文件名 + `MANIFEST.txt` + `README.md`）。提交前改名为 `Capmcm<队号>fj.zip`；**不要**放入赛题原始数据（Data01–11、Supplementary）。
5. **上传**：三件同时上传赛氪；点击“提交电子档”确认；截止 **2026-11-30 09:00**，建议提前 30–60 分钟。

---

## 五、与 `C题_成果汇总/` 的分工

| 文件夹 | 用途 | 给谁看 |
|---|---|---|
| `APMCM_C_Submission/`（本目录） | **对外提交**：论文/控制页/支撑材料，按官方命名 | 评委/报名系统 |
| `C题_成果汇总/` | **内部工作台**：思路、主控文档、核验证据、Q3/Q4 结果 | 你/队友 |

> 提交包只放“官方要的东西”；工作过程、中文文档、核验证据都在 `C题_成果汇总/`，不往提交包里塞。

---

## 六、一句话

> 先把 **队号** 和 **控制页签字** 这两件“行政项”做掉；论文 27 页/0 中文/25 页正文已经合规，支撑材料按需打包。细节逐条看 [05_Submission_Checklist.md](05_Submission_Checklist.md)。
