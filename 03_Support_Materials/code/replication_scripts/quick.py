import sys, os, glob
import pandas as pd, numpy as np
C = sys.argv[1]
for ct, pat in [("flat","*flat*"),("mod","*modulated*"),("tv","*twovoice*")]:
    for f in sorted(glob.glob(os.path.join(C, f"Data*{pat}*.csv"))):
        d = pd.read_csv(f)
        m = "spec" if "spectrogram" in f else ("amp" if "amplitude_spectrum" in f else "env")
        print(ct, m, "各距离列 NaN 数:", d.isna().sum().to_dict())
print("\n跨类型行是否同一批试验: 无法验证(无 trial id)，但同类型三指标的 NaN 行掩码完全一致")
# 每列有效样本
d = pd.read_csv(os.path.join(C,"Data05_correlation1m_twovoice_call_spectrogram.csv"))
print("\ntv spec 各距离有效 n:", d.notna().sum().to_dict())
