# -*- coding: utf-8 -*-
import sys, numpy as np, pandas as pd
DATA = sys.argv[1]
c2 = pd.read_csv(DATA + "/Data02_AfricanPenguins_playbacks_data.csv")
g = c2.groupby("expID").Npenguins.first()
print("实验数:", len(g))
print("群体规模统计:", g.describe().round(2).to_dict())
print("分位:", g.quantile([0,.1,.25,.5,.75,.9,.95,1]).to_dict())
print("\n频数分布:")
print(g.value_counts().sort_index().to_string())
print("\n分箱占比:")
bins = pd.cut(g, [0,1,2,3,5,8,10,15,20,30,40])
print((bins.value_counts(normalize=True).sort_index()*100).round(1).to_string())
print("\n均值/中位/众数:", round(g.mean(),2), g.median(), g.mode().tolist())
# 加权典型群体
print("\n典型群体规模(按实验加权) mean=%.2f median=%.1f" % (g.mean(), g.median()))
