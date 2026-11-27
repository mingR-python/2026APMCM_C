import sys, numpy as np, pandas as pd
from lifelines import LogNormalAFTFitter
DATA=sys.argv[1]
df=pd.read_csv(DATA+"/Data02_AfricanPenguins_playbacks_data.csv")
df["first_lat"]=df[["HeadsUpLatency","VocalLatency","ApproachLatency"]].min(axis=1)
df["responded"]=df.ResponseYesNo.astype(int)
df["logD"]=np.log(df.Distance)
for ct in ["control","flat","mod","tv"]:
    df["ct_"+ct]=(df.CallType==ct).astype(int)
for cx in ["after","commuting","searching"]:
    df["cx_"+cx]=(df.Context==cx).astype(int)
df["T"]=np.where(df.responded==1, df.first_lat.fillna(0.01).clip(lower=0.01), 60.0)
df["E"]=df.responded
print("T range",df["T"].min(),df["T"].max(),"events",int(df["E"].sum()))
f="ct_flat + ct_mod + ct_tv + logD + Npenguins + cx_commuting + cx_searching"
aft=LogNormalAFTFitter().fit(df,"T","E",formula=f)
print(aft.summary)
print("median 50m flat commuting:", aft.predict_median(pd.DataFrame({"ct_flat":[1],"ct_mod":[0],"ct_tv":[0],"logD":[np.log(50)],"Npenguins":[8],"cx_commuting":[1],"cx_searching":[0]})))
