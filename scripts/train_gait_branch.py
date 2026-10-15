
from __future__ import annotations
import json, math, re
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "multimodal" / "Gait_Sensor" / "gait-in-parkinsons-disease-1.0.0"
OUT_DIR = ROOT / "results"
MODEL_DIR = ROOT / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

LEFT_X = np.array([-500,-700,-300,-700,-300,-700,-300,-500], dtype=float)
LEFT_Y = np.array([-800,-400,-400,0,0,400,400,800], dtype=float)
RIGHT_X = np.array([500,700,300,700,300,700,300,500], dtype=float)
RIGHT_Y = np.array([-800,-400,-400,0,0,400,400,800], dtype=float)
EPS = 1e-9
FILENAME_RE = re.compile(r"^(Ga|Ju|Si)(Co|Pt)(\d+)_([0-9]+)\.txt$", re.I)

def safe_corr(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    if len(a)<3 or np.nanstd(a)<EPS or np.nanstd(b)<EPS: return 0.0
    v=np.corrcoef(a,b)[0,1]
    return 0.0 if not np.isfinite(v) else float(v)

def rising_edges(signal, threshold):
    active=np.asarray(signal)>threshold
    if len(active)<2: return np.array([],dtype=int)
    return np.flatnonzero((~active[:-1]) & active[1:]) + 1

def interval_cv(indices, time):
    if len(indices)<3: return 0.0
    ints=np.diff(time[indices]); m=np.mean(ints)
    if not np.isfinite(m) or abs(m)<EPS: return 0.0
    return float(np.std(ints)/m)

def weighted_center(vals, coords):
    denom=np.sum(vals,axis=1); numer=vals@coords
    return np.divide(numer, denom, out=np.zeros_like(numer,dtype=float), where=np.abs(denom)>EPS)

def extract_features(path):
    df=pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    if df.shape[1] != 19:
        raise ValueError(f"{path.name}: expected 19 columns, got {df.shape[1]}")
    a=df.to_numpy(float)
    time=a[:,0]; left=a[:,1:9]; right=a[:,9:17]; lt=a[:,17]; rt=a[:,18]
    duration=float(max(time[-1]-time[0], EPS))
    f={
        "duration_s":duration,
        "left_total_mean":float(np.mean(lt)),
        "left_total_std":float(np.std(lt)),
        "left_total_median":float(np.median(lt)),
        "right_total_mean":float(np.mean(rt)),
        "right_total_std":float(np.std(rt)),
        "right_total_median":float(np.median(rt)),
        "total_force_mean":float(np.mean(lt+rt)),
        "total_force_std":float(np.std(lt+rt)),
        "left_right_force_corr":safe_corr(lt,rt),
    }
    asym=(lt-rt)/(np.abs(lt)+np.abs(rt)+EPS)
    f["asymmetry_mean"]=float(np.mean(asym))
    f["asymmetry_abs_mean"]=float(np.mean(np.abs(asym)))
    f["asymmetry_std"]=float(np.std(asym))
    lthr=max(float(np.nanmax(lt))*0.05,5.0); rthr=max(float(np.nanmax(rt))*0.05,5.0)
    la=lt>lthr; ra=rt>rthr
    f["left_stance_fraction"]=float(np.mean(la))
    f["right_stance_fraction"]=float(np.mean(ra))
    f["double_support_fraction"]=float(np.mean(la & ra))
    f["single_support_fraction"]=float(np.mean(la ^ ra))
    le=rising_edges(lt,lthr); re_=rising_edges(rt,rthr)
    f["left_step_events_per_min"]=float(len(le)/duration*60.0)
    f["right_step_events_per_min"]=float(len(re_)/duration*60.0)
    f["left_step_interval_cv"]=interval_cv(le,time)
    f["right_step_interval_cv"]=interval_cv(re_,time)
    for i in range(8):
        f[f"L{i+1}_mean"]=float(np.mean(left[:,i])); f[f"L{i+1}_std"]=float(np.std(left[:,i]))
        f[f"R{i+1}_mean"]=float(np.mean(right[:,i])); f[f"R{i+1}_std"]=float(np.std(right[:,i]))
    for prefix, series in [
        ("left_cop_x",weighted_center(left,LEFT_X)),
        ("left_cop_y",weighted_center(left,LEFT_Y)),
        ("right_cop_x",weighted_center(right,RIGHT_X)),
        ("right_cop_y",weighted_center(right,RIGHT_Y)),
    ]:
        f[f"{prefix}_mean"]=float(np.mean(series))
        f[f"{prefix}_std"]=float(np.std(series))
        f[f"{prefix}_range"]=float(np.ptp(series))
    return f

def build_table():
    rows=[]
    files=sorted(DATA_DIR.glob("*_01.txt"))
    if not files:
        raise FileNotFoundError(f"No *_01.txt files in {DATA_DIR}")
    for p in files:
        m=FILENAME_RE.match(p.name)
        if not m: continue
        study, group, subject_num, walk_num=m.groups()
        label=1 if group.lower()=="pt" else 0
        sid=f"{study}{group}{subject_num}"
        try: feats=extract_features(p)
        except Exception as e:
            print(f"[skip] {p.name}: {e}"); continue
        rows.append({"file":p.name,"subject_id":sid,"study":study,"label":label,**feats})
    if not rows: raise RuntimeError("No valid gait files parsed.")
    return pd.DataFrame(rows)

def metric_dict(y,prob,pred):
    d={
      "accuracy":float(accuracy_score(y,pred)),
      "sensitivity":float(recall_score(y,pred,pos_label=1,zero_division=0)),
      "specificity":float(recall_score(y,pred,pos_label=0,zero_division=0)),
      "f1":float(f1_score(y,pred,zero_division=0))
    }
    try: d["roc_auc"]=float(roc_auc_score(y,prob))
    except Exception: d["roc_auc"]=None
    return d

def main():
    table=build_table()
    participants=table[["subject_id","label"]].drop_duplicates("subject_id")
    tr_ids, te_ids=train_test_split(
        participants["subject_id"], test_size=0.25, random_state=42,
        stratify=participants["label"]
    )
    tr=table[table.subject_id.isin(set(tr_ids))]
    te=table[table.subject_id.isin(set(te_ids))]
    excluded={"file","subject_id","study","label"}
    cols=[c for c in table.columns if c not in excluded]
    Xtr=tr[cols].replace([np.inf,-np.inf],np.nan).fillna(0.0)
    Xte=te[cols].replace([np.inf,-np.inf],np.nan).fillna(0.0)
    ytr=tr.label.astype(int); yte=te.label.astype(int)
    models={
      "Logistic Regression":Pipeline([("scale",StandardScaler()),("model",LogisticRegression(max_iter=3000,class_weight="balanced",random_state=42))]),
      "RBF SVM":Pipeline([("scale",StandardScaler()),("model",SVC(kernel="rbf",probability=True,class_weight="balanced",random_state=42))]),
      "Random Forest":RandomForestClassifier(n_estimators=350,max_depth=7,min_samples_leaf=2,class_weight="balanced",random_state=42),
    }
    report={
      "dataset":"PhysioNet Gait in Parkinson's Disease",
      "protocol":"usual-walk recordings only (*_01.txt)",
      "samples":int(len(table)),
      "participants":int(table.subject_id.nunique()),
      "controls":int((table.label==0).sum()),
      "parkinsons":int((table.label==1).sum()),
      "train_participants":int(tr.subject_id.nunique()),
      "test_participants":int(te.subject_id.nunique()),
      "n_features":len(cols),
      "split":"participant-separated stratified holdout",
      "models":{}
    }
    best_name=None; best_score=-math.inf; best_model=None
    for name,model in models.items():
        model.fit(Xtr,ytr)
        prob=model.predict_proba(Xte)[:,1]
        pred=(prob>=0.5).astype(int)
        res=metric_dict(yte,prob,pred)
        report["models"][name]=res
        score=res["roc_auc"] if res["roc_auc"] is not None else res["accuracy"]
        if score>best_score:
            best_score=score; best_name=name; best_model=model
    report["selected_model"]=best_name
    joblib.dump({"model":best_model,"model_name":best_name,"feature_names":cols,"protocol":"usual_walk_01"}, MODEL_DIR/"gait_branch.joblib")
    table.to_csv(OUT_DIR/"gait_feature_table.csv",index=False)
    (OUT_DIR/"gait_benchmark_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("\n=== GAIT BRANCH READY ===")
    print(f"Samples/participants : {len(table)} / {table.subject_id.nunique()}")
    print(f"Control / PD         : {(table.label==0).sum()} / {(table.label==1).sum()}")
    print(f"Features             : {len(cols)}")
    print(f"Train / test subjects: {tr.subject_id.nunique()} / {te.subject_id.nunique()}")
    for name,res in report["models"].items():
        auc="NA" if res["roc_auc"] is None else f"{res['roc_auc']:.3f}"
        print(f"{name:20s} acc={res['accuracy']:.3f} sens={res['sensitivity']:.3f} spec={res['specificity']:.3f} auc={auc}")
    print(f"\nSelected model: {best_name}")
    print("Saved: models/gait_branch.joblib")
    print("Saved: results/gait_benchmark_report.json")
    print("Saved: results/gait_feature_table.csv")

if __name__=="__main__":
    main()
