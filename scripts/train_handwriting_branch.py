
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "multimodal" / "Images" / "HandPD"
OUT_DIR = ROOT / "results"
MODEL_DIR = ROOT / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SPIRAL_PATH = DATA_DIR / "Spiral_HandPD.csv"
MEANDER_PATH = DATA_DIR / "Meander_HandPD.csv"

META_COLS = {
    "_ID_EXAM",
    "IMAGE_NAME",
    "ID_PATIENT",
    "CLASS_TYPE",
    "GENDER",
    "RIGH/LEFT-HANDED",
    "AGE",
}


def aggregate_task(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    # Use only handwriting-derived numerical features.
    feature_cols = [c for c in df.columns if c not in META_COLS]

    # One class per examination/session.
    label_check = df.groupby("_ID_EXAM")["CLASS_TYPE"].nunique()
    if (label_check > 1).any():
        bad = label_check[label_check > 1].index.tolist()
        raise ValueError(f"Inconsistent CLASS_TYPE inside exam IDs: {bad[:5]}")

    grouped_mean = df.groupby("_ID_EXAM")[feature_cols].mean().add_prefix(f"{prefix}_mean__")
    grouped_std = (
        df.groupby("_ID_EXAM")[feature_cols]
        .std(ddof=0)
        .fillna(0.0)
        .add_prefix(f"{prefix}_std__")
    )

    meta = (
        df.groupby("_ID_EXAM")
        .agg(
            ID_PATIENT=("ID_PATIENT", "first"),
            CLASS_TYPE=("CLASS_TYPE", "first"),
            N_RECORDS=("IMAGE_NAME", "count"),
        )
    )

    return pd.concat([meta, grouped_mean, grouped_std], axis=1).reset_index()


def build_subject_table() -> pd.DataFrame:
    if not SPIRAL_PATH.exists() or not MEANDER_PATH.exists():
        raise FileNotFoundError(
            "HandPD CSV files not found. Expected:\n"
            f"  {SPIRAL_PATH}\n"
            f"  {MEANDER_PATH}"
        )

    spiral = pd.read_csv(SPIRAL_PATH)
    meander = pd.read_csv(MEANDER_PATH)

    spiral_agg = aggregate_task(spiral, "spiral")
    meander_agg = aggregate_task(meander, "meander")

    merged = spiral_agg.merge(
        meander_agg,
        on="_ID_EXAM",
        suffixes=("_spiral", "_meander"),
        how="inner",
        validate="one_to_one",
    )

    if not np.all(
        merged["CLASS_TYPE_spiral"].to_numpy()
        == merged["CLASS_TYPE_meander"].to_numpy()
    ):
        raise ValueError("Spiral and meander labels disagree for at least one exam.")

    # Dataset encoding: 1 = healthy control, 2 = Parkinson's disease.
    merged["label"] = (merged["CLASS_TYPE_spiral"] == 2).astype(int)

    return merged


def model_set():
    return {
        "Logistic Regression": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=4000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "RBF SVM": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "model",
                    CalibratedClassifierCV(
                        estimator=SVC(
                            kernel="rbf",
                            class_weight="balanced",
                            random_state=42,
                        ),
                        method="sigmoid",
                        cv=3,
                        ensemble=False,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=400,
                        max_depth=7,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def summarize_metrics(y_true: np.ndarray, prob: np.ndarray) -> dict:
    pred = (prob >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "sensitivity": float(
            recall_score(y_true, pred, pos_label=1, zero_division=0)
        ),
        "specificity": float(
            recall_score(y_true, pred, pos_label=0, zero_division=0)
        ),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, prob)),
    }


def main():
    table = build_subject_table()

    excluded = {
        "_ID_EXAM",
        "ID_PATIENT_spiral",
        "ID_PATIENT_meander",
        "CLASS_TYPE_spiral",
        "CLASS_TYPE_meander",
        "N_RECORDS_spiral",
        "N_RECORDS_meander",
        "label",
    }
    feature_cols = [c for c in table.columns if c not in excluded]

    X = table[feature_cols].replace([np.inf, -np.inf], np.nan)
    y = table["label"].astype(int).to_numpy()

    # Subject/exam-level table: each _ID_EXAM appears exactly once after aggregation.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    report = {
        "dataset": "HandPD Spiral + Meander feature CSVs",
        "unit_of_evaluation": "_ID_EXAM (one row per examination/session)",
        "subjects_or_exams": int(table["_ID_EXAM"].nunique()),
        "controls": int((table["label"] == 0).sum()),
        "parkinsons": int((table["label"] == 1).sum()),
        "raw_spiral_records": int(pd.read_csv(SPIRAL_PATH).shape[0]),
        "raw_meander_records": int(pd.read_csv(MEANDER_PATH).shape[0]),
        "aggregated_feature_count": int(len(feature_cols)),
        "validation": "5-fold stratified exam-level cross-validation",
        "class_encoding": {"1": "Healthy Control", "2": "Parkinson's Disease"},
        "demographics_excluded_from_model": [
            "AGE",
            "GENDER",
            "RIGH/LEFT-HANDED",
        ],
        "models": {},
    }

    best_name = None
    best_auc = -1.0
    best_model = None

    for name, model in model_set().items():
        prob = cross_val_predict(
            model,
            X,
            y,
            cv=cv,
            method="predict_proba",
            n_jobs=None,
        )[:, 1]

        result = summarize_metrics(y, prob)
        report["models"][name] = result

        if result["roc_auc"] > best_auc:
            best_auc = result["roc_auc"]
            best_name = name
            best_model = model

    # Fit selected branch on all HandPD exams for product inference.
    best_model.fit(X, y)

    bundle = {
        "model": best_model,
        "model_name": best_name,
        "feature_names": feature_cols,
        "positive_label": "Parkinson's",
        "negative_label": "Healthy Control",
        "input_type": "HandPD aggregated spiral+meander features",
        "group_key": "_ID_EXAM",
    }

    model_path = MODEL_DIR / "handwriting_branch.joblib"
    report_path = OUT_DIR / "handwriting_benchmark_report.json"
    table_path = OUT_DIR / "handwriting_subject_features.csv"

    joblib.dump(bundle, model_path)
    table.to_csv(table_path, index=False)

    report["selected_model"] = best_name
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== HANDWRITING BRANCH READY ===")
    print(f"Exam-level samples : {table['_ID_EXAM'].nunique()}")
    print(f"Control / PD       : {(table['label']==0).sum()} / {(table['label']==1).sum()}")
    print(f"Features           : {len(feature_cols)}")
    print("Validation         : 5-fold stratified exam-level CV")

    for name, result in report["models"].items():
        print(
            f"{name:20s} "
            f"acc={result['accuracy']:.3f} "
            f"sens={result['sensitivity']:.3f} "
            f"spec={result['specificity']:.3f} "
            f"auc={result['roc_auc']:.3f}"
        )

    print(f"\nSelected model: {best_name}")
    print(f"Saved: {model_path}")
    print(f"Saved: {report_path}")
    print(f"Saved: {table_path}")


if __name__ == "__main__":
    main()
