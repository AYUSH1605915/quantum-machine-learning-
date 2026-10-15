
from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from pathlib import Path

import joblib
import mne
import numpy as np
import pandas as pd
from scipy import signal

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=RuntimeWarning)
mne.set_log_level("ERROR")

ROOT = Path(__file__).resolve().parents[1]
EEG_DIR = ROOT / "data" / "multimodal" / "EEG" / "ds007526"
PARTICIPANTS = EEG_DIR / "participants.tsv"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
FEATURE_CACHE = RESULTS_DIR / "eeg_feature_table.csv"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def clean_name(name: str) -> str:
    return (
        str(name)
        .strip()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )


def extract_eeg_features(set_path: Path) -> dict:
    raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose="ERROR")

    # EEG only. Do not use demographics or clinical severity as model inputs.
    eeg_picks = mne.pick_types(
        raw.info,
        meg=False,
        eeg=True,
        eog=False,
        ecg=False,
        emg=False,
        stim=False,
        misc=False,
        exclude="bads",
    )
    if len(eeg_picks) == 0:
        raise RuntimeError(f"No EEG channels found in {set_path}")

    data = raw.get_data(picks=eeg_picks).astype(np.float64, copy=False)
    ch_names = [clean_name(raw.ch_names[i]) for i in eeg_picks]
    sfreq = float(raw.info["sfreq"])

    # Drop a few seconds from each end to reduce edge/setup contamination.
    trim_s = 5.0
    trim = int(round(trim_s * sfreq))
    if data.shape[1] > 2 * trim + int(20 * sfreq):
        data = data[:, trim:-trim]

    # Robust clipping based on each channel's own median absolute deviation.
    # This limits extreme transient artifacts without using label information.
    med = np.nanmedian(data, axis=1, keepdims=True)
    mad = np.nanmedian(np.abs(data - med), axis=1, keepdims=True)
    robust_scale = 1.4826 * mad
    robust_scale[~np.isfinite(robust_scale) | (robust_scale <= 0)] = np.nan
    lower = med - 8.0 * robust_scale
    upper = med + 8.0 * robust_scale
    data = np.where(np.isfinite(lower) & np.isfinite(upper),
                    np.clip(data, lower, upper),
                    data)

    # Welch PSD is computed directly in 1-45 Hz, so a separate time-domain
    # band-pass step is unnecessary for these spectral features.
    nperseg = min(int(round(4.0 * sfreq)), data.shape[1])
    nperseg = max(nperseg, 256)
    noverlap = nperseg // 2

    freqs, psd = signal.welch(
        data,
        fs=sfreq,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        return_onesided=True,
        scaling="density",
        axis=-1,
        average="median",
    )

    total_mask = (freqs >= 1.0) & (freqs <= 45.0)
    total_power = np.trapezoid(psd[:, total_mask], freqs[total_mask], axis=1)
    total_power = np.maximum(total_power, np.finfo(float).eps)

    feats: dict[str, float] = {
        "sampling_rate_hz": sfreq,
        "duration_used_sec": float(data.shape[1] / sfreq),
        "n_eeg_channels": float(len(ch_names)),
    }

    band_rel = {}

    for band, (lo, hi) in BANDS.items():
        # Half-open bands avoid double-counting boundary frequencies.
        if band == "gamma":
            mask = (freqs >= lo) & (freqs <= hi)
        else:
            mask = (freqs >= lo) & (freqs < hi)

        abs_power = np.trapezoid(psd[:, mask], freqs[mask], axis=1)
        rel_power = abs_power / total_power
        band_rel[band] = rel_power

        for ch, value in zip(ch_names, rel_power):
            feats[f"{band}_rel__{ch}"] = float(value)

        # Label-independent global summaries improve robustness if a channel
        # is absent in a small number of recordings.
        feats[f"{band}_global_mean"] = float(np.nanmean(rel_power))
        feats[f"{band}_global_median"] = float(np.nanmedian(rel_power))
        feats[f"{band}_global_std"] = float(np.nanstd(rel_power))

    eps = 1e-12
    alpha = band_rel["alpha"]
    theta = band_rel["theta"]
    beta = band_rel["beta"]
    delta = band_rel["delta"]

    feats["ratio_theta_alpha_global"] = float(
        np.nanmedian(theta / np.maximum(alpha, eps))
    )
    feats["ratio_beta_alpha_global"] = float(
        np.nanmedian(beta / np.maximum(alpha, eps))
    )
    feats["ratio_slow_fast_global"] = float(
        np.nanmedian(
            (delta + theta) / np.maximum(alpha + beta, eps)
        )
    )

    return feats


def build_feature_table() -> pd.DataFrame:
    if not PARTICIPANTS.exists():
        raise FileNotFoundError(f"Missing participant metadata: {PARTICIPANTS}")

    participants = pd.read_csv(PARTICIPANTS, sep="\t")

    required = {"participant_id", "group"}
    missing = required - set(participants.columns)
    if missing:
        raise ValueError(f"participants.tsv missing required columns: {sorted(missing)}")

    participants = participants[
        participants["group"].isin(["HC", "PD"])
    ][["participant_id", "group"]].copy()

    rows = []
    failures = []
    start = time.time()
    total = len(participants)

    print(f"EEG participants queued: {total}")
    print("Extracting resting-state spectral features...")
    print()

    for idx, rec in participants.reset_index(drop=True).iterrows():
        pid = str(rec["participant_id"])
        set_path = EEG_DIR / pid / "eeg" / f"{pid}_task-rest_eeg.set"

        try:
            if not set_path.exists():
                raise FileNotFoundError(str(set_path))

            feats = extract_eeg_features(set_path)
            feats["participant_id"] = pid
            feats["group"] = str(rec["group"])
            feats["label"] = 1 if rec["group"] == "PD" else 0
            rows.append(feats)

        except Exception as exc:
            failures.append({"participant_id": pid, "error": repr(exc)})
            print(f"[WARN] {pid}: {exc}")

        done = idx + 1
        if done == 1 or done % 10 == 0 or done == total:
            elapsed = time.time() - start
            rate = elapsed / done
            eta = rate * (total - done)
            print(
                f"[{done:3d}/{total}] "
                f"elapsed={elapsed/60:5.1f} min "
                f"ETA={eta/60:5.1f} min "
                f"successful={len(rows)}"
            )

    if not rows:
        raise RuntimeError("No EEG recordings were successfully processed.")

    table = pd.DataFrame(rows)

    failure_path = RESULTS_DIR / "eeg_feature_failures.json"
    failure_path.write_text(json.dumps(failures, indent=2), encoding="utf-8")

    return table


def make_models(k: int):
    return {
        "Logistic Regression": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("select", SelectKBest(score_func=f_classif, k=k)),
                (
                    "model",
                    LogisticRegression(
                        max_iter=5000,
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
                ("select", SelectKBest(score_func=f_classif, k=k)),
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
                ("select", SelectKBest(score_func=f_classif, k=k)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        max_depth=8,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def metrics(y_true, prob):
    pred = (prob >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "sensitivity": float(recall_score(y_true, pred, pos_label=1, zero_division=0)),
        "specificity": float(recall_score(y_true, pred, pos_label=0, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, prob)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Recompute EEG spectral features instead of using the cached CSV.",
    )
    args = parser.parse_args()

    if FEATURE_CACHE.exists() and not args.rebuild:
        print(f"Using cached EEG features: {FEATURE_CACHE}")
        table = pd.read_csv(FEATURE_CACHE)
    else:
        table = build_feature_table()
        table.to_csv(FEATURE_CACHE, index=False)
        print(f"\nSaved feature cache: {FEATURE_CACHE}")

    metadata_cols = {
        "participant_id",
        "group",
        "label",
        "sampling_rate_hz",
        "duration_used_sec",
        "n_eeg_channels",
    }
    feature_cols = [c for c in table.columns if c not in metadata_cols]

    # Exclude any accidental non-numeric fields.
    feature_cols = [
        c for c in feature_cols
        if pd.api.types.is_numeric_dtype(table[c])
    ]

    X = table[feature_cols].replace([np.inf, -np.inf], np.nan)
    y = table["label"].astype(int).to_numpy()

    controls = int((y == 0).sum())
    pd_count = int((y == 1).sum())

    if controls < 5 or pd_count < 5:
        raise RuntimeError(
            f"Not enough samples for 5-fold stratified CV: HC={controls}, PD={pd_count}"
        )

    # Feature selection is INSIDE every pipeline, so each CV fold selects
    # features using training subjects only.
    k = min(32, len(feature_cols))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    report = {
        "dataset": "PD-EEG ds007526 resting-state EEG",
        "participants_used": int(len(table)),
        "healthy_controls": controls,
        "parkinsons": pd_count,
        "raw_feature_count": int(len(feature_cols)),
        "selected_features_per_fold": int(k),
        "validation": "5-fold stratified participant-level cross-validation",
        "positive_class": "PD",
        "negative_class": "HC",
        "clinical_and_demographic_inputs_used": False,
        "spectral_bands_hz": BANDS,
        "models": {},
    }

    print("\nTraining EEG classifiers with subject-level cross-validation...")
    best_name = None
    best_auc = -np.inf
    best_model = None

    for name, model in make_models(k).items():
        print(f"  {name} ...", flush=True)
        prob = cross_val_predict(
            model,
            X,
            y,
            cv=cv,
            method="predict_proba",
            n_jobs=None,
        )[:, 1]

        result = metrics(y, prob)
        report["models"][name] = result

        if result["roc_auc"] > best_auc:
            best_auc = result["roc_auc"]
            best_name = name
            best_model = model

    best_model.fit(X, y)

    bundle = {
        "model": best_model,
        "model_name": best_name,
        "feature_names": feature_cols,
        "positive_label": "Parkinson's",
        "negative_label": "Healthy Control",
        "input_type": "resting-state EEG spectral features",
        "dataset": "ds007526",
        "validation_auc": float(best_auc),
    }

    model_path = MODELS_DIR / "eeg_branch.joblib"
    report_path = RESULTS_DIR / "eeg_benchmark_report.json"

    joblib.dump(bundle, model_path)
    report["selected_model"] = best_name
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== EEG BRANCH READY ===")
    print(f"Participants        : {len(table)}")
    print(f"Control / PD        : {controls} / {pd_count}")
    print(f"Raw EEG features    : {len(feature_cols)}")
    print(f"Selected per fold   : {k}")
    print("Validation          : 5-fold participant-level CV")

    for name, result in report["models"].items():
        print(
            f"{name:20s} "
            f"acc={result['accuracy']:.3f} "
            f"sens={result['sensitivity']:.3f} "
            f"spec={result['specificity']:.3f} "
            f"f1={result['f1']:.3f} "
            f"auc={result['roc_auc']:.3f}"
        )

    print(f"\nSelected model: {best_name}")
    print(f"Saved: {model_path}")
    print(f"Saved: {report_path}")
    print(f"Saved: {FEATURE_CACHE}")
    print("\nNOTE: these are cross-validated research-cohort results, not clinical validation.")


if __name__ == "__main__":
    main()
