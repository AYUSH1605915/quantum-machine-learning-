from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd


@dataclass
class BranchStatus:
    name: str
    available: bool
    model_name: str | None = None
    roc_auc: float | None = None
    note: str | None = None


class MultimodalFusionEngine:
    """Late decision-level fusion for independently trained biomedical branches.

    The currently available cohorts are not paired across modalities, so this engine
    intentionally combines *branch-level model scores* rather than training a joint
    cross-modal classifier on mismatched people.
    """

    def __init__(self, root: str | os.PathLike = "."):
        self.root = Path(root)
        self.models_dir = self.root / "models"
        self.results_dir = self.root / "results"
        self._bundles: Dict[str, dict] = {}
        self._tables: Dict[str, pd.DataFrame] = {}
        self._reports: Dict[str, dict] = {}
        self.reload()

    def reload(self):
        self._bundles = {}
        self._tables = {}
        self._reports = {}

        self._load_branch(
            "gait",
            self.models_dir / "gait_branch.joblib",
            self.results_dir / "gait_feature_table.csv",
            self.results_dir / "gait_benchmark_report.json",
        )
        self._load_branch(
            "handwriting",
            self.models_dir / "handwriting_branch.joblib",
            self.results_dir / "handwriting_subject_features.csv",
            self.results_dir / "handwriting_benchmark_report.json",
        )
        self._load_json("voice", self.results_dir / "parkinsons_benchmark_report.json")
        return self

    def _load_json(self, name: str, path: Path):
        if path.exists():
            try:
                self._reports[name] = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                pass

    def _load_branch(self, name: str, model_path: Path, table_path: Path, report_path: Path):
        if model_path.exists():
            try:
                self._bundles[name] = joblib.load(model_path)
            except Exception:
                pass
        if table_path.exists():
            try:
                self._tables[name] = pd.read_csv(table_path)
            except Exception:
                pass
        self._load_json(name, report_path)

    @staticmethod
    def _best_auc(report: dict | None) -> float | None:
        if not report:
            return None

        # Gait / handwriting report shape.
        models = report.get("models")
        if isinstance(models, dict):
            vals = []
            for item in models.values():
                try:
                    auc = item.get("roc_auc")
                    if auc is not None:
                        vals.append(float(auc))
                except Exception:
                    pass
            if vals:
                return max(vals)

        # Voice report shape.
        models = report.get("models_evaluation")
        if isinstance(models, list):
            vals = []
            for item in models:
                try:
                    auc = item.get("roc_auc")
                    if auc is not None:
                        vals.append(float(auc))
                except Exception:
                    pass
            if vals:
                return max(vals)
        return None

    def branch_status(self) -> dict:
        out = {}
        for name in ("gait", "handwriting"):
            bundle = self._bundles.get(name)
            auc = self._best_auc(self._reports.get(name))
            out[name] = {
                "available": bundle is not None,
                "model_name": bundle.get("model_name") if bundle else None,
                "roc_auc": auc,
                "sample_table_available": name in self._tables,
            }
        out["voice"] = {
            "available": True,
            "model_name": "RF + VQC + Fidelity-Kernel QSVM",
            "roc_auc": self._best_auc(self._reports.get("voice")),
            "sample_table_available": True,
        }
        return out

    def get_sample_features(self, modality: str, label: int) -> dict:
        if modality not in self._tables:
            raise FileNotFoundError(f"{modality} feature table is not available")
        table = self._tables[modality]
        if "label" not in table.columns:
            raise ValueError(f"{modality} feature table has no label column")
        subset = table[table["label"].astype(int) == int(label)]
        if subset.empty:
            raise ValueError(f"No {modality} sample found for label={label}")

        # Deterministic middle sample, avoiding a random presentation result.
        row = subset.iloc[len(subset) // 2]
        bundle = self._bundles.get(modality)
        if not bundle:
            raise FileNotFoundError(f"{modality} model bundle is not available")
        features = {
            name: (None if pd.isna(row.get(name)) else float(row.get(name)))
            for name in bundle.get("feature_names", [])
        }
        meta = {}
        for key in ("file", "subject_id", "study", "_ID_EXAM"):
            if key in row.index and not pd.isna(row[key]):
                value = row[key]
                meta[key] = int(value) if isinstance(value, (np.integer,)) else str(value)
        return {"features": features, "meta": meta, "known_label": int(label)}

    def predict_branch(self, modality: str, features: dict) -> dict:
        bundle = self._bundles.get(modality)
        if bundle is None:
            raise FileNotFoundError(f"{modality} model has not been trained/saved")

        names = bundle.get("feature_names", [])
        row = {name: features.get(name, np.nan) for name in names}
        frame = pd.DataFrame([row], columns=names)
        model = bundle["model"]
        score = float(model.predict_proba(frame)[0, 1])
        return {
            "score": score,
            "model_name": bundle.get("model_name", type(model).__name__),
            "feature_count": len(names),
            "roc_auc": self._best_auc(self._reports.get(modality)),
        }

    @staticmethod
    def _reliability_strength(auc: float | None) -> float:
        # AUC=0.5 is chance. Weight only the discrimination above chance and keep
        # a small floor so an available branch is not numerically erased.
        if auc is None or not np.isfinite(auc):
            return 0.10
        return max(float(auc) - 0.50, 0.05)

    def fuse(self, branch_scores: dict[str, float], branch_aucs: dict[str, float | None]) -> dict:
        clean = {
            name: float(np.clip(score, 0.0, 1.0))
            for name, score in branch_scores.items()
            if score is not None and np.isfinite(score)
        }
        if not clean:
            raise ValueError("At least one modality score is required for fusion")

        raw_weights = {
            name: self._reliability_strength(branch_aucs.get(name))
            for name in clean
        }
        denom = sum(raw_weights.values()) or 1.0
        weights = {name: value / denom for name, value in raw_weights.items()}
        fused = sum(clean[name] * weights[name] for name in clean)

        vals = np.array(list(clean.values()), dtype=float)
        dispersion = float(np.std(vals)) if len(vals) > 1 else 0.0
        if len(vals) <= 1:
            agreement = "Single modality"
        elif dispersion < 0.10:
            agreement = "High"
        elif dispersion < 0.20:
            agreement = "Moderate"
        else:
            agreement = "Low"

        if fused < 0.40:
            band = "Lower multimodal pattern"
            band_key = "low"
        elif fused < 0.60:
            band = "Intermediate / indeterminate multimodal pattern"
            band_key = "medium"
        else:
            band = "Elevated multimodal Parkinsonian pattern"
            band_key = "high"

        return {
            "index": fused,
            "index_percent": round(fused * 100.0, 1),
            "band": band,
            "band_key": band_key,
            "agreement": agreement,
            "dispersion": round(dispersion, 4),
            "normalized_weights": {k: round(v, 4) for k, v in weights.items()},
            "branch_scores": {k: round(v, 4) for k, v in clean.items()},
            "method": "validation-weighted late decision fusion",
            "note": (
                "Branch weights are derived from research-validation ROC-AUC above chance and are used for the "
                "prototype fusion demonstration; they are not clinical reliability estimates."
            ),
        }
