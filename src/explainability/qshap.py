"""
Module: src.explainability.qshap
Implements Q-SHAP (Quantum Shapley Value Explainer) via Monte Carlo feature ablation.
Computes patient-specific biomarker attributions:
  - Positive Shapley values (Red) indicate drivers pushing towards MASLD / fatty liver.
  - Negative Shapley values (Green) indicate protective factors pushing towards healthy.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class QSHAPExplainer:
    """
    Q-SHAP Explainer for Quantum Machine Learning Models (VQC & QSVM).
    Evaluates marginal contributions of quantum-encoded features using Monte Carlo permutations.
    """

    def __init__(self, model: Any, feature_names: List[str], baseline_values: Optional[np.ndarray] = None, n_samples: int = 30):
        """
        Args:
            model: Trained VQC or QuantumSVM instance.
            feature_names: Names corresponding to features input to the model.
            baseline_values: Baseline vector for ablated features (defaults to zeros or mean).
            n_samples: Number of Monte Carlo permutation rounds.
        """
        self.model = model
        self.feature_names = feature_names
        self.n_features = len(feature_names)
        self.n_samples = n_samples

        if baseline_values is None:
            self.baseline = np.zeros(self.n_features)
        else:
            self.baseline = np.array(baseline_values)

    def _ablate_and_predict(self, x: np.ndarray, active_indices: List[int]) -> float:
        """
        Runs model prediction with only active_indices retaining their patient values,
        and all other features replaced by the reference baseline.
        """
        x_perturbed = self.baseline.copy()
        for idx in active_indices:
            x_perturbed[idx] = x[idx]

        # Predict probability of positive class (MASLD)
        x_2d = np.array([x_perturbed])
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(x_2d)
            return float(probs[0, 1])
        elif hasattr(self.model, "predict"):
            return float(self.model.predict(x_2d)[0])
        return 0.5

    def _shapley_value(self, x: np.ndarray, feature_idx: int) -> float:
        """
        Approximates the Shapley value for feature_idx via Monte Carlo permutation sampling.
        """
        marginal_contributions = []
        all_indices = list(range(self.n_features))

        for _ in range(self.n_samples):
            # 1. Random permutation of all features
            perm = list(np.random.permutation(all_indices))
            pos = perm.index(feature_idx)

            # 2. Coalition without feature and with feature
            S_without = perm[:pos]
            S_with = perm[:pos + 1]

            # 3. Marginal difference
            val_with = self._ablate_and_predict(x, S_with)
            val_without = self._ablate_and_predict(x, S_without)
            marginal_contributions.append(val_with - val_without)

        return float(np.mean(marginal_contributions))

    def explain(self, x: np.ndarray) -> Dict[str, float]:
        """
        Computes Q-SHAP Shapley values for all features for a single patient record.
        
        Args:
            x: Patient feature vector.

        Returns:
            Dict[str, float]: Mapping of {feature_name: shapley_value}
        """
        explanations = {}
        for i, name in enumerate(self.feature_names):
            phi_i = self._shapley_value(x, i)
            explanations[name] = round(phi_i, 4)
        return explanations

    def plot_explanation(
        self,
        x: np.ndarray,
        patient_id: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Generates and saves a horizontal bar chart of Q-SHAP feature attributions.
        Red = positive contribution (pushes toward MASLD).
        Green = negative contribution (pushes toward Healthy).
        """
        explanations = self.explain(x)

        # Sort features by attribution magnitude
        sorted_feats = sorted(explanations.items(), key=lambda item: abs(item[1]), reverse=False)
        names = [item[0] for item in sorted_feats]
        vals = [item[1] for item in sorted_feats]
        colors = ["#f43f5e" if v >= 0 else "#10b981" for v in vals]

        if save_path is None:
            pid = patient_id if patient_id is not None else "0"
            save_path = f"reports/qshap_patient_{pid}.png"

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(8, 4.8))
        bars = plt.barh(names, vals, color=colors, height=0.55, edgecolor="white", linewidth=0.8)
        plt.axvline(0, color="#94a3b8", linestyle="--", alpha=0.7)

        title_id = f" — Patient #{patient_id}" if patient_id else ""
        plt.title(f"Q-SHAP Quantum Feature Explanation{title_id}", fontweight="bold", fontsize=13)
        plt.xlabel("Shapley Value (Marginal Contribution to MASLD Risk)", fontweight="bold")
        
        # Custom legend elements
        import matplotlib.patches as mpatches
        red_patch = mpatches.Patch(color="#f43f5e", label="Pushes toward MASLD (+)")
        green_patch = mpatches.Patch(color="#10b981", label="Pushes toward Healthy (-)")
        plt.legend(handles=[red_patch, green_patch], loc="lower right", framealpha=0.8)

        plt.grid(axis="x", linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

        logger.info("Saved Q-SHAP explanation plot to %s", save_path)
        return explanations
