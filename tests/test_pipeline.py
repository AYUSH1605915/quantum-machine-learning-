"""
Unit tests for QML-Liver platform modules.
Run with: python -m unittest discover tests
"""

import sys
import unittest
from pathlib import Path

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import numpy as np
from src.data.ingestion import get_dataset, load_ilpd
from src.data.feature_selection import EnsembleFeatureSelector, apply_smoteenn
from src.quantum.feature_maps import normalize_for_encoding, ZZFeatureMapEncoder
from src.quantum.qsvm import QuantumSVM
from src.quantum.vqc import VQC
from src.quantum.error_mitigation import compare_mitigated_vs_unmitigated
from src.explainability.qshap import QSHAPExplainer


class TestQMLLiverPipeline(unittest.TestCase):

    def test_01_data_ingestion(self):
        """Verify ILPD loading and 70/15/15 stratified split."""
        df = load_ilpd()
        self.assertIn("age", df.columns)
        self.assertIn("label", df.columns)
        self.assertFalse(df.isnull().any().any())

        X_tr, X_va, X_te, y_tr, y_va, y_te = get_dataset("ILPD")
        total = len(y_tr) + len(y_va) + len(y_te)
        self.assertAlmostEqual(len(y_tr) / total, 0.70, delta=0.03)
        self.assertAlmostEqual(len(y_va) / total, 0.15, delta=0.03)
        self.assertAlmostEqual(len(y_te) / total, 0.15, delta=0.03)

    def test_02_feature_selection_and_smoteenn(self):
        """Verify reduction to target qubit budget and SMOTEENN balancing."""
        X_tr, _, _, y_tr, _, _ = get_dataset("ILPD")
        target_qubits = 8
        selector = EnsembleFeatureSelector(n_features=target_qubits)
        X_sel = selector.fit_transform(X_tr, y_tr)

        self.assertEqual(X_sel.shape[1], target_qubits)
        X_res, y_res = apply_smoteenn(X_sel, y_tr)
        self.assertGreater(len(y_res), 0)
        self.assertEqual(len(X_res), len(y_res))

    def test_03_quantum_feature_encoding(self):
        """Verify MinMax scaling to [0, 2pi] and state normalization."""
        X = np.random.uniform(10, 100, size=(5, 4))
        X_norm, scaler = normalize_for_encoding(X)
        self.assertTrue(np.all(X_norm >= 0.0))
        self.assertTrue(np.all(X_norm <= 2.0 * np.pi + 1e-5))

        encoder = ZZFeatureMapEncoder(n_qubits=4, reps=1, backend="pennylane")
        state = encoder.encode(X_norm[0])
        self.assertAlmostEqual(np.linalg.norm(state), 1.0, places=5)

    def test_04_error_mitigation_zne(self):
        """Verify Digital ZNE error reduction."""
        res = compare_mitigated_vs_unmitigated(ideal_val=0.85, noise_rate=0.15)
        self.assertIn("mitigated", res)
        self.assertIn("error_reduction_pct", res)
        self.assertGreaterEqual(res["error_reduction_pct"], 0.0)

    def test_05_vqc_and_qshap(self):
        """Verify VQC fit and Q-SHAP feature attribution."""
        np.random.seed(42)
        X_dummy = np.random.randn(20, 4)
        y_dummy = np.random.choice([0, 1], size=20)

        vqc = VQC(n_qubits=4, n_layers=1, max_iter=5, optimizer="spsa")
        vqc.fit(X_dummy, y_dummy)

        preds = vqc.predict(X_dummy[:3])
        self.assertEqual(len(preds), 3)

        explainer = QSHAPExplainer(model=vqc, feature_names=["F1", "F2", "F3", "F4"], n_samples=5)
        shap_vals = explainer.explain(X_dummy[0])
        self.assertEqual(len(shap_vals), 4)


if __name__ == "__main__":
    unittest.main()
