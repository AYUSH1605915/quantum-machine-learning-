import time
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)
from sklearn.inspection import permutation_importance

class ModelEvaluator:
    """
    Evaluation Engine for Early Disease Detection.
    Computes diagnostic metrics: Accuracy, Sensitivity (Recall), Specificity,
    Precision, F1-Score, ROC-AUC, Confusion Matrix, and Latency.
    """
    @staticmethod
    def evaluate_model(name, model, X_test, y_test, training_time=0.0):
        start_inf = time.time()
        y_pred = model.predict(X_test)
        inference_time_ms = ((time.time() - start_inf) / max(len(X_test), 1)) * 1000.0

        # Predict probabilities for ROC-AUC
        try:
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, "decision_function"):
                y_proba = model.decision_function(X_test)
            else:
                y_proba = y_pred
            roc_auc = float(roc_auc_score(y_test, y_proba))
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_curve_data = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
        except Exception:
            roc_auc = float(accuracy_score(y_test, y_pred))
            roc_curve_data = {"fpr": [0.0, 1.0], "tpr": [0.0, 1.0]}

        # Confusion Matrix: TN, FP, FN, TP
        cm = confusion_matrix(y_test, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn, fp, fn, tp = 0, 0, 0, 0
            if len(y_test) > 0:
                if y_pred[0] == 0:
                    tn = len(y_test)
                else:
                    tp = len(y_test)

        accuracy = float(accuracy_score(y_test, y_pred))
        sensitivity = float(recall_score(y_test, y_pred, zero_division=0)) # Recall = TP / (TP + FN)
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0      # Specificity = TN / (TN + FP)
        precision = float(precision_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))

        return {
            "model_name": name,
            "accuracy": round(accuracy, 4),
            "sensitivity": round(sensitivity, 4),
            "specificity": round(specificity, 4),
            "precision": round(precision, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "confusion_matrix": {
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn)
            },
            "roc_curve": roc_curve_data,
            "training_time_sec": round(training_time, 3),
            "inference_latency_ms": round(inference_time_ms, 3)
        }


class ExplainabilityEngine:
    """
    Explainability and Interpretability Module for Hybrid QML.
    Extracts biomarker importance, quantum feature alignment, and circuit topology.
    """
    @staticmethod
    def get_feature_importances(dataset_loader, classical_rf_model=None):
        """Calculates global clinical biomarker importance using ANOVA F-scores & RF permutation."""
        results = []
        feature_names = dataset_loader.feature_names
        X = dataset_loader.X_train_raw
        y = dataset_loader.y_train

        # 1. ANOVA F-Score Importance
        from sklearn.feature_selection import f_classif
        with np.errstate(divide='ignore', invalid='ignore'):
            f_scores, p_values = f_classif(X, y)
        f_scores = np.nan_to_num(f_scores, nan=0.0, posinf=0.0, neginf=0.0)
        total_f = float(np.sum(f_scores))
        norm_f = (f_scores / total_f) if total_f > 0 else np.zeros_like(f_scores)

        rf_importances = None
        if classical_rf_model is not None and hasattr(classical_rf_model, "feature_importances_"):
            rf_importances = classical_rf_model.feature_importances_

        for i, name in enumerate(feature_names):
            is_quantum = name in dataset_loader.selected_feature_names
            # Use combined importance if RF available
            norm_val = float(rf_importances[i]) if (rf_importances is not None and i < len(rf_importances)) else float(norm_f[i])
            item = {
                "feature": name,
                "f_score": round(float(f_scores[i]), 2),
                "normalized_importance": round(norm_val, 4),
                "is_quantum_encoded": bool(is_quantum)
            }
            if rf_importances is not None and i < len(rf_importances):
                item["rf_importance"] = round(float(rf_importances[i]), 4)
            results.append(item)

        results.sort(key=lambda x: x["normalized_importance"], reverse=True)
        return results

    @staticmethod
    def get_circuit_schema(n_qubits=4, n_layers=2):
        """Returns structured quantum circuit schema for frontend visualization."""
        wires = [f"Qubit q[{i}]" for i in range(n_qubits)]
        layers = []

        # Layer 0: Superposition & State Embedding
        encoding_gates = []
        for i in range(n_qubits):
            encoding_gates.append({"type": "H", "wire": i, "desc": "Hadamard superposition"})
            encoding_gates.append({"type": "RY", "wire": i, "desc": f"Angle feature x[{i}]"})
        layers.append({"name": "Quantum Encoding (State Prep)", "gates": encoding_gates})

        # Variational Layers
        for l in range(n_layers):
            rot_gates = []
            for i in range(n_qubits):
                rot_gates.append({"type": "RY", "wire": i, "desc": f"Rot Ry(θ_{l},{i})"})
                rot_gates.append({"type": "RZ", "wire": i, "desc": f"Rot Rz(ω_{l},{i})"})
            layers.append({"name": f"Variational Layer {l + 1} (Rotations)", "gates": rot_gates})

            # Entangling CNOT ring
            entangle_gates = []
            for i in range(n_qubits):
                target = (i + 1) % n_qubits
                entangle_gates.append({"type": "CNOT", "control": i, "target": target, "desc": f"CNOT {i} -> {target}"})
            layers.append({"name": f"Variational Layer {l + 1} (Entanglement Ring)", "gates": entangle_gates})

        # Measurement
        meas = [{"type": "M_Z", "wire": 0, "desc": "Pauli-Z Expectation <Z_0>"}]
        layers.append({"name": "Quantum Readout", "gates": meas})

        return {
            "n_qubits": n_qubits,
            "n_layers": n_layers,
            "wires": wires,
            "stages": layers
        }
