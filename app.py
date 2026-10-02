import json
import os
import time
import traceback
import uuid
from datetime import datetime, timezone

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import numpy as np

from qml_platform.dataset_loader import BiomedicalDatasetLoader
from qml_platform.classical_models import ClassicalBaselines
from qml_platform.quantum_models import VariationalQuantumClassifier, QuantumSVM
from qml_platform.evaluation import ModelEvaluator, ExplainabilityEngine
from qml_platform.severity import ParkinsonSeverityRegressor
from qml_platform.multimodal_fusion import MultimodalFusionEngine


app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

DATA_FILE = os.environ.get("PARKINSONS_DATA", os.path.join("data", "parkinsons.data"))

STATE = {
    "loader": None,
    "classical": None,
    "vqc": None,
    "qsvm": None,
    "evaluation_results": [],
    "feature_importance": [],
    "circuit_ascii": "",
    "circuit_schema": {},
    "last_trained": None,
    "is_training": False,
    "last_error": None,
    "severity": None,
    "severity_error": None,
    "multimodal": None,
}


def get_loader(n_qubits=4):
    if STATE["loader"] is None or STATE["loader"].n_qubits != n_qubits:
        STATE["loader"] = BiomedicalDatasetLoader(DATA_FILE, n_qubits=n_qubits).load_data()
    return STATE["loader"]


def get_multimodal_engine():
    if STATE["multimodal"] is None:
        STATE["multimodal"] = MultimodalFusionEngine(os.getcwd())
    else:
        STATE["multimodal"].reload()
    return STATE["multimodal"]


def voice_branch_score(patient):
    if not STATE["evaluation_results"]:
        raise RuntimeError("Voice QML models are not trained yet. Initialize the QML engine first.")
    loader = STATE["loader"]
    x_classical, x_quantum = loader.transform_single_patient(patient)
    rf = STATE["classical"].get_model("Random Forest")
    rf_score = float(rf.predict_proba(x_classical)[0, 1])
    vqc_score = float(STATE["vqc"].predict_proba(x_quantum)[0, 1])
    qsvm_score = float(STATE["qsvm"].predict_proba(x_quantum)[0, 1])
    return {
        "score": float(np.mean([rf_score, vqc_score, qsvm_score])),
        "components": {
            "Random Forest": rf_score,
            "Quantum VQC": vqc_score,
            "Quantum SVM": qsvm_score,
        },
    }


def severity_validation_gate():
    """Only expose the auxiliary UPDRS estimator if it beats a mean baseline.

    Negative R^2 means the model is worse than predicting the held-out cohort mean.
    In that situation we intentionally suppress patient-level estimates rather than
    displaying an unreliable severity number.
    """
    if STATE["severity"] is None:
        return {
            "passed": False,
            "status": "Unavailable",
            "reason": STATE["severity_error"] or "UPDRS regression model is not loaded.",
            "metrics": None,
        }

    metrics = STATE["severity"].metrics or {}
    motor_r2 = float(metrics.get("motor_r2", -999))
    total_r2 = float(metrics.get("total_r2", -999))
    passed = motor_r2 > 0.0 and total_r2 > 0.0
    if passed:
        reason = "Participant-separated validation outperformed a mean-prediction baseline for both UPDRS targets."
        status = "Passed"
    else:
        reason = (
            "Withheld: participant-separated UPDRS regression did not outperform a mean-prediction baseline "
            "(R² <= 0 on at least one target)."
        )
        status = "Withheld by validation gate"
    return {"passed": passed, "status": status, "reason": reason, "metrics": metrics}


def train_pipeline(epochs=8, n_qubits=4):
    loader = get_loader(n_qubits)
    classical = ClassicalBaselines()
    classical.fit_all(loader.X_train_classical, loader.y_train)

    results = []
    for name, model in classical.get_all_models().items():
        results.append(ModelEvaluator.evaluate_model(
            name, model, loader.X_test_classical, loader.y_test,
            training_time=classical.training_times[name]
        ))

    vqc = VariationalQuantumClassifier(
        n_qubits=n_qubits, n_layers=2, learning_rate=0.06,
        epochs=epochs, batch_size=16, random_state=42
    )
    vqc.fit(loader.X_train_quantum, loader.y_train)
    results.append(ModelEvaluator.evaluate_model(
        "Quantum VQC (PQC)", vqc, loader.X_test_quantum, loader.y_test,
        training_time=vqc.training_time
    ))

    qsvm = QuantumSVM(n_qubits=n_qubits, C=1.0)
    qsvm.fit(loader.X_train_quantum, loader.y_train)
    results.append(ModelEvaluator.evaluate_model(
        "Quantum SVM (Fidelity Kernel)", qsvm, loader.X_test_quantum, loader.y_test,
        training_time=qsvm.training_time
    ))

    rf = classical.get_model("Random Forest")
    importance = ExplainabilityEngine.get_feature_importances(loader, rf)

    severity = None
    severity_error = None
    updrs_path = os.path.join("data", "parkinsons_updrs.data")
    if os.path.exists(updrs_path):
        try:
            severity = ParkinsonSeverityRegressor(updrs_path).fit()
        except Exception as exc:
            severity_error = f"{type(exc).__name__}: {exc}"

    STATE.update({
        "classical": classical,
        "vqc": vqc,
        "qsvm": qsvm,
        "evaluation_results": results,
        "feature_importance": importance,
        "circuit_ascii": vqc.get_circuit_ascii(),
        "circuit_schema": ExplainabilityEngine.get_circuit_schema(n_qubits, 2),
        "last_trained": time.strftime("%Y-%m-%d %H:%M:%S"),
        "last_error": None,
        "severity": severity,
        "severity_error": severity_error,
    })

    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "parkinsons_benchmark_report.json"), "w", encoding="utf-8") as f:
        json.dump({
            "dataset_summary": loader.get_dataset_summary(),
            "models_evaluation": results,
            "feature_importance": importance,
            "circuit_ascii": STATE["circuit_ascii"],
            "validation_note": "Participant-grouped holdout split used when subject IDs are available.",
            "auxiliary_updrs_gate": severity_validation_gate(),
        }, f, indent=2)
    return results


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    data_exists = os.path.exists(DATA_FILE)
    return jsonify({
        "status": "ready" if STATE["evaluation_results"] else ("dataset_ready" if data_exists else "missing_dataset"),
        "dataset": DATA_FILE,
        "last_trained": STATE["last_trained"],
        "is_training": STATE["is_training"],
        "last_error": STATE["last_error"],
    })


@app.route("/api/dataset")
def api_dataset():
    try:
        loader = get_loader()
        return jsonify(loader.get_dataset_summary())
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/train", methods=["POST"])
def api_train():
    if STATE["is_training"]:
        return jsonify({"error": "Training already in progress"}), 409
    body = request.get_json(silent=True) or {}
    epochs = max(2, min(int(body.get("epochs", 8)), 30))
    n_qubits = max(2, min(int(body.get("n_qubits", 4)), 8))
    STATE["is_training"] = True
    try:
        results = train_pipeline(epochs=epochs, n_qubits=n_qubits)
        return jsonify({
            "status": "success",
            "message": f"Hybrid Parkinson's QML pipeline trained with {n_qubits} qubits and {epochs} VQC epochs.",
            "results": results,
            "dataset_summary": STATE["loader"].get_dataset_summary(),
            "auxiliary_updrs_gate": severity_validation_gate(),
        })
    except Exception as exc:
        STATE["last_error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        return jsonify({"error": STATE["last_error"]}), 500
    finally:
        STATE["is_training"] = False


@app.route("/api/benchmark")
def api_benchmark():
    loader = get_loader()
    return jsonify({
        "trained": bool(STATE["evaluation_results"]),
        "dataset_summary": loader.get_dataset_summary(),
        "models_evaluation": STATE["evaluation_results"],
        "feature_importance": STATE["feature_importance"],
        "circuit_ascii": STATE["circuit_ascii"],
        "circuit_schema": STATE["circuit_schema"],
        "severity_available": STATE["severity"] is not None,
        "severity_validation": STATE["severity"].metrics if STATE["severity"] is not None else None,
        "severity_error": STATE["severity_error"],
        "auxiliary_updrs_gate": severity_validation_gate(),
    })


@app.route("/api/sample/<kind>")
def api_sample(kind):
    loader = get_loader()
    label = 1 if kind.lower() in {"pd", "parkinsons", "positive", "high"} else 0
    return jsonify({
        "source": "held-out participant recording" if loader.test_indices is not None else "dataset recording",
        "known_dataset_label": label,
        "known_dataset_label_text": "Parkinson's cohort" if label == 1 else "Control cohort",
        "features": loader.get_demo_sample(label),
    })


def risk_band(score):
    # Presentation bands for the model index only; not validated clinical cutoffs.
    if score < 0.40:
        return "Lower model-indicated pattern", "low"
    if score < 0.60:
        return "Intermediate / indeterminate pattern", "medium"
    return "Elevated Parkinsonian pattern", "high"


def next_step_text(band_key):
    if band_key == "high":
        return (
            "For a research or triage workflow, consider clinician-led neurological evaluation and repeat/confirmatory "
            "assessment with additional modalities. Do not use this model index as a diagnosis."
        )
    if band_key == "medium":
        return (
            "Repeat voice acquisition or combine this result with additional clinical information before interpretation. "
            "An indeterminate model signal should not be treated as positive or negative."
        )
    return (
        "A lower model-indicated pattern does not rule out Parkinson's disease. Persistent clinical concern should still "
        "be evaluated by a qualified clinician."
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if not STATE["evaluation_results"]:
        return jsonify({"error": "Models are not trained yet. Initialize the QML engine first."}), 409

    patient = request.get_json(silent=True) or {}
    loader = STATE["loader"]
    x_classical, x_quantum = loader.transform_single_patient(patient)

    rf = STATE["classical"].get_model("Random Forest")
    rf_score = float(rf.predict_proba(x_classical)[0, 1])
    vqc_score = float(STATE["vqc"].predict_proba(x_quantum)[0, 1])
    qsvm_score = float(STATE["qsvm"].predict_proba(x_quantum)[0, 1])

    # Intentionally an index, not a calibrated disease probability.
    hybrid_index = float(np.mean([rf_score, vqc_score, qsvm_score]))
    band, band_key = risk_band(hybrid_index)
    model_std = float(np.std([rf_score, vqc_score, qsvm_score]))
    agreement = "High" if model_std < 0.08 else ("Moderate" if model_std < 0.16 else "Low")

    cohort = loader.compare_patient_with_cohort(patient)
    encoded = {
        name: round(float(x_quantum[0][i]), 4)
        for i, name in enumerate(loader.selected_feature_names)
    }

    gate = severity_validation_gate()
    severity_result = {
        "available": False,
        "withheld": False,
        "reason": "Symptom-burden estimation is only considered after an elevated model-indicated pattern.",
        "validation_gate": gate,
    }

    if hybrid_index >= 0.60:
        interpretation = (
            "The submitted voice-feature pattern is more consistent with patterns learned from the Parkinson's cohort. "
            "This is a screening-model signal, not a diagnosis."
        )
        if STATE["severity"] is not None and gate["passed"]:
            try:
                severity_result = STATE["severity"].predict_from_classification_features(patient)
                severity_result["validation_gate"] = gate
            except Exception as exc:
                severity_result = {
                    "available": False,
                    "withheld": True,
                    "reason": f"Auxiliary UPDRS module unavailable: {exc}",
                    "validation_gate": gate,
                }
        elif STATE["severity"] is not None and not gate["passed"]:
            severity_result = {
                "available": False,
                "withheld": True,
                "reason": gate["reason"],
                "validation_gate": gate,
            }
    elif hybrid_index < 0.40:
        interpretation = (
            "The submitted voice-feature pattern is more consistent with the control side of the trained model. "
            "A low model index does not rule out Parkinson's disease."
        )
    else:
        interpretation = (
            "The models show an intermediate or mixed pattern. The result should be treated as indeterminate rather than positive or negative."
        )

    provided = sum(1 for f in loader.feature_names if str(patient.get(f, "")).strip() != "")
    total = len(loader.feature_names)
    assessment_id = "QM-PD-" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4].upper()

    return jsonify({
        "meta": {
            "assessment_id": assessment_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model_scope": "Parkinson's voice-pattern research screening",
            "input_type": "Pre-extracted acoustic voice biomarker vector",
            "target_definition": "status: control=0 / Parkinson's cohort=1",
        },
        "assessment": {
            "hybrid_risk_index": round(hybrid_index * 100, 1),
            "risk_band": band,
            "risk_band_key": band_key,
            "model_agreement": agreement,
            "model_score_dispersion": round(model_std, 4),
            "interpretation": interpretation,
            "next_step": next_step_text(band_key),
            "disclaimer": (
                "Research screening prototype only. The index is not a calibrated probability of disease and does not establish or exclude a clinical diagnosis."
            ),
        },
        "model_scores": {
            "Random Forest": round(rf_score * 100, 1),
            "Quantum VQC": round(vqc_score * 100, 1),
            "Quantum SVM": round(qsvm_score * 100, 1),
        },
        "quantum_encoded_features": encoded,
        "cohort_comparison": cohort,
        "top_global_features": STATE["feature_importance"][:8],
        "symptom_burden": severity_result,
        "input_quality": {
            "provided_features": provided,
            "total_features": total,
            "completeness_percent": round(100.0 * provided / total, 1),
            "missing_features": total - provided,
            "missing_value_policy": "Missing inputs are imputed using medians learned from the training split only.",
        },
    })


@app.route("/api/multimodal/status")
def api_multimodal_status():
    engine = get_multimodal_engine()
    return jsonify({
        "branches": engine.branch_status(),
        "fusion_method": "validation-weighted late decision fusion",
        "cohort_note": (
            "Voice, gait, handwriting and EEG benchmark datasets contain different research participants. "
            "The current multimodal prototype therefore fuses independently validated branch scores rather than "
            "training a joint classifier on mismatched subjects."
        ),
    })


@app.route("/api/multimodal/sample/<kind>")
def api_multimodal_sample(kind):
    label = 1 if kind.lower() in {"pd", "parkinsons", "positive", "high"} else 0
    loader = get_loader()
    engine = get_multimodal_engine()
    return jsonify({
        "known_label": label,
        "known_label_text": "Parkinson's cohort" if label == 1 else "Control cohort",
        "bundle_scope": "Cross-cohort demonstration bundle; modality samples are not the same individual.",
        "voice": {
            "features": loader.get_demo_sample(label),
            "source": "held-out voice participant recording",
        },
        "gait": engine.get_sample_features("gait", label),
        "handwriting": engine.get_sample_features("handwriting", label),
        "eeg": engine.get_sample_features("eeg", label),
    })


@app.route("/api/multimodal/predict", methods=["POST"])
def api_multimodal_predict():
    payload = request.get_json(silent=True) or {}
    engine = get_multimodal_engine()

    # Only the voice branch depends on the trained QML engine. Gait,
    # handwriting and EEG can be assessed independently.
    voice_requested = isinstance(payload.get("voice"), dict) and bool(payload.get("voice"))
    if voice_requested and not STATE["evaluation_results"]:
        return jsonify({"error": "Initialize the voice QML engine before using the voice modality."}), 409
    scores = {}
    aucs = {}
    details = {}

    voice = payload.get("voice")
    if isinstance(voice, dict) and voice:
        vr = voice_branch_score(voice)
        scores["voice"] = vr["score"]
        status = engine.branch_status().get("voice", {})
        aucs["voice"] = status.get("roc_auc")
        details["voice"] = {
            "score_percent": round(vr["score"] * 100.0, 1),
            "model_name": "RF + VQC + Fidelity-Kernel QSVM",
            "components": {k: round(v * 100.0, 1) for k, v in vr["components"].items()},
            "roc_auc": aucs["voice"],
        }

    for modality in ("gait", "handwriting", "eeg"):
        features = payload.get(modality)
        if isinstance(features, dict) and features:
            result = engine.predict_branch(modality, features)
            scores[modality] = result["score"]
            aucs[modality] = result.get("roc_auc")
            details[modality] = {
                **result,
                "score_percent": round(result["score"] * 100.0, 1),
            }

    if not scores:
        return jsonify({"error": "No usable modality inputs were supplied."}), 400

    fused = engine.fuse(scores, aucs)
    assessment_id = "QM-MM-" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4].upper()

    return jsonify({
        "meta": {
            "assessment_id": assessment_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "modalities_used": list(scores.keys()),
            "fusion_type": fused["method"],
        },
        "assessment": {
            "multimodal_pattern_index": fused["index_percent"],
            "risk_band": fused["band"],
            "risk_band_key": fused["band_key"],
            "modality_agreement": fused["agreement"],
            "dispersion": fused["dispersion"],
            "interpretation": (
                "This prototype combines independently trained biomedical-signal branches at decision level. "
                "The index is a research screening signal, not a calibrated probability of Parkinson's disease."
            ),
            "disclaimer": (
                "Research prototype only. Current benchmark cohorts are independent across modalities; "
                "external paired-cohort clinical validation is required before clinical use."
            ),
        },
        "branch_details": details,
        "fusion_weights": fused["normalized_weights"],
        "fusion_note": fused["note"],
    })


@app.route("/api/multimodal/demo/<kind>", methods=["POST"])
def api_multimodal_demo(kind):
    label = 1 if kind.lower() in {"pd", "parkinsons", "positive", "high"} else 0
    request_payload = request.get_json(silent=True) or {}

    allowed_modalities = ("voice", "gait", "handwriting", "eeg")
    requested = request_payload.get("modalities")
    if isinstance(requested, list):
        selected = [m for m in allowed_modalities if m in requested]
    else:
        selected = list(allowed_modalities)

    if not selected:
        return jsonify({"error": "Select at least one modality."}), 400

    if "voice" in selected and not STATE["evaluation_results"]:
        return jsonify({"error": "Initialize the voice QML engine before using the voice modality."}), 409

    loader = get_loader()
    engine = get_multimodal_engine()

    scores = {}
    aucs = {}
    details = {}

    if "voice" in selected:
        voice_features = loader.get_demo_sample(label)
        vr = voice_branch_score(voice_features)
        voice_auc = engine.branch_status().get("voice", {}).get("roc_auc")
        scores["voice"] = vr["score"]
        aucs["voice"] = voice_auc
        details["voice"] = {
            "score_percent": round(vr["score"] * 100.0, 1),
            "model_name": "RF + VQC + Fidelity-Kernel QSVM",
            "components": {k: round(v * 100.0, 1) for k, v in vr["components"].items()},
            "roc_auc": voice_auc,
            "sample_meta": {"source": "held-out voice cohort example"},
        }

    for modality in ("gait", "handwriting", "eeg"):
        if modality not in selected:
            continue
        sample = engine.get_sample_features(modality, label)
        result = engine.predict_branch(modality, sample["features"])
        scores[modality] = result["score"]
        aucs[modality] = result.get("roc_auc")
        details[modality] = {
            **result,
            "score_percent": round(result["score"] * 100.0, 1),
            "sample_meta": sample.get("meta", {}),
        }

    fused = engine.fuse(scores, aucs)
    assessment_id = "QM-MM-" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4].upper()

    return jsonify({
        "meta": {
            "assessment_id": assessment_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "modalities_used": list(scores.keys()),
            "modalities_available": list(allowed_modalities),
            "fusion_type": fused["method"],
        },
        "known_label": label,
        "known_label_text": "Parkinson's cohort" if label == 1 else "Control cohort",
        "bundle_scope": (
            "Cross-cohort demonstration bundle; selected modality examples are label-matched "
            "but are not measurements from the same individual."
        ),
        "assessment": {
            "multimodal_pattern_index": fused["index_percent"],
            "risk_band": fused["band"],
            "risk_band_key": fused["band_key"],
            "modality_agreement": fused["agreement"],
            "dispersion": fused["dispersion"],
            "interpretation": (
                "The pattern index combines the selected independently validated signal branches. "
                "Review branch agreement and contribution alongside the overall index."
            ),
            "disclaimer": (
                "Research prototype only. The demonstration uses independent research cohorts and "
                "does not represent a clinical diagnosis or calibrated disease probability."
            ),
        },
        "branch_details": details,
        "fusion_weights": fused["normalized_weights"],
        "fusion_note": fused["note"],
    })


if __name__ == "__main__":
    # Load the dataset quickly so the dashboard can show its metadata; training remains explicit.
    try:
        if os.path.exists(DATA_FILE):
            get_loader()
    except Exception as exc:
        STATE["last_error"] = str(exc)
    app.run(host="127.0.0.1", port=5000, debug=False)
