import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json
import time
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import pandas as pd

from qml_platform.dataset_loader import BiomedicalDatasetLoader
from qml_platform.quantum_models import VariationalQuantumClassifier, QuantumSVM
from qml_platform.classical_models import ClassicalBaselines
from qml_platform.evaluation import ModelEvaluator, ExplainabilityEngine
from qml_platform.report_parser import MedicalReportParser

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# Global in-memory cache for trained models and state
PLATFORM_STATE = {
    "loader": None,
    "classical": None,
    "vqc": None,
    "qsvm": None,
    "evaluation_results": [],
    "circuit_ascii": "",
    "circuit_schema": {},
    "feature_importance": [],
    "last_trained": None,
    "is_training": False
}

def get_or_create_loader(n_qubits=4):
    """Retrieves or instantly initializes the dataset loader without training models."""
    if PLATFORM_STATE["loader"] is None:
        data_file = "inputdata.txt" if os.path.exists("inputdata.txt") else "inputdata.txt.txt"
        loader = BiomedicalDatasetLoader(filepath=data_file, n_qubits=n_qubits)
        loader.load_data()
        PLATFORM_STATE["loader"] = loader
    return PLATFORM_STATE["loader"]

def init_default_pipeline(epochs=6, n_qubits=4):
    """Initializes and trains the pipeline once if not already trained."""
    loader = get_or_create_loader(n_qubits=n_qubits)

    classical = ClassicalBaselines()
    classical.fit_all(loader.X_train_classical, loader.y_train)

    eval_results = []
    for name, model in classical.get_all_models().items():
        res = ModelEvaluator.evaluate_model(
            name, model, loader.X_test_classical, loader.y_test, training_time=classical.training_times[name]
        )
        eval_results.append(res)

    vqc = VariationalQuantumClassifier(n_qubits=n_qubits, n_layers=2, learning_rate=0.07, epochs=epochs)
    vqc.fit(loader.X_train_quantum, loader.y_train)
    res_vqc = ModelEvaluator.evaluate_model("Quantum VQC (PQC)", vqc, loader.X_test_quantum, loader.y_test, training_time=vqc.training_time)
    eval_results.append(res_vqc)

    qsvm = QuantumSVM(n_qubits=n_qubits, C=1.0)
    qsvm.fit(loader.X_train_quantum, loader.y_train)
    res_qsvm = ModelEvaluator.evaluate_model("Quantum SVM (QSVM Kernel)", qsvm, loader.X_test_quantum, loader.y_test, training_time=qsvm.training_time)
    eval_results.append(res_qsvm)

    rf_model = classical.get_model("Random Forest")
    importances = ExplainabilityEngine.get_feature_importances(loader, rf_model)
    circuit_schema = ExplainabilityEngine.get_circuit_schema(n_qubits=n_qubits, n_layers=2)
    circuit_ascii = vqc.get_circuit_ascii()

    PLATFORM_STATE["loader"] = loader
    PLATFORM_STATE["classical"] = classical
    PLATFORM_STATE["vqc"] = vqc
    PLATFORM_STATE["qsvm"] = qsvm
    PLATFORM_STATE["evaluation_results"] = eval_results
    PLATFORM_STATE["circuit_ascii"] = circuit_ascii
    PLATFORM_STATE["circuit_schema"] = circuit_schema
    PLATFORM_STATE["feature_importance"] = importances
    PLATFORM_STATE["last_trained"] = time.strftime("%Y-%m-%d %H:%M:%S")

    # Persist results
    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "benchmark_report.json"), "w") as f:
        json.dump({
            "dataset_summary": loader.get_dataset_summary(),
            "models_evaluation": eval_results,
            "feature_importance": importances,
            "circuit_ascii": circuit_ascii
        }, f, indent=2)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/status")
def api_status():
    return jsonify({
        "status": "ready" if PLATFORM_STATE["loader"] is not None else "uninitialized",
        "last_trained": PLATFORM_STATE["last_trained"],
        "is_training": PLATFORM_STATE["is_training"]
    })

@app.route("/api/dataset")
def api_dataset():
    data_file = "inputdata.txt" if os.path.exists("inputdata.txt") else "inputdata.txt.txt"
    if not os.path.exists(data_file):
        return jsonify({"error": "Dataset file not found"}), 404

    df = pd.read_csv(data_file)
    head_data = df.head(15).to_dict(orient="records")
    columns = list(df.columns)
    summary = {
        "filename": data_file,
        "rows": len(df),
        "columns": columns,
        "class_balance": df['Diagnosis'].value_counts().to_dict() if 'Diagnosis' in df.columns else {},
        "preview": head_data,
        "selected_quantum_features": PLATFORM_STATE["loader"].selected_feature_names if PLATFORM_STATE["loader"] else columns[:4]
    }
    return jsonify(summary)

@app.route("/api/benchmark")
def api_benchmark():
    # If not yet trained, load from file if exists or initialize
    if not PLATFORM_STATE["evaluation_results"]:
        report_file = os.path.join("results", "benchmark_report.json")
        if os.path.exists(report_file):
            with open(report_file, "r") as f:
                data = json.load(f)
                if "circuit_schema" not in data:
                    data["circuit_schema"] = ExplainabilityEngine.get_circuit_schema(n_qubits=4, n_layers=2)
                return jsonify(data)
        else:
            init_default_pipeline(epochs=10, n_qubits=4)

    return jsonify({
        "dataset_summary": PLATFORM_STATE["loader"].get_dataset_summary() if PLATFORM_STATE["loader"] else {},
        "models_evaluation": PLATFORM_STATE["evaluation_results"],
        "feature_importance": PLATFORM_STATE["feature_importance"],
        "circuit_ascii": PLATFORM_STATE["circuit_ascii"],
        "circuit_schema": PLATFORM_STATE["circuit_schema"]
    })

@app.route("/api/train", methods=["POST"])
def api_train():
    req = request.get_json() or {}
    epochs = int(req.get("epochs", 15))
    n_qubits = int(req.get("n_qubits", 4))

    PLATFORM_STATE["is_training"] = True
    try:
        init_default_pipeline(epochs=epochs, n_qubits=n_qubits)
    finally:
        PLATFORM_STATE["is_training"] = False

    return jsonify({
        "status": "success",
        "message": f"Trained hybrid quantum-classical architecture on {n_qubits} qubits with {epochs} VQC epochs",
        "results": PLATFORM_STATE["evaluation_results"]
    })

@app.route("/api/parse-text", methods=["POST"])
def api_parse_text():
    """Extract clinical biomarkers from user-entered or pasted medical report text."""
    data = request.get_json(silent=True) or {}
    raw_text = data.get("text", "")
    if not raw_text:
        return jsonify({"error": "No text provided"}), 400

    parsed = MedicalReportParser.parse_text(raw_text)
    return jsonify(parsed)

@app.route("/api/parse-image", methods=["POST"])
def api_parse_image():
    """Performs OCR and biomarker extraction on an uploaded lab report image."""
    if "image" not in request.files and "file" not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    file = request.files.get("image") or request.files.get("file")
    image_bytes = file.read()
    if not image_bytes:
        return jsonify({"error": "Empty image file"}), 400

    parsed = MedicalReportParser.parse_image_bytes(image_bytes)
    return jsonify(parsed)

@app.route("/api/cohort-stats", methods=["GET"])
def api_cohort_stats():
    """Returns baseline healthy vs diseased cohort statistical benchmarks from inputdata.txt."""
    loader = get_or_create_loader()
    stats = loader.get_cohort_statistics()
    return jsonify(stats)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Predict early disease diagnosis for a single patient record and compare with dataset cohort."""
    if PLATFORM_STATE["loader"] is None or PLATFORM_STATE["vqc"] is None:
        init_default_pipeline(epochs=10, n_qubits=4)

    patient_data = request.get_json(silent=True, force=True) or request.form.to_dict() or {}
    loader = PLATFORM_STATE["loader"]
    vqc = PLATFORM_STATE["vqc"]
    rf = PLATFORM_STATE["classical"].get_model("Random Forest")

    # Transform patient data
    x_classical, x_quantum = loader.transform_single_patient(patient_data)

    # Quantum VQC Prediction
    vqc_prob = vqc.predict_proba(x_quantum)[0]
    vqc_pred = int(vqc.predict(x_quantum)[0])

    # Classical RF Prediction
    rf_prob = rf.predict_proba(x_classical)[0]
    rf_pred = int(rf.predict(x_classical)[0])

    # Hybrid Ensemble Risk Score (weighted quantum + classical)
    hybrid_prob = float(0.60 * vqc_prob[1] + 0.40 * rf_prob[1])
    hybrid_pred = 1 if hybrid_prob >= 0.50 else 0

    # Compare patient vitals with dataset cohort distributions
    cohort_comparison = loader.compare_patient_with_cohort(patient_data)

    # Generate biomarker-specific clinical guidance
    elevated_biomarkers = [
        item["display_name"] for item in cohort_comparison["comparisons"]
        if item["status_level"] in ["danger", "warning"] and item["biomarker"] != "Gender"
    ]
    if hybrid_pred == 1:
        if elevated_biomarkers:
            guidance = (
                f"Elevated biomarkers detected: {', '.join(elevated_biomarkers[:4])}. "
                "Hybrid quantum risk indicates early-stage cellular or metabolic disease pattern. "
                "Recommended follow-up: Confirmatory hepatic ultrasound/fibroscan, comprehensive metabolic panel (CMP), "
                "and consultation with a specialist."
            )
        else:
            guidance = (
                "Hybrid quantum risk indicates subtle multi-biomarker correlation indicative of early disease pattern. "
                "Recommend follow-up clinical screening and repeat blood panel in 4-6 weeks."
            )
    else:
        if elevated_biomarkers:
            guidance = (
                f"Overall hybrid quantum risk is low/healthy, but note mild elevation in: {', '.join(elevated_biomarkers[:3])}. "
                "Routine annual screening and healthy lifestyle/dietary maintenance advised."
            )
        else:
            guidance = (
                "All primary biomedical biomarkers align closely with the dataset healthy cohort baseline. "
                "Normal metabolic and hepatic health profile detected. Routine preventive checkups recommended."
            )

    return jsonify({
        "patient_inputs": patient_data,
        "hybrid_prediction": {
            "disease_detected": bool(hybrid_pred == 1),
            "disease_risk_percentage": round(hybrid_prob * 100, 1),
            "status_label": "High Risk: Early Disease Detected" if hybrid_pred == 1 else "Low Risk: Normal / Healthy",
            "confidence": round(max(hybrid_prob, 1.0 - hybrid_prob) * 100, 1)
        },
        "vqc_model": {
            "prediction": vqc_pred,
            "probability_disease": round(float(vqc_prob[1]) * 100, 1)
        },
        "classical_rf_model": {
            "prediction": rf_pred,
            "probability_disease": round(float(rf_prob[1]) * 100, 1)
        },
        "quantum_encoded_features": {
            name: round(float(x_quantum[0][i]), 4) for i, name in enumerate(loader.selected_feature_names)
        },
        "cohort_comparison": cohort_comparison,
        "clinical_guidance": guidance
    })

@app.route("/results/<path:filename>")
def serve_results(filename):
    return send_from_directory("results", filename)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
