import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from qml_platform.dataset_loader import BiomedicalDatasetLoader
from qml_platform.quantum_models import VariationalQuantumClassifier, QuantumSVM
from qml_platform.classical_models import ClassicalBaselines
from qml_platform.evaluation import ModelEvaluator, ExplainabilityEngine

def print_header(title):
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)

def plot_benchmarks(eval_results, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Bar chart comparing metrics
    model_names = [res["model_name"] for res in eval_results]
    metrics = ["accuracy", "sensitivity", "specificity", "f1_score", "roc_auc"]
    
    x = np.arange(len(model_names))
    width = 0.15
    
    plt.figure(figsize=(12, 6))
    for i, metric in enumerate(metrics):
        values = [res[metric] for res in eval_results]
        plt.bar(x + i * width, values, width, label=metric.replace("_", " ").title())
        
    plt.xlabel("Models", fontweight="bold")
    plt.ylabel("Score", fontweight="bold")
    plt.title("Hybrid Quantum vs Classical Early Disease Detection Benchmark", fontweight="bold", fontsize=14)
    plt.xticks(x + width * 2, model_names, rotation=15, ha='right')
    plt.ylim(0.0, 1.1)
    plt.legend(loc="lower right")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "benchmark_metrics.png"), dpi=300)
    plt.close()

    # 2. ROC Curves
    plt.figure(figsize=(8, 6))
    for res in eval_results:
        roc = res.get("roc_curve", {})
        if "fpr" in roc and "tpr" in roc:
            plt.plot(roc["fpr"], roc["tpr"], label=f'{res["model_name"]} (AUC = {res["roc_auc"]:.2f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.6, label="Random Chance")
    plt.xlabel("False Positive Rate (1 - Specificity)", fontweight="bold")
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontweight="bold")
    plt.title("Receiver Operating Characteristic (ROC) Curves", fontweight="bold", fontsize=13)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curves.png"), dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Hybrid Quantum Machine Learning Platform for Early Disease Detection")
    parser.add_argument("--data", type=str, default="inputdata.txt", help="Path to input data file")
    parser.add_argument("--qubits", type=int, default=4, help="Number of qubits for quantum simulator")
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs for VQC")
    parser.add_argument("--output", type=str, default="results", help="Directory to store results and plots")
    args = parser.parse_args()

    print_header("HYBRID QUANTUM-CLASSICAL MACHINE LEARNING PLATFORM")
    print(f" [*] Biomedical Dataset: {args.data}")
    print(f" [*] Target Quantum Qubits: {args.qubits}")
    print(f" [*] VQC Optimizer Epochs: {args.epochs}")

    # Step 1: Ingestion and Preprocessing
    print("\n[Phase 1] Data Ingestion and Classical Feature Engineering...")
    loader = BiomedicalDatasetLoader(filepath=args.data, n_qubits=args.qubits)
    loader.load_data()
    summary = loader.get_dataset_summary()
    print(f"  -> Samples Loaded: {summary['total_samples']} (Train: {summary['train_samples']}, Test: {summary['test_samples']})")
    print(f"  -> Total Biomarker Features: {summary['total_features']}")
    print(f"  -> Selected Quantum Qubit Features: {', '.join(summary['selected_features'])}")
    print(f"  -> Class Distribution: Healthy/Benign: {summary['class_distribution']['negative (0)']}, Disease Detected: {summary['class_distribution']['positive (1)']}")

    eval_results = []

    # Step 2: Classical Baselines
    print("\n[Phase 2] Training Classical Baseline Models...")
    classical = ClassicalBaselines()
    classical.fit_all(loader.X_train_classical, loader.y_train)

    for name, model in classical.get_all_models().items():
        res = ModelEvaluator.evaluate_model(
            name, model, loader.X_test_classical, loader.y_test, training_time=classical.training_times[name]
        )
        eval_results.append(res)
        print(f"  -> {name:<28} | Accuracy: {res['accuracy']:.4f} | Sensitivity: {res['sensitivity']:.4f} | Specificity: {res['specificity']:.4f} | ROC-AUC: {res['roc_auc']:.4f}")

    # Step 3: Quantum Machine Learning Models
    print("\n[Phase 3] Executing Quantum-Enhanced Machine Learning Models...")
    
    # 3A. Variational Quantum Classifier (VQC)
    print("  -> Training Variational Quantum Classifier (VQC) with Parameterized Quantum Circuit...")
    vqc = VariationalQuantumClassifier(n_qubits=args.qubits, n_layers=2, learning_rate=0.07, epochs=args.epochs)
    vqc.fit(loader.X_train_quantum, loader.y_train)
    res_vqc = ModelEvaluator.evaluate_model("Quantum VQC (PQC)", vqc, loader.X_test_quantum, loader.y_test, training_time=vqc.training_time)
    eval_results.append(res_vqc)
    print(f"  -> {'Quantum VQC (PQC)':<28} | Accuracy: {res_vqc['accuracy']:.4f} | Sensitivity: {res_vqc['sensitivity']:.4f} | Specificity: {res_vqc['specificity']:.4f} | ROC-AUC: {res_vqc['roc_auc']:.4f}")

    # 3B. Quantum Support Vector Machine (QSVM / Quantum Kernel)
    print("  -> Evaluating Quantum Kernel Fidelity Matrix & Training QSVM...")
    qsvm = QuantumSVM(n_qubits=args.qubits, C=1.0)
    qsvm.fit(loader.X_train_quantum, loader.y_train)
    res_qsvm = ModelEvaluator.evaluate_model("Quantum SVM (QSVM Kernel)", qsvm, loader.X_test_quantum, loader.y_test, training_time=qsvm.training_time)
    eval_results.append(res_qsvm)
    print(f"  -> {'Quantum SVM (QSVM Kernel)':<28} | Accuracy: {res_qsvm['accuracy']:.4f} | Sensitivity: {res_qsvm['sensitivity']:.4f} | Specificity: {res_qsvm['specificity']:.4f} | ROC-AUC: {res_qsvm['roc_auc']:.4f}")

    # Step 4: Explainability Module
    print("\n[Phase 4] Model Explainability & Interpretability Analysis...")
    rf_model = classical.get_model("Random Forest")
    importances = ExplainabilityEngine.get_feature_importances(loader, rf_model)
    print(f"  -> Top Clinical Biomarkers Identified:")
    for imp in importances[:5]:
        q_tag = "[Quantum Encoded]" if imp['is_quantum_encoded'] else ""
        print(f"     * {imp['feature']:<28}: Normalized Importance = {imp['normalized_importance']:.4f} {q_tag}")

    # Step 5: Summary Table
    print_header("FINAL BENCHMARK COMPARISON TABLE")
    header_str = f"{'Model':<26} | {'Accuracy':<8} | {'Sens/Recall':<11} | {'Specificity':<11} | {'Precision':<9} | {'F1-Score':<8} | {'ROC-AUC':<8} | {'Train(s)':<8}"
    print(header_str)
    print("-" * len(header_str))
    for res in eval_results:
        print(f"{res['model_name']:<26} | {res['accuracy']:<8.4f} | {res['sensitivity']:<11.4f} | {res['specificity']:<11.4f} | {res['precision']:<9.4f} | {res['f1_score']:<8.4f} | {res['roc_auc']:<8.4f} | {res['training_time_sec']:<8.2f}")

    # Step 6: Save Artifacts
    os.makedirs(args.output, exist_ok=True)
    report_path = os.path.join(args.output, "benchmark_report.json")
    with open(report_path, "w") as f:
        json.dump({
            "dataset_summary": summary,
            "models_evaluation": eval_results,
            "feature_importance": importances,
            "circuit_ascii": vqc.get_circuit_ascii()
        }, f, indent=2)
    print(f"\n[+] Saved detailed evaluation JSON to: {report_path}")

    plot_benchmarks(eval_results, args.output)
    print(f"[+] Saved high-resolution comparison plots to: {args.output}/")

    print("\n[SUCCESS] Hybrid Quantum-Classical Early Disease Detection Execution Completed Successfully!\n")

if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    main()
