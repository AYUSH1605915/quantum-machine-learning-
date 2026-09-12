# QuantumMed: Hybrid Quantum Machine Learning Platform for Early Disease Detection

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![PennyLane](https://img.shields.io/badge/QML-PennyLane%200.45-blueviolet.svg)](https://pennylane.ai)
[![Scikit-Learn](https://img.shields.io/badge/Classical%20ML-Scikit--Learn-orange.svg)](https://scikit-learn.org)
[![Status](https://img.shields.io/badge/Status-Fully%20Functional-brightgreen.svg)]()

## 🌟 Executive Summary

Early and accurate diagnosis of complex diseases (such as metabolic fatty liver disease, early-stage oncology, and cardiovascular disorders) is crucial for improving survival rates and reducing healthcare costs. While classical machine learning algorithms have achieved significant diagnostic milestones, they often struggle to model intricate, multi-gene, and non-linear biomarker correlations in high-dimensional, noisy biomedical datasets.

**QuantumMed** is an end-to-end **Hybrid Quantum-Classical Machine Learning (HQML)** software platform designed to overcome these challenges by combining:
1. **Classical Data Pipelines & Preprocessing**: Missing value imputation, robust standardization, and high-dimensional clinical feature selection (SelectKBest ANOVA / PCA).
2. **Quantum-Enhanced Learning Models**:
   - **Variational Quantum Classifier (VQC)**: Multi-layer Parameterized Quantum Circuits (PQC) with Angle Feature Embedding, rotational variational gates, and circular CNOT entanglement rings.
   - **Quantum Support Vector Machine (QSVM)**: High-dimensional quantum Hilbert-space fidelity kernel estimation.
3. **Classical Baselines**: Rigorous benchmarking against Classical Support Vector Machines (RBF), Random Forest, Logistic Regression, and Multi-Layer Perceptron (MLP) neural networks.
4. **Transparent Explainability Engine**: ANOVA F-score saliency, permutation importance rankings, and quantum circuit topology visualization.
5. **Interactive Full-Stack Web Dashboard & Decision Support**: Real-time diagnostic inference, live quantum circuit diagrams, radar charts, ROC curves, and confusion matrix heatmaps.

---

## 🔬 System Architecture

```
                    ┌─────────────────────────────────────────────────────────┐
                    │               Biomedical Data Source                   │
                    │               (inputdata.txt / Patients)                │
                    └───────────────────────────┬─────────────────────────────┘
                                                │
                                                ▼
                    ┌─────────────────────────────────────────────────────────┐
                    │       Classical Preprocessing & Feature Engineering      │
                    │   - Missing Value Imputation (Median)                   │
                    │   - Robust / Standard Scaling                           │
                    │   - High-Dimensional Reduction: SelectKBest (ANOVA)     │
                    └───────────────────────────┬─────────────────────────────┘
                                                │
                       ┌────────────────────────┴────────────────────────┐
                       ▼                                                 ▼
        ┌─────────────────────────────┐                  ┌─────────────────────────────┐
        │   Classical Baselines       │                  │  Quantum-Enhanced Learning   │
        │ - Logistic Regression       │                  │ - Variational Quantum       │
        │ - Classical SVM (RBF)       │                  │   Classifier (VQC / PQC)    │
        │ - Random Forest             │                  │ - Quantum Kernel (QSVM)     │
        │ - Multilayer Perceptron     │                  │ - Angle / Phase Embedding   │
        └──────────────┬──────────────┘                  │ - Entangling CNOT Rings     │
                       │                                 └──────────────┬──────────────┘
                       │                                                │
                       └────────────────────────┬───────────────────────┘
                                                │
                                                ▼
                    ┌─────────────────────────────────────────────────────────┐
                    │           Benchmarking & Evaluation Engine              │
                    │ - Accuracy, Sensitivity (Recall), Specificity, F1-Score │
                    │ - ROC-AUC Curves, Confusion Matrices, Latency Analysis │
                    │ - Explainability: Permutation Importance, Circuit Maps  │
                    └───────────────────────────┬─────────────────────────────┘
                                                │
                                                ▼
                    ┌─────────────────────────────────────────────────────────┐
                    │   Modern Interactive Platform Web Dashboard (UI)        │
                    │ - Quantum Circuit Live Visualizer                       │
                    │ - Data Ingestion & Live EDA Explorer                    │
                    │ - Quantum vs Classical Comparison & Radar Charts        │
                    │ - Live Single-Patient Clinical Diagnosis Predictor      │
                    └─────────────────────────────────────────────────────────┘
```

---

## ⚛️ Quantum Mathematical Formulations

### 1. Quantum State Preparation (Angle Feature Embedding)
Classical biomedical features $x = (x_0, x_1, \dots, x_{N-1}) \in [0, \pi]^N$ are encoded into quantum state vectors using single-qubit rotations:
$$|\psi(x)\rangle = \bigotimes_{i=0}^{N-1} R_y(x_i) H |0\rangle$$
where $H$ creates an equal superposition and $R_y(x_i) = \exp\left(-i \frac{x_i}{2} \sigma_y\right)$.

### 2. Parameterized Quantum Circuit (PQC)
The state evolves under $L$ variational layers composed of rotational unitary operations and entangling gates:
$$U(\theta) = \prod_{l=1}^L \left( U_{\text{entangle}} \bigotimes_{i=0}^{N-1} R_z(\omega_{l,i}) R_y(\theta_{l,i}) \right)$$
where $U_{\text{entangle}} = \prod_{i=0}^{N-1} \text{CNOT}_{(i, (i+1) \pmod N)}$ establishes quantum entanglement across the qubits.

### 3. Readout & Classical Optimization Loop
The measurement operator evaluates the Pauli-Z expectation value:
$$\hat{y}(x; \theta, b) = \sigma\left(2 \cdot \left(\langle \psi(x)| U^\dagger(\theta) \sigma_z^{(0)} U(\theta) |\psi(x)\rangle + b\right)\right)$$
Trainable parameters $\theta, b$ are optimized via the Adam optimizer using parameter-shift gradient evaluations:
$$\frac{\partial \langle \sigma_z \rangle}{\partial \theta_j} = \frac{1}{2} \left[ \langle \sigma_z \rangle_{\theta_j + \frac{\pi}{2}} - \langle \sigma_z \rangle_{\theta_j - \frac{\pi}{2}} \right]$$

### 4. Quantum Kernel Estimation (QSVM)
Computes the inner-product fidelity in Hilbert space:
$$\kappa(x, x') = |\langle \psi(x) | \psi(x') \rangle|^2 = \left| \langle 0^{\otimes N} | U^\dagger(x') U(x) | 0^{\otimes N} \rangle \right|^2$$
The resulting Gram matrix $K_{ij} = \kappa(x_i, x_j)$ is passed to a maximum-margin dual SVM optimizer.

---

## 📂 Dataset Specification (`inputdata.txt`)

The platform reads patient clinical data from `inputdata.txt`. The dataset contains multi-variate laboratory records:

| Feature Name | Clinical Significance | Normal Range |
|---|---|---|
| `Age` | Patient chronological age | 18 - 85 yrs |
| `Gender` | Biological sex (1 = Male, 0 = Female) | 0 / 1 |
| `BMI` | Body Mass Index | 18.5 - 24.9 kg/m² |
| `Total_Bilirubin` | Biliary excretion / breakdown | 0.2 - 1.2 mg/dL |
| `Direct_Bilirubin` | Conjugated bilirubin index | 0.0 - 0.3 mg/dL |
| `Alkaline_Phosphatase` | Biliary duct enzyme biomarker | 44 - 147 IU/L |
| `Alamine_Aminotransferase (ALT)` | Hepatic cellular integrity indicator | 7 - 56 IU/L |
| `Aspartate_Aminotransferase (AST)` | Mitochondrial hepatic enzyme | 10 - 40 IU/L |
| `Total_Proteins` | Circulating serum protein pool | 6.0 - 8.3 g/dL |
| `Albumin` | Primary hepatic synthesis protein | 3.5 - 5.0 g/dL |
| `Albumin_and_Globulin_Ratio` | Immune-synthesis biomarker balance | 1.0 - 2.5 |
| `Fasting_Glucose` | Glycemic homeostasis indicator | 70 - 99 mg/dL |
| `Serum_Cholesterol` | Lipid panel cardiovascular marker | < 200 mg/dL |
| `Platelet_Count` | Thrombocyte count | 150 - 450 x10³/µL |
| **`Diagnosis`** | **Target Label (1 = Early Disease Detected, 0 = Healthy)** | **0 / 1** |

> **Custom Data**: You can edit or replace `inputdata.txt` with your own CSV/TXT formatted biomedical records. The platform automatically detects headers, imputes missing records, and normalizes columns.

---

## 🚀 How to Run

### 1. Command-Line Benchmark (CLI)
Run end-to-end data loading, classical model training, PennyLane quantum circuit execution, and benchmark generation:
```powershell
.\python_env\python.exe main.py --data inputdata.txt --qubits 4 --epochs 15
```
Output results and high-resolution comparison plots are saved to the `results/` folder:
- `results/benchmark_report.json`
- `results/benchmark_metrics.png`
- `results/roc_curves.png`

### 2. Interactive Web Platform Dashboard
Launch the web interface:
```powershell
.\python_env\python.exe app.py
```
Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 📊 Evaluation & Benchmarking Metrics

The platform evaluates models on:
1. **Accuracy**: Overall correct diagnosis rate $\frac{TP + TN}{TP + TN + FP + FN}$.
2. **Sensitivity (Recall)**: Critical early detection metric measuring the proportion of diseased individuals correctly identified $\frac{TP}{TP + FN}$.
3. **Specificity**: True negative rate measuring healthy individuals correctly identified $\frac{TN}{TN + FP}$.
4. **Precision**: Positive predictive value $\frac{TP}{TP + FP}$.
5. **F1-Score**: Harmonic mean of Precision and Sensitivity $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$.
6. **ROC-AUC**: Area under the Receiver Operating Characteristic curve measuring class separability.
7. **Computational Latency**: Training wall-clock time (s) and inference latency per patient (ms).

---

## 🩺 Interactive Clinical Decision Support

Under the **Clinical Risk Predictor** tab in the dashboard, medical practitioners can:
1. Enter or modify a patient's laboratory biomarker profile.
2. Select standard presets: `Load Healthy Sample` or `Load High-Risk Sample`.
3. Click **Run Hybrid Quantum Inference** to obtain:
   - Early Disease Status (`High Risk: Early Disease Detected` vs `Normal Screening`).
   - Hybrid Quantum Risk Percentage.
   - Quantum Subsystem (VQC) and Classical Subsystem (RF) individual probability scores.
   - Physical Qubit Angle Mappings ($R_y(x_i)$ in radians).
   - Clinical follow-up recommendations (e.g., confirmatory ultrasound/fibroscan).

---

## 📦 Project Structure

```
├── inputdata.txt             # Active clinical biomedical dataset
├── inputdata.txt.txt         # Mirror dataset path for Windows compatibility
├── main.py                   # Master CLI runner and benchmark generator
├── app.py                    # Flask REST API and web platform server
├── qml_platform/             # Core Quantum Machine Learning package
│   ├── __init__.py           # Package exports
│   ├── dataset_loader.py     # Ingestion, imputation, scaling, SelectKBest/PCA
│   ├── quantum_models.py     # PennyLane VQC and Quantum SVM (QSVM)
│   ├── classical_models.py   # Baseline models (SVM, RF, LogReg, MLP)
│   └── evaluation.py         # Diagnostic metrics and explainability engine
├── templates/
│   └── index.html            # Web platform interface
├── static/
│   ├── styles.css            # Modern glassmorphism quantum theme
│   └── app.js                # Dynamic visualization and API orchestration
├── results/                  # Generated benchmark reports and evaluation plots
│   ├── benchmark_report.json
│   ├── benchmark_metrics.png
│   └── roc_curves.png
└── python_env/               # Self-contained Python 3.11 QML environment
```
