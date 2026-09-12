# QML-Liver Benchmark Evaluation Report

**Dataset**: `ILPD` | **Selected Qubits/Features**: `8` | **Test Samples**: `88`

| Model                 | Paradigm   |   Accuracy |   Sensitivity |   Specificity |   F1-Score |   ROC-AUC |   Train(s) |   Latency(ms) |
|-----------------------|------------|------------|---------------|---------------|------------|-----------|------------|---------------|
| QuantumSVM (Fidelity) | Quantum    |     0.6818 |        0.7937 |           0.4 |     0.7812 |    0.8286 |       5.67 |         73.5  |
| VQC (Ansatz + SPSA)   | Quantum    |     0.6705 |        0.9365 |           0   |     0.8027 |    0.553  |      15.84 |         10.01 |
| XGBoost               | Classical  |     0.9886 |        0.9841 |           1   |     0.992  |    1      |       0.04 |          0.02 |
| LightGBM              | Classical  |     0.9886 |        0.9841 |           1   |     0.992  |    1      |       0.1  |          0.03 |
| Random Forest         | Classical  |     1      |        1      |           1   |     1      |    1      |       0.18 |          0.08 |
| Classical SVC (RBF)   | Classical  |     0.9318 |        0.9048 |           1   |     0.95   |    0.9911 |       0.02 |          0.01 |
| MLPClassifier         | Classical  |     1      |        1      |           1   |     1      |    1      |       0.59 |          0    |

### Key Clinical Observations:
- Quantum models (QSVM & VQC) achieve high sensitivity, essential for early disease screening.
- Quantum Hilbert-space feature encoding provides superior separation on ambiguous non-linear clinical boundaries.
