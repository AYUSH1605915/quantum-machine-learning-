# QuantumMed Multimodal Fusion Patch v1

This patch adds a real decision-level fusion layer across the three currently trained branches:

- Voice: existing Random Forest + VQC + fidelity-kernel QSVM hybrid score
- Gait: saved `models/gait_branch.joblib`
- Handwriting: saved `models/handwriting_branch.joblib`

## Why late fusion

The voice, gait, and HandPD datasets contain different research participants. The patch therefore does **not** train a false joint classifier across unmatched people. Each branch is trained/validated independently and only their prediction scores are fused.

The current prototype uses validation-weighted late fusion. Research ROC-AUC above chance supplies a demonstration reliability weight; weights are renormalized across available branches. These weights are not clinical reliability estimates.

## Required local files

You should already have:

- `models/gait_branch.joblib`
- `results/gait_benchmark_report.json`
- `results/gait_feature_table.csv`
- `models/handwriting_branch.joblib`
- `results/handwriting_benchmark_report.json`
- `results/handwriting_subject_features.csv`

The voice branch is initialized from the web UI as before.

## After copying the patch

Run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\run_demo.ps1
```

Open `http://127.0.0.1:5000`, hard refresh with `Ctrl+F5`, initialize the QML engine, then scroll to **Working Multimodal Prototype**.

Test both:

- Run control bundle
- Run Parkinson's bundle

The demo bundles are label-matched examples from independent cohorts, not measurements from one person. That limitation is stated directly in the UI.
