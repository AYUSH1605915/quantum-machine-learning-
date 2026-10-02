
THUNDERSTARS — EEG BRANCH V1
=============================

Purpose
-------
Train a real resting-state EEG Parkinson's-vs-control branch from ds007526.

Inputs expected
---------------
data\multimodal\EEG\ds007526\participants.tsv
data\multimodal\EEG\ds007526\sub-XXX\eeg\sub-XXX_task-rest_eeg.set

The classifier does NOT use:
- age
- sex
- UPDRS
- MoCA
- disease duration
- LEDD
- PIGD/TD scores
- other clinical metadata

It uses EEG-derived spectral measurements only.

Method
------
1. Read one resting-state .set recording per participant with MNE.
2. Keep EEG channels only.
3. Trim 5 seconds from each end.
4. Limit extreme transient artifacts with per-channel robust MAD clipping.
5. Estimate robust Welch spectral density.
6. Build relative delta/theta/alpha/beta/gamma band-power features.
7. Add global spectral summaries and band ratios.
8. Put SelectKBest INSIDE the model pipeline to prevent feature-selection leakage.
9. Evaluate Logistic Regression, calibrated RBF-SVM and Random Forest with
   5-fold stratified participant-level cross-validation.
10. Select the model with the highest cross-validated ROC-AUC and fit it on
    all available participants for application inference.

Run
---
.\.venv\Scripts\python.exe .\scripts\train_eeg_branch.py

The first run computes features from the raw EEG files and caches:
results\eeg_feature_table.csv

A later run reuses that cache.

Force feature recomputation:
.\.venv\Scripts\python.exe .\scripts\train_eeg_branch.py --rebuild

Outputs
-------
models\eeg_branch.joblib
results\eeg_benchmark_report.json
results\eeg_feature_table.csv
results\eeg_feature_failures.json

Important
---------
The cohort is imbalanced (more PD than HC). Do not report accuracy alone.
Use ROC-AUC together with sensitivity and specificity.

These are research-cohort cross-validation results, not clinical validation.
