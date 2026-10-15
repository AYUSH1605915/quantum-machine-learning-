# ThunderStars SIH 2026 — Parkinson's QML Quickstart

This drop-in upgrade keeps the existing Flask/PennyLane/classical-ML core but changes the active disease demonstration to Parkinson's voice-pattern screening.

## 1. Back up the current project
Copy the existing project folder before replacing files.

## 2. Copy this upgrade over the project root
Replace `app.py`, `templates/index.html`, `static/app.js`, `static/styles.css`, and `qml_platform/dataset_loader.py`. The package also includes the existing classical/QML/evaluation modules for convenience.

## 3. Extract only the tiny datasets needed from the 6.97 GB archive
From the project root:

```powershell
.\scripts\prepare_parkinsons_data.ps1 -ZipPath "C:\Users\kv022\Downloads\ParkinsonsData.zip"
```

This extracts only the UCI classification file and the UPDRS telemonitoring file. It does not unpack the full archive.

## 4. Use Python 3.12, not the current 3.14 environment
If Python 3.12 is missing:

```powershell
winget install -e --id Python.Python.3.12 --scope user
```

Close and reopen PowerShell, then:

```powershell
.\scripts\setup_demo.ps1
```

## 5. Run

```powershell
.\scripts\run_demo.ps1
```

Open http://127.0.0.1:5000, click **Initialize QML engine**, then load a held-out control or Parkinson's-cohort sample and run inference.

## What the result means
The UI deliberately reports a **Hybrid Risk Index** / **Parkinsonian Pattern Risk** rather than a diagnosis. The three model scores are model outputs, and their arithmetic mean is used as an ensemble index. It is explicitly not presented as a calibrated clinical probability.

## Validation change
The loader derives participant IDs from the UCI `name` field and uses a participant-grouped stratified split where possible, reducing leakage from repeated voice recordings by the same person.
