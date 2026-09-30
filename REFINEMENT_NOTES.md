# ThunderStars SIH 2026 - Final UI/Report Refinement

This refinement keeps the working Parkinson's QML pipeline and improves the SIH demo presentation.

## What changed

- Clear **Input -> Hybrid Engine -> Output -> Next Action** overview.
- Added **intended users** and a deployment workflow so judges can immediately understand who uses the system and how.
- Grouped the 22 acoustic inputs into Pitch, Jitter, Shimmer, and Noise/Nonlinear categories.
- Added live input-completeness display.
- Added patient-to-cohort evidence to the result, not just global feature importance.
- Improved report metadata: assessment ID, timestamp, input source, and input coverage.
- Improved print CSS: controls and sample buttons are removed from the printed report.
- Added participant-separated validation summary cards.
- Added an **auxiliary-model validation gate**. The UPDRS estimate is automatically withheld when participant-level R-squared is not positive. With the current run (motor R2 -0.178, total R2 -0.412), the system will correctly suppress the unreliable severity numbers and explain why.
- Retains the real RF + PennyLane VQC + fidelity-kernel QSVM inference.

## Apply

Stop the Flask server with Ctrl+C, extract this package over the project root, then restart:

```powershell
.\scripts\run_demo.ps1
```

Because the models live in memory, click **Initialize QML engine** once after restart. Do this before recording the final video.

## Clean PDF tip

When using **Print / Save assessment report**, choose `Save as PDF` and turn off the browser option **Headers and footers** so the localhost URL/date header is not printed.
