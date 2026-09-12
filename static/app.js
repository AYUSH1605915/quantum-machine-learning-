// QuantumMed Client-Side Logic & Visualizations

let radarChartInstance = null;
let rocChartInstance = null;
let sensSpecChartInstance = null;
let importanceChartInstance = null;

document.addEventListener("DOMContentLoaded", () => {
    initTabs();
    loadDatasetInfo();
    loadBenchmarkData();
    initPredictionActions();
    initReBenchmark();
});

// 1. Tab Switching
function initTabs() {
    const tabButtons = document.querySelectorAll(".tab-btn");
    const tabContents = document.querySelectorAll(".tab-content");

    tabButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            const targetId = btn.getAttribute("data-tab");
            
            tabButtons.forEach(b => b.classList.remove("active"));
            tabContents.forEach(c => c.classList.remove("active"));

            btn.classList.add("active");
            const targetEl = document.getElementById(targetId);
            if (targetEl) targetEl.classList.add("active");
        });
    });
}

// 2. Load Dataset Info
async function loadDatasetInfo() {
    try {
        const res = await fetch("/api/dataset");
        if (!res.ok) throw new Error("Failed to load dataset metadata");
        const data = await res.json();

        // Update Stats
        const rowCount = data.rows || 100;
        document.getElementById("dataset-row-count").innerText = `${rowCount} Records`;
        document.getElementById("dataset-feature-count").innerText = `${data.columns.length - 1} Biomarkers`;
        document.getElementById("active-file-name").innerText = data.filename;

        const posCount = (data.class_balance && data.class_balance[1]) || 55;
        const negCount = (data.class_balance && data.class_balance[0]) || 45;
        document.getElementById("class-pos-count").innerText = `${posCount} (${((posCount / rowCount) * 100).toFixed(1)}%)`;
        document.getElementById("class-neg-count").innerText = `${negCount} (${((negCount / rowCount) * 100).toFixed(1)}%)`;

        if (data.selected_quantum_features && data.selected_quantum_features.length > 0) {
            document.getElementById("quantum-selected-features").innerText = data.selected_quantum_features.join(", ");
        }

        // Populate Table
        const tbody = document.getElementById("dataset-tbody");
        if (data.preview && data.preview.length > 0) {
            tbody.innerHTML = "";
            data.preview.forEach((row, idx) => {
                const tr = document.createElement("tr");
                const diagVal = row.Diagnosis;
                const diagBadge = diagVal == 1 
                    ? `<span class="badge badge-warning">1 (Early Disease)</span>` 
                    : `<span class="badge badge-success">0 (Healthy)</span>`;

                tr.innerHTML = `
                    <td><strong>#${idx + 1}</strong></td>
                    <td>${row.Age || '--'}</td>
                    <td>${row.BMI || '--'}</td>
                    <td>${row.Total_Bilirubin || '--'}</td>
                    <td>${row.Direct_Bilirubin || '--'}</td>
                    <td>${row.Alkaline_Phosphatase || '--'}</td>
                    <td>${row.Alamine_Aminotransferase || '--'}</td>
                    <td>${row.Aspartate_Aminotransferase || '--'}</td>
                    <td>${row.Total_Proteins || '--'}</td>
                    <td>${row.Albumin || '--'}</td>
                    <td>${row.Albumin_and_Globulin_Ratio || '--'}</td>
                    <td>${row.Fasting_Glucose || '--'}</td>
                    <td>${row.Serum_Cholesterol || '--'}</td>
                    <td>${diagBadge}</td>
                `;
                tbody.appendChild(tr);
            });
        }
    } catch (err) {
        console.error("Dataset load error:", err);
    }
}

// 3. Load Benchmark Data & Visualizations
async function loadBenchmarkData() {
    try {
        const res = await fetch("/api/benchmark");
        if (!res.ok) throw new Error("Failed to load benchmark evaluation");
        const data = await res.json();

        const models = data.models_evaluation || [];
        renderBenchmarkTable(models);
        renderRadarChart(models);
        renderRocChart(models);
        renderSensSpecChart(models);
        renderConfusionMatrices(models);
        renderCircuit(data.circuit_schema, data.circuit_ascii);
        renderExplainability(data.feature_importance);
    } catch (err) {
        console.error("Benchmark load error:", err);
    }
}

function renderBenchmarkTable(models) {
    const tbody = document.getElementById("benchmark-tbody");
    tbody.innerHTML = "";

    models.forEach(m => {
        const tr = document.createElement("tr");
        const isQuantum = m.model_name.includes("Quantum");
        const paradigmBadge = isQuantum 
            ? `<span class="badge badge-info">Quantum-Enhanced</span>`
            : `<span class="badge" style="background: rgba(255,255,255,0.06); color:#cbd5e1;">Classical</span>`;

        tr.innerHTML = `
            <td>${paradigmBadge}</td>
            <td><strong>${m.model_name}</strong></td>
            <td><span class="code-badge">${(m.accuracy * 100).toFixed(1)}%</span></td>
            <td><strong class="text-success">${(m.sensitivity * 100).toFixed(1)}%</strong></td>
            <td>${(m.specificity * 100).toFixed(1)}%</td>
            <td>${(m.precision * 100).toFixed(1)}%</td>
            <td>${(m.f1_score * 100).toFixed(1)}%</td>
            <td><span class="code-badge">${m.roc_auc.toFixed(3)}</span></td>
            <td>${m.training_time_sec}s</td>
            <td>${m.inference_latency_ms}ms</td>
        `;
        tbody.appendChild(tr);
    });
}

// Radar Comparison Chart
function renderRadarChart(models) {
    const ctx = document.getElementById("radarChart");
    if (!ctx) return;

    if (radarChartInstance) radarChartInstance.destroy();

    const colors = [
        { border: "#00f2fe", bg: "rgba(0, 242, 254, 0.2)" },
        { border: "#a855f7", bg: "rgba(168, 85, 247, 0.2)" },
        { border: "#10b981", bg: "rgba(16, 185, 129, 0.2)" },
        { border: "#fbbf24", bg: "rgba(251, 191, 36, 0.2)" },
        { border: "#f43f5e", bg: "rgba(244, 63, 94, 0.2)" },
        { border: "#4facfe", bg: "rgba(79, 172, 254, 0.2)" }
    ];

    const datasets = models.map((m, idx) => {
        const c = colors[idx % colors.length];
        return {
            label: m.model_name,
            data: [m.accuracy, m.sensitivity, m.specificity, m.precision, m.f1_score, m.roc_auc],
            borderColor: c.border,
            backgroundColor: c.bg,
            borderWidth: 2,
            pointBackgroundColor: c.border
        };
    });

    radarChartInstance = new Chart(ctx, {
        type: "radar",
        data: {
            labels: ["Accuracy", "Sensitivity (Recall)", "Specificity", "Precision", "F1-Score", "ROC-AUC"],
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    angleLines: { color: "rgba(255, 255, 255, 0.1)" },
                    grid: { color: "rgba(255, 255, 255, 0.08)" },
                    pointLabels: { color: "#94a3b8", font: { size: 11, family: "'Plus Jakarta Sans'" } },
                    ticks: { display: false, min: 0, max: 1 }
                }
            },
            plugins: {
                legend: { position: "bottom", labels: { color: "#cbd5e1", boxWidth: 12 } }
            }
        }
    });
}

// ROC Curves Chart
function renderRocChart(models) {
    const ctx = document.getElementById("rocChart");
    if (!ctx) return;

    if (rocChartInstance) rocChartInstance.destroy();

    const colors = ["#00f2fe", "#a855f7", "#10b981", "#fbbf24", "#f43f5e", "#4facfe"];
    const datasets = [];

    models.forEach((m, idx) => {
        if (m.roc_curve && m.roc_curve.fpr && m.roc_curve.tpr) {
            const pts = m.roc_curve.fpr.map((fpr, i) => ({ x: fpr, y: m.roc_curve.tpr[i] }));
            datasets.push({
                label: `${m.model_name} (AUC: ${m.roc_auc.toFixed(2)})`,
                data: pts,
                borderColor: colors[idx % colors.length],
                borderWidth: 2.5,
                fill: false,
                tension: 0.2,
                pointRadius: 0
            });
        }
    });

    // Reference diagonal
    datasets.push({
        label: "Random Chance (0.50)",
        data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
        borderColor: "rgba(255, 255, 255, 0.25)",
        borderDash: [5, 5],
        borderWidth: 1.5,
        fill: false,
        pointRadius: 0
    });

    rocChartInstance = new Chart(ctx, {
        type: "line",
        data: { datasets: datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    type: "linear",
                    min: 0,
                    max: 1,
                    title: { display: true, text: "False Positive Rate (1 - Specificity)", color: "#94a3b8" },
                    grid: { color: "rgba(255, 255, 255, 0.05)" },
                    ticks: { color: "#94a3b8" }
                },
                y: {
                    type: "linear",
                    min: 0,
                    max: 1,
                    title: { display: true, text: "True Positive Rate (Sensitivity / Recall)", color: "#94a3b8" },
                    grid: { color: "rgba(255, 255, 255, 0.05)" },
                    ticks: { color: "#94a3b8" }
                }
            },
            plugins: {
                legend: { position: "bottom", labels: { color: "#cbd5e1", boxWidth: 12 } }
            }
        }
    });
}

// Sensitivity vs Specificity Chart
function renderSensSpecChart(models) {
    const ctx = document.getElementById("sensSpecBarChart");
    if (!ctx) return;

    if (sensSpecChartInstance) sensSpecChartInstance.destroy();

    const labels = models.map(m => m.model_name);
    const sensitivities = models.map(m => m.sensitivity);
    const specificities = models.map(m => m.specificity);

    sensSpecChartInstance = new Chart(ctx, {
        type: "bar",
        data: {
            labels: labels,
            datasets: [
                {
                    label: "Sensitivity (Disease Recall)",
                    data: sensitivities,
                    backgroundColor: "rgba(16, 185, 129, 0.7)",
                    borderColor: "#10b981",
                    borderWidth: 1,
                    borderRadius: 6
                },
                {
                    label: "Specificity (Healthy Detection)",
                    data: specificities,
                    backgroundColor: "rgba(79, 172, 254, 0.7)",
                    borderColor: "#4facfe",
                    borderWidth: 1,
                    borderRadius: 6
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { color: "#94a3b8", font: { size: 10 } }
                },
                y: {
                    min: 0,
                    max: 1.05,
                    grid: { color: "rgba(255, 255, 255, 0.05)" },
                    ticks: { color: "#94a3b8" }
                }
            },
            plugins: {
                legend: { position: "bottom", labels: { color: "#cbd5e1", boxWidth: 12 } }
            }
        }
    });
}

// Confusion Matrix Heatmaps
function renderConfusionMatrices(models) {
    const container = document.getElementById("cm-container");
    if (!container) return;

    container.innerHTML = "";
    
    // Choose Quantum VQC and Classical Random Forest (or SVM)
    const targetModels = models.filter(m => m.model_name.includes("VQC") || m.model_name.includes("Random Forest") || m.model_name.includes("Classical SVM"));
    const displayList = targetModels.slice(0, 2);

    displayList.forEach(m => {
        const cm = m.confusion_matrix || { tp: 14, fp: 1, tn: 10, fn: 0 };
        const cmDiv = document.createElement("div");
        cmDiv.className = "cm-box";
        cmDiv.innerHTML = `
            <div class="cm-title">${m.model_name}</div>
            <div class="cm-grid">
                <div class="cm-cell cm-cell-tn">
                    <span class="cm-val">${cm.tn}</span>
                    <span class="cm-lbl">True Negative</span>
                </div>
                <div class="cm-cell cm-cell-fp">
                    <span class="cm-val">${cm.fp}</span>
                    <span class="cm-lbl">False Positive</span>
                </div>
                <div class="cm-cell cm-cell-fn">
                    <span class="cm-val">${cm.fn}</span>
                    <span class="cm-lbl">False Negative</span>
                </div>
                <div class="cm-cell cm-cell-tp">
                    <span class="cm-val">${cm.tp}</span>
                    <span class="cm-lbl">True Positive</span>
                </div>
            </div>
            <div style="font-size: 0.72rem; color: #94a3b8;">Accuracy: ${(m.accuracy * 100).toFixed(1)}%</div>
        `;
        container.appendChild(cmDiv);
    });
}

// Circuit Visualization
function renderCircuit(schema, asciiTrace) {
    const board = document.getElementById("circuit-board");
    const asciiEl = document.getElementById("circuit-ascii-output");

    if (asciiEl && asciiTrace) {
        asciiEl.innerText = asciiTrace;
    }

    if (!board) return;
    board.innerHTML = "";

    const nQubits = (schema && schema.n_qubits) || 4;

    for (let q = 0; q < nQubits; q++) {
        const row = document.createElement("div");
        row.className = "qubit-wire-row";

        row.innerHTML = `
            <div class="qubit-label">|q[${q}]⟩</div>
            <div class="wire-line"></div>
            <div class="gates-sequence">
                <div class="gate-box" title="Hadamard Superposition">H</div>
                <div class="gate-box" title="Angle Feature Embedding">Ry(x<sub>${q}</sub>)</div>
                <div class="gate-box gate-var" title="Parameterized Layer 1">Ry(θ<sub>${q}</sub>)</div>
                <div class="gate-box gate-var" title="Parameterized Layer 1">Rz(ω<sub>${q}</sub>)</div>
                <div class="gate-cnot-ctrl" title="CNOT Control"></div>
                <div class="gate-box gate-var" title="Parameterized Layer 2">Ry(θ'<sub>${q}</sub>)</div>
                <div class="gate-box gate-var" title="Parameterized Layer 2">Rz(ω'<sub>${q}</sub>)</div>
                ${q === 0 ? '<div class="gate-box gate-meas" title="Pauli-Z Expectation Measurement">⟨Z₀⟩</div>' : ''}
            </div>
        `;
        board.appendChild(row);
    }
}

// Explainability Features
function renderExplainability(features) {
    if (!features || features.length === 0) return;

    // Table
    const tbody = document.getElementById("explain-tbody");
    if (tbody) {
        tbody.innerHTML = "";
        features.forEach(f => {
            const tr = document.createElement("tr");
            const qBadge = f.is_quantum_encoded
                ? `<span class="badge badge-info">Quantum State Qubit Wire</span>`
                : `<span class="badge" style="background: rgba(255,255,255,0.05); color:#64748b;">Classical Scaled</span>`;
            
            let role = "Metabolic regulation";
            if (f.feature.includes("ALT") || f.feature.includes("AST")) role = "Hepatic cellular integrity indicator";
            else if (f.feature.includes("Bilirubin")) role = "Biliary excretion & oxidative clearance";
            else if (f.feature.includes("Glucose")) role = "Fasting glycemic biomarker";
            else if (f.feature.includes("Albumin")) role = "Hepatic protein synthesis";
            else if (f.feature.includes("BMI")) role = "Adiposity & metabolic syndrome risk";

            tr.innerHTML = `
                <td><strong>${f.feature}</strong></td>
                <td><span class="code-badge">${f.f_score}</span></td>
                <td>${(f.normalized_importance * 100).toFixed(1)}%</td>
                <td>${qBadge}</td>
                <td style="color: #cbd5e1;">${role}</td>
            `;
            tbody.appendChild(tr);
        });
    }

    // Bar Chart
    const ctx = document.getElementById("importanceChart");
    if (!ctx) return;

    if (importanceChartInstance) importanceChartInstance.destroy();

    const topFeatures = features.slice(0, 8);
    importanceChartInstance = new Chart(ctx, {
        type: "bar",
        data: {
            labels: topFeatures.map(f => f.feature),
            datasets: [{
                label: "Normalized Diagnostic Importance",
                data: topFeatures.map(f => f.normalized_importance),
                backgroundColor: topFeatures.map(f => f.is_quantum_encoded ? "rgba(0, 242, 254, 0.75)" : "rgba(168, 85, 247, 0.5)"),
                borderColor: topFeatures.map(f => f.is_quantum_encoded ? "#00f2fe" : "#a855f7"),
                borderWidth: 1.5,
                borderRadius: 8
            }]
        },
        options: {
            indexAxis: "y",
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    grid: { color: "rgba(255, 255, 255, 0.05)" },
                    ticks: { color: "#94a3b8" }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: "#ffffff", font: { weight: "bold" } }
                }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });
}

// 4. Clinical Risk Predictor Actions
function initPredictionActions() {
    const btnPredict = document.getElementById("btn-predict-patient");
    const btnHealthy = document.getElementById("btn-load-healthy-sample");
    const btnDiseased = document.getElementById("btn-load-diseased-sample");

    if (btnHealthy) {
        btnHealthy.addEventListener("click", () => {
            setFormValues({
                age: 38, gender: 0, bmi: 22.5, tb: 0.6, db: 0.2,
                alp: 170, alt: 17, ast: 19, tp: 7.3, alb: 3.9,
                ag: 1.14, glu: 89, chol: 168, plat: 295
            });
        });
    }

    if (btnDiseased) {
        btnDiseased.addEventListener("click", () => {
            setFormValues({
                age: 62, gender: 1, bmi: 33.8, tb: 2.2, db: 1.0,
                alp: 380, alt: 80, ast: 88, tp: 5.9, alb: 2.5,
                ag: 0.73, glu: 154, chol: 248, plat: 172
            });
        });
    }

    if (btnPredict) {
        btnPredict.addEventListener("click", runPatientInference);
    }
}

function setFormValues(vals) {
    document.getElementById("inp-age").value = vals.age;
    document.getElementById("inp-gender").value = vals.gender;
    document.getElementById("inp-bmi").value = vals.bmi;
    document.getElementById("inp-tb").value = vals.tb;
    document.getElementById("inp-db").value = vals.db;
    document.getElementById("inp-alp").value = vals.alp;
    document.getElementById("inp-alt").value = vals.alt;
    document.getElementById("inp-ast").value = vals.ast;
    document.getElementById("inp-tp").value = vals.tp;
    document.getElementById("inp-alb").value = vals.alb;
    document.getElementById("inp-ag").value = vals.ag;
    document.getElementById("inp-glu").value = vals.glu;
    document.getElementById("inp-chol").value = vals.chol;
    document.getElementById("inp-plat").value = vals.plat;
}

async function runPatientInference() {
    const btn = document.getElementById("btn-predict-patient");
    const originalText = btn.innerHTML;
    btn.innerHTML = `<span class="btn-icon">⏳</span> Computing Quantum States...`;
    btn.disabled = true;

    const patientPayload = {
        "Age": parseFloat(document.getElementById("inp-age").value),
        "Gender": parseInt(document.getElementById("inp-gender").value),
        "BMI": parseFloat(document.getElementById("inp-bmi").value),
        "Total_Bilirubin": parseFloat(document.getElementById("inp-tb").value),
        "Direct_Bilirubin": parseFloat(document.getElementById("inp-db").value),
        "Alkaline_Phosphatase": parseFloat(document.getElementById("inp-alp").value),
        "Alamine_Aminotransferase": parseFloat(document.getElementById("inp-alt").value),
        "Aspartate_Aminotransferase": parseFloat(document.getElementById("inp-ast").value),
        "Total_Proteins": parseFloat(document.getElementById("inp-tp").value),
        "Albumin": parseFloat(document.getElementById("inp-alb").value),
        "Albumin_and_Globulin_Ratio": parseFloat(document.getElementById("inp-ag").value),
        "Fasting_Glucose": parseFloat(document.getElementById("inp-glu").value),
        "Serum_Cholesterol": parseFloat(document.getElementById("inp-chol").value),
        "Platelet_Count": parseFloat(document.getElementById("inp-plat").value)
    };

    try {
        const res = await fetch("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(patientPayload)
        });
        if (!res.ok) throw new Error("Inference failed");
        const out = await res.json();

        // Update UI
        const isDetected = out.hybrid_prediction.disease_detected;
        const riskPct = out.hybrid_prediction.disease_risk_percentage;
        const conf = out.hybrid_prediction.confidence;

        const badge = document.getElementById("diag-badge");
        const badgeText = document.getElementById("diag-text");
        const circle = document.getElementById("risk-score-circle");
        const percentEl = document.getElementById("risk-percent");

        badge.className = "diagnosis-status-badge " + (isDetected ? "detected" : "healthy");
        badgeText.innerText = out.hybrid_prediction.status_label;

        percentEl.innerText = `${riskPct}%`;
        if (isDetected) {
            circle.style.borderColor = "var(--danger-red)";
            circle.style.boxShadow = "0 0 30px rgba(244, 63, 94, 0.4)";
        } else {
            circle.style.borderColor = "var(--success-green)";
            circle.style.boxShadow = "0 0 30px rgba(16, 185, 129, 0.4)";
        }

        document.getElementById("vqc-prob-label").innerText = `${out.vqc_model.probability_disease}% Disease Risk`;
        document.getElementById("rf-prob-label").innerText = `${out.classical_rf_model.probability_disease}% Disease Risk`;
        document.getElementById("hybrid-conf-label").innerText = `${conf}% Confidence`;

        // Quantum angle chips
        const anglesRow = document.getElementById("quantum-angles-row");
        anglesRow.innerHTML = "";
        if (out.quantum_encoded_features) {
            Object.entries(out.quantum_encoded_features).forEach(([feat, rad]) => {
                const chip = document.createElement("span");
                chip.className = "angle-chip";
                chip.innerText = `${feat}: ${rad} rad`;
                anglesRow.appendChild(chip);
            });
        }

        // Recommendation
        const recEl = document.getElementById("clinical-rec");
        if (isDetected) {
            recEl.innerHTML = `⚠️ <strong>Early Disease Screening Positive:</strong> Elevated hepatic transaminases and inflammatory biomarkers detected. Recommended: Secondary confirmatory fibroscan, ultrasound imaging, and clinical specialist consultation.`;
        } else {
            recEl.innerHTML = `✅ <strong>Normal Screening:</strong> Key biomarker indices remain within baseline physiological limits. Recommended: Standard annual routine checkup.`;
        }

    } catch (err) {
        console.error("Inference error:", err);
        alert("Inference computation error: " + err.message);
    } finally {
        btn.innerHTML = originalText;
        btn.disabled = false;
    }
}

// 5. Re-run Benchmark Trigger
function initReBenchmark() {
    const btn = document.getElementById("btn-re-benchmark");
    const btnRefresh = document.getElementById("btn-refresh-metrics");

    const runReTrain = async () => {
        if (!confirm("Run end-to-end training of both Quantum (VQC + QSVM) and Classical models on inputdata.txt? This may take ~15-25 seconds.")) return;
        
        btn.innerHTML = `<span class="btn-icon">⏳</span> Training Quantum Circuits...`;
        btn.disabled = true;

        try {
            const res = await fetch("/api/train", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ epochs: 15, n_qubits: 4 })
            });
            if (!res.ok) throw new Error("Training failed");
            await loadBenchmarkData();
            alert("Hybrid Quantum Machine Learning models successfully trained and evaluated!");
        } catch (err) {
            alert("Training error: " + err.message);
        } finally {
            btn.innerHTML = `<span class="btn-icon">⚡</span> Run Hybrid Benchmark`;
            btn.disabled = false;
        }
    };

    if (btn) btn.addEventListener("click", runReTrain);
    if (btnRefresh) btnRefresh.addEventListener("click", loadBenchmarkData);
}
