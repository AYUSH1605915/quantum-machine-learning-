// QuantumMed Client-Side Logic & Visualizations

let radarChartInstance = null;
let rocChartInstance = null;
let sensSpecChartInstance = null;
let importanceChartInstance = null;
let cohortComparisonChartInstance = null;

document.addEventListener("DOMContentLoaded", () => {
    initTabs();
    initTopNavAndHero();
    initQuickSearch();
    loadDatasetInfo();
    loadBenchmarkData();
    initPredictionActions();
    initReBenchmark();
    initModalityTabs();
    initImageUploadAndOCR();
    initTextParsing();
});

// Centralized Tab Switching Function
function switchTab(targetId) {
    const tabButtons = document.querySelectorAll(".tab-btn");
    const tabContents = document.querySelectorAll(".tab-content");
    const navLinks = document.querySelectorAll(".nav-menu-link");

    tabButtons.forEach(b => {
        if (b.getAttribute("data-tab") === targetId) b.classList.add("active");
        else b.classList.remove("active");
    });
    tabContents.forEach(c => {
        if (c.id === targetId) c.classList.add("active");
        else c.classList.remove("active");
    });
    navLinks.forEach(link => {
        if (link.getAttribute("data-nav-tab") === targetId) link.classList.add("active");
        else link.classList.remove("active");
    });
}

// 1. Tab Switching
function initTabs() {
    const tabButtons = document.querySelectorAll(".tab-btn");
    tabButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            const targetId = btn.getAttribute("data-tab");
            switchTab(targetId);
        });
    });
}

// Top Sticky Navbar & Hero CTA Handlers
function initTopNavAndHero() {
    const navLinks = document.querySelectorAll(".nav-menu-link");
    const subLinks = document.querySelectorAll(".nav-sub-link, .footer-nav-link");
    const topNavbar = document.getElementById("top-navbar-wrapper");
    const brandHome = document.getElementById("nav-brand-home");

    // Navbar Scroll Background Transition
    window.addEventListener("scroll", () => {
        if (window.scrollY > 25) {
            topNavbar.classList.add("scrolled");
        } else {
            topNavbar.classList.remove("scrolled");
        }
    });

    // Nav Top Links
    navLinks.forEach(link => {
        link.addEventListener("click", (e) => {
            const targetTab = link.getAttribute("data-nav-tab") || link.getAttribute("data-tab");
            const scrollToId = link.getAttribute("data-scroll");
            if (targetTab) switchTab(targetTab);
            if (scrollToId) {
                const el = document.getElementById(scrollToId);
                if (el) el.scrollIntoView({ behavior: "smooth" });
            }
        });
    });

    // Sub-links in Dropdowns & Footer
    subLinks.forEach(link => {
        link.addEventListener("click", (e) => {
            e.preventDefault();
            const targetTab = link.getAttribute("data-tab");
            const scrollToId = link.getAttribute("data-scroll");
            const targetMode = link.getAttribute("data-mode");

            if (targetTab) switchTab(targetTab);
            if (targetMode) {
                const modeBtn = document.getElementById(`btn-mode-${targetMode}`);
                if (modeBtn) modeBtn.click();
            }
            if (scrollToId) {
                const el = document.getElementById(scrollToId);
                if (el) el.scrollIntoView({ behavior: "smooth" });
            }
        });
    });

    if (brandHome) {
        brandHome.addEventListener("click", () => {
            window.scrollTo({ top: 0, behavior: "smooth" });
        });
    }

    // Hero Action Buttons
    const btnHeroScreener = document.getElementById("btn-hero-screener");
    const btnHeroExplore = document.getElementById("btn-hero-explore");
    const btnNavStart = document.getElementById("btn-nav-start-analysis");
    const btnPreviewDemo = document.getElementById("btn-preview-try-demo");
    const btnPreviewLaunch = document.getElementById("btn-preview-launch");
    const btnFinalStart = document.getElementById("btn-final-start");
    const btnFinalExplore = document.getElementById("btn-final-explore");
    const btnViewDetailedBench = document.getElementById("btn-view-detailed-benchmarks");

    const launchScreener = () => {
        switchTab("tab-clinical");
        const el = document.getElementById("screener-section");
        if (el) el.scrollIntoView({ behavior: "smooth" });
    };

    if (btnHeroScreener) btnHeroScreener.addEventListener("click", launchScreener);
    if (btnNavStart) btnNavStart.addEventListener("click", launchScreener);
    if (btnPreviewLaunch) btnPreviewLaunch.addEventListener("click", launchScreener);
    if (btnFinalStart) btnFinalStart.addEventListener("click", launchScreener);

    if (btnHeroExplore) {
        btnHeroExplore.addEventListener("click", () => {
            const el = document.getElementById("capabilities-section");
            if (el) el.scrollIntoView({ behavior: "smooth" });
        });
    }

    if (btnFinalExplore) {
        btnFinalExplore.addEventListener("click", () => {
            const el = document.getElementById("capabilities-section");
            if (el) el.scrollIntoView({ behavior: "smooth" });
        });
    }

    if (btnPreviewDemo) {
        btnPreviewDemo.addEventListener("click", () => {
            switchTab("tab-clinical");
            const btnHealthy = document.getElementById("btn-load-healthy-sample");
            if (btnHealthy) btnHealthy.click();
            const el = document.getElementById("screener-section");
            if (el) el.scrollIntoView({ behavior: "smooth" });
        });
    }

    if (btnViewDetailedBench) {
        btnViewDetailedBench.addEventListener("click", () => {
            switchTab("tab-benchmarks");
            const el = document.getElementById("interactive-workspace");
            if (el) el.scrollIntoView({ behavior: "smooth" });
        });
    }

    // Screening Tool Card Jumps
    const cardQuickScan = document.getElementById("card-tool-quickscan");
    const cardManual = document.getElementById("card-tool-manual");
    const cardReport = document.getElementById("card-tool-report");
    const cardHistory = document.getElementById("card-tool-history");

    if (cardQuickScan) {
        cardQuickScan.addEventListener("click", () => {
            switchTab("tab-clinical");
            const b = document.getElementById("btn-mode-image");
            if (b) b.click();
            const el = document.getElementById("screener-section");
            if (el) el.scrollIntoView({ behavior: "smooth" });
        });
    }
    if (cardManual) {
        cardManual.addEventListener("click", () => {
            switchTab("tab-clinical");
            const b = document.getElementById("btn-mode-form");
            if (b) b.click();
            const el = document.getElementById("screener-section");
            if (el) el.scrollIntoView({ behavior: "smooth" });
        });
    }
    if (cardReport) {
        cardReport.addEventListener("click", () => {
            switchTab("tab-clinical");
            const b = document.getElementById("btn-mode-text");
            if (b) b.click();
            const el = document.getElementById("screener-section");
            if (el) el.scrollIntoView({ behavior: "smooth" });
        });
    }
    if (cardHistory) {
        cardHistory.addEventListener("click", () => {
            switchTab("tab-clinical");
            const el = document.getElementById("cohort-comparison-section");
            if (el) el.scrollIntoView({ behavior: "smooth" });
        });
    }
}

// Quick Search Modal Palette
function initQuickSearch() {
    const trigger = document.getElementById("nav-search-trigger");
    const backdrop = document.getElementById("search-modal-backdrop");
    const closeBtn = document.getElementById("btn-close-search");
    const searchInput = document.getElementById("quick-search-input");
    const resultsList = document.getElementById("search-results-list");

    if (!backdrop || !searchInput) return;

    const openSearch = () => {
        backdrop.classList.add("open");
        searchInput.value = "";
        filterResults("");
        setTimeout(() => searchInput.focus(), 50);
    };

    const closeSearch = () => {
        backdrop.classList.remove("open");
    };

    if (trigger) trigger.addEventListener("click", openSearch);
    if (closeBtn) closeBtn.addEventListener("click", closeSearch);

    backdrop.addEventListener("click", (e) => {
        if (e.target === backdrop) closeSearch();
    });

    // Keyboard Shortcuts (Ctrl+K or Cmd+K to open, ESC to close)
    window.addEventListener("keydown", (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
            e.preventDefault();
            if (backdrop.classList.contains("open")) closeSearch();
            else openSearch();
        } else if (e.key === "Escape" && backdrop.classList.contains("open")) {
            closeSearch();
        }
    });

    // Filter results on typing
    const filterResults = (query) => {
        const q = query.toLowerCase().trim();
        const items = resultsList.querySelectorAll(".search-result-item");
        items.forEach(item => {
            const text = item.innerText.toLowerCase();
            if (!q || text.includes(q)) {
                item.style.display = "flex";
            } else {
                item.style.display = "none";
            }
        });
    };

    searchInput.addEventListener("input", (e) => {
        filterResults(e.target.value);
    });

    // Handle Item Selection
    resultsList.addEventListener("click", (e) => {
        const item = e.target.closest(".search-result-item");
        if (!item) return;

        const action = item.getAttribute("data-action");
        const target = item.getAttribute("data-target");

        closeSearch();

        if (action === "tab") {
            switchTab(target);
        } else if (action === "scroll") {
            switchTab("tab-clinical");
            const targetEl = document.getElementById(target);
            if (targetEl) {
                targetEl.scrollIntoView({ behavior: "smooth", block: "center" });
                targetEl.focus();
                targetEl.style.boxShadow = "0 0 20px #38bdf8";
                setTimeout(() => { targetEl.style.boxShadow = ""; }, 1800);
            }
        }
    });
}

// 2. Load Dataset Info
async function loadDatasetInfo() {
    try {
        const res = await fetch("/api/dataset");
        if (!res.ok) throw new Error("Failed to load dataset metadata");
        const data = await res.json();

        // Update Stats
        const rowCount = data.rows || 300;
        const elRowCount = document.getElementById("dataset-row-count") || document.getElementById("dataset-count-badge");
        if (elRowCount) elRowCount.innerText = `${rowCount} Records`;

        const elFeatCount = document.getElementById("dataset-feature-count") || document.getElementById("feature-count-badge");
        if (elFeatCount && data.columns) elFeatCount.innerText = `${data.columns.length - 1} Biomarkers`;

        const elActiveFile = document.getElementById("active-file-name");
        if (elActiveFile) elActiveFile.innerText = data.filename;

        const posCount = (data.class_balance && data.class_balance[1]) || 55;
        const negCount = (data.class_balance && data.class_balance[0]) || 45;
        const elPos = document.getElementById("class-pos-count");
        if (elPos) elPos.innerText = `${posCount} (${((posCount / rowCount) * 100).toFixed(1)}%)`;

        const elNeg = document.getElementById("class-neg-count");
        if (elNeg) elNeg.innerText = `${negCount} (${((negCount / rowCount) * 100).toFixed(1)}%)`;

        const elQubitFeat = document.getElementById("quantum-selected-features") || document.getElementById("qubit-features-badge");
        if (elQubitFeat && data.selected_quantum_features && data.selected_quantum_features.length > 0) {
            elQubitFeat.innerText = `${data.selected_quantum_features.length} Qubits Selected`;
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

// 4. Input Modality Tabs Switching
function initModalityTabs() {
    const modeBtns = document.querySelectorAll(".modality-btn");
    const modePanels = document.querySelectorAll(".modality-panel");

    modeBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            const mode = btn.getAttribute("data-mode");
            modeBtns.forEach(b => b.classList.remove("active"));
            modePanels.forEach(p => p.classList.remove("active"));

            btn.classList.add("active");
            const targetPanel = document.getElementById(`panel-mode-${mode}`);
            if (targetPanel) targetPanel.classList.add("active");
        });
    });
}

// 5. Image Upload & OCR Handling
function initImageUploadAndOCR() {
    const dropzone = document.getElementById("dropzone-box");
    const fileInput = document.getElementById("lab-image-input");
    const btnBrowse = document.getElementById("btn-browse-image");
    const previewCard = document.getElementById("image-preview-card");
    const previewImg = document.getElementById("image-preview-el");
    const previewFilename = document.getElementById("preview-filename");
    const btnClear = document.getElementById("btn-clear-image");
    const progressContainer = document.getElementById("ocr-progress-container");
    const progressFill = document.getElementById("ocr-progress-fill");
    const progressPercent = document.getElementById("ocr-percentage-text");
    const statusText = document.getElementById("ocr-status-text");
    const feedbackBanner = document.getElementById("image-ocr-feedback");

    if (!dropzone || !fileInput) return;

    btnBrowse.addEventListener("click", (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    dropzone.addEventListener("click", () => fileInput.click());

    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.classList.add("dragover");
    });

    dropzone.addEventListener("dragleave", () => {
        dropzone.classList.remove("dragover");
    });

    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.classList.remove("dragover");
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleImageFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files[0]) {
            handleImageFile(e.target.files[0]);
        }
    });

    if (btnClear) {
        btnClear.addEventListener("click", () => {
            fileInput.value = "";
            previewCard.style.display = "none";
            progressContainer.style.display = "none";
            feedbackBanner.style.display = "none";
        });
    }

    async function handleImageFile(file) {
        if (!file.type.startsWith("image/")) {
            alert("Please upload a valid image file (PNG, JPG, JPEG, WEBP).");
            return;
        }

        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImg.src = e.target.result;
            previewFilename.innerText = file.name;
            previewCard.style.display = "block";
        };
        reader.readAsDataURL(file);

        // Show progress UI
        progressContainer.style.display = "block";
        progressFill.style.width = "5%";
        progressPercent.innerText = "5%";
        statusText.innerText = "Reading image and executing optical character recognition...";
        feedbackBanner.style.display = "none";

        try {
            let extractedText = "";

            // Attempt client-side Tesseract.js if loaded
            if (typeof Tesseract !== "undefined") {
                const res = await Tesseract.recognize(file, "eng", {
                    logger: (m) => {
                        if (m.status === "recognizing text") {
                            const pct = Math.round((m.progress || 0) * 100);
                            progressFill.style.width = `${Math.max(10, pct)}%`;
                            progressPercent.innerText = `${Math.max(10, pct)}%`;
                            statusText.innerText = `Recognizing text (${pct}%)...`;
                        }
                    }
                });
                extractedText = (res && res.data && res.data.text) ? res.data.text : "";
            }

            // If client OCR extracted text, send to backend parser
            let parseResult = null;
            if (extractedText && extractedText.trim().length > 10) {
                statusText.innerText = "Parsing extracted medical biomarkers...";
                const parseRes = await fetch("/api/parse-text", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ text: extractedText })
                });
                parseResult = await parseRes.json();
            } else {
                // Fallback to server-side image OCR
                statusText.innerText = "Processing via backend OCR engine...";
                const formData = new FormData();
                formData.append("image", file);
                const backendOcrRes = await fetch("/api/parse-image", {
                    method: "POST",
                    body: formData
                });
                parseResult = await backendOcrRes.json();
            }

            progressFill.style.width = "100%";
            progressPercent.innerText = "100%";
            statusText.innerText = "Extraction complete!";

            if (parseResult && parseResult.status === "success") {
                const detected = parseResult.detected_count || 0;
                setFormValues(parseResult.complete_vitals || parseResult.extracted);

                feedbackBanner.className = "feedback-banner success";
                feedbackBanner.style.display = "block";
                feedbackBanner.innerHTML = `
                    <strong>✅ Successfully Extracted ${detected} Biomarkers from Lab Report!</strong><br>
                    Detected features: <code>${Object.keys(parseResult.extracted).join(", ")}</code>.<br>
                    Review the auto-populated numbers below and click <em>Run Hybrid Quantum Disease Screening</em>.
                `;
            } else {
                feedbackBanner.className = "feedback-banner warning";
                feedbackBanner.style.display = "block";
                feedbackBanner.innerHTML = `
                    <strong>⚠️ Partial extraction:</strong> Could not detect distinct biomarkers automatically.
                    You can paste the report text under <em>Paste Medical Report Text</em> or enter numbers manually below.
                `;
            }
        } catch (err) {
            console.error("Image OCR Error:", err);
            feedbackBanner.className = "feedback-banner warning";
            feedbackBanner.style.display = "block";
            feedbackBanner.innerHTML = `
                <strong>Notice:</strong> Automated OCR finished with error: ${err.message}. 
                You can type or paste your values directly below.
            `;
        }
    }
}

// 6. Text Report Parsing Handling
function initTextParsing() {
    const btnExtract = document.getElementById("btn-extract-text");
    const btnHighRisk = document.getElementById("btn-sample-diseased-text");
    const btnHealthy = document.getElementById("btn-sample-healthy-text");
    const btnClear = document.getElementById("btn-clear-text");
    const textArea = document.getElementById("paste-text-input");
    const feedbackBanner = document.getElementById("text-extract-feedback");

    if (!btnExtract || !textArea) return;

    if (btnHighRisk) {
        btnHighRisk.addEventListener("click", () => {
            textArea.value = "Patient Age: 62 years, Gender: Male (1), BMI: 33.8. Comprehensive Liver & Metabolic Panel:\nTotal Bilirubin: 2.2 mg/dL, Direct Bilirubin: 1.0 mg/dL, Alkaline Phosphatase: 380 IU/L, ALT / SGPT: 80 IU/L, AST / SGOT: 88 IU/L, Total Proteins: 5.9 g/dL, Serum Albumin: 2.5 g/dL, A/G Ratio: 0.73, Fasting Blood Glucose: 154 mg/dL, Serum Cholesterol: 248 mg/dL, Platelets: 172 x10^3/uL. Clinical Notes: Persistent fatigue, mild right upper quadrant discomfort.";
        });
    }

    if (btnHealthy) {
        btnHealthy.addEventListener("click", () => {
            textArea.value = "Patient Age: 38 years, Gender: Female (0), BMI: 22.5. Routine Preventive Screening Panel:\nTotal Bilirubin: 0.6 mg/dL, Direct Bilirubin: 0.2 mg/dL, Alkaline Phosphatase: 170 IU/L, ALT: 17 IU/L, AST: 19 IU/L, Total Proteins: 7.3 g/dL, Serum Albumin: 3.9 g/dL, A/G Ratio: 1.14, Fasting Glucose: 89 mg/dL, Total Cholesterol: 168 mg/dL, Platelet Count: 295 x10^3/uL. Clinical Notes: Asymptomatic, unremarkable physical examination.";
        });
    }

    if (btnClear) {
        btnClear.addEventListener("click", () => {
            textArea.value = "";
            feedbackBanner.style.display = "none";
        });
    }

    btnExtract.addEventListener("click", async () => {
        const text = textArea.value.trim();
        if (!text) {
            alert("Please paste some medical report text or doctor notes first.");
            return;
        }

        const originalHtml = btnExtract.innerHTML;
        btnExtract.innerHTML = `<span class="btn-icon">⏳</span> Parsing Biomarkers...`;
        btnExtract.disabled = true;

        try {
            const res = await fetch("/api/parse-text", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text })
            });
            if (!res.ok) throw new Error("Failed to parse medical text");
            const data = await res.json();

            if (data.status === "success" && data.detected_count > 0) {
                setFormValues(data.complete_vitals || data.extracted);
                feedbackBanner.className = "feedback-banner success";
                feedbackBanner.style.display = "block";
                feedbackBanner.innerHTML = `
                    <strong>✅ Extracted ${data.detected_count} of 14 Biomarkers!</strong><br>
                    Found: <code>${Object.keys(data.extracted).join(", ")}</code>.<br>
                    The 14 biomarker inputs below have been populated. Click <em>Run Hybrid Quantum Disease Screening</em> below.
                `;
            } else {
                feedbackBanner.className = "feedback-banner warning";
                feedbackBanner.style.display = "block";
                feedbackBanner.innerHTML = `
                    <strong>⚠️ No clear biomarkers recognized.</strong> Please verify names such as 'Age', 'Bilirubin', 'ALT', 'AST', 'Glucose', or adjust values in the form directly below.
                `;
            }
        } catch (err) {
            alert("Parsing error: " + err.message);
        } finally {
            btnExtract.innerHTML = originalHtml;
            btnExtract.disabled = false;
        }
    });
}

// 7. Clinical Risk Predictor & Dataset Cohort Actions
function initPredictionActions() {
    const btnPredict = document.getElementById("btn-predict-patient");
    const btnHealthy = document.getElementById("btn-load-healthy-sample");
    const btnDiseased = document.getElementById("btn-load-diseased-sample");

    if (btnHealthy) {
        btnHealthy.addEventListener("click", () => {
            setFormValues({
                Age: 38, Gender: 0, BMI: 22.5, Total_Bilirubin: 0.6, Direct_Bilirubin: 0.2,
                Alkaline_Phosphatase: 170, Alamine_Aminotransferase: 17, Aspartate_Aminotransferase: 19,
                Total_Proteins: 7.3, Albumin: 3.9, Albumin_and_Globulin_Ratio: 1.14,
                Fasting_Glucose: 89, Serum_Cholesterol: 168, Platelet_Count: 295
            });
        });
    }

    if (btnDiseased) {
        btnDiseased.addEventListener("click", () => {
            setFormValues({
                Age: 62, Gender: 1, BMI: 33.8, Total_Bilirubin: 2.2, Direct_Bilirubin: 1.0,
                Alkaline_Phosphatase: 380, Alamine_Aminotransferase: 80, Aspartate_Aminotransferase: 88,
                Total_Proteins: 5.9, Albumin: 2.5, Albumin_and_Globulin_Ratio: 0.73,
                Fasting_Glucose: 154, Serum_Cholesterol: 248, Platelet_Count: 172
            });
        });
    }

    if (btnPredict) {
        btnPredict.addEventListener("click", runPatientInference);
    }
}

function setFormValues(vals) {
    if (!vals) return;
    const getVal = (stdKey, altKey, fallback) => {
        if (vals[stdKey] !== undefined) return vals[stdKey];
        if (vals[altKey] !== undefined) return vals[altKey];
        if (vals[stdKey.toLowerCase()] !== undefined) return vals[stdKey.toLowerCase()];
        return fallback;
    };

    document.getElementById("inp-age").value = getVal("Age", "age", 45);
    document.getElementById("inp-gender").value = getVal("Gender", "gender", 1);
    document.getElementById("inp-bmi").value = getVal("BMI", "bmi", 26.5);
    document.getElementById("inp-tb").value = getVal("Total_Bilirubin", "tb", 1.0);
    document.getElementById("inp-db").value = getVal("Direct_Bilirubin", "db", 0.3);
    document.getElementById("inp-alp").value = getVal("Alkaline_Phosphatase", "alp", 205);
    document.getElementById("inp-alt").value = getVal("Alamine_Aminotransferase", "alt", 35);
    document.getElementById("inp-ast").value = getVal("Aspartate_Aminotransferase", "ast", 38);
    document.getElementById("inp-tp").value = getVal("Total_Proteins", "tp", 6.7);
    document.getElementById("inp-alb").value = getVal("Albumin", "alb", 3.3);
    document.getElementById("inp-ag").value = getVal("Albumin_and_Globulin_Ratio", "ag", 0.95);
    document.getElementById("inp-glu").value = getVal("Fasting_Glucose", "glu", 110);
    document.getElementById("inp-chol").value = getVal("Serum_Cholesterol", "chol", 195);
    document.getElementById("inp-plat").value = getVal("Platelet_Count", "plat", 235);
}

async function runPatientInference() {
    const btn = document.getElementById("btn-predict-patient");
    const originalText = btn.innerHTML;
    btn.innerHTML = `<span class="btn-icon">⏳</span> Computing Quantum States & Comparing Cohort...`;
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

        // 1. Update Primary Diagnostic Verdict
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

        // Clinical Recommendation
        const recEl = document.getElementById("clinical-rec");
        if (out.clinical_guidance) {
            recEl.innerHTML = `<strong>Clinical Insight:</strong> ${out.clinical_guidance}`;
        }

        // 2. Render Patient vs Dataset Cohort Comparison Table
        if (out.cohort_comparison && out.cohort_comparison.comparisons) {
            renderCohortComparisonTable(out.cohort_comparison.comparisons, out.cohort_comparison);
            renderCohortComparisonChart(out.cohort_comparison.comparisons);
        }

        // Smooth scroll to comparison section if triggered
        const compSection = document.getElementById("cohort-comparison-section");
        if (compSection) {
            compSection.scrollIntoView({ behavior: "smooth", block: "nearest" });
        }

    } catch (err) {
        console.error("Inference error:", err);
        alert("Inference computation error: " + err.message);
    } finally {
        btn.innerHTML = originalText;
        btn.disabled = false;
    }
}

// 8. Render Cohort Comparison Table
function renderCohortComparisonTable(comparisons, summary) {
    const tbody = document.getElementById("cohort-comparison-tbody");
    const summaryBadge = document.getElementById("cohort-summary-badge");

    if (!tbody) return;
    tbody.innerHTML = "";

    if (summaryBadge) {
        const count = summary.abnormal_biomarkers_count || 0;
        if (count >= 3) {
            summaryBadge.className = "cohort-summary-pill high";
            summaryBadge.innerHTML = `🚨 ${count} Biomarkers Out of Normal Range`;
        } else if (count >= 1) {
            summaryBadge.className = "cohort-summary-pill";
            summaryBadge.innerHTML = `⚠️ ${count} Biomarker Mildly Out of Range`;
        } else {
            summaryBadge.className = "cohort-summary-pill optimal";
            summaryBadge.innerHTML = `✅ All 14 Biomarkers Within Reference Limits`;
        }
    }

    comparisons.forEach(item => {
        const tr = document.createElement("tr");
        const statusBadge = `<span class="badge-status ${item.status_level}">${item.status}</span>`;

        tr.innerHTML = `
            <td><strong>${item.display_name}</strong> <small style="color:var(--text-dim);">(${item.unit})</small></td>
            <td><strong style="color:#ffffff; font-size:1rem;">${item.patient_value}</strong></td>
            <td style="color:var(--text-muted);">${item.normal_range}</td>
            <td style="color:#34d399; font-weight:600;">${item.healthy_cohort_mean}</td>
            <td style="color:#f87171; font-weight:600;">${item.diseased_cohort_mean}</td>
            <td>${statusBadge}</td>
            <td>
                <div style="display:flex; align-items:center; gap:6px;">
                    <span style="font-family:var(--font-mono); font-size:0.8rem;">${item.percentile}%</span>
                    <div style="width:40px; height:6px; background:rgba(255,255,255,0.08); border-radius:3px; overflow:hidden;">
                        <div style="width:${item.percentile}%; height:100%; background:var(--primary-cyan);"></div>
                    </div>
                </div>
            </td>
        `;
        tbody.appendChild(tr);
    });
}

// 9. Render Cohort Multi-Bar Comparison Chart
function renderCohortComparisonChart(comparisons) {
    const canvas = document.getElementById("cohortComparisonChart");
    if (!canvas) return;

    // Select key representative biomarkers
    const keyFeatures = [
        "Total_Bilirubin",
        "Alkaline_Phosphatase",
        "Alamine_Aminotransferase",
        "Aspartate_Aminotransferase",
        "Fasting_Glucose",
        "Serum_Cholesterol"
    ];

    const filtered = comparisons.filter(c => keyFeatures.includes(c.biomarker));
    const labels = filtered.map(c => c.display_name);
    const patientVals = filtered.map(c => c.patient_value);
    const healthyMeans = filtered.map(c => c.healthy_cohort_mean);
    const diseasedMeans = filtered.map(c => c.diseased_cohort_mean);

    if (cohortComparisonChartInstance) {
        cohortComparisonChartInstance.destroy();
    }

    const ctx = canvas.getContext("2d");
    cohortComparisonChartInstance = new Chart(ctx, {
        type: "bar",
        data: {
            labels: labels,
            datasets: [
                {
                    label: "Your Vitals",
                    data: patientVals,
                    backgroundColor: "rgba(0, 242, 254, 0.85)",
                    borderColor: "#00f2fe",
                    borderWidth: 1.5,
                    borderRadius: 4
                },
                {
                    label: "Healthy Dataset Baseline (Class 0)",
                    data: healthyMeans,
                    backgroundColor: "rgba(16, 185, 129, 0.7)",
                    borderColor: "#10b981",
                    borderWidth: 1.5,
                    borderRadius: 4
                },
                {
                    label: "Diseased Dataset Baseline (Class 1)",
                    data: diseasedMeans,
                    backgroundColor: "rgba(239, 68, 68, 0.7)",
                    borderColor: "#ef4444",
                    borderWidth: 1.5,
                    borderRadius: 4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: "index", intersect: false },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { color: "#e2e8f0", font: { weight: "600" } }
                },
                y: {
                    grid: { color: "rgba(255, 255, 255, 0.05)" },
                    ticks: { color: "#94a3b8" }
                }
            },
            plugins: {
                legend: {
                    labels: { color: "#ffffff", font: { weight: "600" } }
                },
                tooltip: {
                    backgroundColor: "#0d1527",
                    titleColor: "#00f2fe",
                    bodyColor: "#ffffff",
                    borderColor: "rgba(0, 242, 254, 0.3)",
                    borderWidth: 1
                }
            }
        }
    });
}

// 10. Re-run Benchmark Trigger
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
