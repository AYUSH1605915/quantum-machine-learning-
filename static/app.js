let datasetInfo = null;
let lastResult = null;
let lastMultimodalResult = null;
let currentInputs = {};
let currentSource = "Manual entry";
let initTicker = null;
let initTimer = null;
let initStartedAt = null;

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  localStorage.setItem("quantummed-theme", theme);
  const dark = theme === "dark";
  const icon = document.getElementById("themeIcon");
  const label = document.getElementById("themeLabel");
  if (icon) icon.textContent = dark ? "☀" : "☾";
  if (label) label.textContent = dark ? "Light" : "Dark";
}

function initTheme() {
  const saved = localStorage.getItem("quantummed-theme");
  applyTheme(saved === "dark" ? "dark" : "light");
  document.getElementById("themeToggle")?.addEventListener("click", () => {
    applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");
  });
}

function formatElapsed(ms) {
  const total = Math.max(0, Math.floor(ms / 1000));
  return `${String(Math.floor(total/60)).padStart(2,"0")}:${String(total%60).padStart(2,"0")}`;
}

function updateInitStage(stage) {
  document.querySelectorAll("#initSteps [data-stage]").forEach((el, i) => {
    el.classList.toggle("done", i < stage);
    el.classList.toggle("active", i === stage);
  });
  const progress = document.getElementById("initProgress");
  if (progress) progress.style.width = `${Math.min(92, 12 + stage * 19)}%`;
}

function startInitExperience() {
  const box = document.getElementById("initOnly");
  box?.classList.remove("hidden");
  initStartedAt = Date.now();
  updateInitStage(0);
  let stage = 0;
  initTimer = setInterval(() => {
    const el = document.getElementById("initElapsed");
    if (el) el.textContent = formatElapsed(Date.now() - initStartedAt);
  }, 500);
  initTicker = setInterval(() => {
    stage = Math.min(4, stage + 1);
    updateInitStage(stage);
  }, 15000);
}

function stopInitExperience(success=true) {
  clearInterval(initTicker); clearInterval(initTimer);
  initTicker = initTimer = null;
  if (success) {
    document.querySelectorAll("#initSteps [data-stage]").forEach(el => {el.classList.add("done"); el.classList.remove("active")});
    const progress = document.getElementById("initProgress"); if (progress) progress.style.width = "100%";
    const title = document.getElementById("loadingTitle"); if (title) title.textContent = "QML engine ready";
    const text = document.getElementById("loadingText"); if (text) text.textContent = "Validation metrics and quantum circuit are now available";
  }
}

const prettyMap = {
  "MDVP:Fo(Hz)":"Mean fundamental frequency (Hz)",
  "MDVP:Fhi(Hz)":"Maximum fundamental frequency (Hz)",
  "MDVP:Flo(Hz)":"Minimum fundamental frequency (Hz)",
  "MDVP:Jitter(%)":"Jitter (%)",
  "MDVP:Jitter(Abs)":"Jitter (absolute)",
  "MDVP:RAP":"Relative average perturbation (RAP)",
  "MDVP:PPQ":"Pitch perturbation quotient (PPQ)",
  "Jitter:DDP":"Jitter DDP",
  "MDVP:Shimmer":"Shimmer",
  "MDVP:Shimmer(dB)":"Shimmer (dB)",
  "Shimmer:APQ3":"Shimmer APQ3",
  "Shimmer:APQ5":"Shimmer APQ5",
  "MDVP:APQ":"Amplitude perturbation quotient (APQ)",
  "Shimmer:DDA":"Shimmer DDA",
  "NHR":"Noise-to-harmonics ratio",
  "HNR":"Harmonics-to-noise ratio",
  "RPDE":"Recurrence period density entropy (RPDE)",
  "DFA":"Detrended fluctuation analysis (DFA)",
  "spread1":"Spread 1",
  "spread2":"Spread 2",
  "D2":"Correlation dimension (D2)",
  "PPE":"Pitch period entropy (PPE)"
};

const featureGroups = [
  { title:"Pitch / frequency", note:"Fundamental-frequency characteristics", features:["MDVP:Fo(Hz)","MDVP:Fhi(Hz)","MDVP:Flo(Hz)"] },
  { title:"Jitter", note:"Cycle-to-cycle frequency variation", features:["MDVP:Jitter(%)","MDVP:Jitter(Abs)","MDVP:RAP","MDVP:PPQ","Jitter:DDP"] },
  { title:"Shimmer", note:"Cycle-to-cycle amplitude variation", features:["MDVP:Shimmer","MDVP:Shimmer(dB)","Shimmer:APQ3","Shimmer:APQ5","MDVP:APQ","Shimmer:DDA"] },
  { title:"Noise & nonlinear dynamics", note:"Voice periodicity, complexity and entropy", features:["NHR","HNR","RPDE","DFA","spread1","spread2","D2","PPE"] }
];

const pretty = name => prettyMap[name] || name;
const esc = value => String(value ?? "").replace(/[&<>'"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[c]));

async function api(url, options={}) {
  const res = await fetch(url, {headers:{"Content-Type":"application/json"}, ...options});
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
  return data;
}

function setLoading(on, title="Working…", text="Please wait", mode="generic") {
  document.getElementById("loadingTitle").textContent = title;
  document.getElementById("loadingText").textContent = text;
  const initOnly = document.getElementById("initOnly");
  if (mode !== "training") initOnly?.classList.add("hidden");
  document.getElementById("loadingOverlay").classList.toggle("hidden", !on);
}

async function loadStatus() {
  const s = await api("/api/status");
  const chip = document.getElementById("engineChip");
  chip.classList.remove("ready", "error");
  if (s.status === "ready") {
    chip.classList.add("ready");
    chip.querySelector("span").textContent = "QML engine ready";
  } else if (s.status === "dataset_ready") {
    chip.querySelector("span").textContent = "Dataset ready • models idle";
  } else {
    chip.classList.add("error");
    chip.querySelector("span").textContent = "Dataset missing";
  }
}

async function loadDataset() {
  datasetInfo = await api("/api/dataset");
  document.getElementById("samplesStat").textContent = datasetInfo.total_samples;
  document.getElementById("subjectsStat").textContent = datasetInfo.unique_participants ?? "—";
  document.getElementById("featuresStat").textContent = datasetInfo.total_features;
  document.getElementById("datasetName").textContent = datasetInfo.dataset_name;
  document.getElementById("targetDefinition").textContent = datasetInfo.target_definition || "status: control=0 / Parkinson's cohort=1";
  document.getElementById("splitStrategy").textContent = datasetInfo.split_strategy;
  document.getElementById("selectedFeatures").textContent = (datasetInfo.selected_features || []).map(pretty).join(", ") || "Calculated during training";
  document.getElementById("testSamplesStat").textContent = datasetInfo.test_samples ?? "—";
  document.getElementById("testSubjectsStat").textContent = datasetInfo.test_participants ?? "—";
  renderFeatureForm();
  updateInputCoverage();
}

function renderFeatureForm() {
  const form = document.getElementById("featureForm");
  form.innerHTML = "";
  const available = new Set(datasetInfo.feature_names || []);
  const used = new Set();

  featureGroups.forEach(group => {
    const members = group.features.filter(f => available.has(f));
    if (!members.length) return;
    const section = document.createElement("section");
    section.className = "feature-group";
    section.innerHTML = `<div class="feature-group-head"><h4>${esc(group.title)}</h4><span>${esc(group.note)}</span></div><div class="feature-grid"></div>`;
    const grid = section.querySelector(".feature-grid");
    members.forEach(name => {
      used.add(name);
      const wrap = document.createElement("div");
      wrap.className = "field";
      wrap.innerHTML = `<label>${esc(pretty(name))}</label><input type="number" step="any" data-feature="${esc(name)}" placeholder="blank = impute">`;
      grid.appendChild(wrap);
    });
    form.appendChild(section);
  });

  const extras = (datasetInfo.feature_names || []).filter(f => !used.has(f));
  if (extras.length) {
    const section = document.createElement("section");
    section.className = "feature-group";
    section.innerHTML = `<div class="feature-group-head"><h4>Additional features</h4><span>Other numeric predictors</span></div><div class="feature-grid"></div>`;
    const grid = section.querySelector(".feature-grid");
    extras.forEach(name => {
      const wrap = document.createElement("div");
      wrap.className = "field";
      wrap.innerHTML = `<label>${esc(pretty(name))}</label><input type="number" step="any" data-feature="${esc(name)}" placeholder="blank = impute">`;
      grid.appendChild(wrap);
    });
    form.appendChild(section);
  }

  document.querySelectorAll("[data-feature]").forEach(input => input.addEventListener("input", () => {
    currentSource = "Manual entry";
    document.getElementById("sampleSource").textContent = currentSource;
    updateInputCoverage();
  }));
}

function updateInputCoverage() {
  const inputs = [...document.querySelectorAll("[data-feature]")];
  const filled = inputs.filter(i => i.value !== "").length;
  const total = datasetInfo?.total_features || inputs.length || 0;
  document.getElementById("inputCoverage").textContent = `${filled} / ${total}`;
  document.getElementById("coverageBar").style.width = total ? `${100 * filled / total}%` : "0%";
}

function collectInputs() {
  const data = {};
  document.querySelectorAll("[data-feature]").forEach(input => {
    if (input.value !== "") data[input.dataset.feature] = Number(input.value);
  });
  currentInputs = data;
  return data;
}

function fillInputs(features) {
  currentInputs = features;
  document.querySelectorAll("[data-feature]").forEach(input => {
    const v = features[input.dataset.feature];
    input.value = (v === undefined || v === null) ? "" : v;
  });
  updateInputCoverage();
}

async function loadSample(kind) {
  const data = await api(`/api/sample/${kind}`);
  fillInputs(data.features);
  currentSource = `${data.known_dataset_label_text} • ${data.source}`;
  document.getElementById("sampleSource").textContent = currentSource;
  document.getElementById("screening").scrollIntoView({behavior:"smooth", block:"start"});
}

async function trainModels() {
  setLoading(true, "Initializing hybrid QML engine", "Starting participant-separated training and validation", "training");
  startInitExperience();
  let ok = false;
  try {
    await api("/api/train", {method:"POST", body:JSON.stringify({epochs:8,n_qubits:4})});
    await loadStatus();
    await loadDataset();
    await loadBenchmarks();
    ok = true;
    stopInitExperience(true);
    await new Promise(r => setTimeout(r, 1100));
  } catch (e) {
    stopInitExperience(false);
    alert(`Training failed: ${e.message}`);
  } finally {
    document.getElementById("initOnly")?.classList.add("hidden");
    setLoading(false);
  }
}

async function ensureReady() {
  const s = await api("/api/status");
  if (s.status !== "ready") await trainModels();
}

async function runAssessment() {
  await ensureReady();
  setLoading(true, "Running hybrid inference…", "Random Forest + variational quantum circuit + quantum fidelity kernel");
  try {
    const result = await api("/api/predict", {method:"POST", body:JSON.stringify(collectInputs())});
    lastResult = result;
    renderResult(result);
  } catch (e) {
    alert(`Assessment failed: ${e.message}`);
  } finally { setLoading(false); }
}

function formatDate(iso) {
  try { return new Date(iso).toLocaleString(); } catch { return iso || "—"; }
}

function renderCohortEvidence(r) {
  const encoded = Object.keys(r.quantum_encoded_features || {});
  const rows = r.cohort_comparison?.comparisons || [];
  const chosen = rows.filter(x => encoded.includes(x.feature));
  const displayRows = (chosen.length ? chosen : rows.slice(0,4)).slice(0,4);
  const host = document.getElementById("cohortEvidence");
  if (!displayRows.length) {
    host.innerHTML = `<div class="empty-inline">No cohort comparison available for entered features.</div>`;
    return;
  }
  host.innerHTML = `
    <div class="cohort-head"><span>Feature</span><span>Patient</span><span>Control mean</span><span>PD mean</span><span>Alignment</span></div>
    ${displayRows.map(x => `<div class="cohort-row">
      <span>${esc(pretty(x.feature))}</span>
      <strong>${Number(x.patient_value).toFixed(4)}</strong>
      <span>${Number(x.control_mean).toFixed(4)}</span>
      <span>${Number(x.parkinsons_mean).toFixed(4)}</span>
      <em class="align-${esc(x.alignment_level)}">${esc(x.alignment)}</em>
    </div>`).join("")}`;
}

function renderSeverityGate(r) {
  const severityBlock = document.getElementById("severityBlock");
  const gateBlock = document.getElementById("severityGateBlock");
  severityBlock.classList.add("hidden");
  gateBlock.classList.add("hidden");

  if (r.symptom_burden?.available) {
    severityBlock.classList.remove("hidden");
    document.getElementById("severityScores").innerHTML = `
      <div class="score-tile"><span>Estimated motor_UPDRS</span><strong>${esc(r.symptom_burden.estimated_motor_updrs)}</strong></div>
      <div class="score-tile"><span>Estimated total_UPDRS</span><strong>${esc(r.symptom_burden.estimated_total_updrs)}</strong></div>
      <div class="score-tile"><span>Motor cohort percentile</span><strong>${esc(r.symptom_burden.motor_cohort_percentile)}%</strong></div>`;
    document.getElementById("severityNote").textContent = r.symptom_burden.note;
    return;
  }

  if (r.symptom_burden?.withheld) {
    const gate = r.symptom_burden.validation_gate || {};
    const m = gate.metrics || {};
    gateBlock.classList.remove("hidden");
    document.getElementById("severityGateTitle").textContent = "Auxiliary UPDRS output withheld by validation gate";
    document.getElementById("severityGateText").textContent = r.symptom_burden.reason;
    document.getElementById("severityGateMetrics").innerHTML = `
      <span>motor R² <strong>${m.motor_r2 ?? "—"}</strong></span>
      <span>total R² <strong>${m.total_r2 ?? "—"}</strong></span>
      <span>motor MAE <strong>${m.motor_mae ?? "—"}</strong></span>
      <span>total MAE <strong>${m.total_mae ?? "—"}</strong></span>`;
  }
}

function renderResult(r) {
  document.getElementById("emptyState").classList.add("hidden");
  document.getElementById("resultContent").classList.remove("hidden");
  document.getElementById("resultCard").classList.remove("empty");

  const score = r.assessment.hybrid_risk_index;
  const riskRing = document.getElementById("riskRing");
  riskRing.className = `risk-ring risk-${r.assessment.risk_band_key}`;
  document.getElementById("riskScore").textContent = score.toFixed(1);
  document.getElementById("riskBand").textContent = r.assessment.risk_band;
  document.getElementById("interpretation").textContent = r.assessment.interpretation;
  document.getElementById("agreement").textContent = r.assessment.model_agreement;
  document.getElementById("nextStep").textContent = r.assessment.next_step;
  document.getElementById("disclaimer").textContent = r.assessment.disclaimer;
  riskRing.style.setProperty("--score", `${Math.max(0,Math.min(100,score))*3.6}deg`);

  document.getElementById("modelScores").innerHTML = Object.entries(r.model_scores)
    .map(([k,v]) => `<div class="score-tile"><span>${esc(k)}</span><strong>${Number(v).toFixed(1)}</strong><small>model index</small></div>`).join("");

  document.getElementById("quantumFeatures").innerHTML = Object.entries(r.quantum_encoded_features)
    .map(([k,v]) => `<span class="tag">${esc(pretty(k))}<b>${esc(v)}</b></span>`).join("");

  const global = r.top_global_features || [];
  const maxImp = Math.max(...global.map(x => x.normalized_importance), 0.001);
  document.getElementById("featureImportance").innerHTML = global.map(x => {
    const pct = Math.max(2, (x.normalized_importance / maxImp) * 100);
    return `<div class="importance-row"><span>${esc(pretty(x.feature))}</span><div class="bar"><i style="width:${pct}%"></i></div><strong>${(x.normalized_importance*100).toFixed(1)}%</strong></div>`;
  }).join("");

  renderCohortEvidence(r);
  renderSeverityGate(r);

  document.getElementById("reportAssessmentId").textContent = r.meta?.assessment_id || "—";
  document.getElementById("reportGeneratedAt").textContent = formatDate(r.meta?.generated_at);
  document.getElementById("reportInputSource").textContent = currentSource;
  document.getElementById("reportCoverage").textContent = `${r.input_quality?.provided_features ?? "—"}/${r.input_quality?.total_features ?? "—"} (${r.input_quality?.completeness_percent ?? "—"}%)`;

  document.getElementById("resultCard").scrollIntoView({behavior:"smooth", block:"nearest"});
}

async function loadBenchmarks() {
  const b = await api("/api/benchmark");
  const ds = b.dataset_summary;
  document.getElementById("splitNote").textContent = `${ds.split_strategy}. Test participants are kept separate from training participants.`;
  document.getElementById("selectedFeatures").textContent = (ds.selected_features || []).map(pretty).join(", ");
  document.getElementById("circuitAscii").textContent = b.circuit_ascii || "Initialize the engine to render the trained PennyLane circuit.";
  document.getElementById("testSamplesStat").textContent = ds.test_samples ?? "—";
  document.getElementById("testSubjectsStat").textContent = ds.test_participants ?? "—";

  const body = document.getElementById("benchmarkBody");
  if (!b.trained) {
    body.innerHTML = `<tr><td colspan="6">Initialize the QML engine to calculate metrics.</td></tr>`;
    document.getElementById("severityGateStat").textContent = "Pending";
    return;
  }

  body.innerHTML = b.models_evaluation.map(m => `<tr>
    <td>${esc(m.model_name)}</td><td>${(m.accuracy*100).toFixed(1)}%</td><td>${(m.sensitivity*100).toFixed(1)}%</td>
    <td>${(m.specificity*100).toFixed(1)}%</td><td>${(m.f1_score*100).toFixed(1)}%</td><td>${Number(m.roc_auc).toFixed(3)}</td>
  </tr>`).join("");

  const best = [...b.models_evaluation].sort((a,b) => b.accuracy - a.accuracy)[0];
  document.getElementById("bestAccuracyStat").textContent = `${(best.accuracy*100).toFixed(1)}%`;
  document.getElementById("bestAccuracyModel").textContent = best.model_name;

  const gate = b.auxiliary_updrs_gate || {};
  document.getElementById("severityGateStat").textContent = gate.passed ? "Passed" : "Withheld";
  document.getElementById("severityGateSmall").textContent = gate.passed ? "eligible for experimental display" : "failed participant-level R² gate";

  const highlight = document.getElementById("validationHighlight");
  highlight.classList.remove("hidden");
  highlight.innerHTML = `<strong>Held-out validation:</strong> ${esc(best.model_name)} reached <b>${(best.accuracy*100).toFixed(1)}% accuracy</b> on ${ds.test_samples} held-out recordings using participant-separated evaluation. <span>Small research cohort — not external clinical validation.</span>`;
}


function getSelectedModalities() {
  return [...document.querySelectorAll('input[name="mmModality"]:checked')].map(el => el.value);
}

async function runMultimodalDemo(kind) {
  const modalities = getSelectedModalities();
  if (!modalities.length) {
    alert("Select at least one modality.");
    return;
  }

  if (modalities.includes("voice")) {
    await ensureReady();
  }

  const labels = {voice:"voice", gait:"gait", handwriting:"handwriting", eeg:"EEG"};
  const readable = modalities.map(m => labels[m] || m).join(", ");
  setLoading(true, "Running multimodal fusion…", `Evaluating selected branches: ${readable}`);

  try {
    const result = await api(`/api/multimodal/demo/${kind}`, {
      method:"POST",
      body:JSON.stringify({modalities})
    });
    lastMultimodalResult = result;
    renderMultimodalResult(result);
  } catch (e) {
    alert(`Multimodal assessment failed: ${e.message}`);
  } finally {
    setLoading(false);
  }
}

function renderMultimodalResult(r) {
  document.getElementById("mmEmpty")?.classList.add("hidden");
  document.getElementById("mmResult")?.classList.remove("hidden");
  document.getElementById("mmScore").textContent = Number(r.assessment.multimodal_pattern_index).toFixed(1);
  document.getElementById("mmBand").textContent = r.assessment.risk_band;
  document.getElementById("mmAgreement").textContent = r.assessment.modality_agreement;
  document.getElementById("mmScope").textContent = `${r.known_label_text} demonstration • ${r.bundle_scope}`;
  document.getElementById("mmFusionNote").textContent = r.fusion_note || "";
  document.getElementById("mmInterpretation").textContent = r.assessment.interpretation || "";
  document.getElementById("mmDisclaimer").textContent = r.assessment.disclaimer || "";

  const used = r.meta?.modalities_used || Object.keys(r.branch_details || {});
  const available = r.meta?.modalities_available || ["voice","gait","handwriting","eeg"];
  document.getElementById("mmCoverage").textContent = `${used.length} / ${available.length}`;
  document.getElementById("mmSelectedNames").textContent = used.map(x => ({
    voice:"Voice", gait:"Gait", handwriting:"Handwriting", eeg:"EEG"
  }[x] || x)).join(" + ");
  document.getElementById("mmAssessmentId").textContent = r.meta?.assessment_id || "—";

  const labels = {voice:"Voice", gait:"Gait", handwriting:"Handwriting", eeg:"EEG"};
  const order = ["voice","gait","handwriting","eeg"];
  const rows = order
    .filter(name => (r.branch_details || {})[name])
    .map(name => {
      const d = r.branch_details[name];
      const weight = Number((r.fusion_weights || {})[name] || 0) * 100;
      const auc = d.roc_auc == null ? "—" : Number(d.roc_auc).toFixed(3);
      return `<tr>
        <td><strong>${esc(labels[name] || name)}</strong></td>
        <td>${esc(d.model_name || "—")}</td>
        <td>${Number(d.score_percent).toFixed(1)}/100</td>
        <td>${auc}</td>
        <td>${weight.toFixed(1)}%</td>
      </tr>`;
    }).join("");

  document.getElementById("mmBranchBody").innerHTML = rows;
  document.getElementById("multimodal").scrollIntoView({behavior:"smooth", block:"start"});
}

function downloadMultimodalJSON() {
  if (!lastMultimodalResult) return;
  const blob = new Blob([JSON.stringify(lastMultimodalResult, null, 2)], {type:"application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `${lastMultimodalResult.meta?.assessment_id || "quantummed_multimodal_assessment"}.json`;
  a.click();
  URL.revokeObjectURL(a.href);
}

function printMultimodalReport() {
  const r = lastMultimodalResult;
  if (!r) return;

  const labels = {voice:"Voice", gait:"Gait", handwriting:"Handwriting", eeg:"EEG"};
  const rows = ["voice","gait","handwriting","eeg"]
    .filter(name => (r.branch_details || {})[name])
    .map(name => {
      const d = r.branch_details[name];
      const w = Number((r.fusion_weights || {})[name] || 0) * 100;
      const auc = d.roc_auc == null ? "—" : Number(d.roc_auc).toFixed(3);
      return `<tr>
        <td>${esc(labels[name] || name)}</td>
        <td>${esc(d.model_name || "—")}</td>
        <td>${Number(d.score_percent).toFixed(1)}/100</td>
        <td>${auc}</td>
        <td>${w.toFixed(1)}%</td>
      </tr>`;
    }).join("");

  const used = r.meta?.modalities_used || [];
  const report = window.open("", "_blank");
  if (!report) {
    alert("Allow pop-ups to print the multimodal report.");
    return;
  }
  report.opener = null;
  report.document.write(`<!doctype html>
  <html><head><meta charset="utf-8"><title>${esc(r.meta?.assessment_id || "QuantumMed Multimodal Assessment")}</title>
  <style>
    body{font-family:Arial,sans-serif;color:#172b3d;margin:36px;line-height:1.45}
    h1{font-size:24px;margin:0 0 4px} h2{font-size:18px;margin-top:28px}
    .sub{color:#62778b;margin-bottom:24px}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:18px 0}
    .card{border:1px solid #d7e1e8;border-radius:10px;padding:14px}.card span{display:block;color:#6b7f90;font-size:12px}.card strong{font-size:18px}
    table{border-collapse:collapse;width:100%;margin-top:12px} th,td{border-bottom:1px solid #dbe3e9;padding:10px;text-align:left;font-size:12px}
    th{background:#f3f6f8;color:#50677a}.note{background:#f6f9fa;border-left:4px solid #2d7f89;padding:12px 14px;margin:16px 0}
    .warn{background:#fff8e8;border:1px solid #ead7a3;padding:12px 14px;margin-top:20px;font-size:12px}
    .footer{margin-top:28px;color:#708292;font-size:11px}
    @media print{body{margin:18mm}.no-print{display:none}}
  </style></head><body>
    <h1>QuantumMed Multimodal Research Assessment</h1>
    <div class="sub">Assessment ID: ${esc(r.meta?.assessment_id || "—")} • Generated: ${esc(formatDate(r.meta?.generated_at))}</div>
    <div class="grid">
      <div class="card"><span>Multimodal Pattern Index</span><strong>${Number(r.assessment.multimodal_pattern_index).toFixed(1)} / 100</strong></div>
      <div class="card"><span>Pattern</span><strong>${esc(r.assessment.risk_band)}</strong></div>
      <div class="card"><span>Agreement</span><strong>${esc(r.assessment.modality_agreement)}</strong></div>
    </div>
    <div class="note"><strong>Modalities evaluated:</strong> ${esc(used.map(x => labels[x] || x).join(", "))}<br>
    <strong>Fusion:</strong> ${esc(r.meta?.fusion_type || "validation-weighted late decision fusion")}</div>
    <h2>Branch evidence</h2>
    <table><thead><tr><th>Signal</th><th>Selected model</th><th>Branch index</th><th>Validation ROC-AUC</th><th>Fusion weight</th></tr></thead><tbody>${rows}</tbody></table>
    <h2>Interpretation</h2><p>${esc(r.assessment.interpretation || "")}</p>
    <h2>Demonstration scope</h2><p>${esc(r.bundle_scope || "")}</p>
    <div class="warn"><strong>Important:</strong> ${esc(r.assessment.disclaimer || "")}</div>
    <div class="footer">ThunderStars • SIH 2026 • Research screening prototype</div>
    <script>window.onload=()=>setTimeout(()=>window.print(),250);<\/script>
  </body></html>`);
  report.document.close();
}

function downloadJSON() {
  if (!lastResult) return;
  const payload = {generated_at:new Date().toISOString(), input_source:currentSource, input:currentInputs, result:lastResult};
  const blob = new Blob([JSON.stringify(payload,null,2)], {type:"application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `${lastResult.meta?.assessment_id || "quantummed_parkinsons_assessment"}.json`;
  a.click();
  URL.revokeObjectURL(a.href);
}

function printReport() {
  if (!lastResult) return;
  setTimeout(() => window.print(), 50);
}

document.addEventListener("DOMContentLoaded", async () => {
  initTheme();
  document.querySelectorAll("[data-scroll]").forEach(b => b.addEventListener("click", () => document.getElementById(b.dataset.scroll).scrollIntoView({behavior:"smooth"})));
  document.getElementById("startBtn").addEventListener("click", () => document.getElementById("screening").scrollIntoView({behavior:"smooth"}));
  document.getElementById("trainBtn").addEventListener("click", trainModels);
  document.getElementById("controlSampleBtn").addEventListener("click", () => loadSample("control"));
  document.getElementById("pdSampleBtn").addEventListener("click", () => loadSample("pd"));
  document.getElementById("runBtn").addEventListener("click", runAssessment);
  document.getElementById("mmControlBtn")?.addEventListener("click", () => runMultimodalDemo("control"));
  document.getElementById("mmPdBtn")?.addEventListener("click", () => runMultimodalDemo("pd"));
  document.getElementById("mmReportBtn")?.addEventListener("click", printMultimodalReport);
  document.getElementById("mmJsonBtn")?.addEventListener("click", downloadMultimodalJSON);
  document.getElementById("printBtn").addEventListener("click", printReport);
  document.getElementById("jsonBtn").addEventListener("click", downloadJSON);
  try {
    await loadStatus();
    await loadDataset();
    await loadBenchmarks();
  } catch(e) {
    console.error(e);
    alert(e.message);
  }
});
