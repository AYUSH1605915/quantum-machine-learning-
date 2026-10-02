let datasetInfo = null;
let lastResult = null;
let lastMultimodalResult = null;
const assessmentInputs = {voice:null, gait:null, handwriting:null, eeg:null};
let assessmentInputSource = "user";
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
  if (icon) icon.textContent = dark ? "\u2600" : "\u263E";
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

function setLoading(on, title="Working\u2026", text="Please wait", mode="generic") {
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
    chip.querySelector("span").textContent = "Dataset ready \u2022 models idle";
  } else {
    chip.classList.add("error");
    chip.querySelector("span").textContent = "Dataset missing";
  }
}

async function loadDataset() {
  datasetInfo = await api("/api/dataset");
  document.getElementById("samplesStat").textContent = datasetInfo.total_samples;
  document.getElementById("subjectsStat").textContent = datasetInfo.unique_participants ?? "\u2014";
  document.getElementById("featuresStat").textContent = datasetInfo.total_features;
  document.getElementById("datasetName").textContent = datasetInfo.dataset_name;
  document.getElementById("targetDefinition").textContent = datasetInfo.target_definition || "status: control=0 / Parkinson's cohort=1";
  document.getElementById("splitStrategy").textContent = datasetInfo.split_strategy;
  document.getElementById("selectedFeatures").textContent = (datasetInfo.selected_features || []).map(pretty).join(", ") || "Calculated during training";
  document.getElementById("testSamplesStat").textContent = datasetInfo.test_samples ?? "\u2014";
  document.getElementById("testSubjectsStat").textContent = datasetInfo.test_participants ?? "\u2014";
  renderFeatureForm();
  updateInputCoverage();
}

function renderFeatureForm() {
  const form = document.getElementById("featureForm");
  if (!form) return;
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
  const coverage = document.getElementById("inputCoverage");
  const bar = document.getElementById("coverageBar");
  if (!coverage || !bar) return;
  const inputs = [...document.querySelectorAll("[data-feature]")];
  const filled = inputs.filter(i => i.value !== "").length;
  const total = datasetInfo?.total_features || inputs.length || 0;
  coverage.textContent = `${filled} / ${total}`;
  bar.style.width = total ? `${100 * filled / total}%` : "0%";
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
  currentSource = `${data.known_dataset_label_text} \u2022 ${data.source}`;
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
  setLoading(true, "Running hybrid inference\u2026", "Random Forest + variational quantum circuit + quantum fidelity kernel");
  try {
    const result = await api("/api/predict", {method:"POST", body:JSON.stringify(collectInputs())});
    lastResult = result;
    renderResult(result);
  } catch (e) {
    alert(`Assessment failed: ${e.message}`);
  } finally { setLoading(false); }
}

function formatDate(iso) {
  try { return new Date(iso).toLocaleString(); } catch { return iso || "\u2014"; }
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
      <span>motor R\u00B2 <strong>${m.motor_r2 ?? "\u2014"}</strong></span>
      <span>total R\u00B2 <strong>${m.total_r2 ?? "\u2014"}</strong></span>
      <span>motor MAE <strong>${m.motor_mae ?? "\u2014"}</strong></span>
      <span>total MAE <strong>${m.total_mae ?? "\u2014"}</strong></span>`;
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

  document.getElementById("reportAssessmentId").textContent = r.meta?.assessment_id || "\u2014";
  document.getElementById("reportGeneratedAt").textContent = formatDate(r.meta?.generated_at);
  document.getElementById("reportInputSource").textContent = currentSource;
  document.getElementById("reportCoverage").textContent = `${r.input_quality?.provided_features ?? "\u2014"}/${r.input_quality?.total_features ?? "\u2014"} (${r.input_quality?.completeness_percent ?? "\u2014"}%)`;

  document.getElementById("resultCard").scrollIntoView({behavior:"smooth", block:"nearest"});
}

async function loadBenchmarks() {
  const b = await api("/api/benchmark");
  const ds = b.dataset_summary;
  document.getElementById("splitNote").textContent = `${ds.split_strategy}. Test participants are kept separate from training participants.`;
  document.getElementById("selectedFeatures").textContent = (ds.selected_features || []).map(pretty).join(", ");
  const circuitAscii = document.getElementById("circuitAscii"); if (circuitAscii) circuitAscii.textContent = b.circuit_ascii || "Initialize the engine to render the trained PennyLane circuit.";
  document.getElementById("testSamplesStat").textContent = ds.test_samples ?? "\u2014";
  document.getElementById("testSubjectsStat").textContent = ds.test_participants ?? "\u2014";

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
  document.getElementById("severityGateSmall").textContent = gate.passed ? "eligible for experimental display" : "failed participant-level R\u00B2 gate";

  const highlight = document.getElementById("validationHighlight");
  highlight.classList.remove("hidden");
  highlight.innerHTML = `<strong>Held-out validation:</strong> ${esc(best.model_name)} reached <b>${(best.accuracy*100).toFixed(1)}% accuracy</b> on ${ds.test_samples} held-out recordings using participant-separated evaluation. <span>Small research cohort \u2014 not external clinical validation.</span>`;
}


function getSelectedModalities() {
  return [...document.querySelectorAll('input[name="mmModality"]:checked')].map(el => el.value);
}

function parseFeatureObject(text) {
  const cleaned = String(text || "").trim();
  if (!cleaned) return null;
  const obj = JSON.parse(cleaned);
  if (!obj || Array.isArray(obj) || typeof obj !== "object") {
    throw new Error("Input must be a JSON object of feature-name/value pairs.");
  }
  const numeric = {};
  for (const [key, value] of Object.entries(obj)) {
    if (value === "" || value == null) continue;
    const n = Number(value);
    if (Number.isFinite(n)) numeric[key] = n;
  }
  if (!Object.keys(numeric).length) throw new Error("No numeric feature values were found.");
  return numeric;
}

function parseCsvLine(line) {
  const out = [];
  let current = "", quoted = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === '"') {
      if (quoted && line[i + 1] === '"') { current += '"'; i++; }
      else quoted = !quoted;
    } else if (ch === "," && !quoted) {
      out.push(current.trim()); current = "";
    } else current += ch;
  }
  out.push(current.trim());
  return out;
}

function parseFeatureCsv(text) {
  const lines = String(text || "").split(/\r?\n/).filter(x => x.trim());
  if (lines.length < 2) throw new Error("CSV must contain a header row and at least one data row.");
  const headers = parseCsvLine(lines[0]);
  const values = parseCsvLine(lines[1]);
  const obj = {};
  headers.forEach((h, i) => {
    const n = Number(values[i]);
    if (h && Number.isFinite(n)) obj[h] = n;
  });
  if (!Object.keys(obj).length) throw new Error("No numeric feature values were found in the first CSV data row.");
  return obj;
}

function setAssessmentInput(modality, features, source="user") {
  assessmentInputs[modality] = features && typeof features === "object" ? features : null;
  const editor = document.getElementById(`input-${modality}`);
  if (editor) editor.value = assessmentInputs[modality] ? JSON.stringify(assessmentInputs[modality], null, 2) : "";
  refreshInputCard(modality, source);
  updateAssessmentReadiness();
}

function refreshInputCard(modality, source="user") {
  const obj = assessmentInputs[modality];
  const count = obj ? Object.keys(obj).length : 0;
  const state = document.getElementById(`state-${modality}`);
  const countEl = document.getElementById(`count-${modality}`);
  if (countEl) countEl.textContent = `${count} feature${count === 1 ? "" : "s"} loaded`;
  if (state) {
    state.textContent = count ? (source === "demo" ? "Example loaded" : "Input ready") : "No input";
    state.classList.toggle("empty", !count);
    state.classList.toggle("ready", !!count);
  }
}

function syncEditorToInput(modality, quiet=false) {
  const editor = document.getElementById(`input-${modality}`);
  if (!editor) return null;
  const text = editor.value.trim();
  if (!text) {
    assessmentInputs[modality] = null;
    refreshInputCard(modality);
    updateAssessmentReadiness();
    return null;
  }
  try {
    const obj = parseFeatureObject(text);
    assessmentInputs[modality] = obj;
    refreshInputCard(modality);
    updateAssessmentReadiness();
    return obj;
  } catch (e) {
    if (!quiet) throw e;
    return null;
  }
}

async function handleFeatureFile(modality, file) {
  if (!file) return;
  const text = await file.text();
  let obj;
  if (file.name.toLowerCase().endsWith(".json")) obj = parseFeatureObject(text);
  else obj = parseFeatureCsv(text);
  assessmentInputSource = "user";
  setAssessmentInput(modality, obj, "user");
}

async function loadAssessmentExample(kind) {
  const selected = getSelectedModalities();
  if (!selected.length) {
    alert("Select at least one signal first.");
    return;
  }
  setLoading(true, "Loading visible example inputs\u2026", "Populating the same patient-input containers used by the integrated assessment");
  try {
    const r = await api(`/api/assessment/sample/${kind}`);
    const mapping = {
      voice: r.voice?.features,
      gait: r.gait?.features,
      handwriting: r.handwriting?.features,
      eeg: r.eeg?.features,
    };
    for (const modality of ["voice","gait","handwriting","eeg"]) {
      if (selected.includes(modality)) setAssessmentInput(modality, mapping[modality] || null, "demo");
      else setAssessmentInput(modality, null);
    }
    assessmentInputSource = kind === "control" ? "demo-control" : "demo-pd";
    document.getElementById("mmEmpty")?.classList.remove("hidden");
    document.getElementById("mmResult")?.classList.add("hidden");
  } catch (e) {
    alert(`Could not load example inputs: ${e.message}`);
  } finally {
    setLoading(false);
  }
}

function clearAssessmentInputs() {
  for (const modality of ["voice","gait","handwriting","eeg"]) {
    assessmentInputs[modality] = null;
    const editor = document.getElementById(`input-${modality}`);
    if (editor) editor.value = "";
    refreshInputCard(modality);
  }
  assessmentInputSource = "user";
  lastMultimodalResult = null;
  document.getElementById("mmResult")?.classList.add("hidden");
  document.getElementById("mmEmpty")?.classList.remove("hidden");
  updateAssessmentReadiness();
}

function updateAssessmentReadiness() {
  const selected = getSelectedModalities();
  const ready = selected.filter(m => assessmentInputs[m] && Object.keys(assessmentInputs[m]).length);
  const missing = selected.filter(m => !assessmentInputs[m] || !Object.keys(assessmentInputs[m]).length);
  const title = document.getElementById("inputReadyTitle");
  const text = document.getElementById("inputReadyText");
  const run = document.getElementById("mmRunBtn");

  if (!selected.length) {
    if (title) title.textContent = "No signals selected";
    if (text) text.textContent = "Select at least one modality.";
    if (run) run.disabled = true;
    return;
  }
  if (missing.length) {
    if (title) title.textContent = `${ready.length} of ${selected.length} selected signals have input`;
    if (text) text.textContent = `Still needed: ${missing.map(x => ({voice:"Voice",gait:"Gait",handwriting:"Handwriting",eeg:"EEG"}[x])).join(", ")}`;
    if (run) run.disabled = true;
  } else {
    if (title) title.textContent = `${ready.length} signal${ready.length === 1 ? "" : "s"} ready for assessment`;
    if (text) text.textContent = "QuantumMed will return one integrated result using these loaded measurements.";
    if (run) run.disabled = false;
  }
}

async function runIntegratedAssessment() {
  const selected = getSelectedModalities();
  if (!selected.length) {
    alert("Select at least one signal.");
    return;
  }

  try {
    for (const modality of selected) syncEditorToInput(modality);
  } catch (e) {
    alert(`Please fix the feature JSON before running: ${e.message}`);
    return;
  }

  const missing = selected.filter(m => !assessmentInputs[m] || !Object.keys(assessmentInputs[m]).length);
  if (missing.length) {
    alert(`Input is missing for: ${missing.join(", ")}`);
    updateAssessmentReadiness();
    return;
  }

  if (selected.includes("voice")) await ensureReady();

  const payload = {};
  for (const modality of selected) payload[modality] = assessmentInputs[modality];

  setLoading(true, "Running integrated assessment\u2026", "Evaluating loaded signal values and generating one final Pattern Index");
  try {
    const result = await api("/api/assessment/predict", {
      method:"POST",
      body:JSON.stringify(payload)
    });
    result.meta = result.meta || {};
    result.meta.input_source = assessmentInputSource;
    result.bundle_scope = assessmentInputSource.startsWith("demo-")
      ? "Demonstration generated from visible, label-matched research-cohort feature vectors loaded into the input workspace; modalities are not measurements from the same person."
      : "Integrated assessment generated from the user-provided feature vectors loaded in the patient input workspace.";
    lastMultimodalResult = result;
    renderMultimodalResult(result);
  } catch (e) {
    alert(`Integrated assessment failed: ${e.message}`);
  } finally {
    setLoading(false);
  }
}

function showInputFormatHelp(modality) {
  const help = {
    voice: "Voice accepts a JSON object or one-row CSV containing the 22 acoustic biomarker names used by the voice dataset. Load an example to see exact field names and values.",
    gait: "Gait accepts the derived feature vector expected by the trained 65-feature branch. Use JSON or a one-row CSV whose headers match the trained feature names. Load an example to inspect the exact schema.",
    handwriting: "Handwriting accepts the aggregated spiral + meander motor feature vector expected by the trained 36-feature branch. Use JSON or a one-row CSV. Load an example to inspect the exact schema.",
    eeg: "EEG accepts the derived resting-state spectral feature vector expected by the trained branch. Use JSON or a one-row CSV. Load an example to inspect the exact schema."
  };
  alert(help[modality] || "Provide the trained feature vector as JSON or a one-row CSV.");
}

function renderMultimodalResult(r) {
  document.getElementById("mmEmpty")?.classList.add("hidden");
  document.getElementById("mmResult")?.classList.remove("hidden");
  document.getElementById("mmScore").textContent = Number(r.assessment.multimodal_pattern_index).toFixed(1);
  document.getElementById("mmBand").textContent = r.assessment.risk_band;
  document.getElementById("mmAgreement").textContent = r.assessment.modality_agreement;
  document.getElementById("mmScope").textContent = `${r.known_label_text} demonstration \u2022 ${r.bundle_scope}`;
  document.getElementById("mmFusionNote").textContent = r.fusion_note || "";
  document.getElementById("mmInterpretation").textContent = r.assessment.interpretation || "";
  document.getElementById("mmDisclaimer").textContent = r.assessment.disclaimer || "";

  const used = r.meta?.modalities_used || Object.keys(r.branch_details || {});
  const available = r.meta?.modalities_available || ["voice","gait","handwriting","eeg"];
  document.getElementById("mmCoverage").textContent = `${used.length} / ${available.length}`;
  document.getElementById("mmSelectedNames").textContent = used.map(x => ({
    voice:"Voice", gait:"Gait", handwriting:"Handwriting", eeg:"EEG"
  }[x] || x)).join(" + ");
  document.getElementById("mmAssessmentId").textContent = r.meta?.assessment_id || "\u2014";

  const labels = {voice:"Voice", gait:"Gait", handwriting:"Handwriting", eeg:"EEG"};
  const order = ["voice","gait","handwriting","eeg"];
  const rows = order
    .filter(name => (r.branch_details || {})[name])
    .map(name => {
      const d = r.branch_details[name];
      const weight = Number((r.fusion_weights || {})[name] || 0) * 100;
      const auc = d.roc_auc == null ? "\u2014" : Number(d.roc_auc).toFixed(3);
      return `<tr>
        <td><strong>${esc(labels[name] || name)}</strong></td>
        <td>${esc(d.model_name || "\u2014")}</td>
        <td>${Number(d.score_percent).toFixed(1)}/100</td>
        <td>${auc}</td>
        <td>${weight.toFixed(1)}%</td>
      </tr>`;
    }).join("");

  document.getElementById("mmBranchBody").innerHTML = rows;
  setTimeout(() => document.getElementById("mmResult")?.scrollIntoView({behavior:"smooth", block:"start"}), 100);
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

function printUnifiedReport() {
  const r = lastMultimodalResult;
  if (!r) return;

  const labels = {voice:"Voice", gait:"Gait", handwriting:"Handwriting", eeg:"EEG"};
  const used = r.meta?.modalities_used || [];
  const detail = r.branch_details || {};
  const rows = ["voice","gait","handwriting","eeg"]
    .filter(name => detail[name])
    .map(name => {
      const d = detail[name];
      const w = Number((r.fusion_weights || {})[name] || 0) * 100;
      const auc = d.roc_auc == null ? "\u2014" : Number(d.roc_auc).toFixed(3);
      return `<tr>
        <td><strong>${esc(labels[name] || name)}</strong></td>
        <td>${esc(d.model_name || "\u2014")}</td>
        <td>${Number(d.score_percent).toFixed(1)}/100</td>
        <td>${auc}</td>
        <td>${w.toFixed(1)}%</td>
      </tr>`;
    }).join("");

  const modeText = used.length === 1
    ? `Single-signal assessment \u2022 ${esc(labels[used[0]] || used[0] || "Signal")}`
    : `${used.length}-signal fused assessment \u2022 ${esc(used.map(x => labels[x] || x).join(" + "))}`;

  const report = window.open("", "_blank");
  if (!report) {
    alert("Allow pop-ups to generate the unified assessment report.");
    return;
  }
  report.opener = null;
  report.document.write(`<!doctype html>
  <html><head><meta charset="utf-8"><title>${esc(r.meta?.assessment_id || "QuantumMed Unified Assessment")}</title>
  <style>
    @page{size:A4;margin:14mm}
    *{box-sizing:border-box}html,body{margin:0;padding:0;background:#fff;color:#16293a}
    body{font-family:Arial,Helvetica,sans-serif;font-size:11px;line-height:1.45}
    .page{max-width:780px;margin:0 auto}.brand{display:flex;justify-content:space-between;gap:18px;align-items:flex-start;border-bottom:2px solid #173c50;padding-bottom:12px}
    .brand-name{font-size:20px;font-weight:800;letter-spacing:.08em}.brand-name span{color:#16828b}.brand-meta{text-align:right;color:#6c7b87;font-size:9px}
    h1{font-size:23px;line-height:1.18;margin:18px 0 4px;color:#142a3b}.subtitle{color:#667988;margin:0 0 18px}
    .summary{display:grid;grid-template-columns:1.15fr .85fr .85fr;gap:10px;margin:16px 0}.summary-card{border:1px solid #d7e1e7;border-radius:10px;padding:13px;background:#f8fafb}
    .summary-card span,.meta-grid span{display:block;color:#71818d;font-size:8px;text-transform:uppercase;letter-spacing:.06em}.summary-card strong{display:block;margin-top:4px;font-size:19px;color:#173247}
    .meta-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px 16px;background:#f5f8f9;border:1px solid #d7e1e7;border-radius:9px;padding:10px 12px;margin-bottom:18px}.meta-grid div{display:flex;justify-content:space-between;gap:14px}.meta-grid strong{font-size:9px;text-align:right}
    .section{margin-top:18px;break-inside:avoid}.section-headline{display:flex;justify-content:space-between;align-items:baseline;border-bottom:1px solid #dce4e9;padding-bottom:6px;margin-bottom:8px}.section-headline span{font-size:9px;font-weight:800;letter-spacing:.09em;color:#31546a}.section-headline small{font-size:7px;color:#7c8992}
    table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:8px 7px;border-bottom:1px solid #e1e7eb;font-size:8px}th{background:#f2f6f8;color:#5e7180;text-transform:uppercase;letter-spacing:.04em}td{color:#243a4a}
    .voice-cards{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}.mini-card{border:1px solid #d8e2e7;border-radius:9px;padding:10px;background:#f8fafb}.mini-card span{display:block;color:#687987;font-size:8px}.mini-card strong{display:block;font-size:18px;margin:2px 0;color:#173247}.mini-card small{color:#84919a;font-size:7px}
    .interpret{border-left:4px solid #16828b;background:#f6f9fa;border-radius:8px;padding:10px 12px;color:#334a59}.scope{margin-top:10px;color:#657682;font-size:8px}.warn{margin-top:14px;border:1px solid #ead9a7;background:#fff9ea;border-radius:8px;padding:10px 12px;color:#715c22;font-size:8px}.footer{margin-top:22px;padding-top:9px;border-top:1px solid #dfe6ea;color:#7a8790;font-size:7px;display:flex;justify-content:space-between}
    @media print{body{background:#fff}.page{max-width:none}}
  </style></head><body><div class="page">
    <div class="brand"><div><div class="brand-name">QUANTUM<span>MED</span></div><div style="font-size:8px;color:#77858f">ThunderStars \u2022 Smart India Hackathon 2026</div></div><div class="brand-meta">Research screening prototype<br>${esc(formatDate(r.meta?.generated_at))}</div></div>
    <h1>QuantumMed Integrated Parkinson's Screening Assessment</h1>
    <p class="subtitle">One final result synthesized from the available biomedical signals.</p>

    <div class="summary">
      <div class="summary-card"><span>Unified Pattern Index</span><strong>${Number(r.assessment.multimodal_pattern_index).toFixed(1)} / 100</strong></div>
      <div class="summary-card"><span>Pattern</span><strong>${esc(r.assessment.risk_band)}</strong></div>
      <div class="summary-card"><span>Agreement</span><strong>${esc(r.assessment.modality_agreement)}</strong></div>
    </div>

    <div class="meta-grid">
      <div><span>Assessment ID</span><strong>${esc(r.meta?.assessment_id || "\u2014")}</strong></div>
      <div><span>Assessment mode</span><strong>${modeText}</strong></div>
      <div><span>Signals evaluated</span><strong>${esc(used.map(x => labels[x] || x).join(", "))}</strong></div>
      <div><span>Input source</span><strong>${esc(r.meta?.input_source?.startsWith("demo-") ? "Loaded cohort example" : "Patient input workspace")}</strong></div>
      <div><span>Fusion method</span><strong>${used.length > 1 ? "Validation-weighted late fusion" : "Single-branch inference"}</strong></div>
    </div>

    <section class="section">
      <div class="section-headline"><span>SUPPORTING SIGNAL EVIDENCE</span><small>These scores explain the final integrated result; they are not separate reports.</small></div>
      <table><thead><tr><th>Signal</th><th>Internal model</th><th>Signal index</th><th>Validation ROC-AUC</th><th>Contribution</th></tr></thead><tbody>${rows}</tbody></table>
    </section>


    <section class="section">
      <div class="section-headline"><span>INTERPRETATION</span></div>
      <div class="interpret">${esc(r.assessment.interpretation || "")}</div>
      <div class="scope"><strong>Demonstration scope:</strong> ${esc(r.bundle_scope || "")}</div>
    </section>

    <div class="warn"><strong>Research-use limitation:</strong> ${esc(r.assessment.disclaimer || "")}</div>
    <div class="footer"><span>QuantumMed \u2022 ThunderStars</span><span>${esc(r.meta?.assessment_id || "")}</span></div>
    <script>window.onload=()=>setTimeout(()=>window.print(),250);<\/script>
  </div></body></html>`);
  report.document.close();
}



document.addEventListener("DOMContentLoaded", async () => {
  initTheme();
  document.querySelectorAll("[data-scroll]").forEach(b => b.addEventListener("click", () => document.getElementById(b.dataset.scroll).scrollIntoView({behavior:"smooth"})));
  document.getElementById("startBtn").addEventListener("click", () => document.getElementById("multimodal").scrollIntoView({behavior:"smooth"}));
  document.getElementById("trainBtn").addEventListener("click", trainModels);
  document.getElementById("mmControlBtn")?.addEventListener("click", () => loadAssessmentExample("control"));
  document.getElementById("mmPdBtn")?.addEventListener("click", () => loadAssessmentExample("pd"));
  document.getElementById("mmClearBtn")?.addEventListener("click", clearAssessmentInputs);
  document.getElementById("mmRunBtn")?.addEventListener("click", runIntegratedAssessment);
  document.querySelectorAll('input[name="mmModality"]').forEach(el => el.addEventListener("change", updateAssessmentReadiness));
  for (const modality of ["voice","gait","handwriting","eeg"]) {
    document.getElementById(`file-${modality}`)?.addEventListener("change", async e => {
      try { await handleFeatureFile(modality, e.target.files?.[0]); }
      catch(err) { alert(`${modality} input could not be loaded: ${err.message}`); }
      finally { e.target.value = ""; }
    });
    document.getElementById(`input-${modality}`)?.addEventListener("input", () => syncEditorToInput(modality, true));
  }
  document.querySelectorAll("[data-format-help]").forEach(btn => btn.addEventListener("click", () => showInputFormatHelp(btn.dataset.formatHelp)));
  updateAssessmentReadiness();
  document.getElementById("mmReportBtn")?.addEventListener("click", printUnifiedReport);
  document.getElementById("mmJsonBtn")?.addEventListener("click", downloadMultimodalJSON);
  try {
    await loadStatus();
    await loadDataset();
    await loadBenchmarks();
  } catch(e) {
    console.error(e);
    alert(e.message);
  }
});

