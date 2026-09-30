let datasetInfo = null;
let lastResult = null;
let currentInputs = {};
let currentSource = "Manual entry";

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

function setLoading(on, title="Working…", text="Please wait") {
  document.getElementById("loadingTitle").textContent = title;
  document.getElementById("loadingText").textContent = text;
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
  setLoading(true, "Training hybrid QML engine…", "Participant-separated split → classical baselines → VQC → quantum kernel → validation gates");
  try {
    await api("/api/train", {method:"POST", body:JSON.stringify({epochs:8,n_qubits:4})});
    await loadStatus();
    await loadDataset();
    await loadBenchmarks();
  } catch (e) {
    alert(`Training failed: ${e.message}`);
  } finally { setLoading(false); }
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
  document.querySelectorAll("[data-scroll]").forEach(b => b.addEventListener("click", () => document.getElementById(b.dataset.scroll).scrollIntoView({behavior:"smooth"})));
  document.getElementById("startBtn").addEventListener("click", () => document.getElementById("screening").scrollIntoView({behavior:"smooth"}));
  document.getElementById("trainBtn").addEventListener("click", trainModels);
  document.getElementById("controlSampleBtn").addEventListener("click", () => loadSample("control"));
  document.getElementById("pdSampleBtn").addEventListener("click", () => loadSample("pd"));
  document.getElementById("runBtn").addEventListener("click", runAssessment);
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
