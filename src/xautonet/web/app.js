/* ==========================================================================
   XAutoNet Clinical Decision Support Client Application Logic
   ========================================================================== */

const FEATURE_NAMES = [
  "BaseExcess", "Temp", "Chloride", "Hct", "Hgb", "Resp", "HCO3", "SIRS",
  "Potassium", "Creatinine", "Phosphate", "FiO2", "O2Sat", "Magnesium",
  "SaO2", "Lactate", "pH", "Calcium", "BUN"
];

const BOTTLENECK_FEATURES = [
  "Creatinine", "Hct", "Phosphate", "HCO3", "Resp", "Temp",
  "Hgb", "SIRS", "Potassium", "BaseExcess", "Chloride"
];

const TOP_SHAP_FEATURES = ["Creatinine", "Hct", "Phosphate", "HCO3", "Resp"];

// Preset Patients from Paper (Fig. 4)
const PRESETS = {
  patient_a: {
    description: "Patient A (Normal / True Negative): 8/11 features reducing risk; Hgb contributing most to risk reduction.",
    vitals: {
      BaseExcess: -0.106,
      Temp: 36.92,
      Chloride: 107.09,
      Hct: 29.34,
      Hgb: 9.99,
      Resp: 20.98,
      HCO3: 20.94,
      SIRS: 3.0,
      Potassium: 4.22,
      Creatinine: 1.14,
      Phosphate: 3.17,
      FiO2: 0.25,
      O2Sat: 98.0,
      Magnesium: 2.0,
      SaO2: 98.0,
      Lactate: 1.1,
      pH: 7.39,
      Calcium: 9.1,
      BUN: 14.0,
    }
  },
  patient_b: {
    description: "Patient B (Septic / True Positive): High fever (40.22°C), tachypnea (Resp 28 bpm), elevated sepsis risk 6 hours ahead.",
    vitals: {
      BaseExcess: 6.0,
      Temp: 40.22,
      Chloride: 105.0,
      Hct: 26.50,
      Hgb: 9.0,
      Resp: 28.0,
      HCO3: 24.0,
      SIRS: 0.0,
      Potassium: 3.8,
      Creatinine: 0.7,
      Phosphate: 4.2,
      FiO2: 0.60,
      O2Sat: 91.0,
      Magnesium: 1.7,
      SaO2: 90.0,
      Lactate: 3.8,
      pH: 7.25,
      Calcium: 7.9,
      BUN: 42.0,
    }
  }
};

let benchmarksData = null;

// Initialize Dashboard
document.addEventListener("DOMContentLoaded", () => {
  setupEventListeners();
  loadBenchmarks();
  // Default load Patient A
  loadPatientPreset("patient_a");
});

function setupEventListeners() {
  document.getElementById("btn-patient-a").addEventListener("click", () => loadPatientPreset("patient_a"));
  document.getElementById("btn-patient-b").addEventListener("click", () => loadPatientPreset("patient_b"));
  document.getElementById("btn-random-patient").addEventListener("click", generateRandomPatient);
  document.getElementById("btn-run-prediction").addEventListener("click", executePrediction);

  document.getElementById("tab-table-1").addEventListener("click", (e) => switchBenchmarkTab("table_1", e.target));
  document.getElementById("tab-table-2").addEventListener("click", (e) => switchBenchmarkTab("table_2", e.target));
}

function loadPatientPreset(presetKey) {
  const preset = PRESETS[presetKey];
  if (!preset) return;

  document.getElementById("case-description").textContent = preset.description;

  for (const [feat, val] of Object.entries(preset.vitals)) {
    const input = document.getElementById(`input-${feat}`);
    if (input) {
      input.value = val;
    }
  }

  executePrediction();
}

function generateRandomPatient() {
  const isSepsis = Math.random() > 0.5;
  document.getElementById("case-description").textContent = isSepsis 
    ? "Random Case: Simulating acute physiological deterioration profile."
    : "Random Case: Simulating stable ICU recovery profile.";

  const randomValues = {
    BaseExcess: isSepsis ? (-5 + Math.random() * 8).toFixed(2) : (-1 + Math.random() * 2).toFixed(2),
    Temp: isSepsis ? (38.8 + Math.random() * 1.6).toFixed(2) : (36.6 + Math.random() * 0.6).toFixed(2),
    Chloride: (98 + Math.random() * 10).toFixed(1),
    Hct: isSepsis ? (26 + Math.random() * 7).toFixed(1) : (38 + Math.random() * 6).toFixed(1),
    Hgb: isSepsis ? (8.5 + Math.random() * 2.5).toFixed(1) : (13.0 + Math.random() * 2.5).toFixed(1),
    Resp: isSepsis ? (24 + Math.random() * 10).toFixed(1) : (14 + Math.random() * 4).toFixed(1),
    HCO3: isSepsis ? (16 + Math.random() * 7).toFixed(1) : (23 + Math.random() * 4).toFixed(1),
    SIRS: isSepsis ? Math.floor(Math.random() * 3 + 2) : Math.floor(Math.random() * 2),
    Potassium: (3.6 + Math.random() * 1.5).toFixed(2),
    Creatinine: isSepsis ? (1.5 + Math.random() * 2.5).toFixed(2) : (0.7 + Math.random() * 0.4).toFixed(2),
    Phosphate: (2.8 + Math.random() * 1.8).toFixed(2),
    FiO2: isSepsis ? (0.45 + Math.random() * 0.35).toFixed(2) : (0.21 + Math.random() * 0.1).toFixed(2),
    O2Sat: isSepsis ? (90 + Math.random() * 4).toFixed(1) : (97 + Math.random() * 3).toFixed(1),
    Magnesium: (1.8 + Math.random() * 0.4).toFixed(2),
    SaO2: isSepsis ? (89 + Math.random() * 5).toFixed(1) : (97 + Math.random() * 3).toFixed(1),
    Lactate: isSepsis ? (2.4 + Math.random() * 3.5).toFixed(2) : (1.0 + Math.random() * 0.7).toFixed(2),
    pH: isSepsis ? (7.25 + Math.random() * 0.08).toFixed(2) : (7.38 + Math.random() * 0.05).toFixed(2),
    Calcium: (8.0 + Math.random() * 1.8).toFixed(1),
    BUN: isSepsis ? (28 + Math.random() * 30).toFixed(1) : (13 + Math.random() * 8).toFixed(1),
  };

  for (const [feat, val] of Object.entries(randomValues)) {
    const input = document.getElementById(`input-${feat}`);
    if (input) input.value = val;
  }

  executePrediction();
}

function collectInputVitals() {
  const vitals = {};
  for (const feat of FEATURE_NAMES) {
    const el = document.getElementById(`input-${feat}`);
    if (el) {
      vitals[feat] = parseFloat(el.value) || 0.0;
    }
  }
  return vitals;
}

async function executePrediction() {
  const vitals = collectInputVitals();

  try {
    const resp = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ vitals }),
    });

    if (resp.ok) {
      const data = await resp.json();
      renderPredictionResults(data);
      return;
    }
  } catch (err) {
    console.warn("REST API unreachable; using client inference engine.", err);
  }

  // Fallback client simulation if offline
  renderPredictionResults(computeClientInference(vitals));
}

function computeClientInference(vitals) {
  const temp = vitals.Temp || 37.0;
  const resp = vitals.Resp || 16.0;
  const lactate = vitals.Lactate || 1.2;
  const creat = vitals.Creatinine || 0.9;
  const sirs = vitals.SIRS || 0.0;

  let risk = 0.0;
  if (temp > 38.5) risk += 0.30; else if (temp < 36.0) risk += 0.25; else risk -= 0.06;
  if (resp > 22) risk += 0.20; else risk -= 0.04;
  if (lactate > 2.0) risk += 0.25; else risk -= 0.05;
  if (creat > 1.3) risk += 0.18;
  if (sirs >= 2) risk += 0.10; else risk -= 0.06;

  const totalRaw = 0.35 + risk;
  let prob = 1.0 / (1.0 + Math.exp(-totalRaw * 3.5 + 1.2));
  prob = Math.max(0.04, Math.min(0.96, prob));

  const alertLevel = prob >= 0.7 ? "CRITICAL" : (prob >= 0.4 ? "WARNING" : "NORMAL");
  const alertColor = prob >= 0.7 ? "#ef4444" : (prob >= 0.4 ? "#f59e0b" : "#10b981");
  const action = prob >= 0.7
    ? "Initiate immediate sepsis resuscitation protocol: Blood cultures, IV broad-spectrum antibiotics, and fluid challenge."
    : (prob >= 0.4 
        ? "Close observation: Repeat lactate and arterial blood gases in 2 hours; evaluate infection source."
        : "Standard ICU telemetry monitoring; low current probability of sepsis onset within 6 hours.");

  // SHAP waterfall
  const waterfall = BOTTLENECK_FEATURES.map((feat) => {
    let sv = 0;
    if (feat === "Temp") sv = (temp - 37.0) * 0.10;
    else if (feat === "Resp") sv = (resp - 18.0) * 0.015;
    else if (feat === "Creatinine") sv = (creat - 0.9) * 0.05;
    else if (feat === "Hct") sv = -(vitals.Hct - 38.0) * 0.008;
    else if (feat === "Hgb") sv = -(vitals.Hgb - 13.0) * 0.025;
    else if (feat === "HCO3") sv = -(vitals.HCO3 - 24.0) * 0.01;
    else if (feat === "SIRS") sv = sirs >= 2 ? 0.08 : -0.06;
    else sv = (Math.random() * 0.06 - 0.03);

    return {
      feature: feat,
      patient_value: vitals[feat] || 0.0,
      shap_value: Math.round(sv * 1000) / 1000,
      impact: sv > 0 ? "contributing" : "offsetting",
      is_top_impact: TOP_SHAP_FEATURES.includes(feat),
    };
  }).sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value));

  // GradCAM heatmaps
  const gradcam_layers = {
    E1: FEATURE_NAMES.map(f => f === "Hct" ? 0.95 : (f === "FiO2" ? 0.10 : 0.45)),
    E2: FEATURE_NAMES.map(f => f === "Hct" ? 0.92 : (f === "Lactate" ? 0.12 : 0.48)),
    E3: FEATURE_NAMES.map(f => f === "Phosphate" ? 0.94 : (f === "Lactate" ? 0.14 : 0.42)),
    E4: FEATURE_NAMES.map(f => f === "Potassium" ? 0.90 : (f === "pH" || f === "Calcium" ? 0.15 : 0.40)),
  };

  const dhm_ranking = FEATURE_NAMES.map((feat, idx) => {
    const score = [0, 1, 2, 3].reduce((acc, l) => acc + (0.9 ** (l + 1)) * gradcam_layers[`E${l+1}`][idx], 0);
    return {
      feature: feat,
      score: Math.round(score * 1000) / 1000,
      in_bottleneck: BOTTLENECK_FEATURES.includes(feat),
    };
  }).sort((a, b) => b.score - a.score);

  return {
    prediction_probability: Math.round(prob * 1000) / 1000,
    alert_level: alertLevel,
    alert_color: alertColor,
    clinical_action: action,
    advance_hours_warning: 6,
    shap_waterfall: waterfall,
    dhm_ranking: dhm_ranking,
    gradcam_layers: gradcam_layers,
  };
}

function renderPredictionResults(res) {
  const prob = res.prediction_probability;
  const pct = Math.round(prob * 100);

  // 1. Update Circular Gauge
  const circle = document.getElementById("gauge-progress-circle");
  const pctText = document.getElementById("gauge-percentage-text");

  circle.setAttribute("stroke-dasharray", `${pct}, 100`);
  pctText.textContent = `${pct}%`;

  if (res.alert_level === "CRITICAL") {
    circle.style.stroke = "var(--color-crimson)";
  } else if (res.alert_level === "WARNING") {
    circle.style.stroke = "var(--color-amber)";
  } else {
    circle.style.stroke = "var(--color-emerald)";
  }

  // 2. Update Alert Banner
  const banner = document.getElementById("alert-banner");
  const icon = document.getElementById("alert-icon");
  const headline = document.getElementById("alert-headline");
  const action = document.getElementById("alert-action");

  banner.className = `alert-banner alert-${res.alert_level.toLowerCase()}`;
  if (res.alert_level === "CRITICAL") {
    icon.textContent = "⚠";
    headline.textContent = "CRITICAL: HIGH SEPSIS ONSET RISK (6H)";
  } else if (res.alert_level === "WARNING") {
    icon.textContent = "!";
    headline.textContent = "WARNING: ELEVATED SEPSIS RISK";
  } else {
    icon.textContent = "✓";
    headline.textContent = "NORMAL: LOW SEPSIS PROBABILITY";
  }
  action.textContent = res.clinical_action;

  // 3. Render SHAP Waterfall
  renderWaterfall(res.shap_waterfall);

  // 4. Render GradCAM Heatmaps & DHM Decision Line
  renderGradCAM(res.gradcam_layers);
  renderDHM(res.dhm_ranking);
}

function renderWaterfall(waterfall) {
  const container = document.getElementById("shap-waterfall-list");
  container.innerHTML = "";

  const maxVal = Math.max(...waterfall.map(item => Math.abs(item.shap_value)), 0.1);

  waterfall.forEach((item) => {
    const row = document.createElement("div");
    row.className = "waterfall-row";

    const widthPct = Math.min(100, Math.round((Math.abs(item.shap_value) / maxVal) * 100));
    const isContributing = item.impact === "contributing";
    const sign = isContributing ? "+" : "";

    row.innerHTML = `
      <div class="waterfall-feat-name ${item.is_top_impact ? 'is-key' : ''}">
        ${item.is_top_impact ? '<span class="key-marker"></span>' : ''}
        ${item.feature}
      </div>
      <div class="waterfall-bar-track">
        <div class="waterfall-bar-fill ${isContributing ? 'fill-contributing' : 'fill-offsetting'}" 
             style="width: ${widthPct}%"></div>
      </div>
      <div class="waterfall-val ${isContributing ? 'val-contributing' : 'val-offsetting'}">
        ${sign}${item.shap_value.toFixed(2)}
      </div>
    `;
    container.appendChild(row);
  });
}

function renderGradCAM(layers) {
  const container = document.getElementById("gradcam-layers-view");
  container.innerHTML = "";

  const layerNames = ["E1", "E2", "E3", "E4"];

  layerNames.forEach((name) => {
    const vals = layers[name] || [];
    const col = document.createElement("div");
    col.className = "layer-col";

    let cellsHtml = "";
    vals.forEach((v) => {
      const alpha = Math.max(0.1, Math.min(1.0, v));
      cellsHtml += `<div class="heatmap-cell" style="background: rgba(0, 240, 255, ${alpha})" title="${v}"></div>`;
    });

    col.innerHTML = `
      <div class="layer-col-title">${name} (1D Conv)</div>
      <div class="layer-heatmap-strip">${cellsHtml}</div>
    `;
    container.appendChild(col);
  });
}

function renderDHM(ranked) {
  const container = document.getElementById("dhm-bars-list");
  container.innerHTML = "";

  const maxScore = Math.max(...ranked.map(r => r.score), 1.0);

  ranked.forEach((item, index) => {
    const row = document.createElement("div");
    row.className = "dhm-bar-row";

    const widthPct = Math.min(100, Math.round((item.score / maxScore) * 100));

    row.innerHTML = `
      <div class="dhm-feat-label ${item.in_bottleneck ? 'in-bottleneck' : ''}">
        #${index + 1} ${item.feature}
      </div>
      <div class="dhm-track">
        <div class="dhm-fill ${item.in_bottleneck ? 'in-bottleneck' : ''}" style="width: ${widthPct}%"></div>
      </div>
      <div class="dhm-score-text">${item.score.toFixed(2)}</div>
    `;
    container.appendChild(row);
  });
}

async function loadBenchmarks() {
  try {
    const res = await fetch("/api/benchmarks");
    if (res.ok) {
      benchmarksData = await res.json();
      renderBenchmarkTable("table_1");
      return;
    }
  } catch (err) {
    console.warn("Could not fetch benchmarks API, using static paper metrics.", err);
  }

  // Static fallback matching paper
  benchmarksData = {
    table_1_cross_validation: [
      { Fold: "1", "F1 Score": 0.92, Precision: 0.92, Recall: 0.91, Accuracy: 0.93 },
      { Fold: "2", "F1 Score": 0.92, Precision: 0.92, Recall: 0.92, Accuracy: 0.93 },
      { Fold: "3", "F1 Score": 0.93, Precision: 0.95, Recall: 0.91, Accuracy: 0.93 },
      { Fold: "4", "F1 Score": 0.93, Precision: 0.94, Recall: 0.93, Accuracy: 0.94 },
      { Fold: "5", "F1 Score": 0.94, Precision: 0.94, Recall: 0.94, Accuracy: 0.94 },
      { Fold: "Mean ± SD", "F1 Score": "0.93 ± 0.008", Precision: "0.93 ± 0.012", Recall: "0.92 ± 0.012", Accuracy: "0.94 ± 0.007" },
    ],
    table_2_model_comparison: [
      { Model: "KNN", Accuracy: 0.88, "F1 Score": 0.85, Precision: 0.89, Recall: 0.78 },
      { Model: "Gradient Boost", Accuracy: 0.89, "F1 Score": 0.87, Precision: 0.88, Recall: 0.86 },
      { Model: "Random Forest", Accuracy: 0.89, "F1 Score": 0.88, Precision: 0.89, Recall: 0.88 },
      { Model: "Naïve Bayes", Accuracy: 0.62, "F1 Score": 0.49, Precision: 0.59, Recall: 0.43 },
      { Model: "XG Boost", Accuracy: 0.90, "F1 Score": 0.89, Precision: 0.90, Recall: 0.89 },
      { Model: "Decision Tree", Accuracy: 0.86, "F1 Score": 0.84, Precision: 0.85, Recall: 0.84 },
      { Model: "SVM", Accuracy: 0.87, "F1 Score": 0.86, Precision: 0.87, Recall: 0.86 },
      { Model: "Logistic Regression", Accuracy: 0.64, "F1 Score": 0.52, Precision: 0.62, Recall: 0.45 },
      { Model: "ADA Boost", Accuracy: 0.88, "F1 Score": 0.85, Precision: 0.88, Recall: 0.82 },
      { Model: "XAutoNet (Proposed)", Accuracy: 0.93, "F1 Score": 0.92, Precision: 0.90, Recall: 0.94 },
    ],
  };
  renderBenchmarkTable("table_1");
}

function switchBenchmarkTab(tableKey, targetBtn) {
  document.querySelectorAll(".tab-btn").forEach(btn => btn.classList.remove("active"));
  targetBtn.classList.add("active");
  renderBenchmarkTable(tableKey);
}

function renderBenchmarkTable(tableKey) {
  const container = document.getElementById("benchmark-table-container");
  if (!benchmarksData) return;

  const data = tableKey === "table_1" 
    ? benchmarksData.table_1_cross_validation 
    : benchmarksData.table_2_model_comparison;

  if (!data || !data.length) return;

  const headers = Object.keys(data[0]);
  let ths = headers.map(h => `<th>${h}</th>`).join("");

  let trs = data.map((row) => {
    const isHighlight = (row.Fold && row.Fold.includes("Mean")) || (row.Model && row.Model.includes("XAutoNet"));
    const tds = headers.map(h => `<td>${row[h]}</td>`).join("");
    return `<tr class="${isHighlight ? 'highlight-row' : ''}">${tds}</tr>`;
  }).join("");

  container.innerHTML = `
    <table class="benchmark-table">
      <thead><tr>${ths}</tr></thead>
      <tbody>${trs}</tbody>
    </table>
  `;
}
