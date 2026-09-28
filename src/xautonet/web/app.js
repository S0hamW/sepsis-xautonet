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

const PRESETS = {
  patient_a: {
    description: "Patient A (Normal Profile): Afebrile (36.92°C), normal acid-base and lactate. Routine monitoring indicated.",
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
    description: "Patient B (Septic Profile): High fever (40.22°C), tachypnea (Resp 28 bpm), elevated base excess (+6.0). Elevated risk of sepsis onset within 6 hours.",
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
  },
  patient_healthy: {
    description: "Healthy ICU Baseline: All 19 biomarkers set at physiological midpoints of clinical normal ranges.",
    vitals: {
      BaseExcess: 0.0,
      Temp: 36.65,
      Chloride: 101.0,
      Hct: 43.0,
      Hgb: 14.75,
      Resp: 16.0,
      HCO3: 25.0,
      SIRS: 0.0,
      Potassium: 4.25,
      Creatinine: 0.9,
      Phosphate: 3.5,
      FiO2: 0.355,
      O2Sat: 97.5,
      Magnesium: 1.95,
      SaO2: 97.5,
      Lactate: 1.25,
      pH: 7.40,
      Calcium: 9.5,
      BUN: 13.5,
    }
  }
};

document.addEventListener("DOMContentLoaded", () => {
  setupEventListeners();
  loadPatientPreset("patient_a");
});

function setupEventListeners() {
  const btnA = document.getElementById("btn-patient-a");
  const btnB = document.getElementById("btn-patient-b");
  const btnH = document.getElementById("btn-patient-healthy");
  const btnPredict = document.getElementById("btn-run-prediction");

  if (btnA) btnA.addEventListener("click", () => loadPatientPreset("patient_a"));
  if (btnB) btnB.addEventListener("click", () => loadPatientPreset("patient_b"));
  if (btnH) btnH.addEventListener("click", () => loadPatientPreset("patient_healthy"));
  if (btnPredict) btnPredict.addEventListener("click", executePrediction);
}

function loadPatientPreset(presetKey) {
  const preset = PRESETS[presetKey];
  if (!preset) return;

  const descEl = document.getElementById("case-description");
  if (descEl) descEl.textContent = preset.description;

  for (const [feat, val] of Object.entries(preset.vitals)) {
    const input = document.getElementById(`input-${feat}`);
    if (input) input.value = val;
  }

  executePrediction();
}

function collectInputVitals() {
  const vitals = {};
  for (const feat of FEATURE_NAMES) {
    const el = document.getElementById(`input-${feat}`);
    if (el && el.value.trim() !== "") {
      vitals[feat] = parseFloat(el.value);
    } else {
      vitals[feat] = null;
    }
  }
  return vitals;
}

async function executePrediction() {
  const vitals = collectInputVitals();
  const btnPredict = document.getElementById("btn-run-prediction");
  if (btnPredict) {
    btnPredict.disabled = true;
    btnPredict.textContent = "Analyzing...";
  }

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
    console.warn("Backend API call failed, falling back to local client model.", err);
  } finally {
    if (btnPredict) {
      btnPredict.disabled = false;
      btnPredict.innerHTML = `<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="5 3 19 12 5 21 5 3" /></svg> Analyze Risk`;
    }
  }

  // Fallback client calculation if offline
  renderPredictionResults(computeFallbackInference(vitals));
}

function computeFallbackInference(vitals) {
  const temp = vitals.Temp !== null ? vitals.Temp : 36.65;
  const resp = vitals.Resp !== null ? vitals.Resp : 16.0;
  const sirs = vitals.SIRS !== null ? vitals.SIRS : 0.0;
  const be = vitals.BaseExcess !== null ? vitals.BaseExcess : 0.0;
  const creat = vitals.Creatinine !== null ? vitals.Creatinine : 0.9;

  let logit = -3.5;
  if (temp > 38.3) logit += 2.0; else if (temp < 36.0) logit += 1.5;
  if (resp > 22) logit += 1.2;
  if (sirs >= 2) logit += 0.8;
  if (be > 3.0 || be < -3.0) logit += 0.9;
  if (creat > 1.3) logit += 0.8;

  const prob = 1.0 / (1.0 + Math.exp(-logit));
  const isHigh = prob >= 0.30;

  return {
    prediction_probability: Math.round(prob * 1000) / 1000,
    risk_classification: prob >= 0.60 ? "High Risk" : (prob >= 0.30 ? "Moderate Risk" : "Low Risk"),
    alert_level: prob >= 0.60 ? "HIGH" : (prob >= 0.30 ? "MODERATE" : "LOW"),
    decision_threshold: 0.30,
    is_above_threshold: isHigh,
    interpretation_6h: isHigh
      ? `Elevated risk of sepsis onset within the next 6 hours (probability ${(prob*100).toFixed(1)}% exceeds 30% threshold). Close clinical observation indicated.`
      : `Low risk of sepsis onset within the next 6 hours (probability ${(prob*100).toFixed(1)}% is below 30% threshold). Routine ICU monitoring indicated.`,
    warnings: ["Offline mode: using local estimator."],
    shap_waterfall: BOTTLENECK_FEATURES.map(f => ({
      feature: f,
      patient_value: vitals[f] !== null ? vitals[f] : 0.0,
      shap_value: f === "Temp" && temp > 38 ? 0.35 : (f === "Resp" && resp > 22 ? 0.18 : 0.02),
      impact: (f === "Temp" && temp > 38) || (f === "Resp" && resp > 22) ? "contributing" : "offsetting",
      is_top_impact: ["Temp", "Resp", "Creatinine"].includes(f)
    }))
  };
}

function renderPredictionResults(res) {
  const prob = res.prediction_probability;
  const pct = Math.round(prob * 100);

  // 1. Gauge
  const circle = document.getElementById("gauge-progress-circle");
  const pctText = document.getElementById("gauge-percentage-text");

  if (circle) circle.setAttribute("stroke-dasharray", `${pct}, 100`);
  if (pctText) pctText.textContent = `${pct}%`;

  let alertClass = "alert-normal";
  if (res.alert_level === "HIGH") {
    alertClass = "alert-critical";
    if (circle) circle.style.stroke = "var(--color-crimson)";
  } else if (res.alert_level === "MODERATE") {
    alertClass = "alert-warning";
    if (circle) circle.style.stroke = "var(--color-amber)";
  } else {
    alertClass = "alert-normal";
    if (circle) circle.style.stroke = "var(--color-emerald)";
  }

  // 2. Alert Banner
  const banner = document.getElementById("alert-banner");
  const icon = document.getElementById("alert-icon");
  const headline = document.getElementById("alert-headline");
  const action = document.getElementById("alert-action");

  if (banner) banner.className = `alert-banner ${alertClass}`;
  if (icon) {
    icon.textContent = res.alert_level === "HIGH" ? "⚠" : (res.alert_level === "MODERATE" ? "!" : "✓");
  }
  if (headline) {
    headline.textContent = `${res.risk_classification.toUpperCase()} (${pct}%)`;
  }
  if (action) {
    action.textContent = res.interpretation_6h;
  }

  // 3. Threshold Badge
  const thBadge = document.getElementById("threshold-status-badge");
  if (thBadge) {
    if (res.is_above_threshold) {
      thBadge.textContent = "ELEVATED (≥0.30)";
      thBadge.style.background = "rgba(239, 68, 68, 0.2)";
      thBadge.style.color = "var(--color-crimson)";
    } else {
      thBadge.textContent = "BELOW THRESHOLD (<0.30)";
      thBadge.style.background = "rgba(16, 185, 129, 0.2)";
      thBadge.style.color = "var(--color-emerald)";
    }
  }

  // 4. Warnings
  const warnCard = document.getElementById("warnings-card");
  const warnList = document.getElementById("warnings-list");
  if (warnCard && warnList) {
    if (res.warnings && res.warnings.length > 0) {
      warnList.innerHTML = res.warnings.map(w => `<li>${w}</li>`).join("");
      warnCard.style.display = "block";
    } else {
      warnCard.style.display = "none";
    }
  }

  // 5. SHAP Waterfall
  renderWaterfall(res.shap_waterfall || []);
}

function renderWaterfall(waterfall) {
  const container = document.getElementById("shap-waterfall-list");
  if (!container) return;
  container.innerHTML = "";

  if (waterfall.length === 0) {
    container.innerHTML = `<div style="color: var(--text-muted); font-size: 0.85rem; padding: 1rem;">No feature attribution available.</div>`;
    return;
  }

  const maxVal = Math.max(...waterfall.map(item => Math.abs(item.shap_value)), 0.05);

  waterfall.forEach((item) => {
    const isContributing = item.impact === "contributing";
    const absVal = Math.abs(item.shap_value);
    const barWidthPct = Math.min(100, Math.round((absVal / maxVal) * 100));

    const row = document.createElement("div");
    row.className = `waterfall-item ${isContributing ? "item-contributing" : "item-offsetting"}`;

    row.innerHTML = `
      <div class="waterfall-meta">
        <span class="feat-name">${item.feature}</span>
        <span class="feat-patient-val">${item.patient_value !== undefined ? item.patient_value : "--"}</span>
      </div>
      <div class="waterfall-bar-track">
        <div class="waterfall-bar ${isContributing ? "bar-positive" : "bar-negative"}" style="width: ${barWidthPct}%;"></div>
      </div>
      <div class="waterfall-val">
        ${item.shap_value > 0 ? "+" : ""}${item.shap_value.toFixed(3)}
      </div>
    `;

    container.appendChild(row);
  });
}
