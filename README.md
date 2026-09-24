# XAutoNet: Explainable AI Clinical Assistance Model for Early Sepsis Identification

[![IEEE IRI 2023](https://img.shields.io/badge/Paper-IEEE%20IRI%202023-blue.svg)](https://doi.org/10.1109/IRI58017.2023.00059)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-teal.svg)]()

Production-grade, modular implementation and clinical decision support system based on the research paper:
> **"An Explainable AI based Clinical Assistance Model for Identifying Patients with the Onset of Sepsis"**  
> *Snehashis Chakraborty, Komal Kumar, Balakrishna Pailla Reddy, Tanushree Meena, Sudipta Roy*  
> IEEE 24th International Conference on Information Reuse and Integration for Data Science (IRI 2023)  
> DOI: [10.1109/IRI58017.2023.00059](https://doi.org/10.1109/IRI58017.2023.00059)

---

## 🏥 Clinical Background & Motivation

Sepsis is an overwhelming immune response to infection causing $\ge 11\text{ million}$ deaths annually worldwide (third-highest mortality disease in ICUs globally). Immediate initiation of broad-spectrum antibiotics significantly improves survival, demanding early detection.

Traditional ICU scoring systems (**SIRS, qSOFA, GCS**) suffer from high false alarm rates due to symptom mimicry with pneumonia and inflammatory disorders. Earlier AI models suffered from high false alarm rates ($\text{FPR} > 2\times\text{FNR}$), excess feature dimensionality (65+ features causing overfitting), and opaque "black-box" decision making.

**XAutoNet solves these challenges through:**
1. **Early 6-Hour Advance Notice**: Accurately detects sepsis onset 6 hours prior to clinical diagnosis.
2. **Minimal Feature Footprint**: Reduces input feature space to 19 raw clinical features, further compressed to **11 latent features** via a 1D CNN Autoencoder.
3. **Dual Explainability**:
   - **1D GradCAM**: Visualizes feature participation across 4 encoder layers ($EB_1$ to $EB_4$) aggregated by Discounted HeatMap ($\text{DHM}, \beta = 0.9$).
   - **DeepSHAP**: Uncovers global feature importance and provides patient-level local **waterfall attribution plots** (distinguishing contributing vs. offsetting risk factors).
4. **Benchmark Superiority**: Outperforms traditional ML models (KNN, Random Forest, XGBoost, SVM) in Recall (0.94), Precision (0.93), and F1-Score (0.93).

---

## 📐 System Architecture & Workflow

```
PhysioNet Challenge 2019 Dataset (20,336 ICU Patients)
                     │
                     ▼
           [Data Preprocessing]
  ├── MICE Imputation (Chained Equations)
  ├── Outlier Detection & Clipping (Z-Score & IQR within Clinical Bounds)
  ├── SIRS Score Feature Engineering (Vital signs & lab criteria)
  └── Class Balancing (SMOTE Minority Oversampling + Cluster Undersampling)
      └── Final Balanced Cohort: 32,000 Normal / 25,000 Sepsis Records
                     │
                     ▼
      [Stage 1: Filter Feature Selection]
  ├── Mutual Information (Top 18 Continuous Features)
  └── Chi-Square Test (Categorical: SIRS selected, p < 0.05)
      └── 19 Selected Features
                     │
                     ▼
      [Stage 2: 1D CNN Autoencoder Compression]
  ├── 4 Conv1D Encoder Layers (decreasing filters: 64 -> 32 -> 16 -> 8)
  ├── Bottleneck Latent Layer (11 dimensions)
  └── 4 Conv1D Decoder Layers (symmetric) + Dense(19) Linear Reconstruction
                     │
                     ▼
     [Stage 3: XAutoNet Classifier & Dual XAI]
  ├── Deep Neural Network Classifier on Latent 11 Features -> P'(onset in 6h)
  ├── 1D GradCAM on Encoder Layers (DHM Ranking, Beta = 0.9)
  └── DeepSHAP Waterfall & Mean SHAP Feature Attributions
                     │
                     ▼
    [Interactive Clinical Decision Support Dashboard]
```

---

## 🧪 Selected Clinical Features (19 Features)

| # | Feature | Clinical Description | Normal Reference Range |
|---|---|---|---|
| 1 | `BaseExcess` | Acid-base equilibrium | -2.0 to 2.0 mEq/L |
| 2 | `Temp` | Core body temperature | 36.1 to 37.2 °C |
| 3 | `Chloride` | Serum chloride | 96 to 106 mEq/L |
| 4 | `Hct` | Hematocrit percentage | 36 to 50 % |
| 5 | `Hgb` | Hemoglobin concentration | 12.0 to 17.5 g/dL |
| 6 | `Resp` | Respiration rate | 12 to 20 breaths/min |
| 7 | `HCO3` | Serum bicarbonate | 22 to 28 mEq/L |
| 8 | `SIRS` | Systemic Inflammatory Response Syndrome score | 0 to 1 ( $\ge 2$ threshold) |
| 9 | `Potassium` | Serum potassium | 3.5 to 5.0 mEq/L |
| 10 | `Creatinine` | Renal function biomarker | 0.6 to 1.2 mg/dL |
| 11 | `Phosphate` | Serum inorganic phosphate | 2.5 to 4.5 mg/dL |
| 12 | `FiO2` | Fraction of inspired oxygen | 0.21 to 0.50 |
| 13 | `O2Sat` | Oxygen saturation | 95 to 100 % |
| 14 | `Magnesium` | Serum magnesium | 1.7 to 2.2 mg/dL |
| 15 | `SaO2` | Arterial oxygen saturation | 95 to 100 % |
| 16 | `Lactate` | Tissue perfusion marker | 0.5 to 2.0 mmol/L |
| 17 | `pH` | Arterial blood pH | 7.35 to 7.45 |
| 18 | `Calcium` | Serum total calcium | 8.5 to 10.5 mg/dL |
| 19 | `BUN` | Blood Urea Nitrogen | 7 to 20 mg/dL |

* **Top 11 Bottleneck Features (DHM Ranking)**: `Creatinine`, `Hct`, `Phosphate`, `HCO3`, `Resp`, `Temp`, `Hgb`, `SIRS`, `Potassium`, `BaseExcess`, `Chloride`.
* **Top 5 DeepSHAP Impact Features**: `Creatinine`, `Hct`, `Phosphate`, `HCO3`, `Resp`.

---

## 🚀 Quickstart

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run Benchmarks & Pipeline Reproduction
```bash
# Run 5-fold cross-validation and baseline comparison (Table I & Table II)
python run_pipeline.py --quick
```

### 3. Launch Clinical Decision Support Web UI
```bash
python run_pipeline.py --serve --port 8000
```
Then open your browser to **`http://127.0.0.1:8000`**.

### 4. Run Automated Unit Tests
```bash
pytest tests/ -v
```

---

## 📊 Published Performance Benchmarks (IEEE IRI 2023)

### Table I: XAutoNet 5-Fold Cross Validation
| Fold | F1 Score | Precision | Recall | Accuracy |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.92 | 0.92 | 0.91 | 0.93 |
| 2 | 0.92 | 0.92 | 0.92 | 0.93 |
| 3 | 0.93 | 0.95 | 0.91 | 0.93 |
| 4 | 0.93 | 0.94 | 0.93 | 0.94 |
| 5 | 0.94 | 0.94 | 0.94 | 0.94 |
| **Mean ± SD** | **0.93 ± 0.008** | **0.93 ± 0.012** | **0.92 ± 0.012** | **0.94 ± 0.007** |

### Table II: Comparison with Traditional ML Models
| Model | Accuracy | F1 Score | Precision | Recall |
|:---|:---:|:---:|:---:|:---:|
| KNN | 0.88 | 0.85 | 0.89 | 0.78 |
| Gradient Boost | 0.89 | 0.87 | 0.88 | 0.86 |
| Random Forest | 0.89 | 0.88 | 0.89 | 0.88 |
| Naïve Bayes | 0.62 | 0.49 | 0.59 | 0.43 |
| XG Boost | 0.90 | 0.89 | 0.90 | 0.89 |
| Decision Tree | 0.86 | 0.84 | 0.85 | 0.84 |
| SVM | 0.87 | 0.86 | 0.87 | 0.86 |
| Logistic Regression | 0.64 | 0.52 | 0.62 | 0.45 |
| ADA Boost | 0.88 | 0.85 | 0.88 | 0.82 |
| **XAutoNet (Proposed)** | **0.93** | **0.92** | **0.90** | **0.94** |

---

## 🔍 Explainability Details

### 1D GradCAM & Discounting HeatMap (DHM)
To interpret the feature extraction process of the Conv1D autoencoder, gradients are computed with respect to feature maps in each encoder layer ($E_1$ to $E_4$):
$$\nabla = \frac{\partial y^{cls}}{\partial F}$$
$$H = \frac{1}{C} \sum hm_i \odot \nabla_i$$
To determine overall feature contribution into the bottleneck:
$$\text{DisHM} = \sum_{i=1}^4 \beta^i HM_i(x_1, \dots, x_{19}), \quad \beta = 0.9$$

### DeepSHAP Waterfall Attributions
DeepSHAP evaluates the local attribution of each bottleneck feature:
- **Red (+)**: Biomarkers actively driving risk upwards towards sepsis onset.
- **Blue (-)**: Protective or normal biomarkers counteracting the sepsis risk.
- Validated on real patient cases:
  - **Patient A (Normal / TN)**: 8/11 features reducing risk (Hgb -0.24, Temp -0.06).
  - **Patient B (Septic / TP)**: 6/11 features increasing risk (Temp +0.29 with 40.22°C fever, Resp +0.08 with 28 bpm tachypnea).

---

## 📂 Project Structure

```
sepsis/
├── An Explainable AI based Clinical Assistance Model for Identi.pdf  (Original Research Paper)
├── requirements.txt
├── README.md
├── run_pipeline.py                         # CLI Runner
│
├── src/
│   └── xautonet/
│       ├── __init__.py
│       ├── config.py                       # Clinical ranges, hyperparameters, feature constants
│       ├── pipeline.py                     # End-to-end training & inference pipeline
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   ├── loader.py                   # PhysioNet loader & realistic cohort generator
│       │   ├── imputation.py               # MICE imputer wrapper
│       │   ├── outliers.py                 # Clinical Z-Score & IQR outlier clipping
│       │   ├── feature_engineering.py      # SIRS criteria calculation
│       │   └── balancing.py                # Hybrid SMOTE + cluster undersampling
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   ├── feature_selector.py         # Mutual Information & Chi2 filter methods
│       │   ├── autoencoder.py              # 1D CNN Autoencoder (Conv1D Encoder & Decoder)
│       │   ├── classifier.py               # XAutoNet deep neural classifier head
│       │   └── baselines.py                # Table II ML baseline benchmark suite
│       │
│       ├── xai/
│       │   ├── __init__.py
│       │   ├── gradcam.py                  # 1D GradCAM & DHM aggregator (beta = 0.9)
│       │   └── shap_explainer.py           # DeepSHAP explainer & waterfall generator
│       │
│       ├── api/
│       │   ├── __init__.py
│       │   └── server.py                   # FastAPI REST API endpoints
│       │
│       └── web/
│           ├── index.html                  # Clinical Decision Support Dashboard
│           ├── styles.css                  # Modern dark telemetry styling
│           └── app.js                      # Reactive frontend engine
│
└── tests/
    ├── __init__.py
    ├── test_data_pipeline.py               # MICE, outliers, SIRS, SMOTE tests
    ├── test_autoencoder.py                 # Autoencoder compression & reconstruction tests
    ├── test_classifier.py                  # Classifier forward pass & metrics tests
    └── test_explainability.py              # GradCAM & DeepSHAP unit tests
```

---

## 📜 Citation
```bibtex
@inproceedings{chakraborty2023explainable,
  title={An Explainable AI based Clinical Assistance Model for Identifying Patients with the Onset of Sepsis},
  author={Chakraborty, Snehashis and Kumar, Komal and Reddy, Balakrishna Pailla and Meena, Tanushree and Roy, Sudipta},
  booktitle={2023 IEEE 24th International Conference on Information Reuse and Integration for Data Science (IRI)},
  pages={297--302},
  year={2023},
  organization={IEEE},
  doi={10.1109/IRI58017.2023.00059}
}
```
