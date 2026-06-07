<div align="center">

<img src="https://img.shields.io/badge/Explainable--AI--Bearing--Motor-v1.0-1a237e?style=for-the-badge&logo=gear&logoColor=white" />

# ⚙️ Explainable AI for Bearing & Motor Fault Analysis

### *Signal-to-Insight: Full ML/DL + SHAP & LIME Pipelines Across Five Industrial Fault Datasets*

<p>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/TensorFlow-DNN%7CLSTM%7CCNN-FF6F00?style=flat-square&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/XGBoost-Gradient_Boost-FF6600?style=flat-square"/>
  <img src="https://img.shields.io/badge/LightGBM-Leaf--wise_Trees-00897B?style=flat-square"/>
  <img src="https://img.shields.io/badge/SHAP-TreeExplainer-9C27B0?style=flat-square"/>
  <img src="https://img.shields.io/badge/LIME-Cylindrical_3D-E94560?style=flat-square"/>
  <img src="https://img.shields.io/badge/SMOTETomek-Class_Balance-2196F3?style=flat-square"/>
  <img src="https://img.shields.io/badge/SciPy-Signal_Processing-8BC34A?style=flat-square&logo=scipy&logoColor=white"/>
</p>

<p>
  <img src="https://img.shields.io/badge/Datasets-5_Industrial-success?style=flat-square"/>
  <img src="https://img.shields.io/badge/Fault_Types-10+-critical?style=flat-square"/>
  <img src="https://img.shields.io/badge/Models-8ML%2B3DL_per_Dataset-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/XAI-SHAP_%2B_Cylindrical_LIME-purple?style=flat-square"/>
  <img src="https://img.shields.io/badge/Institute-JIET_Jodhpur-orange?style=flat-square"/>
</p>

<br/>

> **A comprehensive, publication-quality fault diagnosis research platform** that processes raw vibration and electrical signals through full feature engineering pipelines, trains 8 ML + 3 DL models per dataset, and produces SHAP + 3D Cylindrical LIME explanations — across CWRU, IMS, Paderborn, PRONOSTIA, and Banao datasets, plus motor rotor bar fault detection.

</div>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Repository Structure](#-repository-structure)
- [Datasets](#-datasets)
- [Signal Processing & Feature Engineering](#-signal-processing--feature-engineering)
- [Models & Algorithms](#-models--algorithms)
- [Explainable AI (XAI)](#-explainable-ai-xai)
- [Class Imbalance Handling](#-class-imbalance-handling)
- [Visualization Gallery](#-visualization-gallery)
- [Installation & Usage](#-installation--usage)
- [Notebook Guide](#-notebook-guide)
- [Results Overview](#-results-overview)
- [Technical Design Decisions](#-technical-design-decisions)
- [Citation](#-citation)

---

## 🔍 Overview

Industrial machinery failures cost billions annually — but detecting them early, and *understanding why* a model flags a fault, is what separates research from deployment. **Explainable AI for Bearing & Motor Fault Analysis** addresses both:

1. **Raw signal → feature → model**: Complete pipelines from raw vibration (`.mat`, `.csv`, `.rar`) through time/frequency/envelope feature extraction to trained classifiers
2. **Explainability at every level**: SHAP TreeExplainer for global feature ranking + 3D Cylindrical LIME for per-prediction local explanations across all 11 models

The system covers bearing inner race, outer race, and roller faults; rotor bar breakage in induction motors; electrical current/voltage fault signatures; and bearing remaining useful life (RUL) prediction — from healthy to failure.

```
Raw Signals (vibration/current/voltage)
        │
        ▼
Windowing + Feature Extraction
(Time · Frequency · Envelope · Cross-channel)
        │
        ▼
Feature Selection (Fisher Score / Mutual Information)
        │
        ▼
SMOTETomek Balancing (train set only)
        │
        ▼
8 ML Models + 3 DL Models
        │
     ┌──┴──────────────────────┐
     ▼                         ▼
SHAP TreeExplainer        LIME Cylindrical 3D
(Global importance)       (Per-instance explanation)
     │                         │
     └──────────┬──────────────┘
                ▼
      Confusion Matrix · ROC/AUC · MCC · Model Comparison
```

---

## 📂 Repository Structure

```
Explainable-AI-for-Bearing-and-Motor-Fault-Analysis/
│
├── 📓 CWRU_Bearing_Fault_XAI.ipynb          ← CWRU bearing fault (8ML+3DL, SHAP+LIME)
├── 📓 IMS_Bearing_XAI.ipynb                 ← IMS run-to-failure (RAR→features, 8ML+3DL)
├── 📓 Paderborn_FINAL_v2.ipynb              ← Paderborn 32-bearing (SMOTETomek, 8ML+3DL)
├── 📓 PRONOSTIA_final_.ipynb                ← PRONOSTIA RUL regression (ML+LSTM+CNN-LSTM)
├── 📓 Bearing_Faults_XAI.ipynb              ← Inner/Outer race multi-size/load analysis
├── 📓 Bearing_broken_XAI.ipynb              ← Broken rotor bar + bearing fault (ST features)
├── 📓 Bearing_healthy_XAI.ipynb             ← Healthy baseline + bearing fault comparison
├── 📓 Motor_RotorBar_Complete_Pipeline.ipynb ← Motor rotor bar 3-class (mcsadc dataset)
├── 📓 Banao_Electrical_XAI.ipynb            ← Electrical fault (Ia,Ib,Ic,Va,Vb,Vc signals)
│
└── README.md
```

---

## 📊 Datasets

| Dataset | Signals | Classes | Features | Task | Source |
|---|---|---|---|---|---|
| **CWRU** | Vibration (pre-extracted, 48 kHz) | Multi-class fault types | 9 original + 10 engineered = **19** | Classification | Case Western Reserve University |
| **IMS** | Raw vibration 20 kHz, 3 test rigs, 8 channels | 3 health states (Healthy / Degrading / Failure) | **96** (24/channel × 4 channels) | Health state classification | NASA PCOE / University of Cincinnati |
| **Paderborn** | Raw vibration, .mat files, 32 bearings | 3 classes (Healthy / Inner Race / Outer Race) | **26** (14 Time + 9 Freq + 3 Envelope) | Classification | Paderborn University |
| **PRONOSTIA** | Raw vibration 25.6 kHz, H+V axes | RUL regression | **35** (10 Time + 7 Freq per channel × H+V) | Regression (RUL seconds) | FEMTO-ST / IEEE PHM 2012 |
| **Bearing Faults** | Vibration CSV, inner/outer race, multi-size/load | Inner Race vs Outer Race | Time + FFT features | Classification | Custom merged CSV |
| **Broken Rotor + Bearing** | Excel (auto-detect) | Multi-class (ST features) | 51 (17 ops × 3 phases) | Classification | Custom |
| **Healthy + Bearing** | Excel (auto-detect) | Healthy baseline + faults | 51 (17 ops × 3 phases) | Classification | Custom |
| **Motor Rotor Bar** | CSV, 3-phase current+voltage+speed+position | 3 classes (Healthy / 1 Broken Bar / 2 Broken Bars) | **30+** cross-channel features | Classification | mcsadc-motor-rotorbarfailure-1_2023 |
| **Banao Electrical** | Excel, Ia, Ib, Ic, Va, Vb, Vc | 5 classes (rb0–rb4) | 6 raw + 23 engineered = **29** | Classification | Banco de Dados Experimental |

### 📥 Dataset Download Links

| Dataset | Official Source |
|---|---|
| CWRU Bearing Data | [CWRU Bearing Data Center](https://engineering.case.edu/bearingdatacenter) |
| IMS Bearing | [NASA PCOE Data Repository](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/) |
| Paderborn University | [Paderborn Bearing Dataset](https://mb.uni-paderborn.de/en/kat/main-research/datacenter/bearing-datacenter/data-sets-and-download) |
| PRONOSTIA / IEEE PHM 2012 | [IEEE DataPort / FEMTO-ST](https://ieee-dataport.org/open-access/pronostia) |
| Motor Rotor Bar (mcsadc) | [Kaggle / mcsadc-motor-rotorbarfailure-1_2023](https://www.kaggle.com/datasets/uysalserkan/fault-induction-motor-dataset) |

---

## ⚡ Signal Processing & Feature Engineering

Each notebook extracts physically meaningful features from raw signals before training — no black-box end-to-end learning on raw waveforms.

### CWRU — 19 Features (Pre-extracted + Engineered)

**Original 9** (pre-extracted): `max`, `min`, `mean`, `sd`, `rms`, `skewness`, `kurtosis`, `crest`, `form`

**Engineered 10** (derived in-pipeline):

| Feature | Formula | Physical meaning |
|---|---|---|
| `kurtosis_sq` | kurtosis² | Amplified impulsiveness for early fault detection |
| `skew_abs` | \|skewness\| | Unsigned signal asymmetry |
| `impulse_fac` | max / (mean_abs + ε) | Impulse factor (ISO standard metric) |
| `kurt_crest` | kurtosis × crest | Combined impulsiveness indicator |
| `kurt_rms` | kurtosis × rms | Fault energy product |
| `rms_sq` | rms² | Mean square power |
| `crest_sq` | crest² | Squared crest factor |
| `form_sq` | form² | Squared form factor |
| `peak_power` | max² | Peak power |
| `energy_ratio` | rms / form | Energy distribution ratio |

### IMS — 96 Features (24 per channel × 4 channels)

Three feature domains extracted per channel with `scipy`:

| Domain | Features (per channel) |
|---|---|
| **Time** | RMS, Peak, Peak-to-Peak, Mean, Std, Skewness, Kurtosis, Crest Factor, Shape Factor, Impulse Factor, Clearance Factor, Mean Absolute, Variance |
| **Frequency** (FFT) | Spectral Centroid, Spectral Std, Spectral Entropy, Spectral Kurtosis, Peak Frequency, Spectral RMS, Band Energy Ratio |
| **Envelope** (Hilbert) | Envelope RMS, Envelope Kurtosis, Envelope Peak, Envelope Crest |

Health labeling is **adaptive per dataset**: RMS-based percentile thresholds computed independently for each of the 3 test rigs (no fixed threshold leakage across rigs).

### Paderborn — 26 Features per Window

| Domain | Features |
|---|---|
| **Time (14)** | RMS, Peak, P2P, Abs-Mean, Std, Skewness, Kurtosis, Crest Factor, Shape Factor, Impulse Factor, Margin Factor, Root Amplitude, Variance, Energy |
| **Frequency (9)** | Spectral Centroid, Spectral Std, Spectral Entropy, Spectral Kurtosis, Peak Freq, Spectral RMS, Spectral Mean, Band Ratio Low, Band Ratio High |
| **Envelope (3)** | Envelope RMS, Envelope Kurtosis, Envelope Crest |

Feature selection: **Mutual Information** ranking → Top 20 features for model training.

### PRONOSTIA — 35 Features (per recording, H + V axes)

| Domain | Features | Per axis |
|---|---|---|
| **Time** | RMS, Peak, P2P, Mean Abs, Std, Skewness, Kurtosis, Crest Factor, Shape Factor, Clearance | 10 |
| **Frequency** (Welch PSD) | Spectral Centroid, Spectral Std, Spectral Entropy, Spectral Kurtosis, Peak Freq, PSD Mean, RMS Freq | 7 |

Total: 17 × 2 axes (H+V) = 34 + 1 operating condition feature = **35 per recording**.  
RUL labels derived from PHM 2012 challenge ground-truth values.

### Motor Rotor Bar — 30+ Cross-Channel Features

| Category | Features |
|---|---|
| **Per-channel Time** | RMS, Peak, Std, Kurtosis, Skewness, Crest Factor, Shape Factor (7 × 8 channels) |
| **Cross-channel Electrical** | 3-phase current imbalance ratio, 3-phase voltage imbalance ratio, per-phase apparent power (A, B, C) |
| **Speed/Position** | Speed mean, Speed std (ripple — key rotor bar fault indicator), Position std |

**Key diagnostic insight**: Speed ripple (std of rotor speed) is the strongest indicator of rotor bar breakage — broken bars create torque fluctuations that modulate rotor speed at twice slip frequency.

### Banao Electrical — 29 Features (6 raw + 23 engineered)

Raw signals: **Ia, Ib, Ic** (three-phase currents) + **Va, Vb, Vc** (three-phase voltages)

Engineered (all algebraic — no FFT):

| Category | Features |
|---|---|
| **RMS approximations** | RMS per phase (×6) |
| **Phase imbalance** | Current imbalance ratio, Voltage imbalance ratio |
| **Power features** | Per-phase apparent power (Pa, Pb, Pc), Total apparent power |
| **Ratios & products** | V/I ratios, phase-angle proxies, cross-phase products |
| **Statistical** | Skewness, Kurtosis of each signal |

### Bearing Broken / Healthy — Stockwell Transform (51 Features)

**Reference**: M. Singh & A.G. Shaik, *Measurement* 131 (2019) 524–533

- **ST Transform**: Applied to 3-phase current signals
- **17 statistical operations** per phase × **3 phases** = **51 features**
- Fisher Score feature selection ranks all 51 features by discriminability
- Auto-detect file format: `.xlsx` / `.xls` / `.csv`

---

## 🤖 Models & Algorithms

### Core Classifier Suite (Applied per Dataset)

| Model | Library | Role |
|---|---|---|
| **Decision Tree** | `sklearn` | Fast interpretable baseline |
| **Random Forest** | `sklearn` | Ensemble, noise-robust |
| **XGBoost** | `xgboost` | Gradient-boosted trees, early stopping |
| **LightGBM** | `lightgbm` | Leaf-wise SOTA tabular |
| **HistGradientBoosting** | `sklearn` | Sklearn's native GBDT, handles NaN natively |
| **SVM** | `sklearn` (SVC) | Max-margin classifier, RBF kernel |
| **KNN** | `sklearn` | Distance-based, non-parametric |
| **Naive Bayes** | `sklearn` (GaussianNB) | Probabilistic baseline |

### Deep Learning Suite (Applied per Dataset)

| Model | Architecture | Input |
|---|---|---|
| **DNN** | Dense → BatchNorm → Dropout × 3 layers, Adam, EarlyStopping | Engineered feature vector |
| **1D CNN** | Conv1D → MaxPool × 3, Flatten, Dense head; data augmentation (Gaussian noise, ×3 factor) | Engineered features reshaped as 1D sequence |
| **LSTM** | LSTM(128) → LSTM(64) → Dense, ReduceLROnPlateau | Engineered features as time steps |

**For PRONOSTIA (regression)**: SVR, KNN Regressor, DT Regressor, RF Regressor, XGBoost Regressor, LightGBM Regressor, DNN, LSTM, CNN-LSTM

---

## 🔬 Explainable AI (XAI)

Every notebook produces two complementary explanation types:

### 🔵 SHAP — Global Feature Importance (TreeExplainer)

```python
explainer = shap.TreeExplainer(trained_models['LightGBM'])  # or RandomForest
shap_vals = explainer.shap_values(X_shap)

# For multi-class: mean |SHAP| across classes
mean_shap = np.abs(shap_vals).mean(axis=(0, 2))  # shape: (n_samples, n_features, n_classes)
```

- **Output formats** (vary by notebook):
  - Standard horizontal bar chart (CWRU, IMS, Banao)
  - Paderborn-style **SHAP histogram** — per-feature distribution of SHAP values per class
  - **Beeswarm plot** — individual sample SHAP values color-coded by feature magnitude

### 🟠 LIME — Local Explanations (3D Cylindrical)

The signature visualization across all notebooks — a **3D cylindrical bar chart** where each cylinder represents a feature's contribution to the prediction:

```python
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data        = X_train.values,
    feature_names        = X_train.columns.tolist(),
    class_names          = CLASS_NAMES,
    discretize_continuous= True
)
exp = lime_explainer.explain_instance(
    data_row   = sample,
    predict_fn = best_model.predict_proba,
    num_features = 10
)
```

- **Per-model**: LIME explanations generated for all 11 models (8ML + 3DL)
- **Per-class**: Separate explanations for each fault class
- **3D rendering**: Cylinder height = feature weight magnitude; color = positive (fault evidence) vs negative (counter-evidence)

### Fidelity & MCC Reporting

All notebooks compute:
- **LIME fidelity**: R² of the local linear surrogate vs. the original classifier's predictions on the neighborhood
- **MCC (Matthews Correlation Coefficient)**: Robust metric for imbalanced classes, reported alongside Accuracy / Precision / Recall / F1 / AUC

---

## ⚖️ Class Imbalance Handling

### SMOTETomek (Paderborn, Motor Rotor Bar)

```python
from imblearn.combine import SMOTETomek

smt = SMOTETomek(random_state=42)
X_train_bal, y_train_bal = smt.fit_resample(X_train, y_train)
# Applied ONLY on training set — test set untouched (no leakage)
```

SMOTETomek combines:
- **SMOTE**: Synthetic minority oversampling via interpolation between k-nearest neighbors
- **Tomek Links**: Removal of borderline majority-class samples that are nearest neighbors of minority samples

This dual strategy is more effective than SMOTE alone for bearing data, where class boundaries are often noisy.

### SMOTE-only (CWRU, IMS, Banao)

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42, k_neighbors=5)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
```

All notebooks enforce the **no-leakage rule**: balancing is applied exclusively to the training partition after the train/test split.

---

## 🎨 Visualization Gallery

Each notebook produces a complete, numbered figure set saved as `.png` at 130–150 DPI:

| Figure | Description |
|---|---|
| **Fig 1** — Class Distribution | Bar + Donut chart of fault class counts |
| **Fig 2** — Raw Signal Visualization | One file per class, all signal channels overlaid |
| **Fig 2b** — SMOTETomek Before/After | Side-by-side class balance comparison |
| **Fig 3** — Violin / Box Plots | Key feature distributions per health state |
| **Fig 4** — FFT Spectrum | Frequency domain — healthy vs fault, with bearing characteristic frequencies marked |
| **Fig 5** — RMS / Health Timeline | Run-to-failure RMS trend colored by health state |
| **Fig 5b** — Feature Correlation Heatmap | Top 15–20 features, Pearson correlation |
| **Fig 6** — Feature Importance | Fisher Score / Mutual Information ranking bar chart |
| **Fig 7** — SHAP Importance | Histogram or beeswarm + bar chart (TreeExplainer) |
| **Fig 8** — LIME 3D Cylinder | Per-instance local explanation across all models × all classes |
| **Fig 9** — Model Comparison | Grouped bar chart: Accuracy / Precision / Recall / F1 / AUC / MCC |
| **Fig 10** — Confusion Matrices | Per-model confusion matrix grid |
| **Fig 11** — ROC Curves | Macro-averaged OvR ROC for all 11 models |
| **Fig 12** — LIME Fidelity | R² fidelity of LIME surrogate across models |

---

## 🛠️ Installation & Usage

### Prerequisites

```bash
Python >= 3.10
```

### Install All Dependencies

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn \
            xgboost lightgbm tensorflow shap lime imbalanced-learn \
            openpyxl rarfile patool tqdm
```

Or use the per-notebook install cell (each notebook has one at the top):

```bash
# Most notebooks
pip install xgboost lightgbm shap lime imbalanced-learn tensorflow scikit-learn --quiet

# IMS (adds rarfile for .rar extraction)
pip install rarfile patool --quiet
# Also requires 7-Zip installed: https://www.7-zip.org/
```

### Configure Data Paths

Each notebook has a clearly marked `§ Configuration` section. Update the path variable:

```python
# CWRU
FILE_PATH = r"path/to/feature_time_48k_2048_load_1.csv"

# IMS (3 RAR archives)
RAR_PATHS = {
    1: r"path/to/1st_test.rar",
    2: r"path/to/2nd_test.rar",
    3: r"path/to/3rd_test.rar",
}

# Paderborn (.mat files, bearing folders)
DATA_ROOT = r"path/to/Paderborn/KA01/"   # parent folder with all bearing dirs

# PRONOSTIA
BASE_DIR = r"path/to/FEMTO-ST/"

# Motor Rotor Bar
DATA_ROOT = r"path/to/mcsadc-motor-rotorbarfailure-1_2023/"

# Banao Electrical
DATA_DIR = r"path/to/banao/"   # folder with struct_r1b_R1_extracted.xlsx etc.
```

### Run a Notebook

```bash
jupyter notebook CWRU_Bearing_Fault_XAI.ipynb
# or
jupyter lab
```

Each notebook is fully self-contained. Run all cells top-to-bottom — the pipeline installs → loads → processes signals → extracts features → balances → trains → evaluates → explains → saves all figures.

---

## 📋 Notebook Guide

| Notebook | Dataset | Task | Models | Feature Count | XAI Style |
|---|---|---|---|---|---|
| `CWRU_Bearing_Fault_XAI.ipynb` | CWRU | Fault classification | 8ML + DNN + 1D CNN + LSTM | 19 | SHAP bar + LIME 3D cylinder |
| `IMS_Bearing_XAI.ipynb` | IMS (3 rigs) | Health state classification | 8ML + DNN + 1D CNN + LSTM | 96 (24×4ch) | SHAP bar + Cylindrical LIME |
| `Paderborn_FINAL_v2.ipynb` | Paderborn (32 bearings) | 3-class fault classification | 8ML + DNN + LSTM + 1D CNN | 26 | SHAP histogram + beeswarm + Cylindrical LIME |
| `PRONOSTIA_final_.ipynb` | PRONOSTIA | RUL regression (seconds) | RF, XGB, LGBM, SVR, KNN, DT + DNN + LSTM + CNN-LSTM | 35 | SHAP + LIME + PHM scoring |
| `Bearing_Faults_XAI.ipynb` | Custom (inner/outer race) | Binary fault classification | RF, XGB, LightGBM, KNN, MLP, NB, DT | Time+FFT | SHAP + LIME |
| `Bearing_broken_XAI.ipynb` | Custom broken rotor+bearing | Multi-class | DT, RF, XGB, LightGBM, NB, SVM, KNN, MLP + CNN + LSTM | 51 (ST) | SHAP + LIME |
| `Bearing_healthy_XAI.ipynb` | Custom healthy+bearing | Multi-class | DT, RF, XGB, LightGBM, NB, SVM, KNN, MLP + CNN + LSTM | 51 (ST) | SHAP + LIME |
| `Motor_RotorBar_Complete_Pipeline.ipynb` | mcsadc rotor bar | 3-class (0/1/2 broken bars) | 8ML + DNN + LSTM + 1D CNN | 30+ cross-channel | SHAP histogram + Cylindrical LIME |
| `Banao_Electrical_XAI.ipynb` | Banao electrical fault | 5-class (rb0–rb4) | 8ML + DNN + LSTM | 29 (6 raw + 23 eng.) | SHAP + LIME |

---

## 📈 Results Overview

> Representative ranges observed during development. Results vary by random seed and hardware.

### CWRU Bearing Fault

| Model | Accuracy | F1 (Macro) | AUC |
|---|---|---|---|
| LightGBM | ~99.4% | ~0.99 | ~0.999 |
| Random Forest | ~99.1% | ~0.98 | ~0.999 |
| XGBoost | ~98.9% | ~0.98 | ~0.998 |
| 1D CNN | ~97.5% | ~0.96 | ~0.995 |
| SVM | ~96.2% | ~0.94 | ~0.990 |

### IMS Health State Classification (3-class)

| Model | Accuracy | F1 (Macro) | MCC |
|---|---|---|---|
| LightGBM | ~99.7% | ~0.99 | ~0.99 |
| Random Forest | ~99.5% | ~0.99 | ~0.99 |
| DNN | ~98.8% | ~0.98 | ~0.98 |
| KNN | ~97.3% | ~0.96 | ~0.96 |

### Paderborn 3-Class (All 32 Bearings — unique contribution)

| Model | Accuracy | F1 (Macro) | MCC |
|---|---|---|---|
| LightGBM | ~95.8% | ~0.94 | ~0.93 |
| HistGradientBoosting | ~95.2% | ~0.93 | ~0.92 |
| XGBoost | ~94.7% | ~0.92 | ~0.91 |
| 1D CNN | ~93.1% | ~0.90 | ~0.89 |
| Naive Bayes | ~72.4% | ~0.65 | ~0.60 |

### PRONOSTIA RUL Regression

| Model | MAE (s) | RMSE (s) | R² | PHM Score |
|---|---|---|---|---|
| CNN-LSTM | ~850 | ~1,200 | ~0.94 | Best |
| LightGBM | ~980 | ~1,450 | ~0.91 | — |
| LSTM | ~1,050 | ~1,580 | ~0.89 | — |
| SVR | ~1,800 | ~2,600 | ~0.78 | — |

### Motor Rotor Bar (3-class)

| Model | Accuracy | F1 (Macro) | MCC |
|---|---|---|---|
| LightGBM | ~97.3% | ~0.97 | ~0.96 |
| XGBoost | ~96.8% | ~0.96 | ~0.95 |
| DNN | ~95.2% | ~0.94 | ~0.93 |

---

## 🧠 Technical Design Decisions

### Why Engineered Features over Raw Waveform End-to-End?

For small-to-medium industrial datasets (hundreds to low thousands of samples per class), physics-based features dramatically outperform raw-signal CNNs — the model has a strong inductive prior (kurtosis = impulsiveness = fault energy) that data-driven models must learn from scratch with insufficient data.

### Why SMOTETomek over SMOTE Alone?

SMOTE on bearing fault data creates synthetic samples near class boundaries where real fault signatures overlap. Tomek-link removal cleans these ambiguous boundary samples from the majority class, producing cleaner decision boundaries. The Paderborn dataset particularly benefits from this: outer-race and inner-race fault signals share low-frequency envelope features.

### Why Adaptive RMS Thresholds for IMS Health Labeling?

The three IMS test rigs ran under different load conditions and failed at different times. A fixed global RMS threshold (e.g., 0.5g = degrading) misclassifies early degradation in rig 1 (which runs hotter) as failure, and late failure in rig 3 as healthy. Per-rig percentile thresholds (computed on the rig's own RMS distribution) are physically correct.

### Why Stockwell Transform for Broken Rotor / Healthy Notebooks?

The S-Transform provides time-frequency localization without a fixed window size (unlike STFT) — amplitude at each frequency is computed using a frequency-adaptive Gaussian window. For current signals where fault harmonics appear at low frequencies (sidebands around 50 Hz supply) with sharp time-domain transients, the ST gives better feature separability than pure FFT statistics.

### Why SHAP Histogram Style for Paderborn and Motor Notebooks?

For 3-class problems, the standard mean |SHAP| bar chart loses information about class-specific feature importance. The histogram format shows the full SHAP value distribution per feature per class — revealing that some features are diagnostic for outer-race faults but irrelevant for inner-race faults, which is actionable engineering knowledge.

### Why 1D CNN on Engineered Features (Not Raw Signals)?

The 1D CNN in these notebooks does not process raw waveforms. It treats the engineered feature vector as a 1D "sequence" and applies convolutional filters to learn interaction patterns between adjacent features. This is validated by the CWRU literature — engineered-feature CNNs consistently outperform raw-signal CNNs on this dataset for classification.

---

## 📊 Fault Type Taxonomy

```
Industrial Fault Taxonomy (covered across all notebooks)
│
├── 🔴 Bearing Mechanical Faults
│   ├── Inner Race (CWRU, Paderborn, Bearing_Faults, IMS)
│   ├── Outer Race (CWRU, Paderborn, Bearing_Faults, IMS)
│   ├── Roller Element (CWRU, Paderborn)
│   └── Combined / Multi-fault (Paderborn KB bearings)
│
├── 🟠 Bearing Health Progression
│   ├── Healthy (normal operation)
│   ├── Degrading (RMS increase, kurtosis spike)
│   └── Failure (run-to-failure: IMS, PRONOSTIA)
│
├── 🟡 Rotor / Motor Faults
│   ├── 1 Broken Rotor Bar (mcsadc Motor)
│   ├── 2 Broken Rotor Bars (mcsadc Motor)
│   └── Broken Rotor + Bearing Fault (combined, Bearing_broken)
│
└── 🔵 Electrical / Current-Based Faults
    ├── rb0 — Healthy reference
    ├── rb1 — Fault type 1
    ├── rb2 — Fault type 2
    ├── rb3 — Fault type 3
    └── rb4 — Fault type 4
```

---

## 📚 Citation

If you use **Explainable AI for Bearing & Motor Fault Analysis** in your research, please cite:

```bibtex
@misc{xaibearing2026,
  author       = {Golu Kumar Upadhyay},
  title        = {Explainable AI for Bearing and Motor Fault Analysis},
  year         = {2026},
  institution  = {Jodhpur Institute of Engineering and Technology (JIET), Jodhpur},
  howpublished = {\url{https://github.com/GoluKumar-Upadhyay/Explainable-AI-for-Bearing-and-Motor-Fault-Analysis}},
  note         = {Multi-dataset fault diagnosis with SHAP and Cylindrical LIME explainability}
}
```

### Key References

- Qiu, H., Lee, J., Lin, J. (2006). *Wavelet Filter-based Weak Signature Detection Method and its Application on Roller Bearing Prognostics*. Journal of Sound and Vibration 289, 1066–1090.
- Lee, J. et al. IMS, University of Cincinnati. *Bearing Data Set*. NASA PCOE Prognostic Data Repository (2007).
- Lessmeier, C. et al. (2016). *Condition Monitoring of Bearing Damage in Electromechanical Drive Systems by Using Motor Current Signals*. PHM Europe.
- Nectoux, P. et al. (2012). *PRONOSTIA: An Experimental Platform for Bearings Accelerated Degradation Tests*. IEEE PHM.
- Smith, W.A., Randall, R.B. (2015). *Rolling Element Bearing Diagnostics Using the Case Western Reserve University Data*. Mechanical Systems and Signal Processing, 64–65, 100–131.
- Singh, M., Shaik, A.G. (2019). *Faulty Bearing Detection, Classification and Location in a Three-Phase Induction Motor Based on Stockwell Transform and Support Vector Machine*. Measurement, 131, 524–533.
- Lundberg, S., Lee, S-I. (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS.
- Ribeiro, M.T. et al. (2016). *"Why Should I Trust You?" Explaining the Predictions of Any Classifier*. KDD.

---

## 🤝 Contributing

Contributions welcome! Suggested additions:

- [ ] Add `requirements.txt` and `environment.yml` per notebook
- [ ] Parameterize all data paths via `config.yaml`
- [ ] Add Dockerfile for reproducible environment
- [ ] Port PRONOSTIA pipeline to cross-bearing leave-one-out evaluation
- [ ] Add frequency-domain deep learning (1D CNN on FFT spectra)
- [ ] Extend Banao pipeline to real-time inference demo

---

## 📄 License

This project is released under the [MIT License](LICENSE).

---

<div align="center">

**Built with ⚙️ at [JIET Jodhpur](https://www.jiet.ac.in) · Affiliated Research**

*"A model that detects faults but cannot explain them is a black box in a safety-critical system — and that is unacceptable."*

<br/>

[![GitHub](https://img.shields.io/badge/GitHub-GoluKumar--Upadhyay-181717?style=for-the-badge&logo=github)](https://github.com/GoluKumar-Upadhyay/Explainable-AI-for-Bearing-and-Motor-Fault-Analysis)

</div>
