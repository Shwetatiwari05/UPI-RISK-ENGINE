<div align="center">

# 🔐 UPI Fraud Detection System
### Hybrid Anomaly Detection for Identifying Ambiguous UPI Transactions

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1-189AB4?style=for-the-badge)](https://xgboost.readthedocs.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-Vite-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://vitejs.dev)
[![License](https://img.shields.io/badge/License-Academic%20Research-blueviolet?style=for-the-badge)](LICENSE)

> **B.Tech Final Year Major Project** — An end-to-end offline machine learning research system that combines supervised fraud detection and unsupervised anomaly detection with a transparent explainable fusion layer, applied to 7.2 million real UPI transaction records.

</div>

---

## 📌 Project Highlights

| Metric | Value |
|---|---|
| 📊 **Total Records Processed** | 7,218,220 transactions |
| 🎯 **XGBoost ROC-AUC (best, out-of-time)** | **0.941** |
| 🎯 **Random Forest ROC-AUC (out-of-time)** | 0.932 |
| 🎯 **FRAUD_LIKELY precision (tuned)** | 40.0% at ~30 alerts / 10k rows |
| 🔍 **Review band coverage (tuned)** | 10% of traffic · catches 57% of remaining fraud |
| 📁 **Datasets Used** | 4 public datasets (Kaggle + Zenodo) |
| 🧠 **Models Trained** | XGBoost, Random Forest, Isolation Forest, LOF |
| 🖥️ **Frontend** | React + Vite dashboard with 5 pages |

---

## 🧠 What This Project Does

This system performs **post-transaction fraud analysis** on UPI (Unified Payments Interface) transactions using a **hybrid dual-signal approach**:

```
Raw Transactions (7.2M rows)
        │
        ▼
┌───────────────────────┐      ┌─────────────────────────┐
│  SUPERVISED LEARNING  │      │  UNSUPERVISED LEARNING  │
│  XGBoost + RF         │      │  Isolation Forest + LOF │
│  Fraud Probability    │      │  Anomaly Percentile     │
└──────────┬────────────┘      └────────────┬────────────┘
           │                                │
           └──────────┬─────────────────────┘
                      ▼
         ┌────────────────────────┐
         │   FUSION LAYER         │
         │  Weighted Score +      │
         │  Disagreement Band +   │
         │  Uncertainty Penalty   │
         └────────────┬───────────┘
                      ▼
         ┌────────────────────────┐
         │   RESEARCH RESOLUTION  │
         │  LIKELY_LEGITIMATE     │
         │  AMBIGUOUS_REVIEW   ◀─── Novel contribution
         │  FRAUD_LIKELY          │
         └────────────────────────┘
```

The `AMBIGUOUS_REVIEW` band is the **core research contribution** — identifying transactions where supervised and unsupervised models disagree, which traditional binary classifiers miss entirely.

---

## 🏗️ Architecture & Tech Stack

```
upi-fraud-detection/
├── src/                        # Core ML pipeline modules
│   ├── schema_mapping.py       # Multi-dataset schema normalization
│   ├── data_preprocessing.py   # RobustScaler + OneHot encoding
│   ├── feature_engineering.py  # 15+ behavioral features
│   ├── supervised_model.py     # XGBoost + Random Forest training
│   ├── anomaly_detection.py    # Isolation Forest + LOF training
│   ├── fusion_model.py         # Dual-signal fusion layer
│   └── parquet_pipeline.py     # Chunked Parquet ingestion (7.2M rows)
├── api/
│   └── main.py                 # FastAPI REST backend (6 endpoints)
├── app/
│   └── prediction_engine.py    # Model inference engine
├── frontend/                   # React + Vite dashboard
│   └── src/main.jsx            # 5-page interactive UI
├── models/                     # Trained model artifacts (.pkl)
├── reports/                    # Auto-generated charts & metrics
├── data/
│   ├── raw/                    # Input CSVs (not committed)
│   ├── merged/                 # Unified Parquet (7.2M rows)
│   └── processed/              # Feature-engineered Parquet
└── main.py                     # CLI pipeline entry point
```

### Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| **ML / Data** | scikit-learn, XGBoost, pandas, PyArrow | Model training & data pipeline |
| **Storage** | Apache Parquet (Snappy) | Chunked 7.2M-row processing |
| **Backend API** | FastAPI, Uvicorn, Pydantic | REST inference service |
| **Frontend** | React 18, Vite, JavaScript | Interactive research dashboard |
| **Visualization** | Matplotlib, Seaborn, Plotly | Reports & model charts |

---

## 📊 Model Performance

### Supervised Models (out-of-time evaluation)

All scores are computed on the out-of-time evaluation period (the most recent 20% of each source's transactions) — never on training rows:

| Model | ROC-AUC | PR-AUC | Brier (raw → calibrated) | ECE (raw → calibrated) |
|---|---|---|---|---|
| **XGBoost (active model)** | **0.941** | 0.269 | 0.0452 → **0.0050** | 0.0915 → **0.0014** |
| Random Forest | 0.932 | 0.221 | 0.0480 → **0.0051** | 0.0935 → **0.0016** |

> Trained on a fraud-preserving sample with a 3:1 legitimate-to-fraud ratio only — no `class_weight`/`scale_pos_weight` re-weighting on top of it. Raw model scores are mapped to real-world fraud probabilities by isotonic calibration on a held-out, natural-prevalence calibration frame disjoint from the training sample.

### Tuned Operating Points

A naive 0.5 probability threshold is meaningless at ~0.4–0.6% fraud prevalence. These are the empirically tuned operating points persisted in `models/fusion_thresholds.json` and served by the API:

| Band | Condition | Precision | Recall / Coverage |
|---|---|---|---|
| Supervised alert | calibrated p ≥ 0.098 | 25.0% | 35.1% of fraud (~83 alerts / 10k rows) |
| **FRAUD_LIKELY** | fusion score ≥ 0.482 and supervised gate ≥ 0.098 | **40.0%** | 20.0% of fraud (~30 alerts / 10k rows) |
| AMBIGUOUS_REVIEW | fusion score ≥ 0.395 or ambiguity rule fires | 2.7% | 10% of traffic; catches 57.3% of fraud not already in FRAUD_LIKELY |

### Unsupervised Models
- **Isolation Forest** — 250 estimators, `contamination=0.05`
- **Local Outlier Factor** — 35 neighbors, novelty mode

---

## 🗃️ Datasets

Four real-world public datasets, normalized into a unified 10-column schema:

| Dataset | Source | Rows | Domain |
|---|---|---|---|
| **PaySim** | Kaggle | ~6.3M | Synthetic mobile money |
| **UPI Transaction 2024** | Kaggle | ~100K | Indian UPI payments |
| **IEEE Fraud Detection** | Kaggle | ~590K | E-commerce transactions |
| **Digital Payment Transactions** | Zenodo | ~230K | Digital payments |

**Common Schema:** `transaction_id · timestamp · amount · sender_id · receiver_id · device_type · merchant_category · location · transaction_type · fraud_label`

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+ | Node.js 18+

### 1. Clone & Setup Python Environment

```bash
git clone https://github.com/Priyanshu6926/UPI_Fraud_Major_Project.git
cd UPI_Fraud_Major_Project
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Add Datasets

Download the 4 datasets and place them in `data/raw/`:
```
data/raw/paysim.csv
data/raw/upi_transaction_2024.csv
data/raw/ieee_transaction.csv
data/raw/digital_payment_transactions.csv
```

### 3. Run the ML Pipeline

```bash
# Full pipeline (recommended first run)
python main.py --all --chunk-size 100000 --fit-rows 100000 --supervised-rows 500000 --legitimate-ratio 3 --anomaly-rows 200000
```

This runs all 6 stages: data mapping → preprocessing → feature engineering → supervised training (+ isotonic calibration) → anomaly training → fusion threshold tuning.

### 4. Start the API + Dashboard

**Terminal 1 — Backend:**
```bash
uvicorn api.main:app --reload --port 8000
```

**Terminal 2 — Frontend:**
```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173** in your browser.

---

## 🖥️ Dashboard Features

| Page | Description |
|---|---|
| **Simulation** | Enter a transaction manually or select a preset and run real-time model inference |
| **Workflow** | Animated end-to-end pipeline visualization |
| **Analytics** | Dataset distributions, model metrics, feature importance charts |
| **Finance Research Report** | Population distribution, median grey band, fusion score explanation |
| **Prediction Logs** | Persistent history of all tested transactions with detail views |

---

## 🔬 Key Technical Contributions

1. **Multi-dataset schema normalization** — 4 heterogeneous datasets unified into one 10-column schema with alias-based column resolution
2. **Chunked Parquet pipeline** — 7.2M rows processed without loading everything into RAM (100K rows/chunk)
3. **Fraud-preserving stratified sampling** — Retains all fraud rows; legitimate rows sampled at configurable ratio across hour/day strata
4. **Transparent fusion layer** — Not a third classifier. Combines `fraud_probability` (60%) + `anomaly_percentile` (40%) with a disagreement penalty that creates the `AMBIGUOUS_REVIEW` band
5. **Probability calibration** — Isotonic calibration (`CalibratedClassifierCV`, prefit) maps raw model scores to real-world fraud probabilities using a held-out, natural-prevalence calibration set
6. **Sensitivity diagnostics** — Each prediction includes 6 controlled input variations (amount×10, night time, CASH_OUT, etc.) to explain model behavior

---

## 📁 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | API health check |
| `GET` | `/models/status` | Model artifact availability |
| `GET` | `/presets` | Sample transaction presets |
| `GET` | `/reports/summary` | Model metrics + plot list |
| `GET` | `/analytics/data` | Population stats for charts |
| `POST` | `/predict` | Full fraud + anomaly + fusion prediction |

---

## ⚠️ Research Scope & Disclaimer

This system is an **offline, post-transaction research experiment**. It intentionally does **not** implement:
- Real-time payment processing or streaming
- Banking system integration or authentication
- Production deployment or fraud blocking

The fusion resolution (`LIKELY_LEGITIMATE` / `AMBIGUOUS_REVIEW` / `FRAUD_LIKELY`) is a research score for comparing model signals — **not a payment decision**.

### Known Limitation: Cold-Start Blind Spot (measured on out-of-time data)

First-ever transactions are scored with neutralized behavioral features (frequency 0, population-average baseline, no velocity history), which has two measured consequences on the out-of-time test set (1.42M rows; first-ever senders are **87.9% of traffic**):

- **New users are reviewed *less*, not more.** Only **0.085%** of legitimate first-ever transactions land in `AMBIGUOUS_REVIEW` (~1 in 1,200); the two structural cold-start flags (`new_payee_flag`, `unusual_location_flag`) account for ~800 of those rows while also lifting fraud-in-review from 458 to 776 (+318 catches at ~29% incremental precision). The velocity-abuse override never fires on organic cold traffic by construction (it requires history-based signals such as rapid succession).
- **Review-band load concentrates on returning users**, whose richer behavioral features produce higher signal disagreement — the opposite of the "new users get flagged for being new" pattern seen in production fraud systems.
- **57% of first-transaction fraud resolves `LIKELY_LEGITIMATE`.** History-free scoring is inherently blind to account-takeover bursts on their opening transaction; real platforms compensate with deliberate extra scrutiny on first payments (step-up authentication, lower initial limits). Adding an equivalent first-transaction policy is the most impactful future improvement.

> **Deterministic overrides are not quantified offline.** The two rule-based overrides that bypass the statistical fusion score — the Layer-2 absolute amount bound (> ₹20 lakh → `FRAUD_LIKELY`) and the Layer-3 velocity-abuse rule (3+ behavioral flags → `AMBIGUOUS_REVIEW`) — are **not measured for precision/recall on the offline evaluation set**. Both depend on live per-sender history (or physical-amount validation) that point-in-time batch evaluation cannot simulate; their value currently rests on logical design and manual testing, not quantified offline evaluation.

---

## 👥 Contributors

| Name | Role |
|---|---|
| **Priyanshu Shingole** | ML Pipeline, API, Documentation |
| **Rachit Kale** | Frontend Dashboard, UI/UX |

---

## 📄 License

This project is submitted as a B.Tech Final Year Major Project. All datasets used are publicly available under their respective licenses (Kaggle Open Data, Zenodo CC-BY).

---

<div align="center">

**⭐ If you found this project interesting, please consider giving it a star!**

*Built with ❤️ as a B.Tech Final Year Major Project*

</div>
