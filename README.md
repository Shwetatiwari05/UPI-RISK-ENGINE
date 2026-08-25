<div align="center">

# 🔐 UPI Fraud Detection System
### Hybrid Anomaly Detection for Identifying Ambiguous UPI Transactions

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1-189AB4?style=for-the-badge)](https://xgboost.readthedocs.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-Vite-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://vitejs.dev)
[![License](https://img.shields.io/badge/License-Academic%20Research-blueviolet?style=for-the-badge)](LICENSE)

> **B.Tech Final Year Major Project** — An end-to-end post-facto machine learning research system that combines supervised fraud detection and unsupervised anomaly detection with a transparent, explainable fusion layer, applied to 7.2 million unified transaction records across four public payment datasets.

</div>

---

## 📌 Project Highlights

| Metric | Value |
|---|---|
| 📊 **Unified Records Processed** | 7,218,220 transactions (29,356 labeled fraud ≈ 0.41%) |
| 🎯 **XGBoost ROC-AUC (out-of-time)** | **0.941** |
| 🎯 **Random Forest ROC-AUC (out-of-time)** | 0.932 |
| 🎯 **FRAUD_LIKELY precision (tuned)** | **40.0%** at ~29.6 alerts / 10k rows |
| 🔍 **Review-band coverage (tuned)** | 10.0% of traffic · catches **57.3%** of remaining fraud |
| 🧮 **Calibration gain (XGBoost)** | Brier 0.0452 → **0.0050** · ECE 9.15% → **0.14%** |
| 📁 **Datasets Used** | 4 public datasets (Kaggle + Zenodo) |
| 🖥️ **Frontend** | React + Vite dashboard with 5 pages |

Every number in this README was measured directly from committed artifacts (`reports/*.json`, `models/*.json`) or recomputed on the out-of-time evaluation set through the exact production scoring path.

---

## 🧠 What This Project Does

This system performs **post-facto transaction analysis** on UPI-style payment records using a **dual-signal approach**:

```
Raw Transactions (7.2M rows, 4 sources)
        │
        ▼
┌───────────────────────┐      ┌─────────────────────────┐
│  SUPERVISED LEARNING  │      │  UNSUPERVISED LEARNING  │
│  XGBoost (+ RF bench) │      │  Isolation Forest       │
│  calibrated fraud     │      │  anomaly percentile     │
│  probability          │      │  vs training scores     │
└──────────┬────────────┘      └────────────┬────────────┘
           │                                │
           └──────────┬─────────────────────┘
                      ▼
         ┌────────────────────────┐
         │   FUSION LAYER         │
         │  weighted score (0.6/  │
         │  0.4) + ambiguity band │
         │  (disagreement +       │
         │  uncertainty kernel)   │
         └────────────┬───────────┘
                      ▼
         ┌────────────────────────┐
         │   RESEARCH RESOLUTION  │
         │  LIKELY_LEGITIMATE     │
         │  AMBIGUOUS_REVIEW  ◀────  Novel contribution
         │  FRAUD_LIKELY          │
         └────────────────────────┘
```

The `AMBIGUOUS_REVIEW` band is the **core research contribution**: instead of forcing every transaction into a binary verdict, transactions where the two model families disagree — or where evidence sits near the decision boundary — are routed to an explicit human-review state with transparent diagnostics.

Two **deterministic layers** sit above the statistical path and take precedence over it:

1. **Absolute amount bound** — any transaction above ₹20,00,000 resolves `FRAUD_LIKELY` regardless of model output (data-validation ceiling, `UPI_ABSOLUTE_MAX_AMOUNT`).
2. **Velocity-abuse rule** — a sender simultaneously tripping ≥3 behavioral flags (`amount_spike`, `new_payee_flag`, `unusual_location_flag`, `rapid_transactions`) matches the classic account-takeover pattern and escalates to `AMBIGUOUS_REVIEW` independently of scores.

---

## 🏗️ Architecture & Tech Stack

```
upi-fraud-detection/
├── src/                        # Core ML pipeline modules
│   ├── schema_mapping.py       # Multi-dataset schema normalization (alias-based)
│   ├── data_preprocessing.py   # RobustScaler + OneHot encoding state
│   ├── parquet_pipeline.py     # Chunked Parquet ingestion & per-source splits
│   ├── feature_engineering.py  # Causal point-in-time behavioral features
│   ├── supervised_model.py     # XGBoost + Random Forest training
│   ├── anomaly_detection.py    # Isolation Forest + LOF training
│   ├── probability_calibration.py  # Isotonic calibration suite
│   ├── fusion_model.py         # Dual-signal fusion + threshold tuning
│   └── live_history.py         # SQLite store for serving-time personalization
├── api/main.py                 # FastAPI REST backend (6 endpoints)
├── app/prediction_engine.py    # Model inference engine (live feature context)
├── frontend/                   # React + Vite dashboard (5 pages)
├── models/                     # Trained artifacts + tuned thresholds (.pkl/.json)
├── reports/                    # Metrics JSON, row counts, charts
├── data/
│   ├── raw/                    # Input CSVs (not committed)
│   ├── merged/                 # Unified Parquet — mapped_common_schema.parquet
│   └── processed/              # Feature-engineered Parquet + anomaly reference scores
└── main.py                     # CLI pipeline entry point (6 stages)
```

| Layer | Technology | Purpose |
|---|---|---|
| **ML / Data** | scikit-learn, XGBoost, pandas, PyArrow | Model training & streaming pipeline |
| **Storage** | Apache Parquet (Snappy) | Chunked 7.2M-row processing, 100k rows/chunk |
| **Serving state** | SQLite | Per-sender live history for prediction-time features |
| **Backend API** | FastAPI, Uvicorn, Pydantic | REST inference service |
| **Frontend** | React 18, Vite | Interactive research dashboard |

---

## 🗃️ Data

Four heterogeneous public datasets are alias-mapped into one common schema and concatenated into a single Parquet file. Measured contributions (from `mapped_common_schema.parquet`):

| Source | Rows | Fraud rows | Share of all fraud | Domain |
|---|---:|---:|---:|---|
| IEEE-CIS Fraud Detection (Kaggle) | 590,540 | 20,663 | **70.4%** | E-commerce card transactions |
| PaySim (Kaggle) | 6,362,620 | 8,213 | **28.0%** | Synthetic mobile-money logs |
| UPI Transaction 2024 (Kaggle) | 250,000 | 480 | **1.6%** | Indian UPI payments |
| Digital Payment Transactions (Zenodo) | 15,060 | 0 | 0.0% | Digital payments (legit-only) |
| **Total** | **7,218,220** | **29,356** | 100% | — |

**Common schema (10 columns):** `transaction_id · timestamp · amount · sender_id · receiver_id · device_type · merchant_category · location · transaction_type · fraud_label`

> ⚠️ **Honest composition note:** only 1.6% of rows come from a genuine UPI dataset, and they carry just 480 fraud labels. Most *positive* training signal originates from e-commerce (IEEE) and synthetic mobile-money (PaySim) fraud patterns. The system is therefore best described as a **cross-domain fraud-pattern study with a UPI-shaped schema**, not a UPI-native detector — see [Known Limitations](#️-known-limitations--measured).

---

## ⚙️ Pipeline (6 stages — `python main.py --all`)

| Stage | What happens | Leakage controls |
|---|---|---|
| **1. Load & Map** | Stream raw CSVs in chunks, alias-map columns, unify timestamps, write compressed Parquet | — |
| **2. Preprocess** | Fit `RobustScaler` + one-hot state once on a bounded, source-proportional **train-period-only** fit sample; stream-transform all rows | Fit sample drawn exclusively from train period |
| **3. Feature Engineering** | Causal point-in-time behavioral features per sender; persist engineered intermediate + anomaly reference scores | All history features computed from **strictly prior** rows only |
| **4. Supervised Training** | Train XGBoost + Random Forest on a fraud-preserving balanced sample; isotonic calibration on a disjoint natural-prevalence frame; save rank grid | Calibrator frame disjoint from training sample |
| **5. Anomaly Training** | Fit Isolation Forest (250 trees, `contamination=0.05`) + LOF (35 neighbors, novelty mode) on train-period rows; export score reference distribution | Train-period rows only |
| **6. Fusion Tuning** | Sweep FRAUD_LIKELY/review operating points on the **out-of-time** period; persist to `models/fusion_thresholds.json` | Evaluated only on held-out time period |

**Temporal split design:** each source is cut at its own 80th-percentile timestamp (`DEFAULT_TEST_FRACTION = 0.2`), so the evaluation set is strictly *future* relative to training for every domain — no random row-level splits anywhere. The resulting out-of-time evaluation set contains **1,419,809 rows** with 0.59% prevalence.

---

## 🔬 Feature Engineering

All behavioral features are computed **point-in-time** — a row may only see transactions that happened strictly before it:

| Feature group | Examples | Cold-start behavior |
|---|---|---|
| Temporal | `hour_of_day` (IST), night-window flag | Always available |
| Sender history | `transaction_frequency`, `avg_transaction_amount`, amount z-score vs own mean/std, `minutes_since_previous_sender_txn` | Neutralized to population baselines until the sender has history |
| Velocity | `rapid_transactions` — ≥2 txns within a **5-minute** window; high-frequency flag at the train-period **95th-percentile** rate | Fires only with history |
| Structural flags | `new_payee_flag` (first-ever receiver pair), `unusual_location_flag` (location outside sender's observed set), `amount_spike` (vs fitted amount moments) | Fire immediately — these are the cold-start safety net |

At **serving time**, the same feature builders consume live per-sender context from SQLite (`data/live_history.db`), tiered by available history: no history → population pooled fallback; exactly one prior transaction → relative multiple of it; two or more → the sender's real running mean ± std.

---

## 📊 Models & Honest Metrics

### Supervised models (out-of-time evaluation, 1.42M rows)

Trained on a fraud-preserving balanced sample (max 500k rows, 3:1 legitimate-to-fraud ratio, stratified across time strata). No `class_weight` / `scale_pos_weight` stacking on top.

| Model | ROC-AUC | PR-AUC | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| **XGBoost (active)** | **0.941** | **0.269** | 0.932 | 0.061* | 0.733 | 0.113 |
| Random Forest | 0.932 | 0.221 | 0.927 | 0.058* | 0.743 | 0.107 |

\* Raw-model precision/recall at the default 0.5 boundary are reported for completeness only — at 0.59% prevalence they are not meaningful operating points. The deployable operating points are the tuned ones below.

**Reading these numbers honestly:** ROC-AUC ≈ 0.94 looks strong, but PR-AUC ≈ 0.27 reflects the true difficulty of finding ~0.6%-prevalence fraud. Accuracy is a vanity metric here (a "always legitimate" classifier scores 99.4%). This gap between ROC-AUC and PR-AUC is exactly why the project invests in calibration and explicit review bands instead of a single decision threshold.

### Unsupervised models

- **Isolation Forest** (served): 250 estimators; raw anomaly scores are percentile-ranked against the saved training-score distribution so supervised ranks and anomaly percentiles are compared like-for-like.
- **Local Outlier Factor** (trained, benchmark-only): 35 neighbors, novelty mode — trained for comparison but **not loaded in the serving path** (see limitations).

---

## 📏 Probability Calibration

Raw model outputs are not probabilities — especially after rebalanced training (effective fraud rate 25% vs real-world ~0.41%). Every score therefore passes through **isotonic regression**, fit on a held-out calibration frame of **1,284,999 rows / 5,226 fraud** at natural prevalence, disjoint from both training and tuning data.

| Metric | XGBoost raw → isotonic | RandomForest raw → isotonic |
|---|---|---|
| Brier score | 0.04521 → **0.00499** | 0.04805 → **0.00513** |
| Expected Calibration Error | 0.09147 → **0.00140** | 0.09352 → **0.00155** |
| ROC-AUC (ranking quality kept) | 0.94089 → 0.94076 | 0.93218 → 0.93199 |

Isotonic preserves ranking while making scores interpretable as real-world fraud probabilities. The tuned alert threshold (below) sits at **p ≥ 0.098** — far below the naive 0.5 — because honest probabilities rarely exceed 1–2% even for clearly fraudulent patterns.

A legacy analytic prior-correction method (`correct_prior_probability`) remains in the codebase as a fallback for artifacts predating isotonic calibration.

---

## ⚖️ Fusion & Three-Path Resolution

### Score fusion

```
fusion_score = 0.6 × calibrated_fraud_probability
             + 0.4 × anomaly_percentile
```

The supervised rank comes from a 999-point quantile grid of calibration-frame scores, so both signals enter fusion as population percentiles.

### Ambiguity (the review signal — deliberately *not* another risk score)

`ambiguity = 0.50 × disagreement + 0.35 × uncertainty (+ repeatability penalty at serving)`

- **Disagreement** = |supervised rank − anomaly percentile|, counted only when at least one signal claims top-decile risk (the "informed region" — raw percentile differences exceed 0.45 about a third of the time by pure chance).
- **Uncertainty** = Gaussian kernel around the supervised gate in **logit space** (`exp(−½Δ²)`), so "uncertain" means *near the decision boundary*, not merely mid-scale.

### Resolution precedence (as served by `fuse_signals`)

1. Amount > ₹20,00,000 → **FRAUD_LIKELY** (deterministic bound)
2. ≥3 velocity flags → **AMBIGUOUS_REVIEW** (rule-based override)
3. Ambiguity ≥ 0.38 or disagreement ≥ 0.45 → **AMBIGUOUS_REVIEW**
4. Fusion ≥ 0.4817 **and** p ≥ 0.098 → **FRAUD_LIKELY**
5. Fusion ≥ 0.4817 below gate → **AMBIGUOUS_REVIEW** (anomaly unusualness alone can't promote)
6. Fusion ≥ 0.3702 → **AMBIGUOUS_REVIEW**
7. Otherwise → **LIKELY_LEGITIMATE**

### Tuned operating points (persisted & served — verified to reconcile exactly)

Tuned on the full out-of-time period against explicit targets (FL precision floor 0.40; review share target 10% within [5%, 15%]):

| Band | Condition | Precision | Coverage |
|---|---|---|---|
| Supervised alert | p ≥ 0.098 | 25.0% | 35.1% of fraud · ~83 alerts/10k |
| **FRAUD_LIKELY** | fusion ≥ 0.4817 ∧ p ≥ 0.098, outside ambiguity | **40.0%** | **20.0% of all fraud** · 29.56 alerts/10k (4,197 rows) |
| **AMBIGUOUS_REVIEW** | ambiguity-first ∨ fusion ≥ 0.3702 | 2.7% | **10.0% of traffic** (141,981 rows) · catches **57.3%** of fraud not already flagged (3,857 fraud rows) |

Serving-side verification (recomputing resolutions over all 1,419,809 test rows through the exact production formulas) reproduces the stored achieved metrics **bit-exactly** on all eight reported statistics.

---

## 🥶 Cold-Start Behavior (measured, out-of-time)

First-ever transactions have no behavioral history, so history-derived features collapse to neutral values. Measured consequences on the 1.42M-row test set (**87.8%** of its traffic is a sender's first-ever transaction):

- **New users are reviewed *less*, not more.** Only **0.345%** of legitimate first-ever transactions land in `AMBIGUOUS_REVIEW` (~1 in 289) — and every one of those carries a structural cold-start flag.
- Those two structural flags are doing real work offline: they lift fraud-in-review from 1,722 to 4,167 rows (**+2,445 catches at ~57% incremental precision**).
- Even so, **53.4% of first-transaction fraud still resolves `LIKELY_LEGITIMATE`.** History-free scoring is inherently blind to account-takeover bursts on their opening transaction; real platforms compensate with step-up authentication and lower initial limits on first payments. Adding an equivalent first-transaction policy is the most impactful future improvement.
- The velocity-abuse override is nearly silent on organic batch traffic (fires on 182 rows, 3.3% precision, zero additional fraud beyond statistical bands) and the ₹20L bound never fires in-sample — both are online safety nets, not offline detectors.

---

## ⚠️ Known Limitations (measured & stated)

1. **Cold-start blind spot** — quantified above; 53.4% of first-transaction fraud slips through as `LIKELY_LEGITIMATE`.
2. **Cross-domain dependence** — 98%+ of rows and ~98% of fraud labels originate from non-UPI sources (synthetic mobile money + e-commerce). Transfer to genuinely UPI-native traffic is unproven.
3. **Adaptive-baseline grooming ("self-poisoning")** — serving records *every* primary transaction into live history, including ones resolved `AMBIGUOUS_REVIEW` or `FRAUD_LIKELY`. An adversary can deliberately shape their own baseline (e.g., transact near a planned fraud amount, pre-warm payees and locations) so subsequent fraud looks statistically normal. There is no authentication, rate limiting, exclusion of flagged transactions from history, or baseline-drift monitoring. Practical risk in this localhost demo context is low, but the pattern is a genuine production threat class for any adaptive scoring system.
4. **LOF is trained but not served** — the API loads only the Isolation Forest; LOF exists as a training-time benchmark.
5. **No automated test suite** — verification rests on two scripts (`scripts/leakage_probe.py`, `scripts/test_temporal_split_synthetic.py`) plus ad-hoc artifact checks; no CI, no drift monitoring, no retraining triggers.
6. **Single-process SQLite assumptions** — unbounded live-history growth and per-process caching; fine for a demo, not for concurrent serving.
7. **Feature neutralization for known senders** — several population-aggregate features are intentionally zeroed once a sender acquires live history, trading population context for personalization; the sensitivity diagnostics panel makes this visible per prediction.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+ | Node.js 18+

### 1. Clone & Setup Python Environment

```bash
git clone <your-repo-url>
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

Place the four CSVs in `data/raw/`:

```
data/raw/paysim.csv
data/raw/upi_transaction_2024.csv
data/raw/ieee_transaction.csv
data/raw/digital_payment_transactions.csv
```

### 3. Run the ML Pipeline

```bash
# Full pipeline — recommended demonstration run (bounded samples)
python main.py --all --chunk-size 100000 --fit-rows 100000 --supervised-rows 500000 --legitimate-ratio 3 --anomaly-rows 200000
```

Individual stages: `--load`, `--preprocess`, `--features`, `--train-supervised`, `--train-anomaly`, `--tune-thresholds`. See [RUN_COMMANDS.md](RUN_COMMANDS.md) and [Pipeline.md](Pipeline.md) for details, verification snippets, and troubleshooting.

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

Open **http://localhost:5173**. All timestamps shown in the UI are normalized to IST throughout the serving path.

---

## 🖥️ Dashboard Features

| Page | Description |
|---|---|
| **Simulation** | Manual or preset transactions through the real inference stack — dual signals, fusion score, resolution |
| **Workflow** | Animated end-to-end pipeline visualization |
| **Analytics** | Dataset distributions, model metrics, feature importance |
| **Finance Research Report** | Population distribution, median grey-band comparison |
| **Prediction Logs** | Persistent local history of tested transactions |

The Simulation view also surfaces **Personalization Signals** (which cold-start/history features fired), the **velocity-rule override badge**, and **sensitivity diagnostics** (six controlled input variations such as amount×10 or night-time) explaining *why* a resolution was reached.

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/models/status` | Artifact availability |
| `GET` | `/presets` | Sample transaction presets |
| `GET` | `/reports/summary` | Saved training metrics + plots |
| `GET` | `/analytics/data` | Population stats for charts |
| `POST` | `/predict` | Full dual-signal + fusion prediction |

---

## ⚠️ Research Scope & Disclaimer

This system is a **post-facto research experiment**. It intentionally does **not** implement:
- Real-time payment processing or streaming
- Banking-system integration or authentication
- Production deployment or fraud blocking

The three-path resolution is a research instrument for comparing model signals — **not a payment decision**.

---

## 👥 Contributors

| Name | Role |
|---|---|
| **Priyanshu Shingole** | ML Pipeline, API, Documentation |
| **Rachit Kale** | Frontend Dashboard, UI/UX |

---

## 📄 License

Submitted as a B.Tech Final Year Major Project. All datasets used are publicly available under their respective licenses (Kaggle Open Data, Zenodo CC-BY).

---

<div align="center">

**⭐ If you found this project interesting, please consider giving it a star!**

*Built as a B.Tech Final Year Major Project*

</div>
