# Hybrid Anomaly Detection for Identifying Ambiguous UPI Transactions

Offline, batch-processing research project for post-transaction fraud analysis.

The current implementation covers:

- Dataset loading and schema mapping
- Data preprocessing
- Feature engineering
- Supervised fraud detection with XGBoost and Random Forest
- Unsupervised anomaly detection with Isolation Forest and Local Outlier Factor
- Local Vite React testing dashboard for manually checking model behavior
- Explainable research fusion score and downloadable transaction report
- Finance-style transaction report with population distribution and median grey band
- Clickable browser-persisted prediction logs with transaction detail views
- Dedicated Finance Research Report page with ranked transaction line graph and tested-input marker

The project intentionally does not implement real-time streaming, banking integration,
authentication, or production deployment. The local API supports only the React testing UI.

## Datasets

Use only the following datasets and place downloaded files under `data/raw/`:

1. PaySim Dataset from Kaggle
2. UPI Transaction 2024 from Kaggle
3. IEEE Fraud Detection from Kaggle
4. Digital Payment Transactions from Zenodo

No synthetic datasets are required or generated.

## Folder Structure

```text
upi-fraud-detection/
|-- data/
|   |-- raw/
|   |-- processed/
|   `-- merged/
|-- notebooks/
|-- src/
|-- models/
|-- app/
|   `-- components/
|-- reports/
|-- requirements.txt
|-- README.md
`-- main.py
```

## Setup

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Batch Pipeline

```bash
python main.py --all
```

The pipeline discovers supported dataset files in `data/raw/`, maps them into the
common schema, preprocesses them, engineers features, trains supervised models, and
trains anomaly detection models.

For the Parquet layout, chunk-size options, memory behavior, and model-training
limitations, see [Pipeline.md](Pipeline.md).

## Streamlit Testing Dashboard (Optional)

```bash
streamlit run app/app.py
```

This legacy local testing interface displays supervised and unsupervised outputs separately.

## Vite React Testing Dashboard

The React dashboard uses a small local FastAPI service because browser JavaScript cannot
load Python `joblib`, scikit-learn, XGBoost, or Isolation Forest model artifacts directly.

Start the API:

```bash
uvicorn api.main:app --reload
```

Start the React app in a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open:

```text
http://localhost:5173
```

The API is local-only and is used only for manual offline model testing.
The React dashboard additionally presents an explainable fusion score and a downloadable
transaction report. Its resolution can be `LIKELY_LEGITIMATE`, `FRAUD_LIKELY`, or
`AMBIGUOUS_REVIEW`; it is a research interpretation, not a payment decision.

## Common Schema

Every dataset is mapped into:

- `transaction_id`
- `timestamp`
- `amount`
- `sender_id`
- `receiver_id`
- `device_type`
- `merchant_category`
- `location`
- `transaction_type`
- `fraud_label`

## Important Boundary

The fusion layer is deliberately transparent rather than a third opaque classifier. It uses
a weighted dual-signal score plus model disagreement and supervised uncertainty to identify
transactions that belong in an ambiguous research-review band. It does not approve, decline,
block, or prove fraud for a transaction.
