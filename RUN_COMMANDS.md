# UPI Fraud Detection: Run Commands

This guide provides step-by-step instructions for running the offline research pipeline, backend API, and React dashboard on **macOS**, **Linux**, and **Windows**.

Run Python commands from the project root. Run React commands from the `frontend/` folder.

---

## 1. Open the Project Directory

### macOS / Linux
```bash
cd upi-fraud-detection
```

### Windows (PowerShell)
```powershell
cd upi-fraud-detection
```

---

## 2. Create and Activate the Python Environment

Create the environment once:

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Confirm that the prompt starts with `(.venv)`.

---

## 3. Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

The requirements include the Parquet engine (PyArrow), scikit-learn, XGBoost, FastAPI, and other dependencies.

---

## 4. Verify the Datasets

Ensure the approved dataset CSV files are present in `data/raw/`:
- `data/raw/digital_payment_transactions.csv`
- `data/raw/ieee_transaction.csv`
- `data/raw/paysim.csv`
- `data/raw/upi_transaction_2024.csv`

### macOS / Linux
```bash
ls -lh data/raw
```

### Windows (PowerShell)
```powershell
Get-ChildItem ./data/raw
```

> **Note:** The pipeline creates generated Parquet files in `data/merged/` and `data/processed/`.

---

## 5. Run the Complete ML Pipeline

To run the complete data ingestion, preprocessing, feature engineering, and model training:

```bash
python main.py --all
```

The complete execution order:
1. Load each raw CSV in chunks and map it to the common schema.
2. Write the mapped records to compressed Parquet at `data/merged/mapped_common_schema.parquet`.
3. Fit the preprocessing state once on a bounded fitting sample.
4. Read mapped Parquet chunks, apply preprocessing and feature engineering, and write `data/processed/processed_features.parquet`.
5. Train XGBoost and Random Forest once using the configured supervised sample.
6. Train Isolation Forest and LOF once using the configured anomaly sample.
7. Save models, preprocessing state, evaluation metrics, and reports to `models/` and `reports/`.

---

## 6. Recommended Demonstration Run

This command keeps the default chunk size at 100,000 rows and uses bounded training samples for faster execution:

```bash
python main.py --all --chunk-size 100000 --fit-rows 100000 --supervised-rows 500000 --legitimate-ratio 3 --anomaly-rows 200000
```

---

## 7. Run Individual Pipeline Stages (Optional)

Individual stages can be run separately if needed:

- **Load and map Parquet:**
  ```bash
  python main.py --load --chunk-size 100000
  ```
- **Stream preprocessing and feature engineering:**
  ```bash
  python main.py --preprocess --chunk-size 100000 --fit-rows 100000
  ```
- **Train supervised models (XGBoost & Random Forest):**
  ```bash
  python main.py --train-supervised --supervised-rows 500000 --legitimate-ratio 3
  ```
- **Train anomaly models (Isolation Forest & LOF):**
  ```bash
  python main.py --train-anomaly --anomaly-rows 200000
  ```

---

## 8. Verify Parquet Output and Row Counts

Check the mapped Parquet file:
```bash
python -c "import pyarrow.parquet as pq; p=pq.ParquetFile('data/merged/mapped_common_schema.parquet'); print('mapped rows:', p.metadata.num_rows); print('row groups:', p.num_row_groups); print('columns:', p.schema_arrow.names)"
```

Check the processed feature Parquet file:
```bash
python -c "import pyarrow.parquet as pq; p=pq.ParquetFile('data/processed/processed_features.parquet'); print('processed rows:', p.metadata.num_rows); print('row groups:', p.num_row_groups); print('columns:', p.schema_arrow.names)"
```

Check saved model and report files:

### macOS / Linux
```bash
ls -lh models/
ls -lh reports/
```

### Windows (PowerShell)
```powershell
Get-ChildItem ./models
Get-ChildItem ./reports
```

---

## 9. Start the FastAPI Backend

In the project root with the virtual environment activated:

### macOS / Linux
```bash
source .venv/bin/activate
uvicorn api.main:app --reload --port 8000
```

### Windows (PowerShell)
```powershell
.\.venv\Scripts\Activate.ps1
uvicorn api.main:app --reload --port 8000
```

Verify backend health in another terminal:

### macOS / Linux
```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/models/status
```

### Windows (PowerShell)
```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/models/status
```

---

## 10. Start the React Frontend

Open a second terminal, navigate to `frontend/`, install packages, and start the development server:

```bash
cd frontend
npm install
npm run dev
```

Open your browser and navigate to:
```text
http://localhost:5173
```

---

## 11. Test the Prediction Flow

With both the API and Vite dev servers running:

1. Open `http://localhost:5173`.
2. Go to **Simulation**.
3. Select a preset (e.g. *Everyday UPI Transfer* or *High-Risk Ambiguous*) or enter transaction details manually.
4. Click **Run Test**.
5. Review the separate Supervised Fraud Probability and Unsupervised Anomaly Score.
6. Review the **Dual-Signal Fusion Score** and **Research Resolution** (`LIKELY_LEGITIMATE`, `AMBIGUOUS_REVIEW`, or `FRAUD_LIKELY`).
7. Open **Finance Research Report** for the population distribution and median grey-band comparison.
8. Open **Prediction Logs** to inspect previous tests.

---

## 12. Troubleshooting

- **`zsh: command not found: python` (macOS):**
  Use `python3` instead of `python` to create the venv: `python3 -m venv .venv`.
- **Node module binary permission denied on macOS:**
  If copying from Windows, run `cd frontend && rm -rf node_modules package-lock.json && npm install`.
- **Missing models / 500 error on prediction:**
  Ensure model files exist in `models/` or rerun the pipeline: `python main.py --all`.
- **API connection error in React:**
  Ensure FastAPI is running on port 8000 (`uvicorn api.main:app --reload --port 8000`).

---

## 13. More Pipeline Details

See [Pipeline.md](Pipeline.md) for the Parquet architecture, chunk-processing behavior, memory notes, and CLI options.
