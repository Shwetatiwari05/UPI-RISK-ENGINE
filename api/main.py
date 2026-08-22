"""FastAPI service for the React UPI fraud testing dashboard."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.prediction_engine import PredictionEngine  # noqa: E402
from src.fusion_model import build_transaction_report, fuse_signals  # noqa: E402
from src.parquet_pipeline import MAPPED_PARQUET_PATH, iter_parquet_chunks  # noqa: E402
from src.utils import MERGED_DATA_DIR, MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR  # noqa: E402


class TransactionRequest(BaseModel):
    """Manual transaction payload submitted from the React dashboard."""

    amount: float = Field(ge=0)
    transaction_type: str
    device_type: str
    merchant_category: str
    timestamp: datetime
    sender_id: str
    receiver_id: str
    location: str
    transaction_id: str | None = None


app = FastAPI(
    title="UPI Fraud Detection Local API",
    description="Local API for offline model testing with the Vite React dashboard.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_prediction_engine() -> PredictionEngine:
    """Load model artifacts once per API process."""
    return PredictionEngine(MODELS_DIR)


@lru_cache(maxsize=1)
def get_fusion_threshold_overrides() -> dict[str, Any] | None:
    """Load tuned fusion thresholds saved by the offline pipeline."""
    path = MODELS_DIR / "fusion_thresholds.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload.get("thresholds") or None


@app.get("/health")
def health() -> dict[str, str]:
    """Return API health status."""
    return {"status": "ok"}


@app.get("/models/status")
def model_status() -> dict[str, Any]:
    """Return model artifact availability for the UI."""
    engine = get_prediction_engine()
    artifacts = {
        "xgboost_model": (MODELS_DIR / "xgboost_model.pkl").exists(),
        "random_forest": (MODELS_DIR / "random_forest.pkl").exists(),
        "xgboost_calibrator": (MODELS_DIR / "xgboost_model_calibrator.pkl").exists(),
        "random_forest_calibrator": (MODELS_DIR / "random_forest_calibrator.pkl").exists(),
        "fusion_thresholds": (MODELS_DIR / "fusion_thresholds.json").exists(),
        "isolation_forest": (MODELS_DIR / "isolation_forest.pkl").exists(),
        "lof_model": (MODELS_DIR / "lof_model.pkl").exists(),
        "preprocessor": (MODELS_DIR / "preprocessor.pkl").exists(),
        "anomaly_preprocessor": (MODELS_DIR / "anomaly_preprocessor.pkl").exists(),
    }
    return {
        "ready": engine.is_ready,
        "artifacts": artifacts,
        "model_dir": str(MODELS_DIR),
    }


@app.get("/presets")
def presets() -> list[dict[str, Any]]:
    """Return sample transaction presets for quick UI testing."""
    return [
        {
            "name": "Everyday UPI Transfer",
            "amount": 650,
            "transaction_type": "TRANSFER",
            "device_type": "Android",
            "merchant_category": "Personal",
            "sender_id": "user_1024",
            "receiver_id": "user_2048",
            "location": "Mumbai",
            "hour": 13,
        },
        {
            "name": "Late Night High Value",
            "amount": 85000,
            "transaction_type": "PAYMENT",
            "device_type": "Web",
            "merchant_category": "Electronics",
            "sender_id": "user_1024",
            "receiver_id": "merchant_991",
            "location": "Delhi",
            "hour": 2,
        },
        {
            "name": "Rapid Cash Out Pattern",
            "amount": 125000,
            "transaction_type": "CASH_OUT",
            "device_type": "Unknown",
            "merchant_category": "Unknown",
            "sender_id": "user_7777",
            "receiver_id": "receiver_3333",
            "location": "Unknown",
            "hour": 1,
        },
    ]


@app.get("/reports/summary")
def reports_summary() -> dict[str, Any]:
    """Expose saved training metrics when available."""
    metrics_path = REPORTS_DIR / "supervised_metrics.json"
    metrics = None
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    plots = sorted(path.name for path in REPORTS_DIR.glob("*.png"))
    return {
        "metrics": metrics,
        "plots": plots,
    }


@app.get("/analytics/data")
def data_analytics() -> dict[str, Any]:
    """Return cached analytics for the imported mapped dataset."""
    return get_dataset_analytics()


@app.post("/predict")
def predict(payload: TransactionRequest) -> dict[str, Any]:
    """Run supervised and anomaly model predictions for one manual transaction."""
    engine = get_prediction_engine()
    if not engine.is_ready:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts are missing. Run `python main.py --all` first.",
        )

    transaction = payload.model_dump()
    transaction["transaction_id"] = transaction.get("transaction_id") or (
        f"react_manual_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}"
    )

    try:
        prediction = run_model_prediction(engine, transaction)
        diagnostics = build_prediction_diagnostics(engine, transaction, prediction)
        fusion = fuse_signals(
            prediction["supervised"],
            prediction["anomaly"],
            diagnostics,
            thresholds=get_fusion_threshold_overrides(),
        )
        report = build_transaction_report(
            transaction,
            prediction["supervised"],
            prediction["anomaly"],
            fusion,
            diagnostics,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        **prediction,
        "diagnostics": diagnostics,
        "fusion": fusion,
        "report": report,
    }


def run_model_prediction(engine: PredictionEngine, transaction: dict[str, Any]) -> dict[str, Any]:
    """Run both model families and return UI-ready prediction values."""
    outputs = engine.predict(transaction)
    if outputs["supervised"].empty:
        raise RuntimeError("The supervised model did not return a prediction.")
    supervised = outputs["supervised"].iloc[0].to_dict()
    anomaly = outputs["anomaly"].iloc[0].to_dict()
    fraud_probability = float(supervised["fraud_probability"])
    if not np.isfinite(fraud_probability):
        raise RuntimeError("The supervised model returned an invalid fraud probability.")
    calibrated_confidence = calibrate_anomaly_confidence(float(anomaly["anomaly_score"]))
    return {
        "transaction_id": str(supervised["transaction_id"]),
        "supervised": {
            "fraud_probability": fraud_probability,
            "fraud_prediction": int(supervised["fraud_prediction"]),
            "confidence_score": float(supervised["confidence_score"]),
            "model_name": str(supervised.get("model_name", "supervised_model")),
            "signal_status": str(supervised.get("signal_status", "calculated")),
        },
        "anomaly": {
            "anomaly_score": float(anomaly["anomaly_score"]),
            "anomaly_label": str(anomaly["anomaly_label"]),
            "anomaly_confidence": calibrated_confidence,
            "anomaly_percentile": calibrated_confidence,
            "calibration_method": "training-score percentile",
        },
    }


def build_prediction_diagnostics(
    engine: PredictionEngine,
    transaction: dict[str, Any],
    base_prediction: dict[str, Any],
) -> dict[str, Any]:
    """Score controlled variants so the UI can show that inputs affect outputs."""
    repeat_prediction = run_model_prediction(engine, {**transaction, "transaction_id": f"{transaction['transaction_id']}_repeat"})
    base_fraud = base_prediction["supervised"]["fraud_probability"]
    base_anomaly = base_prediction["anomaly"]["anomaly_confidence"]
    repeat_fraud = repeat_prediction["supervised"]["fraud_probability"]
    repeat_anomaly = repeat_prediction["anomaly"]["anomaly_confidence"]

    variants = [
        ("Amount x10", {**transaction, "amount": float(transaction["amount"]) * 10}),
        ("Amount /10", {**transaction, "amount": max(1.0, float(transaction["amount"]) / 10)}),
        ("Force CASH_OUT", {**transaction, "transaction_type": "CASH_OUT"}),
        ("Unknown device", {**transaction, "device_type": "Unknown"}),
        ("Night time", {**transaction, "timestamp": _replace_hour(transaction["timestamp"], 2)}),
        ("Unknown location", {**transaction, "location": "Unknown"}),
    ]

    sensitivity = []
    for name, variant in variants:
        variant["transaction_id"] = f"{transaction['transaction_id']}_{name.lower().replace(' ', '_').replace('/', '_')}"
        variant_prediction = run_model_prediction(engine, variant)
        fraud_probability = variant_prediction["supervised"]["fraud_probability"]
        anomaly_percentile = variant_prediction["anomaly"]["anomaly_confidence"]
        sensitivity.append(
            {
                "name": name,
                "fraud_probability": fraud_probability,
                "fraud_delta": fraud_probability - base_fraud,
                "anomaly_percentile": anomaly_percentile,
                "anomaly_delta": anomaly_percentile - base_anomaly,
            }
        )

    return {
        "deterministic": {
            "repeat_fraud_delta": abs(repeat_fraud - base_fraud),
            "repeat_anomaly_delta": abs(repeat_anomaly - base_anomaly),
        },
        "sensitivity": sensitivity,
        "input_echo": {
            "amount": float(transaction["amount"]),
            "transaction_type": str(transaction["transaction_type"]),
            "device_type": str(transaction["device_type"]),
            "merchant_category": str(transaction["merchant_category"]),
            "sender_id": str(transaction["sender_id"]),
            "receiver_id": str(transaction["receiver_id"]),
            "location": str(transaction["location"]),
            "timestamp": str(transaction["timestamp"]),
        },
    }


def _replace_hour(value: Any, hour: int) -> datetime:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        timestamp = pd.Timestamp.now()
    return timestamp.replace(hour=hour, minute=0, second=0, microsecond=0).to_pydatetime()


@lru_cache(maxsize=1)
def get_dataset_analytics() -> dict[str, Any]:
    """Compute lightweight aggregate analytics from imported data."""
    parquet_path = MAPPED_PARQUET_PATH
    csv_path = MERGED_DATA_DIR / "merged_common_schema.csv"
    if not parquet_path.exists() and not csv_path.exists():
        return {"ready": False, "message": "Run `python main.py --all` to create merged data."}

    usecols = [
        "timestamp",
        "amount",
        "device_type",
        "merchant_category",
        "location",
        "transaction_type",
        "fraud_label",
    ]
    total_rows = 0
    amount_sum = 0.0
    amount_min = None
    amount_max = None
    fraud_counts: dict[str, int] = {}
    type_counts: dict[str, int] = {}
    device_counts: dict[str, int] = {}
    merchant_counts: dict[str, int] = {}
    location_counts: dict[str, int] = {}
    hour_counts = {str(hour): 0 for hour in range(24)}
    amount_bins = {
        "0-500": 0,
        "501-2k": 0,
        "2k-10k": 0,
        "10k-50k": 0,
        "50k+": 0,
    }
    sampled_amounts: list[np.ndarray] = []
    sampled_labels: list[np.ndarray] = []

    if parquet_path.exists():
        chunks = iter_parquet_chunks(parquet_path, chunk_size=250_000, columns=usecols)
    else:
        chunks = pd.read_csv(csv_path, usecols=usecols, chunksize=250_000)

    for chunk_number, chunk in enumerate(chunks):
        total_rows += len(chunk)
        amount = pd.to_numeric(chunk["amount"], errors="coerce").fillna(0)
        sample_size = min(2_500, len(chunk))
        if sample_size:
            sampled = chunk.sample(n=sample_size, random_state=42 + chunk_number)
            sampled_amounts.append(pd.to_numeric(sampled["amount"], errors="coerce").fillna(0).to_numpy(dtype=np.float64, copy=True))
            sampled_labels.append(sampled["fraud_label"].fillna(0).to_numpy(dtype=np.int8, copy=True))
        amount_sum += float(amount.sum())
        current_min = float(amount.min()) if len(amount) else 0.0
        current_max = float(amount.max()) if len(amount) else 0.0
        amount_min = current_min if amount_min is None else min(amount_min, current_min)
        amount_max = current_max if amount_max is None else max(amount_max, current_max)

        _accumulate_counts(fraud_counts, chunk["fraud_label"].fillna(0).astype(int).astype(str))
        _accumulate_counts(type_counts, chunk["transaction_type"].fillna("Unknown").astype(str))
        _accumulate_counts(device_counts, chunk["device_type"].fillna("Unknown").astype(str))
        _accumulate_counts(merchant_counts, chunk["merchant_category"].fillna("Unknown").astype(str))
        _accumulate_counts(location_counts, chunk["location"].fillna("Unknown").astype(str))
        _accumulate_counts(hour_counts, pd.to_datetime(chunk["timestamp"], errors="coerce", format="mixed").dt.hour.dropna().astype(int).astype(str))

        amount_bins["0-500"] += int(amount.between(0, 500).sum())
        amount_bins["501-2k"] += int(amount.between(501, 2_000).sum())
        amount_bins["2k-10k"] += int(amount.between(2_001, 10_000).sum())
        amount_bins["10k-50k"] += int(amount.between(10_001, 50_000).sum())
        amount_bins["50k+"] += int((amount > 50_000).sum())

    fraud_count = fraud_counts.get("1", 0)
    legitimate_count = fraud_counts.get("0", 0)
    average_amount = amount_sum / total_rows if total_rows else 0.0
    transaction_distribution = build_transaction_distribution(
        sampled_amounts,
        sampled_labels,
        represented_rows=total_rows,
        minimum=amount_min or 0.0,
        maximum=amount_max or 0.0,
    )

    return {
        "ready": True,
        "summary": {
            "total_rows": total_rows,
            "fraud_rows": fraud_count,
            "legitimate_rows": legitimate_count,
            "fraud_rate": fraud_count / total_rows if total_rows else 0.0,
            "average_amount": average_amount,
            "min_amount": amount_min or 0.0,
            "max_amount": amount_max or 0.0,
        },
        "transaction_types": _top_items(type_counts, limit=8),
        "device_types": _top_items(device_counts, limit=6),
        "merchant_categories": _top_items(merchant_counts, limit=8),
        "locations": _top_items(location_counts, limit=8),
        "hourly_volume": [{"label": hour, "value": hour_counts[str(hour)]} for hour in range(24)],
        "amount_bins": [{"label": label, "value": value} for label, value in amount_bins.items()],
        "transaction_distribution": transaction_distribution,
        "reports": sorted(path.name for path in REPORTS_DIR.glob("*.png")),
    }


def build_transaction_distribution(
    amount_chunks: list[np.ndarray],
    label_chunks: list[np.ndarray],
    represented_rows: int | None = None,
    minimum: float | None = None,
    maximum: float | None = None,
    bin_count: int = 40,
) -> dict[str, Any]:
    """Summarize the full population with a bounded chart sample.

    Population totals, minimum, and maximum are accumulated from every chunk.
    Quantiles and line points use a deterministic bounded sample so the API does
    not retain millions of amounts just to render a browser visualization.
    """
    if not amount_chunks:
        return {"ready": False, "bins": []}

    amounts = np.concatenate(amount_chunks)
    labels = np.concatenate(label_chunks).astype(np.int8, copy=False)
    amounts = np.nan_to_num(amounts, nan=0.0, posinf=0.0, neginf=0.0)
    if amounts.size == 0:
        return {"ready": False, "bins": []}

    median = float(np.median(amounts))
    first_quartile, third_quartile = np.percentile(amounts, [25, 75])
    first_quartile = float(first_quartile)
    third_quartile = float(third_quartile)
    minimum = float(amounts.min()) if minimum is None else float(minimum)
    maximum = float(amounts.max()) if maximum is None else float(maximum)
    sorted_amounts = np.sort(amounts)
    plot_size = min(1200, sorted_amounts.size)
    plot_indexes = np.unique(np.linspace(0, sorted_amounts.size - 1, plot_size).astype(int))
    quantile_edges = np.quantile(amounts, np.linspace(0.0, 1.0, bin_count + 1))
    edges = np.unique(quantile_edges)

    if len(edges) < 2:
        edges = np.array([minimum, maximum + 1.0], dtype=np.float64)

    bin_indexes = np.clip(np.searchsorted(edges, amounts, side="right") - 1, 0, len(edges) - 2)
    bins = []
    for index in range(len(edges) - 1):
        mask = bin_indexes == index
        left = float(edges[index])
        right = float(edges[index + 1])
        total = int(mask.sum())
        fraud = int(np.count_nonzero(labels[mask] == 1))
        if right <= first_quartile:
            zone = "legitimate_zone"
        elif left >= third_quartile:
            zone = "fraud_zone"
        else:
            zone = "grey_zone"
        bins.append(
            {
                "index": index,
                "label": f"{left:,.0f} - {right:,.0f}",
                "start": left,
                "end": right,
                "center": (left + right) / 2.0,
                "total": total,
                "legitimate": total - fraud,
                "fraud": fraud,
                "zone": zone,
            }
        )

    return {
        "ready": True,
        "metric": "transaction amount",
        "total_transactions": int(represented_rows or amounts.size),
        "sampled_transactions": int(amounts.size),
        "population_transactions": int(represented_rows or amounts.size),
        "minimum": minimum,
        "maximum": maximum,
        "median": median,
        "grey_area_low": first_quartile,
        "grey_area_high": third_quartile,
        "line_points": [
            {
                "rank": int(index + 1),
                "amount": float(sorted_amounts[index]),
            }
            for index in plot_indexes
        ],
        "line_point_count": int(len(plot_indexes)),
        "line_represents_transactions": int(represented_rows or sorted_amounts.size),
        "bins": bins,
    }


@lru_cache(maxsize=1)
def get_anomaly_reference_scores() -> np.ndarray:
    """Load saved anomaly scores for percentile-based confidence calibration."""
    for filename in ["isolation_forest_anomaly_scores.csv", "lof_anomaly_scores.csv"]:
        path = PROCESSED_DATA_DIR / filename
        if path.exists():
            scores = pd.read_csv(path, usecols=["anomaly_score"])["anomaly_score"]
            return np.sort(pd.to_numeric(scores, errors="coerce").dropna().to_numpy())
    return np.array([])


def calibrate_anomaly_confidence(score: float) -> float:
    """Convert one anomaly score into its percentile against saved training scores."""
    reference = get_anomaly_reference_scores()
    if reference.size == 0:
        return 0.0
    percentile = float(np.searchsorted(reference, score, side="right") / reference.size)
    return max(0.0, min(1.0, percentile))


def _accumulate_counts(target: dict[str, int], values: pd.Series) -> None:
    for key, value in values.value_counts().items():
        target[str(key)] = target.get(str(key), 0) + int(value)


def _top_items(counts: dict[str, int], limit: int) -> list[dict[str, Any]]:
    return [
        {"label": label, "value": value}
        for label, value in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:limit]
    ]
