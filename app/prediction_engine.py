"""Prediction helpers for the Streamlit testing dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.anomaly_detection import predict_anomaly
from src.probability_calibration import (
    apply_isotonic_calibration,
    calibrate_probability_array,
    load_isotonic_calibrator,
    load_score_rank_grid,
    score_to_percentile,
)
from src.schema_mapping import COMMON_SCHEMA
from src.supervised_model import predict_fraud_probability
from src.utils import MODELS_DIR, load_joblib


class PredictionEngine:
    """Load trained models and produce separate supervised/anomaly outputs."""

    def __init__(self, model_dir: Path | str = MODELS_DIR) -> None:
        self.model_dir = Path(model_dir)
        self.supervised_model = self._load_first_available(
            ["xgboost_model.pkl", "random_forest.pkl"]
        )
        self.supervised_model_name = self._find_first_available_name(
            ["xgboost_model.pkl", "random_forest.pkl"]
        )
        self.supervised_preprocessor = self._load_optional("preprocessor.pkl") or self._load_optional("scaler.pkl")
        self.anomaly_model = self._load_optional("isolation_forest.pkl")
        self.anomaly_preprocessor = self._load_optional("anomaly_preprocessor.pkl") or self.supervised_preprocessor
        self.feature_context = self._load_optional("feature_context.pkl") or {}
        self.probability_calibration = self._load_probability_calibration()
        self.isotonic_calibrator = self._load_isotonic_for_active_model()
        self.alert_threshold = self._resolve_alert_threshold()
        self.rank_grid = self._load_rank_grid_for_active_model()

    @property
    def is_ready(self) -> bool:
        """Return True when all dashboard predictions can run."""
        return all(
            [
                self.supervised_model is not None,
                self.supervised_preprocessor is not None,
                self.anomaly_model is not None,
                self.anomaly_preprocessor is not None,
                bool(self.feature_context),
                bool(self.probability_calibration),
            ]
        )

    def predict(self, transaction: dict[str, Any]) -> dict[str, pd.DataFrame]:
        """Run supervised fraud and unsupervised anomaly prediction separately."""
        if not self.is_ready:
            raise FileNotFoundError(
                "Trained models are missing. Run `python main.py --all` after placing datasets in data/raw/."
            )

        frame = transaction_to_common_schema(transaction)
        supervised = predict_fraud_probability(
            self.supervised_model,
            self.supervised_preprocessor,
            frame,
            feature_context=self.feature_context,
        )
        if supervised.empty or "fraud_probability" not in supervised.columns:
            raise RuntimeError("The supervised model returned no fraud probability.")
        raw_probability = supervised["fraud_probability"].to_numpy(dtype=np.float64)
        if self.isotonic_calibrator is not None:
            # Isotonic maps raw scores to real-world fraud probabilities using
            # a held-out natural-prevalence calibration set.
            calibrated = apply_isotonic_calibration(raw_probability, self.isotonic_calibrator)
            calibration_method = "isotonic"
        else:
            # Legacy fallback for artifacts trained before isotonic calibration.
            calibrated = calibrate_probability_array(
                raw_probability, self.probability_calibration
            )
            calibration_method = "analytic prior correction (legacy)"
        supervised["fraud_probability"] = calibrated
        if self.rank_grid is not None:
            # Population rank of the calibrated score; fuse_signals compares
            # this percentile against the anomaly percentile (like with like).
            supervised["fraud_probability_rank"] = score_to_percentile(
                self.rank_grid["grid"], calibrated
            )
        supervised["fraud_prediction"] = (
            calibrated >= self.alert_threshold
        ).astype(int)
        supervised["confidence_score"] = np.maximum(calibrated, 1.0 - calibrated)
        supervised["model_name"] = self.supervised_model_name or "supervised_model"
        supervised["calibration_method"] = calibration_method
        supervised["signal_status"] = "calculated"
        anomaly = predict_anomaly(
            self.anomaly_model,
            self.anomaly_preprocessor,
            frame,
            feature_context=self.feature_context,
        )
        return {"supervised": supervised, "anomaly": anomaly}

    def _load_optional(self, filename: str) -> Any | None:
        path = self.model_dir / filename
        if not path.exists():
            return None
        return load_joblib(path)

    def _load_probability_calibration(self) -> dict[str, Any]:
        path = self.model_dir / "fraud_probability_calibration.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def _load_isotonic_for_active_model(self) -> Any | None:
        """Load the calibrator matching the supervised artifact actually in use."""
        if not self.supervised_model_name:
            return None
        return load_isotonic_calibrator(
            self.model_dir / f"{self.supervised_model_name}_calibrator.pkl"
        )

    def _load_rank_grid_for_active_model(self) -> dict[str, Any] | None:
        """Load the score rank grid when it matches the active artifact."""
        if not self.supervised_model_name:
            return None
        payload = load_score_rank_grid(self.model_dir / "supervised_rank_grid.json")
        if payload is None:
            return None
        if payload.get("artifact") != f"{self.supervised_model_name}.pkl":
            return None
        return payload

    def _resolve_alert_threshold(self) -> float:
        """Prefer the tuned per-model threshold, then the shared one, then 0.5."""
        calibration = self.probability_calibration or {}
        model_key = "xgboost" if self.supervised_model_name == "xgboost_model" else "random_forest"
        per_model = (calibration.get("per_model") or {}).get(model_key) or {}
        threshold = per_model.get("alert_threshold", calibration.get("supervised_alert_threshold"))
        try:
            resolved = float(threshold)
        except (TypeError, ValueError):
            return 0.5
        if not np.isfinite(resolved):
            return 0.5
        return float(np.clip(resolved, 1e-6, 1.0 - 1e-6))

    def _load_first_available(self, filenames: list[str]) -> Any | None:
        for filename in filenames:
            model = self._load_optional(filename)
            if model is not None:
                return model
        return None

    def _find_first_available_name(self, filenames: list[str]) -> str | None:
        """Return the artifact name used for the supervised prediction."""
        for filename in filenames:
            if (self.model_dir / filename).exists():
                return filename.removesuffix(".pkl")
        return None


def transaction_to_common_schema(transaction: dict[str, Any]) -> pd.DataFrame:
    """Convert a dashboard input dictionary into the common schema dataframe."""
    row = {
        "transaction_id": transaction.get("transaction_id", "manual_test_0001"),
        "timestamp": pd.to_datetime(transaction.get("timestamp"), errors="coerce"),
        "amount": float(transaction.get("amount", 0.0)),
        "sender_id": str(transaction.get("sender_id", "Unknown")),
        "receiver_id": str(transaction.get("receiver_id", "Unknown")),
        "device_type": str(transaction.get("device_type", "Unknown")),
        "merchant_category": str(transaction.get("merchant_category", "Unknown")),
        "location": str(transaction.get("location", "Unknown")),
        "transaction_type": str(transaction.get("transaction_type", "Unknown")),
        "fraud_label": int(transaction.get("fraud_label", 0)),
    }
    frame = pd.DataFrame([row], columns=COMMON_SCHEMA)
    frame["timestamp"] = frame["timestamp"].fillna(pd.Timestamp.now())
    frame["amount"] = frame["amount"].replace([np.inf, -np.inf], 0).fillna(0)
    return frame
