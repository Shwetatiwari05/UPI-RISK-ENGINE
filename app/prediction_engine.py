"""Prediction helpers for the Streamlit testing dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.anomaly_detection import predict_anomaly
from src.probability_calibration import calibrate_probability_array
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
        supervised["fraud_probability"] = calibrate_probability_array(
            supervised["fraud_probability"].to_numpy(),
            self.probability_calibration,
        )
        supervised["fraud_prediction"] = (supervised["fraud_probability"] >= 0.5).astype(int)
        supervised["confidence_score"] = np.maximum(
            supervised["fraud_probability"],
            1.0 - supervised["fraud_probability"],
        )
        supervised["model_name"] = self.supervised_model_name or "supervised_model"
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
