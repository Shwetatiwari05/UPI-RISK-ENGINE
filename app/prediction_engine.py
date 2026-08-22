"""Prediction helpers for the Streamlit testing dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.anomaly_detection import predict_anomaly
from src.feature_engineering import engineer_features
from src.live_history import (
    DEFAULT_LIVE_HISTORY_DB_PATH,
    UPI_ABSOLUTE_MAX_AMOUNT,
    fetch_sender_summary,
    record_transaction,
)
from src.probability_calibration import (
    apply_isotonic_calibration,
    calibrate_probability_array,
    load_isotonic_calibrator,
    load_score_rank_grid,
    score_to_percentile,
)
from src.schema_mapping import COMMON_SCHEMA
from src.supervised_model import predict_fraud_probability
from src.utils import MODELS_DIR, load_joblib, safe_datetime


class PredictionEngine:
    """Load trained models and produce separate supervised/anomaly outputs."""

    def __init__(
        self,
        model_dir: Path | str = MODELS_DIR,
        live_history_db_path: Path | str = DEFAULT_LIVE_HISTORY_DB_PATH,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.live_history_db_path = Path(live_history_db_path)
        self.supervised_model = self._load_first_available(
            ["xgboost_model.pkl", "random_forest.pkl"]
        )
        self.supervised_model_name = self._find_first_available_name(
            ["xgboost_model.pkl", "random_forest.pkl"]
        )
        self.supervised_preprocessor = self._load_optional("preprocessor.pkl") or self._load_optional("scaler.pkl")
        self.anomaly_model = self._load_optional("isolation_forest.pkl")
        self.anomaly_preprocessor = self._load_optional("anomaly_preprocessor.pkl") or self.supervised_preprocessor
        self.feature_context = dict(self._load_optional("feature_context.pkl") or {})
        self.global_amount_moments = self._derive_global_amount_moments()
        if self.global_amount_moments is not None and self.feature_context:
            self.feature_context["global_amount_std"] = self.global_amount_moments["std"]
            self.feature_context.setdefault(
                "global_amount_mean", self.global_amount_moments["mean"]
            )
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

    def predict(
        self,
        transaction: dict[str, Any],
        record_history: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """Run supervised fraud and unsupervised anomaly prediction separately.

        ``record_history=False`` skips the live-history write for auxiliary
        scoring runs such as sensitivity diagnostics, so synthetic variants
        never pollute a sender's real history.
        """
        if not self.is_ready:
            raise FileNotFoundError(
                "Trained models are missing. Run `python main.py --all` after placing datasets in data/raw/."
            )

        frame = transaction_to_common_schema(transaction)
        sender_id = str(frame.at[0, "sender_id"])
        live_summary = fetch_sender_summary(sender_id, db_path=self.live_history_db_path)
        live_context = None
        if live_summary is not None:
            live_context = {sender_id: live_summary}
        supervised = predict_fraud_probability(
            self.supervised_model,
            self.supervised_preprocessor,
            frame,
            feature_context=self.feature_context,
            live_context=live_context,
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
        amount_value = float(frame.at[0, "amount"])
        exceeds_realistic_bound = bool(amount_value > UPI_ABSOLUTE_MAX_AMOUNT)
        supervised["exceeds_realistic_amount_bound"] = int(exceeds_realistic_bound)
        supervised["live_history_count"] = int(live_summary["count"]) if live_summary else 0
        transparency_frame = engineer_features(
            frame, context=self.feature_context, live_context=live_context
        )
        for column in (
            "avg_transaction_amount",
            "transaction_frequency",
            "amount_spike",
            "new_payee_flag",
            "unusual_location_flag",
            "rapid_transactions",
            "minutes_since_previous_sender_txn",
        ):
            supervised[column] = transparency_frame.at[0, column]
        anomaly = predict_anomaly(
            self.anomaly_model,
            self.anomaly_preprocessor,
            frame,
            feature_context=self.feature_context,
            live_context=live_context,
        )
        if record_history:
            record_transaction(
                sender_id=sender_id,
                receiver_id=str(frame.at[0, "receiver_id"]),
                amount=amount_value,
                location=str(frame.at[0, "location"]),
                timestamp=frame.at[0, "timestamp"],
                transaction_id=str(frame.at[0, "transaction_id"]),
                fraud_probability=float(calibrated[0]),
                fraud_prediction=int(supervised.at[0, "fraud_prediction"]),
                db_path=self.live_history_db_path,
            )
        return {"supervised": supervised, "anomaly": anomaly}

    def _derive_global_amount_moments(self) -> dict[str, float] | None:
        """Derive population amount moments from stored per-sender aggregates.

        Uses the moment identity Var = E[X^2] - E[X]^2 with the per-sender
        frequency/mean/std values already held in the frozen feature context,
        so the training pipeline stays untouched. Returns None when the
        snapshot holds no usable aggregates.
        """
        sender_stats = self.feature_context.get("sender_stats")
        if not isinstance(sender_stats, dict) or not sender_stats:
            return None
        total_frequency = 0
        weighted_mean_sum = 0.0
        weighted_second_moment = 0.0
        for stats in sender_stats.values():
            if not isinstance(stats, dict):
                continue
            try:
                frequency = int(stats.get("transaction_frequency", 0))
                mean = float(stats.get("average_amount", 0.0))
                std = float(stats.get("amount_std", 0.0))
            except (TypeError, ValueError):
                continue
            if frequency <= 0:
                continue
            total_frequency += frequency
            weighted_mean_sum += frequency * mean
            weighted_second_moment += frequency * (std * std + mean * mean)
        if total_frequency <= 0:
            return None
        pooled_mean = weighted_mean_sum / total_frequency
        pooled_variance = max(
            weighted_second_moment / total_frequency - pooled_mean * pooled_mean,
            0.0,
        )
        return {"mean": pooled_mean, "std": float(np.sqrt(pooled_variance))}

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
    frame["timestamp"] = safe_datetime(frame["timestamp"])
    frame["timestamp"] = frame["timestamp"].fillna(pd.Timestamp.now())
    frame["amount"] = frame["amount"].replace([np.inf, -np.inf], 0).fillna(0)
    return frame
