"""Explainable research fusion for ambiguous transaction analysis.

This module combines already-trained supervised and unsupervised outputs. It is
not a third classifier and must not be treated as a production risk decision.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np


FUSION_WEIGHTS = {
    "supervised_fraud_probability": 0.60,
    "unsupervised_unusualness_percentile": 0.40,
}


def fuse_signals(
    supervised: dict[str, Any],
    anomaly: dict[str, Any],
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a transparent score and an ambiguity-oriented resolution.

    The fusion score measures combined concern. The ambiguity score intentionally
    measures a different idea: disagreement between the model families and
    uncertainty around the supervised decision boundary. A high ambiguity score
    sends the transaction to a research-review band instead of calling it an
    anomaly or making a banking decision.
    """
    fraud_probability = _unit_interval(supervised.get("fraud_probability", 0.0))
    unusualness_percentile = _unit_interval(
        anomaly.get("anomaly_percentile", anomaly.get("anomaly_confidence", 0.0))
    )

    disagreement = abs(fraud_probability - unusualness_percentile)
    supervised_uncertainty = 1.0 - abs((2.0 * fraud_probability) - 1.0)
    repeatability_penalty = _repeatability_penalty(diagnostics)
    fusion_score = (
        FUSION_WEIGHTS["supervised_fraud_probability"] * fraud_probability
        + FUSION_WEIGHTS["unsupervised_unusualness_percentile"] * unusualness_percentile
    )
    ambiguity_score = (
        0.50 * disagreement
        + 0.35 * supervised_uncertainty
        + 0.15 * repeatability_penalty
    )

    if ambiguity_score >= 0.38 or disagreement >= 0.45:
        resolution = "AMBIGUOUS_REVIEW"
        resolution_text = "Falls within the ambiguous research-review band"
    elif fraud_probability < 0.50 and fusion_score >= 0.60:
        # Anomaly unusualness must not promote a below-threshold supervised
        # result directly to fraud-likely; retain the disagreement as review.
        resolution = "AMBIGUOUS_REVIEW"
        resolution_text = "Unusual behaviour is present without a supervised fraud signal"
    elif fusion_score >= 0.60:
        resolution = "FRAUD_LIKELY"
        resolution_text = "Fraud signal is elevated across the combined evidence"
    else:
        resolution = "LIKELY_LEGITIMATE"
        resolution_text = "Combined evidence is currently closer to legitimate behaviour"

    thresholds = {
        "supervised_fraud_threshold": 0.50,
        "fraud_likely_fusion_threshold": 0.60,
        "ambiguous_score_threshold": 0.38,
        "ambiguous_disagreement_threshold": 0.45,
        "fraud_likely_requires_supervised_threshold": 0.50,
    }

    return {
        "fusion_score": round(float(fusion_score), 6),
        "ambiguity_score": round(float(ambiguity_score), 6),
        "resolution": resolution,
        "resolution_text": resolution_text,
        "signal_disagreement": round(float(disagreement), 6),
        "supervised_uncertainty": round(float(supervised_uncertainty), 6),
        "repeatability_penalty": round(float(repeatability_penalty), 6),
        "weights": FUSION_WEIGHTS.copy(),
        "thresholds": thresholds,
        "method": "Weighted dual-signal score with disagreement and uncertainty review band",
    }


def build_transaction_report(
    transaction: dict[str, Any],
    supervised: dict[str, Any],
    anomaly: dict[str, Any],
    fusion: dict[str, Any],
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-safe, downloadable post-transaction research report."""
    reasoning = build_resolution_reasoning(transaction, supervised, anomaly, fusion, diagnostics)
    return {
        "report_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_scope": "Offline post-transaction research analysis only",
        "transaction": _json_safe(transaction),
        "supervised_model_output": _json_safe(supervised),
        "unsupervised_model_output": _json_safe(anomaly),
        "fusion_resolution": _json_safe(fusion),
        "model_evidence": _json_safe(diagnostics or {}),
        "reasoning": _json_safe(reasoning),
        "interpretation": (
            "The resolution is a research score for comparing supervised fraud "
            "probability and unsupervised unusualness. It is not an approval, "
            "decline, block, or proof of fraud."
        ),
    }


def build_resolution_reasoning(
    transaction: dict[str, Any],
    supervised: dict[str, Any],
    anomaly: dict[str, Any],
    fusion: dict[str, Any],
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Explain the resolution using model outputs and observable input facts."""
    fraud_probability = _unit_interval(supervised.get("fraud_probability", 0.0))
    anomaly_percentile = _unit_interval(
        anomaly.get("anomaly_percentile", anomaly.get("anomaly_confidence", 0.0))
    )
    fusion_score = _unit_interval(fusion.get("fusion_score", 0.0))
    ambiguity_score = _unit_interval(fusion.get("ambiguity_score", 0.0))
    disagreement = _unit_interval(fusion.get("signal_disagreement", 0.0))
    resolution = str(fusion.get("resolution", "UNKNOWN"))
    fraud_label = "fraud" if fraud_probability >= 0.50 else "legitimate"
    anomaly_label = str(anomaly.get("anomaly_label", "Unknown")).lower()
    amount = float(transaction.get("amount", 0.0) or 0.0)
    hour = _transaction_hour(transaction.get("timestamp"))
    input_factors = []

    if amount >= 50_000:
        input_factors.append(f"High-value amount of {amount:,.2f}")
    if hour is not None and 0 <= hour <= 5:
        input_factors.append(f"Unusual transaction hour of {hour:02d}:00")
    if str(transaction.get("transaction_type", "")).upper() == "CASH_OUT":
        input_factors.append("Cash-out transaction type")
    if str(transaction.get("device_type", "")).strip().lower() in {"", "unknown", "nan"}:
        input_factors.append("Unknown device type")
    if str(transaction.get("location", "")).strip().lower() in {"", "unknown", "nan"}:
        input_factors.append("Unknown location")
    if not input_factors:
        input_factors.append("No simple rule-based warning factor was present in the submitted fields")

    if resolution == "AMBIGUOUS_REVIEW":
        resolution_reason = (
            f"The transaction is ambiguous because the combined score is {fusion_score:.1%}, "
            f"while the ambiguity score is {ambiguity_score:.1%}. The supervised model gives a "
            f"{fraud_probability:.1%} fraud probability ({fraud_label}), whereas the anomaly model "
            f"places it at the {anomaly_percentile:.1%} unusualness percentile ({anomaly_label}); "
            f"their disagreement is {disagreement:.1%}."
        )
    elif resolution == "FRAUD_LIKELY":
        resolution_reason = (
            f"The transaction is fraud-likely in this research experiment because the combined score "
            f"is {fusion_score:.1%}, reaching the 60.0% fraud-likely threshold. The supervised fraud "
            f"probability is {fraud_probability:.1%}, and the anomaly model reports the "
            f"{anomaly_percentile:.1%} unusualness percentile ({anomaly_label})."
        )
    else:
        resolution_reason = (
            f"The transaction currently appears closer to legitimate behaviour because the combined "
            f"score is {fusion_score:.1%}, below the 60.0% fraud-likely threshold. The supervised model "
            f"assigns {fraud_probability:.1%} fraud probability ({fraud_label}) and the anomaly model "
            f"reports the {anomaly_percentile:.1%} unusualness percentile ({anomaly_label})."
        )

    return {
        "resolution_reason": resolution_reason,
        "supervised_reason": (
            f"Supervised output: {fraud_probability:.1%} fraud probability, with a 50.0% "
            f"classification threshold; result is {fraud_label}."
        ),
        "anomaly_reason": (
            f"Unsupervised output: raw anomaly score {float(anomaly.get('anomaly_score', 0.0)):.4f}, "
            f"{anomaly_percentile:.1%} training-score percentile, model label {anomaly_label}."
        ),
        "supporting_numbers": {
            "amount": amount,
            "fraud_probability": round(fraud_probability, 6),
            "anomaly_score": round(float(anomaly.get("anomaly_score", 0.0)), 6),
            "anomaly_percentile": round(anomaly_percentile, 6),
            "fusion_score": round(fusion_score, 6),
            "ambiguity_score": round(ambiguity_score, 6),
            "signal_disagreement": round(disagreement, 6),
        },
        "input_factors": input_factors,
        "sensitivity_note": _sensitivity_note(diagnostics),
    }


def _transaction_hour(value: Any) -> int | None:
    """Extract the transaction hour without failing report generation."""
    try:
        text = str(value)
        return int(text[11:13])
    except (TypeError, ValueError, IndexError):
        return None


def _sensitivity_note(diagnostics: dict[str, Any] | None) -> str:
    """Summarize controlled input sensitivity when diagnostics are available."""
    sensitivity = (diagnostics or {}).get("sensitivity", [])
    if not sensitivity:
        return "No controlled sensitivity run was available."
    largest = max(sensitivity, key=lambda item: abs(float(item.get("fraud_delta", 0.0))))
    return (
        f"The largest controlled fraud-probability change was {float(largest.get('fraud_delta', 0.0)):+.1%} "
        f"under the {largest.get('name', 'tested variation')} variation."
    )


def _repeatability_penalty(diagnostics: dict[str, Any] | None) -> float:
    """Map same-input repeat deltas to a bounded numerical-stability penalty."""
    deterministic = (diagnostics or {}).get("deterministic", {})
    fraud_delta = abs(float(deterministic.get("repeat_fraud_delta", 0.0)))
    anomaly_delta = abs(float(deterministic.get("repeat_anomaly_delta", 0.0)))
    return _unit_interval((fraud_delta + anomaly_delta) / 2.0)


def _unit_interval(value: Any) -> float:
    """Coerce a scalar into the inclusive 0-1 interval."""
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(numeric_value):
        return 0.0
    return float(np.clip(numeric_value, 0.0, 1.0))


def _json_safe(value: Any) -> Any:
    """Convert common numpy/pandas scalar values into JSON-compatible values."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value
