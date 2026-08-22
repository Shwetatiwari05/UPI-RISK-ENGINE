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


# Defaults describe the untuned behaviour. After a training run, main.py sweeps
# these operating points on out-of-time data and persists overrides to
# models/fusion_thresholds.json, which the API passes back in here.
DEFAULT_FUSION_THRESHOLDS: dict[str, float] = {
    # Fused-score boundary for the strong FRAUD_LIKELY label.
    "fraud_likely_fusion_threshold": 0.60,
    # Calibrated supervised evidence floor required alongside the fused score.
    # Honest probabilities at ~0.4% prevalence rarely reach 0.50, so this gate
    # is intentionally far below the naive 0.5 decision boundary.
    "fraud_likely_requires_supervised_probability": 0.10,
    # Middle band: elevated combined concern that stops short of FRAUD_LIKELY.
    "ambiguous_review_fusion_threshold": 0.45,
    "ambiguous_score_threshold": 0.38,
    "ambiguous_disagreement_threshold": 0.45,
}

REVIEW_BAND_RATIO = 0.80


def resolve_thresholds(overrides: dict[str, Any] | None = None) -> dict[str, float]:
    """Merge persisted/one-off threshold overrides onto the defaults."""
    merged = dict(DEFAULT_FUSION_THRESHOLDS)
    if overrides:
        for key, value in overrides.items():
            if key in merged:
                merged[key] = _unit_interval(value)
    return merged


def fuse_signals(
    supervised: dict[str, Any],
    anomaly: dict[str, Any],
    diagnostics: dict[str, Any] | None = None,
    thresholds: dict[str, Any] | None = None,
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
    resolved_thresholds = resolve_thresholds(thresholds)

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

    fraud_likely_threshold = resolved_thresholds["fraud_likely_fusion_threshold"]
    supervised_gate = resolved_thresholds["fraud_likely_requires_supervised_probability"]
    review_threshold = resolved_thresholds["ambiguous_review_fusion_threshold"]

    if ambiguity_score >= resolved_thresholds["ambiguous_score_threshold"] or (
        disagreement >= resolved_thresholds["ambiguous_disagreement_threshold"]
    ):
        resolution = "AMBIGUOUS_REVIEW"
        resolution_text = "Falls within the ambiguous research-review band"
    elif fusion_score >= fraud_likely_threshold and fraud_probability >= supervised_gate:
        resolution = "FRAUD_LIKELY"
        resolution_text = "Fraud signal is elevated across the combined evidence"
    elif fusion_score >= fraud_likely_threshold:
        # Anomaly unusualness must not promote a below-gate supervised result
        # directly to fraud-likely; retain the disagreement as review.
        resolution = "AMBIGUOUS_REVIEW"
        resolution_text = "Unusual behaviour is present without a supervised fraud signal"
    elif fusion_score >= review_threshold:
        resolution = "AMBIGUOUS_REVIEW"
        resolution_text = "Combined concern is elevated enough for research review"
    else:
        resolution = "LIKELY_LEGITIMATE"
        resolution_text = "Combined evidence is currently closer to legitimate behaviour"

    return {
        "fusion_score": round(float(fusion_score), 6),
        "ambiguity_score": round(float(ambiguity_score), 6),
        "resolution": resolution,
        "resolution_text": resolution_text,
        "signal_disagreement": round(float(disagreement), 6),
        "supervised_uncertainty": round(float(supervised_uncertainty), 6),
        "repeatability_penalty": round(float(repeatability_penalty), 6),
        "weights": FUSION_WEIGHTS.copy(),
        "thresholds": resolved_thresholds,
        "method": "Weighted dual-signal score with disagreement and uncertainty review band",
    }


def tune_fusion_thresholds(
    supervised_probabilities: Any,
    anomaly_percentiles: Any,
    y_true: Any,
    min_fraud_precision: float = 0.40,
    review_band_ratio: float = REVIEW_BAND_RATIO,
    gate_quantiles: int = 24,
) -> dict[str, Any]:
    """Sweep operating points on out-of-time data and pick FRAUD_LIKELY thresholds.

    Simulates the exact ``fuse_signals`` resolution path (including the
    ambiguity-first gate, minus the single-transaction repeatability penalty
    that has no meaning in batch scoring). For a grid of calibrated
    supervised-probability gates, the fused-score boundary is swept to find the
    maximum-recall point whose precision meets ``min_fraud_precision`` under
    the joint rule ``not ambiguous and fusion_score >= T and
    fraud_probability >= gate``. When no combination reaches the floor, the
    best-F1 point is returned with ``target_precision_met=False``. The review
    band is set proportionally below the chosen fraud-likely threshold.
    """
    p = np.asarray(supervised_probabilities, dtype=np.float64).ravel()
    unusualness = np.asarray(anomaly_percentiles, dtype=np.float64).ravel()
    y = np.asarray(y_true, dtype=np.int64).ravel()
    if not (len(p) == len(unusualness) == len(y)) or len(y) == 0:
        raise ValueError("Threshold tuning inputs must be non-empty and equally sized.")
    unique_labels = np.unique(y)
    if len(unique_labels) < 2:
        raise ValueError("Threshold tuning needs evaluation rows of both classes.")

    p_clipped = np.clip(p, 0.0, 1.0)
    unusualness_clipped = np.clip(unusualness, 0.0, 1.0)
    fusion_scores = (
        FUSION_WEIGHTS["supervised_fraud_probability"] * p_clipped
        + FUSION_WEIGHTS["unsupervised_unusualness_percentile"] * unusualness_clipped
    )
    # Reproduce the ambiguity-first gate from fuse_signals (repeatability
    # penalty is zero in batch scoring).
    signal_disagreement = np.abs(p_clipped - unusualness_clipped)
    supervised_uncertainty = 1.0 - np.abs(2.0 * p_clipped - 1.0)
    ambiguity_scores = (
        0.50 * signal_disagreement + 0.35 * supervised_uncertainty
    )
    ambiguous_first = (
        ambiguity_scores >= DEFAULT_FUSION_THRESHOLDS["ambiguous_score_threshold"]
    ) | (
        signal_disagreement
        >= DEFAULT_FUSION_THRESHOLDS["ambiguous_disagreement_threshold"]
    )
    prevalence = float(y.mean())
    total_positives = max(int(y.sum()), 1)

    gates = np.unique(
        np.quantile(p, np.linspace(0.50, 0.999, gate_quantiles))
    )
    gates = gates[gates > 0.0]
    if gates.size == 0:
        gates = np.asarray([max(2.0 * prevalence, 1e-4)], dtype=np.float64)

    best: dict[str, Any] | None = None
    fallback: dict[str, Any] | None = None
    for gate in gates:
        gate_mask = (p >= gate) & ~ambiguous_first
        flagged_cap = int(gate_mask.sum())
        if flagged_cap < 10:
            continue
        scores = fusion_scores[gate_mask]
        labels = y[gate_mask]
        order = np.argsort(-scores, kind="mergesort")
        sorted_labels = labels[order]
        true_positive = np.cumsum(sorted_labels)
        false_positive = np.cumsum(1 - sorted_labels)
        precision = true_positive / np.maximum(true_positive + false_positive, 1)
        recall = true_positive / total_positives
        f1 = np.where(
            precision + recall > 0,
            2 * precision * recall / np.maximum(precision + recall, 1e-12),
            0.0,
        )
        sorted_scores = scores[order]

        def _entry(index: int, met: bool) -> dict[str, Any]:
            upper = sorted_scores[index - 1] if index > 0 else sorted_scores[index] + 1.0
            boundary = float((upper + sorted_scores[index]) / 2.0)
            return {
                "gate": float(gate),
                "fusion_threshold": round(float(np.clip(boundary, 0.0, 1.0)), 6),
                "precision": round(float(precision[index]), 6),
                "recall": round(float(recall[index]), 6),
                "f1": round(float(f1[index]), 6),
                "alerts_per_10k_rows": round((index + 1) / len(y) * 10_000, 2),
                "target_precision_met": met,
            }

        feasible = np.flatnonzero(precision >= min_fraud_precision)
        if feasible.size:
            candidate = _entry(int(feasible[-1]), True)
            if best is None or (candidate["recall"], -candidate["alerts_per_10k_rows"]) > (
                best["recall"],
                -best["alerts_per_10k_rows"],
            ):
                best = candidate
        else:
            candidate = _entry(int(np.argmax(f1)), False)
            if fallback is None or (candidate["f1"], candidate["precision"]) > (
                fallback["f1"],
                fallback["precision"],
            ):
                fallback = candidate

    chosen = best or fallback
    if chosen is None:
        raise ValueError("No viable threshold found; check the evaluation inputs.")

    fraud_threshold = chosen["fusion_threshold"]
    review_threshold = round(fraud_threshold * review_band_ratio, 6)
    fraud_band = (
        ~ambiguous_first
        & (fusion_scores >= fraud_threshold)
        & (p >= chosen["gate"])
    )
    review_mask = (
        (ambiguous_first | ((fusion_scores >= review_threshold) & (p < chosen["gate"])))
        & ~fraud_band
    )

    return {
        "thresholds": {
            "fraud_likely_fusion_threshold": fraud_threshold,
            "fraud_likely_requires_supervised_probability": round(chosen["gate"], 8),
            "ambiguous_review_fusion_threshold": review_threshold,
            "ambiguous_score_threshold": DEFAULT_FUSION_THRESHOLDS["ambiguous_score_threshold"],
            "ambiguous_disagreement_threshold": DEFAULT_FUSION_THRESHOLDS[
                "ambiguous_disagreement_threshold"
            ],
        },
        "achieved": {
            "fraud_likely": {
                key: value
                for key, value in chosen.items()
                if key in {"precision", "recall", "f1", "alerts_per_10k_rows"}
            },
            "review_band": {
                "rows": int(review_mask.sum()),
                "fraud_rows": int(y[review_mask].sum()),
                "precision": round(
                    float(y[review_mask].mean()) if review_mask.any() else 0.0, 6
                ),
            },
        },
        "evaluation_rows": int(len(y)),
        "fraud_prevalence": round(prevalence, 8),
        "min_fraud_precision_target": round(float(min_fraud_precision), 4),
        "target_precision_met": bool(chosen["target_precision_met"]),
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
    thresholds = fusion.get("thresholds") or DEFAULT_FUSION_THRESHOLDS
    supervised_gate = _unit_interval(
        thresholds.get("fraud_likely_requires_supervised_probability", 0.10)
    )
    fraud_likely_threshold_pct = _unit_interval(
        thresholds.get("fraud_likely_fusion_threshold", 0.60)
    )
    review_threshold_pct = _unit_interval(
        thresholds.get("ambiguous_review_fusion_threshold", 0.45)
    )
    has_supervised_signal = fraud_probability >= supervised_gate
    fraud_label = "fraud" if has_supervised_signal else "legitimate"
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
            f"is {fusion_score:.1%}, reaching the {fraud_likely_threshold_pct:.1%} fraud-likely "
            f"threshold while the supervised probability clears the {supervised_gate:.1%} evidence "
            f"gate. The supervised fraud probability is {fraud_probability:.1%}, and the anomaly "
            f"model reports the {anomaly_percentile:.1%} unusualness percentile ({anomaly_label})."
        )
    else:
        resolution_reason = (
            f"The transaction currently appears closer to legitimate behaviour because the combined "
            f"score is {fusion_score:.1%}, below both the {review_threshold_pct:.1%} review and "
            f"{fraud_likely_threshold_pct:.1%} fraud-likely thresholds. The supervised model "
            f"assigns {fraud_probability:.1%} fraud probability ({fraud_label}) and the anomaly "
            f"model reports the {anomaly_percentile:.1%} unusualness percentile ({anomaly_label})."
        )

    return {
        "resolution_reason": resolution_reason,
        "supervised_reason": (
            f"Supervised output: {fraud_probability:.1%} fraud probability against a "
            f"{supervised_gate:.1%} calibrated-evidence gate; supervised signal is "
            f"{'present' if has_supervised_signal else 'absent'}."
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
