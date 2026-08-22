"""Explainable research fusion for ambiguous transaction analysis.

This module combines already-trained supervised and unsupervised outputs. It is
not a third classifier and must not be treated as a production risk decision.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np

from src.live_history import UPI_ABSOLUTE_MAX_AMOUNT


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

INFORMED_RANK_FLOOR = 0.95
LOGIT_CLIP = 1e-6

AMBIGUITY_WEIGHTS = {
    "disagreement": 0.50,
    "supervised_uncertainty": 0.35,
    "repeatability_penalty": 0.15,
}


def resolve_thresholds(overrides: dict[str, Any] | None = None) -> dict[str, float]:
    """Merge persisted/one-off threshold overrides onto the defaults."""
    merged = dict(DEFAULT_FUSION_THRESHOLDS)
    if overrides:
        for key, value in overrides.items():
            if key in merged:
                merged[key] = _unit_interval(value)
    return merged


def ambiguity_components(
    fraud_probability: float,
    unusualness_percentile: float,
    supervised_rank: float | None,
    supervised_gate: float,
    repeatability_penalty: float = 0.0,
) -> tuple[float, float, float]:
    """Return (disagreement, uncertainty, ambiguity_score) for one transaction.

    When a population rank for the supervised probability is available the
    redesigned scale-aware formulas apply; otherwise the legacy analytic
    behaviour is preserved for older artifacts.
    """
    disagreement_array, uncertainty_array = ambiguity_arrays(
        [fraud_probability],
        [unusualness_percentile],
        None if supervised_rank is None else [supervised_rank],
        supervised_gate,
    )
    ambiguity_score = combine_ambiguity(
        disagreement_array[0], uncertainty_array[0], repeatability_penalty
    )
    return (
        float(disagreement_array[0]),
        float(uncertainty_array[0]),
        float(ambiguity_score),
    )


def ambiguity_arrays(
    fraud_probabilities: Any,
    unusualness_percentiles: Any,
    supervised_ranks: Any | None,
    supervised_gate: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized (disagreement, uncertainty) matching serving semantics.

    Disagreement compares the supervised probability's population rank against
    the anomaly percentile - like with like - and only counts when at least one
    signal claims top-tier risk (the "informed region"), because two
    near-independent percentiles differ by more than 0.45 roughly a third of
    the time by pure chance. Uncertainty is a Gaussian kernel around the
    supervised evidence gate in logit space; distance from 0.5 was meaningless
    once honest probabilities collapsed to ~0.4% prevalence.
    """
    p = np.clip(np.asarray(fraud_probabilities, dtype=np.float64).ravel(), 0.0, 1.0)
    apct = np.clip(
        np.asarray(unusualness_percentiles, dtype=np.float64).ravel(), 0.0, 1.0
    )
    if supervised_ranks is None:
        return np.abs(p - apct), 1.0 - np.abs((2.0 * p) - 1.0)
    rank = np.clip(np.asarray(supervised_ranks, dtype=np.float64).ravel(), 0.0, 1.0)
    if len(rank) != len(p):
        raise ValueError("Supervised ranks must align with probabilities.")
    informed = np.maximum(rank, apct) >= INFORMED_RANK_FLOOR
    disagreement = np.where(informed, np.abs(rank - apct), 0.0)
    gate = float(np.clip(supervised_gate, LOGIT_CLIP, 1.0 - LOGIT_CLIP))
    delta = _logit(p) - _logit(gate)
    uncertainty = np.exp(-0.5 * delta * delta)
    return disagreement, uncertainty


def combine_ambiguity(
    disagreement: Any, uncertainty: Any, repeatability_penalty: float = 0.0
) -> Any:
    """Apply the fixed ambiguity weights to component signals."""
    return (
        AMBIGUITY_WEIGHTS["disagreement"] * np.asarray(disagreement, dtype=np.float64)
        + AMBIGUITY_WEIGHTS["supervised_uncertainty"]
        * np.asarray(uncertainty, dtype=np.float64)
        + AMBIGUITY_WEIGHTS["repeatability_penalty"] * float(repeatability_penalty)
    )


def _logit(value: Any) -> Any:
    clipped = np.clip(value, LOGIT_CLIP, 1.0 - LOGIT_CLIP)
    return np.log(clipped / (1.0 - clipped))


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
    supervised_rank = supervised.get("fraud_probability_rank")
    if supervised_rank is not None:
        supervised_rank = _unit_interval(supervised_rank)

    repeatability_penalty = _repeatability_penalty(diagnostics)
    fusion_score = (
        FUSION_WEIGHTS["supervised_fraud_probability"] * fraud_probability
        + FUSION_WEIGHTS["unsupervised_unusualness_percentile"] * unusualness_percentile
    )
    disagreement, supervised_uncertainty, ambiguity_score = ambiguity_components(
        fraud_probability,
        unusualness_percentile,
        supervised_rank,
        resolved_thresholds["fraud_likely_requires_supervised_probability"],
        repeatability_penalty,
    )

    fraud_likely_threshold = resolved_thresholds["fraud_likely_fusion_threshold"]
    supervised_gate = resolved_thresholds["fraud_likely_requires_supervised_probability"]
    review_threshold = resolved_thresholds["ambiguous_review_fusion_threshold"]
    exceeds_realistic_bound = bool(
        supervised.get("exceeds_realistic_amount_bound", False)
    )

    if exceeds_realistic_bound:
        resolution = "FRAUD_LIKELY"
        resolution_text = (
            "Amount exceeds the realistic UPI ceiling of "
            f"{UPI_ABSOLUTE_MAX_AMOUNT:,.0f}; deterministic data-validation "
            "bound breached"
        )
    elif ambiguity_score >= resolved_thresholds["ambiguous_score_threshold"] or (
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
        "exceeds_realistic_amount_bound": exceeds_realistic_bound,
        "signal_disagreement": round(float(disagreement), 6),
        "supervised_uncertainty": round(float(supervised_uncertainty), 6),
        "repeatability_penalty": round(float(repeatability_penalty), 6),
        "weights": FUSION_WEIGHTS.copy(),
        "thresholds": resolved_thresholds,
        "method": (
            "Weighted dual-signal score with rank-scale disagreement and "
            "gate-centred uncertainty"
            if supervised_rank is not None
            else "Weighted dual-signal score with disagreement and uncertainty "
            "review band (legacy probability scale)"
        ),
    }


def tune_fusion_thresholds(
    supervised_probabilities: Any,
    anomaly_percentiles: Any,
    y_true: Any,
    supervised_ranks: Any | None = None,
    gate_floor: float = 0.0,
    min_fraud_precision: float = 0.40,
    review_target_share: float = 0.10,
    review_min_share: float = 0.05,
    review_max_share: float = 0.15,
    review_min_precision: float = 0.02,
    gate_quantiles: int = 24,
) -> dict[str, Any]:
    """Sweep operating points on out-of-time data for both resolution bands.

    Simulates the exact ``fuse_signals`` resolution path (including the
    ambiguity-first gate, minus the single-transaction repeatability penalty
    that has no meaning in batch scoring).

    FRAUD_LIKELY: for a grid of calibrated supervised-probability gates - none
    below ``gate_floor`` so the label always requires supervised corroboration -
    the fused-score boundary is swept for the maximum-recall point whose
    precision meets ``min_fraud_precision``. When rank data is supplied, the
    ambiguity uncertainty kernel is centred on each candidate gate during the
    sweep and then re-centred on the chosen gate so the tuned mask matches
    serving semantics exactly.

    AMBIGUOUS_REVIEW: the review threshold is chosen from out-of-time score
    quantiles so the resulting band captures a share of traffic close to
    ``review_target_share`` while staying inside [``review_min_share``,
    ``review_max_share``] and above ``review_min_precision``. Constraints relax
    deterministically (precision floor first, then the share window) when the
    target is unattainable, and the metadata reports which constraints held.
    """
    p = np.clip(np.asarray(supervised_probabilities, dtype=np.float64).ravel(), 0.0, 1.0)
    unusualness = np.clip(np.asarray(anomaly_percentiles, dtype=np.float64).ravel(), 0.0, 1.0)
    y = np.asarray(y_true, dtype=np.int64).ravel()
    if not (len(p) == len(unusualness) == len(y)) or len(y) == 0:
        raise ValueError("Threshold tuning inputs must be non-empty and equally sized.")
    unique_labels = np.unique(y)
    if len(unique_labels) < 2:
        raise ValueError("Threshold tuning needs evaluation rows of both classes.")
    if not (0.0 < review_min_share <= review_target_share <= review_max_share <= 1.0):
        raise ValueError("Review share bounds must satisfy 0 < min <= target <= max <= 1.")

    ranks: np.ndarray | None = None
    if supervised_ranks is not None:
        ranks = np.clip(
            np.asarray(supervised_ranks, dtype=np.float64).ravel(), 0.0, 1.0
        )
        if len(ranks) != len(p):
            raise ValueError("Supervised ranks must align with probabilities.")

    prevalence = float(y.mean())
    total_positives = max(int(y.sum()), 1)
    fusion_scores = (
        FUSION_WEIGHTS["supervised_fraud_probability"] * p
        + FUSION_WEIGHTS["unsupervised_unusualness_percentile"] * unusualness
    )
    score_threshold = DEFAULT_FUSION_THRESHOLDS["ambiguous_score_threshold"]
    disagreement_threshold = DEFAULT_FUSION_THRESHOLDS["ambiguous_disagreement_threshold"]

    def _fraud_entry(gate: float, ambiguous_first: np.ndarray) -> dict[str, Any]:
        gate_mask = (p >= gate) & ~ambiguous_first
        flagged_cap = int(gate_mask.sum())
        if flagged_cap < 10:
            return {}
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
                "gate": round(float(gate), 8),
                "fusion_threshold": round(float(np.clip(boundary, 0.0, 1.0)), 6),
                "precision": round(float(precision[index]), 6),
                "recall": round(float(recall[index]), 6),
                "f1": round(float(f1[index]), 6),
                "alerts_per_10k_rows": round((index + 1) / len(y) * 10_000, 2),
                "target_precision_met": met,
            }

        feasible = np.flatnonzero(precision >= min_fraud_precision)
        if feasible.size:
            return _entry(int(feasible[-1]), True)
        return _entry(int(np.argmax(f1)), False)

    # Pass one: pick the gate. Ambiguity uses a provisional kernel centre so a
    # gate exists at all; with legacy inputs (no ranks) the formulas do not
    # depend on the gate and this equals the historical behaviour.
    provisional_gate = (
        float(gate_floor)
        if gate_floor > 0.0
        else float(max(np.quantile(p, 0.995), LOGIT_CLIP))
    )
    disagreement_p, uncertainty_p = ambiguity_arrays(
        p, unusualness, ranks, provisional_gate
    )
    ambiguity_p = combine_ambiguity(disagreement_p, uncertainty_p)
    ambiguous_p = (ambiguity_p >= score_threshold) | (
        disagreement_p >= disagreement_threshold
    )

    gates = np.unique(np.quantile(p, np.linspace(0.50, 0.999, gate_quantiles)))
    gates = gates[gates > 0.0]
    if gate_floor > 0.0:
        gates = gates[gates >= gate_floor]
        gates = np.unique(np.concatenate([gates, np.asarray([float(gate_floor)])]))
    if gates.size == 0:
        gates = np.asarray([max(2.0 * prevalence, 1e-4)], dtype=np.float64)

    best: dict[str, Any] | None = None
    fallback: dict[str, Any] | None = None
    for gate in gates:
        candidate = _fraud_entry(float(gate), ambiguous_p)
        if not candidate:
            continue
        if candidate["target_precision_met"]:
            if best is None or (
                candidate["recall"],
                -candidate["alerts_per_10k_rows"],
            ) > (best["recall"], -best["alerts_per_10k_rows"]):
                best = candidate
        elif fallback is None or (candidate["f1"], candidate["precision"]) > (
            fallback["f1"],
            fallback["precision"],
        ):
            fallback = candidate

    chosen = best or fallback
    if chosen is None:
        raise ValueError("No viable threshold found; check the evaluation inputs.")

    # Pass two: freeze the ambiguity kernel on the chosen gate so the tuned
    # mask matches fuse_signals exactly, then finalise the fraud band there.
    disagreement_f, uncertainty_f = ambiguity_arrays(
        p, unusualness, ranks, chosen["gate"]
    )
    ambiguity_f = combine_ambiguity(disagreement_f, uncertainty_f)
    ambiguous_first = (ambiguity_f >= score_threshold) | (
        disagreement_f >= disagreement_threshold
    )
    finalised = _fraud_entry(chosen["gate"], ambiguous_first)
    if finalised:
        chosen = finalised

    fraud_threshold = chosen["fusion_threshold"]
    chosen_gate = chosen["gate"]
    fraud_band = (
        ~ambiguous_first & (fusion_scores >= fraud_threshold) & (p >= chosen_gate)
    )

    review_selection = _select_review_threshold(
        fusion_scores=fusion_scores,
        labels=y,
        ambiguous_first=ambiguous_first,
        fraud_band=fraud_band,
        total_rows=len(y),
        target_share=review_target_share,
        min_share=review_min_share,
        max_share=review_max_share,
        min_precision=review_min_precision,
    )
    review_threshold = review_selection["threshold"]

    return {
        "thresholds": {
            "fraud_likely_fusion_threshold": fraud_threshold,
            "fraud_likely_requires_supervised_probability": round(
                float(chosen_gate), 8
            ),
            "ambiguous_review_fusion_threshold": review_threshold,
            "ambiguous_score_threshold": score_threshold,
            "ambiguous_disagreement_threshold": disagreement_threshold,
        },
        "achieved": {
            "fraud_likely": {
                key: value
                for key, value in chosen.items()
                if key in {"precision", "recall", "f1", "alerts_per_10k_rows"}
            },
            "review_band": review_selection["achieved"],
        },
        "evaluation_rows": int(len(y)),
        "fraud_prevalence": round(prevalence, 8),
        "min_fraud_precision_target": round(float(min_fraud_precision), 4),
        "target_precision_met": bool(chosen["target_precision_met"]),
        "gate_floor_applied": round(float(gate_floor), 8),
        "ambiguity_mode": (
            "rank_scale_informed_v2" if ranks is not None else "legacy_probability_scale"
        ),
        "review_band_config": {
            "target_share": round(float(review_target_share), 4),
            "min_share": round(float(review_min_share), 4),
            "max_share": round(float(review_max_share), 4),
            "min_precision": round(float(review_min_precision), 4),
        },
        "review_constraints_met": {
            "precision_floor": bool(review_selection["precision_met"]),
            "share_window": bool(review_selection["share_in_window"]),
        },
    }


def _select_review_threshold(
    fusion_scores: np.ndarray,
    labels: np.ndarray,
    ambiguous_first: np.ndarray,
    fraud_band: np.ndarray,
    total_rows: int,
    target_share: float,
    min_share: float,
    max_share: float,
    min_precision: float,
) -> dict[str, Any]:
    """Choose the review-band boundary closest to the target traffic share.

    The served band is ``(ambiguous_first | fusion >= T_review)`` minus the
    fraud band, so ambiguous rows count toward the share regardless of their
    score. Candidates are evaluated over every distinct fused-score level of
    the non-fraud population using cumulative counts.
    """
    eligible = ~fraud_band
    scores = fusion_scores[eligible]
    labs = labels[eligible]
    ambiguous = ambiguous_first[eligible]

    order = np.argsort(-scores, kind="mergesort")
    s_sorted = scores[order]
    lab_sorted = labs[order]
    amb_sorted = ambiguous[order]

    amb_total = int(amb_sorted.sum())
    amb_fraud_total = int((amb_sorted & (lab_sorted == 1)).sum())
    cum_tp = np.cumsum(lab_sorted)
    cum_amb = np.cumsum(amb_sorted)
    cum_amb_tp = np.cumsum(amb_sorted & (lab_sorted == 1))

    # Only cut at the end of a tied-score block: any threshold strictly inside
    # a tie group is unrepresentable (serving admits whole groups), and cutting
    # mid-block would understate the served band size.
    block_ends = np.flatnonzero(s_sorted[:-1] != s_sorted[1:])
    positions = np.concatenate([block_ends, [len(s_sorted) - 1]])
    review_rows = amb_total + (positions + 1) - cum_amb[positions]
    review_fraud = amb_fraud_total + cum_tp[positions] - cum_amb_tp[positions]
    share = review_rows / float(total_rows)
    precision = review_fraud / np.maximum(review_rows, 1)

    precision_ok = precision >= min_precision
    share_ok = (share >= min_share) & (share <= max_share)
    distance = np.abs(share - target_share)

    def _pick(mask: np.ndarray) -> int | None:
        if not mask.any():
            return None
        candidates = np.flatnonzero(mask)
        best_distance = distance[candidates].min()
        tied = candidates[distance[candidates] <= best_distance + 1e-12]
        return int(tied[np.argmax(review_fraud[tied])])

    index = _pick(precision_ok & share_ok)
    precision_met = True
    share_in_window = True
    if index is None:
        precision_met = False
        index = _pick(share_ok)
    if index is None:
        share_in_window = False
        precision_met = True
        index = _pick(precision_ok)
    if index is None:
        precision_met = False
        share_in_window = False
        index = _pick(np.ones(len(s_sorted), dtype=bool))
    if index is None:
        raise ValueError("No review-band candidate available.")

    if index + 1 < len(s_sorted):
        boundary = float((s_sorted[index] + s_sorted[index + 1]) / 2.0)
    else:
        boundary = float(s_sorted[index]) / 2.0

    remaining_frauds = max(int(labels.sum() - labels[fraud_band].sum()), 1)
    return {
        "threshold": round(float(np.clip(boundary, 0.0, 1.0)), 6),
        "precision_met": precision_met,
        "share_in_window": share_in_window,
        "achieved": {
            "rows": int(review_rows[index]),
            "share": round(float(share[index]), 6),
            "fraud_rows": int(review_fraud[index]),
            "precision": round(float(precision[index]), 6),
            "recall_of_remaining": round(
                float(review_fraud[index] / remaining_frauds), 6
            ),
        },
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
    bound_breached = bool(supervised.get("exceeds_realistic_amount_bound", False))

    if bound_breached:
        input_factors.append(
            f"Amount {amount:,.2f} exceeds the realistic UPI ceiling of "
            f"{UPI_ABSOLUTE_MAX_AMOUNT:,.0f}"
        )
    if int(supervised.get("live_history_count", 0) or 0) == 0:
        input_factors.append("Sender has no previously scored transactions in the live system")
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
    elif resolution == "FRAUD_LIKELY" and bound_breached:
        resolution_reason = (
            f"The transaction is fraud-likely because its amount of {amount:,.2f} breaches the "
            f"deterministic data-validation ceiling of {UPI_ABSOLUTE_MAX_AMOUNT:,.2f} for realistic "
            "UPI payments; this bound applies independently of the model scores. The statistical "
            f"signals are reported separately: a supervised fraud probability of "
            f"{fraud_probability:.1%} and an anomaly unusualness percentile of "
            f"{anomaly_percentile:.1%} ({anomaly_label})."
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
            "exceeds_realistic_amount_bound": bound_breached,
            "live_history_count": int(supervised.get("live_history_count", 0) or 0),
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
