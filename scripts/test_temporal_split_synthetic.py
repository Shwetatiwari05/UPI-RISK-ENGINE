"""Synthetic ground-truth tests for the temporal-split point-in-time pipeline.

Run before touching the real datasets:

    ./venv/bin/python3 scripts/test_temporal_split_synthetic.py

Every assertion is checked against hand-computed expectations on tiny
synthetic frames, so a failure pinpoints the exact broken invariant:
causality of behavioral features, train-only constant fitting, period-plan
math, period-filtered sampling, or end-to-end streamed preprocessing.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import src.parquet_pipeline as pp
from src.feature_engineering import (
    add_temporal_features,
    apply_fitted_features,
    apply_point_in_time_features,
    fit_feature_constants,
)
from src.schema_mapping import COMMON_SCHEMA


def make_base_frame() -> pd.DataFrame:
    """Five transactions from one sender with hand-computable history."""
    return pd.DataFrame(
        {
            "transaction_id": [f"s__t{i}" for i in range(5)],
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01 10:00",
                    "2024-01-01 10:02",
                    "2024-01-01 11:00",
                    "2024-01-02 09:00",
                    "2024-01-02 09:30",
                ]
            ),
            "amount": [100.0, 200.0, 10000.0, 150.0, 150.0],
            "sender_id": ["S1"] * 5,
            "receiver_id": ["R1", "R1", "R2", "R3", "R3"],
            "device_type": ["Android", "Android", "iOS", "Android", "Web"],
            "merchant_category": ["Food", "Food", "Travel", "Food", "Food"],
            "location": ["Mumbai", "Mumbai", "Delhi", "Mumbai", "Mumbai"],
            "transaction_type": ["P2P"] * 5,
            "fraud_label": [0] * 5,
        }
    )


def raw_point_in_time(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    working["timestamp"] = pd.to_datetime(working["timestamp"])
    working["amount"] = working["amount"].astype(float)
    for column in ["sender_id", "receiver_id", "merchant_category", "device_type", "location"]:
        working[column] = working[column].astype(str)
    working = add_temporal_features(working)
    return apply_point_in_time_features(working)


def test_causal_point_in_time_features() -> None:
    out = raw_point_in_time(make_base_frame())
    expected = {
        # row: (freq, avg_prior_mean, spike, tph, minutes_prev, rapid, mdiv, dsw, new_payee)
        0: (0, np.nan, 0, 0, np.nan, 0, 0, 0, 1),
        1: (1, 100.0, 0, 1, 2.0, 1, 1, 1, 0),
        2: (2, 150.0, 1, 0, 58.0, 0, 1, 1, 1),
        3: (3, 10300.0 / 3.0, 0, 0, 1320.0, 0, 2, 2, 1),
        4: (4, 2612.5, 0, 1, 30.0, 0, 2, 2, 0),
    }
    for row, values in expected.items():
        freq, avg, spike, tph, minutes, rapid, mdiv, dsw, new_payee = values
        assert out.loc[row, "transaction_frequency"] == freq, f"row {row} frequency"
        if np.isnan(avg):
            assert np.isnan(out.loc[row, "avg_transaction_amount"]), f"row {row} avg should be NaN"
        else:
            assert abs(out.loc[row, "avg_transaction_amount"] - avg) < 0.51, f"row {row} avg"
        assert out.loc[row, "amount_spike"] == spike, f"row {row} spike"
        assert out.loc[row, "transactions_per_hour"] == tph, f"row {row} tph"
        if np.isnan(minutes):
            assert np.isnan(out.loc[row, "minutes_since_previous_sender_txn"]), f"row {row} minutes"
        else:
            assert abs(out.loc[row, "minutes_since_previous_sender_txn"] - minutes) < 0.01, f"row {row} minutes"
        assert out.loc[row, "rapid_transactions"] == rapid, f"row {row} rapid"
        assert out.loc[row, "merchant_diversity"] == mdiv, f"row {row} merchant diversity"
        assert out.loc[row, "device_switching_frequency"] == dsw, f"row {row} device switching"
        assert out.loc[row, "new_payee_flag"] == new_payee, f"row {row} new payee"

    # amount_spike at row 2 must fire: 10000 > prior_mean(150) + 3 * prior_std(50) = 300
    constants = {"usual_location_map": {}, "high_frequency_threshold": 2.0, "global_amount_mean": 0.0}
    filled = apply_fitted_features(out, constants)
    assert int(filled["amount_spike"].sum()) == 1
    print("PASS causal point-in-time features (freq/avg/spike/velocity/diversity/payee)")


def test_train_only_constants_no_leakage() -> None:
    """Constants fitted on the training rows only; evaluation rows never vote."""
    base = make_base_frame()
    leak_probe = pd.concat([base] * 2, ignore_index=True).head(8)
    # 4 training rows in Mumbai, then 4 later evaluation rows relocated to Delhi.
    # If constants leaked (fit on all rows), the mode location would flip to Delhi.
    leak_probe["location"] = ["Mumbai"] * 4 + ["Delhi"] * 4
    leak_probe["timestamp"] = pd.date_range("2024-01-01", periods=8, freq="6h")
    leak_probe["sender_id"] = ["S1"] * 8

    engineered = raw_point_in_time(leak_probe)
    train_mask = engineered["timestamp"] <= pd.Timestamp("2024-01-01 21:00")
    assert int(train_mask.sum()) == 4 and int((~train_mask).sum()) == 4

    constants = fit_feature_constants(engineered.loc[train_mask])
    assert constants["usual_location_map"] == {"S1": "Mumbai"}, constants["usual_location_map"]
    result = apply_fitted_features(engineered, {**constants, "global_amount_mean": 500.0})
    test_flags = result.loc[~train_mask, "unusual_location_flag"]
    assert (test_flags == 1).all(), "Delhi test rows must be flagged unusual vs Mumbai-trained map"

    # Cold-start fallback uses the supplied training-period global mean.
    cold_rows = result.loc[engineered["transaction_frequency"] == 0]
    assert len(cold_rows) > 0
    assert (cold_rows["avg_transaction_amount"] == 500.0).all()
    print("PASS train-only constants (usual-location map, cold-start fill) without leakage")


def build_synthetic_mapped_parquet(path: Path, rows_per_source: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    parts = []
    start = pd.Timestamp("2024-01-01")
    for source_index, source in enumerate(["alpha", "beta"]):
        hours = np.arange(rows_per_source) + source_index * 3
        senders = rng.integers(0, 40, size=rows_per_source)
        frame = pd.DataFrame(
            {
                "transaction_id": [f"{source}__txn_{i:06d}" for i in range(rows_per_source)],
                "timestamp": start + pd.to_timedelta(hours, unit="h"),
                "amount": np.round(rng.lognormal(mean=6.0, sigma=1.0, size=rows_per_source), 2),
                "sender_id": [f"{source}_S{s}" for s in senders],
                "receiver_id": [f"{source}_R{r}" for r in rng.integers(0, 60, size=rows_per_source)],
                "device_type": rng.choice(["Android", "iOS", "Web"], size=rows_per_source),
                "merchant_category": rng.choice(["Food", "Travel", "Retail"], size=rows_per_source),
                "location": rng.choice(["Mumbai", "Delhi", "Bengaluru"], size=rows_per_source),
                "transaction_type": rng.choice(["P2P", "P2M"], size=rows_per_source),
                "fraud_label": rng.choice([0, 1], size=rows_per_source, p=[0.95, 0.05]),
            }
        )
        parts.append(frame)
    full = pd.concat(parts, ignore_index=True)
    table = pa.Table.from_pandas(full[COMMON_SCHEMA], preserve_index=False)
    pq.write_table(table, path, compression="snappy")
    return full


def test_period_plan_and_end_to_end(tmp: Path) -> None:
    mapped_path = tmp / "mapped_synthetic.parquet"
    truth = build_synthetic_mapped_parquet(mapped_path)

    plan = pp._scan_period_plan(mapped_path, chunk_size=137, test_fraction=0.2)
    assert set(plan["ordered_sources"]) == {"alpha", "beta"}
    total_train = sum(plan["train_row_counts"].values())
    total_test = sum(plan["source_row_counts"].values()) - total_train
    share_test = total_test / (total_train + total_test)
    assert 0.15 <= share_test <= 0.25, f"test share {share_test:.3f} not near 0.20"
    assert plan["source_row_counts"] == {"alpha": 400, "beta": 400}

    cutoffs = plan["cutoffs"]
    for source in ["alpha", "beta"]:
        subset = truth.loc[truth["transaction_id"].str.startswith(source + "__")]
        timestamps = pd.to_datetime(subset["timestamp"])
        expected_cutoff = timestamps.quantile(0.8)
        actual_cutoff = pd.Timestamp(cutoffs[source])
        assert abs((actual_cutoff - expected_cutoff).total_seconds()) < 1.0, (
            f"{source} cutoff drift: {actual_cutoff} vs {expected_cutoff}"
        )

    # Cold-start fallback amount must be the pooled training-period mean.
    source_codes = truth["transaction_id"].str.split("__", n=1).str[0]
    cutoff_series = source_codes.map({key: pd.Timestamp(value) for key, value in cutoffs.items()})
    pooled_train_mask = pd.to_datetime(truth["timestamp"]) <= cutoff_series
    pooled_train_mean = float(truth.loc[pooled_train_mask, "amount"].mean())
    assert abs(plan["global_amount_mean"] - pooled_train_mean) < 1e-9, (
        "global_amount_mean must come from training-period rows only"
    )

    original_dir = pp.PROCESSED_DATA_DIR
    try:
        pp.PROCESSED_DATA_DIR = tmp / "processed"
        pp.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        out_a = tmp / "processed_run_a.parquet"
        out_b = tmp / "processed_run_b.parquet"
        preprocessor, stats = pp.stream_preprocess_to_parquet(
            mapped_parquet_path=mapped_path,
            output_path=out_a,
            fit_rows=300,
            chunk_size=137,
            test_fraction=0.2,
        )
        assert stats["rows_written"] == len(truth), f"row parity broken: {stats['rows_written']}"

        processed = pd.read_parquet(out_a)
        assert "period" in processed.columns
        assert set(processed["period"].unique()) == {"train", "test"}

        merged = processed.merge(
            truth[["transaction_id", "timestamp", "amount"]],
            on="transaction_id",
            validate="one_to_one",
        )
        merged["orig_timestamp"] = pd.to_datetime(merged["timestamp"])
        for source in ["alpha", "beta"]:
            block = merged.loc[merged["transaction_id"].str.startswith(source + "__")]
            cutoff = pd.Timestamp(cutoffs[source])
            train_ts = block.loc[block["period"] == "train", "orig_timestamp"]
            test_ts = block.loc[block["period"] == "test", "orig_timestamp"]
            assert train_ts.max() <= cutoff, f"{source}: train row after cutoff"
            assert test_ts.min() > cutoff, f"{source}: test row at/before cutoff"
        label_match = (
            merged.sort_values("transaction_id")["fraud_label"].reset_index(drop=True)
            == truth.sort_values("transaction_id")["fraud_label"].reset_index(drop=True)
        )
        assert bool(label_match.all()), "fraud_label corrupted during preprocessing"

        forbidden = {"period", "transaction_id", "timestamp", "fraud_label", "source_dataset"}
        leaked = forbidden.intersection(preprocessor.feature_columns_)
        assert not leaked, f"identity/label columns leaked into features: {leaked}"

        pp.stream_preprocess_to_parquet(
            mapped_parquet_path=mapped_path,
            output_path=out_b,
            fit_rows=300,
            chunk_size=137,
            test_fraction=0.2,
        )
        run_a = pd.read_parquet(out_a).sort_values("transaction_id").reset_index(drop=True)
        run_b = pd.read_parquet(out_b).sort_values("transaction_id").reset_index(drop=True)
        key_columns = [
            column
            for column in run_a.columns
            if column not in {"transaction_id"}
        ]
        pd.testing.assert_frame_equal(run_a[key_columns], run_b[key_columns])
        print(
            "PASS end-to-end streamed preprocessing "
            f"(parity={stats['rows_written']}, train={int((run_a['period'] == 'train').sum())}, "
            f"test={int((run_a['period'] == 'test').sum())}, deterministic, labels intact)"
        )

        sample = pp.sample_parquet_rows(
            out_a,
            max_rows=120,
            chunk_size=97,
            strategy="uniform",
            period_value="train",
        )
        assert len(sample) == 120
        assert set(sample["period"].unique()) == {"train"}, "period filter failed"

        train_frame = pp.load_period_frame(out_a, "train", chunk_size=200)
        test_frame = pp.load_period_frame(out_a, "test", chunk_size=200)
        results = __import__("src.supervised_model", fromlist=["train_supervised_models"]).train_supervised_models(
            train_frame,
            model_dir=tmp / "models",
            report_dir=tmp / "reports",
            max_rows=len(train_frame),
            preprocessor=preprocessor,
            preprocessed=True,
            evaluation_frame=test_frame,
        )
        for name, metrics in results.items():
            for key in ["roc_auc", "pr_auc", "precision", "recall", "f1_score"]:
                assert key in metrics, f"{name} missing metric {key}"
            assert np.isfinite(metrics["pr_auc"]), f"{name} pr_auc not finite"
            print(
                f"PASS out-of-time supervised plumbing ({name}): "
                f"ROC-AUC={metrics['roc_auc']:.3f} PR-AUC={metrics['pr_auc']:.3f}"
            )

        broken = test_frame.drop(columns=[preprocessor.feature_columns_[0]])
        try:
            __import__("src.supervised_model", fromlist=["train_supervised_models"]).train_supervised_models(
                train_frame,
                max_rows=len(train_frame),
                preprocessor=preprocessor,
                preprocessed=True,
                evaluation_frame=broken,
            )
        except ValueError as error:
            assert "missing engineered columns" in str(error)
        else:
            raise AssertionError("missing-column guard did not raise")
        print("PASS evaluation-frame column guard raises ValueError")
    finally:
        pp.PROCESSED_DATA_DIR = original_dir


def test_source_blocks_complete_under_small_chunks(tmp: Path) -> None:
    mapped_path = tmp / "mapped_blocks.parquet"
    build_synthetic_mapped_parquet(mapped_path, rows_per_source=60)
    blocks = list(pp._iter_source_blocks(mapped_path, chunk_size=17, columns=pp.COMMON_SCHEMA))
    sources = [source for source, _ in blocks]
    assert sources == ["alpha", "beta"], f"unexpected block order: {sources}"
    for _, block in blocks:
        prefixes = set(pp._source_prefixes(block["transaction_id"]))
        assert len(prefixes) == 1, f"block mixed sources: {prefixes}"
        assert len(block) == 60, f"incomplete block: {len(block)}"
    print("PASS source blocks complete when chunk boundaries fall mid-source")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="pit_synthetic_") as temp_dir:
        tmp = Path(temp_dir)
        test_causal_point_in_time_features()
        test_train_only_constants_no_leakage()
        test_source_blocks_complete_under_small_chunks(tmp)
        test_period_plan_and_end_to_end(tmp)
    print("\nALL SYNTHETIC GROUND-TRUTH TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
