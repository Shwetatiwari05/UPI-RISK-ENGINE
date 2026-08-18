"""Sampling helpers for large offline modeling datasets."""

from __future__ import annotations

import pandas as pd


def balanced_binary_sample(
    df: pd.DataFrame,
    target_column: str,
    max_rows: int,
    random_state: int = 42,
    legitimate_ratio: float = 3.0,
) -> pd.DataFrame:
    """Keep all fraud rows and sample legitimate rows at a controlled ratio."""
    if len(df) <= max_rows or target_column not in df.columns:
        return df.copy()
    if legitimate_ratio <= 0:
        raise ValueError("legitimate_ratio must be greater than zero.")

    label_counts = df[target_column].value_counts(dropna=False)
    if set(label_counts.index) != {0, 1}:
        return df.sample(n=max_rows, random_state=random_state, replace=False).reset_index(drop=True)

    fraud = df[df[target_column] == 1]
    legitimate = df[df[target_column] == 0]
    fraud_target = min(len(fraud), max_rows)
    legitimate_target = min(
        len(legitimate),
        max(0, max_rows - fraud_target),
        int(round(fraud_target * legitimate_ratio)),
    )
    fraud_sample = fraud if len(fraud) <= fraud_target else fraud.sample(
        n=fraud_target,
        random_state=random_state,
        replace=False,
    )
    legitimate_sample = stratified_majority_sample(
        legitimate,
        max_rows=legitimate_target,
        random_state=random_state,
    )

    sampled = pd.concat([fraud_sample, legitimate_sample], ignore_index=True)
    sampled = sampled.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return sampled


def stratified_majority_sample(
    df: pd.DataFrame,
    max_rows: int,
    random_state: int = 42,
) -> pd.DataFrame:
    """Sample a majority class proportionally across useful behavior strata."""
    if max_rows <= 0 or df.empty:
        return df.iloc[0:0].copy()
    if len(df) <= max_rows:
        return df.copy()

    strata_columns = [
        column
        for column in ["hour_of_day", "day_of_week", "transaction_type", "device_type"]
        if column in df.columns
    ]
    if not strata_columns:
        return df.sample(n=max_rows, random_state=random_state, replace=False)

    grouped = df.groupby(strata_columns, dropna=False, sort=False, observed=True)
    counts = grouped.size()
    raw_targets = counts / counts.sum() * max_rows
    targets = raw_targets.astype(int)
    remainder = max_rows - int(targets.sum())
    for key in raw_targets.sub(targets).sort_values(ascending=False).index[:remainder]:
        targets.loc[key] += 1

    selected = []
    for index, subset in grouped:
        target = min(int(targets.loc[index]), len(subset))
        if target:
            selected.append(subset.sample(n=target, random_state=random_state + len(selected), replace=False))
    if not selected:
        return df.sample(n=max_rows, random_state=random_state, replace=False)
    return pd.concat(selected, ignore_index=True).sample(
        frac=1.0,
        random_state=random_state,
    ).reset_index(drop=True)


def stratified_sample(
    df: pd.DataFrame,
    target_column: str,
    max_rows: int,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return a bounded sample that roughly preserves class proportions."""
    if len(df) <= max_rows or target_column not in df.columns:
        return df.copy()

    grouped = []
    total_rows = len(df)
    for label, subset in df.groupby(target_column, dropna=False, sort=False):
        share = len(subset) / total_rows
        n_rows = max(1, int(round(max_rows * share)))
        n_rows = min(n_rows, len(subset))
        grouped.append(subset.sample(n=n_rows, random_state=random_state, replace=False))

    sampled = pd.concat(grouped, ignore_index=True)
    if len(sampled) > max_rows:
        sampled = sampled.sample(n=max_rows, random_state=random_state, replace=False)
    return sampled.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
