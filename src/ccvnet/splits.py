from __future__ import annotations

import numpy as np
import pandas as pd


def make_group_holdout_split(
    df: pd.DataFrame,
    group_col: str,
    test_fraction: float,
    seed: int,
    min_testable_group_size: int = 4,
) -> pd.Series:
    rng = np.random.default_rng(seed)
    split = pd.Series("train", index=df.index, dtype="object")
    for _, sub in df.groupby(group_col):
        idx = sub.index.to_numpy()
        if len(idx) < min_testable_group_size:
            continue
        n_test = max(1, int(round(len(idx) * test_fraction)))
        test_idx = rng.choice(idx, size=n_test, replace=False)
        split.loc[test_idx] = "test"
    return split


def build_repeated_group_holdout_splits(
    df: pd.DataFrame,
    group_col: str,
    test_fraction: float,
    seeds: list[int],
    min_testable_group_size: int = 4,
) -> pd.DataFrame:
    if "row_index" in df.columns:
        row_index = df["row_index"].to_numpy()
    else:
        row_index = np.arange(len(df))

    split_tables = []
    for split_id, seed in enumerate(seeds, start=1):
        split = make_group_holdout_split(
            df, group_col, test_fraction, seed, min_testable_group_size=min_testable_group_size
        )
        split_tables.append(
            pd.DataFrame(
                {
                    "row_index": row_index,
                    group_col: df[group_col].to_numpy(),
                    "split_id": split_id,
                    "seed": seed,
                    "split": split.to_numpy(),
                }
            )
        )
    return pd.concat(split_tables, ignore_index=True)


def build_within_group_holdout_splits(
    df: pd.DataFrame,
    group_col: str,
    test_fraction: float,
    seeds: list[int],
) -> pd.DataFrame:
    split_rows = []
    for split_id, seed in enumerate(seeds, start=1):
        rng = np.random.default_rng(seed)
        for group_value, group_index in df.groupby(group_col, sort=True).indices.items():
            row_index = np.asarray(group_index, dtype=int)
            if len(row_index) <= 1:
                test_index = np.array([], dtype=int)
            else:
                n_test = max(1, int(np.floor(len(row_index) * test_fraction)))
                n_test = min(n_test, len(row_index) - 1)
                test_index = rng.permutation(row_index)[:n_test]
            test_set = set(test_index.tolist())
            for idx in row_index.tolist():
                split_rows.append(
                    {
                        "split_id": split_id,
                        "row_index": idx,
                        group_col: group_value,
                        "split": "test" if idx in test_set else "train",
                    }
                )
    return pd.DataFrame(split_rows).sort_values(["split_id", "row_index"]).reset_index(drop=True)


def split_masks(split_df: pd.DataFrame, split_id: int) -> tuple[np.ndarray, np.ndarray]:
    subset = split_df.loc[split_df["split_id"].eq(split_id)].sort_values("row_index")
    if subset.empty:
        raise ValueError(f"Unknown split_id: {split_id}")
    train_mask = subset["split"].eq("train").to_numpy()
    test_mask = subset["split"].eq("test").to_numpy()
    return train_mask, test_mask

