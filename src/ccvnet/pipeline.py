from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ccvnet.cvd import (
    build_normalized_voltage_cvd_bank,
    load_cvd_bank,
    load_cvd_descriptor_table,
)
from ccvnet.data import (
    DatasetConfig,
    default_dataset_configs,
    infer_value_feature_columns,
)
from ccvnet.splits import build_within_group_holdout_splits
from ccvnet.training import append_cvd_availability_mask


FINE_GROUP_COLUMN = "fine_condition_group"
FINE_META_GROUP_COLUMN = "fine_metadata_group"
COARSE_GROUP_COLUMN = "dataset_group"


@dataclass
class AlignedData:
    pipeline_df: pd.DataFrame
    X_cvd_abs: np.ndarray
    X_cvd_norm: np.ndarray
    X_value_abs: np.ndarray
    X_value_norm: np.ndarray
    y: np.ndarray
    descriptor_abs_cols: list[str]
    descriptor_norm_cols: list[str]
    metadata_cols: list[str]
    descriptor_raw_dim: int
    metadata_raw_dim: int
    fine_group_col: str = FINE_GROUP_COLUMN


def normalized_descriptor_columns(columns: list[str]) -> list[str]:
    normalized_candidates = []
    for col in columns:
        text = str(col).lower()
        if text.startswith("norm_") or text.endswith("_norm") or "normalized" in text:
            normalized_candidates.append(col)
    return normalized_candidates


def add_recovered_group_columns(df: pd.DataFrame, dataset_col: str = "dataset_name") -> pd.DataFrame:
    df = df.copy()
    if COARSE_GROUP_COLUMN not in df.columns:
        df[COARSE_GROUP_COLUMN] = (
            df.get(dataset_col, pd.Series(index=df.index, dtype=object))
            .fillna("unknown_group")
            .astype(str)
        )
    if FINE_GROUP_COLUMN not in df.columns:
        coarse = df[COARSE_GROUP_COLUMN].astype(str)
        chem = df.get("battery_type", pd.Series(index=df.index, dtype=object)).fillna("unknown").astype(str)
        voltage = df.get("voltage_window", pd.Series(index=df.index, dtype=object)).fillna("unknown").astype(str)
        temp = (
            df.get("operation_temperature_C", pd.Series(index=df.index, dtype=object))
            .fillna("unknown")
            .astype(str)
        )
        df[FINE_GROUP_COLUMN] = coarse + " | " + chem + " | " + voltage + " | " + temp
    if FINE_META_GROUP_COLUMN not in df.columns:
        chem = df.get("battery_type", pd.Series(index=df.index, dtype=object)).fillna("unknown").astype(str)
        voltage = df.get("voltage_window", pd.Series(index=df.index, dtype=object)).fillna("unknown").astype(str)
        temp = (
            df.get("operation_temperature_C", pd.Series(index=df.index, dtype=object))
            .fillna("unknown")
            .astype(str)
        )
        df[FINE_META_GROUP_COLUMN] = chem + " | " + voltage + " | " + temp
    return df


def build_metadata_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    numeric_cols = [
        col
        for col in [
            "nominal_capacity_Ah",
            "operation_temperature_C",
            "voltage_min_V",
            "voltage_max_V",
            "voltage_width_V",
        ]
        if col in df.columns
    ]
    categorical_cols = [col for col in ["battery_type", "soc_window", "protocol"] if col in df.columns]
    numeric_df = (
        df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        if numeric_cols
        else pd.DataFrame(index=df.index)
    )
    if categorical_cols:
        categorical_df = pd.get_dummies(
            df[categorical_cols].fillna("unknown").astype(str),
            prefix=categorical_cols,
            dummy_na=False,
        )
    else:
        categorical_df = pd.DataFrame(index=df.index)
    metadata_df = pd.concat([numeric_df, categorical_df], axis=1)
    metadata_df = metadata_df.loc[:, ~metadata_df.columns.duplicated()].copy()
    return metadata_df, metadata_df.columns.tolist()


def dataset_configs_from_config(config: dict[str, Any]) -> list[DatasetConfig]:
    data_cfg = config.get("data", {})
    base_dir = Path(config.get("paths", {}).get("processed_dir", "data/processed"))
    configured = data_cfg.get("datasets")
    if not configured:
        return default_dataset_configs(base_dir)
    configs = []
    for item in configured:
        configs.append(
            DatasetConfig(
                dataset_name=str(item["dataset_name"]),
                folder=str(item["folder"]),
                feature_csv=item.get("feature_csv"),
                base_dir=base_dir,
                battery_type=str(item.get("battery_type", "NMC")),
                nominal_capacity_Ah=float(item.get("nominal_capacity_Ah", np.nan)),
                voltage_window=item.get("voltage_window"),
                default_temperature_C=float(item.get("default_temperature_C", np.nan)),
                default_charging_rate_C=float(item.get("default_charging_rate_C", np.nan)),
                default_discharging_rate_C=float(item.get("default_discharging_rate_C", np.nan)),
                cvd_dir_name=str(item.get("cvd_dir_name", "CVD_curve")),
                notes=str(item.get("notes", "")),
            )
        )
    return configs


def build_aligned_data(config: dict[str, Any]) -> AlignedData:
    data_cfg = config.get("data", {})
    target_col = str(data_cfg.get("target_column", "life"))
    min_target_life = float(data_cfg.get("min_target_life", 100.0))
    cycles = tuple(int(cycle) for cycle in data_cfg.get("cvd_cycles", range(20, 101, 10)))
    value_key = str(data_cfg.get("cvd_value_key", "negative_difference"))
    normalized_voltage_points = int(data_cfg.get("normalized_voltage_points", data_cfg.get("voltage_points", 256)))

    configs = dataset_configs_from_config(config)
    feature_cycles = tuple(int(cycle) for cycle in data_cfg.get("descriptor_cycles", (20, 50, 100)))
    descriptor_df = load_cvd_descriptor_table(configs, feature_cycles=feature_cycles)
    if descriptor_df.empty:
        raise ValueError("No CVD descriptor rows were loaded. Check CVD pkl paths and preprocessing output.")

    cvd_df, cvd_array = load_cvd_bank(configs, cycles=cycles, value_key=value_key, include_voltage_axis=True)
    if cvd_df.empty or cvd_array.size == 0:
        raise ValueError("No CVD records were loaded. Check data paths and selected cycles.")

    norm_cvd_df, norm_cvd_array, _, _ = build_normalized_voltage_cvd_bank(
        cvd_df, cvd_array, cycles=cycles, value_key=value_key, n_points=normalized_voltage_points
    )
    norm_index = norm_cvd_df.set_index("cvd_index").index
    if not cvd_df["cvd_index"].isin(norm_index).all():
        raise ValueError("Normalized CVD array is missing records from the absolute CVD table.")

    merge_keys = [col for col in ["dataset_name", "cell"] if col in descriptor_df.columns and col in cvd_df.columns]
    if len(merge_keys) != 2:
        raise KeyError(f"Expected merge keys dataset_name/cell, got {merge_keys}.")

    cvd_keep_cols = [
        col
        for col in [
            "cvd_index",
            "cvd_path",
            "cvd_life",
            "cvd_group",
            "cvd_cycles",
            "cvd_input_channels",
            "cvd_axis_min_V",
            "cvd_axis_max_V",
            "cvd_axis_length",
            *merge_keys,
        ]
        if col in cvd_df.columns
    ]
    model_df = descriptor_df.merge(cvd_df[cvd_keep_cols], on=merge_keys, how="inner")
    if "target_life" not in model_df.columns:
        if target_col in model_df.columns and "cvd_life" in model_df.columns:
            model_df["target_life"] = model_df[target_col].combine_first(model_df["cvd_life"])
        elif target_col in model_df.columns:
            model_df["target_life"] = model_df[target_col]
        elif "cvd_life" in model_df.columns:
            model_df["target_life"] = model_df["cvd_life"]
        else:
            raise KeyError("Missing target_life source columns.")

    model_df["target_life"] = pd.to_numeric(model_df["target_life"], errors="coerce")
    model_df = model_df.loc[model_df["target_life"].gt(min_target_life)].reset_index(drop=True)
    model_df["row_index"] = np.arange(len(model_df))
    model_df = add_recovered_group_columns(model_df)
    if model_df.empty:
        raise ValueError("No aligned rows remain after target-life filtering.")

    cvd_indices = model_df["cvd_index"].astype(int).to_numpy()
    X_cvd_abs = append_cvd_availability_mask(cvd_array[cvd_indices])
    X_cvd_norm = append_cvd_availability_mask(norm_cvd_array[cvd_indices])

    descriptor_abs_cols = infer_value_feature_columns(model_df)
    descriptor_abs_cols = [col for col in descriptor_abs_cols if col in model_df.columns]
    if not descriptor_abs_cols:
        raise ValueError("No absolute descriptor columns were found.")

    descriptor_norm_cols = normalized_descriptor_columns(list(model_df.columns))
    if descriptor_norm_cols:
        descriptor_norm_df = model_df[descriptor_norm_cols].apply(pd.to_numeric, errors="coerce")
    else:
        descriptor_norm_cols = list(descriptor_abs_cols)
        descriptor_norm_df = model_df[descriptor_abs_cols].apply(pd.to_numeric, errors="coerce")

    descriptor_abs_df = model_df[descriptor_abs_cols].apply(pd.to_numeric, errors="coerce")
    metadata_df, metadata_cols = build_metadata_features(model_df)

    value_abs_df = pd.concat([descriptor_abs_df, metadata_df], axis=1)
    value_abs_df = value_abs_df.loc[:, ~value_abs_df.columns.duplicated()].copy()
    value_norm_df = pd.concat([descriptor_norm_df, metadata_df], axis=1)
    value_norm_df = value_norm_df.loc[:, ~value_norm_df.columns.duplicated()].copy()

    return AlignedData(
        pipeline_df=model_df,
        X_cvd_abs=np.asarray(X_cvd_abs, dtype=np.float32),
        X_cvd_norm=np.asarray(X_cvd_norm, dtype=np.float32),
        X_value_abs=np.asarray(value_abs_df, dtype=np.float32),
        X_value_norm=np.asarray(value_norm_df, dtype=np.float32),
        y=model_df["target_life"].astype(float).to_numpy(),
        descriptor_abs_cols=descriptor_abs_cols,
        descriptor_norm_cols=descriptor_norm_cols,
        metadata_cols=metadata_cols,
        descriptor_raw_dim=len(descriptor_abs_cols),
        metadata_raw_dim=len(metadata_cols),
    )


def build_fine_group_shared_split(aligned: AlignedData, config: dict[str, Any]) -> pd.DataFrame:
    split_cfg = config.get("split", {})
    seeds = [int(seed) for seed in split_cfg.get("seeds", [42, 52, 62, 72, 82])]
    test_fraction = float(split_cfg.get("test_fraction", 0.25))
    group_col = str(split_cfg.get("group_column", aligned.fine_group_col))
    if group_col not in aligned.pipeline_df.columns:
        raise KeyError(f"Split group column {group_col!r} is missing from pipeline_df.")
    split_df = build_within_group_holdout_splits(
        aligned.pipeline_df,
        group_col,
        test_fraction,
        seeds,
    )
    split_df["split_protocol"] = "fine-group-consistent backbone"
    return split_df



