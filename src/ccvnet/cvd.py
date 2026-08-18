from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ccvnet.data import DatasetConfig, parse_cell_metadata


DEFAULT_CVD_CYCLES = tuple(range(20, 101, 10))
DEFAULT_CVD_VALUE_KEY = "negative_difference"


def cycle_payload(cycles: dict, cycle: int) -> dict | None:
    return cycles.get(cycle) or cycles.get(str(cycle))


def cvd_input_channel_names(
    cycles: tuple[int, ...],
    value_key: str,
    include_voltage_axis: bool = True,
) -> list[str]:
    names = [f"cycle_{cycle}_{value_key}" for cycle in cycles]
    if include_voltage_axis:
        names += [f"cycle_{cycle}_voltage_axis" for cycle in cycles]
    return names


def available_cvd_cycles(path: Path) -> tuple[int, ...]:
    with Path(path).open("rb") as handle:
        record = pickle.load(handle)
    cycle_bank = record.get("cycles", {})
    available = []
    for key in cycle_bank.keys():
        try:
            available.append(int(key))
        except (TypeError, ValueError):
            continue
    return tuple(sorted(available))


def inspect_cvd_cycle_coverage(
    configs: list[DatasetConfig],
    requested_cycles: tuple[int, ...] = DEFAULT_CVD_CYCLES,
) -> pd.DataFrame:
    rows = []
    for cfg in configs:
        paths = sorted(cfg.cvd_dir.glob("*CVD.pkl")) if cfg.cvd_dir.exists() else []
        cycle_sets = [set(available_cvd_cycles(path)) for path in paths]
        common_cycles = sorted(set.intersection(*cycle_sets)) if cycle_sets else []
        rows.append(
            {
                "dataset_name": cfg.dataset_name,
                "cvd_count": len(paths),
                "common_cycles_in_dataset": tuple(common_cycles),
                "requested_cycles_available": tuple(
                    cycle for cycle in requested_cycles if cycle in common_cycles
                ),
            }
        )
    return pd.DataFrame(rows)


def fallback_axis(cfg: DatasetConfig, n_points: int) -> np.ndarray:
    v_min, v_max = cfg.voltage_bounds
    if v_min is not None and v_max is not None:
        return np.linspace(v_min, v_max, n_points, dtype=np.float32)
    return np.arange(n_points, dtype=np.float32)


def read_cvd_record(
    path: Path,
    cfg: DatasetConfig,
    cycles: tuple[int, ...],
    value_key: str = DEFAULT_CVD_VALUE_KEY,
    include_voltage_axis: bool = True,
) -> tuple[dict, np.ndarray] | None:
    with Path(path).open("rb") as handle:
        record = pickle.load(handle)

    cycle_bank = record.get("cycles", {})
    signal_channels = []
    voltage_axis_channels = []
    axis_lengths = []
    axis_mins = []
    axis_maxs = []
    for cycle in cycles:
        payload = cycle_payload(cycle_bank, cycle)
        if payload is None or value_key not in payload:
            return None
        values = np.asarray(payload[value_key], dtype=np.float32)
        axis_values = payload.get("axis", None)
        axis = (
            np.asarray(axis_values, dtype=np.float32)
            if axis_values is not None
            else fallback_axis(cfg, values.size)
        )
        if axis.size != values.size:
            axis = fallback_axis(cfg, values.size)

        signal_channels.append(values)
        voltage_axis_channels.append(axis)
        axis_lengths.append(values.size)
        axis_mins.append(float(np.nanmin(axis)))
        axis_maxs.append(float(np.nanmax(axis)))

    if len(set(axis_lengths)) != 1:
        return None

    channel_names = cvd_input_channel_names(cycles, value_key, include_voltage_axis)
    tensor_channels = signal_channels + voltage_axis_channels if include_voltage_axis else signal_channels
    cell = record.get("cell") or Path(path).name.replace("_CVD.pkl", "")
    metadata = parse_cell_metadata(cell, cfg)
    row = {
        "cvd_index": None,
        "dataset_name": cfg.dataset_name,
        "cell": cell,
        "cvd_path": str(path),
        "cvd_life": record.get("life", np.nan),
        "cvd_group": record.get("group", None),
        "cvd_cycles": tuple(cycles),
        "cvd_input_channels": tuple(channel_names),
        "cvd_axis_min_V": float(np.nanmin(axis_mins)),
        "cvd_axis_max_V": float(np.nanmax(axis_maxs)),
        "cvd_axis_length": int(axis_lengths[0]),
    }
    row.update(metadata)
    return row, np.stack(tensor_channels, axis=0)


def load_cvd_bank(
    configs: list[DatasetConfig],
    cycles: tuple[int, ...],
    value_key: str = DEFAULT_CVD_VALUE_KEY,
    include_voltage_axis: bool = True,
) -> tuple[pd.DataFrame, np.ndarray]:
    rows = []
    tensors = []
    channel_names = cvd_input_channel_names(cycles, value_key, include_voltage_axis)
    for cfg in configs:
        if not cfg.cvd_dir.exists():
            continue
        for path in sorted(cfg.cvd_dir.glob("*CVD.pkl")):
            loaded = read_cvd_record(path, cfg, cycles, value_key, include_voltage_axis)
            if loaded is None:
                continue
            row, tensor = loaded
            row["cvd_index"] = len(tensors)
            rows.append(row)
            tensors.append(tensor)

    cvd_df = pd.DataFrame(rows)
    cvd_array = (
        np.stack(tensors, axis=0)
        if tensors
        else np.empty((0, len(channel_names), 0), dtype=np.float32)
    )
    return cvd_df, cvd_array


def extract_cvd_descriptor_row(path: Path, cfg: DatasetConfig, feature_cycles: tuple[int, ...] | None = None) -> dict:
    with Path(path).open("rb") as handle:
        record = pickle.load(handle)

    cell = record.get("cell") or Path(path).name.replace("_CVD.pkl", "")
    cycles = record.get("cycles", {})
    if feature_cycles is None:
        configured = record.get("feature_cycles") or (20, 50, 100)
        feature_cycles = tuple(int(cycle) for cycle in configured)

    row = {
        "dataset_name": cfg.dataset_name,
        "cell": cell,
        "life": record.get("life", np.nan),
        "cvd_group": record.get("group", None),
        "cvd_descriptor_path": str(path),
    }
    row.update(parse_cell_metadata(cell, cfg))

    for cycle in feature_cycles:
        payload = cycle_payload(cycles, int(cycle)) or {}
        metrics = payload.get("metrics", {}) or {}
        shape_features = payload.get("shape_features", {}) or {}
        row[f"vardQ{cycle}"] = metrics.get("log10vardQ", metrics.get("vardQ", np.nan))
        row[f"meandQ{cycle}"] = metrics.get("meandQ", np.nan)
        row[f"rngdQ{cycle}"] = metrics.get("rngdQ", np.nan)
        row[f"SOH{cycle}"] = payload.get("SOH", np.nan)
        for name, value in shape_features.items():
            row[f"{name}{cycle}"] = value
    return row


def load_cvd_descriptor_table(
    configs: list[DatasetConfig],
    feature_cycles: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    rows = []
    for cfg in configs:
        if not cfg.cvd_dir.exists():
            continue
        for path in sorted(cfg.cvd_dir.glob("*CVD.pkl")):
            rows.append(extract_cvd_descriptor_row(path, cfg, feature_cycles=feature_cycles))
    return pd.DataFrame(rows)


def interpolate_on_uniform_voltage_grid(
    axis: np.ndarray,
    values: np.ndarray,
    grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    axis = np.asarray(axis, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    valid = np.isfinite(axis) & np.isfinite(values)
    if valid.sum() < 2:
        return np.full(grid.shape, np.nan, dtype=np.float32), np.zeros(grid.shape, dtype=np.float32)

    axis = axis[valid]
    values = values[valid]
    order = np.argsort(axis)
    axis = axis[order]
    values = values[order]
    axis, unique_idx = np.unique(axis, return_index=True)
    values = values[unique_idx]
    if axis.size < 2:
        return np.full(grid.shape, np.nan, dtype=np.float32), np.zeros(grid.shape, dtype=np.float32)

    in_range = (grid >= float(axis.min())) & (grid <= float(axis.max()))
    interpolated = np.full(grid.shape, np.nan, dtype=np.float32)
    if in_range.any():
        interpolated[in_range] = np.interp(grid[in_range], axis, values).astype(np.float32)
    return interpolated, in_range.astype(np.float32)


def build_uniform_voltage_cvd_bank(
    cvd_df: pd.DataFrame,
    cycles: tuple[int, ...],
    value_key: str,
    voltage_range: tuple[float, float],
    n_points: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    grid = np.linspace(voltage_range[0], voltage_range[1], int(n_points), dtype=np.float32)
    channel_names = [f"cycle_{cycle}_{value_key}_uniform_voltage" for cycle in cycles]
    channel_names += [f"cycle_{cycle}_valid_mask_uniform_voltage" for cycle in cycles]

    rows = []
    tensors = []
    cvd_df_sorted = cvd_df.sort_values("cvd_index").reset_index(drop=True)
    for _, cvd_row in cvd_df_sorted.iterrows():
        with Path(cvd_row["cvd_path"]).open("rb") as handle:
            record = pickle.load(handle)
        cycle_bank = record.get("cycles", {})

        signal_channels = []
        mask_channels = []
        axis_mins = []
        axis_maxs = []
        ok = True
        for cycle in cycles:
            payload = cycle_payload(cycle_bank, cycle)
            if payload is None or value_key not in payload:
                ok = False
                break
            values = np.asarray(payload[value_key], dtype=np.float32)
            axis_values = payload.get("axis", None)
            axis = (
                np.asarray(axis_values, dtype=np.float32)
                if axis_values is not None
                else np.full(values.shape, np.nan, dtype=np.float32)
            )
            if axis.size != values.size:
                axis = np.full(values.shape, np.nan, dtype=np.float32)
            interpolated, valid_mask = interpolate_on_uniform_voltage_grid(axis, values, grid)
            signal_channels.append(
                np.nan_to_num(interpolated, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            )
            mask_channels.append(valid_mask.astype(np.float32))
            finite_axis = axis[np.isfinite(axis)]
            axis_mins.append(float(np.nanmin(finite_axis)) if finite_axis.size else np.nan)
            axis_maxs.append(float(np.nanmax(finite_axis)) if finite_axis.size else np.nan)

        if not ok:
            continue
        if int(cvd_row["cvd_index"]) != len(tensors):
            raise ValueError("Uniform voltage builder expected cvd_index to match tensor order.")

        tensors.append(np.stack(signal_channels + mask_channels, axis=0))
        rows.append(
            {
                "cvd_index": int(cvd_row["cvd_index"]),
                "dataset_name": cvd_row["dataset_name"],
                "cell": cvd_row["cell"],
                "uniform_voltage_min_V": float(voltage_range[0]),
                "uniform_voltage_max_V": float(voltage_range[1]),
                "uniform_voltage_points": int(n_points),
                "source_axis_min_V": float(np.nanmin(axis_mins)) if axis_mins else np.nan,
                "source_axis_max_V": float(np.nanmax(axis_maxs)) if axis_maxs else np.nan,
                "uniform_cvd_input_channels": tuple(channel_names),
            }
        )

    uniform_cvd_df = pd.DataFrame(rows)
    uniform_cvd_array = (
        np.stack(tensors, axis=0)
        if tensors
        else np.empty((0, len(channel_names), int(n_points)), dtype=np.float32)
    )
    return uniform_cvd_df, uniform_cvd_array, grid, channel_names


def build_normalized_voltage_cvd_bank(
    cvd_df: pd.DataFrame,
    cvd_array: np.ndarray,
    cycles: tuple[int, ...],
    value_key: str,
    n_points: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    grid = np.linspace(0.0, 1.0, int(n_points), dtype=np.float32)
    channel_names = [f"cycle_{cycle}_{value_key}_normalized_voltage" for cycle in cycles]
    channel_names += [f"cycle_{cycle}_normalized_voltage_axis" for cycle in cycles]

    cvd_df_sorted = cvd_df.sort_values("cvd_index").reset_index(drop=True)
    n_samples = len(cvd_df_sorted)
    n_cycles = len(cycles)
    if cvd_array.ndim != 3 or cvd_array.shape[1] < 2 * n_cycles:
        raise ValueError("Normalized voltage builder expects signal and voltage-axis channels.")

    normalized_cvd_array = np.empty((n_samples, 2 * n_cycles, int(n_points)), dtype=np.float32)
    rows = []
    for row_pos, cvd_row in enumerate(cvd_df_sorted.itertuples(index=False)):
        cvd_index = int(getattr(cvd_row, "cvd_index"))
        sample = cvd_array[cvd_index]
        signal_bank = sample[:n_cycles, :]
        axis_bank = sample[n_cycles : 2 * n_cycles, :]

        for cycle_idx in range(n_cycles):
            values = np.asarray(signal_bank[cycle_idx], dtype=np.float32)
            axis = np.asarray(axis_bank[cycle_idx], dtype=np.float32)
            valid = np.isfinite(axis) & np.isfinite(values)
            if valid.sum() >= 2:
                axis_valid = axis[valid].astype(np.float32, copy=False)
                values_valid = values[valid].astype(np.float32, copy=False)
            else:
                value_valid = np.isfinite(values)
                values_valid = (
                    values[value_valid].astype(np.float32, copy=False)
                    if value_valid.sum() >= 2
                    else np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(
                        np.float32, copy=False
                    )
                )
                axis_valid = np.linspace(0.0, 1.0, int(values_valid.size), dtype=np.float32)

            order = np.argsort(axis_valid, kind="mergesort")
            axis_valid = axis_valid[order]
            values_valid = values_valid[order]
            axis_valid, unique_idx = np.unique(axis_valid, return_index=True)
            values_valid = values_valid[unique_idx]
            if axis_valid.size < 2:
                axis_valid = np.linspace(0.0, 1.0, max(int(values_valid.size), 2), dtype=np.float32)
                if values_valid.size < 2:
                    values_valid = np.repeat(values_valid[:1], 2).astype(np.float32)

            axis_min = float(axis_valid[0])
            axis_max = float(axis_valid[-1])
            normalized_axis = (
                np.linspace(0.0, 1.0, int(values_valid.size), dtype=np.float32)
                if not np.isfinite(axis_min) or not np.isfinite(axis_max) or axis_max <= axis_min
                else ((axis_valid - axis_min) / (axis_max - axis_min)).astype(np.float32)
            )
            normalized_cvd_array[row_pos, cycle_idx, :] = np.interp(
                grid, normalized_axis, values_valid
            ).astype(np.float32)
            normalized_cvd_array[row_pos, n_cycles + cycle_idx, :] = grid

        rows.append(
            {
                "cvd_index": cvd_index,
                "dataset_name": getattr(cvd_row, "dataset_name"),
                "cell": getattr(cvd_row, "cell"),
                "normalized_voltage_min": 0.0,
                "normalized_voltage_max": 1.0,
                "normalized_voltage_points": int(n_points),
                "normalized_cvd_input_channels": tuple(channel_names),
            }
        )

    normalized_cvd_df = pd.DataFrame(rows)
    return normalized_cvd_df, normalized_cvd_array, grid, channel_names

