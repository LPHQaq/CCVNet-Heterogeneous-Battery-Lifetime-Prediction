from __future__ import annotations

import argparse
import importlib
import pickle
import re
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

SHAPE_FEATURE_NAMES = [
    "V_at_min_dQ",
    "centroid_V",
    "width_halfmin",
    "area_low_frac",
    "area_mid_frac",
    "area_high_frac",
]


def advanced_smooth_capacity_curve(
    capacity_values,
    smooth_method="none",
    smooth_window=11,
    smooth_polyorder=3,
    smooth_window_ratio=20,
    spike_abs=0.20,
    spike_k=6.0,
):
    values = np.asarray(capacity_values, dtype=float)
    if values.size == 0 or smooth_method != "savgol":
        return values

    finite_mask = np.isfinite(values)
    if not np.any(finite_mask):
        return values

    filled = values.copy()
    x = np.arange(filled.size)
    if not np.all(finite_mask):
        filled[~finite_mask] = np.interp(x[~finite_mask], x[finite_mask], filled[finite_mask])

    fixed = filled.copy()
    for index in range(1, fixed.size - 1):
        local_values = filled[index - 1:index + 2]
        local_median = float(np.nanmedian(local_values))
        local_mad = float(np.nanmedian(np.abs(local_values - local_median)))
        local_std = float(np.nanstd(local_values))
        local_scale = local_mad if np.isfinite(local_mad) and local_mad > 1e-12 else local_std
        local_threshold = max(spike_abs, spike_k * local_scale) if np.isfinite(local_scale) else spike_abs
        if np.isfinite(filled[index]) and np.isfinite(local_median) and abs(filled[index] - local_median) > local_threshold:
            left_value = filled[index - 1]
            right_value = filled[index + 1]
            fixed[index] = 0.5 * (left_value + right_value) if np.isfinite(left_value) and np.isfinite(right_value) else local_median

    window = int(fixed.size / smooth_window_ratio) if smooth_window_ratio and smooth_window_ratio > 0 else int(smooth_window)
    if window % 2 == 0:
        window += 1
    window = min(window, fixed.size)
    if window % 2 == 0:
        window -= 1
    if window < max(3, smooth_polyorder + 2):
        return fixed
    try:
        return savgol_filter(fixed, window, smooth_polyorder, mode="interp")
    except Exception:
        return fixed


def parse_preprocess_args(dataset_name: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Preprocess {dataset_name} raw cycles into CCVNet CVD inputs.")
    parser.add_argument("--rawdata-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-feature-csv", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def safe_dataset_tag(dataset_folder_name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in dataset_folder_name).strip("_")


def _loader_name(module: ModuleType) -> str | None:
    for name in (
        "load_analysis_cycles",
        "load_calce_pickle_cycles_by_name",
        "load_mich_joule_pickle_cycles_by_name",
        "load_mich_pickle_cycles_by_name",
        "load_rwth_pickle_cycles_by_name",
        "load_xjtu_pickle_cycles_by_name",
        "load_pickle_cycles_by_name",
        "load_cell_cycles",
    ):
        if hasattr(module, name):
            return name
    return None


def list_raw_files(module: ModuleType, rawdata_dir: Path) -> list[str]:
    files = [p for p in rawdata_dir.iterdir() if p.suffix.lower() in {".pkl", ".pickle"}]
    keep = getattr(module, "keep_mich_cell_file", None)
    if callable(keep):
        files = [p for p in files if keep(p)]
    key = getattr(module, "sort_key_by_battery_id", None)
    if callable(key):
        return [p.name for p in sorted(files, key=lambda p: key(p.name))]
    return [p.name for p in sorted(files)]


def prepare_module(module: ModuleType, rawdata_dir: Path, data_files: list[str]) -> None:
    module.RAWDATA_DIR = rawdata_dir
    module.datafile = list(data_files)
    module.CELL_RAW_CACHE = {}
    module.ANALYSIS_CELL_CACHE = {}
    module.CELL_CAPACITY_SUMMARY_CACHE = {}
    module.cell_name_by_obj_id = {}
    module.analysis_start_cycle_by_name = {Path(name).stem: 1 for name in data_files}


def load_one_cell(module: ModuleType, file_name: str) -> tuple[list[dict], int]:
    stem = Path(file_name).stem
    loader = _loader_name(module)
    if loader is None:
        raise AttributeError(f"No raw-cycle loader found in {module.__name__}")
    payload = getattr(module, loader)(file_name)
    if isinstance(payload, dict) and "cycles" in payload:
        cycles = payload["cycles"]
        start_cycle = int(payload.get("start_cycle", 1))
    else:
        cycles = payload
        start_cycle = 1
        if hasattr(module, "trim_cycles_to_peak_capacity"):
            cycles, peak_idx, _ = module.trim_cycles_to_peak_capacity(cycles)
            start_cycle = int(peak_idx + 1)
    module.cell_name_by_obj_id[id(cycles)] = file_name
    module.analysis_start_cycle_by_name[stem] = start_cycle
    return cycles, start_cycle


def empty_curve_export(n_points: int, shape_feature_names: list[str]) -> dict[str, Any]:
    nan_axis = np.full(int(n_points), np.nan, dtype=np.float32)
    return {
        "axis": nan_axis.copy(),
        "negative_difference": nan_axis.copy(),
        "difference": nan_axis.copy(),
        "ref_interp": nan_axis.copy(),
        "tgt_interp": nan_axis.copy(),
        "metrics": {
            "vardQ": float("nan"),
            "log10vardQ": float("nan"),
            "meandQ": float("nan"),
            "rngdQ": float("nan"),
        },
        "shape_features": {shape_name: float("nan") for shape_name in shape_feature_names},
        "SOH": float("nan"),
    }


def export_dataset(
    module: ModuleType,
    rawdata_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    overwrite: bool = True,
    save_feature_csv: bool = True,
) -> dict[str, Any]:
    settings = module.DATASET_SETTINGS
    source_dir = Path(rawdata_dir) if rawdata_dir is not None else Path(module.CURRENT_DIR) / "rawdata"
    out_dir = Path(output_dir) if output_dir is not None else Path("data") / "processed" / settings["dataset_name"]
    cvd_curve_dir = out_dir / "CVD_curve"
    cvd_curve_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_files = list_raw_files(module, source_dir)
    prepare_module(module, source_dir, data_files)

    cvd_curve_cycles = list(settings["cvd_curve_cycles"])
    feature_cycles = list(settings["feature_cycles"])
    life_threshold = float(settings["life_threshold"])
    n_points = int(settings["n_voltage_grid"])
    shape_feature_names = list(getattr(module, "shape_feature_names", SHAPE_FEATURE_NAMES))

    records = []
    stage_records = []
    exported_cells = []
    skipped_rows = []

    for file_name in data_files:
        cell_name = Path(file_name).stem
        try:
            cell_cycles, start_cycle = load_one_cell(module, file_name)
        except Exception as exc:
            skipped_rows.append({"cell": cell_name, "reason": f"load_failed: {exc}"})
            continue

        should_exclude = getattr(module, "should_exclude_export_cell", None)
        if callable(should_exclude) and should_exclude(cell_name):
            skipped_rows.append({"cell": cell_name, "reason": "dataset_excluded"})
            continue
        excluded_groups = set(settings.get("save_excluded_groups", [])) | set(getattr(module, "EXPORT_EXCLUDED_GROUPS", set()))
        group_name = module.match_plot_group(cell_name) if hasattr(module, "match_plot_group") else None
        if group_name in excluded_groups:
            skipped_rows.append({"cell": cell_name, "group": group_name, "reason": "group_excluded"})
            continue

        capacity_summary = module.summarize_capacity_pipeline_local(cell_cycles, threshold=life_threshold)
        soh_series = np.asarray(capacity_summary["soh"], dtype=float)
        life_value = capacity_summary["life_cycle"]
        record = {"cell": cell_name, "life": float(life_value) if np.isfinite(life_value) else float("nan")}
        stage_record = {"cell": cell_name}
        curve_payload = {
            "cell": cell_name,
            "group": group_name,
            "life": float(life_value) if np.isfinite(life_value) else float("nan"),
            "life_threshold": life_threshold,
            "start_cycle_original": int(start_cycle),
            "curve_cycles": cvd_curve_cycles.copy(),
            "feature_cycles": feature_cycles.copy(),
            "cycles": {},
        }

        for target_cycle in feature_cycles:
            for metric_name in ["vardQ", "meandQ", "rngdQ", "SOH"]:
                record[f"{metric_name}{target_cycle}"] = float("nan")
            for shape_name in shape_feature_names:
                stage_record[f"{shape_name}{target_cycle}"] = float("nan")
            if len(soh_series) >= target_cycle and np.isfinite(soh_series[target_cycle - 1]):
                record[f"SOH{target_cycle}"] = float(soh_series[target_cycle - 1])

        for target_cycle in cvd_curve_cycles:
            curve_export = empty_curve_export(n_points, shape_feature_names)
            curve_export["SOH"] = float(soh_series[target_cycle - 1]) if len(soh_series) >= target_cycle and np.isfinite(soh_series[target_cycle - 1]) else float("nan")
            curve_export["target_cycle_original"] = int(start_cycle + target_cycle - 1)
            profile_i = module.compute_difference_profile(cell_cycles, target_cycle)
            if profile_i is not None:
                metrics_i = module.compute_summary_metrics(profile_i)
                shape_i = module.compute_stage_features(profile_i)
                curve_export.update({
                    "axis": np.asarray(profile_i["axis"], dtype=np.float32),
                    "negative_difference": np.asarray(profile_i["negative_difference"], dtype=np.float32),
                    "difference": np.asarray(profile_i["difference"], dtype=np.float32),
                    "ref_interp": np.asarray(profile_i["ref_interp"], dtype=np.float32),
                    "tgt_interp": np.asarray(profile_i["tgt_interp"], dtype=np.float32),
                    "metrics": metrics_i,
                    "shape_features": shape_i,
                })
                if target_cycle in feature_cycles:
                    record[f"vardQ{target_cycle}"] = float(metrics_i["log10vardQ"])
                    record[f"meandQ{target_cycle}"] = float(metrics_i["meandQ"])
                    record[f"rngdQ{target_cycle}"] = float(metrics_i["rngdQ"])
                    for shape_name in shape_feature_names:
                        stage_record[f"{shape_name}{target_cycle}"] = float(shape_i.get(shape_name, np.nan))
            curve_payload["cycles"][int(target_cycle)] = curve_export

        cvd_path = cvd_curve_dir / f"{cell_name}_CVD.pkl"
        if overwrite or not cvd_path.exists():
            with cvd_path.open("wb") as handle:
                pickle.dump(curve_payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        records.append(record)
        stage_records.append(stage_record)
        exported_cells.append(cell_name)

    feature_csv_path = out_dir / f"{safe_dataset_tag(settings['dataset_name'])}_features.csv"
    if save_feature_csv:
        dfm = pd.DataFrame(records)
        stage_df = pd.DataFrame(stage_records)
        if not dfm.empty and not stage_df.empty:
            dfm_stage = dfm.merge(stage_df, on="cell", how="left")
        else:
            dfm_stage = dfm

        feature_export_cols = ["cell", "life"]
        for target_cycle in feature_cycles:
            for metric_name in ["vardQ", "meandQ", "rngdQ", "SOH"]:
                col = f"{metric_name}{target_cycle}"
                if col in dfm_stage.columns:
                    feature_export_cols.append(col)
        for target_cycle in feature_cycles:
            for shape_name in shape_feature_names:
                col = f"{shape_name}{target_cycle}"
                if col in dfm_stage.columns:
                    feature_export_cols.append(col)

        if not dfm_stage.empty:
            final_feature_df = dfm_stage.loc[:, list(dict.fromkeys(feature_export_cols))].copy()
        else:
            final_feature_df = pd.DataFrame(columns=feature_export_cols)
        final_feature_df.to_csv(feature_csv_path, index=False)

    skipped_df = pd.DataFrame(skipped_rows)
    skipped_path = out_dir / "preprocess_skipped_cells.csv"
    if not skipped_df.empty:
        skipped_df.to_csv(skipped_path, index=False)

    return {
        "dataset_name": settings["dataset_name"],
        "rawdata_dir": str(source_dir),
        "output_dir": str(out_dir),
        "cvd_curve_dir": str(cvd_curve_dir),
        "feature_csv": str(feature_csv_path) if save_feature_csv else None,
        "n_cells": len(data_files),
        "n_exported": len(exported_cells),
        "n_skipped": len(skipped_rows),
    }


def run_preprocess(dataset: str, rawdata_dir: Path | None, output_dir: Path | None, overwrite: bool = True, save_feature_csv: bool = True) -> dict[str, Any]:
    dataset_modules = {
        "MATR": "process_MATR",
        "HUST": "process_HUST",
        "MICH_Joule": "process_MICH_Joule",
        "MICH_JECS": "process_MICH_JECS",
        "CALCE": "process_CALCE",
        "RWTH": "process_RWTH",
        "SDU": "process_SDU",
        "STAN": "process_STAN",
        "TONGJI": "process_TONGJI",
        "XJTU": "process_XJTU",
    }

    if dataset == "all":
        summaries = []
        for name in dataset_modules:
            module = importlib.import_module(dataset_modules[name])
            summaries.append(module.run(rawdata_dir=None, output_dir=None, overwrite=overwrite, save_feature_csv=save_feature_csv))
        return {"datasets": summaries}
    if dataset not in dataset_modules:
        valid = ", ".join(["all", *dataset_modules.keys()])
        raise ValueError(f"Unknown dataset '{dataset}'. Valid choices: {valid}")
    module = importlib.import_module(dataset_modules[dataset])
    return module.run(rawdata_dir=rawdata_dir, output_dir=output_dir, overwrite=overwrite, save_feature_csv=save_feature_csv)
