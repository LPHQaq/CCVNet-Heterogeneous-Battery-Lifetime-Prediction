from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from common import advanced_smooth_capacity_curve, export_dataset

CURRENT_DIR = Path.cwd()
RAWDATA_DIR = CURRENT_DIR / "rawdata"
CELL_RAW_CACHE = {}
ANALYSIS_CELL_CACHE = {}
CELL_CAPACITY_SUMMARY_CACHE = {}
cell_name_by_obj_id = {}
analysis_start_cycle_by_name = {}
datafile = []

CVDCYCLE_KEEP_KEYS = ("current_in_A", "voltage_in_V", "discharge_capacity_in_Ah")


DATASET_SETTINGS = {
    "dataset_name": "MATR(LFP)",
    "cvd_curve_cycles": list(range(10, 101, 10)),
    "feature_cycles": [20, 50, 100],
    "reference_cycle": 10,
    "voltage_window": (2.0, 3.6),
    "n_voltage_grid": 200,
    "life_threshold": 80.0,
    "life_end_tolerance": 1.0,
    "soh_baseline_note": "SOH baseline = fixed nominal capacity of 1.1 Ah for every MATR cell.",
    "segmentation_note": "segmentation rule = keep the MATR-specific last-Vmax to tail-trimmed discharge branch in the dedicated extraction cell below.",
    "plot_group_note": "plot grouping = no subgroup split; use overall Total views only.",
    "plot_groups": [],
    "processing_config": {
        "smooth_polyorder": 3,
        "smooth_window_ratio": 20,
        "spike_abs": 0.20,
        "spike_k": 6.0,
        "current_threshold": -0.05,
        "voltage_round_decimals": 4,
        "q_interp_window_ratio": 0,
        "overlap_pad": 0.0,
        "q_zero_tol": 1e-5,
        "vmax_check_tol": 5e-3,
        "voltage_monotonic_tol": 5e-3,
        "capacity_monotonic_tol": 1e-6,
        "tail_trim": 100,
    },
}


PROCESSING_CONFIG = dict(DATASET_SETTINGS["processing_config"] )

# CVD naming in the publication repo; numerical settings are inherited from the notebook.
dataset_settings = DATASET_SETTINGS
cvd_curve_cycles = list(DATASET_SETTINGS["cvd_curve_cycles"])
cvd_curve_cycles = cvd_curve_cycles
feature_cycles = list(DATASET_SETTINGS["feature_cycles"])
target_cycles = feature_cycles.copy()
cyc_ref = int(DATASET_SETTINGS["reference_cycle"])
Vmin, Vmax = DATASET_SETTINGS["voltage_window"]
nV = int(DATASET_SETTINGS["n_voltage_grid"])
LIFE_THRESHOLD = float(DATASET_SETTINGS["life_threshold"])
LIFE_END_TOLERANCE = float(DATASET_SETTINGS.get("life_end_tolerance", 1.0))


for _config_key, _config_value in PROCESSING_CONFIG.items():
    globals().setdefault(_config_key, _config_value)

def trim_cycle_fields(cycle_rows):
    return [{key: cycle.get(key) for key in CVDCYCLE_KEEP_KEYS} for cycle in cycle_rows]


def load_pickle_cycles_by_name(name):
    cached = CELL_RAW_CACHE.get(name)
    if cached is not None:
        return cached
    with open(RAWDATA_DIR / name, "rb") as f:
        obj = pickle.load(f)
    cycle_rows = obj["cycle_data"] if isinstance(obj, dict) and "cycle_data" in obj else obj
    cell_cycles = trim_cycle_fields(cycle_rows)
    CELL_RAW_CACHE[name] = cell_cycles
    cell_name_by_obj_id[id(cell_cycles)] = name
    return cell_cycles


def load_cell_cycles(cell_name):
    return load_pickle_cycles_by_name(cell_name)


def dataset_nominal_capacity_ah(cell_name):
    return 1.1


def dataset_major_group(cell_name):
    return None


def dataset_plot_group(cell_name):
    return None


def get_nominal_capacity_ah(cell_name):
    return dataset_nominal_capacity_ah(cell_name)


def match_major_group(cell_name):
    return dataset_major_group(cell_name)


def match_plot_group(cell_name):
    return dataset_plot_group(cell_name)


def get_processing_config():
    return PROCESSING_CONFIG.copy()


def resolve_nominal_capacity_ah(cell_cycles):
    cell_name = cell_name_by_obj_id.get(id(cell_cycles), "")
    nominal_capacity_local = get_nominal_capacity_ah(Path(cell_name).stem)
    if np.isfinite(nominal_capacity_local) and nominal_capacity_local > 0.0:
        return float(nominal_capacity_local)
    finite_capacities = []
    for cycle_data in cell_cycles:
        try:
            discharge_capacity = np.asarray(cycle_data["discharge_capacity_in_Ah"], dtype=float)
            finite_capacities.append(float(np.nanmax(discharge_capacity)))
        except Exception:
            continue
    return float(np.nanmax(finite_capacities)) if finite_capacities else float("nan")


def extract_cycle_capacity_ah(cycle_data):
    discharge_capacity = np.asarray(cycle_data["discharge_capacity_in_Ah"], dtype=float)
    current = np.asarray(cycle_data.get("current_in_A", np.full_like(discharge_capacity, np.nan)), dtype=float)
    valid = np.isfinite(discharge_capacity)
    if np.any(np.isfinite(current)):
        discharge_mask = np.isfinite(current) & (current < PROCESSING_CONFIG["current_threshold"]) & valid
        if np.sum(discharge_mask) >= 3:
            return float(np.nanmax(discharge_capacity[discharge_mask]))
    return float(np.nanmax(discharge_capacity[valid])) if np.any(valid) else float("nan")


def collect_cycle_capacities(cell_cycles):
    raw_capacities = []
    for cycle_data in cell_cycles:
        try:
            raw_capacities.append(extract_cycle_capacity_ah(cycle_data))
        except Exception:
            raw_capacities.append(float("nan"))
    return np.asarray(raw_capacities, dtype=float)


def build_capacity_trace_local(cell_cycles):
    cache_key = ("trace", id(cell_cycles))
    cached = CELL_CAPACITY_SUMMARY_CACHE.get(cache_key)
    if cached is not None:
        return cached.copy()

    raw_capacities = collect_cycle_capacities(cell_cycles)
    capacities = advanced_smooth_capacity_curve(
        raw_capacities,
        smooth_method="savgol",
        smooth_polyorder=PROCESSING_CONFIG["smooth_polyorder"],
        smooth_window_ratio=PROCESSING_CONFIG["smooth_window_ratio"],
        spike_abs=PROCESSING_CONFIG["spike_abs"],
        spike_k=PROCESSING_CONFIG["spike_k"],
    )
    CELL_CAPACITY_SUMMARY_CACHE[cache_key] = capacities.copy()
    return capacities.copy()


def build_soh_series_local(cell_cycles):
    cache_key = ("soh", id(cell_cycles))
    cached = CELL_CAPACITY_SUMMARY_CACHE.get(cache_key)
    if cached is not None:
        return cached.copy()

    capacities = build_capacity_trace_local(cell_cycles)
    nominal_capacity_local = resolve_nominal_capacity_ah(cell_cycles)
    if np.isfinite(nominal_capacity_local) and nominal_capacity_local > 0.0:
        soh_series = capacities / float(nominal_capacity_local) * 100.0
    else:
        q0 = float(np.nanmax(capacities)) if np.any(np.isfinite(capacities)) else float("nan")
        soh_series = capacities / q0 * 100.0 if np.isfinite(q0) and q0 > 0.0 else np.full_like(capacities, np.nan)
    CELL_CAPACITY_SUMMARY_CACHE[cache_key] = soh_series.copy()
    return soh_series.copy()


def extract_life_cycle_from_soh(soh_series, threshold=LIFE_THRESHOLD, end_tol=LIFE_END_TOLERANCE):
    soh_array = np.asarray(soh_series, dtype=float)
    finite_idx = np.flatnonzero(np.isfinite(soh_array))
    if finite_idx.size == 0:
        return float("nan")

    for pos in range(finite_idx.size - 1):
        idx = int(finite_idx[pos])
        next_idx = int(finite_idx[pos + 1])
        if soh_array[idx] > threshold and soh_array[next_idx] < threshold:
            return float(idx + 1)

    last_idx = int(finite_idx[-1])
    last_soh = float(soh_array[last_idx])
    if threshold < last_soh < threshold + end_tol:
        return float(last_idx + 1)
    return float("nan")


def summarize_capacity_pipeline_local(cell_cycles, threshold=LIFE_THRESHOLD):
    capacity_trace = build_capacity_trace_local(cell_cycles)
    soh_series = build_soh_series_local(cell_cycles)
    life_cycle = extract_life_cycle_from_soh(soh_series, threshold=threshold)
    return {
        "trace": capacity_trace,
        "soh": soh_series,
        "life_cycle": life_cycle,
    }


def extract_segment_pair(ref_cycle_data, tgt_cycle_data):
    tail_trim_local = int(PROCESSING_CONFIG["tail_trim"])
    ref_voltage = np.asarray(ref_cycle_data["voltage_in_V"], dtype=float)
    ref_capacity = np.asarray(ref_cycle_data["discharge_capacity_in_Ah"], dtype=float)
    tgt_voltage = np.asarray(tgt_cycle_data["voltage_in_V"], dtype=float)
    tgt_capacity = np.asarray(tgt_cycle_data["discharge_capacity_in_Ah"], dtype=float)

    if min(ref_voltage.size, ref_capacity.size, tgt_voltage.size, tgt_capacity.size) < tail_trim_local + 5:
        return None

    sref = int(ref_voltage.size - np.argmax(ref_voltage[::-1]) - 1)
    eref = int(ref_voltage.size - tail_trim_local)
    st = int(tgt_voltage.size - np.argmax(tgt_voltage[::-1]) - 1)
    et = int(tgt_voltage.size - tail_trim_local)
    if eref <= sref + 1 or et <= st + 1:
        return None

    ref_keep_idx = np.arange(sref, eref, dtype=int)
    tgt_keep_idx = np.arange(st, et, dtype=int)
    ref_axis = ref_voltage[ref_keep_idx]
    ref_value = ref_capacity[ref_keep_idx]
    tgt_axis = tgt_voltage[tgt_keep_idx]
    tgt_value = tgt_capacity[tgt_keep_idx]

    ref_valid = np.isfinite(ref_axis) & np.isfinite(ref_value)
    tgt_valid = np.isfinite(tgt_axis) & np.isfinite(tgt_value)
    if np.sum(ref_valid) < 3 or np.sum(tgt_valid) < 3:
        return None

    ref_axis = ref_axis[ref_valid]
    ref_value = ref_value[ref_valid]
    tgt_axis = tgt_axis[tgt_valid]
    tgt_value = tgt_value[tgt_valid]
    ref_keep_idx = ref_keep_idx[ref_valid]
    tgt_keep_idx = tgt_keep_idx[tgt_valid]

    v_lo = max(float(np.nanmin(ref_axis)), float(np.nanmin(tgt_axis)), float(Vmin + overlap_pad))
    v_hi = min(float(np.nanmax(ref_axis)), float(np.nanmax(tgt_axis)), float(Vmax - overlap_pad))
    if not (np.isfinite(v_lo) and np.isfinite(v_hi)) or v_hi <= v_lo:
        return None

    return {
        "ref_axis": ref_axis,
        "ref_value": ref_value,
        "tgt_axis": tgt_axis,
        "tgt_value": tgt_value,
        "ref_interp_axis": ref_axis,
        "ref_interp_value": ref_value,
        "tgt_interp_axis": tgt_axis,
        "tgt_interp_value": tgt_value,
        "plot_axis": ref_axis,
        "plot_value": ref_value,
        "vmax": float(ref_axis[0]),
        "window": (v_lo, v_hi),
        "slice_info": {
            "ref": (sref, eref),
            "tgt": (st, et),
            "ref_keep_indices": ref_keep_idx.tolist(),
            "tgt_keep_indices": tgt_keep_idx.tolist(),
            "ref_raw_points": int(ref_keep_idx.size),
            "tgt_raw_points": int(tgt_keep_idx.size),
            "ref_start_q": float(ref_value[0]),
            "ref_start_v": float(ref_axis[0]),
            "tgt_start_q": float(tgt_value[0]),
            "tgt_start_v": float(tgt_axis[0]),
            "v_lo": v_lo,
            "v_hi": v_hi,
        },
    }


def validate_extracted_branch(branch):
    if branch is None:
        return {
            "extracted": False,
            "start_q_ok": False,
            "start_vmax_ok": False,
            "voltage_desc_ok": False,
            "capacity_asc_ok": False,
        }

    plot_axis = np.asarray(branch["plot_axis"], dtype=float)
    plot_value = np.asarray(branch["plot_value"], dtype=float)

    if plot_axis.size == 0 or plot_value.size == 0:
        return {
            "extracted": False,
            "start_q_ok": False,
            "start_vmax_ok": False,
            "voltage_desc_ok": False,
            "capacity_asc_ok": False,
        }

    return {
        "extracted": True,
        "start_q_ok": bool(np.isfinite(plot_value[0]) and abs(float(plot_value[0])) <= PROCESSING_CONFIG["q_zero_tol"]),
        "start_vmax_ok": bool(np.isfinite(plot_axis[0]) and abs(float(plot_axis[0]) - float(branch["vmax"])) <= PROCESSING_CONFIG["vmax_check_tol"]),
        "voltage_desc_ok": bool(np.all(np.diff(plot_axis) <= PROCESSING_CONFIG["voltage_monotonic_tol"])),
        "capacity_asc_ok": bool(np.all(np.diff(plot_value) >= -PROCESSING_CONFIG["capacity_monotonic_tol"])),
    }


def compute_difference_profile(cell_cycles, target_cycle):
    if "extract_segment_pair" not in globals():
        raise RuntimeError("Run the dataset-specific extraction cell before calling compute_difference_profile.")
    if len(cell_cycles) < max(cyc_ref, target_cycle):
        return None

    segment_pair = extract_segment_pair(cell_cycles[cyc_ref - 1], cell_cycles[target_cycle - 1])
    if segment_pair is None:
        return None

    v_lo, v_hi = segment_pair["window"]
    axis = np.linspace(v_lo, v_hi, int(nV))
    ref_interp_raw = interp1d(segment_pair["ref_interp_axis"], segment_pair["ref_interp_value"], kind="linear", bounds_error=False, fill_value=np.nan)(axis)
    tgt_interp_raw = interp1d(segment_pair["tgt_interp_axis"], segment_pair["tgt_interp_value"], kind="linear", bounds_error=False, fill_value=np.nan)(axis)
    q_interp_window_ratio = PROCESSING_CONFIG["q_interp_window_ratio"] or PROCESSING_CONFIG["smooth_window_ratio"]
    ref_interp = advanced_smooth_capacity_curve(ref_interp_raw, smooth_method="savgol", smooth_polyorder=PROCESSING_CONFIG["smooth_polyorder"], smooth_window_ratio=q_interp_window_ratio, spike_abs=PROCESSING_CONFIG["spike_abs"], spike_k=PROCESSING_CONFIG["spike_k"])
    tgt_interp = advanced_smooth_capacity_curve(tgt_interp_raw, smooth_method="savgol", smooth_polyorder=PROCESSING_CONFIG["smooth_polyorder"], smooth_window_ratio=q_interp_window_ratio, spike_abs=PROCESSING_CONFIG["spike_abs"], spike_k=PROCESSING_CONFIG["spike_k"])
    difference = tgt_interp - ref_interp
    negative_difference = np.where(np.isnan(difference) | (difference < 0.0), difference, 0.0).astype(float)

    return {
        **segment_pair,
        "axis": axis,
        "ref_interp_raw": ref_interp_raw,
        "tgt_interp_raw": tgt_interp_raw,
        "ref_interp": ref_interp,
        "tgt_interp": tgt_interp,
        "difference": difference,
        "negative_difference": negative_difference,
        "axis_label": "Voltage (V)",
        "diff_label": r"$\Delta Q$ (Ah)",
    }


def compute_summary_metrics(profile):
    values = np.asarray(profile["negative_difference"], dtype=float)
    vardq = float(np.nanvar(values))
    return {
        "vardQ": vardq,
        "log10vardQ": float(np.log10(vardq)) if np.isfinite(vardq) and vardq > 0.0 else float("nan"),
        "meandQ": float(np.nanmean(values)) if np.any(np.isfinite(values)) else float("nan"),
        "rngdQ": float(np.nanmax(values) - np.nanmin(values)) if np.any(np.isfinite(values)) else float("nan"),
    }


def compute_stage_features(profile):
    axis = np.asarray(profile["axis"], dtype=float)
    values = np.asarray(profile["negative_difference"], dtype=float)
    valid = np.isfinite(axis) & np.isfinite(values)
    if np.sum(valid) < 3:
        return None

    axis_valid = axis[valid]
    values_valid = values[valid]
    neg = np.abs(values_valid)
    total_neg = float(np.trapezoid(neg, axis_valid))
    if not np.isfinite(total_neg) or total_neg <= 0.0:
        return None

    min_index = int(np.nanargmin(values_valid))
    min_value = float(values_valid[min_index])
    centroid = float(np.trapezoid(axis_valid * neg, axis_valid) / total_neg)
    half_level = 0.5 * min_value
    width_mask = values_valid <= half_level
    axis_lo = float(np.nanmin(axis_valid))
    axis_hi = float(np.nanmax(axis_valid))
    third_1 = axis_lo + (axis_hi - axis_lo) / 3.0
    third_2 = axis_lo + 2.0 * (axis_hi - axis_lo) / 3.0
    low_mask = axis_valid <= third_1
    mid_mask = (axis_valid > third_1) & (axis_valid <= third_2)
    high_mask = axis_valid > third_2
    area_low = float(np.trapezoid(neg[low_mask], axis_valid[low_mask])) if np.sum(low_mask) >= 2 else 0.0
    area_mid = float(np.trapezoid(neg[mid_mask], axis_valid[mid_mask])) if np.sum(mid_mask) >= 2 else 0.0
    area_high = float(np.trapezoid(neg[high_mask], axis_valid[high_mask])) if np.sum(high_mask) >= 2 else 0.0

    return {
        "V_at_min_dQ": float(axis_valid[min_index]),
        "centroid_V": centroid,
        "width_halfmin": float(axis_valid[width_mask][-1] - axis_valid[width_mask][0]) if np.any(width_mask) else float("nan"),
        "area_low_frac": area_low / total_neg,
        "area_mid_frac": area_mid / total_neg,
        "area_high_frac": area_high / total_neg,
    }

def run(rawdata_dir: str | Path | None = None, output_dir: str | Path | None = None, overwrite: bool = True, save_feature_csv: bool = True):
    return export_dataset(__import__(__name__, fromlist=["*"]), rawdata_dir=rawdata_dir, output_dir=output_dir, overwrite=overwrite, save_feature_csv=save_feature_csv)


def main() -> None:
    from common import parse_preprocess_args

    args = parse_preprocess_args(DATASET_SETTINGS["dataset_name"])
    run(rawdata_dir=args.rawdata_dir, output_dir=args.output_dir, overwrite=args.overwrite, save_feature_csv=args.save_feature_csv)


if __name__ == "__main__":
    main()

