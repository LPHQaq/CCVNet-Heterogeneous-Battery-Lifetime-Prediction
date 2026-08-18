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


XJTU_PROTOCOL_GROUPS = [
    "2C",
    "3C",
    "R2.5",
    "R3",
    "RW",
    "Sim_satellite",
]


DATASET_SETTINGS = {
    "dataset_name": "XJTU(NMC)",
    "cvd_curve_cycles": list(range(10, 101, 10)),
    "feature_cycles": [20, 50, 100],
    "reference_cycle": 10,
    "voltage_window": (2.5, 4.2),
    "n_voltage_grid": 200,
    "life_threshold": 80,
    "soh_baseline_note": "SOH baseline = fixed nominal-capacity normalization with 2.0 Ah",
    "segmentation_note": "segmentation rule = XJTU-specific extraction is defined in the dedicated extraction cell below",
    "plot_group_note": "plot grouping = protocol-family styles are defined in the single settings cell",
    "plot_groups": XJTU_PROTOCOL_GROUPS.copy(),
    "group_voltage_limits": {
        "2C": (2.5, 4.2),
        "3C": (2.5, 4.2),
        "R2.5": (2.5, 4.2),
        "R3": (3.0, 4.2),
        "RW": (3.0, 4.2),
        "Sim_satellite": (3.0, 4.2),
    },
    "processing_config": {
        "smooth_polyorder": 3,
        "smooth_window_ratio": 10,
        "spike_abs": 0.20,
        "spike_k": 6.0,
        "current_threshold": -0.5,
        "voltage_round_decimals": 4,
        "q_interp_window_ratio": 0,
        "overlap_pad": 0.0,
        "q_zero_tol": 1e-5,
        "vmax_check_tol": 5e-3,
        "voltage_monotonic_tol": 5e-3,
        "capacity_monotonic_tol": 1e-6,
    },
}


PROCESSING_CONFIG = dict(DATASET_SETTINGS["processing_config"])


EXPORT_EXCLUDED_GROUPS = {"RW", "Sim_satellite"}

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


def load_xjtu_pickle_cycles(path):
    with open(path, "rb") as f:
        payload = pickle.load(f)
    cycle_rows = payload["cycle_data"] if isinstance(payload, dict) and "cycle_data" in payload else payload
    return trim_cycle_fields(cycle_rows)


def load_xjtu_pickle_cycles_by_name(name):
    cached = CELL_RAW_CACHE.get(name)
    if cached is not None:
        return cached
    cell_cycles = load_xjtu_pickle_cycles(RAWDATA_DIR / name)
    CELL_RAW_CACHE[name] = cell_cycles
    return cell_cycles


def dataset_nominal_capacity_ah(cell_name):
    return 2.0 if cell_name is not None else 2.0


def dataset_major_group(cell_name):
    return dataset_plot_group(cell_name)


def dataset_plot_group(cell_name):
    stem = str(cell_name)
    if stem.endswith(".pkl") or stem.endswith(".pickle"):
        stem = Path(stem).stem
    if stem.startswith("XJTU_"):
        stem = stem[len("XJTU_"):]
    return next((group_name for group_name in XJTU_PROTOCOL_GROUPS if stem.startswith(group_name + "_")), None)


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
    nominal_capacity = get_nominal_capacity_ah(Path(cell_name).stem)
    if np.isfinite(nominal_capacity) and nominal_capacity > 0.0:
        return float(nominal_capacity)
    finite_capacities = []
    for cycle_data in cell_cycles:
        try:
            discharge_capacity = np.asarray(cycle_data["discharge_capacity_in_Ah"], dtype=float)
            finite_capacities.append(float(np.nanmax(discharge_capacity)))
        except Exception:
            continue
    return float(np.nanmax(finite_capacities)) if finite_capacities else float("nan")


def resolve_group_voltage_window(group_name):
    return DATASET_SETTINGS["group_voltage_limits"].get(group_name, (Vmin, Vmax))


def resolve_discharge_current_threshold(cycle_data, group_name=None):
    current = np.asarray(cycle_data.get("current_in_A", []), dtype=float)
    negative_current = current[np.isfinite(current) & (current < 0.0)]
    if group_name is not None and group_name not in DATASET_SETTINGS["group_voltage_limits"]:
        group_name = None
    if negative_current.size == 0:
        return float(PROCESSING_CONFIG["current_threshold"])


    representative_current = float(np.nanmedian(negative_current))
    adaptive_threshold = 0.25 * representative_current
    adaptive_threshold = max(float(PROCESSING_CONFIG["current_threshold"]), adaptive_threshold)
    adaptive_threshold = min(-0.05, adaptive_threshold)
    return float(adaptive_threshold)


def extract_cycle_capacity_ah(cycle_data):
    discharge_capacity = np.asarray(cycle_data.get("discharge_capacity_in_Ah", []), dtype=float)
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


def trim_cycles_to_peak_capacity(cell_cycles):
    raw_capacities = collect_cycle_capacities(cell_cycles)
    finite_idx = np.flatnonzero(np.isfinite(raw_capacities))
    if finite_idx.size == 0:
        return list(cell_cycles), 0, raw_capacities

    peak_idx = int(finite_idx[np.nanargmax(raw_capacities[finite_idx])])
    trimmed_cycles = list(cell_cycles[peak_idx:])
    return trimmed_cycles, peak_idx, raw_capacities


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
    nominal_capacity = resolve_nominal_capacity_ah(cell_cycles)
    if np.isfinite(nominal_capacity) and nominal_capacity > 0.0:
        soh_series = capacities / float(nominal_capacity) * 100.0
    else:
        q0 = float(np.nanmax(capacities)) if np.any(np.isfinite(capacities)) else float("nan")
        soh_series = capacities / q0 * 100.0 if np.isfinite(q0) and q0 > 0.0 else np.full_like(capacities, np.nan)
    CELL_CAPACITY_SUMMARY_CACHE[cache_key] = soh_series.copy()
    return soh_series.copy()


def extract_life_cycle_from_soh(soh_series, threshold=LIFE_THRESHOLD, end_tol=0.1):
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
        raise RuntimeError("Run the XJTU-specific extraction cell before calling compute_difference_profile.")
    if len(cell_cycles) < max(cyc_ref, target_cycle):
        return None

    cell_name = cell_name_by_obj_id.get(id(cell_cycles))
    segment_pair = extract_segment_pair(cell_cycles[cyc_ref - 1], cell_cycles[target_cycle - 1], cell_name=cell_name)
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


def _find_longest_true_run(mask):
    mask = np.asarray(mask, dtype=bool)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return None
    splits = np.where(np.diff(idx) > 1)[0] + 1
    runs = np.split(idx, splits)
    run = max(runs, key=len)
    return int(run[0]), int(run[-1] + 1)


def _find_xjtu_anchor_index(voltage, discharge_capacity, before_index):
    if before_index <= 0:
        return None

    finite_mask = np.isfinite(voltage[:before_index]) & np.isfinite(discharge_capacity[:before_index])
    if np.sum(finite_mask) == 0:
        return None

    actual_vmax = float(np.nanmax(voltage[:before_index][finite_mask]))
    candidate_mask = (
        finite_mask
        & (np.abs(discharge_capacity[:before_index]) <= PROCESSING_CONFIG["q_zero_tol"])
        & (np.abs(voltage[:before_index] - actual_vmax) <= PROCESSING_CONFIG["vmax_check_tol"])
    )
    candidate_idx = np.flatnonzero(candidate_mask)
    if candidate_idx.size == 0:
        return None
    return int(candidate_idx[-1]), actual_vmax


def _make_branch_from_axis_value(axis_desc, value_asc, mode, raw_points=None, keep_indices=None, vmax=None, slice_range=None):
    axis_desc = np.asarray(axis_desc, dtype=float)
    value_asc = np.asarray(value_asc, dtype=float)
    finite_mask = np.isfinite(axis_desc) & np.isfinite(value_asc)
    axis_desc = axis_desc[finite_mask]
    value_asc = value_asc[finite_mask]
    if axis_desc.size < 5 or value_asc.size < 5:
        return None
    if not np.all(np.diff(axis_desc) <= PROCESSING_CONFIG["voltage_monotonic_tol"]):
        return None
    if np.any(np.diff(value_asc) < -PROCESSING_CONFIG["capacity_monotonic_tol"]):
        return None

    if vmax is None:
        vmax = float(axis_desc[0])
    if raw_points is None:
        raw_points = int(axis_desc.size)
    if keep_indices is None:
        keep_indices = np.arange(axis_desc.size, dtype=int)
    if slice_range is None:
        slice_range = (0, int(axis_desc.size))

    branch = {
        "axis": axis_desc[::-1].copy(),
        "value": value_asc[::-1].copy(),
        "plot_axis": axis_desc.copy(),
        "plot_value": value_asc.copy(),
        "slice": (int(slice_range[0]), int(slice_range[1])),
        "raw_points": int(raw_points),
        "start_index": int(keep_indices[0]) if len(keep_indices) else 0,
        "keep_indices": np.asarray(keep_indices, dtype=int),
        "vmax": float(vmax),
        "mode": mode,
    }
    checks = validate_extracted_branch(branch)
    if not all(checks.values()):
        return None
    return branch


def _clean_common_discharge_branch(cycle_data, group_name=None):
    voltage = np.asarray(cycle_data.get("voltage_in_V", []), dtype=float)
    discharge_capacity = np.asarray(cycle_data.get("discharge_capacity_in_Ah", []), dtype=float)
    current = np.asarray(cycle_data.get("current_in_A", np.full_like(voltage, np.nan)), dtype=float)

    if voltage.size == 0 or discharge_capacity.size == 0 or current.size == 0:
        return None
    if not np.any(np.isfinite(current)):
        return None

    finite_mask = np.isfinite(voltage) & np.isfinite(discharge_capacity) & np.isfinite(current)
    if np.sum(finite_mask) < 5:
        return None

    voltage_lo, voltage_hi = resolve_group_voltage_window(group_name)
    current_cutoff = resolve_discharge_current_threshold(cycle_data, group_name=group_name)

    discharge_mask = (
        finite_mask
        & (current < current_cutoff)
        & (discharge_capacity > PROCESSING_CONFIG["q_zero_tol"])
        & (voltage >= voltage_lo)
        & (voltage <= voltage_hi + PROCESSING_CONFIG["vmax_check_tol"])
    )
    run = _find_longest_true_run(discharge_mask)
    if run is None:
        return None
    run_start, run_end = run

    anchor = _find_xjtu_anchor_index(voltage, discharge_capacity, run_start)
    if anchor is None:
        return None
    anchor_idx, actual_vmax = anchor

    keep_idx = [anchor_idx]
    last_v = float(voltage[anchor_idx])
    last_q = float(discharge_capacity[anchor_idx])

    for idx in range(run_start, run_end):
        v = float(voltage[idx])
        q = float(discharge_capacity[idx])
        if not (np.isfinite(v) and np.isfinite(q)):
            continue
        if v < voltage_lo or v > voltage_hi + PROCESSING_CONFIG["vmax_check_tol"]:
            continue
        if q < last_q - PROCESSING_CONFIG["capacity_monotonic_tol"]:
            continue
        if v > last_v + PROCESSING_CONFIG["voltage_monotonic_tol"]:
            continue
        if abs(v - last_v) <= PROCESSING_CONFIG["voltage_monotonic_tol"] and abs(q - last_q) <= PROCESSING_CONFIG["capacity_monotonic_tol"]:
            continue
        keep_idx.append(idx)
        last_v = v
        last_q = q

    if len(keep_idx) < 5:
        return None

    keep_idx = np.asarray(keep_idx, dtype=int)
    plot_axis = voltage[keep_idx].copy()
    plot_value = discharge_capacity[keep_idx].copy() - float(discharge_capacity[anchor_idx])
    if plot_value.size < 2 or np.any(plot_value[1:] <= float(PROCESSING_CONFIG["q_zero_tol"])):
        return None

    grouped = (
        pd.DataFrame({
            "V": np.round(plot_axis, int(PROCESSING_CONFIG["voltage_round_decimals"])),
            "V_raw": plot_axis,
            "Q": plot_value,
        })
        .groupby("V", as_index=False)
        .agg(V_raw=("V_raw", "median"), Q=("Q", "median"))
    )
    if len(grouped) < 5:
        return None

    order = np.argsort(grouped["V_raw"].to_numpy(dtype=float))[::-1]
    plot_axis_grouped = grouped["V_raw"].to_numpy(dtype=float)[order]
    plot_value_grouped = grouped["Q"].to_numpy(dtype=float)[order]
    return _make_branch_from_axis_value(
        plot_axis_grouped,
        plot_value_grouped,
        mode="waveform",
        raw_points=int(len(keep_idx)),
        keep_indices=keep_idx,
        vmax=actual_vmax,
        slice_range=(int(keep_idx[0]), int(keep_idx[-1] + 1)),
    )


def _extract_protocol_branch(cycle_data, group_name):
    return _clean_common_discharge_branch(cycle_data, group_name=group_name)


def extract_segment_pair(ref_cycle_data, target_cycle_data, cell_name=None):
    ref_branch = _extract_protocol_branch(ref_cycle_data, match_plot_group(cell_name) if cell_name is not None else None)
    tgt_branch = _extract_protocol_branch(target_cycle_data, match_plot_group(cell_name) if cell_name is not None else None)
    if ref_branch is None or tgt_branch is None:
        return None

    ref_axis_interp, ref_value_interp = ref_branch["axis"], ref_branch["value"]
    tgt_axis_interp, tgt_value_interp = tgt_branch["axis"], tgt_branch["value"]
    v_lo = max(float(np.nanmin(ref_axis_interp)), float(np.nanmin(tgt_axis_interp))) + float(PROCESSING_CONFIG["overlap_pad"])
    v_hi = min(float(np.nanmax(ref_axis_interp)), float(np.nanmax(tgt_axis_interp))) - float(PROCESSING_CONFIG["overlap_pad"])
    if not np.isfinite(v_lo) or not np.isfinite(v_hi) or v_hi <= v_lo:
        return None

    ref_interp_mask = (ref_axis_interp >= v_lo) & (ref_axis_interp <= v_hi)
    tgt_interp_mask = (tgt_axis_interp >= v_lo) & (tgt_axis_interp <= v_hi)
    if np.sum(ref_interp_mask) < 5 or np.sum(tgt_interp_mask) < 5:
        return None

    return {
        "ref_axis": ref_branch["plot_axis"].copy(),
        "ref_value": ref_branch["plot_value"].copy(),
        "tgt_axis": tgt_branch["plot_axis"].copy(),
        "tgt_value": tgt_branch["plot_value"].copy(),
        "ref_interp_axis": ref_axis_interp[ref_interp_mask],
        "ref_interp_value": ref_value_interp[ref_interp_mask],
        "tgt_interp_axis": tgt_axis_interp[tgt_interp_mask],
        "tgt_interp_value": tgt_value_interp[tgt_interp_mask],
        "window": (v_lo, v_hi),
        "slice_info": {
            "ref": ref_branch["slice"],
            "tgt": tgt_branch["slice"],
            "ref_keep_indices": ref_branch["keep_indices"].copy(),
            "tgt_keep_indices": tgt_branch["keep_indices"].copy(),
            "ref_raw_points": ref_branch["raw_points"],
            "tgt_raw_points": tgt_branch["raw_points"],
            "ref_start_q": float(ref_branch["plot_value"][0]),
            "tgt_start_q": float(tgt_branch["plot_value"][0]),
            "ref_start_v": float(ref_branch["plot_axis"][0]),
            "tgt_start_v": float(tgt_branch["plot_axis"][0]),
            "ref_vmax": float(ref_branch["vmax"]),
            "tgt_vmax": float(tgt_branch["vmax"]),
            "ref_mode": ref_branch.get("mode", "unknown"),
            "tgt_mode": tgt_branch.get("mode", "unknown"),
            "ref_summary_meta": {},
            "tgt_summary_meta": {},
            "v_lo": v_lo,
            "v_hi": v_hi,
        },
    }

def run(rawdata_dir: str | Path | None = None, output_dir: str | Path | None = None, overwrite: bool = True, save_feature_csv: bool = True):
    return export_dataset(__import__(__name__, fromlist=["*"]), rawdata_dir=rawdata_dir, output_dir=output_dir, overwrite=overwrite, save_feature_csv=save_feature_csv)


def main() -> None:
    from common import parse_preprocess_args

    args = parse_preprocess_args(DATASET_SETTINGS["dataset_name"])
    run(rawdata_dir=args.rawdata_dir, output_dir=args.output_dir, overwrite=args.overwrite, save_feature_csv=args.save_feature_csv)


if __name__ == "__main__":
    main()

