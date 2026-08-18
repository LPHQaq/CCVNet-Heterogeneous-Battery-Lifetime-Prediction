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

CVDCYCLE_KEEP_KEYS = ("current_in_A", "voltage_in_V", "charge_capacity_in_Ah", "discharge_capacity_in_Ah", "time_in_s")


DATASET_SETTINGS = {
    "dataset_name": "HUST(LFP)",
    "cvd_curve_cycles": list(range(10, 101, 10)),
    "feature_cycles": [20, 50, 100],
    "reference_cycle": 10,
    "voltage_window": (2.0, 3.6),
    "n_voltage_grid": 200,
    "life_threshold": 80.0,
    "life_end_tolerance": 1.0,
    "soh_baseline_note": "SOH baseline = fixed nominal capacity of 1.1 Ah after HUST cycle reindexing from the peak of the smoothed plausible capacity trajectory.",
    "segmentation_note": "segmentation rule = split the longest positive-current charge run by significant current changes, keep the second substantial positive-current stage as the target branch, retain absolute Q from the charge-run start, and use the approximate 3.4-3.6 V subsection only as a soft voltage preference.",
    "plot_group_note": "plot grouping = no subgroup split; use overall Total views only.",
    "plot_groups": [],
    "processing_config": {
        "smooth_polyorder": 3,
        "smooth_window_ratio": 20,
        "spike_abs": 0.20,
        "spike_k": 6.0,
        "current_threshold": -0.1,
        "segment_voltage_window": (3.4, 3.6),
        "segment_voltage_soft_pad": 0.05,
        "charge_current_target": 1.0,
        "charge_current_tol": 0.15,
        "charge_step_change_threshold": 0.05,
        "voltage_round_decimals": 4,
        "q_interp_window_ratio": 35,
        "overlap_pad": 0.01,
        "q_zero_tol": 1e-5,
        "vmax_check_tol": 5e-3,
        "voltage_monotonic_tol": 5e-3,
        "capacity_monotonic_tol": 1e-6,
        "current_plateau_tol": 0.15,
        "edge_check_n": 7,
        "edge_check_band": 3,
        "max_capacity_factor": 1.2,
        "voltage_spike_abs": 0.01,
        "voltage_spike_k": 6.0,
        "hust_reindex_rule": "start each cell at the peak of the smoothed, physically plausible capacity trajectory and drop all earlier cycles",
    },
}


PROCESSING_CONFIG = dict(DATASET_SETTINGS["processing_config"])

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
            finite_capacities.append(extract_cycle_capacity_ah(cycle_data))
        except Exception:
            continue
    return float(np.nanmax(finite_capacities)) if finite_capacities else float("nan")


def _find_longest_true_run_local(mask):
    idx = np.flatnonzero(np.asarray(mask, dtype=bool))
    if idx.size == 0:
        return None
    split_points = np.where(np.diff(idx) > 1)[0] + 1
    runs = np.split(idx, split_points)
    longest = max(runs, key=len)
    return int(longest[0]), int(longest[-1] + 1)


def _find_first_true_run_local(mask):
    idx = np.flatnonzero(np.asarray(mask, dtype=bool))
    if idx.size == 0:
        return None
    split_points = np.where(np.diff(idx) > 1)[0] + 1
    runs = np.split(idx, split_points)
    first = runs[0]
    return int(first[0]), int(first[-1] + 1)


def _split_by_change_points(signal, change_threshold):
    signal = np.asarray(signal, dtype=float)
    if signal.size == 0:
        return []
    diff_signal = np.abs(np.diff(signal))
    change_points = np.flatnonzero(np.isfinite(diff_signal) & (diff_signal >= change_threshold)) + 1
    boundaries = np.concatenate(([0], change_points, [signal.size]))
    segments = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end > start:
            segments.append((int(start), int(end)))
    return segments


def integrate_current_capacity_ah(current, time_in_s, positive_only=False):
    current = np.asarray(current, dtype=float)
    time_in_s = np.asarray(time_in_s, dtype=float)
    capacity = np.full(current.shape, np.nan, dtype=float)
    valid = np.isfinite(current) & np.isfinite(time_in_s)
    if np.sum(valid) < 2:
        return capacity
    valid_idx = np.flatnonzero(valid)
    current_valid = current[valid_idx]
    if positive_only:
        current_valid = np.clip(current_valid, a_min=0.0, a_max=None)
    time_valid = time_in_s[valid_idx]
    dt_hours = np.maximum(np.diff(time_valid), 0.0) / 3600.0
    increments = np.zeros_like(current_valid, dtype=float)
    increments[1:] = 0.5 * (current_valid[1:] + current_valid[:-1]) * dt_hours
    capacity[valid_idx] = np.cumsum(increments)
    return capacity


def locate_hust_charge_window(cycle_data):
    voltage = np.asarray(cycle_data.get("voltage_in_V", []), dtype=float)
    current = np.asarray(cycle_data.get("current_in_A", np.full_like(voltage, np.nan)), dtype=float)
    time_in_s = np.asarray(cycle_data.get("time_in_s", np.full_like(voltage, np.nan)), dtype=float)
    capacity = integrate_current_capacity_ah(current, time_in_s, positive_only=True)

    valid_mask = np.isfinite(voltage) & np.isfinite(current) & np.isfinite(time_in_s) & np.isfinite(capacity)
    positive_run = _find_longest_true_run_local(valid_mask & (current > 0.0))
    if positive_run is None:
        return None
    run_start, run_end = positive_run

    voltage_run = voltage[run_start:run_end]
    current_run = current[run_start:run_end]
    capacity_run = capacity[run_start:run_end]
    if voltage_run.size < 5:
        return None

    raw_segments = _split_by_change_points(current_run, charge_step_change_threshold)
    segment_candidates = []
    for seg_start, seg_end in raw_segments:
        if seg_end - seg_start < 5:
            continue
        seg_current = current_run[seg_start:seg_end]
        seg_voltage = voltage_run[seg_start:seg_end]
        seg_capacity = capacity_run[seg_start:seg_end]
        valid_seg = np.isfinite(seg_current) & np.isfinite(seg_voltage) & np.isfinite(seg_capacity)
        if np.sum(valid_seg) < 5:
            continue
        median_current = float(np.nanmedian(seg_current[valid_seg]))
        if not np.isfinite(median_current) or median_current <= 0.0:
            continue
        segment_candidates.append({
            "start": int(seg_start),
            "end": int(seg_end),
            "median_current": median_current,
        })

    if not segment_candidates:
        return None

    if len(segment_candidates) >= 2:
        chosen_segment = segment_candidates[1]
    else:
        chosen_segment = segment_candidates[0]

    active_start_rel = int(chosen_segment["start"])
    active_end_rel = int(chosen_segment["end"])
    target_current = float(chosen_segment["median_current"])

    plateau_mask = np.isfinite(current_run) & np.isfinite(voltage_run) & np.isfinite(capacity_run)
    plateau_mask &= current_run > 0.0
    plateau_mask &= np.abs(current_run - target_current) <= current_plateau_tol
    plateau_mask[:active_start_rel] = False
    plateau_mask[active_end_rel:] = False
    plateau_run = _find_first_true_run_local(plateau_mask)
    if plateau_run is None:
        plateau_start_rel = active_start_rel
        plateau_end_rel = active_end_rel
    else:
        plateau_start_rel = int(plateau_run[0])
        plateau_end_rel = int(plateau_run[1])

    approx_voltage_mask = np.isfinite(voltage_run) & np.isfinite(capacity_run)
    approx_voltage_mask &= voltage_run >= (segment_voltage_window[0] - segment_voltage_soft_pad)
    approx_voltage_mask &= voltage_run <= (segment_voltage_window[1] + segment_voltage_soft_pad)
    approx_voltage_mask[:plateau_start_rel] = False
    approx_voltage_mask[plateau_end_rel:] = False
    voltage_run_window = _find_first_true_run_local(approx_voltage_mask)
    if voltage_run_window is None:
        segment_start_rel = plateau_start_rel
        segment_end_rel = plateau_end_rel
    else:
        segment_start_rel = int(voltage_run_window[0])
        segment_end_rel = int(voltage_run_window[1])

    segment_start = int(run_start + segment_start_rel)
    segment_end = int(run_start + segment_end_rel)
    if segment_end <= segment_start + 2:
        return None

    return {
        "segment_start": segment_start,
        "segment_end": segment_end,
        "run_start": int(run_start),
        "run_end": int(run_end),
        "change_start": int(run_start + active_start_rel),
        "voltage": voltage,
        "current": current,
        "time_in_s": time_in_s,
        "capacity": capacity,
        "target_current": target_current,
    }


def extract_cycle_capacity_ah(cycle_data):
    segment_info = locate_hust_charge_window(cycle_data)
    if segment_info is not None:
        charge_trace = segment_info["capacity"][segment_info["run_start"]:segment_info["segment_end"]]
        if charge_trace.size >= 2 and np.any(np.isfinite(charge_trace)):
            return float(np.nanmax(charge_trace) - np.nanmin(charge_trace))
    current = np.asarray(cycle_data.get("current_in_A", []), dtype=float)
    time_in_s = np.asarray(cycle_data.get("time_in_s", np.full_like(current, np.nan)), dtype=float)
    integrated_charge = integrate_current_capacity_ah(current, time_in_s, positive_only=True)
    valid = np.isfinite(integrated_charge)
    if np.any(valid):
        return float(np.nanmax(integrated_charge[valid]))
    fallback_charge = np.asarray(cycle_data.get("charge_capacity_in_Ah", []), dtype=float)
    return float(np.nanmax(fallback_charge)) if np.any(np.isfinite(fallback_charge)) else float("nan")


def collect_cycle_capacities(cell_cycles):
    raw_capacities = []
    for cycle_data in cell_cycles:
        try:
            raw_capacities.append(extract_cycle_capacity_ah(cycle_data))
        except Exception:
            raw_capacities.append(float("nan"))
    return np.asarray(raw_capacities, dtype=float)


def reindex_hust_cycles_from_max_capacity(cell_cycles):
    raw_capacities = collect_cycle_capacities(cell_cycles)
    capacities_smooth = advanced_smooth_capacity_curve(
        raw_capacities,
        smooth_method="savgol",
        smooth_polyorder=PROCESSING_CONFIG["smooth_polyorder"],
        smooth_window_ratio=PROCESSING_CONFIG["smooth_window_ratio"],
        spike_abs=PROCESSING_CONFIG["spike_abs"],
        spike_k=PROCESSING_CONFIG["spike_k"],
    )

    plausible_cap = resolve_nominal_capacity_ah(cell_cycles) * PROCESSING_CONFIG["max_capacity_factor"]
    valid_mask = np.isfinite(capacities_smooth) & (capacities_smooth > 0.0) & (capacities_smooth <= plausible_cap)
    if not np.any(valid_mask):
        valid_mask = np.isfinite(capacities_smooth) & (capacities_smooth > 0.0)
    if not np.any(valid_mask):
        return list(cell_cycles), None

    valid_idx = np.where(valid_mask)[0]
    start_idx = int(valid_idx[np.argmax(capacities_smooth[valid_idx])])
    trimmed_cycles = []
    for new_cycle_number, cycle_data in enumerate(cell_cycles[start_idx:], start=1):
        cycle_copy = dict(cycle_data)
        cycle_copy["cycle_number"] = new_cycle_number
        trimmed_cycles.append(cycle_copy)
    return trimmed_cycles, start_idx + 1


def load_analysis_cycles(cell_name):
    cached = ANALYSIS_CELL_CACHE.get(cell_name)
    if cached is not None:
        return cached
    raw_cycles = load_pickle_cycles_by_name(cell_name)
    reindexed_cycles, start_cycle = reindex_hust_cycles_from_max_capacity(raw_cycles)
    ANALYSIS_CELL_CACHE[cell_name] = {
        "cycles": reindexed_cycles,
        "start_cycle": start_cycle,
    }
    return ANALYSIS_CELL_CACHE[cell_name]


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
        if soh_array[idx] >= threshold and soh_array[next_idx] <= threshold:
            return float(next_idx + 1)
        if soh_array[idx] > threshold and soh_array[next_idx] <= threshold:
            return float(next_idx + 1)
        if soh_array[idx] >= threshold and soh_array[next_idx] < threshold:
            return float(idx + 1)

    last_idx = int(finite_idx[-1])
    last_soh = float(soh_array[last_idx])
    if threshold <= last_soh < threshold + end_tol:
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
            "start_vwindow_ok": False,
            "voltage_asc_ok": False,
            "capacity_asc_ok": False,
        }

    plot_axis = np.asarray(branch["plot_axis"], dtype=float)
    plot_value = np.asarray(branch["plot_value"], dtype=float)

    if plot_axis.size == 0 or plot_value.size == 0:
        return {
            "extracted": False,
            "start_q_ok": False,
            "start_vwindow_ok": False,
            "voltage_asc_ok": False,
            "capacity_asc_ok": False,
        }

    return {
        "extracted": True,
         "start_q_ok": bool(np.isfinite(plot_value[0])),
        "start_vwindow_ok": bool(np.isfinite(plot_axis[0]) and np.isfinite(plot_axis[-1]) and (float(plot_axis[0]) >= segment_voltage_window[0] - segment_voltage_soft_pad - PROCESSING_CONFIG["vmax_check_tol"]) and (float(plot_axis[-1]) <= segment_voltage_window[1] + segment_voltage_soft_pad + PROCESSING_CONFIG["vmax_check_tol"])),
        "voltage_asc_ok": bool(np.all(np.diff(plot_axis) >= -PROCESSING_CONFIG["voltage_monotonic_tol"])),
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
    common_voltage = np.linspace(v_lo, v_hi, int(nV))
    ref_interp_raw = interp1d(segment_pair["ref_interp_axis"], segment_pair["ref_interp_value"], kind="linear", bounds_error=False, fill_value=np.nan)(common_voltage)
    tgt_interp_raw = interp1d(segment_pair["tgt_interp_axis"], segment_pair["tgt_interp_value"], kind="linear", bounds_error=False, fill_value=np.nan)(common_voltage)
    q_interp_window_ratio = PROCESSING_CONFIG["q_interp_window_ratio"] or PROCESSING_CONFIG["smooth_window_ratio"]
    ref_interp = advanced_smooth_capacity_curve(ref_interp_raw, smooth_method="savgol", smooth_polyorder=PROCESSING_CONFIG["smooth_polyorder"], smooth_window_ratio=q_interp_window_ratio, spike_abs=PROCESSING_CONFIG["spike_abs"], spike_k=PROCESSING_CONFIG["spike_k"])
    tgt_interp = advanced_smooth_capacity_curve(tgt_interp_raw, smooth_method="savgol", smooth_polyorder=PROCESSING_CONFIG["smooth_polyorder"], smooth_window_ratio=q_interp_window_ratio, spike_abs=PROCESSING_CONFIG["spike_abs"], spike_k=PROCESSING_CONFIG["spike_k"])
    q_diff = tgt_interp - ref_interp
    negative_difference = np.where(np.isnan(q_diff) | (q_diff < 0.0), q_diff, 0.0).astype(float)

    return {
        **segment_pair,
        "common_voltage": common_voltage,
        "axis": common_voltage,
        "ref_interp_raw": ref_interp_raw,
        "tgt_interp_raw": tgt_interp_raw,
        "ref_interp": ref_interp,
        "tgt_interp": tgt_interp,
        "q_diff": q_diff,
        "difference": q_diff,
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
    idx = np.flatnonzero(np.asarray(mask, dtype=bool))
    if idx.size == 0:
        return None
    split_points = np.where(np.diff(idx) > 1)[0] + 1
    runs = np.split(idx, split_points)
    longest = max(runs, key=len)
    return int(longest[0]), int(longest[-1] + 1)


def _build_hust_charge_segment(cycle_data):
    segment_info = locate_hust_charge_window(cycle_data)
    if segment_info is None:
        return None
    voltage = segment_info["voltage"]
    capacity = segment_info["capacity"]
    segment_start = int(segment_info["segment_start"])
    segment_end = int(segment_info["segment_end"])

    voltage_raw = voltage[segment_start:segment_end]
    capacity_raw = capacity[segment_start:segment_end]
    raw_slice = (segment_start, segment_end)

    if voltage_raw.size < 5:
        return None
    capacity_anchor = float(capacity[int(segment_info["run_start"])]) if np.isfinite(capacity[int(segment_info["run_start"])]) else float("nan")
    if np.isfinite(capacity_anchor):
        capacity_raw = capacity_raw - capacity_anchor

    keep_mask = np.ones(voltage_raw.size, dtype=bool)
    left_band = min(edge_check_band, max(voltage_raw.size - 2, 0))
    for j in range(left_band):
        ref_hi = min(voltage_raw.size, j + edge_check_n)
        v_ref = voltage_raw[j + 1:ref_hi]
        if v_ref.size >= 3:
            med_ref = float(np.nanmedian(v_ref))
            mad_ref = float(np.nanmedian(np.abs(v_ref - med_ref)))
            thr_ref = max(v_spike_abs, v_spike_k * mad_ref)
            if np.isfinite(voltage_raw[j]) and np.isfinite(med_ref) and abs(voltage_raw[j] - med_ref) > thr_ref:
                keep_mask[j] = False

    right_band = min(edge_check_band, max(voltage_raw.size - 2, 0))
    for off in range(right_band):
        j = voltage_raw.size - 1 - off
        ref_lo = max(0, j - edge_check_n + 1)
        v_ref = voltage_raw[ref_lo:j]
        if v_ref.size >= 3:
            med_ref = float(np.nanmedian(v_ref))
            mad_ref = float(np.nanmedian(np.abs(v_ref - med_ref)))
            thr_ref = max(v_spike_abs, v_spike_k * mad_ref)
            if np.isfinite(voltage_raw[j]) and np.isfinite(med_ref) and abs(voltage_raw[j] - med_ref) > thr_ref:
                keep_mask[j] = False

    for j in range(left_band, voltage_raw.size - right_band):
        if j <= 0 or j >= voltage_raw.size - 1:
            continue
        v3 = voltage_raw[j - 1:j + 2]
        med3 = float(np.nanmedian(v3))
        mad3 = float(np.nanmedian(np.abs(v3 - med3)))
        thr3 = max(v_spike_abs, v_spike_k * mad3)
        if np.isfinite(voltage_raw[j]) and np.isfinite(med3) and abs(voltage_raw[j] - med3) > thr3:
            keep_mask[j] = False

    voltage_clean = voltage_raw[keep_mask]
    capacity_clean = capacity_raw[keep_mask]
    keep_indices = np.flatnonzero(keep_mask)
    if voltage_clean.size < 5:
        return None

    order = np.argsort(voltage_clean)
    voltage_clean = voltage_clean[order]
    capacity_clean = capacity_clean[order]

    grouped = (
        pd.DataFrame(
            {
                "V_key": np.round(voltage_clean, voltage_round_decimals),
                "V_raw": voltage_clean,
                "Q": capacity_clean,
            }
        )
        .groupby("V_key", as_index=False)
        .agg(V_raw=("V_raw", "median"), Q=("Q", "max"))
    )

    interp_axis = grouped["V_raw"].to_numpy(dtype=float)
    interp_value = grouped["Q"].to_numpy(dtype=float)
    if interp_axis.size < 5:
        return None
    interp_value = np.maximum.accumulate(interp_value)

    plot_axis = interp_axis.copy()
    plot_value = interp_value.copy()

    return {
        "axis": interp_axis,
        "value": interp_value,
        "plot_axis": plot_axis,
        "plot_value": plot_value,
        "slice": raw_slice,
        "raw_points": int(segment_end - segment_start),
        "clean_points": int(interp_axis.size),
        "keep_indices": keep_indices,
        "q_start": float(interp_value[0]) if interp_value.size and np.isfinite(interp_value[0]) else float("nan"),
        "vmin": float(np.nanmin(interp_axis)),
        "vmax": float(np.nanmax(interp_axis)),
        "current_median": float(segment_info["target_current"]),
        "change_start": int(segment_info["change_start"]),
    }


def clean_hust_charge(cycle_data):
    return _build_hust_charge_segment(cycle_data)


def extract_segment_pair(ref_cycle_data, target_cycle_data):
    ref_branch = clean_hust_charge(ref_cycle_data)
    tgt_branch = clean_hust_charge(target_cycle_data)
    if ref_branch is None or tgt_branch is None:
        return None

    v_lo_raw = max(Vmin, float(np.nanmin(ref_branch["axis"])), float(np.nanmin(tgt_branch["axis"])))
    v_hi_raw = min(Vmax, float(np.nanmax(ref_branch["axis"])), float(np.nanmax(tgt_branch["axis"])))
    v_lo = v_lo_raw + overlap_pad
    v_hi = v_hi_raw - overlap_pad
    if not np.isfinite(v_lo) or not np.isfinite(v_hi) or v_hi <= v_lo:
        v_lo = v_lo_raw
        v_hi = v_hi_raw
    if not np.isfinite(v_lo) or not np.isfinite(v_hi) or v_hi <= v_lo:
        return None

    ref_interp_mask = (ref_branch["axis"] >= v_lo) & (ref_branch["axis"] <= v_hi)
    tgt_interp_mask = (tgt_branch["axis"] >= v_lo) & (tgt_branch["axis"] <= v_hi)
    if np.sum(ref_interp_mask) < 5 or np.sum(tgt_interp_mask) < 5:
        return None

    return {
        "ref_axis": ref_branch["plot_axis"].copy(),
        "ref_value": ref_branch["plot_value"].copy(),
        "tgt_axis": tgt_branch["plot_axis"].copy(),
        "tgt_value": tgt_branch["plot_value"].copy(),
        "ref_interp_axis": ref_branch["axis"][ref_interp_mask],
        "ref_interp_value": ref_branch["value"][ref_interp_mask],
        "tgt_interp_axis": tgt_branch["axis"][tgt_interp_mask],
        "tgt_interp_value": tgt_branch["value"][tgt_interp_mask],
        "window": (v_lo, v_hi),
        "slice_info": {
            "ref_raw_points": ref_branch["raw_points"],
            "ref_clean_points": ref_branch["clean_points"],
            "ref_slice": ref_branch["slice"],
            "ref_current_median": ref_branch["current_median"],
            "tgt_raw_points": tgt_branch["raw_points"],
            "tgt_clean_points": tgt_branch["clean_points"],
            "tgt_slice": tgt_branch["slice"],
            "tgt_current_median": tgt_branch["current_median"],
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

