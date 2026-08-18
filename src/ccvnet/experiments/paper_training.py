from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ccvnet.cache import write_manifest, write_resolved_config
from ccvnet.model_registry import (
    ModelSpec,
    model_spec_to_dict,
    paper_experiment_model_specs,
    select_cvd_input,
    select_value_input,
    training_config_for_spec,
    with_cycle_count,
)
from ccvnet.pipeline import AlignedData, build_aligned_data, build_fine_group_shared_split
from ccvnet.training import fit_model_bundle, predict_model_bundle

RESULT_SETS = {"major", "ablation", "all"}
SPLIT_PROTOCOLS = ["total split", "per-dataset split", "fine-group split"]
TRAIN_FRACTIONS = [0.6, 0.4, 0.2]
SUBSET_SEEDS = [42, 52, 62, 72, 82]


def _validate_result_set(result_set: str) -> str:
    result_set = str(result_set).strip().lower()
    if result_set not in RESULT_SETS:
        raise ValueError(f"Unknown result_set={result_set!r}; expected one of {sorted(RESULT_SETS)}.")
    return result_set


def _results_root(config: dict[str, Any]) -> Path:
    return Path(config.get("paths", {}).get("results_dir", "results"))


def result_output_dir(
    config: dict[str, Any], result_set: str, experiment_name: str, suffix: str | None = None
) -> Path:
    parts = [_results_root(config), _validate_result_set(result_set), experiment_name]
    if suffix:
        parts.append(suffix)
    path = Path(*map(str, parts))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _write_outputs(
    output_dir: Path,
    frames: dict[str, pd.DataFrame],
    *,
    config: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, int]:
    row_counts = {}
    for name, frame in frames.items():
        frame = frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()
        _write_csv(output_dir / f"{name}.csv", frame)
        row_counts[name] = int(len(frame))
    manifest = dict(manifest)
    manifest["row_counts"] = row_counts
    write_manifest(output_dir / "cache_manifest.json", manifest)
    write_resolved_config(output_dir / "config_resolved.yaml", config)
    return row_counts


def _read_existing_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _load_existing_frames(output_dir: Path, frame_names: list[str]) -> dict[str, pd.DataFrame]:
    return {name: _read_existing_frame(output_dir / f"{name}.csv") for name in frame_names}


def _spec_mask(df: pd.DataFrame, spec: ModelSpec) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=bool)
    mask = pd.Series(True, index=df.index)
    if "model" in df.columns:
        mask &= df["model"].astype(str).eq(spec.name)
    elif "name" in df.columns:
        mask &= df["name"].astype(str).eq(spec.name)
    if "input_mode" in df.columns:
        mask &= df["input_mode"].astype(str).eq(spec.input_mode)
    if "plot_group" in df.columns:
        mask &= df["plot_group"].astype(str).eq(spec.plot_group)
    elif "result_set" in df.columns and spec.plot_group:
        mask &= df["result_set"].astype(str).eq(spec.plot_group)
    if "key" in df.columns:
        mask &= df["key"].astype(str).eq(spec.key)
    return mask


def _filter_spec(df: pd.DataFrame, spec: ModelSpec) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    mask = _spec_mask(df, spec)
    if mask.empty:
        return df.copy()
    return df.loc[mask].copy()


def _drop_spec(df: pd.DataFrame, spec: ModelSpec) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    mask = _spec_mask(df, spec)
    if mask.empty:
        return df.copy()
    return df.loc[~mask].copy()


def _spec_cached(existing_frames: dict[str, pd.DataFrame], spec: ModelSpec, required_frames: list[str]) -> bool:
    for frame_name in required_frames:
        frame = existing_frames.get(frame_name, pd.DataFrame())
        if _filter_spec(frame, spec).empty:
            return False
    return True


def _merge_spec_frames(
    existing_frames: dict[str, pd.DataFrame],
    new_frames: dict[str, pd.DataFrame],
    spec: ModelSpec,
    frame_names: list[str],
) -> dict[str, pd.DataFrame]:
    merged: dict[str, pd.DataFrame] = {}
    for frame_name in frame_names:
        base = _drop_spec(existing_frames.get(frame_name, pd.DataFrame()), spec)
        fresh = new_frames.get(frame_name, pd.DataFrame())
        if base.empty:
            merged[frame_name] = fresh.copy()
        elif fresh.empty:
            merged[frame_name] = base.copy()
        else:
            merged[frame_name] = pd.concat([base, fresh], ignore_index=True)
    return merged


def _finite_pairs(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[mask], y_pred[mask]


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    y_true, y_pred = _finite_pairs(y_true, y_pred)
    if len(y_true) == 0:
        return {"n": 0, "rmse": np.nan, "mae": np.nan, "mape": np.nan, "pearson_r": np.nan}
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    valid_mape = np.abs(y_true) > 1e-12
    mape = float(np.mean(np.abs(err[valid_mape] / y_true[valid_mape])) * 100.0) if valid_mape.any() else np.nan
    pearson = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) >= 2 and np.std(y_true) > 0 and np.std(y_pred) > 0 else np.nan
    return {"n": int(len(y_true)), "rmse": rmse, "mae": mae, "mape": mape, "pearson_r": pearson}


def summarize_metric_repeats(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    metric_cols = [col for col in ["n", "rmse", "mae", "mape", "pearson_r"] if col in df.columns]
    grouped = df.groupby(group_cols, dropna=False)[metric_cols].agg(["mean", "std"]).reset_index()
    grouped.columns = [
        col[0] if col[1] == "" else (col[0] if col[1] == "mean" else f"{col[0]}_std")
        for col in grouped.columns.to_flat_index()
    ]
    return grouped


def compute_delta_summary(summary_df: pd.DataFrame, key_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()
    max_cycle = int(pd.to_numeric(summary_df["prefix_end_cycle"], errors="coerce").max())
    base = summary_df.loc[pd.to_numeric(summary_df["prefix_end_cycle"], errors="coerce").eq(max_cycle)].copy()
    base = base.set_index(key_cols)
    rows = []
    for _, row in summary_df.iterrows():
        key = tuple(row[col] for col in key_cols)
        out = {col: row[col] for col in key_cols}
        out["prefix_end_cycle"] = int(row["prefix_end_cycle"])
        if key in base.index:
            base_row = base.loc[key]
            if isinstance(base_row, pd.DataFrame):
                base_row = base_row.iloc[0]
            for metric in metric_cols:
                out[f"delta_{metric}"] = float(row.get(metric, np.nan)) - float(base_row.get(metric, np.nan))
                std_col = f"{metric}_std"
                out[f"delta_{metric}_std"] = float(row.get(std_col, np.nan)) if std_col in row else np.nan
        rows.append(out)
    return pd.DataFrame(rows)


def _model_rank(specs: list[ModelSpec], spec: ModelSpec) -> int:
    names = [(item.name, item.input_mode, item.plot_group) for item in specs]
    try:
        return names.index((spec.name, spec.input_mode, spec.plot_group)) + 1
    except ValueError:
        return 0


def _split_specs(aligned: AlignedData, config: dict[str, Any]) -> list[dict[str, Any]]:
    split_df = build_fine_group_shared_split(aligned, config)
    dataset_col = str(config.get("split", {}).get("dataset_column", "dataset_name"))
    fine_col = str(config.get("split", {}).get("group_column", aligned.fine_group_col))
    return [
        {"split_protocol": "total split", "training_mode": "total split", "split_df": split_df, "train_group_col": None},
        {"split_protocol": "per-dataset split", "training_mode": "per-dataset split", "split_df": split_df, "train_group_col": dataset_col},
        {"split_protocol": "fine-group split", "training_mode": "fine-group split", "split_df": split_df, "train_group_col": fine_col},
    ]


def _input_matrices(aligned: AlignedData, spec: ModelSpec) -> tuple[np.ndarray, np.ndarray]:
    return (
        select_cvd_input(spec.input_mode, X_cvd_abs=aligned.X_cvd_abs, X_cvd_norm=aligned.X_cvd_norm),
        select_value_input(spec.input_mode, X_value_abs=aligned.X_value_abs, X_value_norm=aligned.X_value_norm),
    )


def _fit_predict(
    aligned: AlignedData,
    spec: ModelSpec,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    train_cfg: dict[str, Any],
    *,
    seed: int,
    X_cvd: np.ndarray | None = None,
    X_value: np.ndarray | None = None,
) -> np.ndarray:
    if X_cvd is None or X_value is None:
        X_cvd, X_value = _input_matrices(aligned, spec)
    cfg = dict(train_cfg)
    if spec.family == "ccvnet" or spec.key in {"descriptor_mlp", "descriptor_meta_mlp", "spectrum_descriptor", "spectrum_meta"}:
        cfg.setdefault("descriptor_raw_dim", aligned.descriptor_raw_dim)
        cfg.setdefault("metadata_raw_dim", aligned.metadata_raw_dim)
    cfg["seed"] = int(seed)
    bundle = fit_model_bundle(spec.key, X_cvd[train_mask], X_value[train_mask], aligned.y[train_mask], cfg)
    pred = predict_model_bundle(spec.key, bundle, X_cvd[test_mask], X_value[test_mask], {"device": cfg.get("device", "auto")})
    return np.asarray(pred, dtype=float)


def _progress_header(experiment_name: str, result_set: str, current: int, total: int, detail: str) -> None:
    print(f"[{experiment_name}][{result_set}] {current}/{total} {detail}", flush=True)


def _progress_detail(message: str) -> None:
    print(f"  -> {message}", flush=True)


def run_single_benchmark(
    aligned: AlignedData,
    specs: list[ModelSpec],
    spec: ModelSpec,
    split_protocol: str,
    split_df: pd.DataFrame,
    train_group_col: str | None,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overall_rows, dataset_rows, prediction_rows, skipped_rows = [], [], [], []
    X_cvd, X_value = _input_matrices(aligned, spec)
    train_cfg = training_config_for_spec(config.get("training", {}), spec)
    split_ids = sorted(pd.to_numeric(split_df["split_id"], errors="coerce").dropna().astype(int).unique())
    dataset_col = str(config.get("split", {}).get("dataset_column", "dataset_name"))

    for split_id in split_ids:
        _progress_detail(f"split_id={int(split_id)}")
        split_sub = split_df.loc[split_df["split_id"].eq(split_id)].sort_values("row_index")
        split_train_mask = split_sub["split"].eq("train").to_numpy()
        split_test_mask = split_sub["split"].eq("test").to_numpy()
        pred_all = np.full(len(aligned.pipeline_df), np.nan, dtype=float)
        rank = _model_rank(specs, spec)
        if train_group_col is None:
            if int(split_train_mask.sum()) < 2 or int(split_test_mask.sum()) == 0:
                skipped_rows.append({"split_id": int(split_id), "split_protocol": split_protocol, "input_mode": spec.input_mode, "model": spec.name, "plot_group": spec.plot_group, "reason": "too_few_train_or_test_samples", "n_train": int(split_train_mask.sum()), "n_test": int(split_test_mask.sum())})
                continue
            pred_all[split_test_mask] = _fit_predict(aligned, spec, split_train_mask, split_test_mask, train_cfg, seed=10000 + 100 * int(split_id) + rank, X_cvd=X_cvd, X_value=X_value)
        else:
            for group_value in sorted(aligned.pipeline_df[train_group_col].astype(str).dropna().unique()):
                group_mask = aligned.pipeline_df[train_group_col].astype(str).eq(str(group_value)).to_numpy()
                group_train_mask = group_mask & split_train_mask
                group_test_mask = group_mask & split_test_mask
                if int(group_train_mask.sum()) < 2 or int(group_test_mask.sum()) == 0:
                    skipped_rows.append({"split_id": int(split_id), "split_protocol": split_protocol, "input_mode": spec.input_mode, "model": spec.name, "plot_group": spec.plot_group, train_group_col: group_value, "reason": "too_few_train_or_test_samples", "n_train": int(group_train_mask.sum()), "n_test": int(group_test_mask.sum())})
                    continue
                pred_all[group_test_mask] = _fit_predict(aligned, spec, group_train_mask, group_test_mask, train_cfg, seed=20000 + 1000 * int(split_id) + 100 * rank + len(skipped_rows), X_cvd=X_cvd, X_value=X_value)
        valid_test_mask = split_test_mask & np.isfinite(pred_all)
        if int(valid_test_mask.sum()) == 0:
            continue
        pred_df = aligned.pipeline_df.loc[valid_test_mask, ["row_index", "cell", dataset_col, "target_life"]].copy()
        pred_df = pred_df.rename(columns={dataset_col: "dataset_name"})
        pred_df["split_id"] = int(split_id)
        pred_df["split_protocol"] = split_protocol
        pred_df["input_mode"] = spec.input_mode
        pred_df["model"] = spec.name
        pred_df["plot_group"] = spec.plot_group
        pred_df["prediction"] = pred_all[valid_test_mask]
        pred_df["abs_error"] = np.abs(pred_df["prediction"] - pred_df["target_life"])
        prediction_rows.append(pred_df)
        overall = regression_metrics(pred_df["target_life"].to_numpy(), pred_df["prediction"].to_numpy())
        overall.update({"split_id": int(split_id), "split_protocol": split_protocol, "input_mode": spec.input_mode, "model": spec.name, "plot_group": spec.plot_group})
        overall_rows.append(overall)
        for dataset_name, sub in pred_df.groupby("dataset_name", dropna=False):
            metrics = regression_metrics(sub["target_life"].to_numpy(), sub["prediction"].to_numpy())
            metrics.update({"split_id": int(split_id), "split_protocol": split_protocol, "input_mode": spec.input_mode, "model": spec.name, "plot_group": spec.plot_group, "dataset_name": str(dataset_name), "weighted_mape_sum": float(metrics["mape"] * len(sub)) if pd.notna(metrics["mape"]) else np.nan})
            dataset_rows.append(metrics)
    return pd.DataFrame(overall_rows), pd.DataFrame(dataset_rows), pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame(), pd.DataFrame(skipped_rows)


def _sample_train_mask(base_train_mask: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    base_indices = np.flatnonzero(base_train_mask)
    if len(base_indices) <= 1:
        return base_train_mask.copy()
    rng = np.random.default_rng(seed)
    n_keep = max(1, int(round(len(base_indices) * float(fraction))))
    keep = rng.permutation(base_indices)[: min(n_keep, len(base_indices))]
    mask = np.zeros_like(base_train_mask, dtype=bool)
    mask[keep] = True
    return mask


def _sample_groupwise_train_mask(df: pd.DataFrame, group_col: str, base_train_mask: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sampled = np.zeros_like(base_train_mask, dtype=bool)
    for group_value in sorted(df[group_col].astype(str).dropna().unique()):
        group_mask = df[group_col].astype(str).eq(str(group_value)).to_numpy()
        group_train_idx = np.flatnonzero(group_mask & base_train_mask)
        if len(group_train_idx) == 0:
            continue
        n_keep = max(1, int(round(len(group_train_idx) * float(fraction))))
        sampled[rng.permutation(group_train_idx)[: min(n_keep, len(group_train_idx))]] = True
    return sampled


def run_single_small_data(
    aligned: AlignedData,
    specs: list[ModelSpec],
    spec: ModelSpec,
    split_protocol: str,
    training_mode: str,
    split_df: pd.DataFrame,
    train_group_col: str | None,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overall_rows, dataset_rows, prediction_rows = [], [], []
    X_cvd, X_value = _input_matrices(aligned, spec)
    train_cfg = training_config_for_spec(config.get("training", {}), spec)
    split_ids = sorted(pd.to_numeric(split_df["split_id"], errors="coerce").dropna().astype(int).unique())
    dataset_col = str(config.get("split", {}).get("dataset_column", "dataset_name"))
    rank = _model_rank(specs, spec)
    for split_id in split_ids:
        _progress_detail(f"split_id={int(split_id)}")
        split_sub = split_df.loc[split_df["split_id"].eq(split_id)].sort_values("row_index")
        base_train_mask = split_sub["split"].eq("train").to_numpy()
        base_test_mask = split_sub["split"].eq("test").to_numpy()
        if int(base_test_mask.sum()) == 0:
            continue
        for subset_seed in SUBSET_SEEDS:
            for fraction in TRAIN_FRACTIONS:
                pred_all = np.full(len(aligned.pipeline_df), np.nan, dtype=float)
                if train_group_col is None:
                    train_mask = _sample_train_mask(base_train_mask, fraction, subset_seed)
                    if int(train_mask.sum()) < 2:
                        continue
                    seed = 30000 + 1000 * int(split_id) + 100 * rank + int(subset_seed) + int(round(fraction * 100))
                    pred_all[base_test_mask] = _fit_predict(aligned, spec, train_mask, base_test_mask, train_cfg, seed=seed, X_cvd=X_cvd, X_value=X_value)
                else:
                    train_mask = _sample_groupwise_train_mask(aligned.pipeline_df, train_group_col, base_train_mask, fraction, subset_seed)
                    for group_value in sorted(aligned.pipeline_df[train_group_col].astype(str).dropna().unique()):
                        group_mask = aligned.pipeline_df[train_group_col].astype(str).eq(str(group_value)).to_numpy()
                        group_train_mask = group_mask & train_mask
                        group_test_mask = group_mask & base_test_mask
                        if int(group_train_mask.sum()) < 2 or int(group_test_mask.sum()) == 0:
                            continue
                        seed = 40000 + 1000 * int(split_id) + 100 * rank + int(subset_seed) + int(round(fraction * 100)) + len(group_value)
                        pred_all[group_test_mask] = _fit_predict(aligned, spec, group_train_mask, group_test_mask, train_cfg, seed=seed, X_cvd=X_cvd, X_value=X_value)
                valid_test_mask = base_test_mask & np.isfinite(pred_all)
                if int(valid_test_mask.sum()) == 0:
                    continue
                pred_df = aligned.pipeline_df.loc[valid_test_mask, ["row_index", "cell", dataset_col, "target_life"]].copy().rename(columns={dataset_col: "dataset_name"})
                pred_df["split_id"] = int(split_id)
                pred_df["subset_seed"] = int(subset_seed)
                pred_df["train_fraction_total"] = float(fraction)
                pred_df["split_protocol"] = split_protocol
                pred_df["training_mode"] = training_mode
                pred_df["input_mode"] = spec.input_mode
                pred_df["model"] = spec.name
                pred_df["plot_group"] = spec.plot_group
                pred_df["prediction"] = pred_all[valid_test_mask]
                pred_df["abs_error"] = np.abs(pred_df["prediction"] - pred_df["target_life"])
                prediction_rows.append(pred_df)
                overall = regression_metrics(pred_df["target_life"].to_numpy(), pred_df["prediction"].to_numpy())
                overall.update({"split_id": int(split_id), "subset_seed": int(subset_seed), "train_fraction_total": float(fraction), "split_protocol": split_protocol, "training_mode": training_mode, "input_mode": spec.input_mode, "model": spec.name, "plot_group": spec.plot_group})
                overall_rows.append(overall)
                for dataset_name, sub in pred_df.groupby("dataset_name", dropna=False):
                    metrics = regression_metrics(sub["target_life"].to_numpy(), sub["prediction"].to_numpy())
                    metrics.update({"split_id": int(split_id), "subset_seed": int(subset_seed), "train_fraction_total": float(fraction), "split_protocol": split_protocol, "training_mode": training_mode, "input_mode": spec.input_mode, "model": spec.name, "plot_group": spec.plot_group, "dataset_name": str(dataset_name)})
                    dataset_rows.append(metrics)
    return pd.DataFrame(overall_rows), pd.DataFrame(dataset_rows), pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()


def _prefix_cvd_tensor(X: np.ndarray, prefix_end_cycle: int, cycles: list[int]) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    n_cycles = len(cycles)
    n_keep = cycles.index(int(prefix_end_cycle)) + 1
    if X.shape[1] >= 4 * n_cycles:
        pieces = [X[:, :n_keep, :], X[:, n_cycles : n_cycles + n_keep, :], X[:, 2 * n_cycles : 2 * n_cycles + n_keep, :], X[:, 3 * n_cycles : 3 * n_cycles + n_keep, :]]
        return np.concatenate(pieces, axis=1)
    if X.shape[1] >= 2 * n_cycles:
        return np.concatenate([X[:, :n_keep, :], X[:, n_cycles : n_cycles + n_keep, :]], axis=1)
    return X[:, :n_keep, :]


def _descriptor_cycle_map(columns: list[str]) -> dict[int, list[int]]:
    import re
    out: dict[int, list[int]] = {}
    for idx, col in enumerate(columns):
        match = re.search(r"(\d+)$", str(col))
        if match:
            out.setdefault(int(match.group(1)), []).append(idx)
    return out


def _prefix_value_matrix(aligned: AlignedData, X_value: np.ndarray, prefix_end_cycle: int, *, dense: bool = False) -> np.ndarray:
    X_value = np.asarray(X_value, dtype=np.float32).copy()
    desc_dim = aligned.descriptor_raw_dim
    cycle_map = _descriptor_cycle_map(aligned.descriptor_abs_cols)
    desc = X_value[:, :desc_dim]
    meta = X_value[:, desc_dim:]
    masked = np.full_like(desc, np.nan, dtype=np.float32)
    all_cycles = sorted(cycle for cycle in cycle_map if cycle <= int(prefix_end_cycle))
    chosen = all_cycles if dense else [cycle for cycle in [20, 50, 100] if cycle <= int(prefix_end_cycle)]
    selected = []
    for cycle in chosen:
        selected.extend(cycle_map.get(cycle, []))
    if selected:
        masked[:, selected] = desc[:, selected]
    return np.concatenate([masked, meta], axis=1).astype(np.float32)


def run_single_cycle(
    aligned: AlignedData,
    specs: list[ModelSpec],
    spec: ModelSpec,
    split_protocol: str,
    split_df: pd.DataFrame,
    train_group_col: str | None,
    config: dict[str, Any],
    prefix_end_cycle: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cycles = [int(c) for c in config.get("data", {}).get("cvd_cycles", list(range(20, 101, 10)))]
    spec = with_cycle_count(spec, cycles.index(int(prefix_end_cycle)) + 1)
    X_cvd_base, X_value_base = _input_matrices(aligned, spec)
    X_cvd = _prefix_cvd_tensor(X_cvd_base, prefix_end_cycle, cycles)
    X_value = _prefix_value_matrix(aligned, X_value_base, prefix_end_cycle, dense=False)
    overall_rows, dataset_rows, prediction_rows = [], [], []
    train_cfg = training_config_for_spec(config.get("training", {}), spec)
    split_ids = sorted(pd.to_numeric(split_df["split_id"], errors="coerce").dropna().astype(int).unique())
    dataset_col = str(config.get("split", {}).get("dataset_column", "dataset_name"))
    rank = _model_rank(specs, spec)
    for split_id in split_ids:
        _progress_detail(f"split_id={int(split_id)}")
        split_sub = split_df.loc[split_df["split_id"].eq(split_id)].sort_values("row_index")
        split_train_mask = split_sub["split"].eq("train").to_numpy()
        split_test_mask = split_sub["split"].eq("test").to_numpy()
        pred_all = np.full(len(aligned.pipeline_df), np.nan, dtype=float)
        if train_group_col is None:
            if int(split_train_mask.sum()) < 2 or int(split_test_mask.sum()) == 0:
                continue
            pred_all[split_test_mask] = _fit_predict(aligned, spec, split_train_mask, split_test_mask, train_cfg, seed=50000 + 1000 * int(prefix_end_cycle) + 100 * int(split_id) + rank, X_cvd=X_cvd, X_value=X_value)
        else:
            for group_value in sorted(aligned.pipeline_df[train_group_col].astype(str).dropna().unique()):
                group_mask = aligned.pipeline_df[train_group_col].astype(str).eq(str(group_value)).to_numpy()
                group_train_mask = group_mask & split_train_mask
                group_test_mask = group_mask & split_test_mask
                if int(group_train_mask.sum()) < 2 or int(group_test_mask.sum()) == 0:
                    continue
                pred_all[group_test_mask] = _fit_predict(aligned, spec, group_train_mask, group_test_mask, train_cfg, seed=60000 + 1000 * int(prefix_end_cycle) + 100 * int(split_id) + rank + len(group_value), X_cvd=X_cvd, X_value=X_value)
        valid_test_mask = split_test_mask & np.isfinite(pred_all)
        if int(valid_test_mask.sum()) == 0:
            continue
        pred_df = aligned.pipeline_df.loc[valid_test_mask, ["row_index", "cell", dataset_col, "target_life"]].copy().rename(columns={dataset_col: "dataset_name"})
        pred_df["split_id"] = int(split_id)
        pred_df["split_protocol"] = split_protocol
        pred_df["input_mode"] = spec.input_mode
        pred_df["model"] = spec.name
        pred_df["plot_group"] = spec.plot_group
        pred_df["prefix_end_cycle"] = int(prefix_end_cycle)
        pred_df["prediction"] = pred_all[valid_test_mask]
        prediction_rows.append(pred_df)
        metrics = regression_metrics(pred_df["target_life"].to_numpy(), pred_df["prediction"].to_numpy())
        metrics.update({"split_id": int(split_id), "split_protocol": split_protocol, "input_mode": spec.input_mode, "model": spec.name, "plot_group": spec.plot_group, "prefix_end_cycle": int(prefix_end_cycle)})
        overall_rows.append(metrics)
        for dataset_name, sub in pred_df.groupby("dataset_name", dropna=False):
            dmetrics = regression_metrics(sub["target_life"].to_numpy(), sub["prediction"].to_numpy())
            dmetrics.update({"split_id": int(split_id), "split_protocol": split_protocol, "input_mode": spec.input_mode, "model": spec.name, "plot_group": spec.plot_group, "dataset_name": str(dataset_name), "prefix_end_cycle": int(prefix_end_cycle)})
            dataset_rows.append(dmetrics)
    return pd.DataFrame(overall_rows), pd.DataFrame(dataset_rows), pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()


def run_single_descriptor_attribution(
    aligned: AlignedData,
    spec: ModelSpec,
    split_protocol: str,
    split_df: pd.DataFrame,
    train_group_col: str | None,
    config: dict[str, Any],
    prefix_end_cycle: int,
    *,
    dense: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cycles = [int(c) for c in config.get("data", {}).get("cvd_cycles", list(range(20, 101, 10)))]
    spec = with_cycle_count(spec, cycles.index(int(prefix_end_cycle)) + 1)
    X_cvd_base, X_value_base = _input_matrices(aligned, spec)
    X_cvd = _prefix_cvd_tensor(X_cvd_base, prefix_end_cycle, cycles)
    X_value = _prefix_value_matrix(aligned, X_value_base, prefix_end_cycle, dense=dense)
    method = f"{spec.name} {'dense-desc' if dense else 'sparse-desc'}"
    overall_rows, dataset_rows = [], []
    train_cfg = training_config_for_spec(config.get("training", {}), spec)
    split_ids = sorted(pd.to_numeric(split_df["split_id"], errors="coerce").dropna().astype(int).unique())
    dataset_col = str(config.get("split", {}).get("dataset_column", "dataset_name"))

    for split_id in split_ids:
        _progress_detail(f"split_id={int(split_id)}")
        split_sub = split_df.loc[split_df["split_id"].eq(split_id)].sort_values("row_index")
        split_train_mask = split_sub["split"].eq("train").to_numpy()
        split_test_mask = split_sub["split"].eq("test").to_numpy()
        pred_all = np.full(len(aligned.pipeline_df), np.nan, dtype=float)
        dense_offset = 17 if dense else 0
        if train_group_col is None:
            if int(split_train_mask.sum()) < 2 or int(split_test_mask.sum()) == 0:
                continue
            pred_all[split_test_mask] = _fit_predict(
                aligned,
                spec,
                split_train_mask,
                split_test_mask,
                train_cfg,
                seed=70000 + 1000 * int(prefix_end_cycle) + 100 * int(split_id) + dense_offset,
                X_cvd=X_cvd,
                X_value=X_value,
            )
        else:
            for group_value in sorted(aligned.pipeline_df[train_group_col].astype(str).dropna().unique()):
                group_mask = aligned.pipeline_df[train_group_col].astype(str).eq(str(group_value)).to_numpy()
                group_train_mask = group_mask & split_train_mask
                group_test_mask = group_mask & split_test_mask
                if int(group_train_mask.sum()) < 2 or int(group_test_mask.sum()) == 0:
                    continue
                pred_all[group_test_mask] = _fit_predict(
                    aligned,
                    spec,
                    group_train_mask,
                    group_test_mask,
                    train_cfg,
                    seed=80000 + 1000 * int(prefix_end_cycle) + 100 * int(split_id) + len(group_value) + dense_offset,
                    X_cvd=X_cvd,
                    X_value=X_value,
                )
        valid_test_mask = split_test_mask & np.isfinite(pred_all)
        if int(valid_test_mask.sum()) == 0:
            continue
        pred_df = aligned.pipeline_df.loc[valid_test_mask, ["row_index", "cell", dataset_col, "target_life"]].copy().rename(columns={dataset_col: "dataset_name"})
        pred_df["prediction"] = pred_all[valid_test_mask]
        metrics = regression_metrics(pred_df["target_life"].to_numpy(), pred_df["prediction"].to_numpy())
        metrics.update({"split_id": int(split_id), "split_protocol": split_protocol, "method": method, "prefix_end_cycle": int(prefix_end_cycle)})
        overall_rows.append(metrics)
        for dataset_name, sub in pred_df.groupby("dataset_name", dropna=False):
            dmetrics = regression_metrics(sub["target_life"].to_numpy(), sub["prediction"].to_numpy())
            dmetrics.update({"split_id": int(split_id), "split_protocol": split_protocol, "method": method, "dataset_name": str(dataset_name), "prefix_end_cycle": int(prefix_end_cycle)})
            dataset_rows.append(dmetrics)
    return pd.DataFrame(overall_rows), pd.DataFrame(dataset_rows)


def compute_descriptor_gap_summary(overall_summary_df: pd.DataFrame) -> pd.DataFrame:
    if overall_summary_df.empty:
        return pd.DataFrame()
    dense_name = "CCV-basic dense-desc"
    sparse_name = "CCV-basic sparse-desc"
    rows = []
    key_cols = ["split_protocol", "prefix_end_cycle"]
    for key, sub in overall_summary_df.groupby(key_cols, dropna=False):
        lookup = sub.set_index("method")
        if dense_name not in lookup.index or sparse_name not in lookup.index:
            continue
        dense_row = lookup.loc[dense_name]
        sparse_row = lookup.loc[sparse_name]
        if isinstance(dense_row, pd.DataFrame):
            dense_row = dense_row.iloc[0]
        if isinstance(sparse_row, pd.DataFrame):
            sparse_row = sparse_row.iloc[0]
        split_protocol, prefix_end_cycle = key
        rows.append(
            {
                "split_protocol": split_protocol,
                "prefix_end_cycle": int(prefix_end_cycle),
                "descriptor_gap_mape": float(dense_row.get("mape", np.nan)) - float(sparse_row.get("mape", np.nan)),
                "descriptor_gap_mape_std": math.sqrt(float(dense_row.get("mape_std", 0.0) or 0.0) ** 2 + float(sparse_row.get("mape_std", 0.0) or 0.0) ** 2),
                "descriptor_gap_rmse": float(dense_row.get("rmse", np.nan)) - float(sparse_row.get("rmse", np.nan)),
                "descriptor_gap_rmse_std": math.sqrt(float(dense_row.get("rmse_std", 0.0) or 0.0) ** 2 + float(sparse_row.get("rmse_std", 0.0) or 0.0) ** 2),
            }
        )
    return pd.DataFrame(rows)




def _run_moe_like_benchmark(
    config: dict[str, Any],
    result_set: str,
    *,
    experiment_name_default: str,
    runner_name: str,
    small_data: bool = False,
) -> dict[str, Any]:
    result_set = _validate_result_set(result_set)
    aligned = build_aligned_data(config)
    experiment_name = str(config.get("experiment", {}).get("name", experiment_name_default))
    specs = paper_experiment_model_specs(config, experiment_name=experiment_name, result_set=result_set)
    split_specs = _split_specs(aligned, config)
    output_dir = result_output_dir(config, result_set, experiment_name)
    frame_names = ["overall_metric_df", "dataset_metric_df", "prediction_df"] + ([] if small_data else ["skipped_df"])
    existing_frames = _load_existing_frames(output_dir, frame_names)
    required_cached_frames = ["overall_metric_df", "dataset_metric_df", "prediction_df"]
    runnable_specs = [spec for spec in specs if not _spec_cached(existing_frames, spec, required_cached_frames)]
    for spec in specs:
        if spec not in runnable_specs:
            print(f"[{experiment_name}][{result_set}] skip cached model={spec.name} input={spec.input_mode}", flush=True)
    total_tasks = len(runnable_specs) * len(split_specs)
    task_index = 0
    if total_tasks == 0:
        print(f"[{experiment_name}][{result_set}] nothing to run, all model caches already present.", flush=True)
    for spec in runnable_specs:
        overall_frames, dataset_frames, prediction_frames, skipped_frames = [], [], [], []
        for split_spec in split_specs:
            task_index += 1
            _progress_header(runner_name, result_set, task_index, total_tasks, f"model={spec.name} input={spec.input_mode} split={split_spec['split_protocol']}")
            if small_data:
                training_mode = f"{split_spec['split_protocol']} small data"
                overall, dataset, pred = run_single_small_data(
                    aligned, specs, spec, split_spec["split_protocol"], training_mode, split_spec["split_df"], split_spec["train_group_col"], config
                )
                overall_frames.append(overall)
                dataset_frames.append(dataset)
                prediction_frames.append(pred)
            else:
                overall, dataset, pred, skipped = run_single_benchmark(
                    aligned, specs, spec, split_spec["split_protocol"], split_spec["split_df"], split_spec["train_group_col"], config
                )
                overall_frames.append(overall)
                dataset_frames.append(dataset)
                prediction_frames.append(pred)
                skipped_frames.append(skipped)
        spec_frames = {
            "overall_metric_df": pd.concat(overall_frames, ignore_index=True) if overall_frames else pd.DataFrame(),
            "dataset_metric_df": pd.concat(dataset_frames, ignore_index=True) if dataset_frames else pd.DataFrame(),
            "prediction_df": pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame(),
        }
        if not small_data:
            spec_frames["skipped_df"] = pd.concat(skipped_frames, ignore_index=True) if skipped_frames else pd.DataFrame()
        existing_frames = _merge_spec_frames(existing_frames, spec_frames, spec, frame_names)
    frames = {
        "cache_status": cache_status_frame(specs, result_set).assign(cache_ready=True, status="trained"),
        "overall_metric_df": existing_frames["overall_metric_df"],
        "dataset_metric_df": existing_frames["dataset_metric_df"],
        "prediction_df": existing_frames["prediction_df"],
        "overall_summary_df": summarize_metric_repeats(
            existing_frames["overall_metric_df"],
            ["split_protocol", "input_mode", "model", "plot_group"] if not small_data else ["split_protocol", "training_mode", "train_fraction_total", "input_mode", "model", "plot_group"],
        ),
        "dataset_summary_df": summarize_metric_repeats(
            existing_frames["dataset_metric_df"],
            ["split_protocol", "dataset_name", "input_mode", "model", "plot_group"] if not small_data else ["split_protocol", "training_mode", "train_fraction_total", "dataset_name", "input_mode", "model", "plot_group"],
        ),
    }
    if not small_data:
        frames["skipped_df"] = existing_frames["skipped_df"]
    manifest = {"experiment": experiment_name, "result_set": result_set, "models": [model_spec_to_dict(spec) for spec in specs], "status": "trained"}
    _write_outputs(output_dir, frames, config=config, manifest=manifest)
    return {"status": "trained", "output_dir": output_dir}


def run_moe_baseline_benchmark(config: dict[str, Any], result_set: str = "ablation") -> dict[str, Any]:
    return _run_moe_like_benchmark(
        config,
        result_set,
        experiment_name_default="moe_baseline",
        runner_name="moe_baseline",
        small_data=False,
    )


def run_moe_small_data_benchmark(config: dict[str, Any], result_set: str = "ablation") -> dict[str, Any]:
    return _run_moe_like_benchmark(
        config,
        result_set,
        experiment_name_default="moe_small_data",
        runner_name="moe_small_data",
        small_data=True,
    )

def run_moe_early_cycle(config: dict[str, Any], result_set: str = "ablation") -> dict[str, Any]:
    result_set = _validate_result_set(result_set)
    aligned = build_aligned_data(config)
    experiment_name = str(config.get("experiment", {}).get("name", "moe_early_cycle"))
    branch_name = str(config.get("experiment", {}).get("branch", "cycle_effect"))
    specs = paper_experiment_model_specs(config, experiment_name=experiment_name, result_set=result_set)
    split_specs = _split_specs(aligned, config)
    output_dir = result_output_dir(config, result_set, experiment_name, branch_name)
    cycles = [int(c) for c in config.get("data", {}).get("cvd_cycles", list(range(20, 101, 10)))]
    frame_names = ["overall_summary_df", "dataset_summary_df", "overall_delta_summary_df", "dataset_delta_summary_df"]
    existing_frames = _load_existing_frames(output_dir, frame_names)
    required_cached_frames = ["overall_summary_df", "dataset_summary_df"]
    runnable_specs = [spec for spec in specs if not _spec_cached(existing_frames, spec, required_cached_frames)]
    for spec in specs:
        if spec not in runnable_specs:
            print(f"[{experiment_name}][{result_set}] skip cached model={spec.name} input={spec.input_mode}", flush=True)
    total_tasks = len(runnable_specs) * len(cycles) * len(split_specs)
    task_index = 0
    if total_tasks == 0:
        print(f"[{experiment_name}][{result_set}] nothing to run, all model caches already present.", flush=True)
    preserved_overall_summary = existing_frames["overall_summary_df"]
    preserved_dataset_summary = existing_frames["dataset_summary_df"]
    new_overall_summaries = []
    new_dataset_summaries = []
    for spec in runnable_specs:
        overall_metric_frames, dataset_metric_frames = [], []
        for prefix_end_cycle in cycles:
            for split_spec in split_specs:
                task_index += 1
                _progress_header(
                    experiment_name,
                    result_set,
                    task_index,
                    total_tasks,
                    f"model={spec.name} input={spec.input_mode} split={split_spec['split_protocol']} cycle={prefix_end_cycle}",
                )
                overall, dataset, _pred = run_single_cycle(
                    aligned,
                    specs,
                    spec,
                    split_spec["split_protocol"],
                    split_spec["split_df"],
                    split_spec["train_group_col"],
                    config,
                    prefix_end_cycle,
                )
                overall_metric_frames.append(overall)
                dataset_metric_frames.append(dataset)
        overall_metric_df = pd.concat(overall_metric_frames, ignore_index=True) if overall_metric_frames else pd.DataFrame()
        dataset_metric_df = pd.concat(dataset_metric_frames, ignore_index=True) if dataset_metric_frames else pd.DataFrame()
        new_overall_summaries.append(
            summarize_metric_repeats(overall_metric_df, ["split_protocol", "input_mode", "model", "plot_group", "prefix_end_cycle"])
        )
        new_dataset_summaries.append(
            summarize_metric_repeats(dataset_metric_df, ["split_protocol", "input_mode", "model", "plot_group", "dataset_name", "prefix_end_cycle"])
        )
    for spec in runnable_specs:
        preserved_overall_summary = _drop_spec(preserved_overall_summary, spec)
        preserved_dataset_summary = _drop_spec(preserved_dataset_summary, spec)
    overall_summary_df = (
        pd.concat([preserved_overall_summary, *[df for df in new_overall_summaries if not df.empty]], ignore_index=True)
        if (not preserved_overall_summary.empty or any(not df.empty for df in new_overall_summaries))
        else pd.DataFrame()
    )
    dataset_summary_df = (
        pd.concat([preserved_dataset_summary, *[df for df in new_dataset_summaries if not df.empty]], ignore_index=True)
        if (not preserved_dataset_summary.empty or any(not df.empty for df in new_dataset_summaries))
        else pd.DataFrame()
    )
    frames = {
        "cache_status": cache_status_frame(specs, result_set).assign(cache_ready=True, status="trained"),
        "overall_summary_df": overall_summary_df,
        "dataset_summary_df": dataset_summary_df,
        "overall_delta_summary_df": compute_delta_summary(
            overall_summary_df, ["split_protocol", "input_mode", "model", "plot_group"], ["mape", "rmse"]
        ),
        "dataset_delta_summary_df": compute_delta_summary(
            dataset_summary_df, ["split_protocol", "input_mode", "model", "plot_group", "dataset_name"], ["mape", "rmse"]
        ),
    }
    manifest = {
        "experiment": experiment_name,
        "result_set": result_set,
        "branch": branch_name,
        "models": [model_spec_to_dict(spec) for spec in specs],
        "status": "trained",
    }
    _write_outputs(output_dir, frames, config=config, manifest=manifest)
    return {"status": "trained", "cycle_output_dir": output_dir}

def run_descriptor_attribution(config: dict[str, Any], result_set: str = "ablation", aligned: AlignedData | None = None) -> dict[str, Any]:
    result_set = _validate_result_set(result_set)
    aligned = aligned or build_aligned_data(config)
    output_dir = result_output_dir(config, result_set, "small_cycle", "descriptor_attribution")
    cycles = [int(c) for c in config.get("data", {}).get("cvd_cycles", list(range(20, 101, 10)))]
    spec = ModelSpec(
        name="CCV-basic",
        key="ccv_basic",
        input_mode="abs",
        family="ccvnet",
        plot_group="ablation",
        cache_name="ccv_basic_abs_descriptor_attribution",
        notes="Descriptor-density attribution for CCVNet using sparse and dense descriptor schedules.",
    )
    total_tasks = len(cycles) * len(_split_specs(aligned, config)) * 2
    task_index = 0
    overall_frames, dataset_frames = [], []
    for prefix_end_cycle in cycles:
        for split_spec in _split_specs(aligned, config):
            for dense in (True, False):
                task_index += 1
                _progress_header("small_cycle/descriptor_attribution", result_set, task_index, total_tasks, f"model={spec.name} split={split_spec['split_protocol']} cycle={prefix_end_cycle} desc={'dense' if dense else 'sparse'}")
                overall, dataset = run_single_descriptor_attribution(
                    aligned,
                    spec,
                    split_spec["split_protocol"],
                    split_spec["split_df"],
                    split_spec["train_group_col"],
                    config,
                    prefix_end_cycle,
                    dense=dense,
                )
                overall_frames.append(overall)
                dataset_frames.append(dataset)
    overall_metric_df = pd.concat(overall_frames, ignore_index=True) if overall_frames else pd.DataFrame()
    dataset_metric_df = pd.concat(dataset_frames, ignore_index=True) if dataset_frames else pd.DataFrame()
    overall_summary_df = summarize_metric_repeats(overall_metric_df, ["split_protocol", "method", "prefix_end_cycle"])
    dataset_summary_df = summarize_metric_repeats(dataset_metric_df, ["split_protocol", "method", "dataset_name", "prefix_end_cycle"])
    frames = {
        "cache_status": pd.DataFrame(
            [
                {"result_set": result_set, "branch": "descriptor_attribution", "method": "CCV-basic dense-desc", "cache_ready": True, "missing_frames": "", "status": "trained"},
                {"result_set": result_set, "branch": "descriptor_attribution", "method": "CCV-basic sparse-desc", "cache_ready": True, "missing_frames": "", "status": "trained"},
            ]
        ),
        "overall_summary_df": overall_summary_df,
        "dataset_summary_df": dataset_summary_df,
        "gap_summary_df": compute_descriptor_gap_summary(overall_summary_df),
    }
    manifest = {"experiment": "small_cycle", "result_set": result_set, "branch": "descriptor_attribution", "models": [model_spec_to_dict(spec)], "status": "trained"}
    _write_outputs(output_dir, frames, config=config, manifest=manifest)
    return {"status": "trained", "descriptor_output_dir": output_dir}


def cache_status_frame(specs: list[ModelSpec], result_set: str) -> pd.DataFrame:
    rows = []
    for spec in specs:
        row = model_spec_to_dict(spec)
        row.update({"result_set": result_set, "cache_ready": False, "missing_frames": "", "status": "scheduled"})
        rows.append(row)
    return pd.DataFrame(rows)


def run_main_baseline(config: dict[str, Any], result_set: str) -> dict[str, Any]:
    result_set = _validate_result_set(result_set)
    aligned = build_aligned_data(config)
    specs = paper_experiment_model_specs(config, experiment_name="main_baseline", result_set=result_set)
    split_specs = _split_specs(aligned, config)
    output_dir = result_output_dir(config, result_set, "main_baseline")
    frame_names = ["overall_metric_df", "dataset_metric_df", "prediction_df", "skipped_df"]
    existing_frames = _load_existing_frames(output_dir, frame_names)
    required_cached_frames = ["overall_metric_df", "dataset_metric_df", "prediction_df"]
    runnable_specs = [spec for spec in specs if not _spec_cached(existing_frames, spec, required_cached_frames)]
    for spec in specs:
        if spec not in runnable_specs:
            print(f"[main_baseline][{result_set}] skip cached model={spec.name} input={spec.input_mode}", flush=True)
    total_tasks = len(runnable_specs) * len(split_specs)
    task_index = 0
    if total_tasks == 0:
        print(f"[main_baseline][{result_set}] nothing to run, all model caches already present.", flush=True)
    for spec in runnable_specs:
        overall_frames, dataset_frames, prediction_frames, skipped_frames = [], [], [], []
        for split_spec in split_specs:
            task_index += 1
            _progress_header("main_baseline", result_set, task_index, total_tasks, f"model={spec.name} input={spec.input_mode} split={split_spec['split_protocol']}")
            overall, dataset, pred, skipped = run_single_benchmark(aligned, specs, spec, split_spec["split_protocol"], split_spec["split_df"], split_spec["train_group_col"], config)
            overall_frames.append(overall); dataset_frames.append(dataset); prediction_frames.append(pred); skipped_frames.append(skipped)
        spec_frames = {
            "overall_metric_df": pd.concat(overall_frames, ignore_index=True) if overall_frames else pd.DataFrame(),
            "dataset_metric_df": pd.concat(dataset_frames, ignore_index=True) if dataset_frames else pd.DataFrame(),
            "prediction_df": pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame(),
            "skipped_df": pd.concat(skipped_frames, ignore_index=True) if skipped_frames else pd.DataFrame(),
        }
        existing_frames = _merge_spec_frames(existing_frames, spec_frames, spec, frame_names)
    frames = {
        "cache_status": cache_status_frame(specs, result_set).assign(cache_ready=True, status="trained"),
        "overall_metric_df": existing_frames["overall_metric_df"],
        "dataset_metric_df": existing_frames["dataset_metric_df"],
        "prediction_df": existing_frames["prediction_df"],
        "skipped_df": existing_frames["skipped_df"],
    }
    manifest = {"experiment": "main_baseline", "result_set": result_set, "models": [model_spec_to_dict(spec) for spec in specs], "status": "trained"}
    _write_outputs(output_dir, frames, config=config, manifest=manifest)
    return {"status": "trained", "output_dir": output_dir}


def run_small_data(config: dict[str, Any], result_set: str) -> dict[str, Any]:
    result_set = _validate_result_set(result_set)
    aligned = build_aligned_data(config)
    specs = paper_experiment_model_specs(config, experiment_name="small_data", result_set=result_set)
    split_specs = _split_specs(aligned, config)
    output_dir = result_output_dir(config, result_set, "small_data")
    frame_names = ["overall_metric_df", "dataset_metric_df", "prediction_df"]
    existing_frames = _load_existing_frames(output_dir, frame_names)
    required_cached_frames = ["overall_metric_df", "dataset_metric_df", "prediction_df"]
    runnable_specs = [spec for spec in specs if not _spec_cached(existing_frames, spec, required_cached_frames)]
    for spec in specs:
        if spec not in runnable_specs:
            print(f"[small_data][{result_set}] skip cached model={spec.name} input={spec.input_mode}", flush=True)
    total_tasks = len(runnable_specs) * len(split_specs)
    task_index = 0
    if total_tasks == 0:
        print(f"[small_data][{result_set}] nothing to run, all model caches already present.", flush=True)
    for spec in runnable_specs:
        overall_frames, dataset_frames, prediction_frames = [], [], []
        for split_spec in split_specs:
            task_index += 1
            _progress_header("small_data", result_set, task_index, total_tasks, f"model={spec.name} input={spec.input_mode} split={split_spec['split_protocol']}")
            training_mode = f"{split_spec['split_protocol']} small data"
            overall, dataset, pred = run_single_small_data(aligned, specs, spec, split_spec["split_protocol"], training_mode, split_spec["split_df"], split_spec["train_group_col"], config)
            overall_frames.append(overall); dataset_frames.append(dataset); prediction_frames.append(pred)
        spec_frames = {
            "overall_metric_df": pd.concat(overall_frames, ignore_index=True) if overall_frames else pd.DataFrame(),
            "dataset_metric_df": pd.concat(dataset_frames, ignore_index=True) if dataset_frames else pd.DataFrame(),
            "prediction_df": pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame(),
        }
        existing_frames = _merge_spec_frames(existing_frames, spec_frames, spec, frame_names)
    frames = {
        "cache_status": cache_status_frame(specs, result_set).assign(cache_ready=True, status="trained"),
        "overall_metric_df": existing_frames["overall_metric_df"],
        "dataset_metric_df": existing_frames["dataset_metric_df"],
        "prediction_df": existing_frames["prediction_df"],
    }
    manifest = {"experiment": "small_data", "result_set": result_set, "models": [model_spec_to_dict(spec) for spec in specs], "status": "trained"}
    _write_outputs(output_dir, frames, config=config, manifest=manifest)
    return {"status": "trained", "output_dir": output_dir}


def run_small_cycle(config: dict[str, Any], result_set: str) -> dict[str, Any]:
    result_set = _validate_result_set(result_set)
    aligned = build_aligned_data(config)
    specs = paper_experiment_model_specs(config, experiment_name="small_cycle", result_set=result_set)
    split_specs = _split_specs(aligned, config)
    output_dir = result_output_dir(config, result_set, "small_cycle", "cycle_effect")
    cycles = [int(c) for c in config.get("data", {}).get("cvd_cycles", list(range(20, 101, 10)))]
    frame_names = ["overall_summary_df", "dataset_summary_df", "overall_delta_summary_df", "dataset_delta_summary_df"]
    existing_frames = _load_existing_frames(output_dir, frame_names)
    required_cached_frames = ["overall_summary_df", "dataset_summary_df"]
    runnable_specs = [spec for spec in specs if not _spec_cached(existing_frames, spec, required_cached_frames)]
    for spec in specs:
        if spec not in runnable_specs:
            print(f"[small_cycle][{result_set}] skip cached model={spec.name} input={spec.input_mode}", flush=True)
    total_tasks = len(runnable_specs) * len(cycles) * len(split_specs)
    task_index = 0
    if total_tasks == 0:
        print(f"[small_cycle][{result_set}] nothing to run, all model caches already present.", flush=True)
    preserved_overall_summary = existing_frames["overall_summary_df"]
    preserved_dataset_summary = existing_frames["dataset_summary_df"]
    new_overall_summaries = []
    new_dataset_summaries = []
    for spec in runnable_specs:
        overall_metric_frames, dataset_metric_frames, prediction_frames = [], [], []
        for prefix_end_cycle in cycles:
            for split_spec in split_specs:
                task_index += 1
                _progress_header("small_cycle", result_set, task_index, total_tasks, f"model={spec.name} input={spec.input_mode} split={split_spec['split_protocol']} cycle={prefix_end_cycle}")
                overall, dataset, pred = run_single_cycle(aligned, specs, spec, split_spec["split_protocol"], split_spec["split_df"], split_spec["train_group_col"], config, prefix_end_cycle)
                overall_metric_frames.append(overall); dataset_metric_frames.append(dataset); prediction_frames.append(pred)
        overall_metric_df = pd.concat(overall_metric_frames, ignore_index=True) if overall_metric_frames else pd.DataFrame()
        dataset_metric_df = pd.concat(dataset_metric_frames, ignore_index=True) if dataset_metric_frames else pd.DataFrame()
        new_overall_summaries.append(summarize_metric_repeats(overall_metric_df, ["split_protocol", "input_mode", "model", "plot_group", "prefix_end_cycle"]))
        new_dataset_summaries.append(summarize_metric_repeats(dataset_metric_df, ["split_protocol", "input_mode", "model", "plot_group", "dataset_name", "prefix_end_cycle"]))
    for spec in runnable_specs:
        preserved_overall_summary = _drop_spec(preserved_overall_summary, spec)
        preserved_dataset_summary = _drop_spec(preserved_dataset_summary, spec)
    overall_summary_df = pd.concat([preserved_overall_summary, *[df for df in new_overall_summaries if not df.empty]], ignore_index=True) if (not preserved_overall_summary.empty or any(not df.empty for df in new_overall_summaries)) else pd.DataFrame()
    dataset_summary_df = pd.concat([preserved_dataset_summary, *[df for df in new_dataset_summaries if not df.empty]], ignore_index=True) if (not preserved_dataset_summary.empty or any(not df.empty for df in new_dataset_summaries)) else pd.DataFrame()
    frames = {
        "cache_status": cache_status_frame(specs, result_set).assign(cache_ready=True, status="trained"),
        "overall_summary_df": overall_summary_df,
        "dataset_summary_df": dataset_summary_df,
        "overall_delta_summary_df": compute_delta_summary(overall_summary_df, ["split_protocol", "input_mode", "model", "plot_group"], ["mape", "rmse"]),
        "dataset_delta_summary_df": compute_delta_summary(dataset_summary_df, ["split_protocol", "input_mode", "model", "plot_group", "dataset_name"], ["mape", "rmse"]),
    }
    manifest = {"experiment": "small_cycle", "result_set": result_set, "branch": "cycle_effect", "models": [model_spec_to_dict(spec) for spec in specs], "status": "trained"}
    _write_outputs(output_dir, frames, config=config, manifest=manifest)
    result = {"status": "trained", "cycle_output_dir": output_dir}
    if result_set in {"ablation", "all"}:
        result.update(run_descriptor_attribution(config, result_set=result_set, aligned=aligned))
    return result
