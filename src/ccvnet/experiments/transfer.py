from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from torch import nn

from ccvnet.cache import write_manifest, write_resolved_config
from ccvnet.pipeline import AlignedData, COARSE_GROUP_COLUMN, build_aligned_data
from ccvnet.splits import build_within_group_holdout_splits
from ccvnet.training import (
    fit_cvd_scaler,
    make_tensor_loader,
    predict_ccvnet,
    predict_cvd_regressor,
    resolve_device,
    split_descriptor_metadata,
    train_ccvnet,
    train_cnn,
    train_spectrumnn,
    transform_cvd,
)


EXPERIMENT_NAME = "transfer"
DEVICE = resolve_device("auto")
MODULE5_K_VALUES = [1, 2, 4, 8, 16]
MODULE5_SUPPORT_SEEDS = [42, 52, 62, 72, 82]
ZERO_SHOT_MODELS = [
    {"method": "cnn_zero_shot", "method_label": "CNN zero-shot", "input_mode": "norm", "kind": "cnn"},
    {
        "method": "spectrumnn_zero_shot",
        "method_label": "SpectrumNN zero-shot",
        "input_mode": "norm",
        "kind": "spectrumnn",
    },
    {
        "method": "ccv_basic_zero_shot",
        "method_label": "CCV-basic zero-shot",
        "input_mode": "abs",
        "kind": "ccv",
    },
]
TARGET_ONLY_MODELS = [
    {"model": "SpectrumNN", "model_label": "SpectrumNN", "input_mode": "norm", "kind": "spectrumnn"},
    {"model": "CNN", "model_label": "CNN", "input_mode": "norm", "kind": "cnn"},
    {"model": "CCV-basic", "model_label": "CCV-basic", "input_mode": "abs", "kind": "ccv"},
]
MAIN_TRANSFER_MODELS = [
    {
        "model": "SpectrumNN-transfer",
        "model_label": "SpectrumNN-transfer",
        "base_model": "SpectrumNN",
        "input_mode": "norm",
        "kind": "neural",
    },
    {
        "model": "CNN-transfer",
        "model_label": "CNN-transfer",
        "base_model": "CNN",
        "input_mode": "norm",
        "kind": "neural",
    },
    {
        "model": "CCV-transfer",
        "model_label": "CCV-transfer",
        "base_model": "CCV-basic",
        "input_mode": "abs",
        "kind": "ccv",
    },
]
ABLATION_MODELS = [
    {"method": "descriptor_local", "method_label": "Descriptor-local"},
    {"method": "ccv_transfer_meta", "method_label": "CCV-transfer-meta"},
    {"method": "ccv_hybrid", "method_label": "CCV-hybrid"},
]
CCV_TRANSFER_CONFIG = {
    "method": "ccv_transfer_rankdelta_calib",
    "method_label": "CCV-transfer",
    "residual_scale": 0.55,
    "lambda_delta": 0.20,
    "lambda_rank": 0.10,
    "lambda_delta_l2": 0.10,
    "lambda_center": 0.05,
}


def _results_root(config: dict[str, Any]) -> Path:
    return Path(config.get("paths", {}).get("results_dir", "results"))


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _write_outputs(output_dir: Path, frames: dict[str, pd.DataFrame], *, config: dict[str, Any], manifest: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    row_counts: dict[str, int] = {}
    for name, frame in frames.items():
        frame = frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()
        _write_csv(output_dir / f"{name}.csv", frame)
        row_counts[name] = int(len(frame))
    payload = dict(manifest)
    payload["row_counts"] = row_counts
    write_manifest(output_dir / "cache_manifest.json", payload)
    write_resolved_config(output_dir / "config_resolved.yaml", config)


def _group_mean_std(df: pd.DataFrame, group_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    grouped = df.groupby(group_cols, dropna=False)[metric_cols].agg(["mean", "std"]).reset_index()
    grouped.columns = [
        col[0] if col[1] == "" else (col[0] if col[1] == "mean" else f"{col[0]}_std")
        for col in grouped.columns.to_flat_index()
    ]
    return grouped


def _corr(a: np.ndarray, b: np.ndarray, method: str = "pearson") -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    valid = np.isfinite(a) & np.isfinite(b)
    if int(valid.sum()) < 3:
        return np.nan
    a = a[valid]
    b = b[valid]
    if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return np.nan
    if method == "pearson":
        return float(np.corrcoef(a, b)[0, 1])
    if method == "spearman":
        a_rank = pd.Series(a).rank(method="average").to_numpy(dtype=float)
        b_rank = pd.Series(b).rank(method="average").to_numpy(dtype=float)
        return float(np.corrcoef(a_rank, b_rank)[0, 1])
    raise KeyError(method)


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    if len(y_true) == 0:
        return {
            "n": 0,
            "rmse": np.nan,
            "mae": np.nan,
            "mape": np.nan,
            "pearson_r": np.nan,
            "spearman_r": np.nan,
        }
    err = y_pred - y_true
    valid_mape = np.abs(y_true) > 1e-12
    return {
        "n": int(len(y_true)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "mape": float(np.mean(np.abs(err[valid_mape] / y_true[valid_mape])) * 100.0) if valid_mape.any() else np.nan,
        "pearson_r": _corr(y_true, y_pred, method="pearson"),
        "spearman_r": _corr(y_true, y_pred, method="spearman"),
    }


def _ci95(values: np.ndarray) -> float:
    arr = pd.to_numeric(pd.Series(values).dropna(), errors="coerce").dropna().to_numpy(dtype=float)
    if len(arr) <= 1:
        return 0.0
    return float(1.96 * np.std(arr, ddof=1) / np.sqrt(len(arr)))


def _summarize_selected_metrics(df: pd.DataFrame, group_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    grouped = df.groupby(group_cols, dropna=False)[metric_cols].agg(["mean", "std"]).reset_index()
    grouped.columns = [
        col[0] if col[1] == "" else (col[0] if col[1] == "mean" else f"{col[0]}_std")
        for col in grouped.columns.to_flat_index()
    ]
    return grouped


def _build_grouping_settings(aligned: AlignedData, config: dict[str, Any]) -> list[dict[str, Any]]:
    split_cfg = config.get("split", {})
    seeds = [int(seed) for seed in split_cfg.get("seeds", MODULE5_SUPPORT_SEEDS)]
    test_fraction = float(split_cfg.get("test_fraction", 0.25))
    return [
        {
            "grouping_scheme": "by dataset",
            "group_col": COARSE_GROUP_COLUMN,
            "split_df": build_within_group_holdout_splits(aligned.pipeline_df, COARSE_GROUP_COLUMN, test_fraction, seeds),
        },
        {
            "grouping_scheme": "fine subgroup",
            "group_col": aligned.fine_group_col,
            "split_df": build_within_group_holdout_splits(aligned.pipeline_df, aligned.fine_group_col, test_fraction, seeds),
        },
    ]


def _availability_complete_mask(X_cvd: np.ndarray) -> np.ndarray:
    X_cvd = np.asarray(X_cvd, dtype=np.float32)
    if X_cvd.ndim != 3 or X_cvd.shape[1] < 2:
        return np.ones(len(X_cvd), dtype=bool)
    half = X_cvd.shape[1] // 2
    if half <= 0 or half >= X_cvd.shape[1]:
        return np.ones(len(X_cvd), dtype=bool)
    mask_block = X_cvd[:, half:, :]
    channel_has_signal = np.nanmax(mask_block, axis=2) > 0.5
    return np.all(channel_has_signal, axis=1)


def _descriptor_complete_mask(aligned: AlignedData, input_mode: str = "abs") -> np.ndarray:
    X_value = aligned.X_value_abs if input_mode == "abs" else aligned.X_value_norm
    desc = X_value[:, : aligned.descriptor_raw_dim]
    return np.all(np.isfinite(desc), axis=1)


def _complete_case_indices(aligned: AlignedData, kind: str, indices: np.ndarray) -> np.ndarray:
    indices = np.asarray(indices, dtype=int)
    if kind == "cvd_abs":
        mask = _availability_complete_mask(aligned.X_cvd_abs)
        return indices[mask[indices]]
    if kind == "cvd_norm":
        mask = _availability_complete_mask(aligned.X_cvd_norm)
        return indices[mask[indices]]
    if kind == "descriptor_abs":
        mask = _descriptor_complete_mask(aligned, "abs")
        return indices[mask[indices]]
    if kind == "cvd_desc_abs":
        mask = _availability_complete_mask(aligned.X_cvd_norm) & _descriptor_complete_mask(aligned, "abs")
        return indices[mask[indices]]
    raise KeyError(kind)


def _pair_feature_transform(X: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    anchor = np.asarray(anchor, dtype=np.float32)
    if X.ndim == 3:
        return (X - anchor).reshape(len(X), -1).astype(np.float32)
    return (X - anchor).reshape(len(X), -1).astype(np.float32)


def _select_anchor(y: np.ndarray, indices: np.ndarray) -> tuple[int, float]:
    indices = np.asarray(indices, dtype=int)
    y_support = np.asarray(y[indices], dtype=np.float32)
    order = np.argsort(y_support)
    anchor_pos = int(order[len(order) // 2])
    anchor_idx = int(indices[anchor_pos])
    return anchor_idx, float(y[anchor_idx])


def _fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> Ridge | None:
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if len(X) < 2:
        return None
    model = Ridge(alpha=alpha, fit_intercept=True, random_state=0)
    model.fit(X, y)
    return model


def _predict_ridge(model: Ridge | None, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if model is None or len(X) == 0:
        return np.zeros(len(X), dtype=np.float32)
    return np.asarray(model.predict(X), dtype=np.float32).reshape(-1)


def _affine_calibration(y_source: np.ndarray, y_target: np.ndarray) -> tuple[float, float]:
    y_source = np.asarray(y_source, dtype=np.float32).reshape(-1)
    y_target = np.asarray(y_target, dtype=np.float32).reshape(-1)
    valid = np.isfinite(y_source) & np.isfinite(y_target)
    y_source = y_source[valid]
    y_target = y_target[valid]
    if len(y_source) == 0:
        return 1.0, 0.0
    if len(y_source) == 1 or np.std(y_source) <= 1e-12:
        return 1.0, float(np.mean(y_target) - np.mean(y_source))
    slope, intercept = np.polyfit(y_source, y_target, deg=1)
    return float(slope), float(intercept)


def _reliability_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float32).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float32).reshape(-1)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(valid.sum()) < 3:
        return np.nan
    mae = np.mean(np.abs(y_true[valid] - y_pred[valid]))
    scale = np.std(y_true[valid]) + 1e-6
    return float(max(0.0, 1.0 - mae / (scale + 1e-6)))


def _local_kernel_predict(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32).reshape(-1)
    X_test = np.asarray(X_test, dtype=np.float32)
    if len(X_train) == 0:
        return np.zeros(len(X_test), dtype=np.float32)
    if len(X_train) == 1:
        return np.full(len(X_test), float(y_train[0]), dtype=np.float32)
    dist = np.linalg.norm(X_train[:, None, :] - X_test[None, :, :], axis=2)
    bandwidth = float(np.nanmedian(dist))
    bandwidth = bandwidth if np.isfinite(bandwidth) and bandwidth > 1e-6 else 1.0
    weight = np.exp(-(dist**2) / (2.0 * bandwidth**2))
    denom = np.sum(weight, axis=0, keepdims=True) + 1e-8
    pred = (weight.T @ y_train.reshape(-1, 1)).reshape(-1) / denom.reshape(-1)
    return pred.astype(np.float32)


def _local_leave_one_out_predict(X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32).reshape(-1)
    if len(X_train) <= 1:
        return np.zeros(len(X_train), dtype=np.float32)
    preds = []
    for idx in range(len(X_train)):
        keep = np.ones(len(X_train), dtype=bool)
        keep[idx] = False
        preds.append(_local_kernel_predict(X_train[keep], y_train[keep], X_train[idx : idx + 1])[0])
    return np.asarray(preds, dtype=np.float32)


def _train_source_neural(kind: str, X_source: np.ndarray, y_source: np.ndarray, *, seed: int) -> dict[str, Any]:
    if kind == "spectrumnn":
        return train_spectrumnn(X_source, y_source, max_epochs=160, patience=22, batch_size=32, seed=seed)
    if kind == "cnn":
        return train_cnn(X_source, y_source, max_epochs=160, patience=22, batch_size=32, seed=seed)
    raise KeyError(kind)


def _predict_neural(bundle: dict[str, Any], X_cvd: np.ndarray, base_model: str) -> np.ndarray:
    pred, _ = predict_cvd_regressor(bundle, X_cvd)
    return np.asarray(pred, dtype=np.float32).reshape(-1)


def _target_only_schedule(k_shot: int) -> tuple[int, int]:
    if int(k_shot) <= 2:
        return 120, 30
    if int(k_shot) <= 8:
        return 180, 30
    return 240, 30


def _ccv_transform_inputs(bundle: dict[str, Any], X_cvd: np.ndarray, X_value: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_cvd_scaled = transform_cvd(X_cvd, bundle["cvd_scaler"])
    X_desc, X_meta = split_descriptor_metadata(X_value, bundle["descriptor_raw_dim"], bundle["metadata_raw_dim"])
    from ccvnet.models.mlp import tabular_missing_aware_matrix

    X_desc_scaled = bundle["desc_preprocessor"].transform(tabular_missing_aware_matrix(X_desc)).astype(np.float32)
    if bundle["meta_preprocessor"] is not None and X_meta.shape[1]:
        X_meta_scaled = bundle["meta_preprocessor"].transform(tabular_missing_aware_matrix(X_meta)).astype(np.float32)
    else:
        X_meta_scaled = np.zeros((len(X_desc_scaled), 0), dtype=np.float32)
    return X_cvd_scaled.astype(np.float32), X_desc_scaled, X_meta_scaled


def _set_ccv_trainable_layers(model: nn.Module, variant: str) -> int:
    for param in model.parameters():
        param.requires_grad = False
    if hasattr(model, "descriptor_encoder"):
        for param in model.descriptor_encoder.parameters():
            param.requires_grad = False
    if hasattr(model, "base_head"):
        for param in model.base_head.parameters():
            param.requires_grad = True
    if getattr(model, "meta_residual", None) is not None:
        net = getattr(model.meta_residual, "net", None)
        if net is not None:
            for param in net.parameters():
                param.requires_grad = True
    if variant == "meta_head_only":
        return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    if hasattr(model, "cvd_backbone") and hasattr(model.cvd_backbone, "embedding"):
        for param in model.cvd_backbone.embedding.parameters():
            param.requires_grad = True
    if hasattr(model, "cvd_backbone") and hasattr(model.cvd_backbone, "attention_pool"):
        for param in model.cvd_backbone.attention_pool.parameters():
            param.requires_grad = True
    if variant == "spec_last2":
        if hasattr(model, "cvd_backbone") and hasattr(model.cvd_backbone, "encoder"):
            encoder = model.cvd_backbone.encoder
            last_block = encoder[-1] if isinstance(encoder, nn.Sequential) and len(encoder) else None
            if last_block is not None:
                for param in last_block.parameters():
                    param.requires_grad = True
        if hasattr(model, "cvd_backbone") and hasattr(model.cvd_backbone, "cycle_encoder"):
            cycle_encoder = model.cvd_backbone.cycle_encoder
            last_block = cycle_encoder[-1] if isinstance(cycle_encoder, nn.Sequential) and len(cycle_encoder) else None
            if last_block is not None:
                for param in last_block.parameters():
                    param.requires_grad = True
        if hasattr(model, "cvd_backbone") and hasattr(model.cvd_backbone, "cycle_projector"):
            for param in model.cvd_backbone.cycle_projector.parameters():
                param.requires_grad = True
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _pair_losses(pred_scaled: torch.Tensor, y_scaled: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if pred_scaled.ndim == 0:
        pred_scaled = pred_scaled.view(1)
    if y_scaled.ndim == 0:
        y_scaled = y_scaled.view(1)
    if pred_scaled.numel() <= 1:
        zero = pred_scaled.new_tensor(0.0)
        return zero, zero
    delta_pred = pred_scaled[:, None] - pred_scaled[None, :]
    delta_true = y_scaled[:, None] - y_scaled[None, :]
    mask = ~torch.eye(len(pred_scaled), dtype=torch.bool, device=pred_scaled.device)
    pred_vec = delta_pred[mask]
    true_vec = delta_true[mask]
    delta_loss = torch.mean(torch.abs(pred_vec - true_vec))
    rank_target = torch.sign(true_vec)
    rank_margin = 0.05
    rank_loss = torch.relu(rank_margin - rank_target * pred_vec)
    rank_loss = torch.mean(rank_loss)
    return delta_loss, rank_loss


def _predict_ccv_scaled(model: nn.Module, residual_scale: float, xb_cvd: torch.Tensor, xb_desc: torch.Tensor, xb_meta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred_scaled, y_base, delta, *_ = model(xb_cvd, xb_desc, xb_meta, return_details=True)
    pred_used = y_base + float(residual_scale) * delta
    return pred_used, y_base, delta


def _ccv_transfer_meta_finetune_bundle(
    source_bundle: dict[str, Any],
    X_support_cvd: np.ndarray,
    X_support_value: np.ndarray,
    y_support: np.ndarray,
    *,
    variant: str,
    seed: int,
) -> tuple[dict[str, Any], int]:
    model = copy.deepcopy(source_bundle["model"]).to(DEVICE)
    n_trainable = _set_ccv_trainable_layers(model, variant)
    X_cvd_scaled, X_desc_scaled, X_meta_scaled = _ccv_transform_inputs(source_bundle, X_support_cvd, X_support_value)
    y_support = np.asarray(y_support, dtype=np.float32)
    y_scaled = ((y_support - float(source_bundle["y_mean"])) / float(source_bundle["y_std"])).astype(np.float32)
    loader = make_tensor_loader(
        X_cvd_scaled,
        X_desc_scaled,
        X_meta_scaled,
        y_scaled,
        batch_size=min(16, max(1, len(y_scaled))),
        shuffle=True,
    )
    n_support = len(y_support)
    n_epochs = 40 if n_support <= 2 else (55 if n_support <= 8 else 70)
    lr = 4e-4 if variant == "meta_head_only" else 3e-4
    loss_fn = nn.SmoothL1Loss()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    torch.manual_seed(seed)
    for _ in range(n_epochs):
        model.train()
        for xb_cvd, xb_desc, xb_meta, yb in loader:
            xb_cvd = xb_cvd.to(DEVICE)
            xb_desc = xb_desc.to(DEVICE)
            xb_meta = xb_meta.to(DEVICE)
            yb = yb.to(DEVICE)
            opt.zero_grad()
            pred_scaled, _, delta = _predict_ccv_scaled(model, 1.0, xb_cvd, xb_desc, xb_meta)
            abs_loss = loss_fn(pred_scaled, yb)
            loss = abs_loss if variant == "meta_head_only" else abs_loss + 0.05 * torch.mean(delta**2)
            loss.backward()
            opt.step()
    tuned = dict(source_bundle)
    tuned["model"] = model
    return tuned, int(n_trainable)


def _ccv_transfer_finetune_epochs(n_support: int) -> int:
    if int(n_support) <= 2:
        return 60
    if int(n_support) <= 8:
        return 100
    return 140


def _ccv_transfer_hybrid_finetune_bundle(
    source_bundle: dict[str, Any],
    X_support_cvd: np.ndarray,
    X_support_value: np.ndarray,
    y_support: np.ndarray,
    *,
    config: dict[str, Any],
    seed: int,
) -> tuple[dict[str, Any], int]:
    model = copy.deepcopy(source_bundle["model"]).to(DEVICE)
    n_trainable = _set_ccv_trainable_layers(model, "spec_last2")
    residual_scale = float(config.get("residual_scale", 0.55))
    X_cvd_scaled, X_desc_scaled, X_meta_scaled = _ccv_transform_inputs(source_bundle, X_support_cvd, X_support_value)
    y_support = np.asarray(y_support, dtype=np.float32)
    y_scaled = ((y_support - float(source_bundle["y_mean"])) / float(source_bundle["y_std"])).astype(np.float32)
    loader = make_tensor_loader(
        X_cvd_scaled,
        X_desc_scaled,
        X_meta_scaled,
        y_scaled,
        batch_size=min(16, max(1, len(y_scaled))),
        shuffle=True,
    )
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()
    torch.manual_seed(seed)
    for _ in range(_ccv_transfer_finetune_epochs(len(y_support))):
        model.train()
        for xb_cvd, xb_desc, xb_meta, yb in loader:
            xb_cvd = xb_cvd.to(DEVICE)
            xb_desc = xb_desc.to(DEVICE)
            xb_meta = xb_meta.to(DEVICE)
            yb = yb.to(DEVICE)
            opt.zero_grad()
            pred_scaled, _, delta = _predict_ccv_scaled(model, residual_scale, xb_cvd, xb_desc, xb_meta)
            abs_loss = loss_fn(pred_scaled, yb)
            delta_loss, rank_loss = _pair_losses(pred_scaled, yb)
            loss = (
                abs_loss
                + float(config.get("lambda_delta", 0.0)) * delta_loss
                + float(config.get("lambda_rank", 0.0)) * rank_loss
                + float(config.get("lambda_delta_l2", 0.0)) * torch.mean(delta**2)
                + float(config.get("lambda_center", 0.0)) * (torch.mean(delta) ** 2)
            )
            loss.backward()
            opt.step()
    tuned = dict(source_bundle)
    tuned["model"] = model
    tuned["ccv_transfer_residual_scale"] = residual_scale
    return tuned, int(n_trainable)


def _ccv_transfer_hybrid_predict_bundle(bundle: dict[str, Any], X_cvd: np.ndarray, X_value: np.ndarray) -> np.ndarray:
    X_cvd_scaled, X_desc_scaled, X_meta_scaled = _ccv_transform_inputs(bundle, X_cvd, X_value)
    model = bundle["model"].to(DEVICE)
    model.eval()
    residual_scale = float(bundle.get("ccv_transfer_residual_scale", 0.55))
    preds = []
    with torch.no_grad():
        for start in range(0, len(X_cvd_scaled), 128):
            stop = start + 128
            xb_cvd = torch.as_tensor(X_cvd_scaled[start:stop], dtype=torch.float32, device=DEVICE)
            xb_desc = torch.as_tensor(X_desc_scaled[start:stop], dtype=torch.float32, device=DEVICE)
            xb_meta = torch.as_tensor(X_meta_scaled[start:stop], dtype=torch.float32, device=DEVICE)
            pred_scaled, _, _ = _predict_ccv_scaled(model, residual_scale, xb_cvd, xb_desc, xb_meta)
            preds.append(pred_scaled.detach().cpu().numpy())
    pred_scaled = np.concatenate(preds).reshape(-1)
    return (pred_scaled * float(bundle["y_std"]) + float(bundle["y_mean"])).astype(np.float32)


def _finetune_neural(
    source_bundle: dict[str, Any],
    X_support: np.ndarray,
    y_support: np.ndarray,
    *,
    seed: int,
) -> tuple[dict[str, Any], int]:
    model = copy.deepcopy(source_bundle["model"]).to(DEVICE)
    for param in model.parameters():
        param.requires_grad = False
    for layer_name in ["embedding", "regressor"]:
        layer = getattr(model, layer_name)
        for param in layer.parameters():
            param.requires_grad = True
    n_trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    X_scaled = transform_cvd(X_support, source_bundle["scaler"])
    y_support = np.asarray(y_support, dtype=np.float32)
    y_scaled = ((y_support - float(source_bundle["y_mean"])) / float(source_bundle["y_std"])).astype(np.float32)
    loader = make_tensor_loader(
        X_scaled,
        y_scaled,
        batch_size=min(16, max(1, len(y_scaled))),
        shuffle=True,
    )
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()
    torch.manual_seed(seed)
    for _ in range(_ccv_transfer_finetune_epochs(len(y_support))):
        model.train()
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            opt.zero_grad()
            pred_scaled = model(xb)
            abs_loss = loss_fn(pred_scaled, yb)
            delta_loss, rank_loss = _pair_losses(pred_scaled, yb)
            loss = abs_loss + 0.20 * delta_loss + 0.10 * rank_loss
            loss.backward()
            opt.step()
    tuned = dict(source_bundle)
    tuned["model"] = model
    return tuned, int(n_trainable)


def _train_source_bundle(
    aligned: AlignedData,
    group_col: str,
    source_mask: np.ndarray,
) -> dict[str, Any]:
    desc_matrix = np.asarray(aligned.X_value_abs[:, : aligned.descriptor_raw_dim], dtype=np.float32)
    cvd_abs_matrix = np.asarray(aligned.X_cvd_abs, dtype=np.float32)
    group_values = aligned.pipeline_df[group_col].astype(str)
    rows_primary = []
    rows_companion = []
    targets = []
    for source_group in sorted(group_values.loc[source_mask].dropna().unique().tolist()):
        source_idx = np.where(source_mask & group_values.eq(str(source_group)).to_numpy())[0]
        source_idx = _complete_case_indices(aligned, "cvd_desc_abs", source_idx)
        if len(source_idx) < 2:
            continue
        anchor_idx, anchor_life = _select_anchor(aligned.y, source_idx)
        non_anchor = source_idx[source_idx != anchor_idx]
        if len(non_anchor) == 0:
            continue
        anchor_desc = desc_matrix[anchor_idx : anchor_idx + 1]
        anchor_cvd = cvd_abs_matrix[anchor_idx : anchor_idx + 1]
        rows_primary.append(_pair_feature_transform(desc_matrix[non_anchor], anchor_desc))
        rows_companion.append(_pair_feature_transform(cvd_abs_matrix[non_anchor], anchor_cvd))
        targets.append((aligned.y[non_anchor] - anchor_life).astype(np.float32))
    if not targets:
        return {"primary_model": None, "companion_model": None}
    X_primary = np.concatenate(rows_primary, axis=0)
    X_companion = np.concatenate(rows_companion, axis=0)
    y_delta = np.concatenate(targets, axis=0)
    primary_model = _fit_ridge(X_primary, y_delta, alpha=1.0)
    companion_model = _fit_ridge(X_companion, y_delta, alpha=1.0)
    return {"primary_model": primary_model, "companion_model": companion_model}


def _predict_source_bundle(bundle: dict[str, Any], X_primary: np.ndarray, X_companion: np.ndarray) -> np.ndarray:
    preds = []
    if bundle.get("primary_model") is not None:
        preds.append(_predict_ridge(bundle["primary_model"], X_primary))
    if bundle.get("companion_model") is not None:
        preds.append(_predict_ridge(bundle["companion_model"], X_companion))
    if not preds:
        return np.zeros(len(X_primary), dtype=np.float32)
    return np.mean(np.vstack(preds), axis=0).astype(np.float32)


def _source_alpha_cap(k_shot: int) -> float:
    cap_map = {1: 0.00, 2: 0.08, 4: 0.18, 8: 0.30, 16: 0.45}
    return float(cap_map.get(int(k_shot), 0.20))


def _descriptor_gap_factor(X_source_desc: np.ndarray, X_target_desc: np.ndarray) -> tuple[float, float]:
    if len(X_source_desc) < 3 or len(X_target_desc) < 1:
        return np.nan, 0.0
    from ccvnet.models.mlp import tabular_missing_aware_matrix

    scaler = StandardScaler()
    X_source_scaled = scaler.fit_transform(tabular_missing_aware_matrix(X_source_desc))
    X_target_scaled = scaler.transform(tabular_missing_aware_matrix(X_target_desc))
    source_center = np.nanmean(X_source_scaled, axis=0)
    target_center = np.nanmean(X_target_scaled, axis=0)
    gap_value = float(np.sqrt(np.nanmean((target_center - source_center) ** 2)))
    gap_factor = float(1.0 / (1.0 + max(0.0, gap_value)))
    return gap_value, gap_factor


def _run_zero_shot(config: dict[str, Any], aligned: AlignedData, grouping_settings: list[dict[str, Any]]) -> None:
    test_metric_rows: list[dict[str, Any]] = []
    all_metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []

    for grouping_setting in grouping_settings:
        grouping_scheme = grouping_setting["grouping_scheme"]
        group_col = grouping_setting["group_col"]
        split_df = grouping_setting["split_df"]
        group_values = aligned.pipeline_df[group_col].astype(str)
        split_ids = sorted(pd.to_numeric(split_df["split_id"], errors="coerce").dropna().astype(int).unique().tolist())

        for split_id in split_ids:
            split_sub = split_df.loc[split_df["split_id"].eq(split_id)].sort_values("row_index")
            split_train_mask = split_sub["split"].eq("train").to_numpy()
            split_test_mask = split_sub["split"].eq("test").to_numpy()
            split_group_labels = sorted(group_values.loc[split_test_mask].dropna().unique().tolist())
            source_bundle_cache: dict[tuple[str, int, str], dict[str, Any]] = {}

            for group_label in split_group_labels:
                target_test_mask = split_test_mask & group_values.eq(str(group_label)).to_numpy()
                target_all_mask = group_values.eq(str(group_label)).to_numpy()
                source_train_mask = split_train_mask & (~group_values.eq(str(group_label)).to_numpy())
                dataset_name = str(aligned.pipeline_df.loc[target_all_mask, COARSE_GROUP_COLUMN].iloc[0])

                shared_source_idx = np.intersect1d(
                    _complete_case_indices(aligned, "cvd_norm", np.where(source_train_mask)[0]),
                    _complete_case_indices(aligned, "cvd_desc_abs", np.where(source_train_mask)[0]),
                )
                shared_test_idx = np.intersect1d(
                    _complete_case_indices(aligned, "cvd_norm", np.where(target_test_mask)[0]),
                    _complete_case_indices(aligned, "cvd_desc_abs", np.where(target_test_mask)[0]),
                )
                shared_all_idx = np.intersect1d(
                    _complete_case_indices(aligned, "cvd_norm", np.where(target_all_mask)[0]),
                    _complete_case_indices(aligned, "cvd_desc_abs", np.where(target_all_mask)[0]),
                )

                if len(shared_test_idx) < 3:
                    skipped_rows.append(
                        {
                            "grouping_scheme": grouping_scheme,
                            "method": "all",
                            "split_id": int(split_id),
                            "group_label": str(group_label),
                            "dataset_name": dataset_name,
                            "evaluation": "test-set",
                            "reason": "insufficient_shared_test_cases",
                        }
                    )
                    continue
                if len(shared_source_idx) < 12:
                    skipped_rows.append(
                        {
                            "grouping_scheme": grouping_scheme,
                            "method": "all",
                            "split_id": int(split_id),
                            "group_label": str(group_label),
                            "dataset_name": dataset_name,
                            "evaluation": "source-train",
                            "reason": "insufficient_shared_source_cases",
                        }
                    )
                    continue

                for model_spec in ZERO_SHOT_MODELS:
                    cache_key = (str(group_label), int(split_id), str(model_spec["method"]))
                    if cache_key not in source_bundle_cache:
                        if model_spec["kind"] == "spectrumnn":
                            source_bundle_cache[cache_key] = train_spectrumnn(
                                aligned.X_cvd_norm[shared_source_idx],
                                aligned.y[shared_source_idx],
                                max_epochs=160,
                                patience=20,
                                batch_size=32,
                                seed=566000 + 10000 * int(split_id) + 100 * len(str(group_label)),
                            )
                        elif model_spec["kind"] == "cnn":
                            source_bundle_cache[cache_key] = train_cnn(
                                aligned.X_cvd_norm[shared_source_idx],
                                aligned.y[shared_source_idx],
                                max_epochs=160,
                                patience=20,
                                batch_size=32,
                                seed=566100 + 10000 * int(split_id) + 100 * len(str(group_label)),
                            )
                        else:
                            source_bundle_cache[cache_key] = train_ccvnet(
                                aligned.X_cvd_abs[shared_source_idx],
                                aligned.X_value_abs[shared_source_idx],
                                aligned.y[shared_source_idx],
                                descriptor_raw_dim=aligned.descriptor_raw_dim,
                                metadata_raw_dim=aligned.metadata_raw_dim,
                                max_epochs=160,
                                patience=20,
                                batch_size=32,
                                seed=566200 + 10000 * int(split_id) + 100 * len(str(group_label)),
                                model_variant="ccv_basic",
                            )
                    bundle = source_bundle_cache[cache_key]

                    for eval_name, eval_idx in [("test-set", shared_test_idx), ("full-target-data", shared_all_idx)]:
                        if len(eval_idx) < 3:
                            skipped_rows.append(
                                {
                                    "grouping_scheme": grouping_scheme,
                                    "method": model_spec["method"],
                                    "split_id": int(split_id),
                                    "group_label": str(group_label),
                                    "dataset_name": dataset_name,
                                    "evaluation": eval_name,
                                    "reason": "insufficient_shared_eval_cases",
                                }
                            )
                            continue
                        if model_spec["kind"] == "ccv":
                            pred, _, _ = predict_ccvnet(bundle, aligned.X_cvd_abs[eval_idx], aligned.X_value_abs[eval_idx])
                        else:
                            pred, _ = predict_cvd_regressor(bundle, aligned.X_cvd_norm[eval_idx])
                        pred = np.asarray(pred, dtype=np.float32).reshape(-1)
                        metrics = _regression_metrics(aligned.y[eval_idx], pred)
                        row = {
                            "grouping_scheme": grouping_scheme,
                            "method": model_spec["method"],
                            "method_label": model_spec["method_label"],
                            "input_mode": model_spec["input_mode"],
                            "split_id": int(split_id),
                            "group_label": str(group_label),
                            "dataset_name": dataset_name,
                            "evaluation": eval_name,
                            "n_test": int(len(eval_idx)),
                            **metrics,
                        }
                        (test_metric_rows if eval_name == "test-set" else all_metric_rows).append(row)
                        for row_index, y_true_i, pred_i in zip(eval_idx, aligned.y[eval_idx], pred):
                            prediction_rows.append(
                                {
                                    "grouping_scheme": grouping_scheme,
                                    "method": model_spec["method"],
                                    "method_label": model_spec["method_label"],
                                    "input_mode": model_spec["input_mode"],
                                    "split_id": int(split_id),
                                    "group_label": str(group_label),
                                    "dataset_name": dataset_name,
                                    "evaluation": eval_name,
                                    "row_index": int(row_index),
                                    "target_life": float(y_true_i),
                                    "pred_life": float(pred_i),
                                }
                            )

    out_dir = _results_root(config) / "major" / "transfer_zero_shot"
    _write_outputs(
        out_dir,
        {
            "test_metric_df": pd.DataFrame(test_metric_rows),
            "test_summary_df": _summarize_selected_metrics(pd.DataFrame(test_metric_rows), ["grouping_scheme", "method", "method_label", "input_mode"], ["mape", "rmse", "mae", "pearson_r", "spearman_r"]),
            "all_metric_df": pd.DataFrame(all_metric_rows),
            "all_summary_df": _summarize_selected_metrics(pd.DataFrame(all_metric_rows), ["grouping_scheme", "method", "method_label", "input_mode"], ["mape", "rmse", "mae", "pearson_r", "spearman_r"]),
            "prediction_df": pd.DataFrame(prediction_rows),
            "skipped_df": pd.DataFrame(skipped_rows),
        },
        config=config,
        manifest={"experiment": EXPERIMENT_NAME, "branch": "transfer_zero_shot", "status": "trained"},
    )


def _run_target_only(config: dict[str, Any], aligned: AlignedData, grouping_settings: list[dict[str, Any]]) -> None:
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []

    for grouping_setting in grouping_settings:
        grouping_scheme = grouping_setting["grouping_scheme"]
        group_col = grouping_setting["group_col"]
        split_df = grouping_setting["split_df"]
        group_values = aligned.pipeline_df[group_col].astype(str)
        split_ids = sorted(pd.to_numeric(split_df["split_id"], errors="coerce").dropna().astype(int).unique().tolist())

        for split_id in split_ids:
            split_sub = split_df.loc[split_df["split_id"].eq(split_id)].sort_values("row_index")
            split_train_mask = split_sub["split"].eq("train").to_numpy()
            split_test_mask = split_sub["split"].eq("test").to_numpy()
            split_group_labels = sorted(group_values.loc[split_test_mask].dropna().unique().tolist())

            for group_label in split_group_labels:
                target_train_mask = split_train_mask & group_values.eq(str(group_label)).to_numpy()
                target_test_mask = split_test_mask & group_values.eq(str(group_label)).to_numpy()
                dataset_name = str(aligned.pipeline_df.loc[target_test_mask, COARSE_GROUP_COLUMN].iloc[0])
                support_pool = np.intersect1d(
                    _complete_case_indices(aligned, "cvd_norm", np.where(target_train_mask)[0]),
                    _complete_case_indices(aligned, "cvd_desc_abs", np.where(target_train_mask)[0]),
                )
                test_idx = np.intersect1d(
                    _complete_case_indices(aligned, "cvd_norm", np.where(target_test_mask)[0]),
                    _complete_case_indices(aligned, "cvd_desc_abs", np.where(target_test_mask)[0]),
                )

                if len(test_idx) < 3:
                    skipped_rows.append(
                        {
                            "grouping_scheme": grouping_scheme,
                            "model": "all",
                            "split_id": int(split_id),
                            "support_seed": np.nan,
                            "k_shot": np.nan,
                            "group_label": str(group_label),
                            "dataset_name": dataset_name,
                            "reason": "insufficient_shared_test_cases",
                        }
                    )
                    continue

                for support_seed in MODULE5_SUPPORT_SEEDS:
                    for k_shot in MODULE5_K_VALUES:
                        if len(support_pool) < int(k_shot):
                            skipped_rows.append(
                                {
                                    "grouping_scheme": grouping_scheme,
                                    "model": "all",
                                    "split_id": int(split_id),
                                    "support_seed": int(support_seed),
                                    "k_shot": int(k_shot),
                                    "group_label": str(group_label),
                                    "dataset_name": dataset_name,
                                    "reason": "insufficient_shared_support",
                                }
                            )
                            continue
                        rng = np.random.default_rng(567000 + 1000 * int(split_id) + 100 * int(k_shot) + int(support_seed) + len(str(group_label)))
                        support_idx = np.sort(rng.choice(support_pool, size=int(k_shot), replace=False).astype(int))
                        max_epochs, patience = _target_only_schedule(int(k_shot))
                        y_support = aligned.y[support_idx]
                        for model_spec in TARGET_ONLY_MODELS:
                            try:
                                if model_spec["kind"] == "spectrumnn":
                                    bundle = train_spectrumnn(aligned.X_cvd_norm[support_idx], y_support, max_epochs=max_epochs, patience=patience, seed=567100 + int(split_id) + int(support_seed))
                                    pred_test, _ = predict_cvd_regressor(bundle, aligned.X_cvd_norm[test_idx])
                                elif model_spec["kind"] == "cnn":
                                    bundle = train_cnn(aligned.X_cvd_norm[support_idx], y_support, max_epochs=max_epochs, patience=patience, seed=567200 + int(split_id) + int(support_seed))
                                    pred_test, _ = predict_cvd_regressor(bundle, aligned.X_cvd_norm[test_idx])
                                else:
                                    bundle = train_ccvnet(
                                        aligned.X_cvd_abs[support_idx],
                                        aligned.X_value_abs[support_idx],
                                        y_support,
                                        descriptor_raw_dim=aligned.descriptor_raw_dim,
                                        metadata_raw_dim=aligned.metadata_raw_dim,
                                        max_epochs=max_epochs,
                                        patience=patience,
                                        batch_size=min(16, max(1, len(support_idx))),
                                        seed=567300 + int(split_id) + int(support_seed),
                                        model_variant="ccv_basic",
                                    )
                                    pred_test, _, _ = predict_ccvnet(bundle, aligned.X_cvd_abs[test_idx], aligned.X_value_abs[test_idx])
                                pred_test = np.asarray(pred_test, dtype=np.float32).reshape(-1)
                                metric_rows.append(
                                    {
                                        "grouping_scheme": grouping_scheme,
                                        "model": model_spec["model"],
                                        "model_label": model_spec["model_label"],
                                        "input_mode": model_spec["input_mode"],
                                        "split_id": int(split_id),
                                        "support_seed": int(support_seed),
                                        "k_shot": int(k_shot),
                                        "group_label": str(group_label),
                                        "dataset_name": dataset_name,
                                        "train_mode": "target_only_fewshot",
                                        "n_support": int(len(support_idx)),
                                        "n_test": int(len(test_idx)),
                                        **_regression_metrics(aligned.y[test_idx], pred_test),
                                    }
                                )
                                for row_index, y_true_i, pred_i in zip(test_idx, aligned.y[test_idx], pred_test):
                                    prediction_rows.append(
                                        {
                                            "grouping_scheme": grouping_scheme,
                                            "model": model_spec["model"],
                                            "input_mode": model_spec["input_mode"],
                                            "split_id": int(split_id),
                                            "support_seed": int(support_seed),
                                            "k_shot": int(k_shot),
                                            "group_label": str(group_label),
                                            "dataset_name": dataset_name,
                                            "row_index": int(row_index),
                                            "target_life": float(y_true_i),
                                            "pred_life": float(pred_i),
                                        }
                                    )
                            except Exception as exc:
                                skipped_rows.append(
                                    {
                                        "grouping_scheme": grouping_scheme,
                                        "model": model_spec["model"],
                                        "split_id": int(split_id),
                                        "support_seed": int(support_seed),
                                        "k_shot": int(k_shot),
                                        "group_label": str(group_label),
                                        "dataset_name": dataset_name,
                                        "reason": f"train_or_predict_failed: {exc}",
                                    }
                                )

    metric_df = pd.DataFrame(metric_rows)
    out_dir = _results_root(config) / "major" / "transfer_target_only_baseline"
    _write_outputs(
        out_dir,
        {
            "metric_df": metric_df,
            "summary_df": _summarize_selected_metrics(metric_df, ["grouping_scheme", "model", "model_label", "input_mode", "k_shot"], ["mape", "rmse", "mae", "pearson_r", "spearman_r"]),
            "group_summary_df": _summarize_selected_metrics(metric_df, ["grouping_scheme", "model", "model_label", "input_mode", "k_shot", "group_label", "dataset_name"], ["mape", "rmse", "mae", "pearson_r", "spearman_r"]),
            "prediction_df": pd.DataFrame(prediction_rows),
            "skipped_df": pd.DataFrame(skipped_rows),
        },
        config=config,
        manifest={"experiment": EXPERIMENT_NAME, "branch": "transfer_target_only_baseline", "status": "trained"},
    )


def _run_main_transfer(config: dict[str, Any], aligned: AlignedData, grouping_settings: list[dict[str, Any]]) -> None:
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []

    for grouping_setting in grouping_settings:
        grouping_scheme = grouping_setting["grouping_scheme"]
        group_col = grouping_setting["group_col"]
        split_df = grouping_setting["split_df"]
        group_values = aligned.pipeline_df[group_col].astype(str)
        split_ids = sorted(pd.to_numeric(split_df["split_id"], errors="coerce").dropna().astype(int).unique().tolist())

        for split_id in split_ids:
            split_sub = split_df.loc[split_df["split_id"].eq(split_id)].sort_values("row_index")
            split_train_mask = split_sub["split"].eq("train").to_numpy()
            split_test_mask = split_sub["split"].eq("test").to_numpy()
            split_group_labels = sorted(group_values.loc[split_test_mask].dropna().unique().tolist())
            source_bundle_cache: dict[tuple[str, int, str], dict[str, Any]] = {}

            for group_label in split_group_labels:
                target_train_mask = split_train_mask & group_values.eq(str(group_label)).to_numpy()
                target_test_mask = split_test_mask & group_values.eq(str(group_label)).to_numpy()
                source_mask = split_train_mask & (~group_values.eq(str(group_label)).to_numpy())
                dataset_name = str(aligned.pipeline_df.loc[target_test_mask, COARSE_GROUP_COLUMN].iloc[0])
                support_pool = np.intersect1d(
                    _complete_case_indices(aligned, "cvd_norm", np.where(target_train_mask)[0]),
                    _complete_case_indices(aligned, "cvd_desc_abs", np.where(target_train_mask)[0]),
                )
                test_idx = np.intersect1d(
                    _complete_case_indices(aligned, "cvd_norm", np.where(target_test_mask)[0]),
                    _complete_case_indices(aligned, "cvd_desc_abs", np.where(target_test_mask)[0]),
                )
                source_idx = np.intersect1d(
                    _complete_case_indices(aligned, "cvd_norm", np.where(source_mask)[0]),
                    _complete_case_indices(aligned, "cvd_desc_abs", np.where(source_mask)[0]),
                )

                if len(test_idx) < 3:
                    skipped_rows.append({"grouping_scheme": grouping_scheme, "model": "all_transfer", "split_id": int(split_id), "support_seed": np.nan, "k_shot": np.nan, "group_label": str(group_label), "dataset_name": dataset_name, "reason": "insufficient_shared_test_cases"})
                    continue
                if len(source_idx) < 12:
                    skipped_rows.append({"grouping_scheme": grouping_scheme, "model": "all_transfer", "split_id": int(split_id), "support_seed": np.nan, "k_shot": np.nan, "group_label": str(group_label), "dataset_name": dataset_name, "reason": "insufficient_shared_source_cases"})
                    continue

                for model_spec in MAIN_TRANSFER_MODELS:
                    cache_key = (str(grouping_scheme), int(split_id), str(group_label), str(model_spec["model"]))
                    if cache_key not in source_bundle_cache:
                        if model_spec["kind"] == "neural":
                            source_bundle_cache[cache_key] = _train_source_neural(
                                "spectrumnn" if model_spec["base_model"] == "SpectrumNN" else "cnn",
                                aligned.X_cvd_norm[source_idx],
                                aligned.y[source_idx],
                                seed=568000 + 10000 * int(split_id) + len(str(group_label)),
                            )
                        else:
                            source_bundle_cache[cache_key] = train_ccvnet(
                                aligned.X_cvd_abs[source_idx],
                                aligned.X_value_abs[source_idx],
                                aligned.y[source_idx],
                                descriptor_raw_dim=aligned.descriptor_raw_dim,
                                metadata_raw_dim=aligned.metadata_raw_dim,
                                max_epochs=160,
                                patience=22,
                                batch_size=32,
                                seed=568100 + 10000 * int(split_id) + len(str(group_label)),
                                model_variant="ccv_basic",
                            )
                    source_bundle = source_bundle_cache[cache_key]

                    for support_seed in MODULE5_SUPPORT_SEEDS:
                        for k_shot in MODULE5_K_VALUES:
                            if len(support_pool) < int(k_shot):
                                skipped_rows.append({"grouping_scheme": grouping_scheme, "model": model_spec["model"], "split_id": int(split_id), "support_seed": int(support_seed), "k_shot": int(k_shot), "group_label": str(group_label), "dataset_name": dataset_name, "reason": "insufficient_shared_support"})
                                continue
                            rng = np.random.default_rng(568500 + 1000 * int(split_id) + 100 * int(k_shot) + int(support_seed) + len(str(group_label)))
                            support_idx = np.sort(rng.choice(support_pool, size=int(k_shot), replace=False).astype(int))
                            try:
                                if model_spec["kind"] == "neural":
                                    tuned_bundle, n_trainable = _finetune_neural(source_bundle, aligned.X_cvd_norm[support_idx], aligned.y[support_idx], seed=569000 + int(split_id) + int(support_seed))
                                    pred_test = _predict_neural(tuned_bundle, aligned.X_cvd_norm[test_idx], model_spec["base_model"])
                                    train_mode = "source_pretrain_embedding_regressor_finetune_rankdelta"
                                else:
                                    tuned_bundle, n_trainable = _ccv_transfer_hybrid_finetune_bundle(
                                        source_bundle,
                                        aligned.X_cvd_abs[support_idx],
                                        aligned.X_value_abs[support_idx],
                                        aligned.y[support_idx],
                                        config=CCV_TRANSFER_CONFIG,
                                        seed=569100 + int(split_id) + int(support_seed),
                                    )
                                    pred_test = _ccv_transfer_hybrid_predict_bundle(tuned_bundle, aligned.X_cvd_abs[test_idx], aligned.X_value_abs[test_idx])
                                    train_mode = "source_pretrain_ccv_spec_last2_rankdelta_calib"
                                metric_rows.append(
                                    {
                                        "grouping_scheme": grouping_scheme,
                                        "model": model_spec["model"],
                                        "model_label": model_spec["model_label"],
                                        "base_model": model_spec["base_model"],
                                        "input_mode": model_spec["input_mode"],
                                        "split_id": int(split_id),
                                        "support_seed": int(support_seed),
                                        "k_shot": int(k_shot),
                                        "group_label": str(group_label),
                                        "dataset_name": dataset_name,
                                        "train_mode": train_mode,
                                        "n_source": int(len(source_idx)),
                                        "n_support": int(len(support_idx)),
                                        "n_test": int(len(test_idx)),
                                        "n_trainable": int(n_trainable),
                                        **_regression_metrics(aligned.y[test_idx], pred_test),
                                    }
                                )
                                for row_index, y_true_i, pred_i in zip(test_idx, aligned.y[test_idx], pred_test):
                                    prediction_rows.append(
                                        {
                                            "grouping_scheme": grouping_scheme,
                                            "model": model_spec["model"],
                                            "base_model": model_spec["base_model"],
                                            "input_mode": model_spec["input_mode"],
                                            "split_id": int(split_id),
                                            "support_seed": int(support_seed),
                                            "k_shot": int(k_shot),
                                            "group_label": str(group_label),
                                            "dataset_name": dataset_name,
                                            "row_index": int(row_index),
                                            "target_life": float(y_true_i),
                                            "pred_life": float(pred_i),
                                            "n_trainable": int(n_trainable),
                                        }
                                    )
                            except Exception as exc:
                                skipped_rows.append({"grouping_scheme": grouping_scheme, "model": model_spec["model"], "split_id": int(split_id), "support_seed": int(support_seed), "k_shot": int(k_shot), "group_label": str(group_label), "dataset_name": dataset_name, "reason": str(exc)})

    metric_df = pd.DataFrame(metric_rows)
    out_dir = _results_root(config) / "major" / "transfer"
    _write_outputs(
        out_dir,
        {
            "metric_df": metric_df,
            "summary_df": _summarize_selected_metrics(metric_df, ["grouping_scheme", "model", "model_label", "base_model", "input_mode", "k_shot"], ["mape", "rmse", "mae", "pearson_r", "spearman_r", "n_trainable"]),
            "group_summary_df": _summarize_selected_metrics(metric_df, ["grouping_scheme", "group_label", "dataset_name", "model", "model_label", "base_model", "input_mode", "k_shot"], ["mape", "rmse", "mae", "pearson_r", "spearman_r", "n_trainable"]),
            "prediction_df": pd.DataFrame(prediction_rows),
            "skipped_df": pd.DataFrame(skipped_rows),
        },
        config=config,
        manifest={"experiment": EXPERIMENT_NAME, "branch": "transfer", "status": "trained"},
    )


def _run_ablation_transfer(config: dict[str, Any], aligned: AlignedData, grouping_settings: list[dict[str, Any]]) -> None:
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    desc_matrix = np.asarray(aligned.X_value_abs[:, : aligned.descriptor_raw_dim], dtype=np.float32)
    cvd_norm_matrix = np.asarray(aligned.X_cvd_norm, dtype=np.float32)
    cvd_abs_matrix = np.asarray(aligned.X_cvd_abs, dtype=np.float32)

    for grouping_setting in grouping_settings:
        grouping_scheme = grouping_setting["grouping_scheme"]
        group_col = grouping_setting["group_col"]
        split_df = grouping_setting["split_df"]
        group_values = aligned.pipeline_df[group_col].astype(str)
        split_ids = sorted(pd.to_numeric(split_df["split_id"], errors="coerce").dropna().astype(int).unique().tolist())

        for split_id in split_ids:
            split_sub = split_df.loc[split_df["split_id"].eq(split_id)].sort_values("row_index")
            split_train_mask = split_sub["split"].eq("train").to_numpy()
            split_test_mask = split_sub["split"].eq("test").to_numpy()
            split_group_labels = sorted(group_values.loc[split_test_mask].dropna().unique().tolist())
            source_bundle_cache: dict[str, dict[str, Any]] = {}
            source_desc_idx_cache: dict[str, np.ndarray] = {}
            source_ccv_meta_cache: dict[str, dict[str, Any]] = {}

            for group_label in split_group_labels:
                target_train_mask = split_train_mask & group_values.eq(str(group_label)).to_numpy()
                target_test_mask = split_test_mask & group_values.eq(str(group_label)).to_numpy()
                dataset_name = str(aligned.pipeline_df.loc[target_test_mask, COARSE_GROUP_COLUMN].iloc[0])
                support_pool_joint = np.intersect1d(
                    _complete_case_indices(aligned, "descriptor_abs", np.where(target_train_mask)[0]),
                    _complete_case_indices(aligned, "cvd_norm", np.where(target_train_mask)[0]),
                )
                test_idx = np.intersect1d(
                    _complete_case_indices(aligned, "descriptor_abs", np.where(target_test_mask)[0]),
                    _complete_case_indices(aligned, "cvd_norm", np.where(target_test_mask)[0]),
                )
                source_mask = split_train_mask & (~group_values.eq(str(group_label)).to_numpy())
                source_desc_idx = _complete_case_indices(aligned, "descriptor_abs", np.where(source_mask)[0])
                source_joint_idx = np.intersect1d(
                    _complete_case_indices(aligned, "cvd_norm", np.where(source_mask)[0]),
                    _complete_case_indices(aligned, "cvd_desc_abs", np.where(source_mask)[0]),
                )

                if len(test_idx) < 3:
                    skipped_rows.append({"grouping_scheme": grouping_scheme, "method": "all", "split_id": int(split_id), "support_seed": np.nan, "k_shot": np.nan, "group_label": str(group_label), "dataset_name": dataset_name, "reason": "insufficient_joint_test_cases"})
                    continue

                if str(group_label) not in source_bundle_cache:
                    source_bundle_cache[str(group_label)] = _train_source_bundle(aligned, group_col, source_mask)
                    source_desc_idx_cache[str(group_label)] = source_desc_idx.copy()
                    if len(source_joint_idx) >= 12:
                        source_ccv_meta_cache[str(group_label)] = train_ccvnet(
                            aligned.X_cvd_abs[source_joint_idx],
                            aligned.X_value_abs[source_joint_idx],
                            aligned.y[source_joint_idx],
                            descriptor_raw_dim=aligned.descriptor_raw_dim,
                            metadata_raw_dim=aligned.metadata_raw_dim,
                            max_epochs=160,
                            patience=22,
                            batch_size=32,
                            seed=550000 + int(split_id) + len(str(group_label)),
                            model_variant="ccv_basic",
                        )

                for support_seed in MODULE5_SUPPORT_SEEDS:
                    for k_shot in MODULE5_K_VALUES:
                        if len(support_pool_joint) < int(k_shot):
                            skipped_rows.append({"grouping_scheme": grouping_scheme, "method": "all", "split_id": int(split_id), "support_seed": int(support_seed), "k_shot": int(k_shot), "group_label": str(group_label), "dataset_name": dataset_name, "reason": "insufficient_joint_support"})
                            continue
                        rng = np.random.default_rng(310000 + 1000 * int(split_id) + 100 * int(k_shot) + int(support_seed) + len(str(group_label)))
                        support_idx = np.sort(rng.choice(support_pool_joint, size=int(k_shot), replace=False).astype(int))
                        anchor_idx, anchor_life = _select_anchor(aligned.y, support_idx)
                        support_non_anchor = support_idx[support_idx != anchor_idx]
                        anchor_desc = desc_matrix[anchor_idx : anchor_idx + 1]
                        anchor_cvd_norm = cvd_norm_matrix[anchor_idx : anchor_idx + 1]
                        anchor_cvd_abs = cvd_abs_matrix[anchor_idx : anchor_idx + 1]
                        X_support_desc = _pair_feature_transform(desc_matrix[support_non_anchor], anchor_desc) if len(support_non_anchor) else np.zeros((0, desc_matrix.shape[1]), dtype=np.float32)
                        X_test_desc = _pair_feature_transform(desc_matrix[test_idx], anchor_desc)
                        y_support_delta = (aligned.y[support_non_anchor] - anchor_life).astype(np.float32) if len(support_non_anchor) else np.zeros(0, dtype=np.float32)

                        if len(support_non_anchor) == 0:
                            desc_local_test_delta = np.zeros(len(test_idx), dtype=np.float32)
                            desc_local_support_pred = np.zeros(0, dtype=np.float32)
                            local_rel = np.nan
                        elif len(support_non_anchor) == 1:
                            desc_local_test_delta = _local_kernel_predict(X_support_desc, y_support_delta, X_test_desc)
                            desc_local_support_pred = y_support_delta.copy()
                            local_rel = np.nan
                        else:
                            desc_local_test_delta = _local_kernel_predict(X_support_desc, y_support_delta, X_test_desc)
                            desc_local_support_pred = _local_leave_one_out_predict(X_support_desc, y_support_delta)
                            local_rel = _reliability_score(y_support_delta, desc_local_support_pred)

                        for ablation_spec in ABLATION_MODELS:
                            method = ablation_spec["method"]
                            method_label = ablation_spec["method_label"]
                            if method == "descriptor_local":
                                pred_delta = desc_local_test_delta
                                pred_life = pred_delta + anchor_life
                                y_true = aligned.y[test_idx]
                                metric_values = _regression_metrics(y_true, pred_life)
                                metric_values["within_spearman"] = _corr(y_true - anchor_life, pred_delta, method="spearman")
                                metric_rows.append(
                                    {
                                        "grouping_scheme": grouping_scheme,
                                        "method": method,
                                        "method_label": method_label,
                                        "split_id": int(split_id),
                                        "support_seed": int(support_seed),
                                        "k_shot": int(k_shot),
                                        "group_label": str(group_label),
                                        "dataset_name": dataset_name,
                                        "n_support": int(k_shot),
                                        "n_test": int(len(test_idx)),
                                        "retrieved_source_group": np.nan,
                                        "group_life_error": abs(float(np.nanmedian(pred_life)) - float(np.nanmedian(y_true))),
                                        "source_local_gap": np.nan,
                                        "beta": np.nan,
                                        "model": method,
                                        "model_label": method_label,
                                        "alpha": np.nan,
                                        "alpha_cap": np.nan,
                                        "gap_value": np.nan,
                                        "gap_factor": np.nan,
                                        "local_rel": float(local_rel) if np.isfinite(local_rel) else np.nan,
                                        "source_rel": np.nan,
                                        "residual_rel": np.nan,
                                        "base_gap": np.nan,
                                        "n_trainable": np.nan,
                                        **metric_values,
                                    }
                                )
                                for row_index, y_true_i, pred_i in zip(test_idx, y_true, pred_life):
                                    prediction_rows.append(
                                        {
                                            "grouping_scheme": grouping_scheme,
                                            "method": method,
                                            "split_id": int(split_id),
                                            "support_seed": int(support_seed),
                                            "k_shot": int(k_shot),
                                            "group_label": str(group_label),
                                            "dataset_name": dataset_name,
                                            "row_index": int(row_index),
                                            "target_life": float(y_true_i),
                                            "pred_life": float(pred_i),
                                            "retrieved_source_group": np.nan,
                                            "model": method,
                                            "model_label": method_label,
                                            "method_label": method_label,
                                            "pred_base_life": float(pred_i),
                                            "pred_cvd_residual": 0.0,
                                            "alpha": np.nan,
                                            "gap_value": np.nan,
                                            "n_trainable": np.nan,
                                        }
                                    )
                                continue

                            if method == "ccv_transfer_meta":
                                if str(group_label) not in source_ccv_meta_cache:
                                    skipped_rows.append({"grouping_scheme": grouping_scheme, "method": method, "split_id": int(split_id), "support_seed": int(support_seed), "k_shot": int(k_shot), "group_label": str(group_label), "dataset_name": dataset_name, "reason": "insufficient_shared_source_cases"})
                                    continue
                                try:
                                    tuned_bundle, n_trainable = _ccv_transfer_meta_finetune_bundle(
                                        source_ccv_meta_cache[str(group_label)],
                                        aligned.X_cvd_abs[support_idx],
                                        aligned.X_value_abs[support_idx],
                                        aligned.y[support_idx],
                                        variant="meta_head_only",
                                        seed=560000 + int(split_id) + int(support_seed),
                                    )
                                    pred_life, _, _ = predict_ccvnet(tuned_bundle, aligned.X_cvd_abs[test_idx], aligned.X_value_abs[test_idx])
                                    pred_life = np.asarray(pred_life, dtype=np.float32).reshape(-1)
                                    y_true = aligned.y[test_idx]
                                    metric_values = _regression_metrics(y_true, pred_life)
                                    metric_rows.append(
                                        {
                                            "grouping_scheme": grouping_scheme,
                                            "method": method,
                                            "method_label": method_label,
                                            "split_id": int(split_id),
                                            "support_seed": int(support_seed),
                                            "k_shot": int(k_shot),
                                            "group_label": str(group_label),
                                            "dataset_name": dataset_name,
                                            "n_support": int(k_shot),
                                            "n_test": int(len(test_idx)),
                                            "retrieved_source_group": str(group_label),
                                            "group_life_error": abs(float(np.nanmedian(pred_life)) - float(np.nanmedian(y_true))),
                                            "source_local_gap": np.nan,
                                            "beta": np.nan,
                                            "model": method,
                                            "model_label": method_label,
                                            "alpha": np.nan,
                                            "alpha_cap": np.nan,
                                            "gap_value": np.nan,
                                            "gap_factor": np.nan,
                                            "local_rel": np.nan,
                                            "source_rel": np.nan,
                                            "residual_rel": np.nan,
                                            "base_gap": np.nan,
                                            "n_trainable": int(n_trainable),
                                            "within_spearman": _corr(y_true, pred_life, method="spearman"),
                                            **metric_values,
                                        }
                                    )
                                    for row_index, y_true_i, pred_i in zip(test_idx, y_true, pred_life):
                                        prediction_rows.append(
                                            {
                                                "grouping_scheme": grouping_scheme,
                                                "method": method,
                                                "split_id": int(split_id),
                                                "support_seed": int(support_seed),
                                                "k_shot": int(k_shot),
                                                "group_label": str(group_label),
                                                "dataset_name": dataset_name,
                                                "row_index": int(row_index),
                                                "target_life": float(y_true_i),
                                                "pred_life": float(pred_i),
                                                "retrieved_source_group": str(group_label),
                                                "model": method,
                                                "model_label": method_label,
                                                "method_label": method_label,
                                                "pred_base_life": float(pred_i),
                                                "pred_cvd_residual": 0.0,
                                                "alpha": np.nan,
                                                "gap_value": np.nan,
                                                "n_trainable": int(n_trainable),
                                            }
                                        )
                                except Exception as exc:
                                    skipped_rows.append({"grouping_scheme": grouping_scheme, "method": method, "split_id": int(split_id), "support_seed": int(support_seed), "k_shot": int(k_shot), "group_label": str(group_label), "dataset_name": dataset_name, "reason": str(exc)})
                                continue

                            source_bundle = source_bundle_cache[str(group_label)]
                            X_support_cvd_abs = _pair_feature_transform(cvd_abs_matrix[support_non_anchor], anchor_cvd_abs) if len(support_non_anchor) else np.zeros((0, cvd_abs_matrix.shape[1] * cvd_abs_matrix.shape[2]), dtype=np.float32)
                            X_test_cvd_abs = _pair_feature_transform(cvd_abs_matrix[test_idx], anchor_cvd_abs)
                            source_support_raw = _predict_source_bundle(source_bundle, X_support_desc, X_support_cvd_abs) if len(support_non_anchor) else np.zeros(0, dtype=np.float32)
                            source_test_raw = _predict_source_bundle(source_bundle, X_test_desc, X_test_cvd_abs)
                            slope_value, intercept_value = _affine_calibration(source_support_raw, y_support_delta)
                            source_support_aligned = slope_value * source_support_raw + intercept_value if len(source_support_raw) else np.zeros(0, dtype=np.float32)
                            source_test_aligned = slope_value * source_test_raw + intercept_value
                            source_rel = _reliability_score(y_support_delta, source_support_aligned) if len(source_support_aligned) else np.nan
                            gap_value, gap_factor = _descriptor_gap_factor(desc_matrix[source_desc_idx_cache[str(group_label)]], desc_matrix[support_idx])
                            alpha_cap = _source_alpha_cap(int(k_shot))
                            if len(support_non_anchor) == 0:
                                alpha_value = 0.0
                            else:
                                rel_source = max(0.0, float(source_rel)) if np.isfinite(source_rel) else 0.0
                                rel_local = max(0.0, float(local_rel)) if np.isfinite(local_rel) else 0.0
                                rel_ratio = rel_source / (rel_source + rel_local + 1e-6)
                                alpha_value = float(np.clip(alpha_cap * gap_factor * rel_ratio, 0.0, alpha_cap))
                            desc_base_test_delta = (1.0 - alpha_value) * desc_local_test_delta + alpha_value * source_test_aligned
                            desc_base_support_delta = (
                                (1.0 - alpha_value) * desc_local_support_pred + alpha_value * source_support_aligned if len(support_non_anchor) else np.zeros(0, dtype=np.float32)
                            )
                            X_support_cvd_norm = _pair_feature_transform(cvd_norm_matrix[support_non_anchor], anchor_cvd_norm) if len(support_non_anchor) else np.zeros((0, cvd_norm_matrix.shape[1] * cvd_norm_matrix.shape[2]), dtype=np.float32)
                            X_test_cvd_norm = _pair_feature_transform(cvd_norm_matrix[test_idx], anchor_cvd_norm)
                            residual_target = y_support_delta - desc_base_support_delta if len(support_non_anchor) else np.zeros(0, dtype=np.float32)
                            if len(support_non_anchor) <= 1:
                                residual_test_pred = np.zeros(len(test_idx), dtype=np.float32)
                                residual_rel = np.nan
                            else:
                                residual_test_pred = _local_kernel_predict(X_support_cvd_norm, residual_target, X_test_cvd_norm)
                                residual_rel = _reliability_score(residual_target, _local_leave_one_out_predict(X_support_cvd_norm, residual_target))
                            pred_delta = desc_base_test_delta + residual_test_pred
                            pred_life = pred_delta + anchor_life
                            y_true = aligned.y[test_idx]
                            metric_values = _regression_metrics(y_true, pred_life)
                            metric_values["within_spearman"] = _corr(y_true - anchor_life, pred_delta, method="spearman")
                            metric_rows.append(
                                {
                                    "grouping_scheme": grouping_scheme,
                                    "method": method,
                                    "method_label": method_label,
                                    "split_id": int(split_id),
                                    "support_seed": int(support_seed),
                                    "k_shot": int(k_shot),
                                    "group_label": str(group_label),
                                    "dataset_name": dataset_name,
                                    "n_support": int(k_shot),
                                    "n_test": int(len(test_idx)),
                                    "retrieved_source_group": str(group_label),
                                    "group_life_error": abs(float(np.nanmedian(pred_life)) - float(np.nanmedian(y_true))),
                                    "source_local_gap": float(np.nanmean(np.abs(source_test_aligned - desc_local_test_delta))) if len(source_test_aligned) else np.nan,
                                    "beta": np.nan,
                                    "model": method,
                                    "model_label": method_label,
                                    "alpha": float(alpha_value),
                                    "alpha_cap": float(alpha_cap),
                                    "gap_value": float(gap_value) if np.isfinite(gap_value) else np.nan,
                                    "gap_factor": float(gap_factor),
                                    "local_rel": float(local_rel) if np.isfinite(local_rel) else np.nan,
                                    "source_rel": float(source_rel) if np.isfinite(source_rel) else np.nan,
                                    "residual_rel": float(residual_rel) if np.isfinite(residual_rel) else np.nan,
                                    "base_gap": float(np.nanmean(np.abs(source_test_aligned - desc_local_test_delta))) if len(source_test_aligned) else np.nan,
                                    "n_trainable": np.nan,
                                    **metric_values,
                                }
                            )
                            for row_index, y_true_i, pred_i, base_i, res_i in zip(test_idx, y_true, pred_life, desc_base_test_delta + anchor_life, residual_test_pred):
                                prediction_rows.append(
                                    {
                                        "grouping_scheme": grouping_scheme,
                                        "method": method,
                                        "split_id": int(split_id),
                                        "support_seed": int(support_seed),
                                        "k_shot": int(k_shot),
                                        "group_label": str(group_label),
                                        "dataset_name": dataset_name,
                                        "row_index": int(row_index),
                                        "target_life": float(y_true_i),
                                        "pred_life": float(pred_i),
                                        "retrieved_source_group": str(group_label),
                                        "model": method,
                                        "model_label": method_label,
                                        "method_label": method_label,
                                        "pred_base_life": float(base_i),
                                        "pred_cvd_residual": float(res_i),
                                        "alpha": float(alpha_value),
                                        "gap_value": float(gap_value) if np.isfinite(gap_value) else np.nan,
                                        "n_trainable": np.nan,
                                    }
                                )

    metric_df = pd.DataFrame(metric_rows)
    summary_df = _summarize_selected_metrics(
        metric_df,
        ["grouping_scheme", "method", "method_label", "k_shot"],
        [
            "mape",
            "rmse",
            "mae",
            "pearson_r",
            "spearman_r",
            "within_spearman",
            "group_life_error",
            "source_local_gap",
            "beta",
            "alpha",
            "gap_value",
            "gap_factor",
            "local_rel",
            "source_rel",
            "residual_rel",
            "base_gap",
            "n_trainable",
        ],
    )
    if not summary_df.empty:
        summary_df["model"] = summary_df["method"]
        summary_df["model_label"] = summary_df["method_label"]
    group_summary_df = _summarize_selected_metrics(
        metric_df,
        ["grouping_scheme", "group_label", "dataset_name", "method", "method_label", "k_shot"],
        [
            "mape",
            "rmse",
            "mae",
            "pearson_r",
            "spearman_r",
            "within_spearman",
            "group_life_error",
            "source_local_gap",
            "beta",
            "alpha",
            "gap_value",
            "gap_factor",
            "local_rel",
            "source_rel",
            "residual_rel",
            "base_gap",
            "n_trainable",
        ],
    )
    if not group_summary_df.empty:
        group_summary_df["model"] = group_summary_df["method"]
        group_summary_df["model_label"] = group_summary_df["method_label"]
    out_dir = _results_root(config) / "ablation" / "transfer"
    _write_outputs(
        out_dir,
        {
            "metric_df": metric_df,
            "summary_df": summary_df,
            "group_summary_df": group_summary_df,
            "prediction_df": pd.DataFrame(prediction_rows),
            "skipped_df": pd.DataFrame(skipped_rows),
        },
        config=config,
        manifest={"experiment": EXPERIMENT_NAME, "branch": "ablation_transfer", "status": "trained"},
    )


def _run_transfer_diagnose(config: dict[str, Any], aligned: AlignedData, grouping_settings: list[dict[str, Any]]) -> None:
    epoch_rows: list[dict[str, Any]] = []
    shift_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []

    for grouping_setting in grouping_settings:
        grouping_scheme = grouping_setting["grouping_scheme"]
        group_col = grouping_setting["group_col"]
        split_df = grouping_setting["split_df"]
        group_values = aligned.pipeline_df[group_col].astype(str)
        split_ids = sorted(pd.to_numeric(split_df["split_id"], errors="coerce").dropna().astype(int).unique().tolist())
        if not split_ids:
            continue
        rep_split_id = int(split_ids[0])
        split_sub = split_df.loc[split_df["split_id"].eq(rep_split_id)].sort_values("row_index")
        split_train_mask = split_sub["split"].eq("train").to_numpy()
        split_test_mask = split_sub["split"].eq("test").to_numpy()
        split_group_labels = sorted(group_values.loc[split_test_mask].dropna().unique().tolist())
        source_bundle_cache: dict[tuple[str, int], dict[str, Any]] = {}

        for group_label in split_group_labels:
            target_train_mask = split_train_mask & group_values.eq(str(group_label)).to_numpy()
            target_test_mask = split_test_mask & group_values.eq(str(group_label)).to_numpy()
            source_mask = split_train_mask & (~group_values.eq(str(group_label)).to_numpy())
            dataset_name = str(aligned.pipeline_df.loc[target_test_mask, COARSE_GROUP_COLUMN].iloc[0])
            support_pool = np.intersect1d(
                _complete_case_indices(aligned, "cvd_norm", np.where(target_train_mask)[0]),
                _complete_case_indices(aligned, "cvd_desc_abs", np.where(target_train_mask)[0]),
            )
            test_idx = np.intersect1d(
                _complete_case_indices(aligned, "cvd_norm", np.where(target_test_mask)[0]),
                _complete_case_indices(aligned, "cvd_desc_abs", np.where(target_test_mask)[0]),
            )
            source_idx = np.intersect1d(
                _complete_case_indices(aligned, "cvd_norm", np.where(source_mask)[0]),
                _complete_case_indices(aligned, "cvd_desc_abs", np.where(source_mask)[0]),
            )
            if len(test_idx) < 3:
                skipped_rows.append({"grouping_scheme": grouping_scheme, "split_id": rep_split_id, "support_seed": np.nan, "k_shot": np.nan, "group_label": str(group_label), "dataset_name": dataset_name, "reason": "insufficient_shared_test_cases"})
                continue
            if len(source_idx) < 12:
                skipped_rows.append({"grouping_scheme": grouping_scheme, "split_id": rep_split_id, "support_seed": np.nan, "k_shot": np.nan, "group_label": str(group_label), "dataset_name": dataset_name, "reason": "insufficient_shared_source_cases"})
                continue
            cache_key = (str(group_label), int(rep_split_id))
            if cache_key not in source_bundle_cache:
                source_bundle_cache[cache_key] = train_ccvnet(
                    aligned.X_cvd_abs[source_idx],
                    aligned.X_value_abs[source_idx],
                    aligned.y[source_idx],
                    descriptor_raw_dim=aligned.descriptor_raw_dim,
                    metadata_raw_dim=aligned.metadata_raw_dim,
                    max_epochs=160,
                    patience=22,
                    batch_size=32,
                    seed=569000 + 1000 * int(rep_split_id) + len(str(group_label)),
                    model_variant="ccv_basic",
                )
            source_bundle = source_bundle_cache[cache_key]

            for support_seed in MODULE5_SUPPORT_SEEDS:
                for k_shot in MODULE5_K_VALUES:
                    if len(support_pool) < int(k_shot):
                        skipped_rows.append({"grouping_scheme": grouping_scheme, "split_id": rep_split_id, "support_seed": int(support_seed), "k_shot": int(k_shot), "group_label": str(group_label), "dataset_name": dataset_name, "reason": "insufficient_shared_support"})
                        continue
                    rng = np.random.default_rng(569500 + 1000 * int(rep_split_id) + 100 * int(k_shot) + int(support_seed) + len(str(group_label)))
                    support_idx = np.sort(rng.choice(support_pool, size=int(k_shot), replace=False).astype(int))
                    y_support_true = aligned.y[support_idx].astype(np.float32)
                    y_test_true = aligned.y[test_idx].astype(np.float32)
                    model = copy.deepcopy(source_bundle["model"]).to(DEVICE)
                    n_trainable = _set_ccv_trainable_layers(model, "spec_last2")
                    residual_scale = float(CCV_TRANSFER_CONFIG["residual_scale"])
                    X_cvd_support_scaled, X_desc_support_scaled, X_meta_support_scaled = _ccv_transform_inputs(source_bundle, aligned.X_cvd_abs[support_idx], aligned.X_value_abs[support_idx])
                    X_cvd_test_scaled, X_desc_test_scaled, X_meta_test_scaled = _ccv_transform_inputs(source_bundle, aligned.X_cvd_abs[test_idx], aligned.X_value_abs[test_idx])
                    y_support_scaled = ((y_support_true - float(source_bundle["y_mean"])) / float(source_bundle["y_std"])).astype(np.float32)
                    loader = make_tensor_loader(X_cvd_support_scaled, X_desc_support_scaled, X_meta_support_scaled, y_support_scaled, batch_size=min(16, max(1, len(y_support_scaled))), shuffle=True)
                    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4, weight_decay=1e-4)
                    loss_fn = nn.SmoothL1Loss()
                    n_epochs = _ccv_transfer_finetune_epochs(len(y_support_true))

                    def eval_epoch(epoch_num: int) -> tuple[dict[str, Any], np.ndarray]:
                        model.eval()
                        with torch.no_grad():
                            xb_sup_cvd = torch.as_tensor(X_cvd_support_scaled, dtype=torch.float32, device=DEVICE)
                            xb_sup_desc = torch.as_tensor(X_desc_support_scaled, dtype=torch.float32, device=DEVICE)
                            xb_sup_meta = torch.as_tensor(X_meta_support_scaled, dtype=torch.float32, device=DEVICE)
                            yb_sup = torch.as_tensor(y_support_scaled, dtype=torch.float32, device=DEVICE)
                            pred_sup_scaled, _, delta_sup = _predict_ccv_scaled(model, residual_scale, xb_sup_cvd, xb_sup_desc, xb_sup_meta)
                            abs_loss = loss_fn(pred_sup_scaled, yb_sup)
                            delta_loss, rank_loss = _pair_losses(pred_sup_scaled, yb_sup)
                            total_loss = abs_loss + float(CCV_TRANSFER_CONFIG["lambda_delta"]) * delta_loss + float(CCV_TRANSFER_CONFIG["lambda_rank"]) * rank_loss + float(CCV_TRANSFER_CONFIG["lambda_delta_l2"]) * torch.mean(delta_sup**2) + float(CCV_TRANSFER_CONFIG["lambda_center"]) * (torch.mean(delta_sup) ** 2)
                            xb_test_cvd = torch.as_tensor(X_cvd_test_scaled, dtype=torch.float32, device=DEVICE)
                            xb_test_desc = torch.as_tensor(X_desc_test_scaled, dtype=torch.float32, device=DEVICE)
                            xb_test_meta = torch.as_tensor(X_meta_test_scaled, dtype=torch.float32, device=DEVICE)
                            pred_test_scaled, _, _ = _predict_ccv_scaled(model, residual_scale, xb_test_cvd, xb_test_desc, xb_test_meta)
                        pred_support = pred_sup_scaled.detach().cpu().numpy().reshape(-1) * float(source_bundle["y_std"]) + float(source_bundle["y_mean"])
                        pred_test = pred_test_scaled.detach().cpu().numpy().reshape(-1) * float(source_bundle["y_std"]) + float(source_bundle["y_mean"])
                        support_metrics = _regression_metrics(y_support_true, pred_support)
                        test_metrics = _regression_metrics(y_test_true, pred_test)
                        row = {
                            "grouping_scheme": grouping_scheme,
                            "method": "CCV-transfer",
                            "method_label": "CCV-transfer",
                            "split_id": int(rep_split_id),
                            "support_seed": int(support_seed),
                            "k_shot": int(k_shot),
                            "group_label": str(group_label),
                            "dataset_name": dataset_name,
                            "epoch": int(epoch_num),
                            "n_support": int(len(support_idx)),
                            "n_test": int(len(test_idx)),
                            "n_trainable": int(n_trainable),
                            "support_loss_total": float(total_loss.detach().cpu().item()),
                            "support_loss_abs": float(abs_loss.detach().cpu().item()),
                            "support_loss_delta": float((float(CCV_TRANSFER_CONFIG["lambda_delta"]) * delta_loss).detach().cpu().item()),
                            "support_loss_rank": float((float(CCV_TRANSFER_CONFIG["lambda_rank"]) * rank_loss).detach().cpu().item()),
                            "support_loss_delta_l2": float((float(CCV_TRANSFER_CONFIG["lambda_delta_l2"]) * torch.mean(delta_sup**2)).detach().cpu().item()),
                            "support_loss_center": float((float(CCV_TRANSFER_CONFIG["lambda_center"]) * (torch.mean(delta_sup) ** 2)).detach().cpu().item()),
                            "support_mae": float(support_metrics["mae"]),
                            "support_mape": float(support_metrics["mape"]),
                            "test_mape": float(test_metrics["mape"]),
                            "test_rmse": float(test_metrics["rmse"]),
                            "test_mae": float(test_metrics["mae"]),
                            "test_pearson_r": float(test_metrics["pearson_r"]) if np.isfinite(test_metrics["pearson_r"]) else np.nan,
                            "test_spearman_r": float(test_metrics["spearman_r"]) if np.isfinite(test_metrics["spearman_r"]) else np.nan,
                        }
                        return row, pred_test.astype(np.float32)

                    zero_row, pred_test_before = eval_epoch(0)
                    epoch_rows.append(zero_row)
                    for epoch_num in range(1, int(n_epochs) + 1):
                        model.train()
                        for xb_cvd, xb_desc, xb_meta, yb in loader:
                            xb_cvd = xb_cvd.to(DEVICE)
                            xb_desc = xb_desc.to(DEVICE)
                            xb_meta = xb_meta.to(DEVICE)
                            yb = yb.to(DEVICE)
                            opt.zero_grad()
                            pred_scaled, _, delta = _predict_ccv_scaled(model, residual_scale, xb_cvd, xb_desc, xb_meta)
                            abs_loss = loss_fn(pred_scaled, yb)
                            delta_loss, rank_loss = _pair_losses(pred_scaled, yb)
                            loss = abs_loss + float(CCV_TRANSFER_CONFIG["lambda_delta"]) * delta_loss + float(CCV_TRANSFER_CONFIG["lambda_rank"]) * rank_loss + float(CCV_TRANSFER_CONFIG["lambda_delta_l2"]) * torch.mean(delta**2) + float(CCV_TRANSFER_CONFIG["lambda_center"]) * (torch.mean(delta) ** 2)
                            loss.backward()
                            opt.step()
                        row, pred_test_after = eval_epoch(epoch_num)
                        epoch_rows.append(row)
                    for row_index, y_true_i, pred_before_i, pred_after_i in zip(test_idx, y_test_true, pred_test_before, pred_test_after):
                        shift_rows.append({"grouping_scheme": grouping_scheme, "method": "CCV-transfer", "method_label": "CCV-transfer", "split_id": int(rep_split_id), "support_seed": int(support_seed), "k_shot": int(k_shot), "group_label": str(group_label), "dataset_name": dataset_name, "row_index": int(row_index), "target_life": float(y_true_i), "pred_before": float(pred_before_i), "pred_after": float(pred_after_i), "pred_shift": float(pred_after_i - pred_before_i), "abs_shift": float(abs(pred_after_i - pred_before_i))})

    epoch_df = pd.DataFrame(epoch_rows)
    shift_df = pd.DataFrame(shift_rows)
    if epoch_df.empty:
        epoch_summary_df = pd.DataFrame()
        final_metric_df = pd.DataFrame()
        final_summary_df = pd.DataFrame()
    else:
        epoch_summary_df = _summarize_selected_metrics(epoch_df, ["grouping_scheme", "k_shot", "epoch"], ["support_loss_total", "support_loss_abs", "support_loss_delta", "support_loss_rank", "support_mae", "support_mape", "test_mape", "test_rmse", "test_mae", "test_pearson_r", "test_spearman_r"])
        zero_df = epoch_df.loc[epoch_df["epoch"].eq(0), ["grouping_scheme", "split_id", "support_seed", "k_shot", "group_label", "dataset_name", "test_mape", "test_pearson_r"]].rename(columns={"test_mape": "zero_shot_mape", "test_pearson_r": "zero_shot_pearson_r"})
        last_epoch_df = epoch_df.sort_values("epoch").groupby(["grouping_scheme", "split_id", "support_seed", "k_shot", "group_label", "dataset_name"], dropna=False).tail(1)
        final_metric_df = last_epoch_df.merge(zero_df, on=["grouping_scheme", "split_id", "support_seed", "k_shot", "group_label", "dataset_name"], how="left")
        final_metric_df = final_metric_df.rename(columns={"test_mape": "final_mape", "test_rmse": "final_rmse", "test_mae": "final_mae", "test_pearson_r": "final_pearson_r", "test_spearman_r": "final_spearman_r", "support_mae": "support_final_mae", "support_mape": "support_final_mape"})
        final_metric_df["method"] = "CCV-transfer"
        final_metric_df["method_label"] = "CCV-transfer"
        final_metric_df = final_metric_df[["grouping_scheme", "method", "method_label", "split_id", "support_seed", "k_shot", "group_label", "dataset_name", "n_support", "n_test", "n_trainable", "zero_shot_mape", "zero_shot_pearson_r", "final_mape", "final_rmse", "final_mae", "final_pearson_r", "final_spearman_r", "support_final_mae", "support_final_mape"]]
        final_summary_df = _summarize_selected_metrics(final_metric_df, ["grouping_scheme", "k_shot"], ["zero_shot_mape", "zero_shot_pearson_r", "final_mape", "final_rmse", "final_mae", "final_pearson_r", "final_spearman_r", "support_final_mae", "support_final_mape", "n_trainable"])

    out_dir = _results_root(config) / "major" / "transfer_diagnose"
    _write_outputs(
        out_dir,
        {
            "epoch_df": epoch_df,
            "epoch_summary_df": epoch_summary_df,
            "final_metric_df": final_metric_df,
            "final_summary_df": final_summary_df,
            "shift_df": shift_df,
            "skipped_df": pd.DataFrame(skipped_rows),
        },
        config=config,
        manifest={"experiment": EXPERIMENT_NAME, "branch": "transfer_diagnose", "status": "trained"},
    )


def model_plan(config: dict[str, Any]) -> list[dict[str, Any]]:
    experiment_cfg = config.get("experiment", {})
    return list(experiment_cfg.get("models", []))


def run(config: dict[str, Any]) -> None:
    run_major(config)


def run_major(config: dict[str, Any]) -> None:
    aligned = build_aligned_data(config)
    grouping_settings = _build_grouping_settings(aligned, config)
    _run_zero_shot(config, aligned, grouping_settings)
    _run_target_only(config, aligned, grouping_settings)
    _run_main_transfer(config, aligned, grouping_settings)
    _run_transfer_diagnose(config, aligned, grouping_settings)
    print("CCVNet publish transfer caches refreshed under results/major.")


def run_ablation(config: dict[str, Any]) -> None:
    aligned = build_aligned_data(config)
    grouping_settings = _build_grouping_settings(aligned, config)
    _run_ablation_transfer(config, aligned, grouping_settings)
    print("CCVNet publish transfer ablation cache refreshed under results/ablation/transfer.")


def run_all(config: dict[str, Any]) -> None:
    run_major(config)
    run_ablation(config)
