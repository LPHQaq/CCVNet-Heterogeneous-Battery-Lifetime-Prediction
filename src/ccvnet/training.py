from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ccvnet.models.cnn import CNNBaseline
from ccvnet.models.ccvnet import ccvnet
from ccvnet.models.mlp import MLPBaseline, tabular_missing_aware_matrix
from ccvnet.models.spectrumnn import SpectrumNN
from ccvnet.models.spectrum_meta import SpectrumMetaRegressor


def resolve_device(device: str = "auto") -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def fit_cvd_scaler(X_train: np.ndarray) -> dict:
    mean = np.nanmean(X_train, axis=(0, 2), keepdims=True)
    std = np.nanstd(X_train, axis=(0, 2), keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


def transform_cvd(X: np.ndarray, scaler: dict) -> np.ndarray:
    transformed = (X - scaler["mean"]) / scaler["std"]
    return np.nan_to_num(transformed, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def append_cvd_availability_mask(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    channel_available = np.any(np.isfinite(X), axis=2, keepdims=True).astype(np.float32)
    availability_mask = np.broadcast_to(channel_available, X.shape).copy()
    X_filled = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return np.concatenate([X_filled, availability_mask], axis=1)


def split_descriptor_metadata(
    X_value: np.ndarray,
    descriptor_raw_dim: int,
    metadata_raw_dim: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    X_value = np.asarray(X_value, dtype=np.float32)
    if descriptor_raw_dim <= 0:
        raise ValueError("descriptor_raw_dim must be positive.")
    if metadata_raw_dim is None:
        metadata_raw_dim = max(0, X_value.shape[1] - descriptor_raw_dim)
    X_desc = X_value[:, :descriptor_raw_dim]
    X_meta = (
        X_value[:, descriptor_raw_dim : descriptor_raw_dim + metadata_raw_dim]
        if metadata_raw_dim > 0
        else np.zeros((len(X_value), 0), dtype=np.float32)
    )
    return X_desc, X_meta


def make_holdout_indices(n_samples: int, seed: int, min_val: int = 1) -> tuple[np.ndarray, np.ndarray]:
    if n_samples <= 0:
        raise ValueError(f"Need at least 1 sample, got {n_samples}.")
    if n_samples == 1:
        return np.array([0], dtype=int), np.array([], dtype=int)
    if n_samples < 6:
        order = np.random.default_rng(seed).permutation(n_samples)
        n_val = 1 if n_samples >= 3 else 0
        if n_val == 0:
            return order, np.array([], dtype=int)
        return order[n_val:], order[:n_val]
    test_size = max(min_val, int(round(n_samples * 0.2)))
    test_size = min(test_size, n_samples - 1)
    return train_test_split(np.arange(n_samples), test_size=test_size, random_state=seed)


def make_tensor_loader(*arrays: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    tensors = [torch.tensor(arr, dtype=torch.float32) for arr in arrays]
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle)


def train_neural_regressor(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    *,
    max_epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device: str = "auto",
) -> tuple[nn.Module, pd.DataFrame]:
    device = resolve_device(device)
    torch.manual_seed(seed)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.SmoothL1Loss()
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    stale_epochs = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = [item.to(device) for item in batch]
            optimizer.zero_grad()
            pred = model(*batch[:-1]) if len(batch) > 2 else model(batch[0])
            loss = loss_fn(pred, batch[-1])
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        train_loss = float(np.mean(train_losses)) if train_losses else np.nan
        val_loss = train_loss
        if val_loader is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    batch = [item.to(device) for item in batch]
                    pred = model(*batch[:-1]) if len(batch) > 2 else model(batch[0])
                    loss = loss_fn(pred, batch[-1])
                    val_losses.append(float(loss.detach().cpu()))
            val_loss = float(np.mean(val_losses)) if val_losses else train_loss

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if val_loader is not None and stale_epochs >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, pd.DataFrame(history)


def _tabular_slice(
    X_value: np.ndarray,
    *,
    selector: str,
    descriptor_raw_dim: int,
    metadata_raw_dim: int | None = None,
) -> np.ndarray:
    X_desc, X_meta = split_descriptor_metadata(X_value, descriptor_raw_dim, metadata_raw_dim)
    selector = str(selector)
    if selector == "descriptor":
        return X_desc
    if selector == "metadata":
        return X_meta
    if selector == "descriptor_meta":
        return np.concatenate([X_desc, X_meta], axis=1).astype(np.float32)
    raise KeyError(f"Unknown tabular selector: {selector}")


def train_cvd_regressor(
    model_cls: type[nn.Module],
    X: np.ndarray,
    y: np.ndarray,
    *,
    embedding_dim: int = 48,
    max_epochs: int = 220,
    patience: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
    device: str = "auto",
) -> dict:
    scaler = fit_cvd_scaler(X)
    X_scaled = transform_cvd(X, scaler)
    y = np.asarray(y, dtype=np.float32)
    y_mean = float(np.nanmean(y))
    y_std = float(np.nanstd(y)) or 1.0
    y_scaled = ((y - y_mean) / y_std).astype(np.float32)
    train_idx, val_idx = make_holdout_indices(len(X_scaled), seed)
    train_loader = make_tensor_loader(
        X_scaled[train_idx],
        y_scaled[train_idx],
        batch_size=min(batch_size, max(1, len(train_idx))),
        shuffle=True,
    )
    val_loader = (
        make_tensor_loader(
            X_scaled[val_idx],
            y_scaled[val_idx],
            batch_size=max(1, len(val_idx)),
            shuffle=False,
        )
        if len(val_idx)
        else None
    )
    model = model_cls(n_channels=X_scaled.shape[1], embedding_dim=embedding_dim)
    model, history = train_neural_regressor(
        model,
        train_loader,
        val_loader,
        max_epochs=max_epochs,
        patience=patience,
        lr=lr,
        weight_decay=weight_decay,
        seed=seed,
        device=device,
    )
    return {"model": model, "scaler": scaler, "y_mean": y_mean, "y_std": y_std, "history": history}


def train_spectrumnn(X: np.ndarray, y: np.ndarray, **kwargs) -> dict:
    return train_cvd_regressor(SpectrumNN, X, y, **kwargs)


def train_cnn(X: np.ndarray, y: np.ndarray, **kwargs) -> dict:
    return train_cvd_regressor(CNNBaseline, X, y, **kwargs)


def train_mlp(
    X_value: np.ndarray,
    y: np.ndarray,
    *,
    embedding_dim: int = 32,
    hidden_dim: int = 64,
    dropout: float = 0.10,
    max_epochs: int = 220,
    patience: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
    device: str = "auto",
) -> dict:
    X_model = tabular_missing_aware_matrix(X_value)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_model).astype(np.float32)
    y = np.asarray(y, dtype=np.float32)
    y_mean = float(np.nanmean(y))
    y_std = float(np.nanstd(y)) or 1.0
    y_scaled = ((y - y_mean) / y_std).astype(np.float32)

    train_idx, val_idx = make_holdout_indices(len(X_scaled), seed)
    train_loader = make_tensor_loader(
        X_scaled[train_idx],
        y_scaled[train_idx],
        batch_size=min(batch_size, max(1, len(train_idx))),
        shuffle=True,
    )
    val_loader = (
        make_tensor_loader(
            X_scaled[val_idx],
            y_scaled[val_idx],
            batch_size=max(1, len(val_idx)),
            shuffle=False,
        )
        if len(val_idx)
        else None
    )
    model = MLPBaseline(
        input_dim=X_scaled.shape[1],
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        dropout=dropout,
    )
    model, history = train_neural_regressor(
        model,
        train_loader,
        val_loader,
        max_epochs=max_epochs,
        patience=patience,
        lr=lr,
        weight_decay=weight_decay,
        seed=seed,
        device=device,
    )
    return {
        "model": model,
        "scaler": scaler,
        "y_mean": y_mean,
        "y_std": y_std,
        "history": history,
    }


def predict_mlp(
    bundle: dict,
    X_value: np.ndarray,
    batch_size: int = 128,
    device: str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    device = resolve_device(device)
    X_model = tabular_missing_aware_matrix(X_value)
    X_scaled = bundle["scaler"].transform(X_model).astype(np.float32)
    loader = DataLoader(torch.tensor(X_scaled, dtype=torch.float32), batch_size=batch_size, shuffle=False)
    model = bundle["model"].to(device)
    model.eval()
    preds = []
    embeddings = []
    with torch.no_grad():
        for xb in loader:
            xb = xb.to(device)
            pred_scaled, embedding = model(xb, return_embedding=True)
            preds.append(pred_scaled.detach().cpu().numpy())
            embeddings.append(embedding.detach().cpu().numpy())
    pred = np.concatenate(preds) * bundle["y_std"] + bundle["y_mean"]
    emb = np.concatenate(embeddings, axis=0)
    return pred, emb


def predict_cvd_regressor(
    bundle: dict,
    X: np.ndarray,
    batch_size: int = 128,
    device: str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    device = resolve_device(device)
    model = bundle["model"].to(device)
    model.eval()
    X_scaled = transform_cvd(X, bundle["scaler"])
    loader = DataLoader(torch.tensor(X_scaled, dtype=torch.float32), batch_size=batch_size, shuffle=False)
    preds = []
    embeddings = []
    with torch.no_grad():
        for xb in loader:
            xb = xb.to(device)
            pred_scaled, embedding = model(xb, return_embedding=True)
            preds.append(pred_scaled.detach().cpu().numpy())
            embeddings.append(embedding.detach().cpu().numpy())
    pred = np.concatenate(preds) * bundle["y_std"] + bundle["y_mean"]
    emb = np.concatenate(embeddings, axis=0)
    return pred, emb


def train_ccvnet(
    X_cvd: np.ndarray,
    X_value: np.ndarray,
    y: np.ndarray,
    *,
    descriptor_raw_dim: int,
    metadata_raw_dim: int | None = None,
    max_epochs: int = 220,
    patience: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
    model_variant: str = "ccv_basic",
    n_cycles: int | None = None,
    device: str = "auto",
) -> dict:
    y = np.asarray(y, dtype=np.float32)
    cvd_scaler = fit_cvd_scaler(X_cvd)
    X_cvd_scaled = transform_cvd(X_cvd, cvd_scaler)
    X_desc, X_meta = split_descriptor_metadata(X_value, descriptor_raw_dim, metadata_raw_dim)
    X_desc_missing_aware = tabular_missing_aware_matrix(X_desc)
    X_meta_missing_aware = (
        tabular_missing_aware_matrix(X_meta)
        if X_meta.shape[1]
        else np.zeros((len(X_meta), 0), dtype=np.float32)
    )
    desc_preprocessor = StandardScaler()
    meta_preprocessor = StandardScaler() if X_meta.shape[1] else None
    X_desc_scaled = desc_preprocessor.fit_transform(X_desc_missing_aware).astype(np.float32)
    X_meta_scaled = (
        meta_preprocessor.fit_transform(X_meta_missing_aware).astype(np.float32)
        if meta_preprocessor is not None
        else np.zeros((len(X_meta), 0), dtype=np.float32)
    )
    y_mean = float(np.nanmean(y))
    y_std = float(np.nanstd(y)) or 1.0
    y_scaled = ((y - y_mean) / y_std).astype(np.float32)
    train_idx, val_idx = make_holdout_indices(len(X_cvd_scaled), seed)
    train_loader = make_tensor_loader(
        X_cvd_scaled[train_idx],
        X_desc_scaled[train_idx],
        X_meta_scaled[train_idx],
        y_scaled[train_idx],
        batch_size=min(batch_size, max(1, len(train_idx))),
        shuffle=True,
    )
    val_loader = (
        make_tensor_loader(
            X_cvd_scaled[val_idx],
            X_desc_scaled[val_idx],
            X_meta_scaled[val_idx],
            y_scaled[val_idx],
            batch_size=max(1, len(val_idx)),
            shuffle=False,
        )
        if len(val_idx)
        else None
    )

    variant_name = str(model_variant)
    cycle_aware = variant_name == "ccv_abs_cycleaware"
    if variant_name == "ccv_meta_moe":
        meta_variant = "moe"
    elif variant_name == "ccv_meta_mole":
        meta_variant = "mole"
    else:
        meta_variant = "residual"

    model = ccvnet(
        n_channels=X_cvd_scaled.shape[1],
        descriptor_dim=X_desc_scaled.shape[1],
        metadata_dim=X_meta_scaled.shape[1],
        cycle_aware=cycle_aware,
        meta_variant=meta_variant,
        n_cycles=n_cycles,
        n_meta_experts=4,
    )
    model, history = train_neural_regressor(
        model,
        train_loader,
        val_loader,
        max_epochs=max_epochs,
        patience=patience,
        lr=lr,
        weight_decay=weight_decay,
        seed=seed,
        device=device,
    )
    return {
        "model": model,
        "cvd_scaler": cvd_scaler,
        "desc_preprocessor": desc_preprocessor,
        "meta_preprocessor": meta_preprocessor,
        "descriptor_raw_dim": int(descriptor_raw_dim),
        "metadata_raw_dim": int(metadata_raw_dim or max(0, X_value.shape[1] - descriptor_raw_dim)),
        "descriptor_dim": int(X_desc_scaled.shape[1]),
        "metadata_dim": int(X_meta_scaled.shape[1]),
        "y_mean": y_mean,
        "y_std": y_std,
        "history": history,
        "model_variant": variant_name,
        "meta_variant": meta_variant,
        "n_cycles": n_cycles,
    }


def predict_ccvnet(
    bundle: dict,
    X_cvd: np.ndarray,
    X_value: np.ndarray,
    batch_size: int = 128,
    device: str = "auto",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    device = resolve_device(device)
    X_cvd_scaled = transform_cvd(X_cvd, bundle["cvd_scaler"])
    X_desc, X_meta = split_descriptor_metadata(
        X_value, bundle["descriptor_raw_dim"], bundle["metadata_raw_dim"]
    )
    X_desc_scaled = bundle["desc_preprocessor"].transform(
        tabular_missing_aware_matrix(X_desc)
    ).astype(np.float32)
    if bundle["meta_preprocessor"] is not None and X_meta.shape[1]:
        X_meta_scaled = bundle["meta_preprocessor"].transform(
            tabular_missing_aware_matrix(X_meta)
        ).astype(np.float32)
    else:
        X_meta_scaled = np.zeros((len(X_desc_scaled), 0), dtype=np.float32)

    model = bundle["model"].to(device)
    model.eval()
    preds = []
    z_cvd = []
    z_desc = []
    with torch.no_grad():
        for start in range(0, len(X_cvd_scaled), batch_size):
            stop = start + batch_size
            pred_scaled, _, _, batch_z_cvd, batch_z_desc, _, _, _ = model(
                torch.tensor(X_cvd_scaled[start:stop], dtype=torch.float32, device=device),
                torch.tensor(X_desc_scaled[start:stop], dtype=torch.float32, device=device),
                torch.tensor(X_meta_scaled[start:stop], dtype=torch.float32, device=device),
                return_details=True,
            )
            preds.append(pred_scaled.detach().cpu().numpy())
            z_cvd.append(batch_z_cvd.detach().cpu().numpy())
            z_desc.append(batch_z_desc.detach().cpu().numpy())
    pred = np.concatenate(preds, axis=0) * bundle["y_std"] + bundle["y_mean"]
    return pred, np.concatenate(z_cvd, axis=0), np.concatenate(z_desc, axis=0)


def train_spectrum_meta(
    X_cvd: np.ndarray,
    X_value: np.ndarray,
    y: np.ndarray,
    *,
    descriptor_raw_dim: int,
    metadata_raw_dim: int | None = None,
    embedding_dim: int = 48,
    max_epochs: int = 220,
    patience: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
    device: str = "auto",
) -> dict:
    y = np.asarray(y, dtype=np.float32)
    cvd_scaler = fit_cvd_scaler(X_cvd)
    X_cvd_scaled = transform_cvd(X_cvd, cvd_scaler)
    _, X_meta = split_descriptor_metadata(X_value, descriptor_raw_dim, metadata_raw_dim)
    X_meta_missing_aware = (
        tabular_missing_aware_matrix(X_meta)
        if X_meta.shape[1]
        else np.zeros((len(X_meta), 0), dtype=np.float32)
    )
    meta_preprocessor = StandardScaler() if X_meta.shape[1] else None
    X_meta_scaled = (
        meta_preprocessor.fit_transform(X_meta_missing_aware).astype(np.float32)
        if meta_preprocessor is not None
        else np.zeros((len(X_meta), 0), dtype=np.float32)
    )
    y_mean = float(np.nanmean(y))
    y_std = float(np.nanstd(y)) or 1.0
    y_scaled = ((y - y_mean) / y_std).astype(np.float32)
    train_idx, val_idx = make_holdout_indices(len(X_cvd_scaled), seed)
    train_loader = make_tensor_loader(
        X_cvd_scaled[train_idx],
        X_meta_scaled[train_idx],
        y_scaled[train_idx],
        batch_size=min(batch_size, max(1, len(train_idx))),
        shuffle=True,
    )
    val_loader = (
        make_tensor_loader(
            X_cvd_scaled[val_idx],
            X_meta_scaled[val_idx],
            y_scaled[val_idx],
            batch_size=max(1, len(val_idx)),
            shuffle=False,
        )
        if len(val_idx)
        else None
    )
    model = SpectrumMetaRegressor(
        n_channels=X_cvd_scaled.shape[1],
        metadata_dim=X_meta_scaled.shape[1],
        cvd_embedding_dim=embedding_dim,
    )
    model, history = train_neural_regressor(
        model,
        train_loader,
        val_loader,
        max_epochs=max_epochs,
        patience=patience,
        lr=lr,
        weight_decay=weight_decay,
        seed=seed,
        device=device,
    )
    return {
        "model": model,
        "cvd_scaler": cvd_scaler,
        "meta_preprocessor": meta_preprocessor,
        "descriptor_raw_dim": int(descriptor_raw_dim),
        "metadata_raw_dim": int(metadata_raw_dim or max(0, X_value.shape[1] - descriptor_raw_dim)),
        "metadata_dim": int(X_meta_scaled.shape[1]),
        "y_mean": y_mean,
        "y_std": y_std,
        "history": history,
    }


def predict_spectrum_meta(
    bundle: dict,
    X_cvd: np.ndarray,
    X_value: np.ndarray,
    batch_size: int = 128,
    device: str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    device = resolve_device(device)
    X_cvd_scaled = transform_cvd(X_cvd, bundle["cvd_scaler"])
    _, X_meta = split_descriptor_metadata(X_value, bundle["descriptor_raw_dim"], bundle["metadata_raw_dim"])
    if bundle["meta_preprocessor"] is not None and X_meta.shape[1]:
        X_meta_scaled = bundle["meta_preprocessor"].transform(
            tabular_missing_aware_matrix(X_meta)
        ).astype(np.float32)
    else:
        X_meta_scaled = np.zeros((len(X_cvd_scaled), 0), dtype=np.float32)
    model = bundle["model"].to(device)
    model.eval()
    preds = []
    embeddings = []
    with torch.no_grad():
        for start in range(0, len(X_cvd_scaled), batch_size):
            stop = start + batch_size
            xb_cvd = torch.tensor(X_cvd_scaled[start:stop], dtype=torch.float32, device=device)
            xb_meta = torch.tensor(X_meta_scaled[start:stop], dtype=torch.float32, device=device)
            pred_scaled, embedding = model(xb_cvd, xb_meta, return_embedding=True)
            preds.append(pred_scaled.detach().cpu().numpy())
            embeddings.append(embedding.detach().cpu().numpy())
    pred = np.concatenate(preds) * bundle["y_std"] + bundle["y_mean"]
    emb = np.concatenate(embeddings, axis=0)
    return pred, emb


def fit_model_bundle(
    model_name: str,
    X_cvd: np.ndarray,
    X_value: np.ndarray,
    y: np.ndarray,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = config or {}
    model_key = str(model_name).lower()
    if model_key in {"spectrum", "spectrumnn", "spectrum_uniform"}:
        return train_spectrumnn(X_cvd, y, **config)
    if model_key in {"cnn", "cnn_baseline"}:
        return train_cnn(X_cvd, y, **config)
    if model_key in {"mlp", "metadata_mlp", "value_mlp"}:
        return train_mlp(X_value, y, **config)
    if model_key == "descriptor_mlp":
        descriptor_raw_dim = int(config.pop("descriptor_raw_dim"))
        metadata_raw_dim = int(config.pop("metadata_raw_dim", max(0, X_value.shape[1] - descriptor_raw_dim)))
        X_desc = _tabular_slice(X_value, selector="descriptor", descriptor_raw_dim=descriptor_raw_dim, metadata_raw_dim=metadata_raw_dim)
        bundle = train_mlp(X_desc, y, **config)
        bundle["descriptor_raw_dim"] = int(descriptor_raw_dim)
        bundle["metadata_raw_dim"] = int(metadata_raw_dim)
        return bundle
    if model_key == "descriptor_meta_mlp":
        descriptor_raw_dim = int(config.pop("descriptor_raw_dim"))
        metadata_raw_dim = int(config.pop("metadata_raw_dim", max(0, X_value.shape[1] - descriptor_raw_dim)))
        X_desc_meta = _tabular_slice(X_value, selector="descriptor_meta", descriptor_raw_dim=descriptor_raw_dim, metadata_raw_dim=metadata_raw_dim)
        bundle = train_mlp(X_desc_meta, y, **config)
        bundle["descriptor_raw_dim"] = int(descriptor_raw_dim)
        bundle["metadata_raw_dim"] = int(metadata_raw_dim)
        return bundle
    if model_key == "spectrum_descriptor":
        descriptor_raw_dim = int(config.pop("descriptor_raw_dim"))
        metadata_raw_dim = int(config.pop("metadata_raw_dim", max(0, X_value.shape[1] - descriptor_raw_dim)))
        X_desc = _tabular_slice(X_value, selector="descriptor", descriptor_raw_dim=descriptor_raw_dim, metadata_raw_dim=metadata_raw_dim)
        return train_ccvnet(X_cvd, X_desc, y, descriptor_raw_dim=descriptor_raw_dim, metadata_raw_dim=0, model_variant="ccv_basic", **config)
    if model_key == "spectrum_meta":
        descriptor_raw_dim = int(config.pop("descriptor_raw_dim"))
        metadata_raw_dim = int(config.pop("metadata_raw_dim", max(0, X_value.shape[1] - descriptor_raw_dim)))
        return train_spectrum_meta(X_cvd, X_value, y, descriptor_raw_dim=descriptor_raw_dim, metadata_raw_dim=metadata_raw_dim, **config)
    if model_key in {"ccvnet", "ccv_basic", "ccv_norm", "ccv_abs_cycleaware", "ccv_meta_moe", "ccv_meta_mole"}:
        return train_ccvnet(X_cvd, X_value, y, model_variant=model_key, **config)
    raise ValueError(f"Unknown model: {model_name}")


def predict_model_bundle(
    model_name: str,
    bundle: dict[str, Any],
    X_cvd: np.ndarray,
    X_value: np.ndarray,
    config: dict[str, Any] | None = None,
) -> np.ndarray:
    config = config or {}
    model_key = str(model_name).lower()
    if model_key in {"spectrum", "spectrumnn", "spectrum_uniform", "cnn", "cnn_baseline"}:
        return predict_cvd_regressor(bundle, X_cvd, **config)[0]
    if model_key in {"mlp", "metadata_mlp", "value_mlp"}:
        return predict_mlp(bundle, X_value, **config)[0]
    if model_key == "descriptor_mlp":
        X_desc = _tabular_slice(X_value, selector="descriptor", descriptor_raw_dim=bundle["descriptor_raw_dim"], metadata_raw_dim=bundle.get("metadata_raw_dim"))
        return predict_mlp(bundle, X_desc, **config)[0]
    if model_key == "descriptor_meta_mlp":
        X_desc_meta = _tabular_slice(X_value, selector="descriptor_meta", descriptor_raw_dim=bundle["descriptor_raw_dim"], metadata_raw_dim=bundle.get("metadata_raw_dim"))
        return predict_mlp(bundle, X_desc_meta, **config)[0]
    if model_key == "spectrum_descriptor":
        X_desc = _tabular_slice(X_value, selector="descriptor", descriptor_raw_dim=bundle["descriptor_raw_dim"], metadata_raw_dim=bundle.get("metadata_raw_dim"))
        return predict_ccvnet(bundle, X_cvd, X_desc, **config)[0]
    if model_key == "spectrum_meta":
        return predict_spectrum_meta(bundle, X_cvd, X_value, **config)[0]
    if model_key in {"ccvnet", "ccv_basic", "ccv_norm", "ccv_abs_cycleaware", "ccv_meta_moe", "ccv_meta_mole"}:
        return predict_ccvnet(bundle, X_cvd, X_value, **config)[0]
    raise ValueError(f"Unknown model: {model_name}")
