from __future__ import annotations

import numpy as np
import torch
from torch import nn


class MLPBaseline(nn.Module):
    """Tabular MLP baseline for descriptor and/or metadata inputs."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        embedding_dim: int = 32,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
        )
        self.regressor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        embedding = self.embedding(x)
        pred = self.regressor(embedding).squeeze(-1)
        if return_embedding:
            return pred, embedding
        return pred


def tabular_missing_aware_matrix(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.shape[1] == 0:
        return np.zeros((len(X), 0), dtype=np.float32)
    X_mask = np.isfinite(X).astype(np.float32)
    X_filled = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return np.concatenate([X_filled, X_mask], axis=1).astype(np.float32)

