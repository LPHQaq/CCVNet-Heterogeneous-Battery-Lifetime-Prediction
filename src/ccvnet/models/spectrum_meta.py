from __future__ import annotations

import torch
from torch import nn

from ccvnet.models.ccvnet import MetaResidual
from ccvnet.models.spectrumnn import SpectrumNN


class SpectrumMetaRegressor(nn.Module):
    def __init__(
        self,
        n_channels: int,
        metadata_dim: int,
        *,
        cvd_embedding_dim: int = 48,
        fusion_hidden_dim: int = 64,
        dropout: float = 0.12,
    ):
        super().__init__()
        self.cvd_backbone = SpectrumNN(
            n_channels=n_channels,
            embedding_dim=cvd_embedding_dim,
            hidden_dim=64,
            stem_channels=16,
            dropout=dropout,
        )
        self.base_head = nn.Sequential(
            nn.Linear(cvd_embedding_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, 1),
        )
        self.meta_residual = MetaResidual(metadata_dim, hidden_dim=24, dropout=0.05)

    def forward(self, x_cvd: torch.Tensor, x_meta: torch.Tensor, return_embedding: bool = False):
        _, z_cvd = self.cvd_backbone(x_cvd, return_embedding=True)
        y_base = self.base_head(z_cvd).squeeze(-1)
        delta = self.meta_residual(x_meta)
        pred = y_base + delta
        if return_embedding:
            return pred, z_cvd
        return pred
