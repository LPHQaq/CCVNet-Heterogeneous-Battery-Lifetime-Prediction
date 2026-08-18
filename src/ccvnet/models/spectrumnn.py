from __future__ import annotations

import torch
from torch import nn


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 7, dilation: int = 1, dropout: float = 0.10):
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class GatedAttentionPool(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.attention = nn.Conv1d(channels, 1, kernel_size=1)
        self.gate = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention_weight = torch.softmax(self.attention(x), dim=-1)
        gate = torch.sigmoid(self.gate(x))
        return torch.sum(attention_weight * gate * x, dim=-1)


class SpectrumNN(nn.Module):
    def __init__(
        self,
        n_channels: int,
        embedding_dim: int = 48,
        hidden_dim: int = 64,
        stem_channels: int = 16,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.multiscale_stem = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(n_channels, stem_channels, kernel_size=kernel_size, padding=kernel_size // 2),
                    nn.BatchNorm1d(stem_channels),
                    nn.GELU(),
                )
                for kernel_size in (3, 7, 15, 31)
            ]
        )
        self.projection = nn.Sequential(
            nn.Conv1d(stem_channels * 4, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.encoder = nn.Sequential(
            ResidualConvBlock(hidden_dim, kernel_size=7, dilation=1, dropout=dropout),
            ResidualConvBlock(hidden_dim, kernel_size=7, dilation=2, dropout=dropout),
            ResidualConvBlock(hidden_dim, kernel_size=7, dilation=4, dropout=dropout),
        )
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.attention_pool = GatedAttentionPool(hidden_dim)
        self.embedding = nn.Sequential(
            nn.Linear(hidden_dim * 3, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.regressor = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        multiscale = torch.cat([stem(x) for stem in self.multiscale_stem], dim=1)
        encoded = self.encoder(self.projection(multiscale))
        pooled = torch.cat(
            [
                self.avg_pool(encoded).squeeze(-1),
                self.max_pool(encoded).squeeze(-1),
                self.attention_pool(encoded),
            ],
            dim=1,
        )
        embedding = self.embedding(pooled)
        prediction = self.regressor(embedding).squeeze(-1)
        if return_embedding:
            return prediction, embedding
        return prediction

