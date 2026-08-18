from __future__ import annotations

import torch
from torch import nn

from ccvnet.models.spectrumnn import ResidualConvBlock, SpectrumNN


class DescriptorEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, embedding_dim: int = 32, dropout: float = 0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MetaResidual(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 24, dropout: float = 0.05):
        super().__init__()
        if input_dim <= 0:
            self.net = None
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.net is None:
            return torch.zeros((x.shape[0],), device=x.device, dtype=x.dtype)
        return self.net(x).squeeze(-1)


class FinalMoE(nn.Module):
    def __init__(
        self,
        fused_dim: int,
        metadata_dim: int,
        n_experts: int = 4,
        gate_hidden_dim: int = 24,
        expert_hidden_dim: int = 64,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.metadata_dim = int(metadata_dim)
        if self.metadata_dim <= 0:
            self.gate = None
            self.experts = None
        else:
            self.gate = nn.Sequential(
                nn.Linear(metadata_dim, gate_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(gate_hidden_dim, n_experts),
            )
            self.experts = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(fused_dim, expert_hidden_dim),
                        nn.LayerNorm(expert_hidden_dim),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(expert_hidden_dim, 1),
                    )
                    for _ in range(n_experts)
                ]
            )

    def forward(self, fused: torch.Tensor, meta: torch.Tensor, return_details: bool = False):
        if self.gate is None or self.experts is None:
            pred = torch.zeros((fused.shape[0],), device=fused.device, dtype=fused.dtype)
            if return_details:
                return pred, None, None
            return pred
        gate_weight = torch.softmax(self.gate(meta), dim=1)
        expert_outputs = torch.cat([expert(fused) for expert in self.experts], dim=1)
        pred = torch.sum(gate_weight * expert_outputs, dim=1)
        if return_details:
            return pred, gate_weight, expert_outputs
        return pred


class FinalMoLE(nn.Module):
    def __init__(self, fused_dim: int, metadata_dim: int, n_experts: int = 4, gate_hidden_dim: int = 24, dropout: float = 0.05):
        super().__init__()
        self.metadata_dim = int(metadata_dim)
        if self.metadata_dim <= 0:
            self.gate = None
            self.experts = None
        else:
            self.gate = nn.Sequential(
                nn.Linear(metadata_dim, gate_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(gate_hidden_dim, n_experts),
            )
            self.experts = nn.ModuleList([nn.Linear(fused_dim, 1) for _ in range(n_experts)])

    def forward(self, fused: torch.Tensor, meta: torch.Tensor, return_details: bool = False):
        if self.gate is None or self.experts is None:
            pred = torch.zeros((fused.shape[0],), device=fused.device, dtype=fused.dtype)
            if return_details:
                return pred, None, None
            return pred
        gate_weight = torch.softmax(self.gate(meta), dim=1)
        expert_outputs = torch.cat([expert(fused) for expert in self.experts], dim=1)
        pred = torch.sum(gate_weight * expert_outputs, dim=1)
        if return_details:
            return pred, gate_weight, expert_outputs
        return pred


class CycleAwareSpectrum(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_cycles: int,
        embedding_dim: int = 48,
        hidden_dim: int = 48,
        cycle_embedding_dim: int = 32,
        stem_channels: int = 16,
        dropout: float = 0.15,
    ):
        super().__init__()
        if n_cycles <= 0:
            raise ValueError("Cycle-aware spectrum expects n_cycles > 0.")
        if n_channels % n_cycles != 0:
            raise ValueError(f"Cannot factor n_channels={n_channels} by n_cycles={n_cycles}.")
        self.n_cycles = int(n_cycles)
        self.channels_per_cycle = int(n_channels // n_cycles)
        self.cycle_stem = nn.Sequential(
            nn.Conv1d(self.channels_per_cycle, stem_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(stem_channels),
            nn.GELU(),
            nn.Conv1d(stem_channels, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        self.cycle_encoder = nn.Sequential(
            ResidualConvBlock(hidden_dim, kernel_size=7, dilation=1, dropout=dropout),
            ResidualConvBlock(hidden_dim, kernel_size=7, dilation=2, dropout=dropout),
        )
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.cycle_projector = nn.Sequential(
            nn.Linear(hidden_dim * 2, cycle_embedding_dim),
            nn.LayerNorm(cycle_embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.embedding = nn.Sequential(
            nn.Linear(cycle_embedding_dim * 2, embedding_dim),
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
        batch_size, _, n_points = x.shape
        x_cycle = x.view(batch_size, self.n_cycles, self.channels_per_cycle, n_points)
        x_cycle = x_cycle.reshape(batch_size * self.n_cycles, self.channels_per_cycle, n_points)
        encoded = self.cycle_encoder(self.cycle_stem(x_cycle))
        pooled = torch.cat([self.avg_pool(encoded).squeeze(-1), self.max_pool(encoded).squeeze(-1)], dim=1)
        cycle_embedding = self.cycle_projector(pooled).view(batch_size, self.n_cycles, -1)
        cycle_mean = cycle_embedding.mean(dim=1)
        cycle_max = cycle_embedding.max(dim=1).values
        embedding = self.embedding(torch.cat([cycle_mean, cycle_max], dim=1))
        pred = self.regressor(embedding).squeeze(-1)
        if return_embedding:
            return pred, embedding
        return pred


class ccvnet(nn.Module):
    def __init__(
        self,
        n_channels: int,
        descriptor_dim: int,
        metadata_dim: int,
        *,
        n_cycles: int | None = None,
        cvd_embedding_dim: int = 48,
        descriptor_embedding_dim: int = 32,
        fusion_hidden_dim: int = 64,
        dropout: float = 0.12,
        cycle_aware: bool = False,
        meta_variant: str = "residual",
        n_meta_experts: int = 4,
    ):
        super().__init__()
        if cycle_aware:
            if n_cycles is None:
                raise ValueError("cycle_aware=True requires n_cycles.")
            self.cvd_backbone = CycleAwareSpectrum(
                n_channels=n_channels,
                n_cycles=n_cycles,
                embedding_dim=cvd_embedding_dim,
                dropout=dropout,
            )
        else:
            self.cvd_backbone = SpectrumNN(
                n_channels=n_channels,
                embedding_dim=cvd_embedding_dim,
                hidden_dim=64,
                stem_channels=16,
                dropout=dropout,
            )
        self.descriptor_encoder = DescriptorEncoder(
            descriptor_dim, hidden_dim=64, embedding_dim=descriptor_embedding_dim, dropout=dropout
        )
        fused_dim = cvd_embedding_dim + descriptor_embedding_dim
        self.base_head = nn.Sequential(
            nn.Linear(fused_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, 1),
        )
        if str(meta_variant) == "moe":
            self.meta_head = FinalMoE(fused_dim, metadata_dim, n_experts=n_meta_experts, expert_hidden_dim=fusion_hidden_dim)
        elif str(meta_variant) == "mole":
            self.meta_head = FinalMoLE(fused_dim, metadata_dim, n_experts=n_meta_experts)
        else:
            self.meta_head = None
        self.meta_residual = MetaResidual(metadata_dim, hidden_dim=24, dropout=0.05) if str(meta_variant) == "residual" else None
        self.cycle_aware = bool(cycle_aware)
        self.meta_variant = str(meta_variant)

    def forward(self, x_cvd: torch.Tensor, x_desc: torch.Tensor, x_meta: torch.Tensor, return_details: bool = False):
        _, z_cvd = self.cvd_backbone(x_cvd, return_embedding=True)
        z_desc = self.descriptor_encoder(x_desc)
        fused = torch.cat([z_cvd, z_desc], dim=1)

        if self.meta_variant in {"moe", "mole"}:
            pred, gate_weight, expert_outputs = self.meta_head(fused, x_meta, return_details=True)
            y_base = self.base_head(fused).squeeze(-1)
            delta = pred - y_base
        else:
            y_base = self.base_head(fused).squeeze(-1)
            delta = self.meta_residual(x_meta)
            pred = y_base + delta
            gate_weight = None
            expert_outputs = None

        if return_details:
            return pred, y_base, delta, z_cvd, z_desc, fused, gate_weight, expert_outputs
        return pred

