from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as functional

from imbalance_pipeline.features.schema import DEFAULT_FEATURE_REGISTRY


@dataclass(frozen=True, slots=True)
class ModelOutput:
    mixture_logits: Tensor
    mixture_means: Tensor
    mixture_log_scales: Tensor
    flip_logit: Tensor
    delta: Tensor
    auxiliary_horizons: Tensor


class _CausalResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self._first = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            dilation=dilation,
            padding=2 * dilation,
        )
        self._second = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            dilation=dilation,
            padding=2 * dilation,
        )
        self._normalization = nn.GroupNorm(1, channels)

    def forward(self, values: Tensor) -> Tensor:
        first = functional.gelu(_causal_crop(self._first(values), values.shape[-1]))
        second = _causal_crop(self._second(first), values.shape[-1])
        return functional.gelu(self._normalization(values + second))


class ImbalanceForecaster(nn.Module):
    def __init__(
        self,
        *,
        local_features: int,
        context_features: int,
        static_features: int,
        d_model: int = 128,
        tcn_blocks: int = 6,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        mixture_components: int = 5,
        patch_size: int = 15,
    ) -> None:
        super().__init__()
        dimensions = (
            local_features,
            context_features,
            static_features,
            d_model,
            tcn_blocks,
            transformer_layers,
            transformer_heads,
            mixture_components,
            patch_size,
        )
        if min(dimensions) <= 0:
            raise ValueError("feature counts and architecture dimensions must be positive")
        if d_model % transformer_heads != 0:
            raise ValueError("d_model must be divisible by transformer_heads")
        self._patch_size = patch_size
        self._local_projection = nn.Linear(local_features * 2, d_model)
        self._context_projection = nn.Linear(context_features * 2, d_model)
        self._static_projection = nn.Sequential(
            nn.Linear(static_features * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self._tcn = nn.Sequential(
            *[_CausalResidualBlock(d_model, 2**index) for index in range(tcn_blocks)]
        )
        self._positions = nn.Parameter(torch.zeros(1, 64, d_model))
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=transformer_heads,
            dim_feedforward=d_model * 4,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self._transformer = nn.TransformerEncoder(transformer_layer, num_layers=transformer_layers)
        self._context_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self._fusion_gate = nn.Linear(d_model * 2, d_model)
        self._fusion_normalization = nn.LayerNorm(d_model)
        self._mixture = nn.Linear(d_model, mixture_components * 3)
        self._flip = nn.Linear(d_model, 1)
        self._delta = nn.Linear(d_model, 1)
        self._auxiliary = nn.Linear(d_model, 3)

    def forward(
        self,
        local: Tensor,
        local_mask: Tensor,
        context: Tensor,
        context_mask: Tensor,
        static: Tensor,
        static_mask: Tensor,
    ) -> ModelOutput:
        local_encoded = self._local_projection(_masked_inputs(local, local_mask))
        local_tcn = self._tcn(local_encoded.transpose(1, 2)).transpose(1, 2)
        patches = _patch_mean(local_tcn, self._patch_size)
        transformed = self._transformer(patches + self._positions[:, : patches.shape[1]])
        local_summary = transformed.mean(dim=1) + local_tcn[:, -1]
        context_encoded = self._context_projection(_masked_inputs(context, context_mask))
        context_summary = _masked_mean(context_encoded, context_mask.mean(dim=-1, keepdim=True))
        static_summary = self._static_projection(_masked_inputs(static, static_mask))
        exogenous = self._context_gate(torch.cat((context_summary, static_summary), dim=-1))
        gate = torch.sigmoid(self._fusion_gate(torch.cat((local_summary, exogenous), dim=-1)))
        fused = self._fusion_normalization(gate * local_summary + (1.0 - gate) * exogenous)
        mixture = self._mixture(fused)
        logits, means, log_scales = mixture.chunk(3, dim=-1)
        return ModelOutput(
            mixture_logits=logits,
            mixture_means=means,
            mixture_log_scales=log_scales,
            flip_logit=self._flip(fused).squeeze(-1),
            delta=self._delta(fused).squeeze(-1),
            auxiliary_horizons=self._auxiliary(fused),
        )

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


def default_model() -> ImbalanceForecaster:
    return ImbalanceForecaster(
        local_features=len(DEFAULT_FEATURE_REGISTRY.local_names),
        context_features=len(DEFAULT_FEATURE_REGISTRY.context_names),
        static_features=len(DEFAULT_FEATURE_REGISTRY.static_names),
    )


def _masked_inputs(values: Tensor, mask: Tensor) -> Tensor:
    return torch.cat((values * mask, mask), dim=-1)


def _causal_crop(values: Tensor, length: int) -> Tensor:
    return values[..., :length]


def _patch_mean(values: Tensor, patch_size: int) -> Tensor:
    padding = (-values.shape[1]) % patch_size
    if padding:
        values = functional.pad(values, (0, 0, 0, padding))
    return values.reshape(values.shape[0], -1, patch_size, values.shape[-1]).mean(dim=2)


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    weights = mask.to(values.dtype)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
