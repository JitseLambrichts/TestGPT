import math

import torch
from torch import Tensor
from torch.nn import functional as functional

_MIN_SCALE = 1e-4
_SQRT_TWO = math.sqrt(2.0)


def mixture_weights(logits: Tensor) -> Tensor:
    return functional.softmax(logits, dim=-1)


def mixture_scales(log_scales: Tensor) -> Tensor:
    return functional.softplus(log_scales) + _MIN_SCALE


def gaussian_mixture_nll(
    target: Tensor,
    logits: Tensor,
    means: Tensor,
    log_scales: Tensor,
) -> Tensor:
    scales = mixture_scales(log_scales)
    standardized = (target.unsqueeze(-1) - means) / scales
    component_log_probabilities = (
        -0.5 * standardized.square() - torch.log(scales) - 0.5 * math.log(2.0 * math.pi)
    )
    return -torch.logsumexp(
        functional.log_softmax(logits, dim=-1) + component_log_probabilities,
        dim=-1,
    )


def gaussian_mixture_cdf(
    value: Tensor,
    logits: Tensor,
    means: Tensor,
    log_scales: Tensor,
) -> Tensor:
    scales = mixture_scales(log_scales)
    standardized = (value.unsqueeze(-1) - means) / scales
    component_cdf = 0.5 * (1.0 + torch.erf(standardized / _SQRT_TWO))
    return (mixture_weights(logits) * component_cdf).sum(dim=-1)


def gaussian_mixture_quantile(
    probability: Tensor,
    logits: Tensor,
    means: Tensor,
    log_scales: Tensor,
) -> Tensor:
    scales = mixture_scales(log_scales)
    width = 12.0 * scales.max(dim=-1).values
    lower = means.min(dim=-1).values - width
    upper = means.max(dim=-1).values + width
    for _ in range(64):
        midpoint = (lower + upper) / 2.0
        below = gaussian_mixture_cdf(midpoint, logits, means, log_scales) < probability
        lower = torch.where(below, midpoint, lower)
        upper = torch.where(below, upper, midpoint)
    return (lower + upper) / 2.0
