from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as functional

from imbalance_pipeline.model.distribution import gaussian_mixture_cdf, gaussian_mixture_nll
from imbalance_pipeline.model.network import ModelOutput
from imbalance_pipeline.training.data import TrainingBatch


@dataclass(frozen=True, slots=True)
class LossBreakdown:
    total: Tensor
    mixture_nll: Tensor
    flip_focal: Tensor
    delta_huber: Tensor
    auxiliary_huber: Tensor
    consistency: Tensor


class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        *,
        positive_weight: float,
        deadband_mw: float = 10.0,
        mixture_weight: float = 1.0,
        flip_weight: float = 0.5,
        delta_weight: float = 0.2,
        auxiliary_weight: float = 0.1,
        consistency_weight: float = 0.1,
    ) -> None:
        super().__init__()
        if positive_weight <= 0 or deadband_mw <= 0:
            raise ValueError("positive_weight and deadband_mw must be positive")
        if any(
            weight < 0
            for weight in (
                mixture_weight,
                flip_weight,
                delta_weight,
                auxiliary_weight,
                consistency_weight,
            )
        ):
            raise ValueError("loss weights cannot be negative")
        self._positive_weight = min(max(positive_weight, 1.0), 20.0)
        self._deadband_mw = deadband_mw
        self._weights = (
            mixture_weight,
            flip_weight,
            delta_weight,
            auxiliary_weight,
            consistency_weight,
        )

    def forward(self, output: ModelOutput, batch: TrainingBatch) -> LossBreakdown:
        mixture_nll = gaussian_mixture_nll(
            batch.target_next,
            output.mixture_logits,
            output.mixture_means,
            output.mixture_log_scales,
        ).mean()
        flip_focal = focal_binary_loss(
            output.flip_logit,
            batch.flip_target,
            batch.flip_mask,
            positive_weight=self._positive_weight,
        )
        delta_huber = _masked_huber(output.delta, batch.target_delta, batch.delta_mask)
        auxiliary_huber = _masked_huber(
            output.auxiliary_horizons,
            batch.auxiliary_target,
            batch.auxiliary_mask,
        )
        distribution_flip = flip_distribution_probability(
            output.mixture_logits,
            output.mixture_means,
            output.mixture_log_scales,
            batch.current_state,
            deadband_mw=self._deadband_mw,
        )
        known_state = (batch.current_state != 0).to(output.flip_logit.dtype)
        consistency = _masked_binary_cross_entropy_with_logits(
            output.flip_logit,
            distribution_flip,
            known_state,
        )
        (
            mixture_weight,
            flip_weight,
            delta_weight,
            auxiliary_weight,
            consistency_weight,
        ) = self._weights
        total = (
            mixture_weight * mixture_nll
            + flip_weight * flip_focal
            + delta_weight * delta_huber
            + auxiliary_weight * auxiliary_huber
            + consistency_weight * consistency
        )
        if not bool(torch.isfinite(total).all()):
            raise ValueError(f"non-finite multi-task loss for batch {batch.batch_id}")
        return LossBreakdown(
            total=total,
            mixture_nll=mixture_nll,
            flip_focal=flip_focal,
            delta_huber=delta_huber,
            auxiliary_huber=auxiliary_huber,
            consistency=consistency,
        )


def focal_binary_loss(
    logits: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    positive_weight: float,
    gamma: float = 2.0,
) -> Tensor:
    if positive_weight <= 0 or gamma < 0:
        raise ValueError("positive_weight must be positive and gamma cannot be negative")
    probability = torch.sigmoid(logits)
    probability_at_target = torch.where(target > 0.5, probability, 1.0 - probability)
    class_weight = torch.where(
        target > 0.5,
        torch.as_tensor(positive_weight, device=logits.device, dtype=logits.dtype),
        torch.ones_like(logits),
    )
    cross_entropy = functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    weighted_loss = class_weight * (1.0 - probability_at_target).pow(gamma) * cross_entropy
    return _masked_mean(weighted_loss, mask)


def flip_distribution_probability(
    logits: Tensor,
    means: Tensor,
    log_scales: Tensor,
    current_state: Tensor,
    *,
    deadband_mw: float,
) -> Tensor:
    if deadband_mw <= 0:
        raise ValueError("deadband_mw must be positive")
    boundary = torch.full_like(current_state, deadband_mw, dtype=means.dtype)
    below_negative = gaussian_mixture_cdf(-boundary, logits, means, log_scales)
    above_positive = 1.0 - gaussian_mixture_cdf(boundary, logits, means, log_scales)
    return torch.where(
        current_state > 0,
        below_negative,
        torch.where(current_state < 0, above_positive, torch.full_like(below_negative, 0.5)),
    )


def _masked_huber(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    return _masked_mean(functional.huber_loss(prediction, target, reduction="none"), mask)


def _masked_binary_cross_entropy_with_logits(
    logits: Tensor,
    target: Tensor,
    mask: Tensor,
) -> Tensor:
    return _masked_mean(
        functional.binary_cross_entropy_with_logits(logits, target, reduction="none"),
        mask,
    )


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    typed_mask = mask.to(values.dtype)
    return (values * typed_mask).sum() / typed_mask.sum().clamp_min(1.0)
