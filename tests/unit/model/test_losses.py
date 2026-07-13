import torch

from imbalance_pipeline.model.distribution import gaussian_mixture_nll
from imbalance_pipeline.model.losses import (
    MultiTaskLoss,
    flip_distribution_probability,
    focal_binary_loss,
)
from imbalance_pipeline.model.network import ModelOutput
from imbalance_pipeline.training.data import TrainingBatch


def batch() -> TrainingBatch:
    return TrainingBatch(
        local=torch.zeros((2, 3, 2)),
        local_mask=torch.ones((2, 3, 2)),
        context=torch.zeros((2, 2, 1)),
        context_mask=torch.ones((2, 2, 1)),
        static=torch.zeros((2, 1)),
        static_mask=torch.ones((2, 1)),
        target_next=torch.tensor([-12.0, 15.0]),
        target_delta=torch.tensor([-2.0, 3.0]),
        delta_mask=torch.ones(2),
        auxiliary_target=torch.tensor([[-8.0, -6.0, -4.0], [12.0, 14.0, 16.0]]),
        auxiliary_mask=torch.ones((2, 3)),
        flip_target=torch.tensor([1.0, 0.0]),
        flip_mask=torch.tensor([1.0, 1.0]),
        current_state=torch.tensor([1, -1]),
        batch_id="loss-fixture",
    )


def output() -> ModelOutput:
    return ModelOutput(
        mixture_logits=torch.zeros((2, 2), requires_grad=True),
        mixture_means=torch.tensor([[-12.0, -11.0], [15.0, 16.0]], requires_grad=True),
        mixture_log_scales=torch.zeros((2, 2), requires_grad=True),
        flip_logit=torch.tensor([0.2, -0.2], requires_grad=True),
        delta=torch.tensor([-1.0, 2.0], requires_grad=True),
        auxiliary_horizons=torch.tensor(
            [[-7.0, -5.0, -3.0], [11.0, 13.0, 15.0]],
            requires_grad=True,
        ),
    )


def test_mixture_nll_and_focal_loss_are_stable() -> None:
    target = torch.tensor([0.0])
    close = gaussian_mixture_nll(
        target,
        torch.zeros((1, 1)),
        torch.zeros((1, 1)),
        torch.zeros((1, 1)),
    )
    distant = gaussian_mixture_nll(
        target,
        torch.zeros((1, 1)),
        torch.full((1, 1), 20.0),
        torch.zeros((1, 1)),
    )
    masked = focal_binary_loss(
        torch.tensor([100.0, -100.0]),
        torch.tensor([1.0, 0.0]),
        torch.zeros(2),
        positive_weight=20.0,
    )

    assert torch.all(close < distant)
    assert masked.item() == 0.0
    assert torch.isfinite(
        focal_binary_loss(
            torch.tensor([100.0, -100.0]),
            torch.tensor([0.0, 1.0]),
            torch.ones(2),
            positive_weight=20.0,
        )
    )


def test_flip_distribution_consistency_is_state_aware_and_all_heads_receive_gradients() -> None:
    values = output()
    probability = flip_distribution_probability(
        values.mixture_logits,
        values.mixture_means,
        values.mixture_log_scales,
        torch.tensor([1, -1]),
        deadband_mw=10.0,
    )
    loss = MultiTaskLoss(positive_weight=2.0)(values, batch())

    assert probability[0] > 0.5
    assert probability[1] > 0.5
    assert torch.isfinite(loss.total)
    loss.total.backward()
    for tensor in (
        values.mixture_logits,
        values.mixture_means,
        values.mixture_log_scales,
        values.flip_logit,
        values.delta,
        values.auxiliary_horizons,
    ):
        assert tensor.grad is not None
        assert torch.count_nonzero(tensor.grad) > 0
