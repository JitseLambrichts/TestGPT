import math

import torch

from imbalance_pipeline.model.distribution import (
    gaussian_mixture_cdf,
    gaussian_mixture_nll,
    gaussian_mixture_quantile,
    mixture_scales,
    mixture_weights,
)


def standard_normal_parameters() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raw_scale = math.log(math.expm1(1.0 - 1e-4))
    return (
        torch.zeros((2, 1)),
        torch.zeros((2, 1)),
        torch.full((2, 1), raw_scale),
    )


def test_one_component_standard_normal_has_stable_cdf_quantiles_and_nll() -> None:
    logits, means, log_scales = standard_normal_parameters()
    cdf = gaussian_mixture_cdf(torch.zeros(2), logits, means, log_scales)
    median = gaussian_mixture_quantile(torch.full((2,), 0.5), logits, means, log_scales)
    low = gaussian_mixture_quantile(torch.full((2,), 0.1), logits, means, log_scales)
    high = gaussian_mixture_quantile(torch.full((2,), 0.9), logits, means, log_scales)
    nll = gaussian_mixture_nll(torch.tensor([-1_000.0, 1_000.0]), logits, means, log_scales)

    torch.testing.assert_close(cdf, torch.full((2,), 0.5), atol=1e-6, rtol=0.0)
    torch.testing.assert_close(median, torch.zeros(2), atol=1e-5, rtol=0.0)
    assert torch.all(low < median)
    assert torch.all(median < high)
    assert torch.isfinite(nll).all()
    assert torch.all(mixture_scales(log_scales) > 0)
    torch.testing.assert_close(mixture_weights(logits).sum(dim=-1), torch.ones(2))


def test_symmetric_two_component_mixture_has_a_zero_median() -> None:
    logits = torch.zeros((1, 2))
    means = torch.tensor([[-2.0, 2.0]])
    raw_scale = math.log(math.expm1(1.0 - 1e-4))
    log_scales = torch.full((1, 2), raw_scale)

    median = gaussian_mixture_quantile(torch.tensor([0.5]), logits, means, log_scales)

    torch.testing.assert_close(median, torch.zeros(1), atol=1e-5, rtol=0.0)
