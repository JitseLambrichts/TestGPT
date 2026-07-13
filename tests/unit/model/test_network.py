import pytest
import torch

from imbalance_pipeline.model.network import ImbalanceForecaster


@pytest.mark.parametrize(
    "invalid_dimensions",
    [
        {"transformer_heads": 0},
        {"mixture_components": 0},
        {"patch_size": 0},
    ],
)
def test_network_rejects_non_positive_architecture_dimensions(
    invalid_dimensions: dict[str, int],
) -> None:
    with pytest.raises(ValueError, match="architecture dimensions must be positive"):
        ImbalanceForecaster(
            local_features=8,
            context_features=6,
            static_features=4,
            d_model=16,
            **invalid_dimensions,
        )


def test_tiny_mask_aware_tcn_transformer_has_all_heads_and_temporal_gradients() -> None:
    torch.manual_seed(7)
    model = ImbalanceForecaster(
        local_features=8,
        context_features=6,
        static_features=4,
        d_model=16,
        tcn_blocks=2,
        transformer_layers=1,
        transformer_heads=4,
        mixture_components=5,
        patch_size=4,
    )
    local = torch.randn((3, 12, 8), requires_grad=True)
    context = torch.randn((3, 8, 6), requires_grad=True)
    static = torch.randn((3, 4), requires_grad=True)
    local_mask = torch.ones_like(local)
    context_mask = torch.ones_like(context)
    static_mask = torch.ones_like(static)
    local_mask[:, 0, :] = 0
    context_mask[:, -1, :] = 0

    model.eval()
    first = model(local, local_mask, context, context_mask, static, static_mask)
    second = model(local, local_mask, context, context_mask, static, static_mask)
    loss = (
        first.mixture_logits.square().mean()
        + first.mixture_means.square().mean()
        + first.mixture_log_scales.square().mean()
        + first.flip_logit.square().mean()
        + first.delta.square().mean()
        + first.auxiliary_horizons.square().mean()
    )
    loss.backward()

    assert first.mixture_logits.shape == (3, 5)
    assert first.mixture_means.shape == (3, 5)
    assert first.mixture_log_scales.shape == (3, 5)
    assert first.flip_logit.shape == (3,)
    assert first.delta.shape == (3,)
    assert first.auxiliary_horizons.shape == (3, 3)
    assert torch.isfinite(first.mixture_means).all()
    torch.testing.assert_close(first.mixture_logits, second.mixture_logits)
    assert local.grad is not None and torch.count_nonzero(local.grad) > 0
    assert context.grad is not None and torch.count_nonzero(context.grad) > 0
    assert static.grad is not None and torch.count_nonzero(static.grad) > 0
