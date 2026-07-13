from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort  # type: ignore[import-untyped]
import torch

from imbalance_pipeline.model.export_onnx import INPUT_NAMES, OUTPUT_NAMES, export_member
from imbalance_pipeline.model.network import ImbalanceForecaster


def test_exported_member_has_named_inputs_and_outputs(tmp_path: Path) -> None:
    model = ImbalanceForecaster(
        local_features=8,
        context_features=6,
        static_features=4,
        d_model=16,
        tcn_blocks=1,
        transformer_layers=1,
        transformer_heads=4,
    )
    sample = (
        torch.zeros((1, 12, 8)),
        torch.ones((1, 12, 8)),
        torch.zeros((1, 8, 6)),
        torch.ones((1, 8, 6)),
        torch.zeros((1, 4)),
        torch.ones((1, 4)),
    )
    output = tmp_path / "member.onnx"

    export_member(model, sample, output)

    graph = onnx.load(output).graph
    assert not output.with_suffix(".onnx.data").exists()
    assert [item.name for item in graph.input] == [
        "local",
        "local_mask",
        "context",
        "context_mask",
        "static",
        "static_mask",
    ]
    assert [item.name for item in graph.output] == [
        "mixture_logits",
        "mixture_means",
        "mixture_log_scales",
        "flip_logit",
        "delta",
        "auxiliary_horizons",
    ]


def test_onnx_member_matches_pytorch_for_a_dynamic_batch(tmp_path: Path) -> None:
    torch.manual_seed(17)
    model = ImbalanceForecaster(
        local_features=8,
        context_features=6,
        static_features=4,
        d_model=16,
        tcn_blocks=1,
        transformer_layers=1,
        transformer_heads=4,
    ).eval()
    inputs = (
        torch.randn((32, 12, 8)),
        torch.randint(0, 2, (32, 12, 8), dtype=torch.float32),
        torch.randn((32, 8, 6)),
        torch.randint(0, 2, (32, 8, 6), dtype=torch.float32),
        torch.randn((32, 4)),
        torch.randint(0, 2, (32, 4), dtype=torch.float32),
    )
    output = tmp_path / "member.onnx"

    export_member(model, tuple(item[:1] for item in inputs), output)
    with torch.inference_mode():
        expected = model(*inputs)
    actual = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"]).run(
        list(OUTPUT_NAMES),
        {name: value.numpy() for name, value in zip(INPUT_NAMES, inputs, strict=True)},
    )

    for expected_value, actual_value in zip(
        (
            expected.mixture_logits,
            expected.mixture_means,
            expected.mixture_log_scales,
            expected.flip_logit,
            expected.delta,
            expected.auxiliary_horizons,
        ),
        actual,
        strict=True,
    ):
        np.testing.assert_allclose(actual_value, expected_value.numpy(), atol=1e-4, rtol=1e-4)
