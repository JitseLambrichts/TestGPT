from pathlib import Path

import onnx
import torch

from imbalance_pipeline.model.export_onnx import export_member
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
