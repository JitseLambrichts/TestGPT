from collections.abc import Sequence
from pathlib import Path

import onnx
import torch
from torch import Tensor, nn

from imbalance_pipeline.model.network import ImbalanceForecaster

INPUT_NAMES = ("local", "local_mask", "context", "context_mask", "static", "static_mask")
OUTPUT_NAMES = (
    "mixture_logits",
    "mixture_means",
    "mixture_log_scales",
    "flip_logit",
    "delta",
    "auxiliary_horizons",
)


class _OnnxMember(nn.Module):
    def __init__(self, model: ImbalanceForecaster) -> None:
        super().__init__()
        self._model = model

    def forward(
        self,
        local: Tensor,
        local_mask: Tensor,
        context: Tensor,
        context_mask: Tensor,
        static: Tensor,
        static_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        output = self._model(local, local_mask, context, context_mask, static, static_mask)
        return (
            output.mixture_logits,
            output.mixture_means,
            output.mixture_log_scales,
            output.flip_logit,
            output.delta,
            output.auxiliary_horizons,
        )


def export_member(
    model: ImbalanceForecaster,
    sample_batch: Sequence[Tensor],
    output_path: Path,
) -> None:
    if len(sample_batch) != len(INPUT_NAMES):
        raise ValueError("ONNX export requires six model input tensors")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = _OnnxMember(model).eval()
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            tuple(sample_batch),
            output_path,
            input_names=list(INPUT_NAMES),
            output_names=list(OUTPUT_NAMES),
            dynamic_axes={name: {0: "batch"} for name in (*INPUT_NAMES, *OUTPUT_NAMES)},
            opset_version=18,
            do_constant_folding=True,
        )
    onnx.checker.check_model(str(output_path))
