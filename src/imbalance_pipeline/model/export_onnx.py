from collections.abc import Sequence
from pathlib import Path

import onnx
import torch
from torch import Tensor, nn
from torch.export import Dim

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
    export_batch = _dynamic_batch_sample(sample_batch)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = _OnnxMember(model).eval()
    batch = Dim("batch", min=1)
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            export_batch,
            output_path,
            input_names=list(INPUT_NAMES),
            output_names=list(OUTPUT_NAMES),
            external_data=False,
            dynamic_shapes={name: {0: batch} for name in INPUT_NAMES},
            opset_version=18,
            do_constant_folding=True,
        )
    onnx.checker.check_model(str(output_path))


def _dynamic_batch_sample(sample_batch: Sequence[Tensor]) -> tuple[Tensor, ...]:
    batch_size = sample_batch[0].shape[0] if sample_batch[0].ndim else 0
    invalid_batches = any(
        tensor.ndim == 0 or tensor.shape[0] != batch_size for tensor in sample_batch
    )
    if batch_size <= 0 or invalid_batches:
        raise ValueError("ONNX export inputs must have one shared non-empty batch dimension")
    if batch_size > 1:
        return tuple(sample_batch)
    # PyTorch 2.13 specializes a size-one example batch despite a Dim constraint.
    # Duplicate only the export exemplar so the generated graph keeps its batch axis dynamic.
    return tuple(torch.cat((tensor, tensor), dim=0) for tensor in sample_batch)
