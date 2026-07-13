import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from imbalance_pipeline.features.schema import DEFAULT_FEATURE_REGISTRY, FeatureRegistry
from imbalance_pipeline.model.export_onnx import OUTPUT_NAMES, export_member
from imbalance_pipeline.model.network import ImbalanceForecaster
from imbalance_pipeline.training.data import RobustPreprocessor

_MODEL_VERSION = "synthetic-test-v1"
_SEEDS = (17, 29, 43)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a deterministic synthetic ONNX test bundle."
    )
    parser.add_argument("--output", type=Path, default=Path("models/test"))
    parser.add_argument("--force", action="store_true", help="replace a non-empty output directory")
    arguments = parser.parse_args()
    build_bundle(arguments.output, force=arguments.force)


def build_bundle(
    output: Path,
    registry: FeatureRegistry = DEFAULT_FEATURE_REGISTRY,
    *,
    force: bool = False,
) -> None:
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        if not force:
            raise FileExistsError(f"refusing to overwrite non-empty bundle directory: {output}")
        if output.is_dir():
            shutil.rmtree(output)
        else:
            output.unlink()
    output.mkdir(parents=True, exist_ok=True)
    sample = _sample_batch(registry)
    members: list[str] = []
    for index, seed in enumerate(_SEEDS):
        torch.manual_seed(seed)
        member = f"member-{index}.onnx"
        model = ImbalanceForecaster(
            local_features=len(registry.local_names),
            context_features=len(registry.context_names),
            static_features=len(registry.static_names),
            d_model=16,
            tcn_blocks=1,
            transformer_layers=1,
            transformer_heads=4,
        )
        export_member(model, sample, output / member)
        members.append(member)
    (output / "preprocessing.json").write_text(
        _identity_preprocessor(registry).to_json(),
        encoding="utf-8",
    )
    _write_json(
        output / "calibration.json",
        {
            "decision_threshold": 0.5,
            "x_thresholds": [0.0, 1.0],
            "y_thresholds": [0.0, 1.0],
        },
    )
    _write_json(
        output / "evaluation.json",
        {
            "note": (
                "Synthetic deterministic test bundle; never promote for operational forecasting."
            ),
            "synthetic_test_bundle": True,
        },
    )
    artifacts = (*members, "preprocessing.json", "calibration.json", "evaluation.json")
    _write_json(
        output / "manifest.json",
        {
            "checksums": {
                artifact: hashlib.sha256((output / artifact).read_bytes()).hexdigest()
                for artifact in artifacts
            },
            "created_at": "2026-07-13T00:00:00Z",
            "evaluation_file": "evaluation.json",
            "feature_schema_hash": registry.fingerprint,
            "input_shapes": {
                "context": [-1, registry.context_steps, len(registry.context_names)],
                "local": [-1, registry.local_window_minutes, len(registry.local_names)],
                "static": [-1, len(registry.static_names)],
            },
            "members": members,
            "model_version": _MODEL_VERSION,
            "output_names": list(OUTPUT_NAMES),
            "preprocessing_file": "preprocessing.json",
            "calibration_file": "calibration.json",
            "runtime": {
                "architecture": "tiny-tcn-transformer-synthetic",
                "opset": 18,
                "provider": "CPUExecutionProvider",
            },
            "training_period": {
                "end": "2026-07-13T00:00:00Z",
                "start": "2026-07-13T00:00:00Z",
            },
        },
    )


def _sample_batch(registry: FeatureRegistry) -> tuple[torch.Tensor, ...]:
    local = torch.zeros((1, registry.local_window_minutes, len(registry.local_names)))
    context = torch.zeros((1, registry.context_steps, len(registry.context_names)))
    static = torch.zeros((1, len(registry.static_names)))
    masks = (torch.ones_like(local), torch.ones_like(context), torch.ones_like(static))
    return local, masks[0], context, masks[1], static, masks[2]


def _identity_preprocessor(registry: FeatureRegistry) -> RobustPreprocessor:
    return RobustPreprocessor(
        feature_schema_hash=registry.fingerprint,
        local_names=registry.local_names,
        context_names=registry.context_names,
        static_names=registry.static_names,
        local_location=np.zeros(len(registry.local_names), dtype=np.float32),
        local_scale=np.ones(len(registry.local_names), dtype=np.float32),
        context_location=np.zeros(len(registry.context_names), dtype=np.float32),
        context_scale=np.ones(len(registry.context_names), dtype=np.float32),
        static_location=np.zeros(len(registry.static_names), dtype=np.float32),
        static_scale=np.ones(len(registry.static_names), dtype=np.float32),
    )


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
