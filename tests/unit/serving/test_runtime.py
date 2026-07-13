import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import torch

from imbalance_pipeline.domain.imbalance import ConfirmedState
from imbalance_pipeline.features.engine import FeatureSnapshot
from imbalance_pipeline.model.export_onnx import export_member
from imbalance_pipeline.model.network import ImbalanceForecaster
from imbalance_pipeline.serving.runtime import OnnxEnsemble

NOW = datetime(2026, 7, 13, 10, 0, tzinfo=UTC)
SCHEMA = "schema-v1"


def test_onnx_ensemble_returns_deterministic_calibrated_distribution(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path / "bundle")
    ensemble = OnnxEnsemble(bundle, expected_schema_hash=SCHEMA)

    first = ensemble.predict(_snapshot())
    second = ensemble.predict(_snapshot())

    assert first == second
    assert first.model_version == "test-v1"
    assert first.feature_schema_hash == SCHEMA
    assert np.isfinite(first.system_imbalance_mw)
    assert first.p10_mw <= first.system_imbalance_mw <= first.p90_mw
    assert 0.0 <= first.flip_probability <= 1.0
    assert isinstance(first.will_flip, bool)
    assert first.current_state is ConfirmedState.POSITIVE
    assert first.predicted_state in (ConfirmedState.POSITIVE, ConfirmedState.NEGATIVE)


def _snapshot() -> FeatureSnapshot:
    return FeatureSnapshot(
        event_id="feature-001",
        cutoff=NOW,
        knowledge_cutoff=NOW,
        target_time=NOW + timedelta(minutes=1),
        feature_schema_hash=SCHEMA,
        local_values=np.arange(96, dtype=np.float32).reshape(12, 8),
        local_masks=np.ones((12, 8), dtype=np.uint8),
        context_values=np.arange(48, dtype=np.float32).reshape(8, 6),
        context_masks=np.ones((8, 6), dtype=np.uint8),
        static_values=np.arange(4, dtype=np.float32),
        static_masks=np.ones(4, dtype=np.uint8),
        current_state=ConfirmedState.POSITIVE,
        model_eligible=True,
        observed_imbalance_minutes=12,
        created_at=NOW,
    )


def _write_bundle(root: Path) -> Path:
    root.mkdir()
    sample = (
        torch.zeros((1, 12, 8)),
        torch.ones((1, 12, 8)),
        torch.zeros((1, 8, 6)),
        torch.ones((1, 8, 6)),
        torch.zeros((1, 4)),
        torch.ones((1, 4)),
    )
    members: list[str] = []
    for index, seed in enumerate((17, 29, 43)):
        torch.manual_seed(seed)
        filename = f"member-{index}.onnx"
        export_member(
            ImbalanceForecaster(
                local_features=8,
                context_features=6,
                static_features=4,
                d_model=16,
                tcn_blocks=1,
                transformer_layers=1,
                transformer_heads=4,
            ),
            sample,
            root / filename,
        )
        members.append(filename)
    preprocessing = {
        "version": 1,
        "feature_schema_hash": SCHEMA,
        "local": {
            "names": [f"local_{index}" for index in range(8)],
            "location": [0] * 8,
            "scale": [1] * 8,
        },
        "context": {
            "names": [f"context_{index}" for index in range(6)],
            "location": [0] * 6,
            "scale": [1] * 6,
        },
        "static": {
            "names": [f"static_{index}" for index in range(4)],
            "location": [0] * 4,
            "scale": [1] * 4,
        },
    }
    artifacts = {
        "preprocessing.json": json.dumps(preprocessing, sort_keys=True),
        "calibration.json": json.dumps(
            {"x_thresholds": [0.0, 1.0], "y_thresholds": [0.0, 1.0], "decision_threshold": 0.5},
            sort_keys=True,
        ),
        "evaluation.json": "{}",
    }
    for filename, content in artifacts.items():
        (root / filename).write_text(content, encoding="utf-8")
    checksums = {
        filename: hashlib.sha256((root / filename).read_bytes()).hexdigest()
        for filename in (*members, *artifacts)
    }
    manifest = {
        "model_version": "test-v1",
        "created_at": "2026-07-13T00:00:00Z",
        "training_period": {"start": "2025-01-01T00:00:00Z", "end": "2025-02-01T00:00:00Z"},
        "feature_schema_hash": SCHEMA,
        "members": members,
        "preprocessing_file": "preprocessing.json",
        "calibration_file": "calibration.json",
        "evaluation_file": "evaluation.json",
        "runtime": {"opset": 18},
        "input_shapes": {"local": [-1, 12, 8], "context": [-1, 8, 6], "static": [-1, 4]},
        "output_names": [
            "mixture_logits",
            "mixture_means",
            "mixture_log_scales",
            "flip_logit",
            "delta",
            "auxiliary_horizons",
        ],
        "checksums": checksums,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return root
