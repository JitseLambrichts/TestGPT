import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from imbalance_pipeline.domain.imbalance import ConfirmedState
from imbalance_pipeline.features.engine import FeatureSnapshot
from imbalance_pipeline.features.schema import DEFAULT_FEATURE_REGISTRY
from imbalance_pipeline.model.bundle import ModelManifest, validate_bundle
from imbalance_pipeline.serving.runtime import OnnxEnsemble


def test_build_test_bundle_generates_a_valid_synthetic_bundle(tmp_path: Path) -> None:
    output = tmp_path / "test-bundle"

    subprocess.run(
        [
            sys.executable,
            "scripts/build_test_bundle.py",
            "--output",
            str(output),
        ],
        check=True,
    )

    manifest = ModelManifest.load(output)
    validation = validate_bundle(
        output,
        expected_schema_hash=DEFAULT_FEATURE_REGISTRY.fingerprint,
    )

    assert validation.valid is True
    assert manifest.model_version == "synthetic-test-v1"
    assert manifest.input_shapes["local"] == [-1, 180, len(DEFAULT_FEATURE_REGISTRY.local_names)]


def test_build_test_bundle_requires_explicit_force_to_replace_output(tmp_path: Path) -> None:
    output = tmp_path / "test-bundle"
    output.mkdir()
    sentinel = output / "keep-me"
    sentinel.write_text("existing", encoding="utf-8")

    rejected = subprocess.run(
        [sys.executable, "scripts/build_test_bundle.py", "--output", str(output)],
        check=False,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/build_test_bundle.py",
            "--output",
            str(output),
            "--force",
        ],
        check=True,
    )

    assert rejected.returncode != 0
    assert not sentinel.exists()
    assert (output / "manifest.json").is_file()


def test_committed_synthetic_bundle_loads_in_the_cpu_runtime() -> None:
    registry = DEFAULT_FEATURE_REGISTRY
    now = datetime(2026, 7, 13, 10, 0, tzinfo=UTC)
    snapshot = FeatureSnapshot(
        event_id="synthetic-feature-001",
        cutoff=now,
        knowledge_cutoff=now,
        target_time=now + timedelta(minutes=1),
        feature_schema_hash=registry.fingerprint,
        local_values=np.zeros(
            (registry.local_window_minutes, len(registry.local_names)), dtype=np.float32
        ),
        local_masks=np.ones(
            (registry.local_window_minutes, len(registry.local_names)), dtype=np.uint8
        ),
        context_values=np.zeros(
            (registry.context_steps, len(registry.context_names)), dtype=np.float32
        ),
        context_masks=np.ones(
            (registry.context_steps, len(registry.context_names)), dtype=np.uint8
        ),
        static_values=np.zeros(len(registry.static_names), dtype=np.float32),
        static_masks=np.ones(len(registry.static_names), dtype=np.uint8),
        current_state=ConfirmedState.POSITIVE,
        model_eligible=True,
        observed_imbalance_minutes=registry.local_window_minutes,
        created_at=now,
    )

    values = OnnxEnsemble(Path("models/test"), expected_schema_hash=registry.fingerprint).predict(
        snapshot
    )

    assert values.model_version == "synthetic-test-v1"
    assert values.p10_mw <= values.system_imbalance_mw <= values.p90_mw
