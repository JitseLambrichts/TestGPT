import hashlib
import json
from pathlib import Path

import pytest

from imbalance_pipeline.model.bundle import ModelManifest, promote_bundle, validate_bundle

SCHEMA = "a" * 64


def write_bundle(root: Path, *, schema: str = SCHEMA) -> Path:
    root.mkdir()
    artifacts = {
        "member-0.onnx": b"member-0",
        "member-1.onnx": b"member-1",
        "member-2.onnx": b"member-2",
        "preprocessing.json": b'{}',
        "calibration.json": b'{"x_thresholds":[0,1],"y_thresholds":[0,1],"decision_threshold":0.5}',
        "evaluation.json": b"{}",
    }
    for name, content in artifacts.items():
        (root / name).write_bytes(content)
    manifest = {
        "model_version": "test-v1",
        "created_at": "2026-07-13T00:00:00Z",
        "training_period": {"start": "2025-01-01T00:00:00Z", "end": "2025-02-01T00:00:00Z"},
        "feature_schema_hash": schema,
        "members": ["member-0.onnx", "member-1.onnx", "member-2.onnx"],
        "preprocessing_file": "preprocessing.json",
        "calibration_file": "calibration.json",
        "evaluation_file": "evaluation.json",
        "runtime": {"opset": 18},
        "input_shapes": {"local": [-1, 180, 35]},
        "output_names": [
            "mixture_logits",
            "mixture_means",
            "mixture_log_scales",
            "flip_logit",
            "delta",
            "auxiliary_horizons",
        ],
        "checksums": {
            name: hashlib.sha256(content).hexdigest() for name, content in artifacts.items()
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_manifest_validates_hashes_schema_and_safe_artifact_paths(tmp_path: Path) -> None:
    bundle = write_bundle(tmp_path / "bundle")

    manifest = ModelManifest.load(bundle)
    result = validate_bundle(bundle, expected_schema_hash=SCHEMA)

    assert manifest.model_version == "test-v1"
    assert result.valid is True
    assert result.reason is None

    (bundle / "member-0.onnx").write_bytes(b"tampered")
    tampered = validate_bundle(bundle, expected_schema_hash=SCHEMA)
    mismatched = validate_bundle(bundle, expected_schema_hash="b" * 64)

    assert tampered.valid is False
    assert "checksum" in (tampered.reason or "")
    assert mismatched.valid is False


def test_manifest_rejects_wrong_member_count_and_path_traversal(tmp_path: Path) -> None:
    bundle = write_bundle(tmp_path / "bundle")
    payload = json.loads((bundle / "manifest.json").read_text())
    payload["members"] = ["../member.onnx"]
    (bundle / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid model manifest"):
        ModelManifest.load(bundle)


@pytest.mark.parametrize("model_version", ["../outside", "/outside", "production", "."])
def test_manifest_rejects_unsafe_model_versions(tmp_path: Path, model_version: str) -> None:
    bundle = write_bundle(tmp_path / "bundle")
    payload = json.loads((bundle / "manifest.json").read_text())
    payload["model_version"] = model_version
    (bundle / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid model manifest"):
        ModelManifest.load(bundle)


def test_manifest_requires_unique_members_and_validate_rejects_symlink_artifacts(
    tmp_path: Path,
) -> None:
    duplicate = write_bundle(tmp_path / "duplicate")
    payload = json.loads((duplicate / "manifest.json").read_text())
    payload["members"] = ["member-0.onnx", "member-0.onnx", "member-2.onnx"]
    (duplicate / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid model manifest"):
        ModelManifest.load(duplicate)

    linked = write_bundle(tmp_path / "linked")
    outside = tmp_path / "outside.onnx"
    outside.write_bytes(b"member-0")
    (linked / "member-0.onnx").unlink()
    (linked / "member-0.onnx").symlink_to(outside)

    validation = validate_bundle(linked, expected_schema_hash=SCHEMA)

    assert validation.valid is False
    assert "symlink" in (validation.reason or "")


def test_promotion_copies_only_validated_artifacts_and_revalidates_before_activation(
    tmp_path: Path,
) -> None:
    candidate = write_bundle(tmp_path / "candidate")
    (candidate / "untracked.txt").write_text("must not be promoted", encoding="utf-8")

    destination = promote_bundle(candidate, tmp_path / "models")

    assert destination == tmp_path / "models" / "test-v1"
    assert not (destination / "untracked.txt").exists()
    assert validate_bundle(destination, expected_schema_hash=SCHEMA).valid is True
    assert (tmp_path / "models" / "production").resolve() == destination
