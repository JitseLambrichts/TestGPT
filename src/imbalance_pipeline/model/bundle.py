import hashlib
import json
import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath


@dataclass(frozen=True, slots=True)
class ModelManifest:
    model_version: str
    created_at: datetime
    training_period: Mapping[str, str]
    feature_schema_hash: str
    members: tuple[str, str, str]
    preprocessing_file: str
    calibration_file: str
    evaluation_file: str
    runtime: Mapping[str, object]
    input_shapes: Mapping[str, object]
    output_names: tuple[str, ...]
    checksums: Mapping[str, str]

    @classmethod
    def load(cls, bundle_dir: Path) -> "ModelManifest":
        manifest_path = bundle_dir / "manifest.json"
        if manifest_path.is_symlink():
            raise ValueError("model manifest cannot be a symlink")
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("cannot read model manifest") from exc
        if not isinstance(raw, dict):
            raise ValueError("model manifest must be an object")
        try:
            members = _member_paths(raw["members"])
            artifact_paths = (
                *members,
                _safe_path(raw["preprocessing_file"]),
                _safe_path(raw["calibration_file"]),
                _safe_path(raw["evaluation_file"]),
            )
            checksums = _checksums(raw["checksums"], artifact_paths)
            created_at = _utc_datetime(raw["created_at"])
            model_version = _safe_model_version(raw["model_version"])
            schema = _nonempty_string(raw["feature_schema_hash"], "feature_schema_hash")
            training_period = _string_mapping(raw["training_period"], "training_period")
            runtime = _mapping(raw["runtime"], "runtime")
            input_shapes = _mapping(raw["input_shapes"], "input_shapes")
            output_names = _string_tuple(raw["output_names"], "output_names")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid model manifest") from exc
        return cls(
            model_version=model_version,
            created_at=created_at,
            training_period=training_period,
            feature_schema_hash=schema,
            members=members,
            preprocessing_file=_safe_path(raw["preprocessing_file"]),
            calibration_file=_safe_path(raw["calibration_file"]),
            evaluation_file=_safe_path(raw["evaluation_file"]),
            runtime=runtime,
            input_shapes=input_shapes,
            output_names=output_names,
            checksums=checksums,
        )

    @property
    def artifact_paths(self) -> tuple[str, ...]:
        return (*self.members, self.preprocessing_file, self.calibration_file, self.evaluation_file)


@dataclass(frozen=True, slots=True)
class ValidationResult:
    valid: bool
    reason: str | None = None


def validate_bundle(bundle_dir: Path, *, expected_schema_hash: str) -> ValidationResult:
    try:
        manifest = ModelManifest.load(bundle_dir)
        if manifest.feature_schema_hash != expected_schema_hash:
            return ValidationResult(False, "feature schema hash does not match")
        for artifact in manifest.artifact_paths:
            path = bundle_dir / artifact
            if path.is_symlink():
                return ValidationResult(False, f"artifact {artifact} cannot be a symlink")
            if not path.is_file():
                return ValidationResult(False, f"missing artifact {artifact}")
            checksum = hashlib.sha256(path.read_bytes()).hexdigest()
            if checksum != manifest.checksums[artifact]:
                return ValidationResult(False, f"checksum mismatch for {artifact}")
    except ValueError as exc:
        return ValidationResult(False, str(exc))
    return ValidationResult(True)


def promote_bundle(candidate: Path, model_root: Path) -> Path:
    manifest = ModelManifest.load(candidate)
    validation = validate_bundle(candidate, expected_schema_hash=manifest.feature_schema_hash)
    if not validation.valid:
        raise ValueError(validation.reason or "candidate bundle is invalid")
    model_root.mkdir(parents=True, exist_ok=True)
    destination = model_root / manifest.model_version
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"model version already exists: {manifest.model_version}")
    staging = model_root / f".{manifest.model_version}.tmp"
    if staging.exists() or staging.is_symlink():
        raise FileExistsError(f"model staging directory already exists: {staging.name}")
    try:
        staging.mkdir()
        for artifact in ("manifest.json", *manifest.artifact_paths):
            source = candidate / artifact
            if source.is_symlink() or not source.is_file():
                raise ValueError(f"candidate artifact {artifact} is not a regular file")
            destination_artifact = staging / artifact
            shutil.copy2(source, destination_artifact)
            _fsync(destination_artifact)
        copied = validate_bundle(staging, expected_schema_hash=manifest.feature_schema_hash)
        if not copied.valid:
            raise ValueError(copied.reason or "copied bundle validation failed")
        _fsync(staging)
        os.replace(staging, destination)
    except Exception:
        if staging.exists() and staging.is_dir() and not staging.is_symlink():
            shutil.rmtree(staging)
        raise
    production = model_root / "production"
    temporary = model_root / ".production.tmp"
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError("production symlink staging path already exists")
    if production.exists() and not production.is_symlink():
        raise FileExistsError("production path must be a symlink")
    temporary.symlink_to(manifest.model_version)
    os.replace(temporary, production)
    _fsync(model_root)
    return destination


def _member_paths(value: object) -> tuple[str, str, str]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError("model manifest requires exactly three members")
    members = _string_tuple(value, "members")
    if len(set(members)) != len(members):
        raise ValueError("model manifest members must be unique")
    return members[0], members[1], members[2]


def _checksums(value: object, required: tuple[str, ...]) -> Mapping[str, str]:
    mapping = _mapping(value, "checksums")
    checksums: dict[str, str] = {}
    for path in required:
        checksum = mapping.get(path)
        if not isinstance(checksum, str) or len(checksum) != 64:
            raise ValueError(f"missing checksum for {path}")
        checksums[path] = checksum
    members = required[:3]
    if len({checksums[member] for member in members}) != len(members):
        raise ValueError("model manifest member checksums must be unique")
    return checksums


def _safe_path(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("artifact paths must be non-empty strings")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or len(path.parts) != 1:
        raise ValueError("artifact paths must be bundle-relative filenames")
    return value


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _safe_model_version(value: object) -> str:
    version = _nonempty_string(value, "model_version")
    path = PurePosixPath(version)
    if (
        path.is_absolute()
        or ".." in path.parts
        or len(path.parts) != 1
        or version in {".", "production", ".production.tmp"}
    ):
        raise ValueError("model_version must be a safe version directory name")
    return version


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"{name} must be a list of non-empty strings")
    return tuple(_safe_path(item) if name == "members" else item for item in value)


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _string_mapping(value: object, name: str) -> Mapping[str, str]:
    mapping = _mapping(value, name)
    if not all(isinstance(key, str) and isinstance(item, str) for key, item in mapping.items()):
        raise ValueError(f"{name} must map strings to strings")
    return {key: item for key, item in mapping.items() if isinstance(item, str)}


def _utc_datetime(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("created_at must be an ISO timestamp")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("created_at must be timezone-aware")
    return parsed.astimezone(UTC)


def _fsync(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
