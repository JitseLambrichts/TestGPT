import hashlib
import json
import os
import shutil
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from imbalance_pipeline.domain.imbalance import ConfirmedState
from imbalance_pipeline.training.data import TrainingExample

_SHARD_NAME = "shard-00000.npz"
_FORMAT_VERSION = 1


def write_training_dataset(
    examples: Sequence[TrainingExample],
    output: Path,
    *,
    force: bool = False,
) -> Path:
    if not examples:
        raise ValueError("cannot export an empty training dataset")
    schema_hash = _shared_schema(examples)
    output = Path(output)
    if output.exists():
        if not force:
            raise FileExistsError(f"refusing to overwrite training dataset: {output}")
        if output.is_dir():
            shutil.rmtree(output)
        else:
            output.unlink()
    temporary = output.parent / f".{output.name}.tmp"
    if temporary.exists():
        if temporary.is_dir():
            shutil.rmtree(temporary)
        else:
            temporary.unlink()
    temporary.mkdir(parents=True)
    try:
        shard = temporary / _SHARD_NAME
        np.savez_compressed(shard, **_arrays(examples))  # type: ignore[arg-type]
        checksum = _sha256(shard)
        metadata = {
            "count": len(examples),
            "feature_schema_hash": schema_hash,
            "format_version": _FORMAT_VERSION,
            "max_cutoff": max(example.cutoff for example in examples).isoformat(),
            "min_cutoff": min(example.cutoff for example in examples).isoformat(),
            "shards": [{"checksum": checksum, "count": len(examples), "name": _SHARD_NAME}],
        }
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, separators=(",", ":"), sort_keys=True),
            encoding="utf-8",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def load_training_dataset(dataset: Path) -> list[TrainingExample]:
    root = Path(dataset)
    metadata = _metadata(root)
    schema_hash = _required_string(metadata, "feature_schema_hash")
    shards = metadata.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("training dataset metadata requires non-empty shards")
    loaded: list[TrainingExample] = []
    for shard in shards:
        if not isinstance(shard, dict):
            raise ValueError("training dataset shard metadata is invalid")
        name = _required_string(shard, "name")
        expected_checksum = _required_string(shard, "checksum")
        path = root / name
        if not path.is_file():
            raise ValueError(f"missing training dataset shard {name}")
        if _sha256(path) != expected_checksum:
            raise ValueError(f"checksum mismatch for training dataset shard {name}")
        loaded.extend(_load_shard(path, schema_hash))
    expected_count = metadata.get("count")
    if not isinstance(expected_count, int) or expected_count != len(loaded):
        raise ValueError("training dataset count does not match metadata")
    return loaded


def _arrays(examples: Sequence[TrainingExample]) -> dict[str, NDArray[np.generic]]:
    return {
        "auxiliary_mask": np.stack([example.auxiliary_mask for example in examples]),
        "auxiliary_target": np.stack([example.auxiliary_target for example in examples]),
        "current_state": np.asarray([_state_value(example.current_state) for example in examples]),
        "cutoff": np.asarray([example.cutoff.isoformat() for example in examples]),
        "delta_mask": np.asarray([example.delta_mask for example in examples], dtype=np.float32),
        "event_id": np.asarray([example.event_id for example in examples]),
        "flip_mask": np.asarray([example.flip_mask for example in examples], dtype=np.float32),
        "flip_target": np.asarray([example.flip_target for example in examples], dtype=np.float32),
        "local_masks": np.stack([example.local_masks for example in examples]),
        "local_values": np.stack([example.local_values for example in examples]),
        "context_masks": np.stack([example.context_masks for example in examples]),
        "context_values": np.stack([example.context_values for example in examples]),
        "static_masks": np.stack([example.static_masks for example in examples]),
        "static_values": np.stack([example.static_values for example in examples]),
        "target_delta": np.asarray(
            [example.target_delta for example in examples], dtype=np.float32
        ),
        "target_next": np.asarray([example.target_next for example in examples], dtype=np.float32),
    }


def _load_shard(path: Path, schema_hash: str) -> list[TrainingExample]:
    try:
        with np.load(path, allow_pickle=False) as values:
            arrays = {name: np.asarray(values[name]) for name in values.files}
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f"cannot read training dataset shard {path.name}") from exc
    required = {
        "auxiliary_mask",
        "auxiliary_target",
        "current_state",
        "cutoff",
        "delta_mask",
        "event_id",
        "flip_mask",
        "flip_target",
        "local_masks",
        "local_values",
        "context_masks",
        "context_values",
        "static_masks",
        "static_values",
        "target_delta",
        "target_next",
    }
    if arrays.keys() != required:
        raise ValueError(f"training dataset shard {path.name} has an invalid field set")
    count = len(arrays["event_id"])
    if count == 0 or any(array.shape[0] != count for array in arrays.values()):
        raise ValueError(f"training dataset shard {path.name} has inconsistent row counts")
    _binary_masks(arrays, path.name)
    return [
        TrainingExample(
            event_id=str(arrays["event_id"][index]),
            cutoff=_utc_datetime(str(arrays["cutoff"][index])),
            feature_schema_hash=schema_hash,
            local_values=np.asarray(arrays["local_values"][index], dtype=np.float32),
            local_masks=np.asarray(arrays["local_masks"][index], dtype=np.uint8),
            context_values=np.asarray(arrays["context_values"][index], dtype=np.float32),
            context_masks=np.asarray(arrays["context_masks"][index], dtype=np.uint8),
            static_values=np.asarray(arrays["static_values"][index], dtype=np.float32),
            static_masks=np.asarray(arrays["static_masks"][index], dtype=np.uint8),
            target_next=float(arrays["target_next"][index]),
            target_delta=float(arrays["target_delta"][index]),
            delta_mask=float(arrays["delta_mask"][index]),
            auxiliary_target=np.asarray(arrays["auxiliary_target"][index], dtype=np.float32),
            auxiliary_mask=np.asarray(arrays["auxiliary_mask"][index], dtype=np.float32),
            current_state=_state_from_value(arrays["current_state"][index]),
            flip_target=float(arrays["flip_target"][index]),
            flip_mask=float(arrays["flip_mask"][index]),
        )
        for index in range(count)
    ]


def _metadata(root: Path) -> dict[str, object]:
    try:
        value = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("cannot read training dataset metadata") from exc
    if not isinstance(value, dict) or value.get("format_version") != _FORMAT_VERSION:
        raise ValueError("unsupported training dataset metadata")
    return value


def _shared_schema(examples: Sequence[TrainingExample]) -> str:
    schemas = {example.feature_schema_hash for example in examples}
    if len(schemas) != 1:
        raise ValueError("training dataset must use one feature schema hash")
    return next(iter(schemas))


def _required_string(value: dict[str, object], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise ValueError(f"training dataset metadata requires {key}")
    return result


def _binary_masks(arrays: dict[str, NDArray[np.generic]], shard_name: str) -> None:
    for key in ("local_masks", "context_masks", "static_masks"):
        mask = arrays[key]
        if not np.all((mask == 0) | (mask == 1)):
            raise ValueError(f"training dataset shard {shard_name} has non-binary {key}")


def _state_value(value: ConfirmedState | None) -> int:
    if value is ConfirmedState.POSITIVE:
        return 1
    if value is ConfirmedState.NEGATIVE:
        return -1
    return 0


def _state_from_value(value: object) -> ConfirmedState | None:
    if not isinstance(value, (int, np.integer)):
        raise ValueError("training dataset current state is invalid")
    numeric = int(value)
    if numeric == 1:
        return ConfirmedState.POSITIVE
    if numeric == -1:
        return ConfirmedState.NEGATIVE
    if numeric == 0:
        return None
    raise ValueError("training dataset current state is invalid")


def _utc_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("training dataset cutoff must be timezone-aware")
    return parsed.astimezone(UTC)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    raise SystemExit(
        "dataset construction is available through write_training_dataset; "
        "the ClickHouse point-in-time exporter is not configured by this command yet"
    )
