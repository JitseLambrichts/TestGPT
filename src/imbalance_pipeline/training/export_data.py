import hashlib
import json
import os
import shutil
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import chain, islice
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from imbalance_pipeline.domain.imbalance import ConfirmedState
from imbalance_pipeline.training.data import TrainingExample
from imbalance_pipeline.training.splits import IndexRange

_DEFAULT_SHARD_SIZE = 4_096
_FORMAT_VERSION = 1


@dataclass(frozen=True, slots=True)
class TrainingShard:
    name: str
    count: int
    checksum: str


@dataclass(frozen=True, slots=True)
class TrainingDataset:
    root: Path
    feature_schema_hash: str
    dataset_digest: str
    shards: tuple[TrainingShard, ...]
    _verified_shards: set[str] = field(default_factory=set, repr=False, compare=False)

    @property
    def count(self) -> int:
        return sum(shard.count for shard in self.shards)

    def cutoffs(self) -> list[datetime]:
        values: list[datetime] = []
        for shard in self.shards:
            self._verify_shard(shard)
            values.extend(
                _load_shard_cutoffs(
                    self.root / shard.name,
                    expected_count=shard.count,
                )
            )
        return values

    def example_at(self, index: int) -> TrainingExample:
        if index < 0 or index >= self.count:
            raise IndexError("training dataset example index is outside the dataset")
        return next(self.iter_range(IndexRange(index, index + 1)))

    def iter_range(self, section: IndexRange) -> Iterator[TrainingExample]:
        if section.stop > self.count:
            raise ValueError("training dataset range is outside the dataset")
        offset = 0
        for shard in self.shards:
            shard_start = offset
            shard_stop = offset + shard.count
            offset = shard_stop
            start = max(section.start, shard_start)
            stop = min(section.stop, shard_stop)
            if start >= stop:
                continue
            self._verify_shard(shard)
            yield from _load_shard_indices(
                self.root / shard.name,
                self.feature_schema_hash,
                range(start - shard_start, stop - shard_start),
            )

    def iter_batches(
        self,
        section: IndexRange,
        *,
        batch_size: int,
        seed: int | None = None,
    ) -> Iterator[list[TrainingExample]]:
        if batch_size <= 0:
            raise ValueError("training dataset batch_size must be positive")
        if section.stop > self.count:
            raise ValueError("training dataset range is outside the dataset")
        chunks: list[tuple[TrainingShard, int, int]] = []
        offset = 0
        for shard in self.shards:
            start = max(section.start, offset)
            stop = min(section.stop, offset + shard.count)
            if start < stop:
                chunks.append((shard, start - offset, stop - offset))
            offset += shard.count
        generator = np.random.default_rng(seed)
        order = generator.permutation(len(chunks)) if seed is not None else np.arange(len(chunks))
        batch: list[TrainingExample] = []
        for chunk_index in order:
            shard, start, stop = chunks[int(chunk_index)]
            self._verify_shard(shard)
            indices = np.arange(start, stop)
            if seed is not None:
                generator.shuffle(indices)
            for example in _load_shard_indices(
                self.root / shard.name,
                self.feature_schema_hash,
                indices.tolist(),
            ):
                batch.append(example)
                if len(batch) == batch_size:
                    yield batch
                    batch = []
        if batch:
            yield batch

    def _verify_shard(self, shard: TrainingShard) -> None:
        """Verify each immutable shard once for this opened training run."""
        if shard.name in self._verified_shards:
            return
        path = self.root / shard.name
        if _sha256(path) != shard.checksum:
            raise ValueError(f"checksum mismatch for training dataset shard {path.name}")
        self._verified_shards.add(shard.name)


def write_training_dataset(
    examples: Iterable[TrainingExample],
    output: Path,
    *,
    force: bool = False,
    shard_size: int = _DEFAULT_SHARD_SIZE,
) -> Path:
    if shard_size <= 0:
        raise ValueError("training dataset shard_size must be positive")
    iterator = iter(examples)
    try:
        first = next(iterator)
    except StopIteration as exc:
        raise ValueError("cannot export an empty training dataset") from exc
    schema_hash = first.feature_schema_hash
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
        count = 0
        minimum = first.cutoff
        maximum = first.cutoff
        shards: list[dict[str, object]] = []
        for index, chunk in enumerate(_chunks(chain((first,), iterator), shard_size)):
            if any(example.feature_schema_hash != schema_hash for example in chunk):
                raise ValueError("training dataset must use one feature schema hash")
            shard = temporary / f"shard-{index:05d}.npz"
            np.savez_compressed(shard, **_arrays(chunk))  # type: ignore[arg-type]
            shards.append(
                {
                    "checksum": _sha256(shard),
                    "count": len(chunk),
                    "name": shard.name,
                }
            )
            count += len(chunk)
            minimum = min(minimum, *(example.cutoff for example in chunk))
            maximum = max(maximum, *(example.cutoff for example in chunk))
        metadata = {
            "count": count,
            "feature_schema_hash": schema_hash,
            "format_version": _FORMAT_VERSION,
            "max_cutoff": maximum.isoformat(),
            "min_cutoff": minimum.isoformat(),
            "shards": shards,
        }
        metadata["dataset_digest"] = _dataset_digest(metadata)
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
    opened = open_training_dataset(dataset)
    return list(opened.iter_range(IndexRange(0, opened.count)))


def open_training_dataset(dataset: Path) -> TrainingDataset:
    root = Path(dataset)
    metadata = _metadata(root)
    schema_hash = _required_string(metadata, "feature_schema_hash")
    raw_shards = metadata.get("shards")
    if not isinstance(raw_shards, list) or not raw_shards:
        raise ValueError("training dataset metadata requires non-empty shards")
    shards: list[TrainingShard] = []
    for shard in raw_shards:
        if not isinstance(shard, dict):
            raise ValueError("training dataset shard metadata is invalid")
        name = _required_string(shard, "name")
        expected_checksum = _required_string(shard, "checksum")
        count = shard.get("count")
        if not isinstance(count, int) or count <= 0:
            raise ValueError("training dataset shard count is invalid")
        path = root / name
        if not path.is_file():
            raise ValueError(f"missing training dataset shard {name}")
        shards.append(TrainingShard(name=name, count=count, checksum=expected_checksum))
    expected_count = metadata.get("count")
    shard_count = sum(shard.count for shard in shards)
    if not isinstance(expected_count, int) or expected_count != shard_count:
        raise ValueError("training dataset count does not match metadata")
    computed_digest = _dataset_digest(metadata)
    declared_digest = metadata.get("dataset_digest")
    if declared_digest is not None and declared_digest != computed_digest:
        raise ValueError("training dataset metadata digest does not match shard manifest")
    return TrainingDataset(
        root=root,
        feature_schema_hash=schema_hash,
        # Dataset v1 exports predate this field. Their shard checksums are still
        # part of the canonical digest, so retain read compatibility while every
        # newly written export persists the digest explicitly.
        dataset_digest=computed_digest,
        shards=tuple(shards),
    )


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


def _chunks(
    examples: Iterable[TrainingExample],
    size: int,
) -> Iterator[list[TrainingExample]]:
    iterator = iter(examples)
    while chunk := list(islice(iterator, size)):
        yield chunk


def _load_shard(path: Path, schema_hash: str) -> list[TrainingExample]:
    return list(_load_shard_indices(path, schema_hash, None))


def _load_shard_cutoffs(
    path: Path,
    *,
    expected_count: int,
) -> list[datetime]:
    try:
        with np.load(path, allow_pickle=False) as values:
            cutoffs = np.asarray(values["cutoff"])
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f"cannot read training dataset shard {path.name}") from exc
    if len(cutoffs) != expected_count:
        raise ValueError(f"training dataset shard {path.name} has inconsistent row counts")
    return [_utc_datetime(str(value)) for value in cutoffs]


def _load_shard_indices(
    path: Path,
    schema_hash: str,
    indices: Sequence[int] | None,
) -> Iterator[TrainingExample]:
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
    selected: Sequence[int] = range(count) if indices is None else indices
    if any(index < 0 or index >= count for index in selected):
        raise ValueError(f"training dataset shard {path.name} range is outside the shard")
    for index in selected:
        yield _example_from_arrays(arrays, index, schema_hash)


def _example_from_arrays(
    arrays: dict[str, NDArray[np.generic]],
    index: int,
    schema_hash: str,
) -> TrainingExample:
    return TrainingExample(
        event_id=str(arrays["event_id"][index]),
        cutoff=_utc_datetime(str(arrays["cutoff"][index])),
        feature_schema_hash=schema_hash,
        local_values=np.array(arrays["local_values"][index], dtype=np.float32, copy=True),
        local_masks=np.array(arrays["local_masks"][index], dtype=np.uint8, copy=True),
        context_values=np.array(arrays["context_values"][index], dtype=np.float32, copy=True),
        context_masks=np.array(arrays["context_masks"][index], dtype=np.uint8, copy=True),
        static_values=np.array(arrays["static_values"][index], dtype=np.float32, copy=True),
        static_masks=np.array(arrays["static_masks"][index], dtype=np.uint8, copy=True),
        target_next=float(arrays["target_next"][index]),
        target_delta=float(arrays["target_delta"][index]),
        delta_mask=float(arrays["delta_mask"][index]),
        auxiliary_target=np.array(arrays["auxiliary_target"][index], dtype=np.float32, copy=True),
        auxiliary_mask=np.array(arrays["auxiliary_mask"][index], dtype=np.float32, copy=True),
        current_state=_state_from_value(arrays["current_state"][index]),
        flip_target=float(arrays["flip_target"][index]),
        flip_mask=float(arrays["flip_mask"][index]),
    )


def _metadata(root: Path) -> dict[str, object]:
    try:
        value = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("cannot read training dataset metadata") from exc
    if not isinstance(value, dict) or value.get("format_version") != _FORMAT_VERSION:
        raise ValueError("unsupported training dataset metadata")
    return value


def _dataset_digest(metadata: dict[str, object]) -> str:
    """Hash canonical dataset metadata, including every declared shard checksum."""
    canonical = {key: value for key, value in metadata.items() if key != "dataset_digest"}
    encoded = json.dumps(canonical, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


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
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1_048_576):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    raise SystemExit(
        "dataset construction is available through write_training_dataset; "
        "the ClickHouse point-in-time exporter is not configured by this command yet"
    )
