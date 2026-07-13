import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from imbalance_pipeline.domain.imbalance import ConfirmedState, ImbalanceObservation, advance_state
from imbalance_pipeline.features.engine import FeatureSnapshot
from imbalance_pipeline.features.schema import DEFAULT_FEATURE_REGISTRY

_EPSILON = 1e-6
_TRANSFORM_CLIP = 12.0
_AUXILIARY_HORIZONS_MINUTES = (2, 5, 10)


@dataclass(frozen=True, slots=True)
class PreparedFeatures:
    local_values: NDArray[np.float32]
    local_masks: NDArray[np.uint8]
    context_values: NDArray[np.float32]
    context_masks: NDArray[np.uint8]
    static_values: NDArray[np.float32]
    static_masks: NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class TrainingExample:
    event_id: str
    cutoff: datetime
    feature_schema_hash: str
    local_values: NDArray[np.float32]
    local_masks: NDArray[np.uint8]
    context_values: NDArray[np.float32]
    context_masks: NDArray[np.uint8]
    static_values: NDArray[np.float32]
    static_masks: NDArray[np.uint8]
    target_next: float
    target_delta: float
    delta_mask: float
    auxiliary_target: NDArray[np.float32]
    auxiliary_mask: NDArray[np.float32]
    current_state: ConfirmedState | None
    flip_target: float
    flip_mask: float


@dataclass(frozen=True, slots=True)
class TrainingBatch:
    local: Tensor
    local_mask: Tensor
    context: Tensor
    context_mask: Tensor
    static: Tensor
    static_mask: Tensor
    target_next: Tensor
    target_delta: Tensor
    delta_mask: Tensor
    auxiliary_target: Tensor
    auxiliary_mask: Tensor
    flip_target: Tensor
    flip_mask: Tensor
    current_state: Tensor
    batch_id: str

    @classmethod
    def from_examples(
        cls,
        examples: Sequence[TrainingExample],
        *,
        batch_id: str,
    ) -> "TrainingBatch":
        if not examples:
            raise ValueError("cannot build a training batch without examples")
        _require_same_schema(examples)
        return cls(
            local=_stack_tensor(examples, "local_values"),
            local_mask=_stack_tensor(examples, "local_masks"),
            context=_stack_tensor(examples, "context_values"),
            context_mask=_stack_tensor(examples, "context_masks"),
            static=_stack_tensor(examples, "static_values"),
            static_mask=_stack_tensor(examples, "static_masks"),
            target_next=_float_tensor(examples, "target_next"),
            target_delta=_float_tensor(examples, "target_delta"),
            delta_mask=_float_tensor(examples, "delta_mask"),
            auxiliary_target=torch.from_numpy(
                np.stack([example.auxiliary_target for example in examples]).astype(np.float32)
            ),
            auxiliary_mask=torch.from_numpy(
                np.stack([example.auxiliary_mask for example in examples]).astype(np.float32)
            ),
            flip_target=_float_tensor(examples, "flip_target"),
            flip_mask=_float_tensor(examples, "flip_mask"),
            current_state=torch.tensor(
                [_state_value(example.current_state) for example in examples],
                dtype=torch.int64,
            ),
            batch_id=batch_id,
        )

    def to(self, device: torch.device | str) -> "TrainingBatch":
        return TrainingBatch(
            local=self.local.to(device),
            local_mask=self.local_mask.to(device),
            context=self.context.to(device),
            context_mask=self.context_mask.to(device),
            static=self.static.to(device),
            static_mask=self.static_mask.to(device),
            target_next=self.target_next.to(device),
            target_delta=self.target_delta.to(device),
            delta_mask=self.delta_mask.to(device),
            auxiliary_target=self.auxiliary_target.to(device),
            auxiliary_mask=self.auxiliary_mask.to(device),
            flip_target=self.flip_target.to(device),
            flip_mask=self.flip_mask.to(device),
            current_state=self.current_state.to(device),
            batch_id=self.batch_id,
        )


@dataclass(frozen=True, slots=True)
class RobustPreprocessor:
    feature_schema_hash: str
    local_names: tuple[str, ...]
    context_names: tuple[str, ...]
    static_names: tuple[str, ...]
    local_location: NDArray[np.float32]
    local_scale: NDArray[np.float32]
    context_location: NDArray[np.float32]
    context_scale: NDArray[np.float32]
    static_location: NDArray[np.float32]
    static_scale: NDArray[np.float32]

    @classmethod
    def fit(cls, examples: Sequence[TrainingExample]) -> "RobustPreprocessor":
        if not examples:
            raise ValueError("cannot fit preprocessing without training examples")
        _require_same_schema(examples)
        first = examples[0]
        local_location, local_scale = _robust_statistics(
            [example.local_values for example in examples],
            [example.local_masks for example in examples],
        )
        context_location, context_scale = _robust_statistics(
            [example.context_values for example in examples],
            [example.context_masks for example in examples],
        )
        static_location, static_scale = _robust_statistics(
            [example.static_values for example in examples],
            [example.static_masks for example in examples],
        )
        return cls(
            feature_schema_hash=first.feature_schema_hash,
            local_names=_feature_names("local", first.local_values.shape[-1]),
            context_names=_feature_names("context", first.context_values.shape[-1]),
            static_names=_feature_names("static", first.static_values.shape[-1]),
            local_location=local_location,
            local_scale=local_scale,
            context_location=context_location,
            context_scale=context_scale,
            static_location=static_location,
            static_scale=static_scale,
        )

    def transform(self, snapshot: FeatureSnapshot) -> PreparedFeatures:
        if snapshot.feature_schema_hash != self.feature_schema_hash:
            raise ValueError("feature schema hash does not match fitted preprocessing")
        return PreparedFeatures(
            local_values=_transform(
                snapshot.local_values,
                snapshot.local_masks,
                self.local_location,
                self.local_scale,
            ),
            local_masks=_binary_masks(snapshot.local_masks),
            context_values=_transform(
                snapshot.context_values,
                snapshot.context_masks,
                self.context_location,
                self.context_scale,
            ),
            context_masks=_binary_masks(snapshot.context_masks),
            static_values=_transform(
                snapshot.static_values,
                snapshot.static_masks,
                self.static_location,
                self.static_scale,
            ),
            static_masks=_binary_masks(snapshot.static_masks),
        )

    def to_json(self) -> str:
        return json.dumps(
            {
                "context": _serialization_block(
                    self.context_names,
                    self.context_location,
                    self.context_scale,
                ),
                "feature_schema_hash": self.feature_schema_hash,
                "local": _serialization_block(
                    self.local_names,
                    self.local_location,
                    self.local_scale,
                ),
                "static": _serialization_block(
                    self.static_names,
                    self.static_location,
                    self.static_scale,
                ),
                "version": 1,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, serialized: str) -> "RobustPreprocessor":
        payload = json.loads(serialized)
        if not isinstance(payload, dict) or payload.get("version") != 1:
            raise ValueError("unsupported preprocessing payload")
        schema_hash = payload.get("feature_schema_hash")
        if not isinstance(schema_hash, str) or not schema_hash:
            raise ValueError("preprocessing payload requires a feature schema hash")
        local = _deserialization_block(payload.get("local"), "local")
        context = _deserialization_block(payload.get("context"), "context")
        static = _deserialization_block(payload.get("static"), "static")
        return cls(
            feature_schema_hash=schema_hash,
            local_names=local[0],
            local_location=local[1],
            local_scale=local[2],
            context_names=context[0],
            context_location=context[1],
            context_scale=context[2],
            static_names=static[0],
            static_location=static[1],
            static_scale=static[2],
        )


def preprocess_training_examples(
    examples: Sequence[TrainingExample],
    preprocessor: RobustPreprocessor,
) -> list[TrainingExample]:
    transformed: list[TrainingExample] = []
    for example in examples:
        if example.feature_schema_hash != preprocessor.feature_schema_hash:
            raise ValueError("feature schema hash does not match fitted preprocessing")
        transformed.append(
            replace(
                example,
                local_values=_transform(
                    example.local_values,
                    example.local_masks,
                    preprocessor.local_location,
                    preprocessor.local_scale,
                ),
                local_masks=_binary_masks(example.local_masks),
                context_values=_transform(
                    example.context_values,
                    example.context_masks,
                    preprocessor.context_location,
                    preprocessor.context_scale,
                ),
                context_masks=_binary_masks(example.context_masks),
                static_values=_transform(
                    example.static_values,
                    example.static_masks,
                    preprocessor.static_location,
                    preprocessor.static_scale,
                ),
                static_masks=_binary_masks(example.static_masks),
            )
        )
    return transformed


def build_training_examples(
    snapshots: Sequence[FeatureSnapshot],
    observations: Mapping[datetime, float] | Sequence[ImbalanceObservation],
    *,
    deadband_mw: float = 10.0,
) -> list[TrainingExample]:
    if deadband_mw <= 0:
        raise ValueError("deadband_mw must be positive")
    values = _observation_values(observations)
    examples: list[TrainingExample] = []
    for snapshot in snapshots:
        cutoff = _utc(snapshot.cutoff)
        target_time = cutoff + timedelta(minutes=1)
        target_next = values.get(target_time)
        if target_next is None:
            continue
        current_observed = bool(_binary_masks(snapshot.local_masks)[-1, 0])
        current_value = float(snapshot.local_values[-1, 0]) if current_observed else 0.0
        auxiliary_target = np.zeros(len(_AUXILIARY_HORIZONS_MINUTES), dtype=np.float32)
        auxiliary_mask = np.zeros(len(_AUXILIARY_HORIZONS_MINUTES), dtype=np.float32)
        for index, horizon in enumerate(_AUXILIARY_HORIZONS_MINUTES):
            value = values.get(cutoff + timedelta(minutes=horizon))
            if value is not None:
                auxiliary_target[index] = value
                auxiliary_mask[index] = 1.0
        future_state = advance_state(snapshot.current_state, target_next, deadband_mw)
        flip_mask = 1.0 if snapshot.current_state is not None and future_state is not None else 0.0
        flip_target = float(future_state is not snapshot.current_state) if flip_mask else 0.0
        examples.append(
            TrainingExample(
                event_id=snapshot.event_id,
                cutoff=cutoff,
                feature_schema_hash=snapshot.feature_schema_hash,
                local_values=np.asarray(snapshot.local_values, dtype=np.float32),
                local_masks=_binary_masks(snapshot.local_masks),
                context_values=np.asarray(snapshot.context_values, dtype=np.float32),
                context_masks=_binary_masks(snapshot.context_masks),
                static_values=np.asarray(snapshot.static_values, dtype=np.float32),
                static_masks=_binary_masks(snapshot.static_masks),
                target_next=float(target_next),
                target_delta=float(target_next - current_value),
                delta_mask=float(current_observed),
                auxiliary_target=auxiliary_target,
                auxiliary_mask=auxiliary_mask,
                current_state=snapshot.current_state,
                flip_target=flip_target,
                flip_mask=flip_mask,
            )
        )
    return examples


def _observation_values(
    observations: Mapping[datetime, float] | Sequence[ImbalanceObservation],
) -> dict[datetime, float]:
    if isinstance(observations, Mapping):
        return {_utc(timestamp): float(value) for timestamp, value in observations.items()}
    return {
        _utc(observation.timestamp): float(observation.system_imbalance_mw)
        for observation in observations
    }


def _robust_statistics(
    values: Sequence[NDArray[np.float32]],
    masks: Sequence[NDArray[np.uint8]],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    if len(values) != len(masks) or not values:
        raise ValueError("values and masks must have equal non-zero lengths")
    flattened_values = np.concatenate([value.reshape(-1, value.shape[-1]) for value in values])
    flattened_masks = np.concatenate(
        [_binary_masks(mask).reshape(-1, mask.shape[-1]) for mask in masks]
    )
    if flattened_values.shape != flattened_masks.shape:
        raise ValueError("feature values and masks must have equal shapes")
    locations = np.zeros(flattened_values.shape[1], dtype=np.float32)
    scales = np.ones(flattened_values.shape[1], dtype=np.float32)
    for index in range(flattened_values.shape[1]):
        observed = flattened_values[flattened_masks[:, index] == 1, index]
        if observed.size:
            locations[index] = np.float32(np.median(observed))
            lower, upper = np.quantile(observed, (0.25, 0.75))
            scales[index] = np.float32(max(float(upper - lower), _EPSILON))
    return locations, scales


def _transform(
    values: NDArray[np.float32],
    masks: NDArray[np.uint8],
    location: NDArray[np.float32],
    scale: NDArray[np.float32],
) -> NDArray[np.float32]:
    mask = _binary_masks(masks)
    array = np.asarray(values, dtype=np.float32)
    if array.shape != mask.shape or array.shape[-1] != location.shape[0]:
        raise ValueError("feature values, masks, and preprocessing dimensions must match")
    transformed = np.clip((array - location) / scale, -_TRANSFORM_CLIP, _TRANSFORM_CLIP)
    return np.where(mask == 1, transformed, 0.0).astype(np.float32)


def _binary_masks(values: NDArray[np.uint8]) -> NDArray[np.uint8]:
    mask = np.asarray(values, dtype=np.uint8)
    if not np.all((mask == 0) | (mask == 1)):
        raise ValueError("feature masks must be binary")
    return mask


def _feature_names(group: str, width: int) -> tuple[str, ...]:
    if group == "local":
        registry_names = DEFAULT_FEATURE_REGISTRY.local_names
    elif group == "context":
        registry_names = DEFAULT_FEATURE_REGISTRY.context_names
    elif group == "static":
        registry_names = DEFAULT_FEATURE_REGISTRY.static_names
    else:
        raise ValueError("unknown feature group")
    if len(registry_names) == width:
        return registry_names
    return tuple(f"{group}_{index}" for index in range(width))


def _serialization_block(
    names: tuple[str, ...],
    location: NDArray[np.float32],
    scale: NDArray[np.float32],
) -> dict[str, object]:
    return {
        "location": location.astype(float).tolist(),
        "names": list(names),
        "scale": scale.astype(float).tolist(),
    }


def _deserialization_block(
    value: object,
    group: str,
) -> tuple[tuple[str, ...], NDArray[np.float32], NDArray[np.float32]]:
    if not isinstance(value, dict):
        raise ValueError(f"preprocessing payload requires a {group} block")
    names, location, scale = value.get("names"), value.get("location"), value.get("scale")
    if (
        not isinstance(names, list)
        or not all(isinstance(name, str) and name for name in names)
        or not isinstance(location, list)
        or not isinstance(scale, list)
        or len(names) != len(location)
        or len(names) != len(scale)
    ):
        raise ValueError(f"preprocessing {group} block is malformed")
    locations = np.asarray(location, dtype=np.float32)
    scales = np.asarray(scale, dtype=np.float32)
    if (
        not np.isfinite(locations).all()
        or not np.isfinite(scales).all()
        or np.any(scales < _EPSILON)
    ):
        raise ValueError(f"preprocessing {group} block has invalid statistics")
    return tuple(names), locations, scales


def _stack_tensor(examples: Sequence[TrainingExample], field: str) -> Tensor:
    values = [getattr(example, field) for example in examples]
    return torch.from_numpy(np.stack(values).astype(np.float32))


def _float_tensor(examples: Sequence[TrainingExample], field: str) -> Tensor:
    return torch.tensor(
        [float(getattr(example, field)) for example in examples],
        dtype=torch.float32,
    )


def _require_same_schema(examples: Sequence[TrainingExample]) -> None:
    schemas = {example.feature_schema_hash for example in examples}
    if len(schemas) != 1:
        raise ValueError("training examples must use one feature schema hash")


def _state_value(state: ConfirmedState | None) -> int:
    if state is ConfirmedState.POSITIVE:
        return 1
    if state is ConfirmedState.NEGATIVE:
        return -1
    return 0


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("training timestamps must be timezone-aware")
    return value.astimezone(UTC)
