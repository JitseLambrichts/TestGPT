from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import (  # type: ignore[import-untyped]
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)

from imbalance_pipeline.training.data import TrainingExample

_DRIFT_WINDOW = 15


@dataclass(frozen=True, slots=True)
class BaselineValues:
    predicted_mw: NDArray[np.float64]
    flip_probability: NDArray[np.float64]


def persistence_baseline(examples: Iterable[TrainingExample]) -> NDArray[np.float64]:
    return np.asarray([_latest_observed(example) for example in examples], dtype=np.float64)


def clipped_linear_drift_baseline(examples: Iterable[TrainingExample]) -> NDArray[np.float64]:
    return np.asarray([_linear_drift(example) for example in examples], dtype=np.float64)


def rolling_median_baseline(examples: Iterable[TrainingExample]) -> NDArray[np.float64]:
    return np.asarray([_rolling_median(example) for example in examples], dtype=np.float64)


def classical_baseline(
    training: Iterable[TrainingExample],
    evaluation: Iterable[TrainingExample],
    *,
    max_training_examples: int = 100_000,
) -> BaselineValues:
    if max_training_examples <= 0:
        raise ValueError("classical baseline max_training_examples must be positive")
    train_features, train_target, train_flip_target, train_flip_mask = _training_reservoir(
        training,
        max_examples=max_training_examples,
    )
    evaluation_features = _compact_features(evaluation)
    if len(train_features) == 0 or len(evaluation_features) == 0:
        raise ValueError("classical baseline requires non-empty training and evaluation examples")
    regressor = HistGradientBoostingRegressor(
        learning_rate=0.05,
        l2_regularization=1.0,
        max_iter=100,
        max_leaf_nodes=15,
        random_state=17,
    ).fit(
        train_features,
        train_target,
    )
    point_prediction = np.asarray(regressor.predict(evaluation_features), dtype=np.float64)
    flip_probability = _classical_flip_probability(
        train_features,
        evaluation_features,
        train_flip_target,
        train_flip_mask,
    )
    return BaselineValues(predicted_mw=point_prediction, flip_probability=flip_probability)


def _classical_flip_probability(
    train_features: NDArray[np.float32],
    evaluation_features: NDArray[np.float32],
    flip_target: NDArray[np.int64],
    flip_mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    eligible = flip_mask
    target = flip_target[eligible]
    if len(target) == 0:
        return np.zeros(len(evaluation_features), dtype=np.float64)
    base_rate = float(np.mean(target))
    if len(np.unique(target)) < 2:
        return np.full(len(evaluation_features), base_rate, dtype=np.float64)
    classifier = HistGradientBoostingClassifier(
        learning_rate=0.05,
        l2_regularization=1.0,
        max_iter=100,
        max_leaf_nodes=15,
        random_state=17,
    ).fit(train_features[eligible], target)
    return np.asarray(classifier.predict_proba(evaluation_features)[:, 1], dtype=np.float64)


def _training_reservoir(
    examples: Iterable[TrainingExample],
    *,
    max_examples: int,
) -> tuple[NDArray[np.float32], NDArray[np.float64], NDArray[np.int64], NDArray[np.bool_]]:
    generator = np.random.default_rng(17)
    features: list[NDArray[np.float32]] = []
    target: list[float] = []
    flip_target: list[int] = []
    flip_mask: list[bool] = []
    for count, example in enumerate(examples):
        candidate = _compact_example(example)
        if len(features) < max_examples:
            index = len(features)
            features.append(candidate)
            target.append(example.target_next)
            flip_target.append(int(example.flip_target > 0.5))
            flip_mask.append(example.flip_mask > 0.5)
        else:
            index = int(generator.integers(0, count + 1))
            if index >= max_examples:
                continue
            features[index] = candidate
            target[index] = example.target_next
            flip_target[index] = int(example.flip_target > 0.5)
            flip_mask[index] = example.flip_mask > 0.5
    matrix = _feature_matrix(features)
    return (
        matrix,
        np.asarray(target, dtype=np.float64),
        np.asarray(flip_target, dtype=np.int64),
        np.asarray(flip_mask, dtype=bool),
    )


def _compact_features(examples: Iterable[TrainingExample]) -> NDArray[np.float32]:
    return _feature_matrix([_compact_example(example) for example in examples])


def _feature_matrix(features: Sequence[NDArray[np.float32]]) -> NDArray[np.float32]:
    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("baseline features must be finite and have matching shapes")
    return matrix


def _compact_example(example: TrainingExample) -> NDArray[np.float32]:
    summaries = (
        _summary(example.local_values, example.local_masks),
        _summary(example.context_values, example.context_masks),
        _summary(example.static_values, example.static_masks),
    )
    return np.concatenate(summaries)


def _summary(
    values: NDArray[np.float32],
    masks: NDArray[np.uint8],
) -> NDArray[np.float32]:
    if values.shape != masks.shape:
        raise ValueError("baseline feature values and masks must have matching shapes")
    if values.ndim == 0 or values.shape[-1] == 0:
        raise ValueError("baseline feature tensors require at least one feature channel")
    width = values.shape[-1]
    flattened_values = np.asarray(values, dtype=np.float32).reshape(-1, width)
    flattened_masks = np.asarray(masks, dtype=np.uint8).reshape(-1, width)
    observed_values = flattened_values[flattened_masks == 1]
    if not np.isfinite(observed_values).all():
        raise ValueError("observed baseline features must be finite")
    # Keep every source channel separate: MW, prices, ages, flags and calendar
    # inputs must never be pooled into a single synthetic baseline feature.
    result = np.zeros((width, 5), dtype=np.float32)
    for channel in range(width):
        observed = flattened_values[flattened_masks[:, channel] == 1, channel]
        if not len(observed):
            continue
        result[channel] = (
            observed[-1],
            np.mean(observed),
            np.std(observed),
            observed[-1] - observed[0],
            len(observed) / len(flattened_values),
        )
    return result.reshape(-1)


def _observed_history(example: TrainingExample) -> NDArray[np.float64]:
    values = np.asarray(example.local_values[:, 0], dtype=np.float64)
    masks = np.asarray(example.local_masks[:, 0], dtype=np.uint8)
    if values.shape != masks.shape:
        raise ValueError("local imbalance values and masks must have matching shapes")
    observed = values[masks == 1]
    if not np.isfinite(observed).all():
        raise ValueError("observed local imbalance values must be finite")
    return np.asarray(observed, dtype=np.float64)


def _latest_observed(example: TrainingExample) -> float:
    history = _observed_history(example)
    return float(history[-1]) if len(history) else 0.0


def _linear_drift(example: TrainingExample) -> float:
    history = _observed_history(example)[-_DRIFT_WINDOW:]
    if len(history) < 2:
        return _latest_observed(example)
    coordinates = np.arange(len(history), dtype=np.float64)
    slope, intercept = np.polyfit(coordinates, history, deg=1)
    extrapolated = float(intercept + slope * len(history))
    return float(np.clip(extrapolated, np.min(history), np.max(history)))


def _rolling_median(example: TrainingExample) -> float:
    history = _observed_history(example)[-_DRIFT_WINDOW:]
    return float(np.median(history)) if len(history) else 0.0
