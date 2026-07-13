from collections.abc import Sequence
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


def persistence_baseline(examples: Sequence[TrainingExample]) -> NDArray[np.float64]:
    return np.asarray([_latest_observed(example) for example in examples], dtype=np.float64)


def clipped_linear_drift_baseline(examples: Sequence[TrainingExample]) -> NDArray[np.float64]:
    return np.asarray([_linear_drift(example) for example in examples], dtype=np.float64)


def rolling_median_baseline(examples: Sequence[TrainingExample]) -> NDArray[np.float64]:
    return np.asarray([_rolling_median(example) for example in examples], dtype=np.float64)


def classical_baseline(
    training: Sequence[TrainingExample],
    evaluation: Sequence[TrainingExample],
) -> BaselineValues:
    if not training or not evaluation:
        raise ValueError("classical baseline requires non-empty training and evaluation examples")
    train_features = _flatten_features(training)
    evaluation_features = _flatten_features(evaluation)
    regressor = HistGradientBoostingRegressor(
        learning_rate=0.05,
        l2_regularization=1.0,
        max_iter=100,
        max_leaf_nodes=15,
        random_state=17,
    ).fit(
        train_features,
        np.asarray([example.target_next for example in training], dtype=np.float64),
    )
    point_prediction = np.asarray(regressor.predict(evaluation_features), dtype=np.float64)
    flip_probability = _classical_flip_probability(train_features, evaluation_features, training)
    return BaselineValues(predicted_mw=point_prediction, flip_probability=flip_probability)


def _classical_flip_probability(
    train_features: NDArray[np.float64],
    evaluation_features: NDArray[np.float64],
    training: Sequence[TrainingExample],
) -> NDArray[np.float64]:
    eligible = np.asarray([example.flip_mask > 0.5 for example in training])
    target = np.asarray([example.flip_target for example in training], dtype=np.int64)[eligible]
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


def _flatten_features(examples: Sequence[TrainingExample]) -> NDArray[np.float64]:
    matrix = np.asarray([_flatten_example(example) for example in examples], dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("baseline features must be finite and have matching shapes")
    return matrix


def _flatten_example(example: TrainingExample) -> NDArray[np.float64]:
    arrays = (
        example.local_values,
        example.local_masks,
        example.context_values,
        example.context_masks,
        example.static_values,
        example.static_masks,
    )
    return np.concatenate([np.asarray(values, dtype=np.float64).reshape(-1) for values in arrays])


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
