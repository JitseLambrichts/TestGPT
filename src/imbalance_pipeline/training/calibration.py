import json
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.isotonic import IsotonicRegression  # type: ignore[import-untyped]


@dataclass(frozen=True, slots=True)
class IsotonicCalibrator:
    x_thresholds: NDArray[np.float64]
    y_thresholds: NDArray[np.float64]
    decision_threshold: float = 0.5

    @classmethod
    def fit(
        cls,
        probability: NDArray[np.float64],
        target: NDArray[np.int64],
    ) -> "IsotonicCalibrator":
        _validate(probability, target)
        if len(np.unique(target)) < 2:
            return cls(
                x_thresholds=np.asarray([0.0, 1.0]),
                y_thresholds=np.asarray([float(target[0]), float(target[0])]),
                decision_threshold=0.5,
            )
        fitted = IsotonicRegression(out_of_bounds="clip").fit(probability, target)
        calibrator = cls(
            x_thresholds=np.asarray(fitted.X_thresholds_, dtype=np.float64),
            y_thresholds=np.asarray(fitted.y_thresholds_, dtype=np.float64),
        )
        return cls(
            x_thresholds=calibrator.x_thresholds,
            y_thresholds=calibrator.y_thresholds,
            decision_threshold=select_f1_threshold(calibrator.predict(probability), target),
        )

    def predict(self, probability: NDArray[np.float64]) -> NDArray[np.float64]:
        values = np.asarray(probability, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("calibration probabilities must be finite")
        return np.clip(
            np.interp(values, self.x_thresholds, self.y_thresholds),
            0.0,
            1.0,
        )

    def to_json(self) -> str:
        return json.dumps(
            {
                "decision_threshold": self.decision_threshold,
                "x_thresholds": self.x_thresholds.tolist(),
                "y_thresholds": self.y_thresholds.tolist(),
            },
            separators=(",", ":"),
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, serialized: str) -> "IsotonicCalibrator":
        value = json.loads(serialized)
        if not isinstance(value, dict):
            raise ValueError("calibration payload must be an object")
        x = np.asarray(value.get("x_thresholds"), dtype=np.float64)
        y = np.asarray(value.get("y_thresholds"), dtype=np.float64)
        threshold = value.get("decision_threshold")
        if (
            x.ndim != 1
            or y.ndim != 1
            or len(x) == 0
            or len(x) != len(y)
            or not np.isfinite(x).all()
            or not np.isfinite(y).all()
            or np.any(np.diff(x) < 0)
            or np.any(np.diff(y) < 0)
            or not isinstance(threshold, (int, float))
            or not 0.0 <= float(threshold) <= 1.0
        ):
            raise ValueError("invalid calibration payload")
        return cls(x_thresholds=x, y_thresholds=y, decision_threshold=float(threshold))


def select_f1_threshold(probability: NDArray[np.float64], target: NDArray[np.int64]) -> float:
    _validate(probability, target)
    best_threshold = float(np.min(probability))
    best_f1 = -1.0
    for threshold in np.unique(probability):
        predicted = probability >= threshold
        true_positive = int(np.count_nonzero(predicted & (target == 1)))
        false_positive = int(np.count_nonzero(predicted & (target == 0)))
        false_negative = int(np.count_nonzero(~predicted & (target == 1)))
        denominator = 2 * true_positive + false_positive + false_negative
        score = 2 * true_positive / denominator if denominator else 0.0
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold


def _validate(probability: NDArray[np.float64], target: NDArray[np.int64]) -> None:
    if len(probability) == 0 or len(probability) != len(target):
        raise ValueError("calibration arrays must be non-empty and have matching lengths")
    if not np.isfinite(probability).all() or np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("calibration probabilities must be finite within [0, 1]")
    if np.any((target != 0) & (target != 1)):
        raise ValueError("calibration targets must be binary")
