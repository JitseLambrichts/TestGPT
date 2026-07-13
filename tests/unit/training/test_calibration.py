import numpy as np
import pytest

from imbalance_pipeline.training.calibration import IsotonicCalibrator, select_f1_threshold


def test_isotonic_calibration_is_bounded_round_trips_and_does_not_worsen_fit_brier() -> None:
    probability = np.asarray([0.9, 0.8, 0.7, 0.2, 0.1, 0.05])
    target = np.asarray([1, 0, 0, 1, 0, 1])
    calibrator = IsotonicCalibrator.fit(probability, target)

    calibrated = calibrator.predict(probability)
    restored = IsotonicCalibrator.from_json(calibrator.to_json())

    assert np.all((0.0 <= calibrated) & (calibrated <= 1.0))
    assert np.mean((calibrated - target) ** 2) <= np.mean((probability - target) ** 2)
    np.testing.assert_allclose(restored.predict(probability), calibrated)
    assert np.all(np.diff(calibrator.x_thresholds) >= 0)
    assert np.all(np.diff(calibrator.y_thresholds) >= 0)


def test_threshold_selection_uses_the_lower_threshold_for_an_f1_tie() -> None:
    threshold = select_f1_threshold(
        np.asarray([0.8, 0.6, 0.5, 0.4]),
        np.asarray([1, 0, 0, 1]),
    )

    assert threshold == 0.4


@pytest.mark.parametrize(
    "payload",
    [
        '{"x_thresholds":[-0.1,1.0],"y_thresholds":[0.0,1.0],"decision_threshold":0.5}',
        '{"x_thresholds":[0.0,1.0],"y_thresholds":[0.0,1.1],"decision_threshold":0.5}',
    ],
)
def test_isotonic_calibration_rejects_thresholds_outside_probability_range(payload: str) -> None:
    with pytest.raises(ValueError, match="invalid calibration payload"):
        IsotonicCalibrator.from_json(payload)
