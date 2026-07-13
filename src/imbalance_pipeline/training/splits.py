from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

_DEFAULT_INITIAL_TRAINING = timedelta(days=60)
_DEFAULT_EVALUATION = timedelta(days=14)


@dataclass(frozen=True, slots=True)
class IndexRange:
    start: int
    stop: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.stop <= self.start:
            raise ValueError("index ranges must be non-empty and increasing")


@dataclass(frozen=True, slots=True)
class TimeSplit:
    train: IndexRange
    validation: IndexRange
    calibration: IndexRange
    test: IndexRange


def walk_forward_splits(
    timestamps: Sequence[datetime],
    *,
    folds: int,
    gap_minutes: int = 1_440,
    initial_training: timedelta = _DEFAULT_INITIAL_TRAINING,
    evaluation: timedelta = _DEFAULT_EVALUATION,
) -> list[TimeSplit]:
    if folds <= 0:
        raise ValueError("folds must be positive")
    if gap_minutes < 1:
        raise ValueError("gap_minutes must be at least one")
    if initial_training <= timedelta(0) or evaluation <= timedelta(0):
        raise ValueError("split durations must be positive")
    ordered = tuple(_utc(timestamp) for timestamp in timestamps)
    if any(later <= earlier for earlier, later in zip(ordered, ordered[1:], strict=False)):
        raise ValueError("timestamps must be strictly increasing")
    if not ordered:
        raise ValueError("cannot form walk-forward splits from no timestamps")
    gap = timedelta(minutes=gap_minutes)
    origin = ordered[0] + initial_training
    splits: list[TimeSplit] = []
    for _ in range(folds):
        train = _range(ordered, ordered[0], origin)
        validation = _range(ordered, origin + gap, origin + gap + evaluation)
        calibration_start = origin + gap + evaluation + gap
        calibration = _range(ordered, calibration_start, calibration_start + evaluation)
        test_start = calibration_start + evaluation + gap
        test = _range(ordered, test_start, test_start + evaluation)
        if any(index_range is None for index_range in (train, validation, calibration, test)):
            raise ValueError("dataset cannot form the requested purged walk-forward splits")
        assert train is not None
        assert validation is not None
        assert calibration is not None
        assert test is not None
        splits.append(
            TimeSplit(
                train=train,
                validation=validation,
                calibration=calibration,
                test=test,
            )
        )
        origin = test_start + evaluation
    return splits


def _range(
    timestamps: Sequence[datetime],
    start: datetime,
    stop: datetime,
) -> IndexRange | None:
    lower = bisect_left(timestamps, start)
    upper = bisect_left(timestamps, stop)
    if lower >= upper:
        return None
    return IndexRange(lower, upper)


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("split timestamps must be timezone-aware")
    return value.astimezone(UTC)
