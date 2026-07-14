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


def latest_purged_split(
    timestamps: Sequence[datetime],
    *,
    gap_minutes: int = 1_440,
    initial_training: timedelta = _DEFAULT_INITIAL_TRAINING,
    evaluation: timedelta = _DEFAULT_EVALUATION,
) -> TimeSplit:
    """Build one leakage-safe split whose untouched test period ends at the latest row."""
    _validate_parameters(gap_minutes, initial_training, evaluation)
    ordered = _ordered_timestamps(timestamps)
    gap = timedelta(minutes=gap_minutes)
    end = ordered[-1] + _nominal_resolution(ordered)
    test = _range(ordered, end - evaluation, end)
    calibration_end = end - evaluation - gap
    calibration = _range(ordered, calibration_end - evaluation, calibration_end)
    validation_end = calibration_end - evaluation - gap
    validation = _range(ordered, validation_end - evaluation, validation_end)
    train_end = validation_end - evaluation - gap
    train = _range(ordered, ordered[0], train_end)
    if any(section is None for section in (train, validation, calibration, test)):
        raise ValueError("dataset cannot form the requested latest purged split")
    assert train is not None
    assert validation is not None
    assert calibration is not None
    assert test is not None
    if ordered[train.stop - 1] - ordered[0] + _nominal_resolution(ordered) < initial_training:
        raise ValueError("dataset cannot form the requested latest purged split")
    split = TimeSplit(
        train=train,
        validation=validation,
        calibration=calibration,
        test=test,
    )
    validate_time_split(ordered, split, gap_minutes=gap_minutes)
    return split


def validate_time_split(
    timestamps: Sequence[datetime],
    split: TimeSplit,
    *,
    gap_minutes: int = 1_440,
) -> None:
    """Reject out-of-range, overlapping, or insufficiently-purged manual splits."""
    if gap_minutes < 1:
        raise ValueError("gap_minutes must be at least one")
    ordered = _ordered_timestamps(timestamps)
    sections = (split.train, split.validation, split.calibration, split.test)
    if any(section.stop > len(ordered) for section in sections):
        raise ValueError("training split is outside the dataset")
    if any(left.stop > right.start for left, right in zip(sections, sections[1:], strict=False)):
        raise ValueError("training split sections must be strictly chronological and disjoint")
    gap = timedelta(minutes=gap_minutes)
    for earlier, later in zip(sections, sections[1:], strict=False):
        if ordered[later.start] - ordered[earlier.stop - 1] < gap:
            raise ValueError("training split does not satisfy the required purge gap")


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
    _validate_parameters(gap_minutes, initial_training, evaluation)
    ordered = _ordered_timestamps(timestamps)
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


def _validate_parameters(
    gap_minutes: int,
    initial_training: timedelta,
    evaluation: timedelta,
) -> None:
    if gap_minutes < 1:
        raise ValueError("gap_minutes must be at least one")
    if initial_training <= timedelta(0) or evaluation <= timedelta(0):
        raise ValueError("split durations must be positive")


def _ordered_timestamps(timestamps: Sequence[datetime]) -> tuple[datetime, ...]:
    ordered = tuple(_utc(timestamp) for timestamp in timestamps)
    if any(later <= earlier for earlier, later in zip(ordered, ordered[1:], strict=False)):
        raise ValueError("timestamps must be strictly increasing")
    if not ordered:
        raise ValueError("cannot form walk-forward splits from no timestamps")
    return ordered


def _nominal_resolution(timestamps: Sequence[datetime]) -> timedelta:
    if len(timestamps) < 2:
        raise ValueError("cannot form a split from fewer than two timestamps")
    resolution = min(
        later - earlier for earlier, later in zip(timestamps, timestamps[1:], strict=False)
    )
    if resolution <= timedelta(0):
        raise ValueError("timestamps must be strictly increasing")
    return resolution


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
