from datetime import UTC, datetime, timedelta

import pytest

from imbalance_pipeline.training.splits import walk_forward_splits


def minute_timestamps(days: int) -> list[datetime]:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    return [start + timedelta(minutes=index) for index in range(days * 24 * 60)]


def test_walk_forward_splits_are_purged_chronological_and_stable_when_future_is_added() -> None:
    timestamps = minute_timestamps(365)
    splits = walk_forward_splits(timestamps, folds=3, gap_minutes=1_440)
    extended = walk_forward_splits(
        timestamps
        + [timestamps[-1] + timedelta(minutes=index) for index in range(1, 2 * 24 * 60 + 1)],
        folds=3,
        gap_minutes=1_440,
    )

    assert len(splits) == 3
    assert splits == extended
    for split in splits:
        assert split.train.stop <= split.validation.start
        assert split.validation.stop <= split.calibration.start
        assert split.calibration.stop <= split.test.start
        assert (
            timestamps[split.validation.start] - timestamps[split.train.stop - 1]
            >= timedelta(minutes=1_440)
        )
        roles = (
            set(range(split.train.start, split.train.stop)),
            set(range(split.validation.start, split.validation.stop)),
            set(range(split.calibration.start, split.calibration.stop)),
            set(range(split.test.start, split.test.stop)),
        )
        assert all(
            left.isdisjoint(right)
            for index, left in enumerate(roles)
            for right in roles[index + 1 :]
        )


def test_walk_forward_splits_reject_insufficient_or_non_chronological_data() -> None:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)

    with pytest.raises(ValueError, match="strictly increasing"):
        walk_forward_splits([timestamp, timestamp], folds=1, gap_minutes=1)
    with pytest.raises(ValueError, match="cannot form"):
        walk_forward_splits([timestamp + timedelta(minutes=index) for index in range(10)], folds=1)
