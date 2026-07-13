import pytest

from imbalance_pipeline.domain.imbalance import ConfirmedState, advance_state, flip_label


@pytest.mark.parametrize(
    "previous",
    [None, ConfirmedState.POSITIVE, ConfirmedState.NEGATIVE],
)
@pytest.mark.parametrize("value_mw", [-10.0, 0.0, 10.0])
def test_neutral_band_including_boundaries_retains_previous_state(
    previous: ConfirmedState | None,
    value_mw: float,
) -> None:
    assert advance_state(previous, value_mw, 10.0) is previous


def test_values_outside_deadband_establish_state() -> None:
    assert advance_state(None, 10.0001, 10.0) is ConfirmedState.POSITIVE
    assert advance_state(None, -10.0001, 10.0) is ConfirmedState.NEGATIVE


@pytest.mark.parametrize(
    ("current", "value_mw", "expected"),
    [
        (ConfirmedState.POSITIVE, -11.0, ConfirmedState.NEGATIVE),
        (ConfirmedState.NEGATIVE, 11.0, ConfirmedState.POSITIVE),
    ],
)
def test_crossing_opposite_boundary_is_a_flip(
    current: ConfirmedState,
    value_mw: float,
    expected: ConfirmedState,
) -> None:
    future = advance_state(current, value_mw, 10.0)

    assert future is expected
    assert flip_label(current, future) is True


def test_remaining_in_same_confirmed_state_is_not_a_flip() -> None:
    current = advance_state(None, 11.0, 10.0)
    future = advance_state(current, 20.0, 10.0)

    assert flip_label(current, future) is False


def test_first_neutral_value_leaves_state_unknown() -> None:
    assert advance_state(None, 0.0, 10.0) is None
    assert flip_label(None, ConfirmedState.POSITIVE) is None
    assert flip_label(ConfirmedState.NEGATIVE, None) is None


@pytest.mark.parametrize("deadband_mw", [0.0, -1.0])
def test_deadband_must_be_positive(deadband_mw: float) -> None:
    with pytest.raises(ValueError, match="deadband_mw must be positive"):
        advance_state(None, 0.0, deadband_mw)
