from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, field_validator


class ImbalanceObservation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp: datetime
    quarter_hour: datetime
    resolution_code: str
    quality_status: str
    ace_mw: float | None
    system_imbalance_mw: float
    alpha_eur_mwh: float | None
    alpha_prime_eur_mwh: float | None
    marginal_incremental_price_eur_mwh: float | None
    marginal_decremental_price_eur_mwh: float | None
    imbalance_price_eur_mwh: float | None

    @field_validator("timestamp", "quarter_hour")
    @classmethod
    def require_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise ValueError("observation datetimes must be UTC-aware")
        return value.astimezone(UTC)


class ConfirmedState(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


@dataclass(frozen=True, slots=True)
class ConfirmedStateSeed:
    state: ConfirmedState | None
    state_since: datetime | None
    last_observed_at: datetime | None

    def __post_init__(self) -> None:
        if self.state is None and self.state_since is not None:
            raise ValueError("an unknown state cannot have a state_since timestamp")
        if self.state is not None and self.state_since is None:
            raise ValueError("a known state requires a state_since timestamp")
        for value in (self.state_since, self.last_observed_at):
            if value is not None and (value.tzinfo is None or value.utcoffset() != timedelta(0)):
                raise ValueError("state seed timestamps must be UTC-aware")


@dataclass(frozen=True, slots=True)
class VersionedImbalanceObservation:
    observation: ImbalanceObservation
    available_at: datetime
    row_version: int = 0
    event_id: str = ""

    def __post_init__(self) -> None:
        if self.available_at.tzinfo is None or self.available_at.utcoffset() != timedelta(0):
            raise ValueError("version availability timestamps must be UTC-aware")
        if self.row_version < 0:
            raise ValueError("row_version cannot be negative")


def advance_state(
    previous: ConfirmedState | None,
    value_mw: float,
    deadband_mw: float = 10.0,
) -> ConfirmedState | None:
    if deadband_mw <= 0:
        raise ValueError("deadband_mw must be positive")
    if value_mw > deadband_mw:
        return ConfirmedState.POSITIVE
    if value_mw < -deadband_mw:
        return ConfirmedState.NEGATIVE
    return previous


def flip_label(
    current: ConfirmedState | None,
    future: ConfirmedState | None,
) -> bool | None:
    if current is None or future is None:
        return None
    return current is not future
