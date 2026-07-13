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
