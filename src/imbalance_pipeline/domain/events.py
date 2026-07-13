import hashlib
import json
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Subject(StrEnum):
    RAW_ELIA_IMBALANCE = "grid.raw.elia.imbalance.v1"
    RAW_ELIA_LOAD = "grid.raw.elia.load.v1"
    RAW_ELIA_WIND = "grid.raw.elia.wind.v1"
    RAW_ELIA_SOLAR = "grid.raw.elia.solar.v1"
    RAW_WEATHER_FORECAST = "grid.raw.weather.forecast.v1"
    STORED_ELIA_IMBALANCE = "grid.stored.elia.imbalance.v1"
    STORED_IMBALANCE_PREDICTION = "grid.stored.prediction.imbalance.v1"
    FEATURES_IMBALANCE = "grid.features.imbalance.v1"
    PREDICTIONS_IMBALANCE = "grid.predictions.imbalance.v1"
    OUTCOMES_IMBALANCE = "grid.outcomes.imbalance.v1"


def event_id(
    source: str,
    dataset: str,
    natural_key: str,
    version: str = "1",
) -> str:
    identity = json.dumps(
        [source, dataset, natural_key, version],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def utc_milliseconds(value: datetime) -> datetime:
    """Normalize a timestamp to the precision persisted by ClickHouse DateTime64(3)."""
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError("timestamps must be UTC-aware")
    normalized = value.astimezone(UTC)
    return normalized.replace(microsecond=(normalized.microsecond // 1_000) * 1_000)


class EventEnvelope(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: str
    event_type: str
    schema_version: str = "1"
    source: str
    dataset: str
    event_time: datetime
    observed_at: datetime | None = None
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    correlation_id: str
    causation_id: str | None = None
    quality_status: str
    payload: dict[str, object]

    @field_validator("event_time", "observed_at", "ingested_at")
    @classmethod
    def require_utc(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise ValueError("event datetimes must be UTC-aware")
        return value.astimezone(UTC)

    @classmethod
    def create(
        cls,
        event_type: str,
        source: str,
        dataset: str,
        event_time: datetime,
        natural_key: str,
        payload: BaseModel,
        quality_status: str,
        observed_at: datetime | None = None,
    ) -> Self:
        identifier = event_id(source, dataset, natural_key)
        return cls(
            event_id=identifier,
            event_type=event_type,
            source=source,
            dataset=dataset,
            event_time=event_time,
            observed_at=observed_at,
            correlation_id=identifier,
            quality_status=quality_status,
            payload=payload.model_dump(mode="json"),
        )
