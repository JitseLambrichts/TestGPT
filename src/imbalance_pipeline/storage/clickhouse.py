import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Protocol, Self, cast

import clickhouse_connect  # type: ignore[import-untyped]
from clickhouse_connect.driver.asyncclient import AsyncClient  # type: ignore[import-untyped]
from clickhouse_connect.driver.exceptions import ClickHouseError  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from imbalance_pipeline.config import Settings
from imbalance_pipeline.domain.events import EventEnvelope
from imbalance_pipeline.domain.imbalance import (
    ConfirmedState,
    ConfirmedStateSeed,
    ImbalanceObservation,
    VersionedImbalanceObservation,
)
from imbalance_pipeline.sources.elia import (
    LoadObservation,
    SolarObservation,
    WindObservation,
)
from imbalance_pipeline.sources.weather import WeatherForecast

SUPPORTED_SCHEMA_VERSION = "1"
FEATURE_EVENT_TYPES = frozenset(
    {
        "imbalance.feature.snapshot",
        "imbalance.features.built",
        "imbalance.features.created",
    }
)
PREDICTION_EVENT_TYPES = frozenset(
    {
        "imbalance.predicted",
        "imbalance.prediction.created",
        "imbalance.prediction.generated",
    }
)
OUTCOME_EVENT_TYPES = frozenset(
    {
        "imbalance.outcome.observed",
        "imbalance.prediction.outcome",
        "imbalance.prediction.evaluated",
    }
)
MODEL_EVENT_TYPES = frozenset(
    {
        "imbalance.model.version",
        "model.version.registered",
        "model.version.promoted",
    }
)

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class DeadLetterReason(StrEnum):
    UNSUPPORTED_EVENT_TYPE = "unsupported_event_type"
    UNSUPPORTED_SCHEMA_VERSION = "unsupported_schema_version"
    INVALID_PAYLOAD = "invalid_payload"
    DELIVERY_EXHAUSTED = "delivery_exhausted"


class StorageError(RuntimeError):
    """Base class for safe storage-boundary failures."""


class TransientStorageError(StorageError):
    """A retryable ClickHouse failure with deliberately sanitized text."""

    def __init__(self, detail: object | None = None) -> None:
        del detail
        super().__init__("transient ClickHouse operation failed")


class PermanentEventError(StorageError):
    """An event that cannot become valid through redelivery."""

    def __init__(self, reason: DeadLetterReason, detail: object | None = None) -> None:
        del detail
        self.reason = reason
        super().__init__(f"event rejected: {reason.value}")


class FeatureSnapshot(Protocol):
    """Structural input contract kept independent of the future feature package."""

    @property
    def event_id(self) -> str: ...

    @property
    def cutoff(self) -> datetime: ...

    @property
    def target_time(self) -> datetime: ...

    @property
    def feature_schema_hash(self) -> str: ...

    @property
    def local_values(self) -> object: ...

    @property
    def local_masks(self) -> object: ...

    @property
    def context_values(self) -> object: ...

    @property
    def context_masks(self) -> object: ...

    @property
    def static_values(self) -> object: ...

    @property
    def static_masks(self) -> object: ...

    @property
    def current_state(self) -> str | None: ...

    @property
    def created_at(self) -> datetime: ...


class Prediction(BaseModel):
    """Storage-facing prediction view, independent of the future model package."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: str
    cutoff: datetime
    target_time: datetime
    generated_at: datetime
    system_imbalance_mw: float
    p10_mw: float
    p90_mw: float
    flip_probability: float = Field(ge=0.0, le=1.0)
    will_flip: bool
    current_state: str | None
    predicted_state: str | None
    prediction_quality: str
    model_version: str
    feature_schema_hash: str

    @field_validator("cutoff", "target_time", "generated_at")
    @classmethod
    def validate_datetime(cls, value: datetime) -> datetime:
        return _to_utc(value)

    @model_validator(mode="after")
    def validate_interval(self) -> Self:
        if self.p10_mw > self.p90_mw:
            raise ValueError("p10_mw cannot exceed p90_mw")
        return self


class _FeatureSnapshotRecord(BaseModel):
    model_config = ConfigDict(extra="ignore", from_attributes=True)

    event_id: str
    cutoff: datetime
    target_time: datetime
    feature_schema_hash: str
    local_values: object
    local_masks: object
    context_values: object
    context_masks: object
    static_values: object
    static_masks: object
    current_state: str | None = None
    created_at: datetime

    @field_validator("cutoff", "target_time", "created_at")
    @classmethod
    def validate_datetime(cls, value: datetime) -> datetime:
        return _to_utc(value)


class _FeaturePayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    cutoff: datetime
    target_time: datetime
    feature_schema_hash: str
    local_values: object
    local_masks: object
    context_values: object
    context_masks: object
    static_values: object
    static_masks: object
    current_state: str | None = None
    created_at: datetime | None = None

    @field_validator("cutoff", "target_time", "created_at")
    @classmethod
    def validate_datetime(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _to_utc(value)


class _PredictionPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    cutoff: datetime
    target_time: datetime
    generated_at: datetime
    system_imbalance_mw: float
    p10_mw: float
    p90_mw: float
    flip_probability: float = Field(ge=0.0, le=1.0)
    will_flip: bool
    current_state: str | None = None
    predicted_state: str | None = None
    prediction_quality: str
    model_version: str
    feature_schema_hash: str

    @field_validator("cutoff", "target_time", "generated_at")
    @classmethod
    def validate_datetime(cls, value: datetime) -> datetime:
        return _to_utc(value)

    @model_validator(mode="after")
    def validate_interval(self) -> Self:
        if self.p10_mw > self.p90_mw:
            raise ValueError("invalid prediction interval")
        return self


class _OutcomePayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    prediction_event_id: str
    target_time: datetime
    realized_event_id: str
    realized_system_imbalance_mw: float
    realized_state: str | None = None
    flip_actual: bool | None = None
    evaluated_at: datetime

    @field_validator("target_time", "evaluated_at")
    @classmethod
    def validate_datetime(cls, value: datetime) -> datetime:
        return _to_utc(value)


class _ModelVersionPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model_version: str
    feature_schema_hash: str
    manifest_json: str | Mapping[str, object]
    metrics_json: str | Mapping[str, object]
    promoted_at: datetime | None = None
    created_at: datetime

    @field_validator("promoted_at", "created_at")
    @classmethod
    def validate_datetime(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _to_utc(value)


@dataclass(frozen=True, slots=True)
class _PreparedInsert:
    table: str
    columns: tuple[str, ...]
    row: list[object]


RAW_COLUMNS = (
    "event_id",
    "event_type",
    "schema_version",
    "source",
    "dataset",
    "event_time",
    "observed_at",
    "ingested_at",
    "correlation_id",
    "causation_id",
    "quality_status",
    "payload_json",
    "envelope_json",
    "row_version",
)
IMBALANCE_COLUMNS = (
    "event_id",
    "timestamp",
    "quarter_hour",
    "resolution_code",
    "quality_status",
    "ace_mw",
    "system_imbalance_mw",
    "alpha_eur_mwh",
    "alpha_prime_eur_mwh",
    "marginal_incremental_price_eur_mwh",
    "marginal_decremental_price_eur_mwh",
    "imbalance_price_eur_mwh",
    "ingested_at",
    "row_version",
)
LOAD_COLUMNS = (
    "event_id",
    "timestamp",
    "resolution_code",
    "measured_mw",
    "most_recent_forecast_mw",
    "most_recent_confidence_10_mw",
    "most_recent_confidence_90_mw",
    "day_ahead_forecast_mw",
    "day_ahead_confidence_10_mw",
    "day_ahead_confidence_90_mw",
    "week_ahead_forecast_mw",
    "monitored_capacity_mw",
    "load_factor",
    "ingested_at",
    "row_version",
)
WIND_COLUMNS = (
    "event_id",
    "timestamp",
    "resolution_code",
    "offshore_onshore",
    "region",
    "grid_connection_type",
    "real_time_mw",
    "most_recent_forecast_mw",
    "most_recent_confidence_10_mw",
    "most_recent_confidence_90_mw",
    "day_ahead_11h_forecast_mw",
    "day_ahead_11h_confidence_10_mw",
    "day_ahead_11h_confidence_90_mw",
    "day_ahead_forecast_mw",
    "day_ahead_confidence_10_mw",
    "day_ahead_confidence_90_mw",
    "week_ahead_forecast_mw",
    "week_ahead_confidence_10_mw",
    "week_ahead_confidence_90_mw",
    "monitored_capacity_mw",
    "load_factor",
    "decremental_bid_id",
    "ingested_at",
    "row_version",
)
SOLAR_COLUMNS = (
    "event_id",
    "timestamp",
    "resolution_code",
    "region",
    "real_time_mw",
    "most_recent_forecast_mw",
    "most_recent_confidence_10_mw",
    "most_recent_confidence_90_mw",
    "day_ahead_11h_forecast_mw",
    "day_ahead_11h_confidence_10_mw",
    "day_ahead_11h_confidence_90_mw",
    "day_ahead_forecast_mw",
    "day_ahead_confidence_10_mw",
    "day_ahead_confidence_90_mw",
    "week_ahead_forecast_mw",
    "week_ahead_confidence_10_mw",
    "week_ahead_confidence_90_mw",
    "monitored_capacity_mw",
    "load_factor",
    "ingested_at",
    "row_version",
)
FEATURE_COLUMNS = (
    "event_id",
    "cutoff",
    "target_time",
    "feature_schema_hash",
    "local_values",
    "local_masks",
    "context_values",
    "context_masks",
    "static_values",
    "static_masks",
    "current_state",
    "created_at",
    "row_version",
)
PREDICTION_COLUMNS = (
    "event_id",
    "cutoff",
    "target_time",
    "generated_at",
    "system_imbalance_mw",
    "p10_mw",
    "p90_mw",
    "flip_probability",
    "will_flip",
    "current_state",
    "predicted_state",
    "prediction_quality",
    "model_version",
    "feature_schema_hash",
    "row_version",
)
OUTCOME_COLUMNS = (
    "prediction_event_id",
    "target_time",
    "realized_event_id",
    "realized_system_imbalance_mw",
    "realized_state",
    "flip_actual",
    "evaluated_at",
    "row_version",
)
MODEL_COLUMNS = (
    "model_version",
    "feature_schema_hash",
    "manifest_json",
    "metrics_json",
    "promoted_at",
    "created_at",
    "row_version",
)


class ClickHouseRepository:
    def __init__(self, client: AsyncClient, *, database: str = "imbalance") -> None:
        if _IDENTIFIER.fullmatch(database) is None:
            raise ValueError("database must be a ClickHouse identifier")
        self._client = client
        self._database = database

    @classmethod
    async def connect(cls, settings: Settings) -> Self:
        client = await clickhouse_connect.get_async_client(
            dsn=settings.clickhouse_url,
            username=settings.clickhouse_user,
            password=settings.clickhouse_password,
            database=settings.clickhouse_database,
        )
        return cls(client, database=settings.clickhouse_database)

    async def aclose(self) -> None:
        await self._client.close()

    async def insert_event(self, event: EventEnvelope) -> None:
        try:
            prepared = self._prepare_event(event)
            raw = _raw_insert(event)
        except PermanentEventError:
            raise
        except (TypeError, ValueError) as exc:
            raise PermanentEventError(DeadLetterReason.INVALID_PAYLOAD) from exc
        try:
            await self._insert(raw)
            if prepared is not None:
                await self._insert(prepared)
        except (ClickHouseError, OSError, TimeoutError) as exc:
            raise TransientStorageError from exc

    async def insert_feature_snapshot(self, snapshot: FeatureSnapshot) -> None:
        try:
            record = _FeatureSnapshotRecord.model_validate(snapshot, from_attributes=True)
            prepared = _feature_insert(record)
        except (ValidationError, TypeError, ValueError) as exc:
            raise PermanentEventError(DeadLetterReason.INVALID_PAYLOAD) from exc
        try:
            await self._insert(prepared)
        except (ClickHouseError, OSError, TimeoutError) as exc:
            raise TransientStorageError from exc

    async def fetch_imbalance_window(
        self,
        event_cutoff: datetime,
        minutes: int,
        *,
        knowledge_cutoff: datetime,
    ) -> list[ImbalanceObservation]:
        if minutes <= 0:
            raise ValueError("minutes must be positive")
        event_cutoff_utc = _to_utc(event_cutoff)
        knowledge_cutoff_utc = _to_utc(knowledge_cutoff)
        start = event_cutoff_utc - timedelta(minutes=minutes)
        query = f"""
            SELECT
                timestamp,
                argMax(tuple(
                        quarter_hour,
                        resolution_code,
                        quality_status,
                        ace_mw,
                        system_imbalance_mw,
                        alpha_eur_mwh,
                        alpha_prime_eur_mwh,
                        marginal_incremental_price_eur_mwh,
                        marginal_decremental_price_eur_mwh,
                        imbalance_price_eur_mwh
                    ),
                    tuple(row_version, ingested_at, event_id)
                ) AS latest
            FROM {self._database}.imbalance_observations
            WHERE timestamp > {{start:DateTime64(3, 'UTC')}}
              AND timestamp <= {{event_cutoff:DateTime64(3, 'UTC')}}
              AND ingested_at <= {{knowledge_cutoff:DateTime64(3, 'UTC')}}
            GROUP BY timestamp
            ORDER BY timestamp
        """
        try:
            result = await self._client.query(
                query,
                parameters={
                    "start": start,
                    "event_cutoff": event_cutoff_utc,
                    "knowledge_cutoff": knowledge_cutoff_utc,
                },
                tz_mode="aware",
            )
        except (ClickHouseError, OSError, TimeoutError) as exc:
            raise TransientStorageError from exc
        return [_imbalance_from_row(row) for row in result.result_rows]

    async def fetch_imbalance_versions(
        self,
        start: datetime,
        end: datetime,
        *,
        knowledge_cutoff: datetime,
    ) -> list[VersionedImbalanceObservation]:
        start_utc = _to_utc(start)
        end_utc = _to_utc(end)
        knowledge_cutoff_utc = _to_utc(knowledge_cutoff)
        if start_utc > end_utc:
            raise ValueError("start cannot be after end")
        query = f"""
            SELECT
                event_id,
                timestamp,
                quarter_hour,
                resolution_code,
                quality_status,
                ace_mw,
                system_imbalance_mw,
                alpha_eur_mwh,
                alpha_prime_eur_mwh,
                marginal_incremental_price_eur_mwh,
                marginal_decremental_price_eur_mwh,
                imbalance_price_eur_mwh,
                ingested_at,
                row_version
            FROM {self._database}.imbalance_observations
            WHERE timestamp >= {{start:DateTime64(3, 'UTC')}}
              AND timestamp <= {{end:DateTime64(3, 'UTC')}}
              AND ingested_at <= {{knowledge_cutoff:DateTime64(3, 'UTC')}}
            ORDER BY timestamp, row_version, ingested_at, event_id
        """
        try:
            result = await self._client.query(
                query,
                parameters={
                    "start": start_utc,
                    "end": end_utc,
                    "knowledge_cutoff": knowledge_cutoff_utc,
                },
                tz_mode="aware",
            )
        except (ClickHouseError, OSError, TimeoutError) as exc:
            raise TransientStorageError from exc
        return [_versioned_imbalance_from_row(row) for row in result.result_rows]

    async def fetch_imbalance_state_seed(
        self,
        before: datetime,
        *,
        knowledge_cutoff: datetime,
        deadband_mw: float,
    ) -> ConfirmedStateSeed:
        seeds = await self._fetch_imbalance_state_seed_batch(
            ((before, knowledge_cutoff),),
            deadband_mw=deadband_mw,
        )
        return seeds[0]

    async def _fetch_imbalance_state_seed_batch(
        self,
        requests: Sequence[tuple[datetime, datetime]],
        *,
        deadband_mw: float,
    ) -> list[ConfirmedStateSeed]:
        if deadband_mw <= 0:
            raise ValueError("deadband_mw must be positive")
        parameters: dict[str, object] = {"deadband_mw": deadband_mw}
        request_queries: list[str] = []
        for index, (before, knowledge_cutoff) in enumerate(requests):
            before_utc = _to_utc(before)
            knowledge_cutoff_utc = _to_utc(knowledge_cutoff)
            parameters[f"request_{index}_before"] = before_utc
            parameters[f"request_{index}_knowledge_cutoff"] = knowledge_cutoff_utc
            request_queries.append(
                f"""
                    SELECT
                        toUInt32({index}) AS request_index,
                        {{request_{index}_before:DateTime64(3, 'UTC')}} AS before,
                        {{request_{index}_knowledge_cutoff:DateTime64(3, 'UTC')}}
                            AS knowledge_cutoff
                """
            )
        query = f"""
            WITH requests AS (
                {' UNION ALL '.join(request_queries)}
            ),
            canonical AS (
                SELECT
                    requests.request_index,
                    observations.timestamp,
                    argMax(
                        observations.system_imbalance_mw,
                        tuple(
                            observations.row_version,
                            observations.ingested_at,
                            observations.event_id
                        )
                    ) AS system_imbalance_mw
                FROM {self._database}.imbalance_observations AS observations
                INNER JOIN requests
                    ON observations.timestamp <= requests.before
                   AND observations.ingested_at <= requests.knowledge_cutoff
                GROUP BY requests.request_index, observations.timestamp
            ),
            non_neutral AS (
                SELECT
                    request_index,
                    timestamp,
                    if(
                        system_imbalance_mw > {{deadband_mw:Float64}},
                        toInt8(1),
                        toInt8(-1)
                    ) AS state_sign
                FROM canonical
                WHERE abs(system_imbalance_mw) > {{deadband_mw:Float64}}
            ),
            state_transitions AS (
                SELECT
                    request_index,
                    timestamp,
                    state_sign,
                    lagInFrame(state_sign, 1, toInt8(0)) OVER (
                        PARTITION BY request_index
                        ORDER BY timestamp
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                    ) AS previous_state_sign
                FROM non_neutral
            ),
            current_states AS (
                SELECT
                    request_index,
                    argMax(state_sign, timestamp) AS current_state_sign,
                    argMaxIf(
                        timestamp,
                        timestamp,
                        state_sign != previous_state_sign
                    ) AS state_since
                FROM state_transitions
                GROUP BY request_index
            ),
            latest_observations AS (
                SELECT request_index, max(timestamp) AS last_observed_at
                FROM canonical
                GROUP BY request_index
            )
            SELECT
                latest_observations.request_index,
                if(
                    current_states.current_state_sign = 0,
                    NULL,
                    current_states.current_state_sign
                ) AS state_sign,
                if(
                    current_states.current_state_sign = 0,
                    NULL,
                    current_states.state_since
                ) AS state_since,
                latest_observations.last_observed_at
            FROM latest_observations
            LEFT JOIN current_states USING request_index
            ORDER BY latest_observations.request_index
        """
        try:
            result = await self._client.query(query, parameters=parameters, tz_mode="aware")
        except (ClickHouseError, OSError, TimeoutError) as exc:
            raise TransientStorageError from exc
        seeds = [ConfirmedStateSeed(None, None, None) for _ in requests]
        for row in result.result_rows:
            index, state_balance, state_since, last_observed_at = row
            request_index = cast(int, index)
            if request_index < 0 or request_index >= len(seeds):
                raise RuntimeError("ClickHouse returned an invalid state-seed request index")
            seeds[request_index] = _state_seed_from_row(
                state_balance,
                state_since,
                last_observed_at,
            )
        return seeds

    async def latest_prediction(self) -> Prediction | None:
        query = f"""
            SELECT
                event_id,
                cutoff,
                target_time,
                generated_at,
                system_imbalance_mw,
                p10_mw,
                p90_mw,
                flip_probability,
                will_flip,
                current_state,
                predicted_state,
                prediction_quality,
                model_version,
                feature_schema_hash
            FROM {self._database}.predictions
            ORDER BY target_time DESC, row_version DESC, generated_at DESC, event_id DESC
            LIMIT 1
        """
        try:
            result = await self._client.query(query, parameters={}, tz_mode="aware")
        except (ClickHouseError, OSError, TimeoutError) as exc:
            raise TransientStorageError from exc
        if not result.result_rows:
            return None
        values = dict(zip(PREDICTION_COLUMNS[:-1], result.first_row, strict=True))
        for field_name in ("cutoff", "target_time", "generated_at"):
            values[field_name] = _clickhouse_utc(cast(datetime, values[field_name]))
        return Prediction.model_validate(values)

    async def _insert(self, prepared: _PreparedInsert) -> None:
        await self._client.insert(
            f"{self._database}.{prepared.table}",
            [prepared.row],
            column_names=prepared.columns,
        )

    def _prepare_event(self, event: EventEnvelope) -> _PreparedInsert | None:
        if event.schema_version != SUPPORTED_SCHEMA_VERSION:
            raise PermanentEventError(DeadLetterReason.UNSUPPORTED_SCHEMA_VERSION)
        try:
            if event.event_type == "elia.imbalance.observed":
                imbalance = ImbalanceObservation.model_validate(event.payload)
                _require_event_timestamp(event, imbalance.timestamp)
                if imbalance.quality_status != event.quality_status:
                    raise ValueError("payload quality does not match envelope")
                return _imbalance_insert(event, imbalance)
            if event.event_type == "elia.load.observed":
                load = LoadObservation.model_validate(event.payload)
                _require_event_timestamp(event, load.timestamp)
                return _load_insert(event, load)
            if event.event_type == "elia.wind.observed":
                wind = WindObservation.model_validate(event.payload)
                _require_event_timestamp(event, wind.timestamp)
                return _wind_insert(event, wind)
            if event.event_type == "elia.solar.observed":
                solar = SolarObservation.model_validate(event.payload)
                _require_event_timestamp(event, solar.timestamp)
                return _solar_insert(event, solar)
            if event.event_type == "weather.forecast.observed":
                WeatherForecast.model_validate(event.payload)
                return None
            if event.event_type in FEATURE_EVENT_TYPES:
                feature = _FeaturePayload.model_validate(event.payload)
                return _feature_event_insert(event, feature)
            if event.event_type in PREDICTION_EVENT_TYPES:
                prediction = _PredictionPayload.model_validate(event.payload)
                return _prediction_insert(event, prediction)
            if event.event_type in OUTCOME_EVENT_TYPES:
                outcome = _OutcomePayload.model_validate(event.payload)
                return _outcome_insert(event, outcome)
            if event.event_type in MODEL_EVENT_TYPES:
                model = _ModelVersionPayload.model_validate(event.payload)
                return _model_insert(event, model)
        except (ValidationError, TypeError, ValueError) as exc:
            raise PermanentEventError(DeadLetterReason.INVALID_PAYLOAD) from exc
        raise PermanentEventError(DeadLetterReason.UNSUPPORTED_EVENT_TYPE)


def _raw_insert(event: EventEnvelope) -> _PreparedInsert:
    version = _row_version(event.ingested_at, event.event_id)
    return _PreparedInsert(
        "raw_events",
        RAW_COLUMNS,
        [
            event.event_id,
            event.event_type,
            event.schema_version,
            event.source,
            event.dataset,
            _to_utc(event.event_time),
            None if event.observed_at is None else _to_utc(event.observed_at),
            _to_utc(event.ingested_at),
            event.correlation_id,
            event.causation_id,
            event.quality_status,
            _canonical_json(event.payload),
            _canonical_json(event.model_dump(mode="json")),
            version,
        ],
    )


def _imbalance_insert(
    event: EventEnvelope,
    observation: ImbalanceObservation,
) -> _PreparedInsert:
    return _PreparedInsert(
        "imbalance_observations",
        IMBALANCE_COLUMNS,
        [
            event.event_id,
            _to_utc(observation.timestamp),
            _to_utc(observation.quarter_hour),
            observation.resolution_code,
            observation.quality_status,
            observation.ace_mw,
            observation.system_imbalance_mw,
            observation.alpha_eur_mwh,
            observation.alpha_prime_eur_mwh,
            observation.marginal_incremental_price_eur_mwh,
            observation.marginal_decremental_price_eur_mwh,
            observation.imbalance_price_eur_mwh,
            _to_utc(event.ingested_at),
            _row_version(event.ingested_at, event.event_id),
        ],
    )


def _load_insert(event: EventEnvelope, observation: LoadObservation) -> _PreparedInsert:
    return _PreparedInsert(
        "load_observations",
        LOAD_COLUMNS,
        [
            event.event_id,
            _to_utc(observation.timestamp),
            observation.resolution_code,
            observation.measured_mw,
            observation.most_recent_forecast_mw,
            observation.most_recent_confidence_10_mw,
            observation.most_recent_confidence_90_mw,
            observation.day_ahead_forecast_mw,
            observation.day_ahead_confidence_10_mw,
            observation.day_ahead_confidence_90_mw,
            observation.week_ahead_forecast_mw,
            None,
            None,
            _to_utc(event.ingested_at),
            _row_version(event.ingested_at, event.event_id),
        ],
    )


def _wind_insert(event: EventEnvelope, observation: WindObservation) -> _PreparedInsert:
    return _PreparedInsert(
        "wind_observations",
        WIND_COLUMNS,
        [
            event.event_id,
            _to_utc(observation.timestamp),
            observation.resolution_code,
            observation.offshore_onshore,
            observation.region,
            observation.grid_connection_type,
            observation.real_time_mw,
            observation.most_recent_forecast_mw,
            observation.most_recent_confidence_10_mw,
            observation.most_recent_confidence_90_mw,
            observation.day_ahead_11h_forecast_mw,
            observation.day_ahead_11h_confidence_10_mw,
            observation.day_ahead_11h_confidence_90_mw,
            observation.day_ahead_forecast_mw,
            observation.day_ahead_confidence_10_mw,
            observation.day_ahead_confidence_90_mw,
            observation.week_ahead_forecast_mw,
            observation.week_ahead_confidence_10_mw,
            observation.week_ahead_confidence_90_mw,
            observation.monitored_capacity_mw,
            observation.load_factor,
            None if observation.decremental_bid_id is None else str(observation.decremental_bid_id),
            _to_utc(event.ingested_at),
            _row_version(event.ingested_at, event.event_id),
        ],
    )


def _solar_insert(event: EventEnvelope, observation: SolarObservation) -> _PreparedInsert:
    return _PreparedInsert(
        "solar_observations",
        SOLAR_COLUMNS,
        [
            event.event_id,
            _to_utc(observation.timestamp),
            observation.resolution_code,
            observation.region,
            observation.real_time_mw,
            observation.most_recent_forecast_mw,
            observation.most_recent_confidence_10_mw,
            observation.most_recent_confidence_90_mw,
            observation.day_ahead_11h_forecast_mw,
            observation.day_ahead_11h_confidence_10_mw,
            observation.day_ahead_11h_confidence_90_mw,
            observation.day_ahead_forecast_mw,
            observation.day_ahead_confidence_10_mw,
            observation.day_ahead_confidence_90_mw,
            observation.week_ahead_forecast_mw,
            observation.week_ahead_confidence_10_mw,
            observation.week_ahead_confidence_90_mw,
            observation.monitored_capacity_mw,
            observation.load_factor,
            _to_utc(event.ingested_at),
            _row_version(event.ingested_at, event.event_id),
        ],
    )


def _feature_insert(record: _FeatureSnapshotRecord) -> _PreparedInsert:
    return _feature_values_insert(
        event_id=record.event_id,
        cutoff=record.cutoff,
        target_time=record.target_time,
        feature_schema_hash=record.feature_schema_hash,
        local_values=record.local_values,
        local_masks=record.local_masks,
        context_values=record.context_values,
        context_masks=record.context_masks,
        static_values=record.static_values,
        static_masks=record.static_masks,
        current_state=record.current_state,
        created_at=record.created_at,
    )


def _feature_event_insert(event: EventEnvelope, payload: _FeaturePayload) -> _PreparedInsert:
    return _feature_values_insert(
        event_id=event.event_id,
        cutoff=payload.cutoff,
        target_time=payload.target_time,
        feature_schema_hash=payload.feature_schema_hash,
        local_values=payload.local_values,
        local_masks=payload.local_masks,
        context_values=payload.context_values,
        context_masks=payload.context_masks,
        static_values=payload.static_values,
        static_masks=payload.static_masks,
        current_state=payload.current_state,
        created_at=payload.created_at or event.ingested_at,
    )


def _feature_values_insert(
    *,
    event_id: str,
    cutoff: datetime,
    target_time: datetime,
    feature_schema_hash: str,
    local_values: object,
    local_masks: object,
    context_values: object,
    context_masks: object,
    static_values: object,
    static_masks: object,
    current_state: str | None,
    created_at: datetime,
) -> _PreparedInsert:
    local = _flatten_floats(local_values)
    local_mask = _flatten_masks(local_masks)
    context = _flatten_floats(context_values)
    context_mask = _flatten_masks(context_masks)
    static = _flatten_floats(static_values)
    static_mask = _flatten_masks(static_masks)
    _require_matching_vectors(local, local_mask, "local")
    _require_matching_vectors(context, context_mask, "context")
    _require_matching_vectors(static, static_mask, "static")
    return _PreparedInsert(
        "feature_snapshots",
        FEATURE_COLUMNS,
        [
            event_id,
            _to_utc(cutoff),
            _to_utc(target_time),
            feature_schema_hash,
            local,
            local_mask,
            context,
            context_mask,
            static,
            static_mask,
            current_state,
            _to_utc(created_at),
            _row_version(created_at, event_id),
        ],
    )


def _prediction_insert(event: EventEnvelope, payload: _PredictionPayload) -> _PreparedInsert:
    return _PreparedInsert(
        "predictions",
        PREDICTION_COLUMNS,
        [
            event.event_id,
            _to_utc(payload.cutoff),
            _to_utc(payload.target_time),
            _to_utc(payload.generated_at),
            payload.system_imbalance_mw,
            payload.p10_mw,
            payload.p90_mw,
            payload.flip_probability,
            int(payload.will_flip),
            payload.current_state,
            payload.predicted_state,
            payload.prediction_quality,
            payload.model_version,
            payload.feature_schema_hash,
            _row_version(payload.generated_at, event.event_id),
        ],
    )


def _outcome_insert(event: EventEnvelope, payload: _OutcomePayload) -> _PreparedInsert:
    return _PreparedInsert(
        "prediction_outcomes",
        OUTCOME_COLUMNS,
        [
            payload.prediction_event_id,
            _to_utc(payload.target_time),
            payload.realized_event_id,
            payload.realized_system_imbalance_mw,
            payload.realized_state,
            None if payload.flip_actual is None else int(payload.flip_actual),
            _to_utc(payload.evaluated_at),
            _row_version(payload.evaluated_at, event.event_id),
        ],
    )


def _model_insert(event: EventEnvelope, payload: _ModelVersionPayload) -> _PreparedInsert:
    return _PreparedInsert(
        "model_versions",
        MODEL_COLUMNS,
        [
            payload.model_version,
            payload.feature_schema_hash,
            _json_field(payload.manifest_json),
            _json_field(payload.metrics_json),
            None if payload.promoted_at is None else _to_utc(payload.promoted_at),
            _to_utc(payload.created_at),
            _row_version(payload.created_at, event.event_id),
        ],
    )


def _imbalance_from_row(row: Sequence[object]) -> ImbalanceObservation:
    timestamp, values = row
    latest = cast(Sequence[object], values)
    return ImbalanceObservation(
        timestamp=_clickhouse_utc(cast(datetime, timestamp)),
        quarter_hour=_clickhouse_utc(cast(datetime, latest[0])),
        resolution_code=cast(str, latest[1]),
        quality_status=cast(str, latest[2]),
        ace_mw=cast(float | None, latest[3]),
        system_imbalance_mw=cast(float, latest[4]),
        alpha_eur_mwh=cast(float | None, latest[5]),
        alpha_prime_eur_mwh=cast(float | None, latest[6]),
        marginal_incremental_price_eur_mwh=cast(float | None, latest[7]),
        marginal_decremental_price_eur_mwh=cast(float | None, latest[8]),
        imbalance_price_eur_mwh=cast(float | None, latest[9]),
    )


def _versioned_imbalance_from_row(row: Sequence[object]) -> VersionedImbalanceObservation:
    (
        event_id,
        timestamp,
        quarter_hour,
        resolution_code,
        quality_status,
        ace_mw,
        system_imbalance_mw,
        alpha_eur_mwh,
        alpha_prime_eur_mwh,
        marginal_incremental_price_eur_mwh,
        marginal_decremental_price_eur_mwh,
        imbalance_price_eur_mwh,
        ingested_at,
        row_version,
    ) = row
    observation = ImbalanceObservation(
        timestamp=_clickhouse_utc(cast(datetime, timestamp)),
        quarter_hour=_clickhouse_utc(cast(datetime, quarter_hour)),
        resolution_code=cast(str, resolution_code),
        quality_status=cast(str, quality_status),
        ace_mw=cast(float | None, ace_mw),
        system_imbalance_mw=cast(float, system_imbalance_mw),
        alpha_eur_mwh=cast(float | None, alpha_eur_mwh),
        alpha_prime_eur_mwh=cast(float | None, alpha_prime_eur_mwh),
        marginal_incremental_price_eur_mwh=cast(
            float | None,
            marginal_incremental_price_eur_mwh,
        ),
        marginal_decremental_price_eur_mwh=cast(
            float | None,
            marginal_decremental_price_eur_mwh,
        ),
        imbalance_price_eur_mwh=cast(float | None, imbalance_price_eur_mwh),
    )
    return VersionedImbalanceObservation(
        observation=observation,
        available_at=_clickhouse_utc(cast(datetime, ingested_at)),
        row_version=cast(int, row_version),
        event_id=cast(str, event_id),
    )


def _state_seed_from_row(
    state_balance: object,
    state_since: object,
    last_observed_at: object,
) -> ConfirmedStateSeed:
    last_observed = (
        None
        if last_observed_at is None
        else _clickhouse_utc(cast(datetime, last_observed_at))
    )
    if state_balance is None:
        return ConfirmedStateSeed(None, None, last_observed)
    state_since_utc = _clickhouse_utc(cast(datetime, state_since))
    state = (
        ConfirmedState.POSITIVE
        if cast(float, state_balance) > 0
        else ConfirmedState.NEGATIVE
    )
    return ConfirmedStateSeed(state, state_since_utc, last_observed)


def _require_event_timestamp(event: EventEnvelope, timestamp: datetime) -> None:
    if _to_utc(timestamp) != _to_utc(event.event_time):
        raise ValueError("payload timestamp does not match envelope")


def _to_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("datetime must be timezone-aware")
    return value.astimezone(UTC)


def _clickhouse_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _row_version(timestamp: datetime, identity: str) -> int:
    value = _to_utc(timestamp)
    epoch_milliseconds = int(value.timestamp() * 1_000)
    if epoch_milliseconds < 0:
        raise ValueError("row version timestamp cannot predate the Unix epoch")
    identity_suffix = int(hashlib.sha256(identity.encode("utf-8")).hexdigest()[:5], 16)
    return epoch_milliseconds * 1_048_576 + identity_suffix


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _json_field(value: str | Mapping[str, object]) -> str:
    if isinstance(value, str):
        parsed = json.loads(value)
        return _canonical_json(parsed)
    return _canonical_json(value)


def _as_sequence(value: object) -> Sequence[object]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return cast(Sequence[object], value)
    raise TypeError("feature vector must be an array")


def _flatten_floats(value: object) -> list[float]:
    flattened: list[float] = []
    for item in _as_sequence(value):
        if hasattr(item, "tolist") or (
            isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray))
        ):
            flattened.extend(_flatten_floats(item))
        elif isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError("feature values must be numeric")
        else:
            flattened.append(float(item))
    return flattened


def _flatten_masks(value: object) -> list[int]:
    flattened: list[int] = []
    for item in _as_sequence(value):
        if hasattr(item, "tolist") or (
            isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray))
        ):
            flattened.extend(_flatten_masks(item))
        elif isinstance(item, bool):
            flattened.append(int(item))
        elif isinstance(item, int) and item in (0, 1):
            flattened.append(item)
        else:
            raise TypeError("feature masks must contain only zero or one")
    return flattened


def _require_matching_vectors(values: Sequence[object], masks: Sequence[object], name: str) -> None:
    if len(values) != len(masks):
        raise ValueError(f"{name} values and masks must have equal flattened lengths")
