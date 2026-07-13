import asyncio
import math
import random
import statistics
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import httpx
from pydantic import BaseModel, ConfigDict, field_validator

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
WEATHER_TIMEOUT = httpx.Timeout(10.0, connect=10.0, read=10.0)
MAX_ATTEMPTS = 5
CIRCULAR_RESULTANT_TOLERANCE = 1e-12
WEATHER_VARIABLES = (
    "temperature_2m",
    "cloud_cover",
    "surface_pressure",
    "precipitation",
    "wind_speed_10m",
    "wind_speed_100m",
    "wind_direction_100m",
    "wind_gusts_10m",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
)


@dataclass(frozen=True)
class WeatherLocation:
    name: str
    latitude: float
    longitude: float


WEATHER_LOCATIONS = (
    WeatherLocation("Brussels", 50.8503, 4.3517),
    WeatherLocation("Zeebrugge", 51.33, 3.2),
    WeatherLocation("Hasselt", 50.9307, 5.3325),
    WeatherLocation("Liège", 50.6337, 5.5675),
    WeatherLocation("Arlon", 49.6833, 5.8167),
)


class WeatherVariableAggregate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    mean: float | None
    minimum: float | None
    maximum: float | None
    point_mask: dict[str, bool]


class WeatherForecast(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    valid_time: datetime
    available_at: datetime
    locations: tuple[str, ...]
    variables: dict[str, WeatherVariableAggregate]

    @field_validator("valid_time", "available_at")
    @classmethod
    def require_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise ValueError("weather datetimes must be UTC-aware")
        return value.astimezone(UTC)


class WeatherClient:
    def __init__(
        self,
        enabled: bool = False,
        client: httpx.AsyncClient | None = None,
        clock: Callable[[], datetime] | None = None,
        retry_base_seconds: float = 0.25,
    ) -> None:
        if retry_base_seconds < 0:
            raise ValueError("retry_base_seconds cannot be negative")
        self._enabled = enabled
        self._client = client or httpx.AsyncClient()
        self._owns_client = client is None
        self._clock = clock or _utc_now
        self._retry_base_seconds = retry_base_seconds

    async def __aenter__(self) -> "WeatherClient":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def iter_current_state(
        self,
        start: datetime,
        end: datetime,
        historical: bool,
        availability_cutoff: datetime | None = None,
    ) -> AsyncIterator[WeatherForecast]:
        if not self._enabled:
            return

        start_utc = _normalize_boundary(start)
        end_utc = _normalize_boundary(end)
        if start_utc >= end_utc:
            raise ValueError("weather start must be before end")
        cutoff_utc = (
            _normalize_boundary(availability_cutoff)
            if availability_cutoff is not None
            else None
        )

        endpoint = HISTORICAL_FORECAST_URL if historical else FORECAST_URL
        response, receipt_time = await self._get_with_retry(
            endpoint,
            _weather_query(start_utc, end_utc),
            capture_receipt_time=not historical,
        )
        payloads = _weather_payloads(response)

        point_hours = [_point_hourly(payload) for payload in payloads]
        reference_times = point_hours[0]["time"]
        for point in point_hours[1:]:
            if point["time"] != reference_times:
                raise ValueError("Open-Meteo point responses must share hourly timestamps")

        for index, source_time in enumerate(reference_times):
            valid_time = _open_meteo_utc_time(source_time)
            if valid_time < start_utc or valid_time >= end_utc:
                continue
            if historical:
                available_at = valid_time
            else:
                if receipt_time is None:
                    raise RuntimeError("live weather receipt time was not captured")
                available_at = receipt_time
            if cutoff_utc is not None and available_at > cutoff_utc:
                continue
            yield WeatherForecast(
                valid_time=valid_time,
                available_at=available_at,
                locations=tuple(location.name for location in WEATHER_LOCATIONS),
                variables={
                    variable: _aggregate_variable(variable, point_hours, index)
                    for variable in WEATHER_VARIABLES
                },
            )

    async def _get_with_retry(
        self,
        endpoint: str,
        params: Mapping[str, str],
        capture_receipt_time: bool,
    ) -> tuple[httpx.Response, datetime | None]:
        for attempt in range(MAX_ATTEMPTS):
            try:
                response = await self._client.get(
                    endpoint,
                    params=params,
                    timeout=WEATHER_TIMEOUT,
                )
            except (httpx.ConnectError, httpx.ConnectTimeout):
                if attempt == MAX_ATTEMPTS - 1:
                    raise
                await self._wait_before_retry(attempt)
                continue

            if response.status_code == 429 or response.status_code >= 500:
                if attempt < MAX_ATTEMPTS - 1:
                    await self._wait_before_retry(attempt)
                    continue
            if response.is_success:
                receipt_time = (
                    _normalize_boundary(self._clock()) if capture_receipt_time else None
                )
                return response, receipt_time
            response.raise_for_status()

        raise RuntimeError("unreachable retry state")

    async def _wait_before_retry(self, attempt: int) -> None:
        ceiling = self._retry_base_seconds * 2**attempt
        await asyncio.sleep(random.uniform(0.0, ceiling))


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _normalize_boundary(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("weather boundaries must be timezone-aware")
    return value.astimezone(UTC)


def _weather_query(start: datetime, end: datetime) -> dict[str, str]:
    return {
        "latitude": ",".join(format(location.latitude, "g") for location in WEATHER_LOCATIONS),
        "longitude": ",".join(
            format(location.longitude, "g") for location in WEATHER_LOCATIONS
        ),
        "hourly": ",".join(WEATHER_VARIABLES),
        "timezone": "UTC",
        "start_date": start.date().isoformat(),
        "end_date": end.date().isoformat(),
    }


def _weather_payloads(response: httpx.Response) -> list[Mapping[str, object]]:
    payload: object = response.json()
    if isinstance(payload, dict):
        payloads: list[object] = [payload]
    elif isinstance(payload, list):
        payloads = payload
    else:
        raise ValueError("Open-Meteo response must be an object or list of objects")
    if len(payloads) != len(WEATHER_LOCATIONS):
        raise ValueError(f"Open-Meteo response must contain {len(WEATHER_LOCATIONS)} points")
    if not all(isinstance(item, dict) for item in payloads):
        raise ValueError("Open-Meteo point responses must be JSON objects")
    return payloads  # type: ignore[return-value]


def _point_hourly(payload: Mapping[str, object]) -> dict[str, Sequence[object]]:
    if payload.get("utc_offset_seconds") != 0:
        raise ValueError("Open-Meteo response must use UTC")
    raw_hourly = payload.get("hourly")
    if not isinstance(raw_hourly, dict):
        raise ValueError("Open-Meteo point response must contain hourly values")

    hourly: dict[str, Sequence[object]] = {}
    expected_length: int | None = None
    for key in ("time", *WEATHER_VARIABLES):
        values = raw_hourly.get(key)
        if not isinstance(values, list):
            raise ValueError(f"Open-Meteo hourly {key!r} must be a list")
        if expected_length is None:
            expected_length = len(values)
        elif len(values) != expected_length:
            raise ValueError("Open-Meteo hourly arrays must have equal lengths")
        hourly[key] = values
    return hourly


def _open_meteo_utc_time(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("Open-Meteo hourly timestamps must be strings")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _aggregate_variable(
    variable: str,
    point_hours: Sequence[Mapping[str, Sequence[object]]],
    index: int,
) -> WeatherVariableAggregate:
    available_values: list[float] = []
    point_mask: dict[str, bool] = {}
    for location, hourly in zip(WEATHER_LOCATIONS, point_hours, strict=True):
        raw_value = hourly[variable][index]
        if raw_value is None:
            point_mask[location.name] = False
            continue
        if isinstance(raw_value, bool) or not isinstance(raw_value, int | float):
            raise ValueError(f"Open-Meteo {variable!r} values must be numeric or null")
        point_mask[location.name] = True
        available_values.append(float(raw_value))

    if not available_values:
        return WeatherVariableAggregate(
            mean=None,
            minimum=None,
            maximum=None,
            point_mask=point_mask,
        )
    if variable == "wind_direction_100m":
        return WeatherVariableAggregate(
            mean=_circular_mean_degrees(available_values),
            minimum=None,
            maximum=None,
            point_mask=point_mask,
        )
    return WeatherVariableAggregate(
        mean=statistics.fmean(available_values),
        minimum=min(available_values),
        maximum=max(available_values),
        point_mask=point_mask,
    )


def _circular_mean_degrees(values: Sequence[float]) -> float | None:
    sine = statistics.fmean(math.sin(math.radians(value)) for value in values)
    cosine = statistics.fmean(math.cos(math.radians(value)) for value in values)
    if math.hypot(sine, cosine) < CIRCULAR_RESULTANT_TOLERANCE:
        return None
    return math.degrees(math.atan2(sine, cosine)) % 360.0
