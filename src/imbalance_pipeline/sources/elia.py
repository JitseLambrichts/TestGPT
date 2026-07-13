import asyncio
import json
import random
from collections.abc import AsyncIterator, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from urllib.parse import quote

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator

from imbalance_pipeline.domain.imbalance import ImbalanceObservation

DEFAULT_ELIA_BASE_URL = "https://opendata.elia.be/api/explore/v2.1"
ELIA_TIMEOUT = httpx.Timeout(10.0, connect=10.0, read=10.0)
MAX_ATTEMPTS = 5
MAX_PAGE_SIZE = 100
MAX_RESULT_WINDOW = 10_000
DATASET_ORDER_BY = {
    "ods086": "datetime asc, offshoreonshore asc, region asc, gridconnectiontype asc",
    "ods087": "datetime asc, region asc",
}


@dataclass(frozen=True)
class _EliaPage:
    total_count: int
    records: list[dict[str, object]]


class _TimedEliaRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    timestamp: datetime = Field(validation_alias="datetime")
    resolution_code: str = Field(validation_alias="resolutioncode")

    @field_validator("timestamp")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        return _normalize_source_datetime(value)


class _ImbalanceSourceRecord(_TimedEliaRecord):
    quarter_hour: datetime = Field(validation_alias="quarterhour")
    quality_status: str = Field(validation_alias="qualitystatus")
    ace_mw: float | None = Field(validation_alias="ace")
    system_imbalance_mw: float = Field(validation_alias="systemimbalance")
    alpha_eur_mwh: float | None = Field(validation_alias="alpha")
    alpha_prime_eur_mwh: float | None = Field(validation_alias="alpha_prime")
    marginal_incremental_price_eur_mwh: float | None = Field(
        validation_alias="marginalincrementalprice"
    )
    marginal_decremental_price_eur_mwh: float | None = Field(
        validation_alias="marginaldecrementalprice"
    )
    imbalance_price_eur_mwh: float | None = Field(validation_alias="imbalanceprice")

    @field_validator("quarter_hour")
    @classmethod
    def normalize_quarter_hour(cls, value: datetime) -> datetime:
        return _normalize_source_datetime(value)


class LoadObservation(_TimedEliaRecord):
    measured_mw: float | None = Field(validation_alias="measured")
    most_recent_forecast_mw: float | None = Field(validation_alias="mostrecentforecast")
    most_recent_confidence_10_mw: float | None = Field(
        validation_alias="mostrecentconfidence10"
    )
    most_recent_confidence_90_mw: float | None = Field(
        validation_alias="mostrecentconfidence90"
    )
    day_ahead_forecast_mw: float | None = Field(validation_alias="dayaheadforecast")
    day_ahead_confidence_10_mw: float | None = Field(
        validation_alias="dayaheadconfidence10"
    )
    day_ahead_confidence_90_mw: float | None = Field(
        validation_alias="dayaheadconfidence90"
    )
    week_ahead_forecast_mw: float | None = Field(validation_alias="weekaheadforecast")


class WindObservation(_TimedEliaRecord):
    offshore_onshore: str = Field(validation_alias="offshoreonshore")
    region: str = Field(validation_alias="region")
    grid_connection_type: str = Field(validation_alias="gridconnectiontype")
    real_time_mw: float | None = Field(validation_alias="realtime")
    most_recent_forecast_mw: float | None = Field(validation_alias="mostrecentforecast")
    most_recent_confidence_10_mw: float | None = Field(
        validation_alias="mostrecentconfidence10"
    )
    most_recent_confidence_90_mw: float | None = Field(
        validation_alias="mostrecentconfidence90"
    )
    day_ahead_11h_forecast_mw: float | None = Field(validation_alias="dayahead11hforecast")
    day_ahead_11h_confidence_10_mw: float | None = Field(
        validation_alias="dayahead11hconfidence10"
    )
    day_ahead_11h_confidence_90_mw: float | None = Field(
        validation_alias="dayahead11hconfidence90"
    )
    day_ahead_forecast_mw: float | None = Field(validation_alias="dayaheadforecast")
    day_ahead_confidence_10_mw: float | None = Field(
        validation_alias="dayaheadconfidence10"
    )
    day_ahead_confidence_90_mw: float | None = Field(
        validation_alias="dayaheadconfidence90"
    )
    week_ahead_forecast_mw: float | None = Field(validation_alias="weekaheadforecast")
    week_ahead_confidence_10_mw: float | None = Field(
        validation_alias="weekaheadconfidence10"
    )
    week_ahead_confidence_90_mw: float | None = Field(
        validation_alias="weekaheadconfidence90"
    )
    monitored_capacity_mw: float | None = Field(validation_alias="monitoredcapacity")
    load_factor: float | None = Field(validation_alias="loadfactor")
    decremental_bid_id: str | int | None = Field(validation_alias="decrementalbidid")


class SolarObservation(_TimedEliaRecord):
    region: str = Field(validation_alias="region")
    real_time_mw: float | None = Field(validation_alias="realtime")
    most_recent_forecast_mw: float | None = Field(validation_alias="mostrecentforecast")
    most_recent_confidence_10_mw: float | None = Field(
        validation_alias="mostrecentconfidence10"
    )
    most_recent_confidence_90_mw: float | None = Field(
        validation_alias="mostrecentconfidence90"
    )
    day_ahead_11h_forecast_mw: float | None = Field(validation_alias="dayahead11hforecast")
    day_ahead_11h_confidence_10_mw: float | None = Field(
        validation_alias="dayahead11hconfidence10"
    )
    day_ahead_11h_confidence_90_mw: float | None = Field(
        validation_alias="dayahead11hconfidence90"
    )
    day_ahead_forecast_mw: float | None = Field(validation_alias="dayaheadforecast")
    day_ahead_confidence_10_mw: float | None = Field(
        validation_alias="dayaheadconfidence10"
    )
    day_ahead_confidence_90_mw: float | None = Field(
        validation_alias="dayaheadconfidence90"
    )
    week_ahead_forecast_mw: float | None = Field(validation_alias="weekaheadforecast")
    week_ahead_confidence_10_mw: float | None = Field(
        validation_alias="weekaheadconfidence10"
    )
    week_ahead_confidence_90_mw: float | None = Field(
        validation_alias="weekaheadconfidence90"
    )
    load_factor: float | None = Field(validation_alias="loadfactor")
    monitored_capacity_mw: float | None = Field(validation_alias="monitoredcapacity")


def normalize_imbalance(record: Mapping[str, object]) -> ImbalanceObservation:
    source = _ImbalanceSourceRecord.model_validate(record)
    if source.resolution_code != "PT1M":
        raise ValueError(
            f"ODS161 one-minute observations require PT1M, got {source.resolution_code!r}"
        )
    return ImbalanceObservation(
        timestamp=source.timestamp,
        quarter_hour=source.quarter_hour,
        resolution_code=source.resolution_code,
        quality_status=source.quality_status,
        ace_mw=source.ace_mw,
        system_imbalance_mw=source.system_imbalance_mw,
        alpha_eur_mwh=source.alpha_eur_mwh,
        alpha_prime_eur_mwh=source.alpha_prime_eur_mwh,
        marginal_incremental_price_eur_mwh=source.marginal_incremental_price_eur_mwh,
        marginal_decremental_price_eur_mwh=source.marginal_decremental_price_eur_mwh,
        imbalance_price_eur_mwh=source.imbalance_price_eur_mwh,
    )


def normalize_load(record: Mapping[str, object]) -> LoadObservation:
    observation = LoadObservation.model_validate(record)
    _require_quarter_hour_resolution(observation.resolution_code, "ODS002")
    return observation


def normalize_wind(record: Mapping[str, object]) -> WindObservation:
    observation = WindObservation.model_validate(record)
    _require_quarter_hour_resolution(observation.resolution_code, "ODS086")
    return observation


def normalize_solar(record: Mapping[str, object]) -> SolarObservation:
    observation = SolarObservation.model_validate(record)
    _require_quarter_hour_resolution(observation.resolution_code, "ODS087")
    return observation


class EliaClient:
    def __init__(
        self,
        base_url: str = DEFAULT_ELIA_BASE_URL,
        client: httpx.AsyncClient | None = None,
        page_size: int = 100,
        retry_base_seconds: float = 0.25,
    ) -> None:
        if not 1 <= page_size <= MAX_PAGE_SIZE:
            raise ValueError("page_size must be between 1 and 100")
        if retry_base_seconds < 0:
            raise ValueError("retry_base_seconds cannot be negative")
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.AsyncClient()
        self._owns_client = client is None
        self._page_size = page_size
        self._retry_base_seconds = retry_base_seconds

    async def __aenter__(self) -> "EliaClient":
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

    async def fetch_page(
        self,
        dataset: str,
        limit: int,
        offset: int,
        where: str | None,
        order_by: str,
    ) -> list[dict[str, object]]:
        page = await self._fetch_page(dataset, limit, offset, where, order_by)
        return page.records

    async def _fetch_page(
        self,
        dataset: str,
        limit: int,
        offset: int,
        where: str | None,
        order_by: str,
    ) -> _EliaPage:
        url = f"{self._base_url}/catalog/datasets/{quote(dataset, safe='')}/records"
        params: dict[str, str | int] = {
            "limit": limit,
            "offset": offset,
            "order_by": order_by,
        }
        if where is not None:
            params["where"] = where

        for attempt in range(MAX_ATTEMPTS):
            try:
                response = await self._client.get(
                    url,
                    params=params,
                    timeout=ELIA_TIMEOUT,
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
            response.raise_for_status()
            return _response_page(response)

        raise RuntimeError("unreachable retry state")

    async def iter_records(
        self,
        dataset: str,
        start: datetime | None,
        end: datetime | None,
    ) -> AsyncIterator[dict[str, object]]:
        start_utc = _normalize_query_boundary(start)
        end_utc = _normalize_query_boundary(end)
        for window_start, window_end in _query_windows(start_utc, end_utc):
            async for record in self._iter_window(
                dataset,
                window_start,
                window_end,
            ):
                yield record

    async def _iter_window(
        self,
        dataset: str,
        start: datetime | None,
        end: datetime | None,
    ) -> AsyncIterator[dict[str, object]]:
        conditions: list[str] = []
        if start is not None:
            conditions.append(f"datetime >= {_query_string(start.isoformat())}")
        if end is not None:
            conditions.append(f"datetime < {_query_string(end.isoformat())}")
        where = " AND ".join(conditions) or None

        offset = 0
        consumed = 0
        expected_total: int | None = None
        seen_pages: set[str] = set()
        while True:
            if offset + self._page_size >= MAX_RESULT_WINDOW:
                raise RuntimeError("Explore pagination would reach the 10,000-row result window")
            page = await self._fetch_page(
                dataset=dataset,
                limit=self._page_size,
                offset=offset,
                where=where,
                order_by=DATASET_ORDER_BY.get(dataset.lower(), "datetime asc"),
            )
            if expected_total is None:
                expected_total = page.total_count
                if _last_request_reaches_result_limit(expected_total, self._page_size):
                    raise RuntimeError(
                        "Explore query exceeds the safe result window; use a narrower time range"
                    )
            if consumed >= expected_total:
                if page.records:
                    raise RuntimeError("Explore returned records beyond total_count")
                return
            if not page.records:
                raise RuntimeError("Explore pagination made no progress before total_count")

            fingerprint = json.dumps(
                page.records,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            if fingerprint in seen_pages:
                raise RuntimeError("Explore pagination returned a repeated page")
            seen_pages.add(fingerprint)

            remaining = expected_total - consumed
            if len(page.records) > remaining:
                raise RuntimeError("Explore returned more records than total_count")
            for record in page.records:
                yield record
            consumed += len(page.records)
            if consumed == expected_total:
                return
            if len(page.records) < self._page_size:
                raise RuntimeError("Explore pagination made no progress before total_count")
            offset += len(page.records)

    async def _wait_before_retry(self, attempt: int) -> None:
        ceiling = self._retry_base_seconds * 2**attempt
        await asyncio.sleep(random.uniform(0.0, ceiling))


def _normalize_source_datetime(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("source datetimes must be timezone-aware")
    return value.astimezone(UTC)


def _normalize_query_boundary(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return _normalize_source_datetime(value)


def _query_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


def _response_page(response: httpx.Response) -> _EliaPage:
    payload: object = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Elia response must be a JSON object")
    total_count = payload.get("total_count")
    if isinstance(total_count, bool) or not isinstance(total_count, int) or total_count < 0:
        raise ValueError("Elia response must contain a non-negative total_count")
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError("Elia response must contain a results list")

    records: list[dict[str, object]] = []
    for result in results:
        if not isinstance(result, dict) or not all(isinstance(key, str) for key in result):
            raise ValueError("Elia result records must be JSON objects with string keys")
        records.append(result)
    return _EliaPage(total_count=total_count, records=records)


def _require_quarter_hour_resolution(resolution_code: str, dataset: str) -> None:
    if resolution_code != "PT15M":
        raise ValueError(f"{dataset} quarter-hour observations require PT15M")


def _query_windows(
    start: datetime | None,
    end: datetime | None,
) -> Iterator[tuple[datetime | None, datetime | None]]:
    if start is None or end is None:
        yield start, end
        return
    cursor = start
    while cursor < end:
        next_midnight = (cursor + timedelta(days=1)).replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )
        window_end = min(next_midnight, end)
        yield cursor, window_end
        cursor = window_end


def _last_request_reaches_result_limit(total_count: int, page_size: int) -> bool:
    if total_count == 0:
        return False
    last_offset = ((total_count - 1) // page_size) * page_size
    return last_offset + page_size >= MAX_RESULT_WINDOW
