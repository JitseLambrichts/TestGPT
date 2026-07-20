from datetime import UTC, datetime
from types import SimpleNamespace
from typing import cast

import pytest
from clickhouse_connect.driver.asyncclient import AsyncClient
from clickhouse_connect.driver.exceptions import OperationalError

from imbalance_pipeline.storage.clickhouse import (
    ClickHouseRepository,
    DashboardRow,
    TransientStorageError,
)

START = datetime(2026, 7, 13, 4, 0, tzinfo=UTC)
END = datetime(2026, 7, 13, 10, 2, tzinfo=UTC)


def dashboard_row(
    *, realized_mw: float | None = -18.0, flip_actual: bool | None = True
) -> tuple[object, ...]:
    target = datetime(2026, 7, 13, 10, 2, tzinfo=UTC)
    return (
        "prediction-001",
        datetime(2026, 7, 13, 10, 1, tzinfo=UTC),
        target,
        datetime(2026, 7, 13, 10, 1, 7, tzinfo=UTC),
        -20.0,
        -35.0,
        -5.0,
        0.8,
        True,
        "positive",
        "negative",
        "model",
        "model-v1",
        "feature-schema-001",
        realized_mw,
        None if realized_mw is None else "negative",
        flip_actual,
        None if realized_mw is None else datetime(2026, 7, 13, 10, 2, 8, tzinfo=UTC),
    )


class RecordingClient:
    def __init__(
        self,
        *,
        rows: list[tuple[object, ...]] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.rows = rows or []
        self.error = error
        self.query_text = ""
        self.parameters: dict[str, object] = {}

    async def query(
        self,
        query: str,
        *,
        parameters: dict[str, object],
        tz_mode: str,
    ) -> SimpleNamespace:
        assert tz_mode == "aware"
        self.query_text = query
        self.parameters = parameters
        if self.error is not None:
            raise self.error
        return SimpleNamespace(result_rows=self.rows)


def repository(client: RecordingClient) -> ClickHouseRepository:
    return ClickHouseRepository(cast(AsyncClient, client), database="imbalance")


@pytest.mark.asyncio
async def test_dashboard_rows_join_canonical_outcomes() -> None:
    client = RecordingClient(rows=[dashboard_row()])

    rows = await repository(client).list_dashboard_rows(start=START, end=END, limit=500)

    assert isinstance(rows[0], DashboardRow)
    assert rows[0].realized_system_imbalance_mw == -18.0
    assert rows[0].flip_actual is True
    assert "LEFT JOIN canonical_outcomes" in client.query_text
    assert "GROUP BY prediction_event_id, target_time" in client.query_text
    assert client.parameters == {"start": START, "end": END, "limit": 500}


@pytest.mark.asyncio
async def test_dashboard_rows_preserve_pending_outcomes() -> None:
    client = RecordingClient(rows=[dashboard_row(realized_mw=None, flip_actual=None)])

    rows = await repository(client).list_dashboard_rows(start=START, end=END, limit=500)

    assert rows[0].realized_system_imbalance_mw is None
    assert rows[0].realized_state is None
    assert rows[0].flip_actual is None
    assert rows[0].evaluated_at is None


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [0, 10_001])
async def test_dashboard_rows_reject_invalid_limits(limit: int) -> None:
    with pytest.raises(ValueError, match="between 1 and 10000"):
        await repository(RecordingClient()).list_dashboard_rows(
            start=START,
            end=END,
            limit=limit,
        )


@pytest.mark.asyncio
async def test_dashboard_rows_reject_reversed_ranges() -> None:
    with pytest.raises(ValueError, match="end cannot precede start"):
        await repository(RecordingClient()).list_dashboard_rows(
            start=END,
            end=START,
            limit=500,
        )


@pytest.mark.asyncio
async def test_dashboard_rows_require_utc_aware_ranges() -> None:
    with pytest.raises(ValueError, match="UTC-aware"):
        await repository(RecordingClient()).list_dashboard_rows(
            start=START.replace(tzinfo=None),
            end=END,
            limit=500,
        )


@pytest.mark.asyncio
async def test_dashboard_query_sanitizes_storage_failures() -> None:
    client = RecordingClient(error=OperationalError("secret ClickHouse detail"))

    with pytest.raises(TransientStorageError, match="transient ClickHouse operation failed"):
        await repository(client).list_dashboard_rows(start=START, end=END, limit=500)
