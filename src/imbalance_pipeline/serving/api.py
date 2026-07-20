import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request, Response, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CollectorRegistry, Counter, generate_latest
from prometheus_client.exposition import CONTENT_TYPE_LATEST
from pydantic import BaseModel

from imbalance_pipeline.config import Settings, get_settings
from imbalance_pipeline.model.bundle import ModelManifest
from imbalance_pipeline.storage.clickhouse import (
    ClickHouseRepository,
    DashboardRow,
    Prediction,
    TransientStorageError,
)


class PredictionRepository(Protocol):
    async def latest_prediction(self) -> Prediction | None: ...

    async def list_predictions(
        self,
        *,
        start: datetime,
        end: datetime,
        limit: int,
        after_target_time: datetime | None,
        after_event_id: str | None,
    ) -> list[Prediction]: ...

    async def list_dashboard_rows(
        self,
        *,
        start: datetime,
        end: datetime,
        limit: int,
    ) -> list[DashboardRow]: ...


@dataclass(frozen=True, slots=True)
class ModelStatus:
    model_version: str | None
    feature_schema_hash: str | None
    fallback_enabled: bool


class PredictionCursor(BaseModel):
    event_id: str
    target_time: datetime


class PredictionPage(BaseModel):
    items: list[Prediction]
    next_cursor: PredictionCursor | None


class DashboardPage(BaseModel):
    items: list[DashboardRow]


class _ApiMetrics:
    def __init__(self, registry: CollectorRegistry) -> None:
        self.requests = Counter(
            "imbalance_api_requests",
            "HTTP requests handled by the imbalance prediction API.",
            ("route", "status"),
            registry=registry,
        )
        self.registry = registry


def create_app(
    repository: PredictionRepository,
    *,
    registry: CollectorRegistry | None = None,
    model_status: ModelStatus | None = None,
) -> FastAPI:
    """Create the read-only prediction API around an already-connected repository."""
    metrics = _ApiMetrics(registry or CollectorRegistry())
    active_model = model_status or ModelStatus(
        model_version=None,
        feature_schema_hash=None,
        fallback_enabled=True,
    )
    app = FastAPI(title="Belgian Imbalance Prediction API", version="1")
    dashboard_dir = Path(__file__).with_name("dashboard")
    app.mount(
        "/dashboard-assets",
        StaticFiles(directory=dashboard_dir),
        name="dashboard-assets",
    )

    @app.middleware("http")
    async def record_request(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        response = await call_next(request)
        metrics.requests.labels(route=request.url.path, status=str(response.status_code)).inc()
        return response

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz", include_in_schema=False)
    @app.get("/health/ready")
    async def readyz() -> dict[str, str]:
        await _probe_repository(repository)
        if active_model.model_version is None and not active_model.fallback_enabled:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="no validated model or fallback is available",
            )
        return {"status": "ready"}

    @app.get("/v1/predictions/latest", response_model=Prediction)
    async def latest_prediction() -> Prediction:
        try:
            prediction = await repository.latest_prediction()
        except TransientStorageError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="prediction store is temporarily unavailable",
            ) from exc
        if prediction is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="no prediction is available yet",
            )
        return prediction

    @app.get("/v1/predictions", response_model=PredictionPage)
    async def prediction_range(
        start: datetime,
        end: datetime,
        limit: int = Query(default=1_000, ge=1, le=10_000),
        after_target_time: datetime | None = None,
        after_event_id: str | None = None,
    ) -> PredictionPage:
        start = _utc_query_timestamp(start, "start")
        end = _utc_query_timestamp(end, "end")
        if end < start:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="end must not precede start",
            )
        if (after_target_time is None) != (after_event_id is None):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="after_target_time and after_event_id must be supplied together",
            )
        cursor_time = (
            None
            if after_target_time is None
            else _utc_query_timestamp(after_target_time, "after_target_time")
        )
        try:
            records = await repository.list_predictions(
                start=start,
                end=end,
                limit=limit + 1,
                after_target_time=cursor_time,
                after_event_id=after_event_id,
            )
        except TransientStorageError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="prediction store is temporarily unavailable",
            ) from exc
        items = records[:limit]
        next_cursor = (
            PredictionCursor(event_id=items[-1].event_id, target_time=items[-1].target_time)
            if len(records) > limit
            else None
        )
        return PredictionPage(items=items, next_cursor=next_cursor)

    @app.get("/v1/models/current", response_model=ModelStatus)
    async def current_model() -> ModelStatus:
        return active_model

    @app.get("/v1/dashboard", response_model=DashboardPage)
    async def dashboard_data(
        start: datetime,
        end: datetime,
        limit: int = Query(default=500, ge=1, le=10_000),
    ) -> DashboardPage:
        start = _utc_query_timestamp(start, "start")
        end = _utc_query_timestamp(end, "end")
        if end < start:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="end must not precede start",
            )
        try:
            rows = await repository.list_dashboard_rows(start=start, end=end, limit=limit)
        except TransientStorageError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="prediction store is temporarily unavailable",
            ) from exc
        return DashboardPage(items=rows)

    @app.get("/dashboard", response_class=FileResponse, include_in_schema=False)
    async def dashboard_page() -> FileResponse:
        return FileResponse(dashboard_dir / "index.html")

    @app.get("/metrics", include_in_schema=False)
    async def metrics_endpoint() -> Response:
        return Response(content=generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    return app


async def _probe_repository(repository: PredictionRepository) -> None:
    try:
        await repository.latest_prediction()
    except TransientStorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="prediction store is temporarily unavailable",
        ) from exc


def _utc_query_timestamp(value: datetime, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"{name} must be UTC-aware",
        )
    return value.astimezone(UTC)


async def _run_api(settings: Settings) -> None:
    repository = await ClickHouseRepository.connect(settings)
    model_status = _model_status(settings)
    server = uvicorn.Server(
        uvicorn.Config(
            create_app(repository, model_status=model_status),
            host=settings.api_host,
            port=settings.api_port,
            log_level="info",
        )
    )
    try:
        await server.serve()
    finally:
        await repository.aclose()


def _model_status(settings: Settings) -> ModelStatus:
    try:
        manifest = ModelManifest.load(settings.model_dir)
    except ValueError:
        return ModelStatus(
            model_version=None,
            feature_schema_hash=None,
            fallback_enabled=settings.allow_fallback,
        )
    return ModelStatus(
        model_version=manifest.model_version,
        feature_schema_hash=manifest.feature_schema_hash,
        fallback_enabled=settings.allow_fallback,
    )


def main() -> None:
    asyncio.run(_run_api(get_settings()))
