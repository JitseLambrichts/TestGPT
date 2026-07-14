import asyncio
import math
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Final, Protocol, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator

from imbalance_pipeline.config import get_settings
from imbalance_pipeline.domain.events import EventEnvelope, Subject, event_id, utc_milliseconds
from imbalance_pipeline.domain.imbalance import ConfirmedState, advance_state
from imbalance_pipeline.features.engine import FeatureSnapshot
from imbalance_pipeline.features.schema import FeatureRegistry
from imbalance_pipeline.messaging.base import EventBus, Message
from imbalance_pipeline.services.sink import DeadLetterPayload
from imbalance_pipeline.serving.runtime import OnnxEnsemble, PredictionValues
from imbalance_pipeline.storage.clickhouse import DeadLetterReason

PREDICTOR_DLQ_SUBJECT: Final = "grid.dlq.predictor.v1"
PREDICTOR_SUBSCRIPTION: Final = (Subject.FEATURES_IMBALANCE.value, "predictor-v1")
_FALLBACK_MODEL_VERSION: Final = "fallback-v1"


class PredictorRuntime(Protocol):
    @property
    def model_version(self) -> str: ...

    @property
    def feature_schema_hash(self) -> str: ...

    def predict(self, snapshot: FeatureSnapshot) -> PredictionValues: ...


class _FeaturePayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    cutoff: datetime
    target_time: datetime
    feature_schema_hash: str
    local_values: object
    local_masks: object
    context_values: object
    context_masks: object
    static_values: object
    static_masks: object
    current_state: ConfirmedState | None = None
    model_eligible: bool
    observed_imbalance_minutes: int
    created_at: datetime

    @field_validator("cutoff", "target_time", "created_at")
    @classmethod
    def require_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
            raise ValueError("feature snapshot timestamps must be UTC-aware")
        return value.astimezone(UTC)

    @field_validator("feature_schema_hash")
    @classmethod
    def require_schema_hash(cls, value: str) -> str:
        if not value:
            raise ValueError("feature_schema_hash cannot be empty")
        return value

    @field_validator("observed_imbalance_minutes")
    @classmethod
    def require_non_negative_count(cls, value: int) -> int:
        if value < 0:
            raise ValueError("observed_imbalance_minutes cannot be negative")
        return value

    @model_validator(mode="after")
    def validate_snapshot_semantics(self) -> "_FeaturePayload":
        if self.target_time != self.cutoff + timedelta(minutes=1):
            raise ValueError("feature snapshot target_time must be one minute after cutoff")
        if self.model_eligible and self.current_state is None:
            raise ValueError("an eligible feature snapshot requires a confirmed state")
        return self


class _FallbackUnavailable(ValueError):
    pass


class PredictorService:
    """Turns immutable feature snapshots into idempotent prediction events."""

    def __init__(
        self,
        runtime: PredictorRuntime | None,
        bus: EventBus,
        *,
        allow_fallback: bool = True,
        deadband_mw: float = 10.0,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if deadband_mw <= 0:
            raise ValueError("deadband_mw must be positive")
        self._runtime = runtime
        self._bus = bus
        self._allow_fallback = allow_fallback
        self._deadband_mw = deadband_mw
        self._clock = clock or (lambda: datetime.now(UTC))

    async def handle(self, message: Message) -> None:
        try:
            snapshot = _feature_snapshot(_feature_payload(message.event))
        except (TypeError, ValidationError, ValueError):
            await self._reject(message, _invalid_reason(message.event))
            return

        try:
            values, quality, degraded_reason = self._predict(snapshot)
        except _FallbackUnavailable:
            await self._reject(message, DeadLetterReason.INVALID_PAYLOAD)
            return
        except Exception:
            await message.nak(5.0)
            return

        try:
            generated_at = utc_milliseconds(self._clock())
            await self._bus.publish(
                Subject.PREDICTIONS_IMBALANCE.value,
                _prediction_event(
                    message.event,
                    snapshot,
                    values,
                    generated_at=generated_at,
                    prediction_quality=quality,
                    degraded_reason=degraded_reason,
                ),
            )
            await message.ack()
        except Exception:
            await message.nak(5.0)

    async def run(self) -> None:
        subject, durable = PREDICTOR_SUBSCRIPTION
        async for message in self._bus.messages(subject, durable):
            await self.handle(message)

    def _predict(self, snapshot: FeatureSnapshot) -> tuple[PredictionValues, str, str | None]:
        if not snapshot.model_eligible:
            return self._fallback(snapshot, "insufficient_feature_history")
        if self._runtime is None:
            if not self._allow_fallback:
                raise RuntimeError("no validated prediction runtime is available")
            return self._fallback(snapshot, "model_bundle_unavailable")
        try:
            values = self._runtime.predict(snapshot)
            _validate_prediction(values, snapshot)
        except Exception:
            if not self._allow_fallback:
                raise
            return self._fallback(snapshot, "model_inference_failed")
        return values, "model", None

    def _fallback(
        self,
        snapshot: FeatureSnapshot,
        reason: str,
    ) -> tuple[PredictionValues, str, str]:
        if not self._allow_fallback:
            raise RuntimeError("fallback predictions are disabled")
        if snapshot.local_values.shape[0] < 1 or snapshot.local_values.shape[1] < 1:
            raise _FallbackUnavailable("feature snapshot has no local imbalance value")
        if snapshot.local_masks.shape != snapshot.local_values.shape or not bool(
            snapshot.local_masks[-1, 0]
        ):
            raise _FallbackUnavailable("feature snapshot has no observed latest imbalance")
        latest_imbalance = float(snapshot.local_values[-1, 0])
        if not math.isfinite(latest_imbalance):
            raise _FallbackUnavailable("latest imbalance must be finite")
        predicted_state = advance_state(
            snapshot.current_state,
            latest_imbalance,
            deadband_mw=self._deadband_mw,
        )
        model_version = _runtime_model_version(self._runtime)
        return (
            PredictionValues(
                model_version=model_version,
                feature_schema_hash=snapshot.feature_schema_hash,
                system_imbalance_mw=latest_imbalance,
                p10_mw=latest_imbalance,
                p90_mw=latest_imbalance,
                flip_probability=0.0,
                will_flip=False,
                current_state=snapshot.current_state,
                predicted_state=predicted_state,
                delta_mw=0.0,
            ),
            "degraded",
            reason,
        )

    async def _reject(self, message: Message, reason: DeadLetterReason) -> None:
        event = message.event
        payload = DeadLetterPayload(
            original_event=event,
            reason=reason,
            delivery_count=message.delivery_count,
        )
        try:
            await self._bus.publish(
                PREDICTOR_DLQ_SUBJECT,
                EventEnvelope(
                    event_id=event_id(
                        "predictor",
                        "dead-letter",
                        f"{event.event_id}:{reason.value}",
                    ),
                    event_type="predictor.event.rejected",
                    source="predictor",
                    dataset="dead-letter",
                    event_time=event.event_time,
                    observed_at=event.observed_at,
                    ingested_at=event.ingested_at,
                    correlation_id=event.correlation_id,
                    causation_id=event.event_id,
                    quality_status="rejected",
                    payload=payload.model_dump(mode="json"),
                ),
            )
            await message.ack()
        except Exception:
            await message.nak(5.0)


def _feature_payload(event: EventEnvelope) -> _FeaturePayload:
    if event.event_type != "imbalance.feature.snapshot":
        raise ValueError("predictor accepts only feature snapshots")
    if event.schema_version != "1":
        raise ValueError("unsupported feature snapshot schema version")
    payload = _FeaturePayload.model_validate(event.payload)
    if payload.cutoff != event.event_time:
        raise ValueError("feature snapshot cutoff does not match event time")
    if payload.created_at != event.ingested_at:
        raise ValueError("feature snapshot creation time does not match envelope")
    return payload


def _feature_snapshot(payload: _FeaturePayload) -> FeatureSnapshot:
    local_values = _float_array(payload.local_values, dimensions=2, name="local_values")
    local_masks = _mask_array(payload.local_masks, shape=local_values.shape, name="local_masks")
    context_values = _float_array(payload.context_values, dimensions=2, name="context_values")
    context_masks = _mask_array(
        payload.context_masks,
        shape=context_values.shape,
        name="context_masks",
    )
    static_values = _float_array(payload.static_values, dimensions=1, name="static_values")
    static_masks = _mask_array(payload.static_masks, shape=static_values.shape, name="static_masks")
    for values in (
        local_values,
        local_masks,
        context_values,
        context_masks,
        static_values,
        static_masks,
    ):
        values.setflags(write=False)
    return FeatureSnapshot(
        event_id="",
        cutoff=payload.cutoff,
        knowledge_cutoff=payload.created_at,
        target_time=payload.target_time,
        feature_schema_hash=payload.feature_schema_hash,
        local_values=local_values,
        local_masks=local_masks,
        context_values=context_values,
        context_masks=context_masks,
        static_values=static_values,
        static_masks=static_masks,
        current_state=payload.current_state,
        model_eligible=payload.model_eligible,
        observed_imbalance_minutes=payload.observed_imbalance_minutes,
        created_at=payload.created_at,
    )


def _float_array(
    value: object,
    *,
    dimensions: int,
    name: str,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float32]]:
    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric array") from exc
    if array.ndim != dimensions or array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a non-empty finite {dimensions}-D array")
    return array


def _mask_array(
    value: object,
    *,
    shape: tuple[int, ...],
    name: str,
) -> np.ndarray[tuple[int, ...], np.dtype[np.uint8]]:
    try:
        array = np.asarray(value, dtype=np.uint8)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a mask array") from exc
    if array.shape != shape or not np.isin(array, (0, 1)).all():
        raise ValueError(f"{name} must contain only 0/1 values matching its feature shape")
    return array


def _validate_prediction(values: PredictionValues, snapshot: FeatureSnapshot) -> None:
    if values.feature_schema_hash != snapshot.feature_schema_hash:
        raise ValueError("model prediction schema does not match feature snapshot")
    if values.current_state is not snapshot.current_state:
        raise ValueError("model prediction current state does not match feature snapshot")
    numeric_values = (
        values.system_imbalance_mw,
        values.p10_mw,
        values.p90_mw,
        values.flip_probability,
        values.delta_mw,
    )
    if not all(math.isfinite(value) for value in numeric_values):
        raise ValueError("model prediction contains non-finite values")
    if values.p10_mw > values.p90_mw or not 0.0 <= values.flip_probability <= 1.0:
        raise ValueError("model prediction has invalid interval or flip probability")
    if values.will_flip and (
        values.current_state is None or values.predicted_state is values.current_state
    ):
        raise ValueError("model flip decision is inconsistent with predicted state")


def _runtime_model_version(runtime: PredictorRuntime | None) -> str:
    if runtime is None:
        return _FALLBACK_MODEL_VERSION
    try:
        return runtime.model_version
    except Exception:
        return _FALLBACK_MODEL_VERSION


def _prediction_event(
    trigger: EventEnvelope,
    snapshot: FeatureSnapshot,
    values: PredictionValues,
    *,
    generated_at: datetime,
    prediction_quality: str,
    degraded_reason: str | None,
) -> EventEnvelope:
    current_state = values.current_state.value if values.current_state is not None else None
    predicted_state = values.predicted_state.value if values.predicted_state is not None else None
    payload: dict[str, object] = {
        "cutoff": _timestamp(snapshot.cutoff),
        "target_time": _timestamp(snapshot.target_time),
        "generated_at": _timestamp(generated_at),
        "system_imbalance_mw": values.system_imbalance_mw,
        "p10_mw": values.p10_mw,
        "p90_mw": values.p90_mw,
        "flip_probability": values.flip_probability,
        "will_flip": values.will_flip,
        "current_state": current_state,
        "predicted_state": predicted_state,
        "prediction_quality": prediction_quality,
        "model_version": values.model_version,
        "feature_schema_hash": values.feature_schema_hash,
    }
    if degraded_reason is not None:
        payload["degraded_reason"] = degraded_reason
    return EventEnvelope(
        event_id=event_id(
            "predictor",
            "system-imbalance",
            f"{trigger.event_id}:{values.model_version}:{values.feature_schema_hash}",
        ),
        event_type="imbalance.prediction.generated",
        source="predictor",
        dataset="system-imbalance",
        event_time=snapshot.target_time,
        observed_at=trigger.observed_at,
        ingested_at=generated_at,
        correlation_id=trigger.correlation_id,
        causation_id=trigger.event_id,
        quality_status=prediction_quality,
        payload=payload,
    )


def _invalid_reason(event: EventEnvelope) -> DeadLetterReason:
    if event.event_type != "imbalance.feature.snapshot":
        return DeadLetterReason.UNSUPPORTED_EVENT_TYPE
    if event.schema_version != "1":
        return DeadLetterReason.UNSUPPORTED_SCHEMA_VERSION
    return DeadLetterReason.INVALID_PAYLOAD


def _timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


async def _run_service() -> None:
    from imbalance_pipeline.messaging.nats import NatsEventBus

    settings = get_settings()
    registry = FeatureRegistry.default(deadband_mw=settings.flip_deadband_mw)
    runtime: PredictorRuntime | None
    try:
        runtime = OnnxEnsemble(
            settings.model_dir,
            expected_schema_hash=registry.fingerprint,
            deadband_mw=settings.flip_deadband_mw,
        )
    except ValueError:
        if not settings.allow_fallback:
            raise
        runtime = None
    bus = await NatsEventBus.connect(settings)
    try:
        await bus.ensure_grid_stream()
        await PredictorService(
            runtime,
            cast(EventBus, bus),
            allow_fallback=settings.allow_fallback,
            deadband_mw=settings.flip_deadband_mw,
        ).run()
    finally:
        await bus.aclose()


def main() -> None:
    asyncio.run(_run_service())
