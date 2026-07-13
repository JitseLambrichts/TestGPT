import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol
from zoneinfo import ZoneInfo

import holidays
import numpy as np
from numpy.typing import NDArray

from imbalance_pipeline.domain.events import event_id
from imbalance_pipeline.domain.imbalance import ConfirmedState, ImbalanceObservation, advance_state
from imbalance_pipeline.features.schema import DEFAULT_FEATURE_REGISTRY, FeatureRegistry

LOCAL_HISTORY_MINUTES = 180
CONTEXT_STEPS = 96
CONTEXT_RESOLUTION_MINUTES = 15
MIN_OBSERVED_MINUTES_FOR_MODEL = 30
BRUSSELS = ZoneInfo("Europe/Brussels")


class FeatureSource(Protocol):
    async def fetch_imbalance_window(
        self,
        event_cutoff: datetime,
        minutes: int,
        *,
        knowledge_cutoff: datetime,
    ) -> list[ImbalanceObservation]: ...


@dataclass(frozen=True, slots=True)
class FeatureSnapshot:
    event_id: str
    cutoff: datetime
    knowledge_cutoff: datetime
    target_time: datetime
    feature_schema_hash: str
    local_values: NDArray[np.float32]
    local_masks: NDArray[np.uint8]
    context_values: NDArray[np.float32]
    context_masks: NDArray[np.uint8]
    static_values: NDArray[np.float32]
    static_masks: NDArray[np.uint8]
    current_state: ConfirmedState | None
    model_eligible: bool
    observed_imbalance_minutes: int
    created_at: datetime


class FeatureEngine:
    def __init__(
        self,
        source: FeatureSource,
        registry: FeatureRegistry = DEFAULT_FEATURE_REGISTRY,
    ) -> None:
        self._source = source
        self._registry = registry
        self._local_index = {name: index for index, name in enumerate(registry.local_names)}
        self._context_index = {name: index for index, name in enumerate(registry.context_names)}

    async def build(
        self,
        event_cutoff: datetime,
        *,
        knowledge_cutoff: datetime,
    ) -> FeatureSnapshot:
        cutoff = _utc(event_cutoff)
        knowledge = _utc(knowledge_cutoff)
        required_minutes = max(
            self._registry.local_window_minutes,
            self._registry.context_steps * self._registry.context_resolution_minutes,
        )
        observations = await self._source.fetch_imbalance_window(
            cutoff,
            required_minutes,
            knowledge_cutoff=knowledge,
        )
        return self._transform(cutoff, knowledge, observations)

    async def build_many(
        self,
        event_cutoffs: Sequence[datetime],
        *,
        knowledge_cutoffs: Sequence[datetime],
    ) -> list[FeatureSnapshot]:
        if len(event_cutoffs) != len(knowledge_cutoffs):
            raise ValueError("event_cutoffs and knowledge_cutoffs must have equal lengths")
        return [
            await self.build(event_cutoff, knowledge_cutoff=knowledge_cutoff)
            for event_cutoff, knowledge_cutoff in zip(
                event_cutoffs,
                knowledge_cutoffs,
                strict=True,
            )
        ]

    def _transform(
        self,
        cutoff: datetime,
        knowledge_cutoff: datetime,
        observations: Sequence[ImbalanceObservation],
    ) -> FeatureSnapshot:
        history_minutes = self._registry.context_steps * self._registry.context_resolution_minutes
        history_start = cutoff - timedelta(minutes=history_minutes - 1)
        history_times = tuple(
            history_start + timedelta(minutes=index) for index in range(history_minutes)
        )
        by_timestamp = {
            _utc(observation.timestamp): observation
            for observation in sorted(observations, key=lambda row: row.timestamp)
            if history_start <= observation.timestamp <= cutoff
        }
        balance, states, durations, ages = _balance_history(
            history_times,
            by_timestamp,
            self._registry.deadband_mw,
        )
        local_values, local_masks = self._local_window(
            history_times,
            by_timestamp,
            balance,
            states,
            durations,
            ages,
        )
        context_values, context_masks = self._context_window(
            history_times,
            balance,
            cutoff,
        )
        static_values = _static_values(cutoff)
        static_masks = np.ones(static_values.shape, dtype=np.uint8)
        local_balance_index = self._local_index["system_imbalance_mw"]
        observed_minutes = int(local_masks[:, local_balance_index].sum())
        current_state = states[-1]
        event_identity = event_id(
            "feature-engine",
            self._registry.fingerprint,
            ":".join(
                (
                    cutoff.isoformat(),
                    knowledge_cutoff.isoformat(),
                )
            ),
        )
        for array in (
            local_values,
            local_masks,
            context_values,
            context_masks,
            static_values,
            static_masks,
        ):
            array.setflags(write=False)
        return FeatureSnapshot(
            event_id=event_identity,
            cutoff=cutoff,
            knowledge_cutoff=knowledge_cutoff,
            target_time=cutoff + timedelta(minutes=1),
            feature_schema_hash=self._registry.fingerprint,
            local_values=local_values,
            local_masks=local_masks,
            context_values=context_values,
            context_masks=context_masks,
            static_values=static_values,
            static_masks=static_masks,
            current_state=current_state,
            model_eligible=observed_minutes >= MIN_OBSERVED_MINUTES_FOR_MODEL,
            observed_imbalance_minutes=observed_minutes,
            created_at=knowledge_cutoff,
        )

    def _local_window(
        self,
        history_times: Sequence[datetime],
        observations: dict[datetime, ImbalanceObservation],
        balance: NDArray[np.float64],
        states: Sequence[ConfirmedState | None],
        durations: NDArray[np.float64],
        ages: NDArray[np.float64],
    ) -> tuple[NDArray[np.float32], NDArray[np.uint8]]:
        local_length = self._registry.local_window_minutes
        offset = len(history_times) - local_length
        values = np.zeros((local_length, len(self._registry.local_names)), dtype=np.float32)
        masks = np.zeros(values.shape, dtype=np.uint8)
        for output_index, history_index in enumerate(range(offset, len(history_times))):
            observation = observations.get(history_times[history_index])
            self._set_local_raw(values, masks, output_index, observation)
            self._set_local_dynamics(values, masks, output_index, history_index, balance)
            if math.isfinite(balance[history_index]):
                self._set_local(
                    values,
                    masks,
                    output_index,
                    "boundary_distance_mw",
                    abs(abs(balance[history_index]) - self._registry.deadband_mw),
                )
                state = states[history_index]
                if state is not None:
                    sign = 1.0 if state is ConfirmedState.POSITIVE else -1.0
                    self._set_local(values, masks, output_index, "confirmed_state_sign", sign)
                    self._set_local(
                        values,
                        masks,
                        output_index,
                        "state_duration_minutes",
                        durations[history_index],
                    )
            self._set_local(
                values,
                masks,
                output_index,
                "imbalance_age_minutes",
                ages[history_index],
            )
        return values, masks

    def _set_local_raw(
        self,
        values: NDArray[np.float32],
        masks: NDArray[np.uint8],
        row: int,
        observation: ImbalanceObservation | None,
    ) -> None:
        if observation is None:
            return
        self._set_local(values, masks, row, "system_imbalance_mw", observation.system_imbalance_mw)
        self._set_local_optional(values, masks, row, "ace_mw", observation.ace_mw)
        self._set_local_optional(
            values,
            masks,
            row,
            "imbalance_price_eur_mwh",
            observation.imbalance_price_eur_mwh,
        )
        self._set_local_optional(
            values,
            masks,
            row,
            "marginal_incremental_price_eur_mwh",
            observation.marginal_incremental_price_eur_mwh,
        )
        self._set_local_optional(
            values,
            masks,
            row,
            "marginal_decremental_price_eur_mwh",
            observation.marginal_decremental_price_eur_mwh,
        )
        self._set_local(
            values,
            masks,
            row,
            "quality_validated",
            float(observation.quality_status.casefold() == "validated"),
        )

    def _set_local_dynamics(
        self,
        values: NDArray[np.float32],
        masks: NDArray[np.uint8],
        output_index: int,
        history_index: int,
        balance: NDArray[np.float64],
    ) -> None:
        current = balance[history_index]
        if not math.isfinite(current):
            return
        if history_index >= 1 and math.isfinite(balance[history_index - 1]):
            delta_1 = current - balance[history_index - 1]
            self._set_local(values, masks, output_index, "delta_1_mw", delta_1)
            if history_index >= 2 and math.isfinite(balance[history_index - 2]):
                previous_delta = balance[history_index - 1] - balance[history_index - 2]
                self._set_local(
                    values,
                    masks,
                    output_index,
                    "acceleration_mw",
                    delta_1 - previous_delta,
                )
        if history_index >= 2 and math.isfinite(balance[history_index - 2]):
            self._set_local(
                values,
                masks,
                output_index,
                "delta_2_mw",
                current - balance[history_index - 2],
            )
        for window in (5, 15, 60):
            sample = _complete_window(balance, history_index, window)
            if sample is not None:
                self._set_local(
                    values,
                    masks,
                    output_index,
                    f"ewm_{window}_mw",
                    _ewm(sample),
                )
        for window in (5, 15):
            sample = _complete_window(balance, history_index, window)
            if sample is not None:
                self._set_local(
                    values,
                    masks,
                    output_index,
                    f"rolling_median_{window}_mw",
                    float(np.median(sample)),
                )
                self._set_local(
                    values,
                    masks,
                    output_index,
                    f"slope_{window}_mw_per_minute",
                    _slope(sample),
                )
                if window == 15:
                    self._set_local(
                        values,
                        masks,
                        output_index,
                        "rolling_min_15_mw",
                        float(np.min(sample)),
                    )
                    self._set_local(
                        values,
                        masks,
                        output_index,
                        "rolling_max_15_mw",
                        float(np.max(sample)),
                    )
        for window in (15, 60):
            sample = _complete_window(balance, history_index, window)
            if sample is not None:
                median = float(np.median(sample))
                scale = 1.4826 * float(np.median(np.abs(sample - median)))
                self._set_local(
                    values,
                    masks,
                    output_index,
                    f"robust_scale_{window}_mw",
                    scale,
                )

    def _set_local(
        self,
        values: NDArray[np.float32],
        masks: NDArray[np.uint8],
        row: int,
        name: str,
        value: float,
    ) -> None:
        column = self._local_index[name]
        values[row, column] = value
        masks[row, column] = 1

    def _set_local_optional(
        self,
        values: NDArray[np.float32],
        masks: NDArray[np.uint8],
        row: int,
        name: str,
        value: float | None,
    ) -> None:
        if value is not None:
            self._set_local(values, masks, row, name, value)

    def _context_window(
        self,
        history_times: Sequence[datetime],
        balance: NDArray[np.float64],
        cutoff: datetime,
    ) -> tuple[NDArray[np.float32], NDArray[np.uint8]]:
        values = np.zeros(
            (self._registry.context_steps, len(self._registry.context_names)),
            dtype=np.float32,
        )
        masks = np.zeros(values.shape, dtype=np.uint8)
        boundary = _floor_quarter_hour(cutoff)
        first_boundary = boundary - timedelta(
            minutes=(self._registry.context_steps - 1) * self._registry.context_resolution_minutes
        )
        position = {timestamp: index for index, timestamp in enumerate(history_times)}
        for row in range(self._registry.context_steps):
            bin_end = first_boundary + timedelta(
                minutes=row * self._registry.context_resolution_minutes
            )
            bin_start = bin_end - timedelta(minutes=self._registry.context_resolution_minutes - 1)
            sample = [
                balance[position[timestamp]]
                for timestamp in _minute_range(bin_start, bin_end)
                if timestamp in position and math.isfinite(balance[position[timestamp]])
            ]
            if sample:
                array = np.asarray(sample, dtype=np.float64)
                self._set_context(values, masks, row, "imbalance_mean_mw", float(np.mean(array)))
                self._set_context(values, masks, row, "imbalance_min_mw", float(np.min(array)))
                self._set_context(values, masks, row, "imbalance_max_mw", float(np.max(array)))
                self._set_context(values, masks, row, "imbalance_std_mw", float(np.std(array)))
                self._set_context(values, masks, row, "imbalance_age_minutes", 0.0)
            else:
                self._set_context(values, masks, row, "imbalance_age_minutes", float(row * 15))
            for source_name in ("load", "wind", "solar"):
                self._set_context(
                    values,
                    masks,
                    row,
                    f"{source_name}_actual_age_minutes",
                    float(row * self._registry.context_resolution_minutes),
                )
        return values, masks

    def _set_context(
        self,
        values: NDArray[np.float32],
        masks: NDArray[np.uint8],
        row: int,
        name: str,
        value: float,
    ) -> None:
        column = self._context_index[name]
        values[row, column] = value
        masks[row, column] = 1


def _balance_history(
    times: Sequence[datetime],
    observations: dict[datetime, ImbalanceObservation],
    deadband_mw: float,
) -> tuple[
    NDArray[np.float64],
    list[ConfirmedState | None],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    values = np.full(len(times), np.nan, dtype=np.float64)
    states: list[ConfirmedState | None] = []
    durations = np.zeros(len(times), dtype=np.float64)
    ages = np.zeros(len(times), dtype=np.float64)
    previous_state: ConfirmedState | None = None
    duration = 0
    age = 0
    for index, timestamp in enumerate(times):
        observation = observations.get(timestamp)
        if observation is None:
            age += 1
            states.append(previous_state)
            durations[index] = duration
            ages[index] = age
            continue
        value = observation.system_imbalance_mw
        values[index] = value
        age = 0
        next_state = advance_state(previous_state, value, deadband_mw)
        duration = duration + 1 if next_state is previous_state else 1
        previous_state = next_state
        states.append(next_state)
        durations[index] = duration
        ages[index] = age
    return values, states, durations, ages


def _complete_window(
    values: NDArray[np.float64],
    end_index: int,
    window: int,
) -> NDArray[np.float64] | None:
    start = end_index - window + 1
    if start < 0:
        return None
    sample = values[start : end_index + 1]
    if not np.isfinite(sample).all():
        return None
    return sample


def _ewm(sample: NDArray[np.float64]) -> float:
    alpha = 2.0 / (len(sample) + 1.0)
    value = float(sample[0])
    for item in sample[1:]:
        value = alpha * float(item) + (1.0 - alpha) * value
    return value


def _slope(sample: NDArray[np.float64]) -> float:
    x = np.arange(len(sample), dtype=np.float64)
    centered = x - x.mean()
    denominator = float(np.dot(centered, centered))
    if denominator == 0:
        return 0.0
    return float(np.dot(centered, sample - sample.mean()) / denominator)


def _static_values(cutoff: datetime) -> NDArray[np.float32]:
    local = cutoff.astimezone(BRUSSELS)
    minute_of_quarter = local.minute % 15
    minute_of_hour = local.minute
    hour = local.hour + local.minute / 60.0
    weekday = local.weekday()
    day_of_year = local.timetuple().tm_yday - 1
    holiday_calendar = holidays.country_holidays("BE", years=[local.year])
    dst_before = (local - timedelta(hours=1)).utcoffset()
    dst_after = (local + timedelta(hours=1)).utcoffset()
    return np.asarray(
        (
            math.sin(2.0 * math.pi * minute_of_quarter / 15.0),
            math.cos(2.0 * math.pi * minute_of_quarter / 15.0),
            math.sin(2.0 * math.pi * minute_of_hour / 60.0),
            math.cos(2.0 * math.pi * minute_of_hour / 60.0),
            math.sin(2.0 * math.pi * hour / 24.0),
            math.cos(2.0 * math.pi * hour / 24.0),
            math.sin(2.0 * math.pi * weekday / 7.0),
            math.cos(2.0 * math.pi * weekday / 7.0),
            math.sin(2.0 * math.pi * day_of_year / 366.0),
            math.cos(2.0 * math.pi * day_of_year / 366.0),
            float(weekday >= 5),
            float(local.date() in holiday_calendar),
            float(dst_before != dst_after),
        ),
        dtype=np.float32,
    )


def _floor_quarter_hour(value: datetime) -> datetime:
    return value.replace(minute=(value.minute // 15) * 15, second=0, microsecond=0)


def _minute_range(start: datetime, end: datetime) -> tuple[datetime, ...]:
    count = int((end - start).total_seconds() // 60) + 1
    return tuple(start + timedelta(minutes=offset) for offset in range(count))


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("feature timestamps must be timezone-aware")
    return value.astimezone(UTC)
