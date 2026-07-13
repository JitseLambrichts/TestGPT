import math
from bisect import bisect_left
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol
from zoneinfo import ZoneInfo

import holidays
import numpy as np
from numpy.typing import NDArray

from imbalance_pipeline.domain.events import event_id
from imbalance_pipeline.domain.imbalance import (
    ConfirmedState,
    ConfirmedStateSeed,
    ImbalanceObservation,
    VersionedImbalanceObservation,
    advance_state,
)
from imbalance_pipeline.features.schema import DEFAULT_FEATURE_REGISTRY, FeatureRegistry

LOCAL_HISTORY_MINUTES = 180
CONTEXT_STEPS = 96
CONTEXT_RESOLUTION_MINUTES = 15
MIN_OBSERVED_MINUTES_FOR_MODEL = 30
REPLAY_START = datetime(1970, 1, 1, tzinfo=UTC)
BRUSSELS = ZoneInfo("Europe/Brussels")


class FeatureSource(Protocol):
    async def fetch_imbalance_window(
        self,
        event_cutoff: datetime,
        minutes: int,
        *,
        knowledge_cutoff: datetime,
    ) -> list[ImbalanceObservation]: ...

    async def fetch_imbalance_state_seed(
        self,
        before: datetime,
        *,
        knowledge_cutoff: datetime,
        deadband_mw: float,
    ) -> ConfirmedStateSeed: ...

    async def fetch_imbalance_versions(
        self,
        start: datetime,
        end: datetime,
        *,
        knowledge_cutoff: datetime,
    ) -> list[VersionedImbalanceObservation]: ...


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


@dataclass(frozen=True, slots=True)
class _StateTransformer:
    outputs: tuple[ConfirmedState | None, ...]
    last_changes: tuple[datetime | None, ...]


_STATE_OPTIONS: tuple[ConfirmedState | None, ...] = (
    None,
    ConfirmedState.POSITIVE,
    ConfirmedState.NEGATIVE,
)
_STATE_INDICES = {state: index for index, state in enumerate(_STATE_OPTIONS)}
_IDENTITY_TRANSFORMER = _StateTransformer(_STATE_OPTIONS, (None, None, None))


class _StateReplay:
    def __init__(self, timestamps: Sequence[datetime], deadband_mw: float) -> None:
        self._timestamps = tuple(timestamps)
        self._positions = {timestamp: index for index, timestamp in enumerate(self._timestamps)}
        self._deadband_mw = deadband_mw
        self._size = 1
        while self._size < len(self._timestamps):
            self._size *= 2
        self._tree = [_IDENTITY_TRANSFORMER] * (2 * self._size)

    def update(self, timestamp: datetime, value_mw: float) -> None:
        position = self._positions[timestamp] + self._size
        self._tree[position] = _state_transformer(timestamp, value_mw, self._deadband_mw)
        position //= 2
        while position:
            self._tree[position] = _compose_state_transformers(
                self._tree[position * 2],
                self._tree[position * 2 + 1],
            )
            position //= 2

    def seed_before(
        self,
        before: datetime,
        observations: Mapping[datetime, ImbalanceObservation],
    ) -> ConfirmedStateSeed:
        transformer = self._prefix(bisect_left(self._timestamps, before))
        state = transformer.outputs[0]
        state_since = transformer.last_changes[0] if state is not None else None
        index = bisect_left(self._timestamps, before) - 1
        while index >= 0:
            timestamp = self._timestamps[index]
            if timestamp in observations:
                return ConfirmedStateSeed(state, state_since, timestamp)
            index -= 1
        return ConfirmedStateSeed(state, state_since, None)

    def _prefix(self, end: int) -> _StateTransformer:
        left = _IDENTITY_TRANSFORMER
        right = _IDENTITY_TRANSFORMER
        start = self._size
        stop = self._size + end
        while start < stop:
            if start % 2:
                left = _compose_state_transformers(left, self._tree[start])
                start += 1
            if stop % 2:
                stop -= 1
                right = _compose_state_transformers(self._tree[stop], right)
            start //= 2
            stop //= 2
        return _compose_state_transformers(left, right)


class FeatureReplaySession:
    """Reuses one point-in-time version stream across chronological training batches."""

    def __init__(
        self,
        engine: "FeatureEngine",
        versions: Sequence[VersionedImbalanceObservation],
        *,
        end: datetime,
        knowledge_cutoff: datetime,
    ) -> None:
        self._engine = engine
        self._end = end
        self._knowledge_cutoff = knowledge_cutoff
        timestamps = tuple(sorted({_utc(version.observation.timestamp) for version in versions}))
        self._state_replay = _StateReplay(timestamps, engine._registry.deadband_mw)
        self._versions = tuple(
            sorted(
                versions,
                key=lambda item: (
                    item.available_at,
                    item.row_version,
                    item.event_id,
                    item.observation.timestamp,
                ),
            )
        )
        self._best_versions: dict[datetime, tuple[int, datetime, str]] = {}
        self._observations: dict[datetime, ImbalanceObservation] = {}
        self._version_index = 0
        self._last_knowledge_cutoff: datetime | None = None

    def build_many(
        self,
        event_cutoffs: Sequence[datetime],
        *,
        knowledge_cutoffs: Sequence[datetime],
    ) -> list[FeatureSnapshot]:
        if len(event_cutoffs) != len(knowledge_cutoffs):
            raise ValueError("event_cutoffs and knowledge_cutoffs must have equal lengths")
        cutoffs = tuple(_utc(cutoff) for cutoff in event_cutoffs)
        knowledge = tuple(_utc(cutoff) for cutoff in knowledge_cutoffs)
        if any(cutoff > self._end for cutoff in cutoffs):
            raise ValueError("replay session does not include the requested event cutoff")
        if any(cutoff > self._knowledge_cutoff for cutoff in knowledge):
            raise ValueError("replay session does not include the requested knowledge cutoff")
        if any(later < earlier for earlier, later in zip(knowledge, knowledge[1:], strict=False)):
            raise ValueError("replay session batches must have non-decreasing knowledge cutoffs")
        if self._last_knowledge_cutoff is not None and knowledge:
            if knowledge[0] < self._last_knowledge_cutoff:
                raise ValueError("replay session cannot move backwards in knowledge time")

        snapshots: list[FeatureSnapshot] = []
        for cutoff, known_at in zip(cutoffs, knowledge, strict=True):
            self._advance_to(known_at)
            history_times = _history_times(cutoff, self._engine._registry)
            state_seed = self._state_replay.seed_before(history_times[0], self._observations)
            history_observations = [
                self._observations[timestamp]
                for timestamp in history_times
                if timestamp in self._observations
            ]
            snapshots.append(
                self._engine._transform(
                    cutoff,
                    known_at,
                    history_observations,
                    state_seed,
                )
            )
        if knowledge:
            self._last_knowledge_cutoff = knowledge[-1]
        return snapshots

    def _advance_to(self, known_at: datetime) -> None:
        while (
            self._version_index < len(self._versions)
            and self._versions[self._version_index].available_at <= known_at
        ):
            version = self._versions[self._version_index]
            timestamp = _utc(version.observation.timestamp)
            rank = (version.row_version, version.available_at, version.event_id)
            if rank >= self._best_versions.get(timestamp, (-1, REPLAY_START, "")):
                self._best_versions[timestamp] = rank
                self._observations[timestamp] = version.observation
                self._state_replay.update(timestamp, version.observation.system_imbalance_mw)
            self._version_index += 1


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
            len(_history_times(cutoff, self._registry)),
        )
        observations = await self._source.fetch_imbalance_window(
            cutoff,
            required_minutes,
            knowledge_cutoff=knowledge,
        )
        seed = await self._source.fetch_imbalance_state_seed(
            _history_times(cutoff, self._registry)[0] - timedelta(minutes=1),
            knowledge_cutoff=knowledge,
            deadband_mw=self._registry.deadband_mw,
        )
        return self._transform(cutoff, knowledge, observations, seed)

    async def build_many(
        self,
        event_cutoffs: Sequence[datetime],
        *,
        knowledge_cutoffs: Sequence[datetime],
    ) -> list[FeatureSnapshot]:
        if len(event_cutoffs) != len(knowledge_cutoffs):
            raise ValueError("event_cutoffs and knowledge_cutoffs must have equal lengths")
        if not event_cutoffs:
            return []
        cutoffs = tuple(_utc(cutoff) for cutoff in event_cutoffs)
        knowledge = tuple(_utc(cutoff) for cutoff in knowledge_cutoffs)
        replay = await self.open_replay(
            end=max(cutoffs),
            knowledge_cutoff=max(knowledge),
        )
        requests = sorted(
            enumerate(zip(cutoffs, knowledge, strict=True)),
            key=lambda item: (item[1][1], item[1][0], item[0]),
        )
        ordered_snapshots = replay.build_many(
            [cutoff for _, (cutoff, _) in requests],
            knowledge_cutoffs=[known_at for _, (_, known_at) in requests],
        )
        snapshots: list[FeatureSnapshot | None] = [None] * len(cutoffs)
        for (original_index, _), snapshot in zip(requests, ordered_snapshots, strict=True):
            snapshots[original_index] = snapshot
        if any(snapshot is None for snapshot in snapshots):
            raise RuntimeError("feature replay did not return every requested snapshot")
        return [snapshot for snapshot in snapshots if snapshot is not None]

    async def open_replay(
        self,
        *,
        end: datetime,
        knowledge_cutoff: datetime,
    ) -> FeatureReplaySession:
        replay_end = _utc(end)
        replay_knowledge_cutoff = _utc(knowledge_cutoff)
        versions = await self._source.fetch_imbalance_versions(
            REPLAY_START,
            replay_end,
            knowledge_cutoff=replay_knowledge_cutoff,
        )
        return FeatureReplaySession(
            self,
            versions,
            end=replay_end,
            knowledge_cutoff=replay_knowledge_cutoff,
        )

    def _transform(
        self,
        cutoff: datetime,
        knowledge_cutoff: datetime,
        observations: Sequence[ImbalanceObservation],
        state_seed: ConfirmedStateSeed,
    ) -> FeatureSnapshot:
        history_times = _history_times(cutoff, self._registry)
        history_start = history_times[0]
        by_timestamp = {
            _utc(observation.timestamp): observation
            for observation in sorted(observations, key=lambda row: row.timestamp)
            if history_start <= observation.timestamp <= cutoff
        }
        balance, states, durations, ages = _balance_history(
            history_times,
            by_timestamp,
            self._registry.deadband_mw,
            state_seed,
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
            state_seed,
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
            model_eligible=bool(
                observed_minutes >= MIN_OBSERVED_MINUTES_FOR_MODEL
                and current_state is not None
                and math.isfinite(ages[-1])
                and ages[-1] <= _imbalance_max_age(self._registry)
            ),
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
            if math.isfinite(ages[history_index]):
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
        for window in self._registry.ewm_windows:
            sample = _complete_window(balance, history_index, window)
            if sample is not None:
                self._set_local(
                    values,
                    masks,
                    output_index,
                    f"ewm_{window}_mw",
                    _ewm(sample),
                )
        for window in self._registry.median_windows:
            sample = _complete_window(balance, history_index, window)
            if sample is not None:
                self._set_local(
                    values,
                    masks,
                    output_index,
                    f"rolling_median_{window}_mw",
                    float(np.median(sample)),
                )
                if window == max(self._registry.median_windows):
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
        for window in self._registry.slope_windows:
            sample = _complete_window(balance, history_index, window)
            if sample is not None:
                self._set_local(
                    values,
                    masks,
                    output_index,
                    f"slope_{window}_mw_per_minute",
                    _slope(sample),
                )
        for window in self._registry.robust_scale_windows:
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
        state_seed: ConfirmedStateSeed,
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
            last_observed_at = _last_observed_at(
                history_times,
                balance,
                bin_end,
                state_seed.last_observed_at,
            )
            if last_observed_at is not None:
                self._set_context(
                    values,
                    masks,
                    row,
                    "imbalance_age_minutes",
                    _minutes_between(last_observed_at, bin_end),
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
    state_seed: ConfirmedStateSeed,
) -> tuple[
    NDArray[np.float64],
    list[ConfirmedState | None],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    values = np.full(len(times), np.nan, dtype=np.float64)
    states: list[ConfirmedState | None] = []
    durations = np.zeros(len(times), dtype=np.float64)
    ages = np.full(len(times), np.nan, dtype=np.float64)
    previous_state = state_seed.state
    state_since = state_seed.state_since
    last_observed_at = state_seed.last_observed_at
    for index, timestamp in enumerate(times):
        observation = observations.get(timestamp)
        if observation is None:
            states.append(previous_state)
            if previous_state is not None and state_since is not None:
                durations[index] = _minutes_between(state_since, timestamp)
            if last_observed_at is not None:
                ages[index] = _minutes_between(last_observed_at, timestamp)
            continue
        value = observation.system_imbalance_mw
        values[index] = value
        next_state = advance_state(previous_state, value, deadband_mw)
        if next_state is not previous_state:
            state_since = timestamp
        previous_state = next_state
        last_observed_at = timestamp
        states.append(next_state)
        if next_state is not None and state_since is not None:
            durations[index] = _minutes_between(state_since, timestamp)
        ages[index] = 0.0
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
    dst_before = (cutoff - timedelta(hours=1)).astimezone(BRUSSELS).utcoffset()
    dst_after = (cutoff + timedelta(hours=1)).astimezone(BRUSSELS).utcoffset()
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


def _history_times(cutoff: datetime, registry: FeatureRegistry) -> tuple[datetime, ...]:
    context_end = _floor_quarter_hour(cutoff)
    first_context_end = context_end - timedelta(
        minutes=(registry.context_steps - 1) * registry.context_resolution_minutes
    )
    history_start = first_context_end - timedelta(minutes=registry.context_resolution_minutes - 1)
    return _minute_range(history_start, cutoff)


def _minute_range(start: datetime, end: datetime) -> tuple[datetime, ...]:
    count = int((end - start).total_seconds() // 60) + 1
    return tuple(start + timedelta(minutes=offset) for offset in range(count))


def _state_transformer(
    timestamp: datetime,
    value_mw: float,
    deadband_mw: float,
) -> _StateTransformer:
    outputs: list[ConfirmedState | None] = []
    changes: list[datetime | None] = []
    for previous in _STATE_OPTIONS:
        current = advance_state(previous, value_mw, deadband_mw)
        outputs.append(current)
        changes.append(timestamp if current is not previous else None)
    return _StateTransformer(tuple(outputs), tuple(changes))


def _compose_state_transformers(
    left: _StateTransformer,
    right: _StateTransformer,
) -> _StateTransformer:
    outputs: list[ConfirmedState | None] = []
    changes: list[datetime | None] = []
    for index, middle in enumerate(left.outputs):
        right_index = _STATE_INDICES[middle]
        outputs.append(right.outputs[right_index])
        changes.append(right.last_changes[right_index] or left.last_changes[index])
    return _StateTransformer(tuple(outputs), tuple(changes))


def _last_observed_at(
    history_times: Sequence[datetime],
    balance: NDArray[np.float64],
    at: datetime,
    seed_last_observed_at: datetime | None,
) -> datetime | None:
    observed = [
        timestamp
        for timestamp, value in zip(history_times, balance, strict=True)
        if timestamp <= at and math.isfinite(value)
    ]
    if observed:
        return observed[-1]
    return seed_last_observed_at


def _minutes_between(start: datetime, end: datetime) -> float:
    return (end - start).total_seconds() / 60.0


def _imbalance_max_age(registry: FeatureRegistry) -> int:
    definition = next(
        feature
        for feature in registry.features
        if feature.group == "local" and feature.name == "imbalance_age_minutes"
    )
    if definition.max_source_age_minutes is None:
        raise RuntimeError("imbalance_age_minutes requires a maximum source age")
    return definition.max_source_age_minutes


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("feature timestamps must be timezone-aware")
    return value.astimezone(UTC)
