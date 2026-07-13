from datetime import UTC, datetime, timedelta

from imbalance_pipeline.domain.imbalance import (
    ConfirmedState,
    ConfirmedStateSeed,
    ImbalanceObservation,
    VersionedImbalanceObservation,
    advance_state,
)

VersionedObservation = VersionedImbalanceObservation


def observation(
    timestamp: datetime,
    value: float,
    *,
    ace_mw: float | None = 0.0,
    quality_status: str = "Validated",
) -> ImbalanceObservation:
    timestamp = timestamp.astimezone(UTC).replace(second=0, microsecond=0)
    quarter_hour = timestamp.replace(minute=(timestamp.minute // 15) * 15)
    return ImbalanceObservation(
        timestamp=timestamp,
        quarter_hour=quarter_hour,
        resolution_code="PT1M",
        quality_status=quality_status,
        ace_mw=ace_mw,
        system_imbalance_mw=value,
        alpha_eur_mwh=None,
        alpha_prime_eur_mwh=None,
        marginal_incremental_price_eur_mwh=120.0,
        marginal_decremental_price_eur_mwh=95.0,
        imbalance_price_eur_mwh=95.0,
    )


class MemoryFeatureSource:
    def __init__(
        self,
        rows: list[VersionedObservation],
        *,
        state_seed: ConfirmedStateSeed | None = None,
    ) -> None:
        self.rows = rows
        self.state_seed = state_seed
        self.calls: list[tuple[datetime, int, datetime]] = []
        self.seed_calls: list[tuple[datetime, datetime]] = []
        self.version_calls: list[tuple[datetime, datetime, datetime]] = []

    async def fetch_imbalance_window(
        self,
        event_cutoff: datetime,
        minutes: int,
        *,
        knowledge_cutoff: datetime,
    ) -> list[ImbalanceObservation]:
        self.calls.append((event_cutoff, minutes, knowledge_cutoff))
        start = event_cutoff - timedelta(minutes=minutes)
        return [
            row.observation
            for row in self.rows
            if start < row.observation.timestamp <= event_cutoff
            and row.available_at <= knowledge_cutoff
        ]

    async def fetch_imbalance_state_seed(
        self,
        before: datetime,
        *,
        knowledge_cutoff: datetime,
        deadband_mw: float,
    ) -> ConfirmedStateSeed:
        self.seed_calls.append((before, knowledge_cutoff))
        if self.state_seed is not None:
            return self.state_seed
        state: ConfirmedState | None = None
        state_since: datetime | None = None
        last_observed_at: datetime | None = None
        for row in sorted(self.rows, key=lambda item: item.observation.timestamp):
            if row.observation.timestamp > before or row.available_at > knowledge_cutoff:
                continue
            next_state = advance_state(state, row.observation.system_imbalance_mw, deadband_mw)
            if next_state is not state:
                state_since = row.observation.timestamp
            state = next_state
            last_observed_at = row.observation.timestamp
        return ConfirmedStateSeed(state, state_since, last_observed_at)

    async def fetch_imbalance_versions(
        self,
        start: datetime,
        end: datetime,
        *,
        knowledge_cutoff: datetime,
    ) -> list[VersionedObservation]:
        self.version_calls.append((start, end, knowledge_cutoff))
        return [
            row
            for row in self.rows
            if start <= row.observation.timestamp <= end
            and row.available_at <= knowledge_cutoff
        ]

    def add(self, row: VersionedObservation) -> None:
        self.rows.append(row)


def minute_rows(
    cutoff: datetime,
    count: int,
    *,
    value: float = 25.0,
    available_delay: timedelta = timedelta(seconds=5),
) -> list[VersionedObservation]:
    start = cutoff - timedelta(minutes=count - 1)
    return [
        VersionedObservation(
            observation(timestamp, value + index),
            timestamp + available_delay,
        )
        for index, timestamp in enumerate(
            start + timedelta(minutes=offset) for offset in range(count)
        )
    ]
