from datetime import UTC, datetime, timedelta

from imbalance_pipeline.domain.imbalance import (
    ConfirmedStateSeed,
    ImbalanceObservation,
    VersionedImbalanceObservation,
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
        self.state_seed = state_seed or ConfirmedStateSeed(None, None, None)
        self.calls: list[tuple[datetime, int, datetime]] = []
        self.seed_calls: list[tuple[datetime, datetime]] = []
        self.seed_batch_calls: list[tuple[tuple[datetime, datetime], ...]] = []
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
        del deadband_mw
        self.seed_calls.append((before, knowledge_cutoff))
        return self.state_seed

    async def fetch_imbalance_state_seeds(
        self,
        requests: tuple[tuple[datetime, datetime], ...],
        *,
        deadband_mw: float,
    ) -> list[ConfirmedStateSeed]:
        del deadband_mw
        self.seed_batch_calls.append(requests)
        return [self.state_seed for _ in requests]

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
