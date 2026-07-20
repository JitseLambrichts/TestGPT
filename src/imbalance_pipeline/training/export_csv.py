"""Build a leakage-bounded training export directly from an Elia ODS133 CSV."""

import argparse
import csv
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime, timedelta
from itertools import islice
from pathlib import Path
from typing import cast

from imbalance_pipeline.domain.imbalance import (
    ImbalanceObservation,
    VersionedImbalanceObservation,
)
from imbalance_pipeline.features.engine import FeatureEngine, FeatureReplaySession, FeatureSource
from imbalance_pipeline.features.schema import FeatureRegistry
from imbalance_pipeline.sources.elia import normalize_imbalance
from imbalance_pipeline.training.data import TrainingExample, build_training_examples
from imbalance_pipeline.training.export_data import write_training_dataset

_ODS133_FIELDS = {
    "Datetime",
    "Resolution code",
    "Quarter hour",
    "Quality status",
    "ACE",
    "System imbalance",
    "Alpha",
    "Alpha'",
    "Marginal incremental price",
    "Marginal decremental price",
    "Imbalance Price",
}
_HISTORY_MINUTES = 24 * 60


def read_ods133_csv(path: Path) -> list[ImbalanceObservation]:
    """Read the portal's semicolon CSV and return unique chronological UTC rows."""
    source = Path(path)
    observations: dict[datetime, ImbalanceObservation] = {}
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=";")
        fields = set(reader.fieldnames or ())
        missing = sorted(_ODS133_FIELDS - fields)
        if missing:
            raise ValueError(f"ODS133 CSV is missing required columns: {', '.join(missing)}")
        for line_number, row in enumerate(reader, start=2):
            observation = _ods133_observation(row, line_number)
            if observation.timestamp in observations:
                raise ValueError(
                    f"duplicate ODS133 minute at {observation.timestamp.isoformat()}"
                )
            observations[observation.timestamp] = observation
    if not observations:
        raise ValueError("ODS133 CSV contains no observations")
    return [observations[timestamp] for timestamp in sorted(observations)]


def select_training_cutoffs(
    timestamps: Sequence[datetime],
    *,
    history_minutes: int = _HISTORY_MINUTES,
    stride_minutes: int = 30,
    start: datetime | None = None,
    end: datetime | None = None,
    max_examples: int | None = None,
) -> list[datetime]:
    """Select chronological cutoffs with contiguous history and a real t+1 label."""
    if history_minutes <= 0 or stride_minutes <= 0:
        raise ValueError("history_minutes and stride_minutes must be positive")
    if max_examples is not None and max_examples <= 0:
        raise ValueError("max_examples must be positive")
    ordered = sorted({_utc(timestamp) for timestamp in timestamps})
    if not ordered:
        return []
    lower = _utc(start) if start is not None else None
    upper = _utc(end) if end is not None else None
    if lower is not None and upper is not None and upper <= lower:
        raise ValueError("end must be after start")

    origin = ordered[0]
    consecutive = 0
    selected: list[datetime] = []
    for index, timestamp in enumerate(ordered):
        if index and timestamp == ordered[index - 1] + timedelta(minutes=1):
            consecutive += 1
        else:
            consecutive = 1
        has_target = (
            index + 1 < len(ordered)
            and ordered[index + 1] == timestamp + timedelta(minutes=1)
        )
        on_stride = int((timestamp - origin).total_seconds() // 60) % stride_minutes == 0
        in_range = (lower is None or timestamp >= lower) and (upper is None or timestamp < upper)
        if consecutive >= history_minutes and has_target and on_stride and in_range:
            selected.append(timestamp)
    if max_examples is not None and len(selected) > max_examples:
        selected = _evenly_spaced(selected, max_examples)
    return selected


async def export_ods133_training_dataset(
    csv_path: Path,
    output: Path,
    *,
    stride_minutes: int = 30,
    batch_size: int = 256,
    deadband_mw: float = 10.0,
    start: datetime | None = None,
    end: datetime | None = None,
    max_examples: int | None = None,
) -> Path:
    """Convert ODS133 into the immutable NPZ-shard format consumed by the trainer."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    observations = read_ods133_csv(csv_path)
    cutoffs = select_training_cutoffs(
        [observation.timestamp for observation in observations],
        stride_minutes=stride_minutes,
        start=start,
        end=end,
        max_examples=max_examples,
    )
    if not cutoffs:
        raise ValueError("ODS133 CSV has no cutoffs with complete history and a t+1 target")

    # Keep the exact online feature contract so the resulting bundle can be
    # loaded by the predictor; the raw-source provenance is captured by the
    # export command and dataset checksum rather than by changing the schema.
    registry = FeatureRegistry.default(deadband_mw=deadband_mw)
    versions = tuple(
        VersionedImbalanceObservation(
            observation=observation,
            # ODS133 records represent the minute at which the estimate was calculated.
            # Treat that minute as the earliest usable cutoff and never expose t+1.
            available_at=observation.timestamp,
            event_id=f"ods133-csv:{observation.timestamp.isoformat()}",
        )
        for observation in observations
    )
    values = {
        observation.timestamp: observation.system_imbalance_mw
        for observation in observations
    }
    engine = FeatureEngine(cast(FeatureSource, _UnusedFeatureSource()), registry)
    replay = FeatureReplaySession(
        engine,
        versions,
        end=observations[-1].timestamp,
        knowledge_cutoff=observations[-1].timestamp,
    )

    def examples() -> Iterator[TrainingExample]:
        iterator = iter(cutoffs)
        while batch := list(islice(iterator, batch_size)):
            snapshots = replay.build_many(batch, knowledge_cutoffs=batch)
            yield from build_training_examples(
                snapshots,
                values,
                deadband_mw=deadband_mw,
            )

    return write_training_dataset(examples(), Path(output))


class _UnusedFeatureSource:
    """Replay construction is synchronous; online source methods are intentionally unused."""


def _ods133_observation(row: dict[str, str], line_number: int) -> ImbalanceObservation:
    try:
        return normalize_imbalance(
            {
                "datetime": row["Datetime"],
                "resolutioncode": row["Resolution code"],
                "quarterhour": row["Quarter hour"],
                "qualitystatus": row["Quality status"],
                "ace": _optional_float(row["ACE"]),
                "systemimbalance": float(row["System imbalance"]),
                "alpha": _optional_float(row["Alpha"]),
                "alpha_prime": _optional_float(row["Alpha'"]),
                "marginalincrementalprice": _optional_float(
                    row["Marginal incremental price"]
                ),
                "marginaldecrementalprice": _optional_float(
                    row["Marginal decremental price"]
                ),
                "imbalanceprice": _optional_float(row["Imbalance Price"]),
            }
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid ODS133 CSV row at line {line_number}") from exc


def _optional_float(value: str) -> float | None:
    stripped = value.strip()
    return float(stripped) if stripped else None


def _evenly_spaced(values: Sequence[datetime], count: int) -> list[datetime]:
    if count == 1:
        return [values[-1]]
    last = len(values) - 1
    indices = [round(index * last / (count - 1)) for index in range(count)]
    return [values[index] for index in indices]


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("training cutoffs must be timezone-aware")
    return value.astimezone(UTC)


def _parse_iso(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid ISO timestamp: {value}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("timestamp must be timezone-aware")
    return parsed.astimezone(UTC)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a causal flip-training dataset from an Elia ODS133 CSV."
    )
    parser.add_argument("csv", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--stride-minutes", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--deadband-mw", type=float, default=10.0)
    parser.add_argument("--start", type=_parse_iso)
    parser.add_argument("--end", type=_parse_iso)
    parser.add_argument("--max-examples", type=int)
    args = parser.parse_args()

    import asyncio

    asyncio.run(
        export_ods133_training_dataset(
            args.csv,
            args.output,
            stride_minutes=args.stride_minutes,
            batch_size=args.batch_size,
            deadband_mw=args.deadband_mw,
            start=args.start,
            end=args.end,
            max_examples=args.max_examples,
        )
    )


__all__ = [
    "export_ods133_training_dataset",
    "main",
    "read_ods133_csv",
    "select_training_cutoffs",
]
