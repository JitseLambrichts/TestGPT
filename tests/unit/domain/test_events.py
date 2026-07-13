import hashlib
import json
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal

import pytest
from pydantic import ValidationError

from imbalance_pipeline.domain.events import EventEnvelope, Subject, event_id
from imbalance_pipeline.domain.imbalance import ImbalanceObservation


def observation() -> ImbalanceObservation:
    return ImbalanceObservation(
        timestamp=datetime(2026, 7, 13, 10, 1, tzinfo=UTC),
        quarter_hour=datetime(2026, 7, 13, 10, 0, tzinfo=UTC),
        resolution_code="PT1M",
        quality_status="Validated",
        ace_mw=Decimal("-49.05"),
        system_imbalance_mw=Decimal("133.96"),
        alpha_eur_mwh=Decimal("0.0"),
        alpha_prime_eur_mwh=Decimal("0.0"),
        marginal_incremental_price_eur_mwh=Decimal("120.13"),
        marginal_decremental_price_eur_mwh=Decimal("99.97"),
        imbalance_price_eur_mwh=Decimal("99.97"),
    )


def envelope() -> EventEnvelope:
    return EventEnvelope.create(
        event_type="elia.imbalance.observed",
        source="elia",
        dataset="ods161",
        event_time=datetime(2026, 7, 13, 10, 1, tzinfo=UTC),
        natural_key="2026-07-13T10:01:00Z",
        payload=observation(),
        quality_status="Validated",
    )


def test_event_id_is_a_stable_sha256_of_the_versioned_natural_identity() -> None:
    identity = ["elia", "ods161", "2026-07-13T10:01:00Z", "1"]
    canonical_identity = json.dumps(identity, ensure_ascii=False, separators=(",", ":"))
    expected = hashlib.sha256(canonical_identity.encode("utf-8")).hexdigest()

    assert event_id(*identity[:3]) == expected
    assert event_id(*identity[:3]) == event_id(*identity[:3])
    assert event_id(*identity[:3], version="2") != expected


def test_subject_constants_match_the_versioned_grid_contract() -> None:
    assert tuple(subject.value for subject in Subject) == (
        "grid.raw.elia.imbalance.v1",
        "grid.raw.elia.load.v1",
        "grid.raw.elia.wind.v1",
        "grid.raw.elia.solar.v1",
        "grid.raw.weather.forecast.v1",
        "grid.stored.elia.imbalance.v1",
        "grid.features.imbalance.v1",
        "grid.predictions.imbalance.v1",
        "grid.outcomes.imbalance.v1",
    )

    with pytest.raises(ValueError):
        Subject("grid.raw.elia.imbalance.v2")


def test_envelope_create_preserves_transport_neutral_contract_fields() -> None:
    payload = observation()
    created = EventEnvelope.create(
        event_type="elia.imbalance.observed",
        source="elia",
        dataset="ods161",
        event_time=payload.timestamp,
        natural_key="2026-07-13T10:01:00Z",
        payload=payload,
        quality_status=payload.quality_status,
    )

    assert created.event_id == event_id("elia", "ods161", "2026-07-13T10:01:00Z")
    assert created.schema_version == "1"
    assert created.correlation_id == created.event_id
    assert created.causation_id is None
    assert created.observed_at is None
    assert created.payload == payload.model_dump(mode="json")
    assert isinstance(created.payload, dict)
    assert created.ingested_at.utcoffset() == timedelta(0)


def test_envelope_json_round_trip_is_lossless() -> None:
    created = envelope()

    restored = EventEnvelope.model_validate_json(created.model_dump_json())

    assert restored == created
    assert ImbalanceObservation.model_validate(restored.payload) == observation()


@pytest.mark.parametrize("field", ["event_time", "observed_at", "ingested_at"])
@pytest.mark.parametrize(
    "invalid_datetime",
    [
        datetime(2026, 7, 13, 10, 1),
        datetime(2026, 7, 13, 10, 1, tzinfo=timezone(timedelta(hours=2))),
    ],
)
def test_envelope_rejects_non_utc_datetimes(
    field: str,
    invalid_datetime: datetime,
) -> None:
    values = envelope().model_dump()
    values[field] = invalid_datetime

    with pytest.raises(ValidationError, match="UTC"):
        EventEnvelope.model_validate(values)


def test_imbalance_observation_uses_utc_datetimes_and_float_source_values() -> None:
    created = observation()

    assert created.resolution_code == "PT1M"
    assert created.timestamp.tzinfo is UTC
    assert created.quarter_hour.tzinfo is UTC
    assert isinstance(created.ace_mw, float)
    assert isinstance(created.system_imbalance_mw, float)
    assert isinstance(created.imbalance_price_eur_mwh, float)


@pytest.mark.parametrize("field", ["timestamp", "quarter_hour"])
@pytest.mark.parametrize(
    "invalid_datetime",
    [
        datetime(2026, 7, 13, 10, 1),
        datetime(2026, 7, 13, 10, 1, tzinfo=timezone(timedelta(hours=2))),
    ],
)
def test_imbalance_observation_rejects_non_utc_datetimes(
    field: str,
    invalid_datetime: datetime,
) -> None:
    values = observation().model_dump()
    values[field] = invalid_datetime

    with pytest.raises(ValidationError, match="UTC"):
        ImbalanceObservation.model_validate(values)


def test_imbalance_observation_preserves_nullable_source_values() -> None:
    values = observation().model_dump()
    for field in (
        "ace_mw",
        "alpha_eur_mwh",
        "alpha_prime_eur_mwh",
        "marginal_incremental_price_eur_mwh",
        "marginal_decremental_price_eur_mwh",
        "imbalance_price_eur_mwh",
    ):
        values[field] = None

    created = ImbalanceObservation.model_validate(values)

    assert created.ace_mw is None
    assert created.alpha_eur_mwh is None
    assert created.alpha_prime_eur_mwh is None
    assert created.marginal_incremental_price_eur_mwh is None
    assert created.marginal_decremental_price_eur_mwh is None
    assert created.imbalance_price_eur_mwh is None


def test_domain_models_are_frozen() -> None:
    with pytest.raises(ValidationError, match="frozen"):
        envelope().quality_status = "Corrected"

    with pytest.raises(ValidationError, match="frozen"):
        observation().resolution_code = "PT15M"
