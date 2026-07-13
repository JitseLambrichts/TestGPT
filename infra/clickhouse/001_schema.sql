CREATE DATABASE IF NOT EXISTS imbalance;

CREATE TABLE IF NOT EXISTS imbalance.schema_migrations
(
    version UInt32,
    name String,
    checksum FixedString(64),
    applied_at DateTime64(3, 'UTC')
)
ENGINE = ReplacingMergeTree(applied_at)
ORDER BY version;

CREATE TABLE IF NOT EXISTS imbalance.raw_events
(
    event_id String,
    event_type LowCardinality(String),
    schema_version LowCardinality(String),
    source LowCardinality(String),
    dataset LowCardinality(String),
    event_time DateTime64(3, 'UTC'),
    observed_at Nullable(DateTime64(3, 'UTC')),
    ingested_at DateTime64(3, 'UTC'),
    correlation_id String,
    causation_id Nullable(String),
    quality_status LowCardinality(String),
    payload_json String,
    envelope_json String,
    row_version UInt64
)
ENGINE = ReplacingMergeTree(row_version)
PARTITION BY toYYYYMM(event_time)
ORDER BY (event_id, event_time)
TTL toDateTime(event_time, 'UTC') + INTERVAL 90 DAY DELETE;

CREATE TABLE IF NOT EXISTS imbalance.imbalance_observations
(
    event_id String,
    timestamp DateTime64(3, 'UTC'),
    quarter_hour DateTime64(3, 'UTC'),
    resolution_code LowCardinality(String),
    quality_status LowCardinality(String),
    ace_mw Nullable(Float64),
    system_imbalance_mw Float64,
    alpha_eur_mwh Nullable(Float64),
    alpha_prime_eur_mwh Nullable(Float64),
    marginal_incremental_price_eur_mwh Nullable(Float64),
    marginal_decremental_price_eur_mwh Nullable(Float64),
    imbalance_price_eur_mwh Nullable(Float64),
    ingested_at DateTime64(3, 'UTC'),
    row_version UInt64
)
ENGINE = ReplacingMergeTree(row_version)
PARTITION BY toYYYYMM(timestamp)
ORDER BY (timestamp, event_id);

CREATE TABLE IF NOT EXISTS imbalance.load_observations
(
    event_id String,
    timestamp DateTime64(3, 'UTC'),
    resolution_code LowCardinality(String),
    measured_mw Nullable(Float64),
    most_recent_forecast_mw Nullable(Float64),
    most_recent_confidence_10_mw Nullable(Float64),
    most_recent_confidence_90_mw Nullable(Float64),
    day_ahead_forecast_mw Nullable(Float64),
    day_ahead_confidence_10_mw Nullable(Float64),
    day_ahead_confidence_90_mw Nullable(Float64),
    week_ahead_forecast_mw Nullable(Float64),
    monitored_capacity_mw Nullable(Float64),
    load_factor Nullable(Float64),
    ingested_at DateTime64(3, 'UTC'),
    row_version UInt64
)
ENGINE = ReplacingMergeTree(row_version)
PARTITION BY toYYYYMM(timestamp)
ORDER BY (timestamp, event_id);

CREATE TABLE IF NOT EXISTS imbalance.wind_observations
(
    event_id String,
    timestamp DateTime64(3, 'UTC'),
    resolution_code LowCardinality(String),
    offshore_onshore LowCardinality(String),
    region LowCardinality(String),
    grid_connection_type LowCardinality(String),
    real_time_mw Nullable(Float64),
    most_recent_forecast_mw Nullable(Float64),
    most_recent_confidence_10_mw Nullable(Float64),
    most_recent_confidence_90_mw Nullable(Float64),
    day_ahead_11h_forecast_mw Nullable(Float64),
    day_ahead_11h_confidence_10_mw Nullable(Float64),
    day_ahead_11h_confidence_90_mw Nullable(Float64),
    day_ahead_forecast_mw Nullable(Float64),
    day_ahead_confidence_10_mw Nullable(Float64),
    day_ahead_confidence_90_mw Nullable(Float64),
    week_ahead_forecast_mw Nullable(Float64),
    week_ahead_confidence_10_mw Nullable(Float64),
    week_ahead_confidence_90_mw Nullable(Float64),
    monitored_capacity_mw Nullable(Float64),
    load_factor Nullable(Float64),
    decremental_bid_id Nullable(String),
    ingested_at DateTime64(3, 'UTC'),
    row_version UInt64
)
ENGINE = ReplacingMergeTree(row_version)
PARTITION BY toYYYYMM(timestamp)
ORDER BY (timestamp, offshore_onshore, region, grid_connection_type, event_id);

CREATE TABLE IF NOT EXISTS imbalance.solar_observations
(
    event_id String,
    timestamp DateTime64(3, 'UTC'),
    resolution_code LowCardinality(String),
    region LowCardinality(String),
    real_time_mw Nullable(Float64),
    most_recent_forecast_mw Nullable(Float64),
    most_recent_confidence_10_mw Nullable(Float64),
    most_recent_confidence_90_mw Nullable(Float64),
    day_ahead_11h_forecast_mw Nullable(Float64),
    day_ahead_11h_confidence_10_mw Nullable(Float64),
    day_ahead_11h_confidence_90_mw Nullable(Float64),
    day_ahead_forecast_mw Nullable(Float64),
    day_ahead_confidence_10_mw Nullable(Float64),
    day_ahead_confidence_90_mw Nullable(Float64),
    week_ahead_forecast_mw Nullable(Float64),
    week_ahead_confidence_10_mw Nullable(Float64),
    week_ahead_confidence_90_mw Nullable(Float64),
    monitored_capacity_mw Nullable(Float64),
    load_factor Nullable(Float64),
    ingested_at DateTime64(3, 'UTC'),
    row_version UInt64
)
ENGINE = ReplacingMergeTree(row_version)
PARTITION BY toYYYYMM(timestamp)
ORDER BY (timestamp, region, event_id);

CREATE TABLE IF NOT EXISTS imbalance.feature_snapshots
(
    event_id String,
    cutoff DateTime64(3, 'UTC'),
    target_time DateTime64(3, 'UTC'),
    feature_schema_hash String,
    local_values Array(Float32),
    local_masks Array(UInt8),
    context_values Array(Float32),
    context_masks Array(UInt8),
    static_values Array(Float32),
    static_masks Array(UInt8),
    current_state LowCardinality(Nullable(String)),
    created_at DateTime64(3, 'UTC'),
    row_version UInt64
)
ENGINE = ReplacingMergeTree(row_version)
PARTITION BY toYYYYMM(target_time)
ORDER BY (target_time, feature_schema_hash, event_id);

CREATE TABLE IF NOT EXISTS imbalance.predictions
(
    event_id String,
    cutoff DateTime64(3, 'UTC'),
    target_time DateTime64(3, 'UTC'),
    generated_at DateTime64(3, 'UTC'),
    system_imbalance_mw Float64,
    p10_mw Float64,
    p90_mw Float64,
    flip_probability Float32,
    will_flip UInt8,
    current_state Nullable(String),
    predicted_state Nullable(String),
    prediction_quality LowCardinality(String),
    model_version String,
    feature_schema_hash String,
    row_version UInt64
)
ENGINE = ReplacingMergeTree(row_version)
PARTITION BY toYYYYMM(target_time)
ORDER BY (target_time, model_version, event_id);

CREATE TABLE IF NOT EXISTS imbalance.prediction_outcomes
(
    prediction_event_id String,
    target_time DateTime64(3, 'UTC'),
    realized_event_id String,
    realized_system_imbalance_mw Float64,
    realized_state Nullable(String),
    flip_actual Nullable(UInt8),
    evaluated_at DateTime64(3, 'UTC'),
    row_version UInt64
)
ENGINE = ReplacingMergeTree(row_version)
PARTITION BY toYYYYMM(target_time)
ORDER BY (prediction_event_id, target_time);

CREATE TABLE IF NOT EXISTS imbalance.model_versions
(
    model_version String,
    feature_schema_hash String,
    manifest_json String,
    metrics_json String,
    promoted_at Nullable(DateTime64(3, 'UTC')),
    created_at DateTime64(3, 'UTC'),
    row_version UInt64
)
ENGINE = ReplacingMergeTree(row_version)
PARTITION BY toYYYYMM(created_at)
ORDER BY (model_version, feature_schema_hash);

GRANT SELECT, INSERT ON imbalance.* TO imbalance;
