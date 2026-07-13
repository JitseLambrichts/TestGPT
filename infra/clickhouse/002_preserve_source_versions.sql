-- Rebuild source tables because ClickHouse cannot append an existing column to
-- an existing sorting key. Run before source writers start, as the Docker
-- migration job does. Pre-migration revisions that have already been merged
-- cannot be reconstructed, but every revision written after this migration remains
-- available for point-in-time feature reconstruction.

CREATE TABLE imbalance.raw_events__v2 AS imbalance.raw_events
ENGINE = ReplacingMergeTree(row_version)
PARTITION BY toYYYYMM(event_time)
ORDER BY (event_id, event_time, row_version);

ALTER TABLE imbalance.raw_events__v2
    MODIFY TTL toDateTime(event_time, 'UTC') + INTERVAL 90 DAY DELETE;

CREATE TABLE imbalance.imbalance_observations__v2 AS imbalance.imbalance_observations
ENGINE = ReplacingMergeTree(row_version)
PARTITION BY toYYYYMM(timestamp)
ORDER BY (timestamp, event_id, row_version);

CREATE TABLE imbalance.load_observations__v2 AS imbalance.load_observations
ENGINE = ReplacingMergeTree(row_version)
PARTITION BY toYYYYMM(timestamp)
ORDER BY (timestamp, event_id, row_version);

CREATE TABLE imbalance.wind_observations__v2 AS imbalance.wind_observations
ENGINE = ReplacingMergeTree(row_version)
PARTITION BY toYYYYMM(timestamp)
ORDER BY (timestamp, offshore_onshore, region, grid_connection_type, event_id, row_version);

CREATE TABLE imbalance.solar_observations__v2 AS imbalance.solar_observations
ENGINE = ReplacingMergeTree(row_version)
PARTITION BY toYYYYMM(timestamp)
ORDER BY (timestamp, region, event_id, row_version);

INSERT INTO imbalance.raw_events__v2 SELECT * FROM imbalance.raw_events;

INSERT INTO imbalance.imbalance_observations__v2 SELECT * FROM imbalance.imbalance_observations;

INSERT INTO imbalance.load_observations__v2 SELECT * FROM imbalance.load_observations;

INSERT INTO imbalance.wind_observations__v2 SELECT * FROM imbalance.wind_observations;

INSERT INTO imbalance.solar_observations__v2 SELECT * FROM imbalance.solar_observations;

RENAME TABLE
    imbalance.raw_events TO imbalance.raw_events__v1_backup,
    imbalance.raw_events__v2 TO imbalance.raw_events,
    imbalance.imbalance_observations TO imbalance.imbalance_observations__v1_backup,
    imbalance.imbalance_observations__v2 TO imbalance.imbalance_observations,
    imbalance.load_observations TO imbalance.load_observations__v1_backup,
    imbalance.load_observations__v2 TO imbalance.load_observations,
    imbalance.wind_observations TO imbalance.wind_observations__v1_backup,
    imbalance.wind_observations__v2 TO imbalance.wind_observations,
    imbalance.solar_observations TO imbalance.solar_observations__v1_backup,
    imbalance.solar_observations__v2 TO imbalance.solar_observations;
