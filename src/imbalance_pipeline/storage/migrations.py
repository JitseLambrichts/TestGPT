from dataclasses import dataclass
from typing import cast

from clickhouse_connect.driver.asyncclient import AsyncClient  # type: ignore[import-untyped]


@dataclass(frozen=True, slots=True)
class SourceVersionMigration:
    table: str
    partition_by: str
    order_by: str
    ttl: str | None = None


SOURCE_VERSION_MIGRATIONS = (
    SourceVersionMigration(
        table="raw_events",
        partition_by="toYYYYMM(event_time)",
        order_by="(event_id, event_time, row_version)",
        ttl="toDateTime(event_time, 'UTC') + INTERVAL 90 DAY DELETE",
    ),
    SourceVersionMigration(
        table="imbalance_observations",
        partition_by="toYYYYMM(timestamp)",
        order_by="(timestamp, event_id, row_version)",
    ),
    SourceVersionMigration(
        table="load_observations",
        partition_by="toYYYYMM(timestamp)",
        order_by="(timestamp, event_id, row_version)",
    ),
    SourceVersionMigration(
        table="wind_observations",
        partition_by="toYYYYMM(timestamp)",
        order_by=(
            "(timestamp, offshore_onshore, region, grid_connection_type, event_id, row_version)"
        ),
    ),
    SourceVersionMigration(
        table="solar_observations",
        partition_by="toYYYYMM(timestamp)",
        order_by="(timestamp, region, event_id, row_version)",
    ),
)


class MigrationError(RuntimeError):
    """A migration cannot safely determine its next recovery step."""


async def migrate_source_version_retention(
    client: AsyncClient,
    *,
    database: str = "imbalance",
) -> None:
    for definition in SOURCE_VERSION_MIGRATIONS:
        await _migrate_source_table(client, database, definition)


async def _migrate_source_table(
    client: AsyncClient,
    database: str,
    definition: SourceVersionMigration,
) -> None:
    source = f"{database}.{definition.table}"
    recovery = f"{source}__v2"
    if await _has_expected_order(client, source, definition.order_by):
        return
    if await _table_exists(client, recovery):
        if not await _has_expected_order(client, recovery, definition.order_by):
            raise MigrationError(
                f"cannot resume {definition.table}: recovery table has an unknown schema"
            )
    else:
        await client.command(
            f"""
            CREATE TABLE {recovery} AS {source}
            ENGINE = ReplacingMergeTree(row_version)
            PARTITION BY {definition.partition_by}
            ORDER BY {definition.order_by}
            """
        )
    if definition.ttl is not None:
        await client.command(f"ALTER TABLE {recovery} MODIFY TTL {definition.ttl}")
    await client.command(f"TRUNCATE TABLE {recovery}")
    await client.command(f"INSERT INTO {recovery} SELECT * FROM {source}")
    await client.command(f"EXCHANGE TABLES {source} AND {recovery}")
    if not await _has_expected_order(client, source, definition.order_by):
        raise MigrationError(f"source version migration did not activate {definition.table}")


async def _table_exists(client: AsyncClient, table: str) -> bool:
    result = await client.query(f"EXISTS TABLE {table}")
    return bool(cast(int, result.first_row[0]))


async def _has_expected_order(
    client: AsyncClient,
    table: str,
    order_by: str,
) -> bool:
    if not await _table_exists(client, table):
        return False
    result = await client.query(f"SHOW CREATE TABLE {table}")
    return f"ORDER BY {order_by}" in str(result.first_row[0])
