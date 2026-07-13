import asyncio
import hashlib
import re
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, cast

import clickhouse_connect  # type: ignore[import-untyped]
from clickhouse_connect.driver.asyncclient import AsyncClient  # type: ignore[import-untyped]

from imbalance_pipeline.config import get_settings
from imbalance_pipeline.storage.migrations import migrate_source_version_retention

_MIGRATION_FILE = re.compile(r"(?P<version>\d{3})_(?P<name>[a-z0-9_]+)\.sql$")
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*$")
_LEDGER_COLUMNS: Final = ("version", "name", "checksum", "applied_at")


@dataclass(frozen=True, slots=True)
class SchemaMigration:
    version: int
    name: str
    path: Path
    sql: str
    checksum: str


class MigrationChecksumError(RuntimeError):
    """An applied migration changed and can no longer be safely replayed."""


ManagedMigration = Callable[[AsyncClient], Awaitable[None]]


async def _source_version_migration(client: AsyncClient, *, database: str) -> None:
    await migrate_source_version_retention(client, database=database)


MANAGED_MIGRATIONS: dict[int, Callable[..., Awaitable[None]]] = {
    2: _source_version_migration,
}


def discover_migrations(directory: Path | None = None) -> tuple[SchemaMigration, ...]:
    migration_directory = directory or _default_migration_directory()
    migrations: list[SchemaMigration] = []
    for path in migration_directory.glob("[0-9][0-9][0-9]_*.sql"):
        match = _MIGRATION_FILE.fullmatch(path.name)
        if match is None:
            raise ValueError(f"invalid migration filename: {path.name}")
        contents = path.read_text()
        migrations.append(
            SchemaMigration(
                version=int(match["version"]),
                name=match["name"],
                path=path,
                sql=contents,
                checksum=hashlib.sha256(contents.encode()).hexdigest(),
            )
        )
    ordered = tuple(sorted(migrations, key=lambda migration: migration.version))
    if not ordered or ordered[0].version != 1:
        raise ValueError("migration directory must begin with 001_*.sql")
    if len({migration.version for migration in ordered}) != len(ordered):
        raise ValueError("migration versions must be unique")
    return ordered


async def apply_migrations(
    client: AsyncClient,
    *,
    database: str = "imbalance",
    directory: Path | None = None,
) -> None:
    _require_identifier(database)
    migrations = discover_migrations(directory)
    if not await _ledger_exists(client, database):
        await _execute_sql(client, migrations[0].sql)
    applied = await _applied_checksums(client, database)
    for migration in migrations:
        checksum = applied.get(migration.version)
        if checksum is not None:
            if checksum != migration.checksum:
                raise MigrationChecksumError(
                    f"migration {migration.version:03d}_{migration.name} checksum changed"
                )
            continue
        if migration.version != 1:
            await _execute_sql(client, migration.sql)
        managed = MANAGED_MIGRATIONS.get(migration.version)
        if managed is not None:
            await managed(client, database=database)
        await client.insert(
            f"{database}.schema_migrations",
            [(migration.version, migration.name, migration.checksum, datetime.now(UTC))],
            column_names=_LEDGER_COLUMNS,
        )


async def _ledger_exists(client: AsyncClient, database: str) -> bool:
    database_result = await client.query(f"EXISTS DATABASE {database}")
    if not bool(cast(int, database_result.first_row[0])):
        return False
    table_result = await client.query(f"EXISTS TABLE {database}.schema_migrations")
    return bool(cast(int, table_result.first_row[0]))


async def _applied_checksums(client: AsyncClient, database: str) -> dict[int, str]:
    result = await client.query(
        f"SELECT version, checksum FROM {database}.schema_migrations FINAL"
    )
    return {
        cast(int, version): _checksum_text(checksum)
        for version, checksum in result.result_rows
    }


async def _execute_sql(client: AsyncClient, contents: str) -> None:
    for statement in _sql_statements(contents):
        await client.command(statement)


def _sql_statements(contents: str) -> Iterable[str]:
    for fragment in contents.split(";"):
        statement = "\n".join(
            line for line in fragment.splitlines() if not line.lstrip().startswith("--")
        ).strip()
        if statement:
            yield statement


def _checksum_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("ascii")
    return str(value)


def _default_migration_directory() -> Path:
    return Path(__file__).parents[3] / "infra" / "clickhouse"


def _require_identifier(value: str) -> None:
    if _IDENTIFIER.fullmatch(value) is None:
        raise ValueError("database must be a ClickHouse identifier")


async def _run() -> None:
    settings = get_settings()
    client = await clickhouse_connect.get_async_client(
        dsn=settings.clickhouse_url,
        username=settings.clickhouse_user,
        password=settings.clickhouse_password,
        database="default",
    )
    try:
        await apply_migrations(client, database=settings.clickhouse_database)
    finally:
        await client.close()


def main() -> None:
    asyncio.run(_run())
