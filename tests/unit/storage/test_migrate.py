from pathlib import Path
from types import SimpleNamespace

import pytest

from imbalance_pipeline.storage.migrate import (
    MANAGED_MIGRATIONS,
    MigrationChecksumError,
    apply_migrations,
)


class FakeMigrationClient:
    def __init__(self) -> None:
        self.database_exists = False
        self.ledger_exists = False
        self.applied: dict[int, str | bytes] = {}
        self.commands: list[str] = []

    async def query(self, statement: str) -> SimpleNamespace:
        if statement.startswith("EXISTS DATABASE"):
            return SimpleNamespace(first_row=(int(self.database_exists),))
        if statement.startswith("EXISTS TABLE"):
            return SimpleNamespace(first_row=(int(self.ledger_exists),))
        if statement.startswith("SELECT version, checksum"):
            return SimpleNamespace(result_rows=list(self.applied.items()))
        raise AssertionError(f"unexpected query: {statement}")

    async def command(self, statement: str) -> None:
        self.commands.append(statement)
        if "CREATE DATABASE" in statement:
            self.database_exists = True
        if "schema_migrations" in statement:
            self.ledger_exists = True

    async def insert(
        self,
        table: str,
        data: list[tuple[int, str, str, object]],
        *,
        column_names: tuple[str, ...],
    ) -> None:
        assert table == "demo.schema_migrations"
        assert column_names == ("version", "name", "checksum", "applied_at")
        for version, _name, checksum, _applied_at in data:
            self.applied[version] = checksum


def write_migrations(directory: Path) -> None:
    (directory / "001_schema.sql").write_text(
        "CREATE DATABASE IF NOT EXISTS demo;\n"
        "CREATE TABLE IF NOT EXISTS demo.schema_migrations (version UInt32);\n"
    )
    (directory / "002_preserve_source_versions.sql").write_text("-- managed by Python\n")


@pytest.mark.asyncio
async def test_runner_applies_managed_source_version_migration_once(tmp_path: Path) -> None:
    write_migrations(tmp_path)
    client = FakeMigrationClient()
    managed_calls: list[str] = []
    original = MANAGED_MIGRATIONS[2]

    async def managed(_client: FakeMigrationClient, *, database: str) -> None:
        managed_calls.append(database)

    MANAGED_MIGRATIONS[2] = managed
    try:
        await apply_migrations(client, database="demo", directory=tmp_path)
        await apply_migrations(client, database="demo", directory=tmp_path)
    finally:
        MANAGED_MIGRATIONS[2] = original

    assert managed_calls == ["demo"]
    assert sorted(client.applied) == [1, 2]


@pytest.mark.asyncio
async def test_runner_rejects_drift_in_an_already_applied_migration(tmp_path: Path) -> None:
    write_migrations(tmp_path)
    client = FakeMigrationClient()
    original = MANAGED_MIGRATIONS[2]

    async def managed(_client: FakeMigrationClient, *, database: str) -> None:
        del database

    MANAGED_MIGRATIONS[2] = managed
    try:
        await apply_migrations(client, database="demo", directory=tmp_path)
    finally:
        MANAGED_MIGRATIONS[2] = original

    (tmp_path / "002_preserve_source_versions.sql").write_text("-- changed managed migration\n")

    with pytest.raises(MigrationChecksumError, match="checksum changed"):
        await apply_migrations(client, database="demo", directory=tmp_path)


@pytest.mark.asyncio
async def test_runner_accepts_clickhouse_fixed_string_checksums_as_bytes(tmp_path: Path) -> None:
    write_migrations(tmp_path)
    client = FakeMigrationClient()
    original = MANAGED_MIGRATIONS[2]

    async def managed(_client: FakeMigrationClient, *, database: str) -> None:
        del database

    MANAGED_MIGRATIONS[2] = managed
    try:
        await apply_migrations(client, database="demo", directory=tmp_path)
        client.applied = {
            version: checksum.encode() if isinstance(checksum, str) else checksum
            for version, checksum in client.applied.items()
        }
        await apply_migrations(client, database="demo", directory=tmp_path)
    finally:
        MANAGED_MIGRATIONS[2] = original
