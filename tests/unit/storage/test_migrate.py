import asyncio
from pathlib import Path
from tomllib import loads
from types import SimpleNamespace

import pytest
from clickhouse_connect.driver.exceptions import ClickHouseError

from imbalance_pipeline.storage.migrate import (
    MANAGED_MIGRATIONS,
    MigrationChecksumError,
    MigrationLocked,
    _execute_sql,
    apply_migrations,
)


class FakeMigrationClient:
    def __init__(self) -> None:
        self.database_exists = False
        self.ledger_exists = False
        self.migration_lock_held = False
        self.applied: dict[int, str | bytes] = {}
        self.commands: list[str] = []

    async def query(self, statement: str) -> SimpleNamespace:
        if statement.startswith("EXISTS DATABASE"):
            return SimpleNamespace(first_row=(int(self.database_exists),))
        if statement == "EXISTS TABLE demo.__migration_lock":
            return SimpleNamespace(first_row=(int(self.migration_lock_held),))
        if statement.startswith("EXISTS TABLE"):
            return SimpleNamespace(first_row=(int(self.ledger_exists),))
        if statement.startswith("SELECT version, checksum"):
            return SimpleNamespace(result_rows=list(self.applied.items()))
        raise AssertionError(f"unexpected query: {statement}")

    async def command(self, statement: str) -> None:
        normalized = statement.lstrip()
        if normalized.startswith("CREATE TABLE demo.__migration_lock"):
            if self.migration_lock_held:
                raise ClickHouseError("migration lock already exists")
            self.migration_lock_held = True
            return
        if normalized.startswith("DROP TABLE IF EXISTS demo.__migration_lock"):
            self.migration_lock_held = False
            return
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


@pytest.mark.asyncio
async def test_runner_completes_an_unregistered_bootstrap_after_a_crash(tmp_path: Path) -> None:
    write_migrations(tmp_path)
    client = FakeMigrationClient()
    client.database_exists = True
    client.ledger_exists = True
    original = MANAGED_MIGRATIONS[2]

    async def managed(_client: FakeMigrationClient, *, database: str) -> None:
        del database

    MANAGED_MIGRATIONS[2] = managed
    try:
        await apply_migrations(client, database="demo", directory=tmp_path)
    finally:
        MANAGED_MIGRATIONS[2] = original

    assert "CREATE TABLE IF NOT EXISTS demo.schema_migrations (version UInt32)" in client.commands
    assert sorted(client.applied) == [1, 2]


@pytest.mark.asyncio
async def test_runner_rejects_a_second_concurrent_migration_process(tmp_path: Path) -> None:
    write_migrations(tmp_path)
    client = FakeMigrationClient()
    entered = asyncio.Event()
    release = asyncio.Event()
    original = MANAGED_MIGRATIONS[2]

    async def managed(_client: FakeMigrationClient, *, database: str) -> None:
        del database
        entered.set()
        await release.wait()

    MANAGED_MIGRATIONS[2] = managed
    first = asyncio.create_task(apply_migrations(client, database="demo", directory=tmp_path))
    try:
        await entered.wait()
        with pytest.raises(MigrationLocked, match="already running"):
            await apply_migrations(client, database="demo", directory=tmp_path)
    finally:
        release.set()
        await first
        MANAGED_MIGRATIONS[2] = original

    assert client.migration_lock_held is False


@pytest.mark.asyncio
async def test_runner_renders_the_database_identifier_after_identifier_validation(
    tmp_path: Path,
) -> None:
    (tmp_path / "001_schema.sql").write_text(
        "CREATE DATABASE IF NOT EXISTS imbalance;\n"
        "CREATE TABLE IF NOT EXISTS imbalance.schema_migrations (version UInt32);\n"
    )
    (tmp_path / "002_preserve_source_versions.sql").write_text("-- managed by Python\n")
    client = FakeMigrationClient()
    original = MANAGED_MIGRATIONS[2]

    async def managed(_client: FakeMigrationClient, *, database: str) -> None:
        del database

    MANAGED_MIGRATIONS[2] = managed
    try:
        await apply_migrations(client, database="demo", directory=tmp_path)
    finally:
        MANAGED_MIGRATIONS[2] = original

    assert "CREATE TABLE IF NOT EXISTS demo.schema_migrations (version UInt32)" in client.commands
    assert all("imbalance.schema_migrations" not in command for command in client.commands)


def test_wheel_configuration_includes_sql_migrations_as_package_resources() -> None:
    pyproject = Path(__file__).parents[3] / "pyproject.toml"
    configuration = loads(pyproject.read_text())

    assert configuration["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"] == {
        "infra/clickhouse": "imbalance_pipeline/storage/sql_migrations"
    }


@pytest.mark.asyncio
async def test_runner_accepts_the_docker_xml_user_grant_limitation() -> None:
    client = FakeMigrationClient()

    async def readonly_grant(statement: str) -> None:
        if statement.startswith("GRANT "):
            raise ClickHouseError("ACCESS_STORAGE_READONLY")
        client.commands.append(statement)

    client.command = readonly_grant  # type: ignore[method-assign]

    await _execute_sql(
        client, "CREATE TABLE demo.example (id UInt8); GRANT SELECT ON demo.* TO demo;"
    )

    assert client.commands == ["CREATE TABLE demo.example (id UInt8)"]
