from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="IMBALANCE_",
        env_file=".env",
        extra="forbid",
        frozen=True,
    )

    nats_url: str = "nats://localhost:4222"
    clickhouse_url: str = "http://localhost:8123"
    clickhouse_database: str = "imbalance"
    clickhouse_user: str = "imbalance"
    clickhouse_password: str = "imbalance"
    elia_base_url: str = "https://opendata.elia.be/api/explore/v2.1"
    elia_imbalance_live_dataset: str = "ods161"
    elia_imbalance_history_dataset: str = "ods133"
    flip_deadband_mw: float = Field(default=10.0, gt=0)
    local_window_minutes: int = Field(default=180, ge=30)
    context_steps: int = Field(default=96, ge=24)
    ensemble_size: int = Field(default=3, ge=1)
    gmm_components: int = Field(default=5, ge=2)
    model_dir: Path = Path("/models/production")
    allow_fallback: bool = True
    api_host: str = "0.0.0.0"
    api_port: int = Field(default=8000, ge=1, le=65535)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
