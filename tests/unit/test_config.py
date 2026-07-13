import pytest
from pydantic import ValidationError

from imbalance_pipeline.config import Settings


def test_settings_use_safe_local_defaults() -> None:
    settings = Settings()
    assert settings.flip_deadband_mw == 10.0
    assert settings.local_window_minutes == 180
    assert settings.context_steps == 96
    assert settings.ensemble_size == 3
    assert settings.gmm_components == 5


def test_deadband_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        Settings(flip_deadband_mw=0)
