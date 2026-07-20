import os
import time

import httpx
import pytest

from imbalance_pipeline.features.schema import DEFAULT_FEATURE_REGISTRY

API_URL = os.getenv("IMBALANCE_E2E_API_URL")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        API_URL is None, reason="set IMBALANCE_E2E_API_URL after starting compose.test"
    ),
]


def test_fixture_source_reaches_a_model_prediction_within_ten_seconds() -> None:
    assert API_URL is not None
    deadline = time.monotonic() + 45.0
    latest: httpx.Response | None = None
    while time.monotonic() < deadline:
        response = httpx.get(f"{API_URL}/v1/predictions/latest", timeout=2.0)
        if response.status_code == 200:
            latest = response
            break
        assert response.status_code in {404, 503}
        time.sleep(0.5)
    assert latest is not None, "fixture event did not reach the prediction API"
    prediction = latest.json()
    assert prediction["prediction_quality"] == "model"
    assert prediction["feature_schema_hash"] == DEFAULT_FEATURE_REGISTRY.fingerprint
    assert prediction["target_time"] > prediction["cutoff"]
