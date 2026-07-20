import os
from datetime import UTC, datetime, timedelta

import pytest

from imbalance_pipeline.sources.elia import EliaClient, normalize_imbalance

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.getenv("IMBALANCE_RUN_LIVE_TESTS") != "1",
        reason="set IMBALANCE_RUN_LIVE_TESTS=1 to call Elia Open Data",
    ),
]


@pytest.mark.asyncio
async def test_elia_ods161_has_a_normalizable_recent_minute() -> None:
    end = datetime.now(UTC)
    start = end - timedelta(minutes=15)
    async with EliaClient() as client:
        rows = [row async for row in client.iter_records("ods161", start, end)]

    assert rows
    observation = normalize_imbalance(rows[-1])
    assert observation.resolution_code == "PT1M"
