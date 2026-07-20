"""Deterministic Explore API fixture for the Docker end-to-end stack."""

import json
import math
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

_ANCHOR = datetime.now(UTC).replace(second=0, microsecond=0) - timedelta(minutes=240)


def _imbalance_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for index in range(240):
        timestamp = _ANCHOR + timedelta(minutes=index)
        quarter_hour = timestamp.replace(minute=(timestamp.minute // 15) * 15)
        value = round(120.0 * math.sin(index / 9.0) + 20.0 * math.cos(index / 3.0), 3)
        records.append(
            {
                "datetime": timestamp.isoformat().replace("+00:00", "Z"),
                "resolutioncode": "PT1M",
                "quarterhour": quarter_hour.isoformat().replace("+00:00", "Z"),
                "qualitystatus": "Validated",
                "ace": round(value / 2.0, 3),
                "systemimbalance": value,
                "alpha": 5.0,
                "alpha_prime": None,
                "marginalincrementalprice": 120.0,
                "marginaldecrementalprice": 100.0,
                "imbalanceprice": 110.0,
            }
        )
    return records


_RECORDS = _imbalance_records()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        parts = parsed.path.split("/")
        dataset = parts[-2] if len(parts) >= 2 and parts[-1] == "records" else ""
        records = _RECORDS if dataset == "ods161" else []
        query = parse_qs(parsed.query)
        offset = int(query.get("offset", ["0"])[0])
        limit = int(query.get("limit", ["100"])[0])
        payload = {"total_count": len(records), "results": records[offset : offset + limit]}
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        del format, args


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", 8080), Handler).serve_forever()
