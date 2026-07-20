import os
import time
from urllib.error import URLError
from urllib.request import urlopen


def main() -> None:
    url = os.getenv("IMBALANCE_E2E_API_URL", "http://localhost:18000") + "/health/ready"
    deadline = time.monotonic() + float(os.getenv("IMBALANCE_STACK_TIMEOUT_SECONDS", "90"))
    while time.monotonic() < deadline:
        try:
            with urlopen(url, timeout=2) as response:  # noqa: S310 - local test URL
                if response.status == 200:
                    return
        except URLError:
            pass
        time.sleep(1)
    raise SystemExit(f"stack did not become ready: {url}")


if __name__ == "__main__":
    main()
