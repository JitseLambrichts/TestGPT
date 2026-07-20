#!/usr/bin/env sh
set -eu

python scripts/wait_for_stack.py
IMBALANCE_E2E_API_URL="${IMBALANCE_E2E_API_URL:-http://localhost:18000}" \
  python -m pytest tests/e2e/test_pipeline.py -m integration -q
