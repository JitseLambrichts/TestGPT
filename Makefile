.DEFAULT_GOAL := help

.PHONY: help up down clean test train export-training smoke backfill observability

help:
	@printf '%s\n' 'up: start the realtime pipeline' 'observability: start the pipeline plus Prometheus and Grafana' 'backfill: publish an explicit UTC day window (pass START=... END=...)' 'export-training: export a checksummed dataset (pass START=... END=... OUTPUT=...)' 'down: stop the pipeline' 'test: run unit tests' 'train: run the trainer profile (pass DATASET=... OUTPUT=...)' 'smoke: run the opt-in live Elia test' 'clean: remove local Docker volumes (requires CONFIRM_CLEAN=1)'

up:
	docker compose up --build --detach

observability:
	docker compose --profile observability up --build --detach

backfill:
	@test -n "$(START)" && test -n "$(END)" || (echo "Set START and END as UTC ISO-8601 timestamps" && exit 1)
	docker compose run --rm ingestor imbalance-ingestor --start "$(START)" --end "$(END)"

down:
	docker compose down

clean:
	@test "$(CONFIRM_CLEAN)" = "1" || (echo "Set CONFIRM_CLEAN=1 to remove local volumes" && exit 1)
	docker compose down --volumes --remove-orphans

test:
	python -m pytest -m "not integration and not live"

train:
	@test -n "$(DATASET)" && test -n "$(OUTPUT)" || (echo "Set DATASET and OUTPUT" && exit 1)
	docker compose --profile training run --rm trainer imbalance-train /data/$(DATASET) /models/$(OUTPUT)

export-training:
	@test -n "$(START)" && test -n "$(END)" && test -n "$(OUTPUT)" || (echo "Set START, END, and OUTPUT as UTC ISO-8601 values" && exit 1)
	docker compose --profile training run --rm trainer imbalance-export-training --start "$(START)" --end "$(END)" --output /data/$(OUTPUT)

smoke:
	IMBALANCE_RUN_LIVE_TESTS=1 python -m pytest tests/e2e/test_live_smoke.py -m live -q
