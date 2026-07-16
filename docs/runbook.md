# Runbook

## Gezonde toestand

```bash
docker compose ps
curl --fail http://localhost:8000/health/ready
curl http://localhost:8000/v1/models/current
```

NATS en ClickHouse moeten `healthy` zijn; `migrate` moet met code 0 afgerond zijn. De API-readiness accepteert een gevalideerd model of een expliciet toegestane fallback.

## Bronstaleness of ingestiefouten

Controleer de ingestorlogs en de Prometheus-teller `imbalance_ingestor_source_errors_total`. Elia ODS161 is polling, geen pushstream. Controleer eerst de bronbeschikbaarheid; start daarna uitsluitend de ingestor opnieuw:

```bash
docker compose restart ingestor
docker compose logs --tail=200 ingestor
```

Gebruik voor een gecontroleerde historische replay `make backfill START=… END=…`, met één UTC-dag per run. Start niet met willekeurige grote ranges: de service weigert ranges langer dan een dag.

## Historische training

Backfill historische imbalance-data uit ODS133 (ODS161 is uitsluitend de live bron). Backfill minstens de volledige contextgeschiedenis vóór de eerste exportdag; exporteer daarna point-in-time voorbeelden en train een kandidaatbundle:

```bash
make backfill START=2026-07-13T00:00:00Z END=2026-07-14T00:00:00Z
make export-training START=2026-07-13T00:00:00Z END=2026-07-14T00:00:00Z OUTPUT=exports/imbalance-2026-07-13
make train DATASET=exports/imbalance-2026-07-13 OUTPUT=candidates/imbalance-2026-07-13
```

Controleer de evaluatie- en checksuminformatie en gebruik daarna de bestaande `promote_bundle`-functie als expliciete operatoractie. Na promotie herlaad je alleen de predictor en API:

```bash
docker compose restart predictor api
```

## JetStream-lag, redelivery en DLQ

Open NATS monitoring op `http://localhost:8222` vanuit het Docker-netwerk of inspecteer de stream met een NATS-adminclient. Controleer durable consumers `clickhouse-*`, `feature-builder-v1`, `predictor-v1` en `outcome-*`. Een poison message wordt na begrensde retries naar het relevante `grid.dlq.*` subject gestuurd.

Herstel eerst de oorzaak, inspecteer daarna één DLQ-event en republish uitsluitend dat event met dezelfde schema-versie en een nieuwe causation-id. Ruim een DLQ nooit blind op: een malformed payload moet reproduceerbaar blijven.

## ClickHouse

```bash
docker compose exec clickhouse clickhouse-client --user imbalance --password imbalance --query 'SELECT 1'
docker compose logs --tail=200 migrate
```

Migraties zijn checksum-geledgerd. Een gewijzigde reeds toegepaste migratie is fataal; maak dan een nieuwe genummerde migratie, wijzig de oude niet. Maak backups met ClickHouse-native backup tooling of consistente snapshots van het named volume voordat je schema of retentie verandert.

## Modelproblemen en rollback

Een ontbrekend, corrupt of schema-incompatibel bundle mag geen stille modeloutput geven. Met `IMBALANCE_ALLOW_FALLBACK=true` publiceert de predictor gemarkeerde fallbackrecords; met `false` wordt readiness onbeschikbaar.

Rollback is atomair: laat `models/production` naar een eerdere gevalideerde versie wijzen en herstart alleen de predictor en API.

```bash
ln -sfn imbalance-YYYYMMDDTHHMMSSZ models/production
docker compose restart predictor api
curl --fail http://localhost:8000/health/ready
```

Gebruik alleen bundles waarvan `validate_bundle` succesvol is; wijzig artifacts nooit in-place.

## Veilige stop en destructieve cleanup

```bash
docker compose down
CONFIRM_CLEAN=1 make clean
```

De eerste opdracht laat persistente volumes intact. De tweede verwijdert ClickHouse- en JetStream-data; maak eerst een backup en bevestig dat dit werkelijk een lokale niet-productieomgeving is.
