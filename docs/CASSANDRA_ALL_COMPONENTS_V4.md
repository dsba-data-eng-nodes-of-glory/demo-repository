# Version 4: Cassandra/CSV selectable operational storage

This is an **incremental patch** for the ZIP uploaded as `Version 4.zip` (archive root `Version 2/`). **The original directory structure is preserved.** Model training logic, checkpoints, station registry and the Kafka/Spark topology are not replaced.

## What is integrated

| Data flow | CSV (`backend: csv`) | Cassandra (`backend: cassandra`) |
|---|---|---|
| Demand observations | `src/storage/forecast_store.py` CSV | `actuals_by_station_month` |
| Demand 96-step forecasts | CSV | `forecasts_by_station_day` and current-run pointer |
| Solar and wind forecasts | `renewable_forecasts.csv` | `renewable_forecasts_by_station_day` |
| Solar and wind observed generation | `renewable_actuals.csv` *when an observation exists* | `renewable_actuals_by_station_month` *when an observation exists* |
| Demand anomaly score snapshots | SQLite | `anomaly_assessments_by_station_day` |
| Quality events | JSONL | `quality_events_by_station_day` |
| Dashboard reads | Existing file-based API paths | Same API routes, Cassandra-backed |
| Training datasets and model checkpoints | CSV/Excel and `.pt`/`.joblib` | **Remain files (not migrated)** |

**Important:** Cassandra cannot supply renewable *observations* that do not exist in the source feeds. The existing forecaster uses registered historical generation files and only records generation actuals when the accepted replay timestamp exists in those files. If the stream proceeds beyond the end of the historical solar/wind data, neither backend fabricates observations. A real operational generation ingestion feed is still a separate future requirement.

## Apply the patch

1. Stop FastAPI, Streamlit and the station's producer, Spark worker and predictor. Back up `config/`, `data/`, `models/` and `checkpoints/`.
2. Extract `Cassandra_All_Models_V4_Patch.zip` into the **Version 4 project root**, preserving all relative paths. The patch includes code, documentation and tests only. It does **not** contain data or trained models.
3. Install the existing dependencies if needed: `pip install -r requirements.txt`. This project already specifies `cassandra-driver>=3.29,<4.0`.
4. Start the Cassandra database and verify that its port is reachable from the machine where the API and predictor run (default `127.0.0.1:9042`). Docker containers must use the reachable Cassandra host, not `127.0.0.1` unless Cassandra is in the same container.
5. Initialize Cassandra tables: `python -m scripts.init_cassandra`. This creates the missing renewable, anomaly and partition-index tables alongside existing demand/quality tables; it does not delete old data.
6. Before switching backends, perform the explicit import preflight: `python -m scripts.migrate_csv_to_cassandra --station amprion`. When satisfied, run `python -m scripts.migrate_csv_to_cassandra --station amprion --apply` to import existing **operational** history. The command is optional if you prefer to start a new empty Cassandra operational history. It never touches training data or deletes source files.
7. Set `backend: cassandra` in `config/storage.yaml`, or set `STORAGE_BACKEND=cassandra` consistently for the API, predictor, quality consumer and other services. **Environment override takes precedence**. Restart those processes.
8. Verify `http://127.0.0.1:8000/health/ready` reports `"storage": {"backend":"cassandra","available":true}` and check:
   - `/results/actuals?station=amprion&limit=10`
   - `/results/forecasts?station=amprion&limit=96`
   - `/results/renewables?station=amprion`
   - `/results/renewables/performance?station=amprion`
   - `/results/anomaly/forecast?station=amprion`
   - `/results/quality/events?station=amprion`
9. Only then start replay, confirm that the predictor reports `Operational storage: ... backend: cassandra`, and watch disk capacity, API latency and Kafka lag.

### Switch back to CSV

Set `backend: csv` in `config/storage.yaml` (and remove or change any `STORAGE_BACKEND` environment override), then restart the services. CSV remains a first-class mode. **Backend switching does not silently copy records between stores.** Any data written only to Cassandra will not appear in CSV mode until explicitly exported/migrated. Keep your old CSV files and backups. This patch's import script is one-way (CSV/SQLite -> Cassandra), not a transparent bidirectional synchronization mechanism.

### Storage retention and history

`config/storage.yaml` adds:

```yaml
cassandra:
  renewable_forecast_history_days: 14
  renewable_actual_history_months: 12
  anomaly_history_days: 30
  anomaly_ttl_days: 30
```

`anomaly_ttl_days` expires **new** anomaly assessments; their logical origin can be historical replay time. Cassandra TTL uses real write time. Other forecast rows currently have no automatic TTL; monitor capacity and add a planned retention policy before sustained production use. API history windows are *query limits*, not retention policies.

### Reset and delete

The patch registers each Cassandra partition written by the new code in `storage_buckets_by_station`. Reset and deletion preflight the Cassandra connection and inspect this station-scoped manifest; operations delete only the selected station's known partitions, without table-wide `TRUNCATE`. They also preserve the existing stopped-process/Kafka-topic safety checks.

**Legacy Cassandra records written before the manifest existed are an exception.** If existing demand pointers have no registered buckets, reset/delete refuses rather than claiming to have removed all data. Mixed pre- and post-patch unindexed legacy records may also require an administrator-assisted migration/cleanup. Do not use Cassandra reset as a general purge of unknown preexisting tables. Cassandra partition deletion and Kafka log deletion are not one atomic transaction: back up before reset.

### Accuracy and interpretation

The anomaly scoring algorithm itself is unchanged. Cassandra persists scored forecast-origin snapshots, preserving threshold and model fingerprints without altering the Dense Autoencoder. An anomaly prediction can legitimately be revised by a newer forecast. Stored old assessments are **previous predictions**, not confirmed grid events.

Solar/wind performance currently uses registered historical generation constrained by the replay's acknowledged-demand watermark. Operational production accuracy requires actual live solar/wind ingestion, not only historical weather or observed demand.

### Testing and deployment limitations

The added focused tests simulate Cassandra sessions (including checks for partitioned writes and station-isolated deletes), verify CSV switching and anomaly persistence routing, and exercise the renewable results endpoint. These are not a substitute for a real local Cassandra integration test. Tests here ran without a Cassandra server or `cassandra-driver`, and Streamlit rendering/Kafka/Spark Windows deployment were not exercised. Run the endpoints and replay in a real Cassandra environment before customer deployment.

The project already has several older monitoring assertions expecting former refresh defaults/labels. Those are unchanged by this patch and should be updated or investigated separately.

## Changed files

- `src/storage/cassandra_store.py` — new Cassandra tables, prepared statements, read/write paths and station-scoped cleanup
- `src/storage/renewable_store.py` — CSV/Cassandra switching for renewable forecast and actual generation
- `src/storage/forecast_store.py` — validated backend selector; existing demand and quality routing retained
- `src/models/renewable_forecaster.py` — expose actual generation only at an observed replay timestamp
- `src/models/streaming_predictor.py` — persist generation actuals before Kafka offset acknowledgement
- `src/models/anomaly/snapshot_store.py` — Cassandra or SQLite score snapshots
- `backend/results_api.py` — backend-aware renewable forecast, actual and performance reads; existing anomaly API automatically dispatches via snapshot facade
- `backend/pipeline_manager.py` — safe Cassandra reset/delete preflight and partition cleanup
- `config/storage.yaml` — additive optional query/retention configuration, CSV remains default
- `scripts/migrate_csv_to_cassandra.py` — explicit non-destructive import utility
- `tests/test_cassandra_full_pipeline_patch.py` — targeted tests

## Important operational advice

- Cassandra is not a remedy for full disks. Its own commit logs, SSTables and compaction require free space. Keep it on adequately provisioned storage and avoid filling the Windows system drive.
- Run Cassandra in a separate service or host. A single-node Cassandra install does **not** provide distributed fault tolerance.
- Use a consistent backend setting across **all** runtime processes. Do not change backend while streaming; stop/restart first.
- Do not delete Kafka KRaft storage or Spark checkpoints merely to activate Cassandra.
