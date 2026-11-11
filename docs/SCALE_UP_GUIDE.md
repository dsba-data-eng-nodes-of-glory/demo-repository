# Scale-up implementation guide (without Data Lakes)

## Revised architecture

```text
TSO CSV / replay connectors
        |
        v
Python Kafka Producers
        |
        v
Kafka KRaft cluster (3 brokers; RF=3; min ISR=2)
        |
        v
Spark Structured Streaming (master + workers)
  - JSON parsing and schema validation
  - UTC timestamp validation
  - calendar enrichment
  - invalid-record routing to DLQ
        |
        v
Kafka featured topics
        |
        v
Forecast predictor (LSTM / RF / GBT / XGBoost)
        |
        +---- actual observations ----+
        +---- forecast history -------+----> Cassandra 3-node ring (RF=3)
        +---- latest complete run ---+
        +---- quality events --------+
                                            |
                                      FastAPI + Prometheus
                                            |
                                      Streamlit + Grafana
```

No Data Lake is included. Historical training CSVs and local model artifacts remain as they are. Cassandra is the operational time-series store for live actuals, forecast runs, latest forecasts and quality events.

## What this patch changes

### 1. Multi-source ingestion and Kafka

- Keeps the existing station-configured producer and separate raw/featured topics.
- Topic creation now reads `topic_partitions` and `replication_factor` from Kafka configuration/environment rather than hardcoding 1/1.
- The Compose cluster provisions three KRaft brokers. The application config sets RF=3 and min ISR=2.
- Each station still has one ordered aggregate time series and its own topic. One partition per station topic is deliberate: the producer key is the station ID, so all messages for that station would land in the same partition anyway. Scale across stations; for intra-station parallelism, first introduce independent zones/substations/assets and key by that entity.

### 2. Spark Structured Streaming

- Reads raw Kafka events, parses the shared event schema, validates station/timestamp/demand/temperature, and enriches valid records with calendar features.
- Sends invalid records to the shared `electricity-dlq` topic instead of silently dropping them.
- Uses station-specific checkpoints for both featured and DLQ queries.
- Derives the Kafka connector Maven coordinate from the installed PySpark version and Scala binary unless explicitly overridden. This prevents the common Spark/connector version mismatch (for example, Spark 4.x with a Spark 3.5 connector). The Compose image pins Spark/PySpark 3.5.6 consistently.
- Reads `SPARK_MASTER_URL`, `SPARK_SHUFFLE_PARTITIONS`, `SPARK_MAX_OFFSETS_PER_TRIGGER` and `KAFKA_BOOTSTRAP_SERVERS` environment overrides.
- Emits micro-batch progress to logs.

**Boundary:** Spark currently performs distributed event validation and calendar enrichment. The model-specific rolling/lag features and 96 lead-time rows are still constructed in the predictor using the same Python feature contract used by training. This is intentional in this patch to avoid training/serving feature drift. Moving the full stateful lag feature contract into Spark is a separate, test-driven step; it should not be claimed as completed yet.

### 3. Cassandra operational storage

`config/storage.yaml` selects `csv` or `cassandra`. CSV remains the default for a safe local transition. The Compose stack sets `STORAGE_BACKEND=cassandra`.

Tables:

- `actuals_by_station_month`: partitioned by `(station_id, month_bucket)`, clustered by UTC timestamp.
- `latest_actual_month_by_station`: small pointer table so API reads are anchored to the latest replayed/live data month rather than the machine's wall-clock month.
- `forecasts_by_station_day`: partitioned by `(station_id, origin_date)`, clustered by forecast origin, run ID and horizon step.
- `latest_forecast_points_by_station_run`: stores the 96 points for a specific run.
- `latest_forecast_run_by_station`: pointer updated only after all points have been written. This prevents readers switching to a partially written latest forecast.
- `quality_events_by_station_day`: rejected input and quality warnings.

Actuals and forecast points use prepared statements and unlogged batches scoped to a single partition. Local single-node mode defaults to RF=1 and consistency ONE. The three-node Compose environment uses NetworkTopologyStrategy RF=3 and LOCAL_QUORUM.

The API reads actuals, forecast history, latest forecast and quality events from Cassandra when the backend is set to Cassandra. Historical CSV files are still merged into the actuals response for the existing dashboard timeline.

### 4. Reliability, quality and recovery

- Predictor auto-commit is disabled. It commits explicit offsets only after storage succeeds, or after an invalid event has been acknowledged by the DLQ.
- A storage/model exception stops the predictor without committing the failing offset; the event is replayed after restart.
- Actual writes are idempotent by `(station_id, utc_timestamp)` in Cassandra. Forecast writes are idempotent by their partition/clustering key and run ID.
- Spark rejects malformed/invalid raw events into DLQ. The shared `src.quality.dlq_consumer` persists those events to Cassandra (or JSONL in CSV mode).
- Timestamp gaps are recorded as quality warnings. The predictor resets its rolling history and waits for a new contiguous history window before forecasting.
- Prometheus counters/histograms expose received, accepted, rejected, forecast runs, forecast points and event-processing latency. Each station gets its own predictor metrics port starting at 9102.

**Delivery guarantee:** This is at-least-once processing with idempotent writes, not an end-to-end exactly-once transaction across Kafka and Cassandra. A crash between database write and Kafka offset commit can cause replay; Cassandra primary keys make the replay safe.

### 5. Orchestration and monitoring

- `docker-compose.scale.yml` manages 3 Kafka brokers, 3 Cassandra nodes, Spark master + 2 workers, API, dashboard, Prometheus and Grafana.
- The API's existing station process manager starts station-specific producer, Spark and predictor processes inside the API container. The quality DLQ consumer runs as a separate Compose service with restart policy.
- `scripts/cluster_control.py` provides `up`, `down`, `status` and `logs` commands.
- FastAPI exposes `/health/live`, `/health/ready` and `/metrics`. Streamlit Monitoring displays readiness, storage backend and recent quality events. Prometheus scrapes API and predictor metrics; Grafana is available for dashboards.

The local FastAPI process manager remains a lightweight supervisor for station child processes. It is not a full Kubernetes-style scheduler. Container-level restart policies cover the infrastructure and standalone quality consumer; station child-process auto-restart and distributed job rescheduling remain future hardening work.

## Setup A — Local single-node Cassandra (keep current Windows POC)

1. Install dependencies in the active virtual environment:

   ```powershell
   pip install -r requirements.txt
   ```

2. Start a local Cassandra development node (Docker Desktop):

   ```powershell
   docker run -d --name electricity-cassandra -p 9042:9042 cassandra:4.1
   ```

3. In `config/storage.yaml`, set:

   ```yaml
   backend: cassandra
   ```

   Keep the local defaults: host `127.0.0.1`, RF=1, consistency `ONE`.

4. Start FastAPI and the station pipeline as before. The Cassandra schema is created on first connection when `create_schema_on_start: true`. You can also initialize it explicitly:

   ```powershell
   python -m scripts.init_cassandra
   ```

5. Verify:

   - `GET http://127.0.0.1:8000/health/ready`
   - `GET http://127.0.0.1:8000/results/actuals?station=tennet&limit=100`
   - `GET http://127.0.0.1:8000/results/forecasts?station=tennet&limit=100`
   - `GET http://127.0.0.1:8000/results/quality/events?station=tennet`

To roll back, set `backend: csv` and restart predictor/API. Existing CSV outputs are not deleted.

## Setup B — Distributed learning cluster with Docker Compose

**Resource note:** Three Kafka brokers + three Cassandra nodes + two Spark workers are a meaningful local cluster. Use a machine with ample RAM (recommend 16 GB or more for comfortable experimentation). Reduce worker memory/replicas if your machine is constrained.

1. Ensure Docker Desktop is running and Compose v2 is available.
2. Ensure the repo contains the configured station input/history CSVs and model artifacts. They are bind-mounted into the application container.
3. Start the stack from the repository root:

   ```powershell
   python -m scripts.cluster_control up --build
   ```

4. Wait for Kafka and Cassandra health checks, then open:

   - Streamlit: `http://localhost:8501`
   - FastAPI docs: `http://localhost:8000/docs`
   - Spark Master UI: `http://localhost:8081`
   - Prometheus: `http://localhost:9090`
   - Grafana: `http://localhost:3000` (initial login commonly admin/admin; change it immediately)

5. Select a configured station in Streamlit and start its pipeline. The API container uses the internal Docker network endpoints for Kafka, Cassandra and Spark.

6. Check service state/logs:

   ```powershell
   python -m scripts.cluster_control status
   python -m scripts.cluster_control logs
   ```

7. Stop the stack (persistent named volumes are retained):

   ```powershell
   python -m scripts.cluster_control down
   ```

To remove the persistent test data as well, use `docker compose -f docker-compose.scale.yml down -v` only when you intentionally want to erase Kafka/Cassandra/Prometheus/Grafana volumes.

## Validation plan

Run the following before claiming the platform scales:

1. **Functional:** publish one station replay and verify raw → Spark featured → predictor → Cassandra → API → dashboard.
2. **Data quality:** publish invalid JSON, negative demand, missing temperature and a timestamp gap. Verify DLQ records, quality table/API and no forecast from a broken history window.
3. **Replay:** restart predictor after a forced stop. Verify it resumes from the last committed offset and does not duplicate actual/forecast rows.
4. **Kafka resilience:** stop one broker. With RF=3 and min ISR=2, writes should continue while two replicas remain. Restore broker and verify replication catches up.
5. **Cassandra resilience:** stop one node. With RF=3 and LOCAL_QUORUM, writes/reads should continue with two replicas. Restore node and verify it rejoins.
6. **Spark scale:** compare one worker versus two workers for the same input volume; measure input rows/sec, processed rows/sec and end-to-end latency.
7. **Load:** increase producer replay rate and record Kafka lag, Spark throughput, predictor latency, Cassandra write latency, CPU and memory.
8. **Recovery:** stop API, Spark worker, predictor process and Cassandra node separately; document which components auto-recover and which require restart.

Do not use synthetic throughput figures. Record measurements from your actual run.
