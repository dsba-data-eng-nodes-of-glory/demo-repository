# Nodes of Glory multi-station performance update

This update uses `Full Final Version.zip` as its baseline and uses the existing CSV and local SQLite storage profile. It retains the station registry, data files, trained models, Kafka topics and per-station Spark checkpoints from that archive.

## Apply on Windows

1. Stop the running FastAPI application and any Spark or station processes launched separately for this project. The updated FastAPI application owns the shared Spark worker.
2. Extract the updated full archive into a separate folder, or extract the upgrade ZIP into your current project root. When applying the upgrade, keep the values from your machine for `java_home`, the Kafka connector and other custom settings in `config/spark.yaml`. The shared worker is enabled by `shared_worker: true`.
3. From the project root in your existing environment, start FastAPI in one terminal and Streamlit in another.

   ```powershell
   conda activate electricity-spark
   $env:STORAGE_BACKEND = "csv"
   python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
   ```

   ```powershell
   conda activate electricity-spark
   python -m streamlit run dashboard/app.py
   ```

4. Use **Start all stations** and open **Monitoring → Germany**. The supplied archive contains Amprion and 50 Hertz. Register and deploy the additional stations with their own model artifacts, topics and hierarchy entries for a four-station run. Germany aligns displayed totals to the slowest reporting station and shows the number of stations contributing to each metric.

## Improvements

- Germany reuses the same calculation for overlapping dashboard requests. The brief cache expires after one second. Rollups reuse parsed UTC timestamps, check forecast grids with integer arithmetic and read bounded tails from station-owned CSVs. The dashboard keeps the last successful view during a timeout and displays its age.
- A single Spark JVM hosts the station queries. Each station retains its own input topic, output topic and checkpoint. Stopping one station removes only its query. A failed query can retry independently, and the API supervises the shared process.
- Spark publishes valid features and rejected records in one Kafka write per batch by setting each record’s output topic. The source checkpoint advances after the write completes.
- Activation starts the worker processes without fixed waits. Java version checks are reused, and station inference defaults to one numerical-library thread per predictor. Initial Demand forecasts can be generated from the approved lookback while Spark starts. Live forecasts still advance from station observations.
- Log views read the end of each file. Shared Spark messages appear under `logs/spark_shared.log` with station identifiers. Predictor and producer logs are still station-specific.

## What the uploaded logs showed

- A cold 50 Hertz predictor startup reported 108.64 seconds. A later run reported 13.97 seconds.
- In the slower run both Spark workers reported heartbeat and BlockManager errors around 20:12:31. Their streaming queries reported startup at 20:13:04. A port conflict also caused Spark to retry the UI port.
- The baseline code used separate JVMs, two Kafka writes per micro-batch and fixed waits during single-station activation. Germany read and processed recent forecast runs for each refresh. The archive did not contain a FastAPI request log, so the Germany bottleneck was reproduced with a controlled API benchmark.

## Verification

The four-stream integration test used the two supplied stations plus two isolated replay clones in Linux with Kafka 4.1.2, Spark 3.5.6, Java 17 and the archive’s models. The clones are test fixtures. All four used the configured two-second producer interval. Test timings describe that environment.

| Check | Result |
| --- | --- |
| Start all four | 2.71 seconds to the activation response |
| First Demand forecasts | 3.54 to 4.54 seconds from the start request |
| Spark | Four station queries registered in one Spark worker |
| Germany live API | 40 successful polls, 0.21 seconds p95 |
| Longer four-station replay | 80 successful polls, 0.37 seconds p95, and 42 intervals persisted per station |
| Forecast alignment | 96 points with 4 of 4 contributors for Demand, Solar, Wind Energy and Residual Load |
| Quality routing | An off-grid event reached the dead-letter quality log |
| Station isolation | Stopping one query left the other three running in the same worker |

A separate four-station, skewed-CSV HTTP benchmark returned identical response payloads with the original and updated code. Median fresh response time went from 0.67 to 0.25 seconds. The slowest response in a four-request burst went from 3.03 to 0.33 seconds. Validate throughput on your Windows machine with your real station inputs and disk before relying on the lab timings.
