# Streaming reliability and performance patch

## Changes

1. **Historical context resolution**
   - `/results/actuals` now loads the configured station `history_file` (for Amprion, `data/training/amprion/demand_train.csv`) and the configured `input_file`.
   - Legacy stations without `history_file` retain the derived `*_history.csv` fallback.
   - UTC parsing now supports both European day-first source timestamps and ISO Kafka timestamps. This avoids dropping valid dates such as `30-09-2026` during chart/anomaly context assembly.

2. **Producer restart safety**
   - `csv_producer` saves the last Kafka-acknowledged source timestamp at `data/state/<station>/producer_progress.json`.
   - Normal restarts resume strictly after that timestamp. The cursor must still be present in the source CSV; changed/replaced sources fail closed rather than silently skipping or replaying data.
   - `--replay` discards the cursor and starts from the first source row. Use it only after resetting the matching Kafka topics and result outputs.
   - An OS-level per-station lock prevents two local producer processes from publishing the same station concurrently. The lock is automatically released when the process exits.
   - The producer now uses the station's configured `history_file` when no `--history-file` override is provided.

3. **Reset integration**
   - Dashboard `reset_station_pipeline()` removes the producer cursor along with the selected station's Kafka records, outputs and Spark checkpoints.
   - `full_replay_amprion.bat` clears the Amprion producer cursor and starts the producer with `--replay` after it clears Kafka and result outputs.

4. **CSV persistence efficiency and idempotency**
   - Actual and forecast key indexes are cached in process and rebuilt only when the backing CSV signature changes.
   - Forecast CSV schema checks read only the header during normal appends; full-file migration occurs only when the schema actually needs upgrading.
   - A retried demand forecast with the same `forecast_run_id` and horizon step is not appended twice.
   - Renewable forecast duplicate keys are cached in the same way instead of scanning the full renewable CSV for every run.

5. **Forecast input and feature efficiency**
   - Forecast-weather lookups are cached by path, modification time and file size, avoiding a full weather CSV read on every Dense demand inference.
   - Renewable train/test frames remain sorted/unique after initial load; the per-origin context extractor no longer sorts and copies the entire multi-year frame for every forecast.
   - Amprion demand forecast cadence is configured to every 4 observations (hourly at 15-minute resolution). Set `forecast_every_steps: 1` in `config/stations.yaml` when a new 24-hour forecast is required every 15 minutes.
   - Actual CSV storage is batched up to 50 rows or 5 seconds, unless a forecast event forces a flush first.

6. **Spark execution**
   - Featured records and DLQ records are now written from one `foreachBatch` callback and a single Kafka source query. The parsed micro-batch is cached for both sink writes, removing the duplicate source query, consumer group and independently evaluated parsing plan.
   - Delivery remains at-least-once if one sink succeeds and the other fails. Predictor timestamp de-duplication and DLQ source-offset idempotency handle retries.

## Apply / run

1. Stop the station's Spark, predictor and producer processes.
2. Back up the station output folder and current Spark checkpoints.
3. For an intentional clean replay, use **Stop + reset** in Pipeline Control, then start the station. That reset clears the matching Kafka records, output files, Spark checkpoints and producer cursor. `full_replay_amprion.bat` performs the equivalent Amprion holdout reset.
4. For an ordinary process restart without resetting Kafka or outputs, start the producer normally. It resumes from its saved acknowledged cursor; do not pass `--replay`.

The patch does not change model weights, feature definitions, the 672-to-96 forecast contract, or the anomaly scorer's point-in-time rules.
