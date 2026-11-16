# Residual Load + Renewable Monitoring API Timeout Fix

## Problem
The previous monitoring page recomputed the residual Dense Autoencoder in a synchronous
GET /results/residual/forecast request, competing with predictor-side scoring and
blocking the browser for up to 30 seconds. The renewable results API has a first-load
CSV history cache that can take >5 seconds on older Windows machines/HDDs.

## What changed
- Monitoring and Model Analysis now request `cached_only=true` for residual forecasts.
  This fetches an assessment **already saved by the predictor** from SQLite,
  without running any ML model or loading historical CSVs in the API request.
  The origin is checked so stale assessments are not overlaid on a newer Demand forecast.
- Snapshot reads now open a read-only SQLite connection (250 ms lock timeout) rather than
  trying to update the SQLite schema on each HTTP poll.
- `/results/renewables?defer_heavy=true` preloads reference history on a bounded background
  executor and responds immediately with `pending: true` while warming.
  Existing API callers retain their previous synchronous behavior by default.
- Monitoring shows a preparing-data status, not a timeout or false missing-data message.
- No Demand/Solar/Wind or residual model training, checkpoint or scoring changes.

## Installation
1. STOP FastAPI and Streamlit (do not delete data files or Kafka checkpoints).
2. Back up the four changed source files. Extract this ZIP in your existing
   Residual Load Spike Detection project root, preserving subfolders.
3. Start FastAPI and Streamlit from **the same project directory**.
4. Start the station predictor if it is not already running: it is responsible for
   scoring new residual forecast origins and persisting assessments.
5. Leave `config/storage.yaml` set to `backend: csv` if Cassandra is not active.

## Verification on Windows
```
curl.exe --max-time 5 http://127.0.0.1:8000/health/live
curl.exe --max-time 5 "http://127.0.0.1:8000/results/residual/forecast?station=amprion&cached_only=true"
curl.exe --max-time 5 "http://127.0.0.1:8000/results/renewables?station=amprion&defer_heavy=true"
```
- For residual scoring, `pending: true` means the selected origin hasn't been scored and
  saved yet; wait for the predictor. It does not mean the FastAPI service is down.
- For renewables, the first call can say `pending: true`, and a later call should return
  actual and forecast records once the history cache is warm.
- If `/health/live` itself times out, check whether FastAPI is running and responsive.
  Check CPU/RAM/disk load, broker failure or a process crash; the dashboard patch cannot
  repair an unavailable server.

## Tests
Latest source-level regression: 144 passed, 4 failed. The four failures are
legacy Monitoring assertions for obsolete start-boundary rendering and the earlier
2-second default refresh, unchanged from the previous version. New focused tests pass.
The Kafka/Spark Windows runtime and Streamlit browser could not be exercised here.

## Caveat
Saved assessments are intentionally not re-scored on each GET; if thresholds are
changed, the predictor must score the origin with the new configuration before a
new result appears. Do not mistake an earlier stored assessment for a current risk
judgment. No spike/alert detection accuracy change is claimed.
