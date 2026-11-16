# Residual Load — Forecast Origin Alignment Fix

## What was broken
- Demand generates a complete forecast at each 15-minute origin. The predictor normally persists renewable/residual scoring less frequently (approximately hourly).
- Monitoring requested a saved residual assessment for the *exact* newest Demand origin, so it showed "Assessment is being prepared" in the intervening 15-minute steps even though valid recent assessments existed.
- The cached API only inspected the newest snapshot for an origin (or globally). Existing database contains multiple calibration/forecast fingerprints for an origin, so the first selected one may not match active station thresholds.
- Model Analysis could pick an unrelated latest-origin record from a previous replay.

## Changes
- `src/models/residual/snapshots.py`: new bounded read-only `candidate_snapshots` query. Uses an <=origin window to avoid selecting future assessment, newest-on-disk insertion order when no origin is supplied, and does no model inference.
- `backend/results_api.py`: adds optional `lookback_minutes` (0-240), selects the first candidate *compatible with active station calibration*; distinguishes missing/mismatched snapshots from API failures, includes `snapshot_lag_minutes` and `is_exact_origin`.
- `dashboard/pages/monitoring.py`: asks for up to a 60-minute earlier scored origin; displays the latest completed residual graph with the age labelled. Anomaly intervals are overlaid on the latest Demand graph ONLY when forecast origins exactly match. In-between, avoiding overlay prevents falsely attributing previous-origin predictions to the newest Demand forecast.
- `dashboard/pages/model_management.py`: retrieves current Demand origin, then uses the same 60-minute bounded lookup for operational anomaly analysis.
- `tests/test_nonblocking_monitoring_api.py`: new tests for stale/future results, historic replay, threshold compatibility, and hourly scoring versus 15-minute Demand frequency.

## Install
1. Stop Streamlit and FastAPI. Predictor can remain running. Back up these four Python source files.
2. From this ZIP, copy `backend/`, `src/`, and `dashboard/` into your project root, preserving relative paths. This does not replace checkpoints, station YAML, training data, or SQLite databases.
3. Restart FastAPI from the same project root and restart Streamlit. No model retraining or SQLite reset.
4. Verify `curl.exe --max-time 10 "http://127.0.0.1:8000/results/residual/forecast?station=amprion&cached_only=true&origin_utc=<ISO-origin>&lookback_minutes=60"` (URL-encode + if necessary).
5. In Monitoring, a residual graph with `snapshot_lag_minutes` 0..60 should appear when there is a compatible saved origin in that window; if no such origin exists it explains that explicitly.

## Calibration requirements
Saving developer thresholds **before** the pipeline is correct. Scoring uses `residual_detection_settings` first. Results from old thresholds are NOT silently reclassified, so wait until a compatible scored origin is saved, or regenerate on the predictor. The patch only fetches saved records; no expensive autoencoder execution is added to API GET.

## What is not changed
- No changes to the residual autoencoder/checkpoints or to forecast cadence (important for an 8 GB Windows machine).
- No overlays across different forecast origins; that would mix incompatible model vintages. If a marker must appear every 15 minutes, re-evaluate renewable/residual scoring cadence and system load separately.
- No guarantee of live performance on Windows Kafka/Spark; source level only.

## Tests
Focused: 21 passed. Full suite before final UI origin alignment change: 149 passed, 4 failed (the same four old static Monitoring/refresh tests from the prior project). Model Analysis file change was syntax checked and focused tests rerun.
