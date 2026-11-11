ACTUALS API PERFORMANCE PATCH
=============================

Patch target
------------
This patch was built against the uploaded electricity_mlop_final V1 source
archive (the exact backend/results_api.py and dashboard/pages/monitoring.py
versions included there). It deliberately changes only:

  backend/results_api.py
  tests/test_actuals_historical_cache.py   (new regression test)
  scripts/benchmark_actuals_api.py         (new HTTP timing tool)

No monitoring page, forecast selection, anomaly scoring, renewable API,
Kafka/Spark pipeline, station configuration, or model files are replaced.

Why this is needed
------------------
The Amprion station config points to demand_train.csv (166,464 rows) and
demand_test.csv (1,824 rows). The previous /results/actuals implementation
cached raw CSV reads, but still copied, parsed timestamps, concatenated,
sorted and de-duplicated the complete historical data on every request.
That repeated historical preparation was the main source of recurring latency.

What changes
------------
1. Adds a prepared historical-actuals cache, keyed by station and each source
   file's path, mtime_ns and size.
2. Parses day-first/mixed timestamps and numeric demand once per source version.
3. Stable-sorts and de-duplicates static source rows once, retaining the
   existing source precedence (history_file first, input_file second).
4. Applies the request cutoff and selects only the required historical tail
   before copying/merging with live actuals.
5. Retains live-over-historical precedence for duplicate timestamps.
6. Rebuilds the prepared cache automatically when a source file changes.
7. Keeps the API response shape unchanged; internal timestamp/priority columns
   are removed before JSON serialization.

Apply safely
------------
1. Stop the FastAPI process.
2. Back up your current backend/results_api.py.
3. Overlay the backend/results_api.py file from this patch bundle.
4. Add the two new files under tests/ and scripts/ respectively.
5. Run these checks from the project root:

   python -m py_compile backend/results_api.py scripts/benchmark_actuals_api.py
   pytest -q tests/test_api.py tests/test_actuals_historical_cache.py tests/test_amprion_forecasting_integration.py::test_historical_actual_api_uses_configured_history_file_for_first_forecast_origin

6. Restart FastAPI. The first /actuals request builds the cache; later requests
   in the same FastAPI worker reuse the prepared history.
7. Measure the real HTTP endpoint with:

   python scripts/benchmark_actuals_api.py --station amprion --limit 5000

   Add --end-time <forecast-origin-UTC> to match the Monitoring page request.

Expected behaviour
------------------
The first request after API process start still reads/prepares the static
source files. Repeated requests should be substantially faster until a source
file changes or the FastAPI worker restarts. Each worker maintains its own
in-process cache; your current one-worker Uvicorn setup is compatible.

Validation performed against the included V1 archive
----------------------------------------------------
- Before patch, direct calls to the same actuals route took approximately
  4.58s, 4.50s and 4.88s in this test environment.
- After patch, first call took approximately 4.54s, then repeat calls took
  approximately 0.05s and 0.04s.
- Targeted actuals/cache regression tests passed (3 passed).
- A broader adjacent test selection also surfaced four failures in existing
  anomaly-overlay / renewable-selector tests outside the files changed here;
  those were not modified by this patch and are not claimed as passing.

Important
---------
This patch was made against the V1 ZIP uploaded in this conversation. If the
live project directory has changed after that ZIP was created, compare the
current backend/results_api.py with the patch before applying it. No changes
are made to the user's local files by creating this bundle.
