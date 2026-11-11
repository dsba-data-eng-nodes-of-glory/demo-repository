# Amprion-only runbook: October 2026 replay and monitoring

## 1. Expected monitoring behaviour

The current project configuration contains **Amprion only**.

- The demand chart shows a solid blue actual-demand history ending at the forecast origin and a dashed orange 96-step forecast.
- Forecast anomaly risk is scored on the **forecast trajectory only**. Historical actual demand is input context, never an anomaly target or highlighted interval.
- Exactly two high-demand forecast event types are highlighted directly on the orange forecast curve: **Demand level (historical baseline)** (`LEVEL_SPIKE`) and **Ramp-up** (`RAMP_UP_SPIKE`). Downward/low-demand events remain in the inference output but receive no chart indication, matching the supplied `classify_spike_types()` implementation.
- There are no separate anomaly subplot rows.
- Renewable monitoring is one chart with separate Solar PV and Onshore Wind rows. Each row has actual generation (solid line) and the latest complete model forecast (dashed line). Separate y-axes keep the solar curve visible beside the larger wind values.

## 2. Replay boundary: 1 October 2026

The station config now defines:

- History: `data/training/amprion/demand_train.csv`
- Stream/test input: `data/training/amprion/demand_test.csv`
- Expected first streamed timestamp: `2026-10-01T00:00:00Z`

The training file ends at **2026-09-30 23:45 UTC**. The test file starts at **2026-10-01 00:00 UTC** and is continuous at 15-minute resolution. The predictor uses the final 672 training observations as the seed window. Its first forecast origin is **2026-09-30 23:45 UTC**, therefore the first forecast target is **2026-10-01 00:00 UTC**.

The CSV producer now checks both the history/stream boundary and `expected_stream_start_utc`. This prevents accidentally starting the older January replay input.

## 3. Safe full replay

Before replaying, stop the Amprion producer, predictor and Spark processes. Start Kafka at `localhost:9092` and then run from the project root:

```bat
full_replay_amprion.bat
```

The script asks before making changes. On confirmation, it:

1. Backs up and clears Amprion `actuals.csv`, `forecasts.csv`, and `renewable_forecasts.csv`.
2. Clears old Amprion Kafka topic records and the Amprion Spark checkpoint (it does not delete model weights or source data).
3. Starts Spark, then the predictor using `demand_train.csv`, then the producer using `demand_test.csv`.
4. The predictor creates the initial demand, solar and wind forecasts from the 30 September history boundary before replaying the first 1 October event.

For the dashboard-driven path, use Pipeline Control's **Reset station pipeline** action first, then **Start station**. Reset is station-scoped and also clears the renewable forecast output file to prevent January runs from being selected after an October restart.

## 4. Forecast weather and renewable coverage

Demand future temperature comes from `data/raw/stations/amprion/demand_weather_forecast.csv`. Solar and wind future covariates come from their respective `forecast_*` columns in `solar_test.csv` and `wind_test.csv`. The serving code does not substitute future observed weather or future realised generation when building forecasts.

The supplied demand, solar and wind test/weather sources cover **2026-10-01 00:00 UTC through 2026-10-19 23:45 UTC**. To make a full 96-step forecast, the selected origin must have all 96 future weather timestamps available. Consequently, the last complete forecast origin supported by these supplied files is **2026-10-18 23:45 UTC**. Later origins are expected to be skipped until the weather files are extended.

The renewable predictor writes two 96-step runs at startup (solar and wind) and refreshes them every `forecast_every_steps` observations. Check `logs/amprion_predictor.log` for `Persisted renewable forecast rows=192` messages. A zero-row result is accompanied by a warning; model-specific reasons are logged by the renewable forecaster.

## 5. Verification and diagnostics

From the project root:

```bash
pytest -q tests/test_amprion_forecasting_integration.py tests/test_forecast_anomaly_scoring.py tests/test_forecast_anomaly_strips.py tests/test_forecast_selection.py
python -m scripts.test_demand_ae
python -m scripts.evaluate_renewable_holdout
```

The FastAPI endpoint `/results/renewables?station=amprion` returns:

- combined solar/wind actuals and source coverage;
- selected latest complete solar and wind forecast runs;
- the configured storage path, row counts, completeness and read errors.

The monitoring page displays an explicit warning when actual or forecast series do not fall inside the selected time window. When solar/wind data is missing, inspect the API `forecast_status`, the coverage dates and `logs/amprion_predictor.log` rather than assuming a Plotly rendering problem.

## 6. Dependency note

The saved demand anomaly bundle contains a scikit-learn `RobustScaler` created with scikit-learn 1.7.2. Use the package's pinned compatible range (`>=1.7.2,<1.8.0`) to avoid an `InconsistentVersionWarning` when loading the artifact.
