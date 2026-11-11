# Amprion forecasting and monitoring integration

## What is now integrated

The Amprion station has a 15-minute, 672-step history (7 days) to 96-step (24-hour) multi-output demand model. The supplied Dense NN trainer and its matching feature builder are retained under `src/models/training/amprion/`. The predictor loads the supplied weights and metadata from `models/amprion/demand_dense_nn/`.

Future covariates are sourced from forecast columns only:

| Model | Historical context | Future weather used by the serving feature builder |
|---|---|---|
| Demand | Observed demand and observed temperature | `forecast_temperature` |
| Solar PV | Historical solar generation, observed cloud cover and observed radiation | `forecast_cloud_cover_mean_pct`, `forecast_shortwave_radiation_mean_prev_hour_w_m2` |
| Onshore wind | Historical generation, observed wind speed and observed wind direction | Forecast wind speed and forecast wind direction columns |

For future horizon rows, target values are placeholders and are not model inputs. Missing or incomplete future-weather coverage suppresses that forecast run. The demand predictor does not replace missing future forecast temperature with the latest observed temperature. Renewable predictions are isolated from the demand stream: a solar or wind failure is logged and does not terminate demand forecasting.

The serving feature builders are the same builders used by the supplied training scripts. This keeps feature ordering, lag definitions, weather selection and scaling aligned between training and inference.

## Metrics recorded in the supplied model artifacts

The model-artifact metrics below are copied from the supplied checkpoints/metadata. Solar and wind October holdout metrics are computed separately using 19 daily 23:45 UTC forecast origins. Future realised generation is masked before serving feature construction; it is used only to evaluate predictions. Demand metrics are stored in target-scaled units in the demand checkpoint metadata.

| Model | Test metric | Supplied result |
|---|---|---:|
| Demand Dense NN | RMSE (scaled target) | 0.3000 |
| Demand Dense NN | MAE (scaled target) | 0.2229 |
| Solar MLP | MAE | 89.2490 MWh |
| Solar MLP | RMSE | 154.7806 MWh |
| Solar MLP | WAPE | 12.9141% |
| Solar MLP | R² | 0.9732 |
| Wind Dense NN | MAE | 134.1819 MWh |
| Wind Dense NN | RMSE | 183.1859 MWh |
| Wind Dense NN | WAPE | 22.4577% |

October holdout evaluation (19 daily origins; 1,824 predicted 15-minute points per model):

| Model | MAE (MWh) | RMSE (MWh) | WAPE | R² | Bias (MWh) |
|---|---:|---:|---:|---:|---:|
| Solar PV | 159.9810 | 282.5333 | 31.9373% | 0.8742 | -103.7653 |
| Onshore wind | 116.3707 | 157.8378 | 18.2684% | 0.9352 | -44.1778 |

Solar's supplied weekly baseline metadata records MAE 237.8563 MWh, RMSE 461.4358 MWh and WAPE 34.4171%. These scores describe different evaluation sets and should not be compared directly across different generation targets or datasets.

## Dashboard additions

`dashboard/pages/monitoring.py` includes:

- The actual demand line (blue) and latest demand forecast (orange dashed).
- Exactly two anomaly strips directly below the demand chart: **Demand level (historical baseline)** and **Ramp-up**. They use the supplied `classify_spike_types()` logic (`LEVEL_SPIKE` and `RAMP_UP_SPIKE`) and are scored on the 96-step forecast trajectory only. Actual demand is used solely as preceding context and is never labelled. Normal/insufficient timestamps are blank; there is no generic third strip.
- One combined renewable monitoring panel with two vertically aligned rows: Solar PV and Onshore Wind. Each row has its own readable y-scale and can show actual generation as a solid line and the latest stored forecast as a dashed line. The rows share the UTC time axis with each other and the demand view.
- Renewable coverage information showing actual timestamp coverage, whether the separate test file exists, and how many solar/wind actual and forecast points are being drawn.

The screenshot supplied for review showed the demand graph and only the renewable section heading/caption; it ended before the renewable chart area. The updated panel now displays Solar and Wind in distinct rows, avoiding the smaller solar series being visually compressed by the wind range.

The endpoint is `GET /results/renewables?station=amprion&limit=5000`. It merges configured Amprion train/test CSVs for realised generation and reads the latest renewable forecast run from `data/processed/stations/amprion/renewable_forecasts.csv`. Actual curves can appear when the train CSVs are present. Dashed forecast curves appear only after the Amprion predictor has generated and persisted the 96-step solar/wind forecast run.

## Current supplied data coverage

The uploaded material includes:

- `data/training/amprion/demand_train.csv`
- `data/training/amprion/demand_test.csv`
- `data/training/amprion/solar_train.csv`
- `data/training/amprion/solar_test.csv`
- `data/training/amprion/wind_train.csv`
- `data/training/amprion/wind_test.csv`

It now includes `solar_test.csv` and `wind_test.csv` at `data/training/amprion/`. The files extend renewable actual and forecast-weather coverage from 2026-10-01 00:00 through 2026-10-19 23:45 UTC. The serving adapters and renewable actual-history API merge them with the corresponding training files automatically. Run `python -m scripts.evaluate_renewable_holdout` to reproduce the daily-origin holdout evaluation and per-horizon outputs.

The demand weather lookup file is generated at `data/raw/stations/amprion/demand_weather_forecast.csv` from the supplied forecast-temperature column in the demand training and test inputs. It contains 168,288 unique 15-minute timestamps from 2022-01-01 00:00 UTC through 2026-10-19 23:45 UTC. Its timestamps are normalized to UTC and the file is validated for duplicate rows and cadence gaps during packaging.

**Forecast-vintage limitation:** the supplied CSV columns are keyed by valid timestamp but do not contain a forecast issue/creation timestamp. The serving code uses the supplied `forecast_*` values for future valid times, never the corresponding observed weather columns. A strict historical point-in-time backtest (using the exact forecast vintage that was available at each historical origin) requires forecast issue-time/vintage data; that provenance is not present in the uploaded files, so no such backtest claim is made here.

## Paths and commands

Run commands from the repository root in the project's Python environment.

### Re-train demand forecast model

```bash
python -m src.models.training.amprion.Dense_NN_Forecast_Training
```

Default input: `data/training/amprion/demand_train.csv`  
Default output: `models/amprion/demand_dense_nn/`

### Re-train solar MLP

```bash
python -m src.models.training.amprion.train_solar_mlp
```

Default input: `data/training/amprion/solar_train.csv`  
Default output: `models/amprion/solar_dense_nn/`

### Re-train wind Dense NN

```bash
python -m src.models.training.amprion.Amprion_Wind_Dense
```

Default input: `data/training/amprion/wind_train.csv`  
Default output: `models/amprion/wind_dense_nn/`

### Re-train demand anomaly autoencoder

```bash
python -m src.models.anomaly.train_demand_ae --input data/training/amprion/demand_train.csv --timestamp-col utc_timestamp --demand-col demand
```

Default output: `models/amprion/demand_anomaly/`

Retraining replaces the files in the respective model output directory. Back up deployed checkpoints before retraining. Update model version tags in `config/stations.yaml` after promoting new artifacts.

### Enable the Amprion station

The packaged configuration preserves the station's existing stopped state (`enabled: false`). Enable it from the existing station controls or set `stations.amprion.enabled: true` in `config/stations.yaml` before starting its producer/Spark/predictor workflow. Renewable forecast generation is controlled by `stations.amprion.renewables.enabled` and is currently enabled in the supplied configuration.

## Validation performed for this package

- Unit/integration suite: 48 passed (two scikit-learn unpickle-version warnings because the environment here uses 1.8.0 while the supplied scaler was serialized with 1.7.2).
- Python syntax compilation: passed for `backend`, `dashboard`, `src`, `scripts` and `tests`.
- Demand Dense inference smoke test: 96 finite non-negative predictions generated using future `forecast_temperature` values.
- Solar and wind inference smoke test: 96 solar and 96 wind forecast points generated using forecast weather covariates.
- Demand anomaly scorer smoke test: one score/classification record returned per supplied timestamp; 96-step forecast scorer and both component strip outputs verified.
- FastAPI `/results/renewables` smoke test: HTTP 200, station-specific actual history and test-file coverage reported.
- October renewable holdout: 19 daily 96-step origins; Solar MAE 159.98 MWh, RMSE 282.53 MWh, WAPE 31.94%, R² 0.8742, bias -103.77 MWh; Wind MAE 116.37 MWh, RMSE 157.84 MWh, WAPE 18.27%, R² 0.9352, bias -44.18 MWh.
- No live Kafka/Spark/Cassandra cluster or Docker Compose run is claimed as part of these model/dashboard checks.

The local test environment had scikit-learn 1.8.0 while the supplied anomaly scaler was serialized with scikit-learn 1.7.2. `requirements.txt` pins the compatible minor version range (`>=1.7.2,<1.8.0`) for clean installations. Reinstall dependencies from the updated requirements before running the dashboard in a fresh environment.
