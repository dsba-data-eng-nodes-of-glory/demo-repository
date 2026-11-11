# Station onboarding, component training and deployment

**Implementation stage:** Workbook onboarding + component-specific Model Management + training adapters + independent renewable serving.

## How to apply

Use the full updated project ZIP as a replacement for your working **code** only, or extract the patch ZIP onto your existing project root. **Before replacing anything, back up `config/stations.yaml`, `models/`, `data/`, `checkpoints/`, and `logs/`.** The patch ZIP only includes code/config/defaults/tests/template; it deliberately does **not** overwrite your station registry or customer model/data artifacts.

From your activated `electricity-spark` environment, run:

```bat
pip install -r requirements.txt
python -m uvicorn backend.main:app --reload --port 8000
```

In a second terminal in the **same project root**:

```bat
python -m streamlit run dashboard/app.py
```

Adjust backend port/URL if the existing setup already uses different settings. Do not reformat Kafka storage or delete Spark checkpoints.

## Station onboarding

1. Open **Station Management** and click **Download station training Excel template**. The backend also exposes `GET /stations/template`.
2. Replace sample rows in **Demand**, **Solar**, **Wind** with the station's training records. Keep the exact sheet headers. Provide one continuous 15-minute timeline on each selected sheet.
3. Select station ID, timezone, interval and upload the single Excel workbook (`.xlsx` or `.xlsm`). All four components start **checked**: `demand_forecast`, `solar_forecast`, `wind_forecast`, `demand_anomaly`. Uncheck any component you do not need.
4. The backend validates only the required worksheets for selected components; writes normalized per-component CSVs; keeps the source workbook; registers all component flags in `config/stations.yaml`.
5. **Data availability:** Demand forecast needs a meaningful chronological training/stream split. Renewable models need 672 historic steps, 96 forward steps, and chronological train/validation/test coverage. The autoencoder additionally requires at least three sufficiently represented calendar years and a minimum 30 days of validation data. Anomaly-only stations may be registered and trained but cannot be activated as demand streaming stations.

For renewables, forecast weather columns must contain forecasts known at the origin, never observed future weather. The supplied single value per valid timestamp is **not sufficient to reconstruct all old forecasts issued at different origins**; operational-grade backtesting requires weather forecast issue-time/vintage data and a point-in-time join.

## Training component algorithms

| Model component | Available algorithms | Training target | Evaluation |
| --- | --- | --- | --- |
| Demand | Dense NN, LSTM, Random Forest, XGBoost, GBT | Demand in MW, 96-step next-day horizon | MW MAE/RMSE/WAPE |
| Solar | Dense NN, LSTM, Random Forest, XGBoost, GBT | Photovoltaics in MWh per interval, 96 steps | MWh MAE/RMSE/WAPE |
| Wind | Dense NN, LSTM, Random Forest, XGBoost, GBT | Onshore wind in MWh per interval, 96 steps | MWh MAE/RMSE/WAPE |
| Demand anomaly | **Dense Autoencoder only** | Reconstruction loss for 96-step demand feature windows | Validation reconstruction loss, thresholds |

Solar/wind **share one hyperparameter profile per algorithm across stations** in `config/algorithm_configs.yaml`. The sequence model uses past generation/weather and forecast future weather, without exposing future observed generation. Tree models are one lead-aware estimator rather than 96 separate estimators. The demand Dense NN uses sklearn MLP with standardized features/target. The autoencoder reuses the original 96×10 feature/threshold algorithm.

Open **Model Management**, select the station, select **Model component to train**, select its algorithm, optionally edit/save hyperparameters, and click **Run training**. The job runs in the managed background process of the user's backend session and logs to station training logs. Jobs run one at a time per backend process to protect limited RAM. Each completed model is a **candidate** and gets separate metadata/registry entries. Saving an experiment does not replace a deployed model.

Use **Compare experiments** for models *within the same component*, and **Deploy selected model** to update that component's active checkpoint. Renewables can be deployed independently of each other and of demand. Stop the station predictor first, deploy, then restart the pipeline; existing demand streaming must have an active trained demand checkpoint. Anomaly deployment changes the inference bundle used by forecast-path monitoring.

## Important operational boundaries

- **15-minute frequency:** Solar, wind, autoencoder and their current 96-step/672-step feature contracts require it. Demand frequency may follow the existing contract.
- **Demand-driven streaming:** Kafka/Spark/CSV producer/predictor are started by the existing demand stream driver. A station with *only* solar or *only* wind can onboard, train and deploy, but cannot yet activate streaming without a demand component.
- **Renewable live weather:** Serving requires 96 target timestamps of **forecast** weather for each forecast origin. An uploaded historical workbook does not continually refresh future weather; a real-time weather forecast data feed must be integrated separately for sustained live operations.
- **Training metrics are not audited benchmarks:** Synthetic smoke checks verify execution and artifact compatibility only. Use real holdout data, weather forecast vintage and proper backtesting before making performance claims.
- **Version artifacts:** `models/<station>/experiments/<experiment-id>/...` are immutable candidate locations and registry metadata controls deployment.
- **ML lifecycle:** The default selection registers a component but does not automatically train or deploy it. Each component can be trained in any order.
- **Compatibility:** Existing Amprion-trained model files and station-config entries remain unchanged. Do not delete original model checkpoints.

## Verification performed

- New workbook onboarding tests, API validation/config/deployment tests and model-registry tests pass.
- Synthetic 1,850-row, 15-minute solar/wind datasets: **all 10 algorithm × component training-to-serving contracts** loaded and returned 96 forecasts. This tests functionality, not real-world accuracy.
- The original repository has four unrelated failing tests in the monitoring/startup and legacy renewable result helpers. They remain out of scope and are reported separately rather than silently modified.

## New and changed files

- `templates/Station_Training_Template.xlsx`; `src/ingestion/workbook_loader.py`
- `backend/station_api.py`, `backend/model_api.py`, `backend/pipeline_manager.py`
- `dashboard/pages/station_management.py`, `dashboard/pages/model_management.py`
- `config/algorithm_configs.yaml`
- `src/models/train_renewable.py` (new common renewable training runner)
- `src/models/train_tabular.py` (demand Dense NN)
- `src/models/run_experiment.py`, `src/models/model_registry.py`
- `src/models/streaming_predictor.py`, `src/models/renewable_forecaster.py`
- `src/models/training/amprion/solar_mlp_features.py` (ISO timestamp parsing correctness)
- `requirements.txt`; `tests/test_component_model_training_api.py` and prior workbook tests
