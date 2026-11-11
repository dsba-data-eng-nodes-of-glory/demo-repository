# Model Management — Component Retraining + New Data Upload (V2)

## What changed

The *original* `amprion` entry in `config/stations.yaml` uses the legacy registry format. It already provides demand history, solar/wind `train_file` paths and an anomaly checkpoint, but it **does not define** `model_components`. The previous `/models/components` endpoint therefore returned **demand-only** for this station. This patch detects those legacy paths at read time, without editing or re-registering Amprion.

Model Management now has one common **Model component to manage / retrain** dropdown above all four tabs. Selections are Demand forecasting, Solar forecasting, Wind forecasting and Demand anomaly autoencoder. Training, comparisons, registry/deployment and analysis all use the selected component. For a newer station, an explicit onboarding deselection remains disabled and is shown as such; it is not silently overridden.

**Algorithms:** Demand, Solar and Wind expose Dense NN, LSTM, Random Forest, XGBoost and GBT. Demand anomaly exposes **Dense Autoencoder only**. Solar/Wind share algorithm hyperparameter defaults across stations.

## Retraining an existing model

1. Back up `config/stations.yaml`, `models/`, `data/`, `checkpoints/` and custom dashboard files.
2. Stop FastAPI, Streamlit and any station training processes. Keep your station data, registry and trained models.
3. Extract `Station_Component_Retraining_V2_Patch.zip` into your current project **root**, preserving folders and replacing only files included in the patch. The new `backend/training_data.py` must be present. Do **not** replace your `config/stations.yaml`.
4. In your `electricity-spark` environment, install dependencies from the existing `requirements.txt` if required; this patch introduces no new dependency beyond the project's installed FastAPI, pandas and openpyxl.
5. Restart FastAPI (one worker) and Streamlit. Refresh the browser with Ctrl+F5 if the old CSS persists.
6. Open **Model Management** → select Amprion → choose the desired component in the common dropdown. All four should now be available for the existing Amprion station.
7. Select the algorithm and modify parameters as desired.
8. For a normal retrain choose **Registered historical dataset**. For a new complete historical snapshot, open **Upload and validate new historical training data**, upload a file, click **Validate & save dataset**, and select the new snapshot in **Dataset for this training run**.
9. Click **Run training**. Watch the training service logs. Review candidate metrics under Model Management and use **Model Registry & Deployment** to deploy an approved version. Stop the predictor before deploying and restart the station pipeline afterwards.

## Upload schemas

A CSV may contain data for **the selected component only**. Alternatively, upload the single Excel workbook used for new station onboarding; the server reads only the relevant **Demand**, **Solar** or **Wind** worksheet. Download the workbook via `/stations/template`, or the template link on the training form.

* Demand forecasting: `utc_timestamp,demand,temperature` (temperature is needed by the existing demand model).
* Demand anomaly: `utc_timestamp,demand` (or CSV `timestamp,demand_mw`). The backend normalizes to `timestamp,demand_mw`. The anomaly baseline requires three sufficiently represented calendar years of training data and sufficient validation coverage.
* Solar: `utc_timestamp,Photovoltaics [MWh],cloud_cover_mean_pct,shortwave_radiation_mean_prev_hour_w_m2,forecast_cloud_cover_mean_pct,forecast_shortwave_radiation_mean_prev_hour_w_m2,Photovoltaics_installed_capacity [MW]`.
* Wind: `utc_timestamp,Wind onshore [MWh],wind_speed_100m_mean_ms,wind_direction_100m_circular_mean_deg,Wind_onshore_installed_capacity [MW],future_wind_speed,future_wind_direction`. Existing forecast-column aliases `forecast_wind_speed_100m_mean_ms` and `forecast_wind_direction_100m_circular_mean_deg` are also accepted for CSV uploads.

CSV data must have regular timestamps at the station's configured interval with no gaps, duplicate timestamps, blank required columns, invalid numeric values, or invalid physical ranges. Existing Solar, Wind and Anomaly trainers currently require **15-minute** observations. Renewable trainers use a 672-step history and a 96-step horizon. For renewable *future weather* inputs, only the weather forecast available at the forecast issue/origin time is valid; observed future weather would leak information.

**An uploaded snapshot must contain complete continuous history for that model.** The upload is not a partial append and is not merged automatically with earlier data. There is a 100 MB HTTP upload cap. New sources are stored under `data/training_uploads/<station>/<model_component>/<source_id>/`; each keeps its normalized CSV, source file and manifest with SHA-256, dates and row count. The training request selects sources by an ID scoped to station/component. Data integrity is verified before training. On station deletion, its training uploads are removed with its other files.

**No active data is overwritten on upload:** `data/raw`, original demand replay files, deployed checkpoints, streaming output and the station registry are left intact. The experiment config carries the new data path and checksum. Training creates a separate candidate. Deploying that candidate changes the model assignment but does *not* automatically switch live Kafka/producer inputs or the live future-weather feed to the training snapshot. Future-weather availability and the streaming timeline must still be verified before accepting a retrained model for live operation.

## API additions

* `GET /models/components?station=amprion`: includes inferred enabled components, registered source availability and checkpoint availability for both legacy and new stations.
* `POST /models/training-data/upload` multipart fields: `station`, `component`, `file`, optional `timestamp_format` (`auto` default), `source_timezone` (`UTC` default).
* `GET /models/training-data?station=amprion&component=solar_forecast`: lists successfully validated uploaded snapshots without scanning their entire hashes each refresh.
* `POST /models/experiments/train`: retains the prior JSON body but optionally accepts `training_data_id`. If omitted, uses the station's configured historical dataset. On legacy Amprion this correctly selects `data/training/amprion/demand_train.csv`, rather than its replay/test CSV.

Experiment records and metadata include `training_data_path`, `training_data_id` and `training_data_sha256` for training lineage. Experiment comparison warns when dataset hashes differ. Existing records without these fields are still readable.

## Dashboard contrast corrections

`dashboard/ui.py` now applies shared high-contrast label, caption, tab, expander-header and file-uploader styling throughout Home, Stations, Models, Pipeline and Monitoring. Form values remain dark on their light input fields. The original screenshot showed a pale expander header with nearly white text and low-contrast labels; this stylesheet addresses both. Exact appearance still depends on the installed Streamlit version and the browser theme, so validate the rendering locally after restart.

## Verification and known limitations

* **122 tests passed; 4 pre-existing unrelated tests failed** in the complete repository suite. Existing failures: two assertions for older forecast/renewable chart behavior, missing `_latest_renewable_run` in `results_api.py`, and two obsolete monitoring source-string assertions (four failures in total).
* Additional focused HTTP tests cover legacy Amprion model discovery, upload normalization and isolation, snapshot selection, checksum mismatch detection, out-of-range rejection, trainer source-path propagation and anomaly legacy-data adaptation.
* A real synthetic-data training smoke test succeeded for **both solar and wind Random Forest**, using datasets uploaded through the new validation function and producing versioned model artifacts. These synthetic scores do not establish operational accuracy.
* **Not verified:** Streamlit pixel rendering in your Windows/Streamlit installation, full Kafka→Spark→Predictor streaming with a newly retrained checkpoint, and live forecast weather feeds. Test these on one station before activating new models for all stations.
