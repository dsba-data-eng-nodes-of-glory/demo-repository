# Station workbook onboarding — backend contract

## Endpoint

`POST /stations/onboard` accepts a multipart form upload containing one `.xlsx` or `.xlsm` workbook. The backend stores the source workbook and normalized model-specific CSVs below `data/raw/stations/<station_id>/`.

| Form field | Required | Default | Purpose |
|---|---:|---|---|
| `station_id` | Yes | — | Station key; normalized to lowercase letters, numbers, `-` and `_` |
| `file` | Yes | — | Station workbook, maximum 100 MB |
| `display_name` | No | Uppercase station key | User-facing name |
| `stream_fraction` | No | `0.20` | Demand forecaster's final stream tail; accepted range 0.10–0.50 |
| `timestamp_format` | No | `auto` | Timestamp parsing hint |
| `source_timezone` | No | `UTC` | Timezone used for timestamps without an explicit UTC offset |
| `interval_minutes` | No | `15` | Source observation interval |
| `selected_models` | No | All four | Comma-separated IDs or JSON array; see below |
| `raw_topic` | No | `<station>-raw` | Kafka raw topic |
| `featured_topic` | No | `<station>-featured` | Kafka featured topic |

An empty or omitted `selected_models` field enables all four components:

- `demand_forecast`
- `solar_forecast`
- `wind_forecast`
- `demand_anomaly`

Example form value to deselect solar and anomaly while retaining demand and wind:

`["demand_forecast", "wind_forecast"]`

Only the worksheets required by selected models are parsed. The Demand worksheet is shared by the demand forecaster and anomaly autoencoder.

## Worksheet contract

### Demand

Required for `demand_forecast`: `utc_timestamp`, `demand`, `temperature`.

Required for `demand_anomaly` only: `utc_timestamp`, `demand`. Temperature is not required when demand forecasting is deselected. Common alternatives such as `timestamp`, `datetime`, `demand_mw`, `load_mw`, `temperature_c` and `temp_c` are normalized to the canonical names.

The backend writes:

- `<station>_demand_training.csv` — chronological training prefix for the existing demand forecaster.
- `<station>_demand_history.csv` — final configured history window before the simulated stream.
- `<station>_demand_stream.csv` — final `stream_fraction` tail.
- `<station>_demand_anomaly.csv` — full demand history with the anomaly trainer's expected headers `timestamp,demand_mw`.

The anomaly file is written only when `demand_anomaly` is selected.

### Solar

Required columns are the seven headers used by the existing solar Dense NN feature builder:

`utc_timestamp`, `Photovoltaics [MWh]`, `cloud_cover_mean_pct`, `shortwave_radiation_mean_prev_hour_w_m2`, `forecast_cloud_cover_mean_pct`, `forecast_shortwave_radiation_mean_prev_hour_w_m2`, `Photovoltaics_installed_capacity [MW]`.

The complete normalized series is written to `<station>_solar_data.csv` and registered as the solar model's training data. The model training script retains its chronological 70/15/15 split.

### Wind

Required columns are:

`utc_timestamp`, `Wind onshore [MWh]`, `wind_speed_100m_mean_ms`, `wind_direction_100m_circular_mean_deg`, `Wind_onshore_installed_capacity [MW]`, `future_wind_speed`, `future_wind_direction`.

The complete normalized series is written to `<station>_wind_data.csv` and registered as the wind model's training data. The model training script retains its chronological 70/15/15 split.

Future weather inputs must represent forecasts available at the forecast origin. The backend can validate numeric values and coverage, but cannot independently prove forecast-vintage provenance from the workbook alone.

## Validation behavior

- Required sheets and columns are checked only for selected models.
- Timestamps are normalized to UTC. Naive timestamps use `source_timezone`.
- Duplicate timestamps, invalid timestamps, non-numeric values, non-finite values and irregular time gaps are rejected with worksheet-specific errors.
- Demand and generation cannot be negative. Solar cloud cover must be 0–100%, wind direction 0–360°, and installed capacity must be positive.
- Solar, wind and demand-anomaly models currently require 15-minute data.
- Solar and wind need sufficient rows for the existing 672-step history and 96-step horizon with their chronological 70/15/15 split.
- Demand anomaly training additionally needs at least 30 days in its 10% validation tail and three sufficiently represented training years (at least 180 days per year in the first 90% of the history), as required by its current annual-trend baseline.
- Upload processing uses a staging folder. Invalid workbooks do not create a station registry entry or leave a partial station data folder.

## Station registry additions

New stations contain `enabled_models` and `model_components`. Every model component has an independent `enabled` and `status` field. Selected components start as `pending_training`; deselected components start as `deselected`.

Renewable source paths and model artifact paths are stored under `renewables.solar` and `renewables.wind`. The upgraded serving adapter loads solar and wind **independently**, so `renewables.enabled` is true whenever either component is selected. Selected but untrained renewable components are skipped safely until a valid model checkpoint is deployed. Streaming activation still requires a demand forecasting component.

Onboarding registers inputs and paths. The later Model Management integration (documented in `docs/STATION_COMPONENT_TRAINING_README.md`) supports separate training and explicit deployment; onboarding does not automatically launch model training or start Kafka/Spark.
