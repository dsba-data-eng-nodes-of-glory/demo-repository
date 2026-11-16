# Version 6 station weather automation

## Apply on Windows

1. Stop the station pipelines and the FastAPI/Streamlit processes. Back up the existing project.
2. Open the existing Version 6 project root, the folder with `backend`, `config`, `dashboard`, `src` and `templates`.
3. Merge the ZIP's `overlay` folder into that project root and replace matching files. Alternatively, extract the ZIP outside your project and run `git apply --check "<extracted-folder>/Version_6_Automatic_Weather.patch"` followed by `git apply "<extracted-folder>/Version_6_Automatic_Weather.patch"` from your project root. Choose one method.
4. Restart FastAPI and Streamlit using your existing commands. The patch uses the project's existing dependencies, including requests, pandas, openpyxl and Python's built-in SQLite.
5. In Station Management, download the new template. Fill **Station Location** with 1–12 service-area points and add the selected electricity worksheets. Enter latitude and longitude in decimal degrees. A single coordinate is sufficient for a local station. For a wide service area, each representative point has equal weight. Wind direction uses a circular mean.

Demand requires timestamp and load. Solar and Wind Energy each require timestamp, actual generation in MWh per interval and installed capacity in MW. Residual Load uses those three electricity histories. The Model Management upload uses the saved station location for retraining. A station without a managed weather configuration can still accept the original weather-inclusive input format.

## Weather and model inputs

| Stage | Source and behavior |
| --- | --- |
| Historical inputs | Open-Meteo ERA5 reanalysis in UTC. For recent dates where ERA5 may lag, the system uses the Historical Forecast API's recent historical estimates. |
| Training forecast covariates | Archived Open-Meteo ECMWF IFS `previous_day1` values for temperature, cloud cover, radiation and 100 m wind. The lookup is keyed by valid UTC hour. |
| CSV replay | Archived day-ahead weather for the replay timestamp, including targets beyond the last actual generation row when the archive covers those targets. |
| Live Kafka | The current forecast, shared by Demand, Solar and Wind Energy. The predictor refreshes it when needed after 30 minutes. Provider failures can use a previously downloaded forecast up to six hours old if it still covers the required horizon. |
| Residual Load | The existing Dense Autoencoder scores synchronized Demand, Solar and Wind Energy forecasts using aligned load and generation history. It needs no separate weather file. |

Each electricity timestamp uses the weather hour at or before it. Repeated values on the 15-minute grid remain hourly data. Radiation is the provider's preceding-hour mean in W/m². Wind speed is requested in m/s. Weather includes relative humidity in the database for reuse.

Day-ahead values represent model weather predicted at least 24 hours before each valid hour. A training window can contain values from different weather model runs. Live serving uses the latest forecast available to the application. For an exact historical simulation of a particular forecast issuance, an archived run per issuance time would be needed.

Historical energy rows are preserved. Forecast training uses the longest continuous span with the necessary archived forecast fields inside the relevant training source. Registration shows the number of electricity rows missing archived forecasts and checks that the usable span supports a train/validation split. When a complete future horizon is unavailable, actual readings continue to be saved and the affected forecast is skipped with a log entry.

SQLite is created automatically at `data/state/<station_id>/weather.sqlite3`. It stores hourly values and retrieval metadata. Downloaded months are reused across uploads and replay. The cache follows the coordinates and weather model in the station configuration. For paid customer use, set `OPEN_METEO_API_KEY` in the backend environment before startup. The provider's Professional tier or higher covers the archive APIs. See [Open-Meteo pricing](https://open-meteo.com/en/pricing).

## Live Kafka input

1. Select **Live Kafka feed** when registering. Upload the full electricity history through the last interval before the live feed so the seven-day lookback is ready. Select **Historical CSV replay** if you want to use the existing CSV producer instead.
2. Train and deploy the selected models in Model Management. Activate the station. The live setting starts Spark and the predictor while your meter or gateway publishes to the station's raw topic, which defaults to `<station_id>-raw`.
3. Use the same UTC time grid, station ID, interval and measurement units as the workbook. Supply generation and installed capacity alongside demand for the selected renewable models. For example, replace the timestamp below with the current interval.

```json
{
  "station_id": "region-a",
  "utc_timestamp": "2026-11-16T14:00:00Z",
  "demand": 4090.0,
  "solar_mwh": 125.0,
  "solar_capacity_mw": 12500.0,
  "wind_mwh": 820.0,
  "wind_capacity_mw": 18000.0
}
```

A gap resets the affected lookback until enough consecutive measurements are available. On process restart, live context starts from the workbook seed plus subsequent consumed events. If the seed is stale and Kafka resumes after committed events, the forecasts warm up as the live context fills again. Forecasts use the latest installed capacity at the origin as the future capacity input.

## Prepared Amprion database

The package includes a separate `amprion-seed` folder with the downloaded database and the 11 reference coordinates from the attached downloader scripts. This lets you enable the downloaded feed for the existing Amprion entry without changing its station paths or checkpoints.

- ERA5 historical weather contains 42,360 complete hourly rows covering 1 January 2022 through 31 October 2026.
- The IFS day-ahead data has 9,138 complete hourly rows from 1 October 2025 through 31 October 2026, with gaps in March, April and May 2026. Earlier requested months without all variables remain unavailable. The bundled October 2026 replay dates have full coverage.
- The database includes a live forecast snapshot fetched on 16 November 2026. The application refreshes it automatically at use time.

To enable the seed, copy `amprion_weather.sqlite3` to `data/state/amprion/weather.sqlite3` with the predictor stopped. Under the existing `stations.amprion` entry in `config/stations.yaml`, add the `weather` block from `amprion_weather.yaml`, indented to the same level as `display_name`. Use the existing CSV replay activation flow. For model retraining on managed weather, upload the electricity history through Model Management so it is enriched with archived inputs before training.

Provider documentation: [Historical Weather](https://open-meteo.com/en/docs/historical-weather-api), [Previous Runs](https://open-meteo.com/en/docs/previous-runs-api), [Forecast](https://open-meteo.com/en/docs).
