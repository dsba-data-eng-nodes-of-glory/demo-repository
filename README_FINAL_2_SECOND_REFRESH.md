# Final 2-second dashboard auto-refresh patch

Apply to the latest Amprion **Version 3** project, after the Residual Load Spike Detection, timeout and origin-alignment patches. Changes ONLY the five Streamlit dashboard Python files listed below; no models, predictors, configurations, Kafka topics, data or SQLite files are replaced.

## Default refresh behavior

* **Live Monitoring**: auto-refresh already enabled; default interval changed from 10 to **2 seconds**. The 2-second selection is applied once even to an existing Streamlit session, but later user changes remain respected. Only the live chart panel reruns.
* **Model Management > Model Analysis**: auto-refresh **enabled by default** at **2 seconds**. The optional third-party `streamlit-autorefresh` full-page rerun was replaced with a native `st.fragment`. This refreshes only the selected demand/solar/wind/residual analysis panel. The API cache TTL is now 2 seconds. Other Model Management tabs (Configure & Retrain, Compare Models, Registry & Deployment) never auto-refresh.
* **Pipeline Control**: station status and Kafka/service readiness strip update every **2 seconds**. Start/Stop/Reset buttons and logs do **not** auto-run.
* **Station Management**: selected station's status/technical service strip refreshes every **2 seconds**. Uploads, registration, training controls, log retrieval, and delete forms do **not** auto-refresh.
* **Platform Overview (Home)**: four live summary counts refresh every **2 seconds** using one batched `/stations/overview` request. Navigation, station cards and activate/stop controls retain their normal manual interactions.

These are UI polling intervals. If a backend response or chart calculation takes longer than 2 seconds, the next update can lag. A 2-second UI refresh does NOT mean the models train or generate forecasts every 2 seconds. For a slow Windows 8-GB system, disable auto-refresh or select 5/10/15/30 seconds in Model Analysis or Monitoring when needed.

## Install

1. Stop Streamlit (`Ctrl+C`). The producer, Spark, predictor, and FastAPI do NOT need to be stopped.
2. Make a backup of the five files below in your `C:\Users\hp\Documents\Nodes of Glory\Version 3` project.
3. Extract ZIP at project root, overwriting the **five dashboard files only** and retaining directory paths. Do not copy the `tests` file unless you want to run its optional regression checks.
4. Start Streamlit as before (for example `streamlit run dashboard/app.py`).
5. Verify monitoring shows 'Every 2 seconds' and Model Analysis has auto-refresh checked with '2 seconds'. Pipeline/Station status and Overview KPIs should update without page reloading.

## Files updated

- `dashboard/app.py`
- `dashboard/pages/monitoring.py`
- `dashboard/pages/model_management.py`
- `dashboard/pages/pipeline_control.py`
- `dashboard/pages/station_management.py`

Optional test: `python -m pytest -q tests/test_default_two_second_refresh_final.py`

## Notes

* Status fragments only poll backend GET endpoints; no POST or DELETE operations are triggered automatically.
* No threshold settings, model checkpoints or source datasets are changed.
* Streamlit >=1.38 is required (already required by existing Monitoring fragments).
* Source regression: 21 tests passed across this refresh change and nearby monitoring/model tests. Prior full suite had 2 unrelated legacy history-window assertion failures; those remain outside this patch.
