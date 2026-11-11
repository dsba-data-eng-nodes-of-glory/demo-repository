# Residual Load Spike Detection — monitoring cleanup

Apply this patch to the **complete Residual Load Intelligence project** (not the earlier demand-anomaly version). It does not change the three forecasting models, residual AE checkpoint, or scoring algorithm.

## Changes
- Section title: **Residual Load Spike Detection**.
- Forecast spike intervals highlighted as orange vertical bands and markers on both the Demand forecast chart and residual-load chart, only for matching forecast origins.
- Monitoring shows residual-load forecast and compact KPIs, not the dual-score diagnostic plot. The dual-score plot stays in **Model Management → Model Analysis**.
- Removed sustained risk alert section and state-machine processing (WATCH/WARNING/CRITICAL/RESOLVING/RESOLVED), including the `/results/residual/alerts` endpoint.
- Immutable per-origin assessment history remains in SQLite to prevent losing earlier scored snapshots on refresh. For existing stations it reuses the legacy SQLite file and **never deletes existing alert-event tables**.
- Station threshold calibration moved to sidebar **Workspace settings → Edit residual spike thresholds**. The selector is lazy-loaded, per station, and writes to `/stations/{station}/residual-settings`. Older threshold values remain readable for stations not yet migrated; new writes remove the legacy alert policy.
- Global dark-theme text contrast corrected for headings, widget labels, metrics, and Plotly titles/axes.

## Apply
1. Stop Streamlit and FastAPI. Back up `config/stations.yaml`, data/processed, and your trained models.
2. Extract the ZIP contents into your existing project root, preserving paths. The patch ZIP includes only the changed/new files.
3. Remove the obsolete **source file** `src/models/residual/alerts.py` (the patch cannot delete a file automatically). **Do not delete** `data/processed/stations/*/residual_alerts.sqlite`; those files hold historical scored snapshots.
4. Restart FastAPI and Streamlit, select Amprion, and open Monitoring. Settings are in the collapsed sidebar.
5. Verify the API returns records and history but no `alerts` field, and that `/results/residual/alerts` is not served.

## Operational notes
- Bands and markers are **predicted residual-load spike intervals**, not observed demand abnormalities. They appear only if the residual and selected Demand charts have the same forecast origin.
- Modifying thresholds changes subsequent scored assessments and produces a new fingerprint; old saved snapshots remain immutable.
- Old state-machine tables remain on disk for historical safety but are neither used nor exposed by the project.
- Keep `config/storage.yaml` at `backend: csv` if Cassandra is not active.
