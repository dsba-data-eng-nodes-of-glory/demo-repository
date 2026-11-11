# Compact Customer Dashboard — Layout Patch

## Source baseline
Apply over Version 2 **after both** Version 2 Customer Dashboard Patch and
Version 2 Live Model Analysis Patch. The archive preserves the original project
structure. This patch updates dashboard UI only; it does not replace data,
station configuration, models, backend code or checkpoints.

## Changes

- Monitoring: demand KPIs use smaller cards; Solar/Wind now each display latest
  replay-visible actual, first forecast step, forecast peak and forecast average.
  Solar/Wind values are interval energy (MWh); demand values are power (MW).
- Monitoring: API readiness, quality-event counts, storage, raw API row counts,
  replay details, saved-run charts and debug captions move behind sidebar
  **Show technical details**, default OFF. The explicit **Show forecast table**
  setting still displays the table independently.
- Monitoring: anomalies are **not** disabled; forecast markers, saved origin
  history, risk review and their calibrated threshold remain unchanged.
  Data delay remains visible as a compact warning indicator.
- Station Management: model-component inventory is collapsed initially and
  station/pipeline status uses compact indicators rather than tall metric cards.
- Pipeline: service status, Kafka and model readiness share one status strip.
  Component inventory is on-demand; Kafka topics and logs are in Advanced.
  Stop/reset and activation safety warnings remain visible.
- Model Analysis: replace the three large configuration/deployment cards with
  compact status indicators; keep graphs visible and place technical paths
  inside an optional details expander.
- Home: station state is shown once, compactly; new-customer setup steps and
  architecture explanation are collapsed when not needed. Remove incorrect
  LSTM-only architecture label.
- Shared theme: tighter metric-card padding, improved informational-alert
  contrast, slightly smaller headings, help text accessible via `?` tooltips.
- Renewable figure height reduces from 620 to 500 while retaining two plots.

## Apply (Windows)

1. Stop Streamlit. Back up the project folder before overwriting any source.
2. Extract this ZIP into the project root, retaining paths such as
   `dashboard/pages/monitoring.py`.
3. Restart: `streamlit run dashboard/app.py` (from the project root).
4. On Monitoring open the sidebar to find **Show technical details**.
   Leave unchecked for the customer-facing view.
5. Check Demand, Solar, Wind cards against the latest 96-step forecasts and
   verify monitoring still shows anomaly markers and two renewable plots.

## Verification

- Python `compileall` passed on the dashboard package.
- 140 pytest tests passed and 5 failed in the last full suite. The five failures
  reproduce older tests expecting pre-existing anomaly-marker labels, removed
  graph cutoff logic, or the earlier 2-second refresh default. No new failures.
- Focused data-summary, replay-cutoff, analysis and component UI tests passed.
- Browser-level Streamlit rendering and Windows Kafka/Spark lifecycle were not
  tested in this environment (Streamlit is not installed here).
- The implementation reduces **visible** clutter and skips health/quality API
  requests unless diagnostics is enabled. It is not a measured latency benchmark.

## Data correctness

Solar/Wind actual values are taken from the preexisting replay-filtered
`/results/renewables` response, never from future observations that the replay
has not acknowledged. The forecast summaries choose each component's latest
forecast origin and avoid combining overlapping saved runs. When no records are
available, the card shows an em dash, not a fabricated zero. A partial run is
labelled `Forecast peak / Forecast average` rather than `24h peak / 24h average`.
