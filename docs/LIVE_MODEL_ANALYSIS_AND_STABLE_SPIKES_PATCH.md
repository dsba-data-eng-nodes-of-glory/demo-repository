# Version 2 — Live Model Analysis and Stable Forecast Risk

## Applies to

Your **Version 2 project with the Customer Dashboard Patch already applied**. This is a targeted patch. It **does not overwrite** `config/stations.yaml`, trained models, customer uploads, producer checkpoints, stream outputs or existing forecasts. It preserves the original directory layout.

## Changes

### 1. Customer-facing Model Analysis

- Model Management -> Amprion -> Solar or Wind -> Model Analysis now shows **forecast/observation matching and operational replay metrics**: matched points, MAE, RMSE, WAPE, actual-vs-predicted trend and signed error trend. Choose lead time 15 minutes, 1 hour, 6 hours or 24 hours.
- Forecasts are matched only when their targets have matured. **The eligible actual generation observations currently come from the station's historical renewable train/test CSVs, not a live generation ingestion topic.** The maximum eligible timestamp is strictly limited by the latest demand actual actually acknowledged and stored by the streaming pipeline. Thus this is a **replay-aligned operational evaluation**, not proof of live real-world renewable generation accuracy. A genuine renewable actual-ingestion stream is required for production live evaluation.
- New endpoint: `GET /results/renewables/performance?station=amprion&horizon_step=1`. Returns model-specific matched records/metrics, observed replay watermark and explicit data provenance. If no acknowledged actual is present it returns `0` matched points, not fabricated accuracy.
- The demand forecasting analysis remains unchanged.

### 2. Two forecast anomaly charts

- Model Management -> Demand anomaly autoencoder -> Model Analysis:
  1. **Actual demand history -> future demand forecast**, with *current forecast-only* demand-high and ramp-up markers.
  2. **Combined autoencoder reconstruction robust-z score vs checkpoint/configured threshold** (horizontal dashed threshold). Every target uses the checkpoint's real threshold, not a fake probability.
- Shows current origin, number of scored steps and count of anomaly-risk points. Never uses realized demand at future target timestamps to classify forecast spike risk.
- New endpoint: `GET /results/anomaly/forecast?station=amprion`.

### 3. Stable, auditable spike assessments

- Previous dashboard rescored only the currently selected rolling 24-hour forecast. Different forecast origins naturally give different demand curves and spike predictions. That made earlier *predicted* spikes appear to disappear even though they were not retracted as historical observations.
- The backend now persists a short **origin-specific score snapshot** in `data/processed/stations/<station>/anomaly_forecast_history.sqlite3` and only scores each unchanged origin/checkpoint/threshold/prediction set once. New forecasts do not overwrite old assessments; retrained or changed models produce a new fingerprint.
- Monitoring shows a collapsible **Past forecast anomaly assessments** table and optionally renders superseded high-demand spike predictions as **hollow gray marks at their own earlier predicted values**. Current risk markers remain colored orange/red. Old predictions are historical *forecasts*, not confirmed grid incidents, and are never counted as current alerts.
- The SQLite history retains a bounded 800 assessments (about a week of 15-minute updates, plus restarts). Checkpoint parameters and data history remain unchanged. The existing alert-review controls remain available.

### 4. Cleaner navigation

- The left sidebar is collapsed at initial page load. The FastAPI URL is under `⚙️ Workspace settings`, an expander inside the optional sidebar. The redundant fixed Platform Modules text has been removed.
- Shared alert text uses higher contrast with the dark dashboard.
- Model Analysis can be refreshed on demand; optional automatic refresh is available when `streamlit-autorefresh` is installed.

## Installation

1. Stop FastAPI and Streamlit; keep Kafka/Spark data/checkpoints intact.
2. Back up the project, especially `config/stations.yaml`, `models/`, `data/` and `checkpoints/`.
3. Extract the patch ZIP into the **existing project root**, preserving the `backend/`, `dashboard/`, `src/` and `tests/` folders. It changes only the files in the manifest.
4. Restart FastAPI using your project's current environment and launch Streamlit as normal. On Windows, from the project root, the typical commands are:

   ```powershell
   python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
   streamlit run dashboard/app.py
   ```

5. Validate endpoints in the browser:
   - `http://127.0.0.1:8000/results/anomaly/forecast?station=amprion`
   - `http://127.0.0.1:8000/results/renewables/performance?station=amprion&horizon_step=1`
6. In Model Management -> Model Analysis, switch between Solar, Wind and Demand anomaly.
7. On Monitoring, advance the 15-minute replay and check that older forecast flags remain in the history while new-risk markers correctly update with each forecast origin.

## Tests and boundaries

- Focused tests cover timestamp maturity, lead-time selection, replay wall-clock issue times, missing watermark, model/prediction fingerprint changes and stable per-origin snapshots.
- Amprion Dense Autoencoder checkpoint was exercised using an actual model-generated 96-step forecast: 96 scores were produced and the checkpoint/configured threshold was applied; repeated requests for the same origin returned identical scores.
- Full test run: **134 passed / 5 failed**. All five failures are older monitoring regressions carried over from the previous project: one outdated plot legend assertion, two stream-start textual assertions and two tests that require the old two-second default refresh. See existing project tests for the specific failures. The new focused tests passed.
- Streamlit is not installed in the patch-building environment: page code passes syntax/static checks, but its appearance has **not** been browser-tested on your installed Streamlit version. Full Kafka/Spark runtime and a persistent production store were not load-tested.
- Because the anomaly outputs are *forecast risk* rather than observed incidents, forecast flags may legitimately change as future load predictions change. The archive preserves earlier opinions; it does **not** suppress all revisions or establish improved anomaly detection precision. Validate alert policy on retrospective known incidents before treating it as production-grade alarming.
- Since the score snapshots are produced by the API when a forecast is requested, a long period during which nobody accesses the endpoint may leave past intermediate origins unscored. For guaranteed capture of every origin in a 24/7 deployment, move this scoring trigger to the predictor/stream consumer (future enhancement).
