# Residual Load Intelligence — full-model replacement (Version 2 project)

## Overview

This package replaces the **active demand-only anomaly component** with a
capacity-aware **Residual Load anomaly** component. All three previously
trained forecasting models (Demand, Solar Energy, Wind Energy) are reused
unchanged. The new fourth component is a 15-feature / 96-step (24-hour)
**Dense Autoencoder plus statistical-deviation detector**, adapted from the
uploaded Residual_Load_Package benchmark. The pre-trained Amprion benchmark
checkpoint has been packaged in portable `.pt` format for a compatible Amprion
reference deployment. This is not an assertion of production incident-detection
accuracy.

### Data contracts and no-leakage rules

`Residual load (MWh / 15 min) = Demand energy − Solar Energy − Wind Energy`.

* The three **96-step forecasts** must share identical UTC forecast origins,
  target times, 15-minute intervals and geographical/system coverage.
* The original wind dataset has a fixed internal header `Wind onshore [MWh]`.
  Its **user-facing label is Wind Energy**; never rename underlying model
  feature columns without updating the trained forecasting checkpoint.
* Solar and Wind Energy use MWh per 15-minute interval. Onboarding provides a
  per-station demand-unit choice: **MWh per interval** or **average MW**, which
  is converted into MWh by dividing by four.
* Installed Solar/Wind capacity uses **MW**. It must be historical capacity
  known at or before the forecast origin. Capacity bounds are validated.
* Historical observations are never read after the forecast origin for anomaly
  scoring. Forecast issue wall-clock time is not the replay origin.
* Missing/unaligned model forecasts, gaps and absent history result in **not
  ready**, not fabricated residual-risk scores.
* All highlighted intervals are **forecast risk assessments**, not confirmed grid
  incidents or guaranteed grid-security assessments.

### Architecture

```
Demand CSV + Solar CSV + Wind CSV → Kafka raw → Spark features → Kafka featured
       ├─ Demand forecasting checkpoint ──┐
       ├─ Solar Energy checkpoint ─────────┼─ aligned 96-step forecasts ─┐
       └─ Wind Energy checkpoint ──────────┘                             │
                                                                         ▼
                                       Residual load = Demand − Solar − Wind
                                                                         │
                                               Capacity-aware residual features
                                                                         │
                                               Statistical deviation + Dense AE
                                                                         │
                                        Dual thresholds + upward-ramp review
                                                                         │
                                  Flagged forecast target intervals
                                                                         │
                                         SQLite forecast-origin snapshots
                                                                         │
                                        FastAPI → Streamlit monitoring & model lab
```

The demand/renewable forecasters, Kafka topics, Spark processing and
onboarding project structure have **not** been replaced. Only the active
anomaly branch is changed.

## User workflows

**Station onboarding:** `Stations` → Download workbook → Upload Demand,
Solar and Wind worksheets → select the four components (all selected by
default) → select demand units → register. Residual Anomaly requires all three
forecasting components. Registered source data are normalized into a derived
residual training CSV. Existing Amprion uses the packaged reference `.pt`.
Other new stations need a separately trained compatible checkpoint.

**Model management:** choose `Residual Load anomaly autoencoder` →
`Configure & Retrain` → Dense Autoencoder → optional new training dataset
(CSV in canonical combined layout or three-sheet workbook) → versioned
training candidate → explicitly deploy. Renewables and Demand retain their
previous Dense NN/LSTM/tree forecasting choices.

**Dashboard settings:** Open the sidebar → **Workspace settings** →
**Edit residual spike thresholds**. Select the station and configure the
statistical score, autoencoder score, detector agreement and upward-only
highlighting. Setting a threshold to zero uses the checkpoint-calibrated value.
Settings apply to the next scored forecast and do **not** retrain the autoencoder.

**Monitoring:** The section is **Residual Load Spike Detection**, positioned
below the Demand chart. Both the Demand forecast and residual-load forecast
shade/mark anomalous forecast target intervals. The former uses the same
selected forecast origin to prevent stale and misleading overlays. The
monitoring page does **not** display a score-vs-threshold chart or sustained
warning ledger. Model Analysis retains the two score curves and thresholds
for technical validation. Per-origin forecast assessments are saved in SQLite
and may be reviewed under historical assessments.

## Model training limitations

The residual benchmark uses a time-aware baseline and separate chronological
fit/validation/calibration periods. New residual training datasets require
substantial continuous history: with the default 80% fit fraction, the fit
period must contain **at least two full years** and the validation and
calibration periods at least 20 days each. Onboarding rejects selected residual
training components without adequate historical data; users can deselect
residual detection and register the three forecasting models independently.
For fully model-version-independent evaluation, use chronological held-out
forecast origins and labelled operational incidents when those become
available. Alert thresholds are unsupervised quantile calibration, not
measured false-positive rates or precision/recall.

## API endpoints and storage

```
GET  /models/components?station=amprion
GET  /stations/amprion/residual-settings
PUT  /stations/amprion/residual-settings
GET  /results/residual/forecast?station=amprion
```

Residual assessments are stored per station in SQLite WAL mode. Existing
installations reuse `data/processed/stations/<station>/residual_alerts.sqlite`
only to preserve previously saved `residual_assessments` rows. New stations
use `data/processed/stations/<station>/residual_assessments.sqlite`.
Legacy `alert_events` tables are not read, written or deleted by the updated
code. The CSV/Cassandra switch for operational Demand/Solar/Wind forecasts
is preserved. The sustained event state machine and
`/results/residual/alerts` route have been retired.

## Deployment and rollback

1. Stop Kafka/Spark producer/predictor, FastAPI and Streamlit.
2. **Back up the entire original Version 2 folder** (`config/stations.yaml`,
   `models/`, `data/`, `logs/`, `checkpoints/`) before replacing it.
3. For a fresh/isolated installation, extract the **complete-project ZIP**
   into a new folder. Do not overwrite existing historical data blindly.
4. Keep `config/storage.yaml` at `backend: csv` if Cassandra is unavailable.
5. Install existing project `requirements.txt` in your compatible Python
   environment, and start FastAPI and Streamlit using your usual commands.
6. In `Model Management`, confirm all four components and verify that the
   Amprion residual portable `.pt` is available.
7. Check `/results/residual/forecast?station=amprion`: expect 96 rows only
   when all three 96-step model forecasts and adequate **past** observations
   are available. An initial cold request may take over ten seconds; later
   requests use memory/disk snapshot caches.
8. Start replay, verify saved origin assessments persist across refreshes
   and remain isolated by station. Validate that changing thresholds affects
   subsequent scores; forecast anomaly markers are not confirmed grid events.

All three forecasting checkpoints and their source modules are unchanged. The
old demand-only anomaly checkpoint, training/inference code and UI overlays
have been retired from the **complete-project package**; old historical
project ZIPs remain the appropriate rollback source.

## Verification performed for this delivery

* Real Amprion benchmark `.pt` loaded and scored a **96-step** synthetic
  contract trajectory made from the supplied historical sources.
* Real included Amprion replay forecast files were processed end-to-end by
  FastAPI; the residual endpoint returned **96 valid scored steps** with the
  configured thresholds and saved a station-specific snapshot.
* Duplicate assessment for the same forecast origin returned an identical
  stored result. Snapshots use stable (forecast-origin, model/input fingerprint) identities.
* All three forecasting and source files were retained. The legacy demand
  anomaly component is not selectable in Model Management.
* Full regression suite: see the test summary in the delivery message;
  existing monitoring test failures tied to superseded source text remain.
* A live Windows Kafka/Spark cluster, customer-specific accuracy backtesting,
  Streamlit browser rendering, and true future solar/wind weather feeds were
  **not** validated in this environment.
