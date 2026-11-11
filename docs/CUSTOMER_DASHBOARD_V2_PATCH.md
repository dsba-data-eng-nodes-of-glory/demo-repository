# Customer Dashboard, Performance & Demand Spike Review Patch

**Scope:** Targeted changes to the uploaded **Version 2** codebase. No core package or directory is renamed. The existing training algorithms, model artifacts, station registry, Kafka topics, data files and Spark checkpoint contracts are unchanged.

## Summary

| Area | Update | Expected benefit |
|---|---|---|
| Home | New `GET /stations/overview` endpoint; Home uses it with 404 fallback | One status API request instead of `2 × number of stations + 2` separate requests |
| API performance | Java runtime/version status cached for 120 seconds; start checks remain fresh | Avoid repeated Windows `java -version` subprocess launches on every refresh |
| Model Management | Replaced eager `st.tabs` rendering with a single active `st.radio` section | Prevents hidden panels fetching 20,000-row histories, experiments and registries while editing training parameters |
| Station Management | Show selected / historical data available / checkpoint deployed **separately** for all four components | Correct legacy Amprion interpretation; less confusion between deselected, untrained and stopped |
| Station actions | Confirmation for stop/reset on Home and Pipeline | Avoid accidental clearing of station replay results and Spark checkpoints |
| UI accessibility | Central CSS for widgets, labels, tabs, expanders, error messages, keyboard focus and mobile widths | Better contrast and legibility across all dashboard pages |
| Monitoring | Default refresh 10 seconds with 2/5/10/15-second options; forecast tail reduced to 2,000 records | Less frequent heavy chart recomputation and data transfer; 2-second demo mode still available |
| Renewable monitoring | Select the most recent **complete** 96-step solar and wind run, fallback to partial with diagnostics | Avoids incomplete newer forecast hiding an earlier complete forecast |
| Chart theme | Dark renewable and saved-forecast chart styles | Consistency with the app shell |
| Demand anomaly | Optional event-level risk review over unchanged Dense Autoencoder classifications | Customer-readable grouped events; separate sustained alerts from short spikes without retraining or recalibrating the model |
| Anomaly checkpoint I/O | LRU cache deserialized joblib bundle by path + size + mtime | Reduces repeated disk reads during Streamlit reruns |

## Demand anomaly — current algorithm and limits

The existing checkpoint uses a **Dense Autoencoder**, not an LSTM. It scores the final timestamp of each trailing **96 × 15-minute** demand window using ten features: demand deviation, 30-minute ramp, 1-hour ramp, 24-hour change, and time-of-day/day-of-week/year cycle sin/cos features. An annual trend and calendar baseline are fitted on training data. The level and ramp reconstruction errors are robust-z standardized against training calibration. A station-level threshold override, if set, changes only the inference decision boundary. Anomalous **forecast** demand above the baseline is classified as either `LEVEL_SPIKE` or `RAMP_UP_SPIKE`; low-demand anomalies are retained in raw classifications but intentionally not highlighted as demand-high events.

The new `src/models/anomaly/event_review.py` does **not** modify the trained model, the checkpoint, the anomaly threshold or original classifications. It groups consecutive forecast-only demand-high/ramp-up signals and independently assigns `alert_review_status` using:

- Minimum contiguous forecast intervals (default 2 = 30 minutes)
- Minimum **event peak** forecast demand above the historical baseline (default 1.0%)
- No future observed demand at or after the forecast origin
- Missing 15-minute timestamps break an event; brief events remain visible as model signals but are not confirmed for the review list

These review thresholds are **operator heuristics** for alert triage, not learned or statistically calibrated performance guarantees. The UI exposes both values in **Monitoring → Spike alert sensitivity**. Orange/red markers still show original Autoencoder output, while the table groups signals and distinguishes those meeting review criteria. The table is downloadable as CSV.

**Important:** There are no verified spike labels in the submitted dataset. False positives, recall, precision and event lead time **cannot** be established without a historical event audit and a held-out backtest. Demand-only anomalies cannot identify *renewable shortfall* or *residual-load spike* causes reliably; they require matching solar/wind forecast and observed generation and an explicit residual-load forecasting model. Do not use flags as autonomous dispatch instructions.

### Recommended further validation before live rollout

1. Audit training/test chronological separation and holdout at least several weeks from AE fitting and threshold calibration.
2. Build an event annotation set across seasons; record true-positive rate, false alarms per day, event precision, mean lead time, and duration error by station.
3. Analyze forecast bias/uncertainty: AE flags applied to predicted demand may reflect forecasting errors rather than physical spikes.
4. Calibrate operator-review thresholds on validation events, then freeze settings for untouched holdout tests.
5. Add a separate renewable/residual-load event classifier **only** once same-origin solar/wind predictions are validated. Do not use actual future weather, future generation, or tomorrow's prices at forecast origin.

## Installation (patch ZIP)

1. **Stop FastAPI and Streamlit**. Keep Kafka and station data backed up before changing code.
2. Back up `config/stations.yaml`, `data/`, `models/`, `checkpoints/`, and `logs/`.
3. Extract `Version_2_Customer_Dashboard_Patch.zip` into the existing project root (the folder containing `backend/`, `dashboard/`, `src/`, `tests/`). Select **replace changed files**; do not replace the whole project directory.
4. Restart FastAPI from the same project root, **one worker, no `--reload`**:

   ```bat
   python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
   ```

5. Restart Streamlit:

   ```bat
   python -m streamlit run dashboard/app.py
   ```

6. Open `http://127.0.0.1:8000/stations/overview`. Confirm `statuses` and `components` contain Amprion and all four enabled components.
7. In Home confirm Kafka readiness, in Stations check data/deployment statuses, in Models test switching all four components, in Monitoring inspect alert review and try a refresh of 2 or 10 seconds.
8. To rollback, restore the pre-patch code files from backup. Nothing in the patch overwrites `stations.yaml` or model checkpoint binaries.

## Testing and limitations

- Validated Python syntax using `compileall`.
- Ran full Python test suite on this source snapshot: **130 passed, 5 failed, 1 warning**. The remaining failures are static checks expecting older chart/stream-start behavior (3), or the old **2-second default** instead of the new 10-second default (2). No known newly introduced functional-test failure remains.
- All 8 new event/performance unit tests passed, including Amprion legacy component discovery, checkpoint-cache invalidation, shared Java status caching, baseline guardrails, event contiguity and incomplete-run selection. The renewable API smoke/integration tests also passed.
- **Streamlit was not installed in the code-execution environment and could not be downloaded.** The actual UI therefore was not render-tested. Validate layout and CSS on the exact Windows browser + Streamlit version you deploy.
- **Kafka/Spark/FastAPI/Streamlit Windows end-to-end load test was not performed.** Do not present this patch as certified high-availability production software.

### Additional production requirements not covered by this patch

Authentication / role-based permissions for upload, train, deploy and station reset; HTTPS and network isolation; persistent backup and recovery; per-customer tenant isolation; audit logs for actions; monitoring and alert notifications; worker/process supervision; broker and storage durability testing; resource-capacity and multiuser load testing; and statistically validated anomaly alert policy. The current PipelineManager is a **single-host process manager** and should be run with one FastAPI worker. These are prerequisites before external customer deployment; they are deliberately **not** introduced here because the user asked to preserve the core architecture.
