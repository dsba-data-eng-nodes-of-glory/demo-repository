# Version 7 station hierarchy

## Apply the patch

1. Stop the running station pipelines, FastAPI and Streamlit.
2. Extract this ZIP. In the project root from the attached Version 7 archive, run `git apply --check "<path>\Version_7_Station_Hierarchy.patch"`, followed by `git apply "<path>\Version_7_Station_Hierarchy.patch"`. Alternatively, copy the contents of `overlay` into the project root with the same relative paths.
3. Restart FastAPI and Streamlit, and start Kafka as usual. Use the existing environment and requirements. The patch applies directly to the attached Version 7 source.

## Register four German substations

Register each measuring substation from **Station Management → Register a station or substation**. Give it a unique ID and its own Excel workbook. The workbook still uses the station's real location, UTC-normalized timestamps, and inputs for the selected models. Select Demand forecasting to enable live monitoring. Select the renewable models where you have the corresponding actuals and capacity inputs.

For the first substation, put this into **Parent hierarchy (top to bottom)**, replacing the state and station names with your own:

```text
Country: Germany
State: Hesse
Station: Station A
```

Enter the fourth level, the substation, in the **Station or substation ID** and **Display name** fields. Reuse the same parent path for siblings. For another station or state, select the nearest existing branch and extend it. To insert another parent level, add a line like `Region: Rhine-Main` at the appropriate position. The project supports up to 32 parent levels and has no configured station-count limit. Existing standalone stations, including the uploaded Amprion entry, remain visible as standalone sources. Keep any regional total that already includes the four substations outside their branch to avoid double counting.

For a replay demo, use the same holdout start timestamp and overlapping data for all four workbooks. Each workbook's holdout begins after the configured fraction of its own rows. Parent sums are computed at exact UTC timestamps on the same interval grid. Demand values in MWh per interval are converted to mean MW before summing.

## Run the demo

1. Train, deploy and activate the four source stations through their existing controls. Group nodes are created from the paths automatically.
2. Open **Monitoring**, expand the sidebar if needed and choose **Germany**. **View Germany** selects the country total. The successive choices let you drill down through the same tree. Choose a substation for its own operational view or open its individual model and anomaly panels.
3. Pause a substation while the others advance. The **Registered substations reporting** KPI and amber points show incomplete intervals. The top-right UTC clock keeps ticking. Activate the substation again to replay from its saved progress. Avoid **Reset Replay** during the demo, since reset starts a fresh replay.
4. Show the historical points changing from 3/4 to 4/4 when the readings arrive. The source intervals keep their original timestamps.

## Calculation and operation

- The station registry stores parent paths only on source entries. Kafka topics, Spark jobs, model checkpoints, stop/start and reset remain source-scoped. The group view sums descendant results without launching a pipeline per parent.
- Demand actuals use processed readings. With the default single-partition featured topics, a supervised per-substation reader also supplies live demand independently of the forecasting model and keeps its Kafka cursor in SQLite. Multi-partition configurations use the predictor's operational actuals path. The predictor publishes available Solar and Wind Energy measurements.
- Coverage is shown per metric and interval. A gap with zero reports is empty. Residual load uses demand, solar and wind readings that exist for the same source and timestamp. Parent-level anomaly classification would require a model for the rolled-up residual-load series.
- Summed forecasts use complete child runs with matching origins and target timestamps, with partial group coverage flagged per point. The API prefers the newest shared demand origin in the recent run window. Parent totals cover registered substations. Historical source uploads are not used as future actuals in the hierarchy view.
- Group reads use the configured CSV or Cassandra backend and bounded recent result windows parallelized across four read workers. Every group selector can browse many sources, while concurrent pipeline capacity depends on the machine's Spark and model resources.

## Verification

The patch was checked against a clean copy of the uploaded archive. Automated tests cover registration, shared parent paths, deeper hierarchy levels, partial/full roll-ups, forecast-origin alignment, unit conversion and existing pipeline/model contracts. Streamlit smoke tests checked the country display, drill-down and parent reuse. A local Kafka run with four reader processes killed one reader, observed 3/4 coverage, and verified all eight affected intervals completed at 4/4 after recovery. The full Windows setup and concurrent four-model pipelines still need a run on the target machine.
