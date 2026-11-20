# Version 11 monitoring unit selector

Extract the replacement files into the Version 11 project root, the folder containing `dashboard` and `backend`. Restart Streamlit and open **Live Monitoring → Monitoring Controls → Display unit**.

The selector starts on **MWh** for energy per observation interval. Choose **MW** for average power over that same interval. Demand, Solar, Wind Energy, and Residual Load use the selection on station and hierarchy charts, summary cards, and forecast tables. The demand forecast CSV includes the selection in its filename and records it in the `demand_unit` and `interval_minutes` columns.

The conversion uses the configured interval and source units. At 15 minutes, 3,033 MWh corresponds to 12,132 MW. Anomaly scores, thresholds, and spike counts use their existing scales.

For a clean Version 11 Git checkout, you can instead extract the enclosed Git diff, run `git apply --check Version_11_Monitoring_Unit_Selector.patch` from the project root, and then run `git apply Version_11_Monitoring_Unit_Selector.patch`.
