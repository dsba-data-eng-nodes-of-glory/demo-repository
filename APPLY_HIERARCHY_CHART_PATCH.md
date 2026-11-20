# Nodes of Glory · Hierarchy monitoring patch

Use this patch with **Nodes of Glory Full Final Performance**.

1. Stop the station pipelines and close FastAPI and Streamlit.
2. Extract the ZIP into the existing project root, where the `backend`, `dashboard` and `src` folders are located. Replace the matching Python files.
3. Start FastAPI and Streamlit with your usual commands. Start the stations, then open **Monitoring → Germany** with auto-refresh enabled.

## Chart behavior

Germany and other parent nodes use the same chart layout as Amprion Demand across Demand, Solar, Wind Energy and Residual Load. The default axis shows 72 hours of observation history followed by the configured forecast horizon. The solid blue line represents actuals, the dashed orange line represents the forecast, and the vertical Now line marks the shared forecast origin. The MW and MWh selector applies to all four charts. Each metric uses its own y-axis range.

Before drawing a chart, the dashboard waits until a complete live forecast advances to a newer origin. A hierarchy forecast also needs observations and forecasts from all registered descendants at matching timestamps. With **Show forecasts** off, the chart starts after the next full-coverage observation arrives. A stopped or lagging substation can keep its parent at the last common interval. **Substation reporting details** shows each station’s latest Demand timestamp.

Germany uses available pre-replay history from the registered training CSVs to fill the left side of the chart. Live observations take priority and establish the startup clock. Incomplete observation coverage leaves a gap in the total. Turn on **Partial actual coverage** in the chart legend to inspect those values.

Residual Load at hierarchy level is calculated from aligned Demand, Solar and Wind Energy. Station-level anomaly assessments remain available in the individual monitoring view.

## Verification

The included tests cover stationary startup, advancing forecast origins, restart detection, four hierarchy chart layouts, unit switching, bounded data reads and pre-replay history. Run the targeted checks from the project root.

```text
python -m pytest -q tests/test_monitoring_initialization_only.py tests/test_monitoring_display_units.py tests/test_forecast_startup_placeholder.py tests/test_monitoring_fragment_refresh_v5.py tests/test_station_hierarchy.py
```
