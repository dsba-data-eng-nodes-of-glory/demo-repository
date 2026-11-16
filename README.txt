15-MINUTE STREAM STEP EVERY 2 REAL SECONDS — PATCH

Root cause confirmed in the uploaded Version 1.zip:
- config/stations.yaml was still set to stream_delay_seconds: 1.5
- demand forecast_every_steps was 4 (one forecast origin per 4 source rows = 60 simulated minutes)
- renewables.forecast_every_steps was 4 (renewable forecasts also updated hourly)
- dashboard/pages/monitoring.py defaulted to a 5-second auto-refresh (selectbox index=1)

Patch changes:
1. stream_delay_seconds: 2.0
   Producer publishes one source CSV row every 2 seconds. Each source row is 15 minutes.
2. forecast_every_steps: 1
   Demand predictor creates a new forecast origin after every accepted 15-minute observation.
3. renewables.forecast_every_steps: 1
   Solar/wind forecast origins also update after every accepted observation.
4. Monitoring auto-refresh default: 2 seconds.

Files to replace in the project root:
- config/stations.yaml
- dashboard/pages/monitoring.py

Restart requirements:
- Stop the running pipeline.
- Restart FastAPI/backend so the process manager reads the updated station config.
- Start the pipeline again so the producer receives --delay 2.0 and the predictor reads cadence=1.
- Restart/refresh Streamlit; select "Every 2 seconds" in the monitoring sidebar if the current browser session retained its old selection.

Important:
- This controls replay and forecast-update cadence; it does not reduce horizon_steps=96.
- Each source CSV row must be 15 minutes apart. The producer preserves source timestamps and publishes one row per delay.
- Forecasting/renewable compute may take longer than 2 seconds. If processing falls behind, Kafka can queue records and the displayed cursor may catch up in bursts. The cadence is a target replay rate, not a hard real-time guarantee.
- Do not reset producer_progress.json or Kafka topics just to change the delay. Resume will continue from the last acknowledged timestamp.

Validation:
- Python compilation passed for monitoring.py, csv_producer.py and streaming_predictor.py.
- 6 tests passed (new cadence configuration tests plus producer resume tests).
