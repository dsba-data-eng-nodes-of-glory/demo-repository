# Version 6 startup performance patch

## Install

1. Apply the **Automatic Weather** and **Fault Tolerance** patches first.
2. Stop the station and close FastAPI and Streamlit.
3. Copy the contents of the ZIP's `overlay` directory into the Version 6 folder that contains `backend`, `src`, `dashboard`, and `config`. Alternatively, from that folder run `git apply --check "<path>/Version_6_Startup_Performance.patch"` followed by `git apply "<path>/Version_6_Startup_Performance.patch"`.
4. Start FastAPI and Streamlit and activate Amprion. Use the existing station configuration, model files, and weather database. You can resume the current replay with its saved Kafka offsets and predictor state.

## Changes

| Area | Behavior |
| --- | --- |
| Demand lookback | Recovery is checked first. A complete recovered window skips the seed CSV. For a new Amprion replay, the predictor reads and validates the last 672 rows of the configured chronological demand CSV. A custom seed or an unusable tail uses full-file validation. |
| Solar and Wind Energy | Load recent rows around the current replay origin from the configured histories. Use the full source when the bounded read cannot cover the required earlier context. Save each component's forecast as soon as it is generated. |
| Residual Load | Reuse the generation and capacity history already loaded by the predictor. A background worker assesses synchronized Demand, Solar, and Wind Energy forecasts while Kafka consumption continues. It retains the active assessment and the newest waiting origin. During a backlog it can skip intermediate assessments. |
| Monitoring | Publish Demand before loading renewable model history. Display available renewable forecasts while their historical observations are loading. Use the most recent Demand observations for the live chart and retain full-file fallback for earlier requests. Anomaly markers require the assessment and plotted forecast to share the same origin. |

The 24-hour, 96-step horizon and two-second dashboard refresh continue to apply. The existing FastAPI service startup sequence and Kafka group session settings are used.

## Verify

Watch Monitoring after activation. Demand should become available first, then Solar and Wind Energy, followed by the first completed Residual Load assessment. The predictor log records lookback restoration, first Demand forecast, saved Solar and Wind Energy forecasts, and Residual Load scoring. Compare those timestamps with the time you activated the station if your laptop still takes longer than expected.

In the Linux test with local Kafka 4.3.1 and the Amprion model files and weather cache, the first Demand output was saved about 1.5 seconds after predictor launch. Solar, Wind Energy and the first Residual Load output were saved within about 2 seconds. This test published directly to the featured topic. Windows activation, Spark startup, and browser rendering were outside those timings.
