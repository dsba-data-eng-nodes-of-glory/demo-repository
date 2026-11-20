# Version 10 hierarchy and model quality patch

Apply to the project root from the supplied Version 10 archive, the folder containing `backend`, `dashboard`, `config`, and `src`.

1. Stop the dashboard and FastAPI. Extract the folders from the patch ZIP into the project root and replace the matching Python files. The ZIP also includes a Git patch. For the Git route, run `git apply --check Version_10_Hierarchy_and_Model_Quality.patch` and then `git apply Version_10_Hierarchy_and_Model_Quality.patch`.
2. Restart FastAPI and Streamlit. Select **Germany** from the hierarchy in **Monitoring**. Keep **Show forecasts** enabled. The parent displays Demand, Solar, Wind Energy and Residual Load together. Open **Substation reporting details** to see each station's newest recorded interval.
3. For 50 Hertz, go to **Model Management → Configure & Retrain**. Select the **Registered historical dataset** and **Dense NN** for each of Demand, Solar and Wind Energy. After reviewing each candidate, stop the station pipeline, select the candidate under **Registry & Deployment**, deploy it, and restart the station. The new models contribute as those forecast origins are generated.

## Germany hierarchy verification

The archive has two Germany leaves, Amprion and 50 Hertz. Amprion had saved demand through 16 October 07:00 UTC. 50 Hertz had saved demand through 6 October 03:00 UTC. The previous code read separate recent tails and plotted the faster station's time range. When the saved cursors were this far apart, the recent tails never overlapped.

The endpoint now reads faster stations through the slowest available demand timestamp and searches for a shared forecast origin. It reports coverage by metric and shows the independent station timestamps. When all forecast components share an origin, all four charts use it. If a complete common residual origin is unavailable, Demand, Solar and Wind Energy can each display a complete group forecast at their own explicitly marked origin. A residual forecast requires all three components at the same origin, lead and station. The parent view shows residual values without anomaly scoring.

In this archive, the last shared observation was **6 October 2026 at 03:00 UTC**, with **2 of 2** substations. Amprion contributed **16,900 MW** and 50 Hertz contributed **10,528 MW**, for **27,428 MW**. Solar was **0 MWh**, Wind Energy was **1,025.54 MWh**, and residual load was **5,831.46 MWh** for that 15-minute interval. The verified common forecast origin was **6 October 2026 at 02:45 UTC**, with 96 forecast intervals per metric through **7 October 02:45 UTC**.

Demand monitoring converts uploaded energy per interval to average MW using `MWh × 60 / interval_minutes`. Residual load sums `Demand MW × interval_minutes / 60 − Solar MWh − Wind Energy MWh` per substation and timestamp. The replay progress for each station determines when the group advances.

## Why the 50 Hertz Demand result differed

The workbook contains electricity readings starting in January 2022 and has 166,464 pre-replay rows. The legacy Amprion demand artifact is a PyTorch 96-output network. Registered 50 Hertz Demand used a scikit-learn network with one row for each forecast lead and 240 sampled training origins. The shared Dense NN label describes a model family. The saved checkpoints have different feature and training implementations.

The former training code kept only the single longest uninterrupted archived-weather segment. For 50 Hertz Demand and Wind Energy, that left a 16,896-interval segment ending in March 2026. Solar had a much longer complete segment. The patch keeps later eligible forecast windows, requires a continuous actual-history window and complete archived forecast weather for every target, and keeps the train, validation and test target periods chronological. The October replay stays outside fitting, validation and test.

Demand selects a blend with the same quarter-hour one week earlier. For 50 Hertz, the June–August validation period selected 25% Dense NN and 75% weekly baseline. Wind Energy selects the blend on validation and gives the latest observed generation more weight at short lead times, fading it toward the weather-driven forecast. Solar benefits from the adjusted history and reproducible training seed.

| 50 Hertz metric | Archived model WAPE | Retrained candidate WAPE |
| --- | ---: | ---: |
| Demand | 8.42% | 4.55% |
| Solar | 34.63% | 33.78% |
| Wind Energy | 56.15% | 50.78% |

The comparison uses 10,680 matched forecast-origin and target pairs per metric. Origins were sampled hourly during the common October replay, with targets through 6 October 02:45 UTC. All candidates were trained on pre-October data and evaluated through the serving code using the archive's cached forecast weather. Wind Energy still underpredicted on average by 149.08 MWh per interval in this sample. Wind locations in the workbook are equally weighted, so generation-capacity-weighted locations are worth evaluating against a longer replay before making a broader accuracy claim.

## Verification scope

Verified the bundled CSV/SQLite Germany results, four Streamlit charts and forecast tables, the candidate training flow, and the October replay predictions. The full suite passed 190 tests. Eight failures also reproduced on the original archive. The suite exercised the alternative storage wrappers. A running Cassandra cluster and live Windows Kafka/Spark demonstration were outside the verification environment.
