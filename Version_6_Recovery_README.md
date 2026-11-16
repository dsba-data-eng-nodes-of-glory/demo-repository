# Version 6 fault tolerance patch

## Apply on Windows

1. Use the Version 6 project with the **Automatic Weather Patch** already applied.
2. Stop the current station processes and close FastAPI and Streamlit. When replacing an older build, shut down FastAPI from its terminal to preserve the current replay position.
3. Copy the contents of `overlay` into the project root, the folder containing `backend`, `dashboard`, `src` and `config`. Replace the matching files. Alternatively, run `git apply --check "<path>/Version_6_Fault_Tolerance.patch"` and then `git apply "<path>/Version_6_Fault_Tolerance.patch"` from the project root.
4. Start Kafka, FastAPI (one worker, without `--reload`) and Streamlit as usual. Activate Amprion from **Pipeline Control** or **Home**. The patch uses existing dependencies and built-in SQLite.

## Changes

| Component | Behavior |
| --- | --- |
| Predictor recovery | Stores the recent electricity readings in `data/state/<station>/predictor_recovery.sqlite3`. For Amprion that is up to 672 readings, with demand, available temperature, streamed generation and installed capacities. |
| Restart sequence | Writes results, then saves the recovery readings and Kafka positions in one SQLite transaction, then commits to Kafka. A Kafka redelivery after the local transaction reuses the saved result and avoids processing the same event again. |
| First activation after applying | If the existing Kafka group already has committed positions, the predictor reads a bounded slice of retained committed records and joins them with the workbook seed. |
| Missing intervals | Uses the latest continuous lookback. A late reading inside the retained window can repair a gap. If a measurement never arrived or is no longer available, forecasts wait for a continuous window. No substitute electricity readings are generated. |
| Service supervision | FastAPI watches Spark, the predictor, the CSV producer, and the shared DLQ consumer when managed locally. Unexpected exits trigger independent retries with delays of 2, 4, 8, 16, 30 and 30 seconds. After six failed restart attempts, the service is shown as failed. Two minutes of stable operation clears the consecutive failure count when scheduling the next retry. An explicit activation can retry a failed service. |
| Stop and Reset | **Stop** cancels pending retries and preserves Kafka positions, Spark checkpoints, producer progress, recovery state, and outputs. **Reset replay** is a separate confirmed action under Pipeline Control. Reset clears the predictor checkpoint with the replay state and retains the station weather database. A CSV producer that reaches end of file is shown as completed. |
| Compatibility | The Windows replay launcher clears the recovery database when starting a deliberate fresh replay. The Kafka admin call now supports the installed `kafka-python` topic creation response. |

## Verify after applying

1. Activate Amprion through the dashboard and wait for rolling forecasts.
2. Find the predictor PID in **Pipeline Control → Advanced** or in the response from `GET http://127.0.0.1:8000/pipeline/status`. End that process in Task Manager while the replay still has readings.
3. Look for **Restarting** and then **Running**. The predictor log should show `Restored predictor lookback: 672/672`, then a full 96-step Demand forecast at the next valid origin. Solar and Wind Energy each produce 96 points when their context and weather are available.
4. Test **Stop**, wait briefly, and confirm the processes remain stopped. Activate again and confirm the replay advances from the saved position. Use **Reset replay** only when you want to start the replay from the beginning.

FastAPI must be running to supervise its child processes. This patch targets process recovery on the local setup. Kafka broker and machine availability still depend on the deployment.
