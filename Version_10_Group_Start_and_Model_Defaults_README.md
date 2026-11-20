# Version 10 group start and model defaults

This package applies to the Version 10 project root, identified by the `backend`, `dashboard`, `config`, and `src` folders. It includes the earlier Version 10 hierarchy and model quality changes.

1. Stop FastAPI and Streamlit. Extract the ZIP into the project root and replace the matching source files.
2. Restart FastAPI and Streamlit.
3. On Home, use **Start all stations** directly below **Refresh station cards**. The summary shows which stations started or were already running and why any station could not start.

For a clean Version 10 checkout, the enclosed cumulative Git diff can also be checked using `git apply --check Version_10_Group_Start_and_Model_Defaults.patch` and then applied using `git apply Version_10_Group_Start_and_Model_Defaults.patch`. If you already installed the earlier hierarchy and model quality patch, use the replacement files.

## Station startup

The group start uses the existing station activation checks. Newly launched CSV producers prepare their files, load their checkpoints, and connect to their Kafka topics before they publish on a shared wall-clock schedule. They keep their own saved source cursors and configured replay intervals. Producers already running continue from their current positions. A failed or unready station is reported separately, and ready stations can still start.

A shared start removes the delay from launching each station separately. Stations with the same next source timestamp and replay interval start from the same point. Stations whose saved cursors are already apart continue from their respective timestamps. The Germany view aligns actuals and forecasts by shared source timestamps.

## Monitoring and training

- Individual model and anomaly panels open by default when a substation is selected.
- Germany chart legends sit above the plot with dedicated space below for date and time labels. The monitoring clock is removed.
- New station onboarding and the Model Management selector default to **Dense NN for Demand**, **LSTM for Solar**, **LSTM for Wind Energy**, and **Dense Autoencoder for residual-load anomalies**. Existing stations can adopt the defaults by retraining and deploying the selected models.

## Verification

Verified a shared release using two Python worker processes with different startup durations, as well as an unready station, a failed producer, and an already-running station. Streamlit checks exercised the Home button and its result message, individual monitoring defaults, the four Germany chart layouts, and the training selectors. The Windows Kafka and Spark startup path still needs a live demo check.
