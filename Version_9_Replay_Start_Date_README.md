# Version 9 replay start date

This patch applies directly to the supplied Version 9 archive. Open the project root containing `backend`, `dashboard`, and `src`.

1. Stop FastAPI and Streamlit and extract this ZIP.
2. Copy the files from `overlay` into the project root, preserving their relative paths. If you use Git, run `git apply --check "C:\path\to\Version_9_Replay_Start_Date.patch"` first, then run the same command without `--check`.
3. Restart FastAPI and Streamlit.

## Register stations for the Germany demo

Select **Historical CSV replay** and choose **Replay start date (UTC)** in Station Management. Set **1 October 2026** for each demo station. Each Demand sheet must have a reading at **00:00 UTC** on that date and cover at least one forecast horizon from that timestamp. At the default 15-minute interval with a 24-hour horizon, this means 96 replay readings. The upload also needs sufficient earlier history for the models you selected. Input timestamps are normalized using your selected source timezone.

| Input period | Use |
| --- | --- |
| Before 1 October, 00:00 UTC | Chronological training, validation, and offline testing for forecast models. Residual Load uses its existing fit, validation, and calibration splits. |
| 1 October, 00:00 UTC onward | CSV replay. Demand becomes the producer's input, and registered Solar and Wind Energy observations are available to the replay at the relevant timestamps. |

With the default validation setting, the forecast model splits are approximately 70% training, 15% validation, and 15% offline test within the pre-replay period. The existing validation setting remains configurable. For demand at 15-minute resolution, the replay history seed ends at 30 September, 23:45 UTC.

The date aligns the first data timestamp of each newly registered station. The stations still begin processing when activated. Legacy API submissions using the percentage and previously registered stations keep their original files and configuration. The date picker appears only for CSV replay.

## Verification

The patch was checked against a fresh extraction of the supplied Version 9 archive. The focused suite ran 53 tests, including two real workbook registrations with different historical lengths. Both produced the same first Kafka payload timestamp using a mocked Kafka transport. Dense NN and a short LSTM training run produced separate validation and test periods that ended before the cutoff. The Streamlit date picker was checked in both replay and live Kafka modes. The full Windows Kafka/Spark demo was outside this verification.
