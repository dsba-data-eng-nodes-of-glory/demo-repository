# Station Management update

This patch upgrades the existing POC's Station Management page into a station onboarding and lifecycle control page.

## What it adds

- Upload and validate a historical station CSV through the dashboard.
- Split the uploaded series chronologically: the first 80% becomes model training/validation data; the final configured fraction (default 20%) becomes a simulated stream; the last `history_steps` rows before the stream become the seed history.
- Train a station-specific encoder-decoder LSTM with the same 8 historical and 9 future features expected by the streaming predictor.
- Save a PyTorch checkpoint and `metadata.json` under `models/<station>/v1/`.
- Create missing Kafka topics without deleting/truncating existing topics.
- Start and stop Spark, predictor, producer and training jobs independently per station.
- Show station-specific process status and logs.

## CSV contract

The uploaded CSV must have these columns:

- `utc_timestamp`
- `demand`
- `temperature`

Timestamps must be continuous at the interval configured in `config/model.yaml` (currently 15 minutes). The current station loader removes invalid/duplicate rows and then rejects irregular timestamp gaps. Rename source columns to this contract before upload if necessary.

## Install dependency

In the existing `electricity-spark` Conda environment, from the POC directory:

```cmd
pip install python-multipart
```

Or install only missing packages from `requirements.txt`; avoid upgrading working PySpark/Torch packages unnecessarily.

## Start services

Start Kafka separately, then start FastAPI with one worker and no reload:

```cmd
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Start Streamlit in another terminal:

```cmd
streamlit run dashboard/app.py
```

Open the Station Management page, upload the station CSV and register it. Then use the lifecycle buttons in order:

1. Train / retrain model.
2. Create / verify topics.
3. Activate station.

Activation also checks/creates missing topics and verifies the station's training outputs, seed history, stream CSV and model checkpoint. Kafka must already be running.

## Important behavior

- Amprion's existing checkpoint, source CSVs, topics and Spark checkpoint are not reset by onboarding another station.
- New stations are assigned `<station>-raw` and `<station>-featured` topic names, a separate Spark checkpoint (the Spark job already suffixes its configured checkpoint base with the station ID), and a separate LSTM consumer group (the predictor already suffixes the configured group with the station ID).
- The process manager uses station-qualified identities (`<station>:spark`, `<station>:predictor`, etc.), so the same service types can run for multiple stations concurrently.
- The new trainer uses last-observed-temperature persistence in both training and inference to avoid using future temperature observations unavailable at prediction time. This is not a meteorological forecast; a future weather forecast feed is a later enhancement.
- The trainer uses a capped, evenly spaced sample of rolling windows by default (400 train, 100 validation) to be practical on a CPU-only 8 GB laptop. Increase the limits/epochs after confirming the first onboarding run completes.
- This code has been syntax-checked in the build environment, but Windows Kafka/Spark execution and training on a second station must be validated on the user's machine with that station's dataset.
