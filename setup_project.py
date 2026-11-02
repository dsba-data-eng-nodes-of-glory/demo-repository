
from pathlib import Path

# ==================================================
# PROJECT ROOT
# ==================================================

ROOT = Path(__file__).resolve().parent

if ROOT.name != "POC":
    raise RuntimeError(
        "Please place this script inside the POC folder."
    )

print("=" * 55)
print(" ELECTRICITY FORECASTING PLATFORM - POC SETUP")
print("=" * 55)

# ==================================================
# DIRECTORY STRUCTURE
# ==================================================

directories = [
    "dashboard/pages",
    "backend",
    "config",

    "src/producers",
    "src/ingestion",
    "src/spark",
    "src/features",
    "src/models",
    "src/analytics",
    "src/storage",

    "models/current",
    "models/versions",
    "models/metadata",

    "data/raw/existing_station",
    "data/raw/new_stations",
    "data/processed",
    "data/features",
    "data/predictions",

    "logs",

    "checkpoints/existing_station",

    "tests"
]

# ==================================================
# FILES AND INITIAL CONTENT
# ==================================================

files = {

    # ---------------- BACKEND ----------------

    "backend/__init__.py": "",

    "backend/main.py": """
from fastapi import FastAPI

from backend.station_api import router as station_router
from backend.pipeline_api import router as pipeline_router
from backend.model_api import router as model_router

app = FastAPI(
    title="Electricity Forecasting Platform",
    version="1.0.0"
)

app.include_router(station_router)
app.include_router(pipeline_router)
app.include_router(model_router)


@app.get("/")
def home():
    return {
        "message": "Electricity Forecasting API is running"
    }
""",

    "backend/station_api.py": """
from fastapi import APIRouter

router = APIRouter(
    prefix="/stations",
    tags=["Station Management"]
)

@router.get("/")
def get_stations():
    return {"message": "Station API ready"}
""",

    "backend/pipeline_api.py": """
from fastapi import APIRouter

router = APIRouter(
    prefix="/pipeline",
    tags=["Pipeline Control"]
)

@router.get("/status")
def pipeline_status():
    return {"status": "Not started"}
""",

    "backend/model_api.py": """
from fastapi import APIRouter

router = APIRouter(
    prefix="/model",
    tags=["Model Management"]
)

@router.get("/status")
def model_status():
    return {"status": "Ready"}
""",

    "backend/pipeline_manager.py": """
# Pipeline process management
# Kafka, Spark and model process control
""",

    # ---------------- DASHBOARD ----------------

    "dashboard/app.py": """
import streamlit as st

st.set_page_config(
    page_title="Electricity Forecasting Platform",
    layout="wide"
)

st.title("Electricity Forecasting Platform")

st.write(
    "Multi-region electricity demand and generation monitoring"
)
""",

    "dashboard/pages/monitoring.py": """
import streamlit as st

st.title("Electricity Monitoring")
st.info("Monitoring dashboard will be connected to storage.")
""",

    "dashboard/pages/station_management.py": """
import streamlit as st

st.title("Station Management")
st.info("Dynamic station onboarding.")
""",

    "dashboard/pages/pipeline_control.py": """
import streamlit as st

st.title("Pipeline Control")
st.info("Kafka and Spark process management.")
""",

    "dashboard/pages/model_management.py": """
import streamlit as st

st.title("Model Management")
st.info("Training, evaluation and model versioning.")
""",

    # ---------------- PRODUCERS ----------------

    "src/producers/__init__.py": "",

    "src/producers/producer_existing.py": """
# Connect your existing regional Kafka producer here.
""",

    "src/producers/producer_factory.py": """
# Dynamic producer creation and station configuration.
""",

    # ---------------- INGESTION ----------------

    "src/ingestion/__init__.py": "",

    "src/ingestion/historical_loader.py": """
# Historical data ingestion for new stations.
""",

    "src/ingestion/kafka_consumer.py": """
# Kafka consumer and ingestion logic.
""",

    # ---------------- SPARK ----------------

    "src/spark/__init__.py": "",

    "src/spark/preprocessing.py": """
# Spark data cleaning and preprocessing.
""",

    "src/spark/streaming.py": """
# Spark Structured Streaming pipeline.
""",

    # ---------------- FEATURES ----------------

    "src/features/__init__.py": "",

    "src/features/feature_engineering.py": """
# Common feature engineering for all stations.
""",

    # ---------------- MODELS ----------------

    "src/models/__init__.py": "",

    "src/models/train.py": """
# Global LSTM training.
""",

    "src/models/predict.py": """
# Demand and generation forecasting.
""",

    "src/models/evaluate.py": """
# Model evaluation and comparison.
""",

    # ---------------- ANALYTICS ----------------

    "src/analytics/__init__.py": "",

    "src/analytics/residual_load.py": """
# Residual load and demand-generation balance.
""",

    "src/analytics/spike_detection.py": """
# Spike and anomaly detection.
""",

    # ---------------- STORAGE ----------------

    "src/storage/__init__.py": "",

    "src/storage/data_writer.py": """
# Write processed data and forecasts.
""",

    "src/storage/data_reader.py": """
# Read data for dashboard and APIs.
""",

    # ---------------- TESTS ----------------

    "tests/test_api.py": "",
    "tests/test_pipeline.py": "",
    "tests/test_features.py": "",

    # ---------------- CONFIGURATION ----------------

    "config/stations.yaml": """
stations:
  - id: existing_station
    name: Existing Station
    enabled: true
    status: active
    kafka_topic: existing_station_data
    historical_data_path: data/raw/existing_station/
    checkpoint_path: checkpoints/existing_station/
""",

    "config/kafka.yaml": """
kafka:
  bootstrap_servers: localhost:9092
  consumer_group: electricity-consumer

  topics:
    existing_station: existing_station_data
""",

    "config/model.yaml": """
model:
  name: global_lstm
  forecast_horizon: 24
  training_strategy: global
  retraining_enabled: false

evaluation:
  metrics:
    - RMSE
    - MAE
    - MAPE
""",

    # ---------------- ROOT FILES ----------------

    "requirements.txt": """
fastapi
uvicorn
streamlit
requests
pandas
numpy
pyyaml
pyspark
scikit-learn
tensorflow
python-dotenv
""",

    ".env": """
API_HOST=127.0.0.1
API_PORT=8000

KAFKA_BOOTSTRAP_SERVERS=localhost:9092

ENVIRONMENT=development
""",

    "README.md": """
# Electricity Forecasting Platform - POC

## Objective

Scalable multi-region electricity demand and generation
forecasting platform.

## Initial configuration

- One active station
- Three additional stations supported through onboarding
- Kafka for data streaming
- PySpark for preprocessing and feature engineering
- LSTM for demand and generation forecasting
- FastAPI for backend control
- Streamlit for monitoring and pipeline management

## Architecture

Data Producers
    -> Kafka
    -> Spark
    -> Feature Engineering
    -> ML Models
    -> Analytics
    -> Storage
    -> FastAPI
    -> Streamlit

## Run FastAPI

uvicorn backend.main:app --reload --port 8000

## Run Streamlit

streamlit run dashboard/app.py
"""
}

# ==================================================
# CREATE DIRECTORIES
# ==================================================

print("\\nCreating directories...")

for directory in directories:
    path = ROOT / directory
    path.mkdir(parents=True, exist_ok=True)
    print(f"[DIR] {directory}")

# ==================================================
# CREATE FILES WITHOUT OVERWRITING
# ==================================================

print("\\nCreating files...")

created = 0
skipped = 0

for filename, content in files.items():

    path = ROOT / filename

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if path.exists():
        print(f"[SKIP] {filename} (already exists)")
        skipped += 1
        continue

    path.write_text(
        content.strip() + "\\n",
        encoding="utf-8"
    )

    print(f"[FILE] {filename}")
    created += 1

# ==================================================
# CREATE PACKAGE FILES
# ==================================================

packages = [
    "src",
    "src/producers",
    "src/ingestion",
    "src/spark",
    "src/features",
    "src/models",
    "src/analytics",
    "src/storage"
]

for package in packages:

    init_file = ROOT / package / "__init__.py"

    if not init_file.exists():
        init_file.write_text("", encoding="utf-8")
        created += 1

# ==================================================
# FINAL REPORT
# ==================================================

print("\\n" + "=" * 55)
print(" POC DIRECTORY SETUP COMPLETED")
print("=" * 55)

print(f"Project location : {ROOT}")
print(f"Directories      : {len(directories)}")
print(f"Files created    : {created}")
print(f"Files preserved  : {skipped}")

print("\\nNext steps:")
print("1. Install requirements.")
print("2. Start FastAPI.")
print("3. Open FastAPI Swagger documentation.")
print("4. Start Streamlit.")
print("5. Connect existing Kafka and Spark scripts.")

print("\\nSetup completed successfully!")
