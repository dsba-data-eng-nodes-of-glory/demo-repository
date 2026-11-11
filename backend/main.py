from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi import Request
import time
import logging
from backend.pipeline_api import router as pipeline_router
from backend.station_api import router as station_router
from backend.model_api import router as model_router
from backend.results_api import router as results_router
from backend.pipeline_manager import manager
from src.storage.forecast_store import storage_health

LOGGER = logging.getLogger("electricity.api")

try:
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
    API_REQUESTS = Counter("electricity_api_requests_total", "API requests", ["method", "path", "status"])
    API_LATENCY = Histogram("electricity_api_request_seconds", "API request duration", ["method", "path"])
except ImportError:  # metrics remain optional in lightweight POC installs
    API_REQUESTS = API_LATENCY = None
    generate_latest = None
    CONTENT_TYPE_LATEST = "text/plain"

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    manager.stop_pipeline()
    try:
        from src.storage.cassandra_store import close as close_cassandra
        close_cassandra()
    except Exception:
        pass

app = FastAPI(title="Electricity Forecasting POC API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(pipeline_router)
app.include_router(station_router)
app.include_router(model_router)
app.include_router(results_router)

@app.middleware("http")
async def observe_requests(request: Request, call_next):
    started = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        if API_REQUESTS is not None:
            path = request.url.path
            API_REQUESTS.labels(request.method, path, str(status)).inc()
            elapsed = time.perf_counter() - started
            API_LATENCY.labels(request.method, path).observe(elapsed)
            if elapsed >= 2.0:
                LOGGER.warning("Slow API request method=%s path=%s status=%s elapsed_seconds=%.3f", request.method, path, status, elapsed)

@app.get("/health/live")
def liveness():
    return {"status": "alive"}

@app.get("/health/ready")
def readiness():
    storage = storage_health()
    kafka = manager.kafka_available()
    ready = bool(storage.get("available")) and bool(kafka)
    return {"ready": ready, "storage": storage, "kafka_available": kafka}

@app.get("/metrics")
def metrics_endpoint():
    if generate_latest is None:
        return Response("prometheus-client is not installed", status_code=501, media_type="text/plain")
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/")
def root():
    return {"message": "Electricity Forecasting API", "docs": "/docs", "health": "/health/ready", "metrics": "/metrics"}
