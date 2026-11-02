from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.pipeline_api import router as pipeline_router
from backend.station_api import router as station_router
from backend.model_api import router as model_router
from backend.results_api import router as results_router
from backend.pipeline_manager import manager

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    manager.stop_pipeline()

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

@app.get("/")
def root():
    return {"message": "Electricity Forecasting POC API", "docs": "/docs"}
