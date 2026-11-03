from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from backend.pipeline_manager import manager

router = APIRouter(prefix="/pipeline", tags=["pipeline"])


class StationRequest(BaseModel):
    station: str | None = None


class ResetRequest(BaseModel):
    station: str
    confirm: bool = False


@router.get("/status")
def status():
    return manager.status()


@router.post("/start")
def start(request: StationRequest | None = None):
    try:
        return manager.start_pipeline(request.station if request else None)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/stop")
def stop(request: StationRequest | None = None):
    try:
        if request and request.station:
            # Keep the legacy pipeline endpoint consistent with Home: a
            # station-scoped stop also performs the station reset.
            from backend.station_api import stop_station
            return stop_station(request.station)
        return manager.stop_pipeline(None)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/reset")
def reset_station(request: ResetRequest):
    if not request.confirm:
        raise HTTPException(
            status_code=400,
            detail="Explicit confirmation is required before resetting station data.",
        )
    try:
        return manager.reset_station_pipeline(request.station)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/logs/{process_name}")
def logs(process_name: str, station: str | None = Query(default=None)):
    try:
        return {
            "name": process_name,
            "station": station,
            "content": manager.read_log(process_name, station=station),
        }
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/stations/{station}/status")
def station_pipeline_status(station: str):
    try:
        return manager.station_status(station)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
