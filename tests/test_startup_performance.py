from threading import Event
from pathlib import Path

import pandas as pd

from backend import results_api
from src.models import streaming_predictor
from src.models.residual.background import ResidualWorker
from src.models.residual.operational import build_residual_history


def test_registered_demand_seed_reads_only_needed_tail(tmp_path, monkeypatch):
    path = tmp_path / "demand.csv"
    times = pd.date_range("2025-10-01", periods=10_000, freq="15min", tz="UTC")
    pd.DataFrame({"utc_timestamp": times, "demand": 3000.0, "temperature": 18.0}).to_csv(path, index=False)
    existing_loader = streaming_predictor.load_station_csv

    def tail_only(*args, **kwargs):
        assert len(kwargs["source_frame"]) == 672
        return existing_loader(*args, **kwargs)

    monkeypatch.setattr(streaming_predictor, "load_station_csv", tail_only)
    frame = streaming_predictor.load_history_source(path, steps=672, expected_end=times[-1])
    assert frame.utc_timestamp.tolist() == times[-672:].tolist()


def test_worker_does_not_block_stream_or_queue_all_origins():
    entered, finish, latest_done = Event(), Event(), Event()
    calls = []

    def assess(origin):
        calls.append(origin)
        if origin == 1:
            entered.set()
            assert finish.wait(timeout=3)
        if origin == 3:
            latest_done.set()

    worker = ResidualWorker(assess)
    try:
        worker.submit(1)
        assert entered.wait(timeout=2)
        worker.submit(2)
        worker.submit(3)
        assert calls == [1]
        finish.set()
        assert latest_done.wait(timeout=2)
        assert calls == [1, 3]
    finally:
        finish.set()
        worker.close()


def test_residual_uses_prepared_observations_without_registered_csv_reads(tmp_path):
    past = pd.date_range("2026-10-01", periods=450, freq="15min", tz="UTC")
    dates = past.append(pd.DatetimeIndex([past[-1] + pd.Timedelta(minutes=15)]))
    demand = [{"utc_timestamp": date, "demand": 400.0} for date in dates]
    prepared = {kind: pd.DataFrame({kind: 5.0, f"{kind}_capacity_mw": 25.0}, index=dates)
                for kind in ("solar", "wind")}
    history, solar, wind = build_residual_history(tmp_path, {}, past[-1], demand, renewable_context=prepared)
    assert history.index[-1] == past[-1] and len(history) == 450
    assert history["residual"].eq(390).all()
    assert (solar, wind) == (25, 25)


def test_renewables_returns_saved_forecasts_while_actual_history_loads(tmp_path, monkeypatch):
    origin = pd.Timestamp("2026-10-01T00:00Z")
    cfg = tmp_path / "stations.yaml"
    forecasts = tmp_path / "renewables.csv"
    cfg.write_text(f"stations:\n  amprion:\n    renewable_forecasts_file: {forecasts.as_posix()}\n")
    pd.DataFrame([{
        "station_id": "amprion", "generation_type": "solar", "forecast_origin_utc": origin.isoformat(),
        "forecast_timestamp_utc": (origin + pd.Timedelta(minutes=15 * step)).isoformat(),
        "horizon_step": step, "predicted_generation_mwh": 10.0, "forecast_run_id": "solar-run",
    } for step in range(1, 97)]).to_csv(forecasts, index=False)
    monkeypatch.setattr(results_api, "STATION_CONFIG_FILE", cfg)
    monkeypatch.setattr(results_api, "_renewable_warmed_or_schedule", lambda station: False)
    monkeypatch.setattr(results_api, "storage_backend", lambda: "csv")
    monkeypatch.setattr(results_api, "read_renewable_actuals", lambda *a, **k: [])
    out = results_api.renewables("amprion", limit=100, end_time=origin.isoformat(), defer_heavy=True)
    assert out["history_pending"] is True and len(out["forecasts"]["solar"]) == 96
    assert out["forecasts"]["wind"] == []


def test_bounded_renewable_read_matches_full_features(monkeypatch):
    from src.models.renewable_forecaster import AmprionRenewableForecaster
    from src.models.training.amprion import solar_mlp_features as solar

    file = Path(__file__).resolve().parents[1] / "data/training/amprion/solar_train.csv"
    origin = pd.Timestamp("2026-09-30T23:45Z")
    runner = AmprionRenewableForecaster(file.parent, "amprion", {}, startup_origin=origin)
    runner.weather = object()  # Managed weather supplies the future independently of the file.
    small, _ = runner._read_history([file], solar.load_solar_csv)
    full, _ = solar.load_solar_csv(file)
    assert len(small) == 8192
    pd.testing.assert_frame_equal(small.tail(672).reset_index(drop=True), full.tail(672).reset_index(drop=True))


def test_monitoring_demand_tail_matches_regular_api_at_start_of_replay():
    cutoff = "2026-09-30T23:45:00Z"
    standard = results_api.actuals("amprion", limit=20, end_time=cutoff)
    fast = results_api.actuals("amprion", limit=20, end_time=cutoff, recent_only=True)
    assert fast == standard
    assert len(fast["records"]) == 20
