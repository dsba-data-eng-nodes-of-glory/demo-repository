"""Prometheus metrics shared by the streaming predictor process."""
from __future__ import annotations

import logging

LOG = logging.getLogger(__name__)
_STARTED = False

try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server

    EVENTS_RECEIVED = Counter("electricity_events_received_total", "Kafka events received", ["station"])
    EVENTS_ACCEPTED = Counter("electricity_events_accepted_total", "Kafka events accepted", ["station"])
    EVENTS_REJECTED = Counter("electricity_events_rejected_total", "Kafka events rejected", ["station", "rule"])
    FORECAST_RUNS = Counter("electricity_forecast_runs_total", "Forecast runs completed", ["station", "algorithm"])
    FORECAST_POINTS = Counter("electricity_forecast_points_written_total", "Forecast points persisted", ["station"])
    PROCESSING_SECONDS = Histogram("electricity_event_processing_seconds", "Event processing time", ["station"])
    LAST_EVENT_TIMESTAMP = Gauge("electricity_last_event_timestamp_seconds", "Unix timestamp of latest accepted event", ["station"])
    STORAGE_UP = Gauge("electricity_storage_up", "Whether configured storage is reachable")
except ImportError:  # metrics are optional for local development
    EVENTS_RECEIVED = EVENTS_ACCEPTED = EVENTS_REJECTED = None
    FORECAST_RUNS = FORECAST_POINTS = PROCESSING_SECONDS = None
    LAST_EVENT_TIMESTAMP = STORAGE_UP = None


def start_metrics_server(port: int, enabled: bool = True) -> int | None:
    """Start Prometheus on the requested port, or a nearby free port.

    A port conflict must not stop forecasting. Metrics stay optional when no
    candidate port can be bound.
    """
    global _STARTED
    if not enabled or _STARTED or EVENTS_RECEIVED is None:
        return None
    base = int(port)
    candidates = [base + offset for offset in (0, 100, 200, 300) if 1024 <= base + offset <= 65535]
    last_error: OSError | None = None
    for candidate in candidates:
        try:
            start_http_server(candidate, addr="0.0.0.0")
        except OSError as exc:
            last_error = exc
            LOG.warning("Prometheus metrics port %s is unavailable (%s).", candidate, exc)
            continue
        _STARTED = True
        LOG.info("Prometheus metrics available on :%s/metrics", candidate)
        return candidate
    LOG.warning("Prometheus metrics disabled because no port was available: %s", last_error)
    return None
