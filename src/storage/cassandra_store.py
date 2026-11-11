"""Cassandra operational store for electricity actuals and forecasts.

The tables are query-oriented and intentionally denormalized. This module is
not a data lake: Cassandra stores operational time-series observations and
forecast runs used by the live API/dashboard.
"""
from __future__ import annotations

import calendar
import logging
import os
from datetime import date, datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
LOG = logging.getLogger(__name__)
_LOCK = RLock()
_CLUSTER = None
_SESSION = None
_PREPARED: dict[str, Any] = {}
_SCHEMA_READY = False


def _config() -> dict[str, Any]:
    path = ROOT / "config" / "storage.yaml"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _utc_naive(value: Any) -> datetime:
    """Cassandra TIMESTAMP uses a Python datetime; normalize it to naive UTC."""
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime):
        value = datetime.fromtimestamp(float(value), tz=timezone.utc)
    if value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def _month_bucket(value: datetime) -> str:
    return value.strftime("%Y-%m")


def _origin_date(value: datetime) -> date:
    return value.date()


def get_session():
    """Return a process-wide Cassandra session, connecting lazily."""
    global _CLUSTER, _SESSION, _SCHEMA_READY
    with _LOCK:
        if _SESSION is not None:
            return _SESSION
        try:
            from cassandra.cluster import Cluster
            from cassandra.auth import PlainTextAuthProvider
            from cassandra.policies import DCAwareRoundRobinPolicy
        except ImportError as exc:  # pragma: no cover - depends on optional service extra
            raise RuntimeError(
                "Cassandra backend selected but cassandra-driver is not installed. "
                "Run: pip install -r requirements.txt"
            ) from exc

        cfg = _config().get("cassandra", {})
        env_hosts = os.getenv("CASSANDRA_CONTACT_POINTS")
        if env_hosts:
            cfg["hosts"] = [host.strip() for host in env_hosts.split(",") if host.strip()]
        if os.getenv("CASSANDRA_LOCAL_DATACENTER"):
            cfg["local_datacenter"] = os.environ["CASSANDRA_LOCAL_DATACENTER"]
        if cfg.get("create_schema_on_start", True) and not _SCHEMA_READY:
            ensure_schema()
            _SCHEMA_READY = True
        hosts = cfg.get("hosts") or ["127.0.0.1"]
        auth = None
        if cfg.get("username"):
            auth = PlainTextAuthProvider(
                username=str(cfg["username"]),
                password=str(cfg.get("password") or os.getenv("CASSANDRA_PASSWORD", "")),
            )
        _CLUSTER = Cluster(
            contact_points=hosts,
            port=int(cfg.get("port", 9042)),
            auth_provider=auth,
            load_balancing_policy=DCAwareRoundRobinPolicy(local_dc=str(cfg.get("local_datacenter", "datacenter1"))),
            connect_timeout=float(cfg.get("connect_timeout_seconds", 10)),
        )
        try:
            _SESSION = _CLUSTER.connect()
            _SESSION.default_timeout = float(cfg.get("request_timeout_seconds", 15))
            keyspace = str(cfg.get("keyspace", "electricity"))
            _SESSION.set_keyspace(keyspace)
            _prepare(_SESSION)
            return _SESSION
        except Exception:
            _CLUSTER.shutdown()
            _CLUSTER = None
            _SESSION = None
            raise


def _prepare(session) -> None:
    global _PREPARED
    from cassandra import ConsistencyLevel
    configured_level = os.getenv("CASSANDRA_CONSISTENCY_LEVEL", str(_config().get("cassandra", {}).get("consistency_level", "ONE"))).upper()
    consistency = getattr(ConsistencyLevel, configured_level, ConsistencyLevel.ONE)
    _PREPARED = {
        "actual": session.prepare(
            "INSERT INTO actuals_by_station_month "
            "(station_id, month_bucket, utc_timestamp, actual_demand, temperature) "
            "VALUES (?, ?, ?, ?, ?)"
        ),
        "latest_actual_read": session.prepare(
            "SELECT month_bucket, latest_timestamp FROM latest_actual_month_by_station WHERE station_id = ?"
        ),
        "latest_actual_write": session.prepare(
            "INSERT INTO latest_actual_month_by_station (station_id, month_bucket, latest_timestamp) VALUES (?, ?, ?)"
        ),
        "forecast": session.prepare(
            "INSERT INTO forecasts_by_station_day "
            "(station_id, origin_date, forecast_origin_utc, forecast_run_id, "
            "forecast_issued_at_utc, horizon_step, forecast_timestamp_utc, "
            "predicted_demand, model_file, algorithm, model_version, experiment_id, weather_mode) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        ),
        "latest_point": session.prepare(
            "INSERT INTO latest_forecast_points_by_station_run "
            "(station_id, forecast_run_id, horizon_step, forecast_timestamp_utc, "
            "predicted_demand, model_file, algorithm, model_version, experiment_id, weather_mode) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        ),
        "latest_pointer": session.prepare(
            "INSERT INTO latest_forecast_run_by_station "
            "(station_id, forecast_run_id, forecast_origin_utc, forecast_issued_at_utc, "
            "algorithm, model_version, experiment_id, weather_mode) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
        ),
        "quality": session.prepare(
            "INSERT INTO quality_events_by_station_day "
            "(station_id, event_date, event_timestamp, event_id, severity, rule_name, "
            "message, source_topic, source_partition, source_offset) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        ),
        "actual_read": session.prepare(
            "SELECT station_id, utc_timestamp, actual_demand, temperature "
            "FROM actuals_by_station_month WHERE station_id = ? AND month_bucket = ?"
        ),
        "forecast_read": session.prepare(
            "SELECT station_id, forecast_issued_at_utc, forecast_origin_utc, forecast_run_id, "
            "forecast_timestamp_utc, horizon_step, predicted_demand, model_file, algorithm, "
            "model_version, experiment_id, weather_mode "
            "FROM forecasts_by_station_day WHERE station_id = ? AND origin_date = ?"
        ),
        "latest_pointer_read": session.prepare(
            "SELECT forecast_run_id, forecast_origin_utc, forecast_issued_at_utc, "
            "algorithm, model_version, experiment_id, weather_mode "
            "FROM latest_forecast_run_by_station WHERE station_id = ?"
        ),
        "latest_points_read": session.prepare(
            "SELECT station_id, horizon_step, forecast_timestamp_utc, predicted_demand, model_file, "
            "algorithm, model_version, experiment_id, weather_mode "
            "FROM latest_forecast_points_by_station_run "
            "WHERE station_id = ? AND forecast_run_id = ?"
        ),
        "renewable_forecast": session.prepare(
            "INSERT INTO renewable_forecasts_by_station_day "
            "(station_id, generation_type, origin_date, forecast_origin_utc, forecast_run_id, "
            "forecast_issued_at_utc, horizon_step, forecast_timestamp_utc, "
            "predicted_generation_mwh, model_version, weather_mode) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        ),
        "renewable_forecast_read": session.prepare(
            "SELECT station_id, generation_type, forecast_origin_utc, forecast_run_id, "
            "forecast_issued_at_utc, horizon_step, forecast_timestamp_utc, "
            "predicted_generation_mwh, model_version, weather_mode "
            "FROM renewable_forecasts_by_station_day "
            "WHERE station_id = ? AND generation_type = ? AND origin_date = ?"
        ),
        "renewable_actual": session.prepare(
            "INSERT INTO renewable_actuals_by_station_month "
            "(station_id, generation_type, month_bucket, utc_timestamp, actual_generation_mwh, source) "
            "VALUES (?, ?, ?, ?, ?, ?)"
        ),
        "renewable_actual_read": session.prepare(
            "SELECT station_id, generation_type, utc_timestamp, actual_generation_mwh, source "
            "FROM renewable_actuals_by_station_month "
            "WHERE station_id = ? AND generation_type = ? AND month_bucket = ?"
        ),
        "renewable_latest_read": session.prepare(
            "SELECT latest_origin_utc FROM latest_renewable_origin_by_station_kind "
            "WHERE station_id = ? AND generation_type = ?"
        ),
        "renewable_latest_insert": session.prepare(
            "INSERT INTO latest_renewable_origin_by_station_kind "
            "(station_id, generation_type, latest_origin_utc) VALUES (?, ?, ?) IF NOT EXISTS"
        ),
        "renewable_latest_cas": session.prepare(
            "UPDATE latest_renewable_origin_by_station_kind SET latest_origin_utc = ? "
            "WHERE station_id = ? AND generation_type = ? IF latest_origin_utc = ?"
        ),
        "renewable_actual_latest_read": session.prepare(
            "SELECT latest_timestamp FROM latest_renewable_actual_by_station_kind "
            "WHERE station_id = ? AND generation_type = ?"
        ),
        "renewable_actual_latest_insert": session.prepare(
            "INSERT INTO latest_renewable_actual_by_station_kind "
            "(station_id, generation_type, latest_timestamp) VALUES (?, ?, ?) IF NOT EXISTS"
        ),
        "renewable_actual_latest_cas": session.prepare(
            "UPDATE latest_renewable_actual_by_station_kind SET latest_timestamp = ? "
            "WHERE station_id = ? AND generation_type = ? IF latest_timestamp = ?"
        ),
        "anomaly_write": session.prepare(
            "INSERT INTO anomaly_assessments_by_station_day "
            "(station_id, origin_date, origin_utc, fingerprint, assessed_at_utc, signals, "
            "max_score, records_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?) IF NOT EXISTS USING TTL ?"
        ),
        "anomaly_read": session.prepare(
            "SELECT records_json FROM anomaly_assessments_by_station_day "
            "WHERE station_id = ? AND origin_date = ? AND origin_utc = ? AND fingerprint = ?"
        ),
        "anomaly_day_read": session.prepare(
            "SELECT origin_utc, fingerprint, assessed_at_utc, signals, max_score, records_json "
            "FROM anomaly_assessments_by_station_day WHERE station_id = ? AND origin_date = ?"
        ),
        "anomaly_latest_read": session.prepare(
            "SELECT latest_origin_utc FROM latest_anomaly_origin_by_station WHERE station_id = ?"
        ),
        "anomaly_latest_insert": session.prepare(
            "INSERT INTO latest_anomaly_origin_by_station (station_id, latest_origin_utc) "
            "VALUES (?, ?) IF NOT EXISTS"
        ),
        "anomaly_latest_cas": session.prepare(
            "UPDATE latest_anomaly_origin_by_station SET latest_origin_utc = ? "
            "WHERE station_id = ? IF latest_origin_utc = ?"
        ),
        "bucket_write": session.prepare(
            "INSERT INTO storage_buckets_by_station "
            "(station_id, bucket_type, bucket_value) VALUES (?, ?, ?)"
        ),
        "bucket_read": session.prepare(
            "SELECT bucket_value FROM storage_buckets_by_station "
            "WHERE station_id = ? AND bucket_type = ?"
        ),
        "quality_read": session.prepare(
            "SELECT event_timestamp, event_id, severity, rule_name, message, "
            "source_topic, source_partition, source_offset "
            "FROM quality_events_by_station_day WHERE station_id = ? AND event_date = ?"
        ),
    }
    for statement in _PREPARED.values():
        statement.consistency_level = consistency


def ensure_schema() -> None:
    """Create development keyspace/tables if configured; safe to call repeatedly."""
    from cassandra.cluster import Cluster
    from cassandra.auth import PlainTextAuthProvider

    cfg = _config().get("cassandra", {})
    env_hosts = os.getenv("CASSANDRA_CONTACT_POINTS")
    if env_hosts:
        cfg["hosts"] = [host.strip() for host in env_hosts.split(",") if host.strip()]
    if os.getenv("CASSANDRA_LOCAL_DATACENTER"):
        cfg["local_datacenter"] = os.environ["CASSANDRA_LOCAL_DATACENTER"]
    hosts = cfg.get("hosts") or ["127.0.0.1"]
    auth = None
    if cfg.get("username"):
        auth = PlainTextAuthProvider(
            username=str(cfg["username"]),
            password=str(cfg.get("password") or os.getenv("CASSANDRA_PASSWORD", "")),
        )
    cluster = Cluster(contact_points=hosts, port=int(cfg.get("port", 9042)), auth_provider=auth)
    session = None
    try:
        session = cluster.connect()
        keyspace = str(cfg.get("keyspace", "electricity"))
        replication = cfg.get("replication", {})
        strategy = os.getenv("CASSANDRA_REPLICATION_STRATEGY", str(replication.get("class", "SimpleStrategy")))
        if strategy == "NetworkTopologyStrategy":
            env_rf = os.getenv("CASSANDRA_REPLICATION_RF")
            dc_map = replication.get("datacenters", {str(cfg.get("local_datacenter", "datacenter1")): int(env_rf or 3)})
            replication_map = ", ".join(f"'{k}': {int(v)}" for k, v in dc_map.items())
            replication_clause = f"'class': 'NetworkTopologyStrategy', {replication_map}"
        else:
            replication_clause = (
                "'class': 'SimpleStrategy', 'replication_factor': "
                f"{int(replication.get('replication_factor', 1))}"
            )
        session.execute(
            f"CREATE KEYSPACE IF NOT EXISTS {keyspace} WITH replication = {{{replication_clause}}}"
        )
        session.set_keyspace(keyspace)
        statements = [
            """CREATE TABLE IF NOT EXISTS actuals_by_station_month (
                station_id text, month_bucket text, utc_timestamp timestamp,
                actual_demand double, temperature double,
                PRIMARY KEY ((station_id, month_bucket), utc_timestamp)
            ) WITH CLUSTERING ORDER BY (utc_timestamp ASC)""",
            """CREATE TABLE IF NOT EXISTS latest_actual_month_by_station (
                station_id text PRIMARY KEY, month_bucket text, latest_timestamp timestamp
            )""",
            """CREATE TABLE IF NOT EXISTS forecasts_by_station_day (
                station_id text, origin_date date, forecast_origin_utc timestamp,
                forecast_run_id text, forecast_issued_at_utc timestamp,
                horizon_step smallint, forecast_timestamp_utc timestamp,
                predicted_demand double, model_file text, algorithm text, model_version text,
                experiment_id text, weather_mode text,
                PRIMARY KEY ((station_id, origin_date), forecast_origin_utc, forecast_run_id, horizon_step)
            ) WITH CLUSTERING ORDER BY (forecast_origin_utc DESC, forecast_run_id ASC, horizon_step ASC)""",
            """CREATE TABLE IF NOT EXISTS latest_forecast_points_by_station_run (
                station_id text, forecast_run_id text, horizon_step smallint,
                forecast_timestamp_utc timestamp, predicted_demand double, model_file text,
                algorithm text, model_version text, experiment_id text, weather_mode text,
                PRIMARY KEY ((station_id, forecast_run_id), horizon_step)
            ) WITH CLUSTERING ORDER BY (horizon_step ASC)""",
            """CREATE TABLE IF NOT EXISTS latest_forecast_run_by_station (
                station_id text PRIMARY KEY, forecast_run_id text,
                forecast_origin_utc timestamp, forecast_issued_at_utc timestamp,
                algorithm text, model_version text, experiment_id text, weather_mode text
            )""",
            """CREATE TABLE IF NOT EXISTS renewable_forecasts_by_station_day (
                station_id text, generation_type text, origin_date date,
                forecast_origin_utc timestamp, forecast_run_id text,
                forecast_issued_at_utc timestamp, horizon_step smallint,
                forecast_timestamp_utc timestamp, predicted_generation_mwh double,
                model_version text, weather_mode text,
                PRIMARY KEY ((station_id, generation_type, origin_date),
                    forecast_origin_utc, forecast_run_id, horizon_step)
            ) WITH CLUSTERING ORDER BY (
                forecast_origin_utc DESC, forecast_run_id ASC, horizon_step ASC)""",
            """CREATE TABLE IF NOT EXISTS renewable_actuals_by_station_month (
                station_id text, generation_type text, month_bucket text,
                utc_timestamp timestamp, actual_generation_mwh double, source text,
                PRIMARY KEY ((station_id, generation_type, month_bucket), utc_timestamp)
            ) WITH CLUSTERING ORDER BY (utc_timestamp DESC)""",
            """CREATE TABLE IF NOT EXISTS latest_renewable_origin_by_station_kind (
                station_id text, generation_type text, latest_origin_utc timestamp,
                PRIMARY KEY (station_id, generation_type)
            )""",
            """CREATE TABLE IF NOT EXISTS latest_renewable_actual_by_station_kind (
                station_id text, generation_type text, latest_timestamp timestamp,
                PRIMARY KEY (station_id, generation_type)
            )""",
            """CREATE TABLE IF NOT EXISTS anomaly_assessments_by_station_day (
                station_id text, origin_date date, origin_utc timestamp,
                fingerprint text, assessed_at_utc timestamp, signals int,
                max_score double, records_json text,
                PRIMARY KEY ((station_id, origin_date), origin_utc, fingerprint)
            ) WITH CLUSTERING ORDER BY (origin_utc DESC, fingerprint ASC)""",
            """CREATE TABLE IF NOT EXISTS latest_anomaly_origin_by_station (
                station_id text PRIMARY KEY, latest_origin_utc timestamp
            )""",
            """CREATE TABLE IF NOT EXISTS storage_buckets_by_station (
                station_id text, bucket_type text, bucket_value text,
                PRIMARY KEY ((station_id, bucket_type), bucket_value)
            )""",
            """CREATE TABLE IF NOT EXISTS quality_events_by_station_day (
                station_id text, event_date date, event_timestamp timestamp,
                event_id text, severity text, rule_name text, message text,
                source_topic text, source_partition int, source_offset bigint,
                PRIMARY KEY ((station_id, event_date), event_timestamp, event_id)
            ) WITH CLUSTERING ORDER BY (event_timestamp DESC, event_id ASC)""",
        ]
        for statement in statements:
            session.execute(statement)
    finally:
        if session is not None:
            session.shutdown()
        cluster.shutdown()


def write_actuals(records: list[dict[str, Any]]) -> int:
    if not records:
        return 0
    from cassandra import ConsistencyLevel
    from cassandra.query import BatchStatement, BatchType

    session = get_session()
    grouped: dict[tuple[str, str], list[tuple]] = {}
    for record in records:
        ts = _utc_naive(record["utc_timestamp"])
        partition_key = (str(record["station_id"]), _month_bucket(ts))
        grouped.setdefault(partition_key, []).append((
            str(record["station_id"]), _month_bucket(ts), ts,
            float(record["actual_demand"]), float(record["temperature"]),
        ))
    level_name = os.getenv("CASSANDRA_CONSISTENCY_LEVEL", str(_config().get("cassandra", {}).get("consistency_level", "ONE"))).upper()
    level = getattr(ConsistencyLevel, level_name, ConsistencyLevel.ONE)
    for (station_id, month), values in grouped.items():
        # Keep unlogged batches small, and within one Cassandra partition.
        for start in range(0, len(values), 64):
            batch = BatchStatement(batch_type=BatchType.UNLOGGED, consistency_level=level)
            for value in values[start:start + 64]:
                batch.add(_PREPARED["actual"], value)
            session.execute(batch)
        _register_bucket(session, station_id, "demand_actual_month", month)

    latest_by_station: dict[str, datetime] = {}
    for record in records:
        station_id = str(record["station_id"])
        ts = _utc_naive(record["utc_timestamp"])
        if station_id not in latest_by_station or ts > latest_by_station[station_id]:
            latest_by_station[station_id] = ts
    for station_id, candidate in latest_by_station.items():
        current = session.execute(_PREPARED["latest_actual_read"], (station_id,)).one()
        if current is None or current.latest_timestamp is None or candidate > current.latest_timestamp:
            session.execute(_PREPARED["latest_actual_write"], (station_id, _month_bucket(candidate), candidate))
    return len(records)


def write_forecasts(records: list[dict[str, Any]]) -> int:
    if not records:
        return 0
    session = get_session()
    saved = 0
    first = records[0]
    origin = _utc_naive(first["forecast_origin_utc"])
    issued = _utc_naive(first.get("forecast_issued_at_utc") or datetime.now(timezone.utc))
    run_id = str(first.get("forecast_run_id") or issued.isoformat(timespec="microseconds"))
    station_id = str(first["station_id"])
    algorithm = str(first.get("algorithm", "unknown"))
    model_version = str(first.get("model_version", "unknown"))
    experiment_id = str(first.get("experiment_id", ""))
    weather_mode = str(first.get("weather_mode", "unknown"))

    # Both forecast point collections are single-partition writes. Unlogged
    # batches are appropriate here because each batch targets one partition.
    from cassandra import ConsistencyLevel
    from cassandra.query import BatchStatement, BatchType
    level_name = os.getenv("CASSANDRA_CONSISTENCY_LEVEL", str(_config().get("cassandra", {}).get("consistency_level", "ONE"))).upper()
    level = getattr(ConsistencyLevel, level_name, ConsistencyLevel.ONE)
    history_batch = BatchStatement(batch_type=BatchType.UNLOGGED, consistency_level=level)
    latest_batch = BatchStatement(batch_type=BatchType.UNLOGGED, consistency_level=level)
    for record in records:
        if str(record["station_id"]) != station_id:
            raise ValueError("A Cassandra forecast batch must contain exactly one station.")
        if _utc_naive(record["forecast_origin_utc"]) != origin:
            raise ValueError("A Cassandra forecast batch must contain exactly one forecast origin.")
        target = _utc_naive(record["forecast_timestamp_utc"])
        horizon = int(record["horizon_step"])
        value = float(record["predicted_demand"])
        model_file = str(record.get("model_file", ""))
        history_batch.add(_PREPARED["forecast"], (
            station_id, _origin_date(origin), origin, run_id, issued,
            horizon, target, value, model_file, algorithm, model_version,
            experiment_id, weather_mode,
        ))
        latest_batch.add(_PREPARED["latest_point"], (
            station_id, run_id, horizon, target, value, model_file,
            algorithm, model_version, experiment_id, weather_mode,
        ))
        saved += 1
    session.execute(history_batch)
    session.execute(latest_batch)
    _register_bucket(session, station_id, "demand_forecast_day", _origin_date(origin).isoformat())
    _register_bucket(session, station_id, "demand_run_id", run_id)

    # Update the pointer last; readers continue seeing the previous complete
    # forecast run if either preceding partition write fails.
    session.execute(
        _PREPARED["latest_pointer"],
        (station_id, run_id, origin, issued, algorithm, model_version,
         experiment_id, weather_mode),
    )
    return saved


def write_quality_event(event: dict[str, Any]) -> None:
    session = get_session()
    timestamp = _utc_naive(event.get("event_timestamp") or datetime.now(timezone.utc))
    session.execute(
        _PREPARED["quality"],
        (
            str(event["station_id"]), timestamp.date(), timestamp,
            str(event.get("event_id") or f"{event.get('source_partition', -1)}:{event.get('source_offset', -1)}"),
            str(event.get("severity", "warning")), str(event.get("rule_name", "unknown")),
            str(event.get("message", ""))[:2000], str(event.get("source_topic", "")),
            int(event.get("source_partition", -1)), int(event.get("source_offset", -1)),
        ),
    )
    _register_bucket(session, str(event["station_id"]), "quality_day", timestamp.date().isoformat())


def read_actuals(station: str, limit: int = 5000) -> list[dict[str, Any]]:
    session = get_session()
    cfg = _config().get("cassandra", {})
    months = max(1, int(cfg.get("actual_history_months", 12)))
    latest = session.execute(_PREPARED["latest_actual_read"], (station,)).one()
    if latest is not None and latest.month_bucket:
        year, month = (int(part) for part in latest.month_bucket.split("-", 1))
    else:
        now = datetime.now(timezone.utc)
        year, month = now.year, now.month
    buckets: list[str] = []
    for _ in range(months):
        buckets.append(f"{year:04d}-{month:02d}")
        month -= 1
        if month == 0:
            month, year = 12, year - 1
    rows: list[dict[str, Any]] = []
    for bucket in buckets:
        rows.extend(dict(row._asdict()) for row in session.execute(_PREPARED["actual_read"], (station, bucket)))
        if len(rows) >= limit:
            break
    rows.sort(key=lambda row: row["utc_timestamp"])
    rows = rows[-max(1, int(limit)):]
    for row in rows:
        row["utc_timestamp"] = row["utc_timestamp"].replace(tzinfo=timezone.utc).isoformat()
    return rows


def read_forecasts(station: str, limit: int = 5000) -> list[dict[str, Any]]:
    session = get_session()
    days = max(1, int(_config().get("cassandra", {}).get("forecast_history_days", 14)))
    latest_run = session.execute(_PREPARED["latest_pointer_read"], (station,)).one()
    anchor_day = (latest_run.forecast_origin_utc.date()
                  if latest_run is not None and latest_run.forecast_origin_utc is not None
                  else datetime.now(timezone.utc).date())
    rows: list[dict[str, Any]] = []
    from datetime import timedelta
    for offset in range(days):
        day = anchor_day - timedelta(days=offset)
        rows.extend(dict(row._asdict()) for row in session.execute(_PREPARED["forecast_read"], (station, day)))
        if len(rows) >= limit:
            break
    rows.sort(key=lambda row: (row["forecast_origin_utc"], row["horizon_step"]))
    rows = rows[-max(1, int(limit)):]
    for row in rows:
        for key in ("forecast_issued_at_utc", "forecast_origin_utc", "forecast_timestamp_utc"):
            value = row.get(key)
            if value is not None:
                row[key] = value.replace(tzinfo=timezone.utc).isoformat()
    return rows


def read_latest_forecast(station: str) -> list[dict[str, Any]]:
    session = get_session()
    pointer = session.execute(_PREPARED["latest_pointer_read"], (station,)).one()
    if pointer is None:
        return []
    run_id = pointer.forecast_run_id
    rows = [dict(row._asdict()) for row in session.execute(
        _PREPARED["latest_points_read"], (station, run_id)
    )]
    for row in rows:
        row["forecast_run_id"] = run_id
        row["forecast_origin_utc"] = pointer.forecast_origin_utc.replace(tzinfo=timezone.utc).isoformat()
        row["forecast_issued_at_utc"] = pointer.forecast_issued_at_utc.replace(tzinfo=timezone.utc).isoformat()
        if row.get("forecast_timestamp_utc") is not None:
            row["forecast_timestamp_utc"] = row["forecast_timestamp_utc"].replace(tzinfo=timezone.utc).isoformat()
    return sorted(rows, key=lambda row: row["horizon_step"])


def read_quality_events(station: str, limit: int = 100) -> list[dict[str, Any]]:
    session = get_session()
    from datetime import timedelta
    rows: list[dict[str, Any]] = []
    today = datetime.now(timezone.utc).date()
    for offset in range(14):
        day = today - timedelta(days=offset)
        rows.extend(dict(row._asdict()) for row in session.execute(_PREPARED["quality_read"], (station, day)))
        if len(rows) >= limit:
            break
    rows.sort(key=lambda row: row["event_timestamp"], reverse=True)
    rows = rows[:max(1, int(limit))]
    for row in rows:
        if row.get("event_timestamp") is not None:
            row["event_timestamp"] = row["event_timestamp"].replace(tzinfo=timezone.utc).isoformat()
    return rows


def health_check() -> dict[str, Any]:
    try:
        session = get_session()
        session.execute("SELECT release_version FROM system.local", timeout=3)
        return {"available": True, "backend": "cassandra"}
    except Exception as exc:
        return {"available": False, "backend": "cassandra", "error": str(exc)}


def close() -> None:
    global _CLUSTER, _SESSION, _SCHEMA_READY
    with _LOCK:
        if _SESSION is not None:
            _SESSION.shutdown()
        if _CLUSTER is not None:
            _CLUSTER.shutdown()
        _SESSION = None
        _CLUSTER = None
        _PREPARED.clear()
        _SCHEMA_READY = False


# --- Renewable generation and forecast anomaly operational storage ---

def _register_bucket(session, station: str, kind: str, value: str) -> None:
    # Station-scoped partition manifest. Required for deleting only this station.
    session.execute(_PREPARED["bucket_write"], (str(station), kind, str(value)))


def _advance_latest(session, read_key: str, insert_key: str, cas_key: str,
                    read_args: tuple, insert_args: tuple, new_ts: datetime,
                    cas_args_builder) -> None:
    """Monotonic pointer update. Older replay origins must not hide newer runs."""
    for _ in range(4):
        row = session.execute(_PREPARED[read_key], read_args).one()
        prior = getattr(row, "latest_origin_utc", None) if row is not None else None
        if prior is None and row is not None:
            prior = getattr(row, "latest_timestamp", None)
        if prior is not None and _utc_naive(prior) >= new_ts:
            return
        if prior is None:
            result = session.execute(_PREPARED[insert_key], insert_args).one()
        else:
            result = session.execute(_PREPARED[cas_key], cas_args_builder(prior)).one()
        if result is not None:
            # Cassandra's driver may expose the conditional result column as
            # "[applied]", "applied" or only as the first positional field.
            applied = getattr(result, "[applied]", None)
            if applied is None:
                applied = getattr(result, "applied", None)
            if applied is None and hasattr(result, "_asdict"):
                applied = result._asdict().get("[applied]")
            if applied is None:
                try:
                    applied = result[0]
                except (TypeError, IndexError, KeyError):
                    applied = False
            if applied:
                return
    # Another writer raced: the insert itself is durable, the next query
    # still uses the latest successfully published pointer.
    LOG.warning("Latest pointer update raced with another writer: %s", read_args)


def _batch_write(records: list[tuple], statement_name: str, level=None):
    from cassandra.query import BatchStatement, BatchType
    from cassandra import ConsistencyLevel
    session = get_session()
    consistency = level or getattr(ConsistencyLevel,
        os.getenv("CASSANDRA_CONSISTENCY_LEVEL", str(_config().get("cassandra", {}).get("consistency_level", "ONE"))).upper(),
        ConsistencyLevel.ONE)
    for offset in range(0, len(records), 64):
        batch = BatchStatement(batch_type=BatchType.UNLOGGED, consistency_level=consistency)
        for row in records[offset:offset + 64]:
            batch.add(_PREPARED[statement_name], row)
        session.execute(batch)
    return session


def write_renewable_forecasts(records: list[dict[str, Any]]) -> int:
    """Idempotent upserts partitioned by station/kind/origin day.

    A batch can contain both wind and solar, and retries use stable run keys.
    """
    if not records:
        return 0
    partitions: dict[tuple[str, str, date], list[tuple]] = {}
    latest: dict[tuple[str, str], datetime] = {}
    for item in records:
        kind = str(item["generation_type"])
        if kind not in {"solar", "wind"}:
            raise ValueError(f"Unknown generation type {kind}")
        station = str(item["station_id"])
        origin = _utc_naive(item["forecast_origin_utc"])
        target = _utc_naive(item["forecast_timestamp_utc"])
        issued = _utc_naive(item.get("forecast_issued_at_utc") or origin)
        key = (station, kind, origin.date())
        run_id = str(item["forecast_run_id"])
        partitions.setdefault(key, []).append((
            station, kind, origin.date(), origin, run_id, issued,
            int(item["horizon_step"]), target,
            float(item["predicted_generation_mwh"]),
            str(item.get("model_version") or ""), str(item.get("weather_mode") or ""),
        ))
        latest[(station, kind)] = max(latest.get((station, kind), origin), origin)
    session = get_session()
    for (station, kind, day), items in partitions.items():
        _batch_write(items, "renewable_forecast")
        _register_bucket(session, station, f"renewable_forecast_day_{kind}", day.isoformat())
    for (station, kind), candidate in latest.items():
        _advance_latest(session, "renewable_latest_read", "renewable_latest_insert", "renewable_latest_cas",
            (station, kind), (station, kind, candidate), candidate,
            lambda old, s=station, k=kind, t=candidate: (t, s, k, old))
    return len(records)


def write_renewable_actuals(records: list[dict[str, Any]]) -> int:
    if not records:
        return 0
    partitions: dict[tuple[str, str, str], list[tuple]] = {}
    latest: dict[tuple[str, str], datetime] = {}
    for item in records:
        station, kind = str(item["station_id"]), str(item["generation_type"])
        if kind not in {"solar", "wind"}:
            raise ValueError(f"Unknown generation type {kind}")
        timestamp = _utc_naive(item["utc_timestamp"])
        month = _month_bucket(timestamp)
        partitions.setdefault((station, kind, month), []).append((
            station, kind, month, timestamp, float(item["actual_generation_mwh"]),
            str(item.get("source") or "stream_observed"),
        ))
        latest[(station, kind)] = max(latest.get((station, kind), timestamp), timestamp)
    session = get_session()
    for (station, kind, month), items in partitions.items():
        _batch_write(items, "renewable_actual")
        _register_bucket(session, station, f"renewable_actual_month_{kind}", month)
    for (station, kind), candidate in latest.items():
        _advance_latest(session, "renewable_actual_latest_read", "renewable_actual_latest_insert", "renewable_actual_latest_cas",
            (station, kind), (station, kind, candidate), candidate,
            lambda old, s=station, k=kind, t=candidate: (t, s, k, old))
    return len(records)


def _iso_rows(records: list[dict], fields: tuple[str, ...]) -> list[dict]:
    for row in records:
        for field in fields:
            if row.get(field) is not None:
                row[field] = _utc_naive(row[field]).replace(tzinfo=timezone.utc).isoformat()
    return records


def _recent_months(anchor: datetime, count: int):
    year, month = anchor.year, anchor.month
    for _ in range(count):
        yield f"{year:04d}-{month:02d}"
        month -= 1
        if month < 1:
            year -= 1
            month = 12


def read_renewable_forecasts(station: str, limit: int = 20000,
                              end_time: str | None = None) -> list[dict[str, Any]]:
    from datetime import timedelta
    session = get_session()
    cfg = _config().get("cassandra", {})
    days = max(1, min(90, int(cfg.get("renewable_forecast_history_days", 14))))
    cutoff = _utc_naive(end_time) if end_time else None
    rows = []
    # Reserve capacity for each generation source. An active solar feed must not
    # evict all wind records from the shared response if wind is behind.
    per_kind_limit = max(96, int(limit) // 2)
    for kind in ("solar", "wind"):
        pointer = session.execute(_PREPARED["renewable_latest_read"], (station, kind)).one()
        anchor = cutoff or (pointer.latest_origin_utc if pointer else None)
        if anchor is None:
            continue
        kind_rows = []
        for offset in range(days):
            day = _utc_naive(anchor).date() - timedelta(days=offset)
            for item in session.execute(_PREPARED["renewable_forecast_read"], (station, kind, day)):
                obj = dict(item._asdict())
                if cutoff and _utc_naive(obj["forecast_origin_utc"]) > cutoff:
                    continue
                kind_rows.append(obj)
            if len(kind_rows) >= per_kind_limit:
                break
        kind_rows.sort(key=lambda r: (_utc_naive(r["forecast_origin_utc"]), r["horizon_step"]))
        rows.extend(kind_rows[-per_kind_limit:])
    rows.sort(key=lambda r: (_utc_naive(r["forecast_origin_utc"]), r["generation_type"], r["horizon_step"]))
    return _iso_rows(rows, ("forecast_issued_at_utc", "forecast_origin_utc", "forecast_timestamp_utc"))


def read_renewable_actuals(station: str, limit: int = 20000,
                            end_time: str | None = None) -> list[dict[str, Any]]:
    session = get_session()
    cfg = _config().get("cassandra", {})
    months = max(1, min(36, int(cfg.get("renewable_actual_history_months", 12))))
    cutoff = _utc_naive(end_time) if end_time else None
    rows = []
    for kind in ("solar", "wind"):
        pointer = session.execute(_PREPARED["renewable_actual_latest_read"], (station, kind)).one()
        anchor = cutoff or (pointer.latest_timestamp if pointer else None)
        if anchor is None:
            continue
        for month in _recent_months(_utc_naive(anchor), months):
            for item in session.execute(_PREPARED["renewable_actual_read"], (station, kind, month)):
                obj = dict(item._asdict())
                if cutoff and _utc_naive(obj["utc_timestamp"]) > cutoff:
                    continue
                rows.append(obj)
            if len(rows) >= 2 * max(1, limit):
                break
    rows.sort(key=lambda r: _utc_naive(r["utc_timestamp"]))
    return _iso_rows(rows[-max(1, int(limit)):], ("utc_timestamp",))


def write_anomaly_assessment(station: str, origin_utc: str, fingerprint: str,
                             records: list[dict]) -> None:
    import json
    import math
    session = get_session()
    origin = _utc_naive(origin_utc)
    ttl = max(60, min(365 * 86400, int(_config().get("cassandra", {}).get("anomaly_ttl_days", 30)) * 86400))
    scores = [float(r["combined_anomaly_score"]) for r in records
              if r.get("combined_anomaly_score") is not None and math.isfinite(float(r["combined_anomaly_score"]))]
    session.execute(_PREPARED["anomaly_write"], (
        station, origin.date(), origin, fingerprint, datetime.now(timezone.utc).replace(tzinfo=None),
        sum(r.get("classification") == "YES" for r in records), max(scores) if scores else None,
        json.dumps(records, separators=(",", ":"), allow_nan=False), ttl,
    ))
    _register_bucket(session, station, "anomaly_day", origin.date().isoformat())
    _advance_latest(session, "anomaly_latest_read", "anomaly_latest_insert", "anomaly_latest_cas",
        (station,), (station, origin), origin, lambda old: (origin, station, old))


def read_anomaly_assessment(station: str, origin_utc: str, fingerprint: str) -> list[dict] | None:
    import json
    session = get_session()
    origin = _utc_naive(origin_utc)
    item = session.execute(_PREPARED["anomaly_read"], (station, origin.date(), origin, fingerprint)).one()
    return json.loads(item.records_json) if item is not None else None


def read_recent_anomaly_assessments(station: str, limit: int = 24) -> list[dict]:
    import json
    from datetime import timedelta
    session = get_session()
    item = session.execute(_PREPARED["anomaly_latest_read"], (station,)).one()
    if item is None or item.latest_origin_utc is None:
        return []
    days = max(1, min(365, int(_config().get("cassandra", {}).get("anomaly_history_days", 30))))
    rows = []
    for offset in range(days):
        day = _utc_naive(item.latest_origin_utc).date() - timedelta(days=offset)
        rows.extend(dict(r._asdict()) for r in session.execute(_PREPARED["anomaly_day_read"], (station, day)))
        if len(rows) >= limit:
            break
    rows.sort(key=lambda r: (r["origin_utc"], r["assessed_at_utc"]), reverse=True)
    results = []
    for r in rows[:max(1, min(200, int(limit)))]:
        records = json.loads(r["records_json"])
        results.append({
            "origin_utc": _utc_naive(r["origin_utc"]).replace(tzinfo=timezone.utc).isoformat(),
            "fingerprint": r["fingerprint"][:12],
            "assessed_at_utc": _utc_naive(r["assessed_at_utc"]).replace(tzinfo=timezone.utc).isoformat(),
            "signals": r["signals"], "max_score": r["max_score"],
            "flagged_target_times": [v.get("timestamp_utc") for v in records if v.get("classification") == "YES"],
            "flagged_points": [{"timestamp_utc": v.get("timestamp_utc"),
                                "forecast_demand_mw": v.get("forecast_demand_mw"),
                                "combined_anomaly_score": v.get("combined_anomaly_score"),
                                "spike_cause": v.get("spike_cause")}
                               for v in records if v.get("classification") == "YES"
                               and v.get("spike_cause") in ("LEVEL_SPIKE", "RAMP_UP_SPIKE")],
        })
    return results


def clear_station_operational_data(station: str, *, dry_run: bool = True) -> dict[str, Any]:
    """Delete only partitions registered for one station; no tables are truncated.

    Existing pre-manifest Cassandra rows must be migrated separately. Refuse
    clearing a legacy station with demand pointers but no registered partitions.
    """
    session = get_session()
    kinds = (
        "demand_actual_month", "demand_forecast_day", "demand_run_id",
        "renewable_actual_month_solar", "renewable_actual_month_wind",
        "renewable_forecast_day_solar", "renewable_forecast_day_wind",
        "anomaly_day", "quality_day",
    )
    entries = {kind: [r.bucket_value for r in session.execute(_PREPARED["bucket_read"], (station, kind))]
               for kind in kinds}
    demand_actual = session.execute(_PREPARED["latest_actual_read"], (station,)).one()
    demand_forecast = session.execute(_PREPARED["latest_pointer_read"], (station,)).one()
    if ((demand_actual is not None and getattr(demand_actual, "latest_timestamp", None) is not None and not entries["demand_actual_month"])
        or (demand_forecast is not None and getattr(demand_forecast, "forecast_run_id", None) is not None
            and not entries["demand_forecast_day"])):
        raise RuntimeError("Legacy Cassandra demand data has no partition manifest. "
                           "Refusing an incomplete station reset; migrate or archive those partitions first.")
    summary = {"station": station, "dry_run": dry_run,
               "partition_counts": {kind: len(values) for kind, values in entries.items()}}
    if dry_run:
        return summary
    statements = {
        "demand_actual_month": "DELETE FROM actuals_by_station_month WHERE station_id = %s AND month_bucket = %s",
        "demand_forecast_day": "DELETE FROM forecasts_by_station_day WHERE station_id = %s AND origin_date = %s",
        "demand_run_id": "DELETE FROM latest_forecast_points_by_station_run WHERE station_id = %s AND forecast_run_id = %s",
        "renewable_actual_month_solar": "DELETE FROM renewable_actuals_by_station_month WHERE station_id = %s AND generation_type = %s AND month_bucket = %s",
        "renewable_actual_month_wind": "DELETE FROM renewable_actuals_by_station_month WHERE station_id = %s AND generation_type = %s AND month_bucket = %s",
        "renewable_forecast_day_solar": "DELETE FROM renewable_forecasts_by_station_day WHERE station_id = %s AND generation_type = %s AND origin_date = %s",
        "renewable_forecast_day_wind": "DELETE FROM renewable_forecasts_by_station_day WHERE station_id = %s AND generation_type = %s AND origin_date = %s",
        "anomaly_day": "DELETE FROM anomaly_assessments_by_station_day WHERE station_id = %s AND origin_date = %s",
        "quality_day": "DELETE FROM quality_events_by_station_day WHERE station_id = %s AND event_date = %s",
    }
    for kind, values in entries.items():
        for value in values:
            if "forecast_day" in kind or kind in ("anomaly_day", "quality_day"):
                arg = date.fromisoformat(value)
            else:
                arg = value
            if kind.startswith("renewable_"):
                args = (station, "solar" if kind.endswith("solar") else "wind", arg)
            else:
                args = (station, arg)
            session.execute(statements[kind], args)
            session.execute("DELETE FROM storage_buckets_by_station "
                            "WHERE station_id = %s AND bucket_type = %s AND bucket_value = %s", (station, kind, value))
    for table in ("latest_actual_month_by_station", "latest_forecast_run_by_station",
                  "latest_anomaly_origin_by_station"):
        session.execute(f"DELETE FROM {table} WHERE station_id = %s", (station,))
    for table in ("latest_renewable_origin_by_station_kind", "latest_renewable_actual_by_station_kind"):
        for kind in ("solar", "wind"):
            session.execute(f"DELETE FROM {table} WHERE station_id = %s AND generation_type = %s", (station, kind))
    summary["cleared"] = True
    return summary
