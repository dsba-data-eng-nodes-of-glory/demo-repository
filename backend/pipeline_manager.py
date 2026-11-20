"""Station-aware local process manager for the electricity forecasting POC.

Run FastAPI with one worker and without --reload. Kafka broker is managed
separately; station topic creation is performed through Kafka AdminClient.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock, RLock, Thread
from typing import Any

import yaml

from src.spark.control import read_control, write_control

from scripts.station_replay_helper import (
    prepare_station_outputs,
    validate_station_output_files,
    delete_station_output_rows,
)

LOG = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
SPARK_CONFIG = ROOT / "config" / "spark.yaml"
STATION_CONFIG = ROOT / "config" / "stations.yaml"
KAFKA_CONFIG = ROOT / "config" / "kafka.yaml"
STORAGE_CONFIG = ROOT / "config" / "storage.yaml"


STATION_REGISTRY_LOCK = RLock()
_REGISTRY_CACHE_LOCK = Lock()
_REGISTRY_CACHE: tuple[tuple, dict] | None = None


def read_station_config() -> dict[str, Any]:
    global _REGISTRY_CACHE
    path = STATION_CONFIG
    with _REGISTRY_CACHE_LOCK:
        stat = path.stat()
        signature = (path, stat.st_mtime_ns, stat.st_size)
        if _REGISTRY_CACHE is None or _REGISTRY_CACHE[0] != signature:
            with path.open(encoding="utf-8") as handle:
                _REGISTRY_CACHE = (signature, yaml.safe_load(handle) or {})
        return deepcopy(_REGISTRY_CACHE[1])


def write_station_config(config: dict[str, Any]) -> None:
    """Atomically persist the station registry."""
    temporary = STATION_CONFIG.with_suffix(".yaml.tmp")
    temporary.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    temporary.replace(STATION_CONFIG)


def read_spark_config() -> dict[str, Any]:
    if not SPARK_CONFIG.exists():
        return {}
    with SPARK_CONFIG.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def read_kafka_config() -> dict[str, Any]:
    if not KAFKA_CONFIG.exists():
        config = {"bootstrap_servers": "127.0.0.1:9092"}
    else:
        with KAFKA_CONFIG.open(encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    if os.getenv("KAFKA_BOOTSTRAP_SERVERS"):
        config["bootstrap_servers"] = os.environ["KAFKA_BOOTSTRAP_SERVERS"]
    if os.getenv("KAFKA_TOPIC_PARTITIONS"):
        config["topic_partitions"] = int(os.environ["KAFKA_TOPIC_PARTITIONS"])
    if os.getenv("KAFKA_TOPIC_REPLICATION_FACTOR"):
        config["replication_factor"] = int(os.environ["KAFKA_TOPIC_REPLICATION_FACTOR"])
    return config


def resolve_root_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def _tail_log(path: Path, lines: int) -> str:
    try:
        with path.open("rb") as handle:
            end = handle.seek(0, os.SEEK_END)
            start = max(0, end - 128 * 1024)
            handle.seek(start)
            if start:
                handle.readline()
            return "\n".join(handle.read().decode("utf-8", errors="replace").splitlines()[-lines:])
    except FileNotFoundError:
        return ""


@dataclass
class ServiceRestart:
    command: list[str]
    name: str
    station: str | None
    process: subprocess.Popen | None
    started_at: float
    next_attempt: float | None = None
    failures: int = 0


class PipelineManager:
    """Own independent local processes for every station and service."""

    SERVICE_NAMES = ("spark", "predictor", "producer", "training", "live_actuals")
    RESTART_DELAYS = (2, 4, 8, 16, 30, 30)
    STABLE_SECONDS = 120

    def __init__(self) -> None:
        self.processes: dict[str, subprocess.Popen] = {}
        self.log_handles: dict[str, Any] = {}
        self.lock = Lock()
        self.lifecycle_lock = STATION_REGISTRY_LOCK
        self._supervised: dict[str, ServiceRestart] = {}
        self._supervisor_stop = Event()
        self._supervisor_thread: Thread | None = None
        # Java version detection starts a subprocess on Windows. Status panels
        # must never repeat that check for every station on every refresh.
        self._runtime_status_cache: dict[str, Any] | None = None
        self._runtime_status_cache_at = 0.0
        self._cassandra_status_cache: dict[str, Any] | None = None
        self._cassandra_status_cache_at = 0.0
        self._java_version: tuple[Path, str] | None = None
        self._spark_dir: Path | None = None
        self._spark_requests: dict[str, str] = {}

    def get_runtime_environment(self) -> dict[str, Any]:
        env = os.environ.copy()
        python_exe = Path(sys.executable).resolve()
        if not python_exe.exists():
            raise RuntimeError(f"Python executable not found: {python_exe}")
        env["PYSPARK_PYTHON"] = str(python_exe)
        env["PYSPARK_DRIVER_PYTHON"] = str(python_exe)
        configured_java = read_spark_config().get("java_home") if os.name == "nt" else None
        java_home = configured_java or env.get("JAVA_HOME")
        if java_home:
            java_home_path = Path(java_home).expanduser().resolve()
            java_exe = java_home_path / "bin" / ("java.exe" if os.name == "nt" else "java")
        else:
            java_exe_value = shutil.which("java")
            if not java_exe_value:
                raise RuntimeError("Java is not available. Install a supported JDK or set JAVA_HOME.")
            java_exe = Path(java_exe_value).resolve()
            java_home_path = java_exe.parent.parent
        if not java_exe.exists():
            raise RuntimeError(f"Java executable not found: {java_exe}")
        env["JAVA_HOME"] = str(java_home_path)
        env["PATH"] = str(java_home_path / "bin") + os.pathsep + env.get("PATH", "")
        cached_version = self._java_version
        if cached_version is None or cached_version[0] != java_exe:
            result = subprocess.run([str(java_exe), "-version"], capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                raise RuntimeError("Java version check failed: " + result.stderr)
            self._java_version = cached_version = (java_exe, result.stderr + result.stdout)
        return {
            "env": env,
            "python_executable": str(python_exe),
            "java_home": str(java_home_path),
            "java_executable": str(java_exe),
            "java_version": cached_version[1],
        }

    def kafka_available(self, host: str | None = None, port: int | None = None) -> bool:
        if host is None or port is None:
            first = str(read_kafka_config().get("bootstrap_servers", "127.0.0.1:9092")).split(",")[0].strip()
            if ":" in first:
                host_value, port_value = first.rsplit(":", 1)
                host = host or host_value
                port = port or int(port_value)
            else:
                host = host or first
                port = port or 9092
        try:
            with socket.create_connection((host, int(port)), timeout=2):
                return True
        except OSError:
            return False

    def cassandra_status(self) -> dict[str, Any]:
        """Report whether the configured operational store is reachable.

        The dashboard polls station status every few seconds. A TCP probe is
        cached so a stopped Cassandra node cannot stall that refresh, and the
        API process does not open a driver session just to paint the chip.
        """
        now = time.monotonic()
        with self.lock:
            cached = self._cassandra_status_cache
            if cached is not None and now - self._cassandra_status_cache_at < 5.0:
                return dict(cached)
        from src.storage.forecast_store import storage_backend

        backend = storage_backend()
        status: dict[str, Any] = {
            "backend": backend,
            "available": False,
            "host": None,
            "port": None,
        }
        if backend == "cassandra":
            cfg: dict[str, Any] = {}
            if STORAGE_CONFIG.exists():
                with STORAGE_CONFIG.open(encoding="utf-8") as handle:
                    cfg = (yaml.safe_load(handle) or {}).get("cassandra") or {}
            hosts = [host.strip() for host in os.getenv("CASSANDRA_CONTACT_POINTS", "").split(",") if host.strip()]
            host = hosts[0] if hosts else str((cfg.get("hosts") or ["127.0.0.1"])[0])
            port = int(os.getenv("CASSANDRA_PORT") or cfg.get("port") or 9042)
            status["host"] = host
            status["port"] = port
            try:
                with socket.create_connection((host, port), timeout=1):
                    status["available"] = True
            except OSError:
                status["available"] = False
        with self.lock:
            self._cassandra_status_cache = dict(status)
            self._cassandra_status_cache_at = time.monotonic()
        return status

    @staticmethod
    def process_key(name: str, station: str | None = None) -> str:
        return f"{station}:{name}" if station else name

    @staticmethod
    def log_path(name: str, station: str | None = None) -> Path:
        return LOG_DIR / (f"{station}_{name}.log" if station else f"{name}.log")

    def _watch_services(self) -> None:
        while not self._supervisor_stop.wait(1):
            try:
                self._restart_failed_services()
            except Exception:
                LOG.exception("Service supervisor check failed.")
            try:
                from backend.onboarding_training import advance_initial_training
                advance_initial_training(self)
            except Exception:
                LOG.exception("Initial training queue check failed.")

    def start_supervisor(self) -> None:
        with self.lock:
            if self._supervisor_thread is None or not self._supervisor_thread.is_alive():
                self._supervisor_stop.clear()
                self._supervisor_thread = Thread(target=self._watch_services, name="pipeline-supervisor", daemon=True)
                self._supervisor_thread.start()

    def _supervise(self, name: str, command: list[str], station: str | None = None) -> None:
        key = self.process_key(name, station)
        with self.lock:
            process = self.processes.get(key)
            if process is None:
                return
            previous = self._supervised.get(key)
            if previous is None or previous.process is not process:
                self._supervised[key] = ServiceRestart(command, name, station, process, time.monotonic())
        self.start_supervisor()

    def shutdown_supervisor(self) -> None:
        self._supervisor_stop.set()
        with self.lock:
            self._supervised.clear()
            thread = self._supervisor_thread
        if thread is not None:
            thread.join(timeout=3)

    def _restart_failed_services(self) -> None:
        with self.lock:
            services = list(self._supervised.items())
        for key, service in services:
            with self.lock:
                if self._supervised.get(key) is not service:
                    continue
                code = service.process.poll() if service.process is not None else -1
                if code is None or (service.name == "producer" and code == 0):
                    continue
                now = time.monotonic()
                if service.next_attempt is None:
                    if now - service.started_at >= self.STABLE_SECONDS:
                        service.failures = 0
                    if service.failures >= len(self.RESTART_DELAYS):
                        handle = self.log_handles.pop(key, None)
                        if handle:
                            handle.close()
                        continue
                    service.next_attempt = now + self.RESTART_DELAYS[service.failures]
                    LOG.warning("%s exited (%s). Automatic retry %d in %s seconds.",
                                key, code, service.failures + 1, self.RESTART_DELAYS[service.failures])
                    continue
                if now < service.next_attempt:
                    continue
            # Resolve the runtime without holding the process lock, then recheck
            # that a concurrent manual stop did not cancel the restart.
            try:
                runtime = self.get_runtime_environment()
                with self.lock:
                    if self._supervised.get(key) is not service or self._supervisor_stop.is_set():
                        continue
                    service.failures += 1
                    service.next_attempt = None
                    service.started_at = time.monotonic()
                    service.process = None
                    self._spawn_locked(service.name, service.command, service.station, runtime)
                    service.process = self.processes[key]
                LOG.info("Automatically restarted %s (attempt %d).", key, service.failures)
            except Exception:
                with self.lock:
                    if self._supervised.get(key) is service and service.next_attempt is not None:
                        service.failures += 1
                        service.next_attempt = None
                        service.started_at = time.monotonic()
                LOG.exception("Could not restart %s.", key)

    def status(self) -> dict[str, Any]:
        with self.lock:
            processes = {}
            for key, proc in self.processes.items():
                code = proc.poll()
                service = self._supervised.get(key)
                name = key.rsplit(":", 1)[-1]
                state = "running" if code is None else "stopped"
                if code is not None and service is not None:
                    if name == "producer" and code == 0:
                        state = "completed"
                    elif service.next_attempt is None and service.failures >= len(self.RESTART_DELAYS):
                        state = "failed"
                    else:
                        state = "restarting"
                processes[key] = {
                    "running": code is None,
                    "pid": proc.pid,
                    "return_code": code,
                    "state": state,
                    "restart_attempts": service.failures if service is not None else 0,
                    "retry_in_seconds": (max(0, round(service.next_attempt - time.monotonic()))
                                         if service is not None and service.next_attempt is not None else None),
                }
        shared_process = processes.get("spark_shared")
        for station, state in self._shared_query_status(shared_process).items():
            processes[self.process_key("spark", station)] = state
        now = time.monotonic()
        with self.lock:
            cached = self._runtime_status_cache
            cache_age = now - self._runtime_status_cache_at
        if cached is None or cache_age >= 120.0:
            try:
                runtime_info = self.get_runtime_environment()
                runtime = {
                    "python_executable": runtime_info["python_executable"],
                    "java_home": runtime_info["java_home"],
                    "java_executable": runtime_info["java_executable"],
                    "java_version": runtime_info["java_version"].strip(),
                }
            except Exception as exc:
                runtime = {"error": str(exc)}
            with self.lock:
                self._runtime_status_cache = runtime
                self._runtime_status_cache_at = time.monotonic()
        else:
            runtime = cached
        return {
            "kafka_tcp_available": self.kafka_available(),
            "cassandra": self.cassandra_status(),
            "runtime": runtime,
            "processes": processes,
        }

    def station_status(self, station: str) -> dict[str, Any]:
        snapshot = self.status()
        selected = {
            service: snapshot["processes"].get(self.process_key(service, station), {
                "running": False, "pid": None, "return_code": None
            })
            for service in self.SERVICE_NAMES
        }
        cfg = read_station_config().get("stations", {}).get(station, {})
        active_model = cfg.get("active_model") or {}
        active_artifact = active_model.get("artifact_path")
        checkpoint = resolve_root_path(active_artifact or cfg.get("model_checkpoint", "__missing_model__"))
        return {
            "station": station,
            "kafka_tcp_available": snapshot["kafka_tcp_available"],
            "cassandra": snapshot.get("cassandra") or {"backend": "csv", "available": False},
            "services": selected,
            "quality_consumer": snapshot["processes"].get("quality", {"running": False, "pid": None, "return_code": None}),
            "model_checkpoint": str(checkpoint),
            "model_ready": checkpoint.is_file(),
            "active_model": active_model or None,
            "topics": {
                "raw": cfg.get("raw_topic"),
                "featured": cfg.get("featured_topic"),
            },
        }

    def start(self, name: str, command: list[str], station: str | None = None) -> dict[str, Any]:
        key = self.process_key(name, station)
        with self.lock:
            existing = self.processes.get(key)
            if existing and existing.poll() is None:
                return {"name": name, "station": station, "started": False, "message": "Already running", "pid": existing.pid}
        runtime = (self.get_runtime_environment() if name in ("spark", "spark_shared")
                   else {"env": os.environ.copy()})
        with self.lock:
            existing = self.processes.get(key)
            if existing and existing.poll() is None:
                return {"name": name, "station": station, "started": False, "message": "Already running", "pid": existing.pid}
            return self._spawn_locked(name, command, station, runtime)

    def _spawn_locked(self, name: str, command: list[str], station: str | None, runtime: dict) -> dict[str, Any]:
        key = self.process_key(name, station)
        log_path = self.log_path(name, station)
        log_handle = log_path.open("a", encoding="utf-8")
        kwargs: dict[str, Any] = {}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        env = runtime["env"]
        if name == "predictor":
            threads = os.getenv("ELECTRICITY_MODEL_THREADS", "1")
            env = {**env, "OMP_NUM_THREADS": threads, "MKL_NUM_THREADS": threads,
                   "OPENBLAS_NUM_THREADS": threads, "NUMEXPR_NUM_THREADS": threads}
        try:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log_handle,
                                       stderr=subprocess.STDOUT, text=True, env=env, **kwargs)
        except Exception:
            log_handle.close()
            raise
        old_log = self.log_handles.get(key)
        if old_log:
            old_log.close()
        self.processes[key] = process
        self.log_handles[key] = log_handle
        return {"name": name, "station": station, "started": True, "pid": process.pid, "log": str(log_path)}

    def stop(self, name: str, station: str | None = None) -> dict[str, Any]:
        if name == "spark" and station is not None and (
            station in self._spark_requests or station in self._shared_query_status()
        ):
            return self._stop_shared_query(station)
        key = self.process_key(name, station)
        with self.lock:
            self._supervised.pop(key, None)
            proc = self.processes.get(key)
            if not proc or proc.poll() is not None:
                handle = self.log_handles.pop(key, None)
                if handle:
                    handle.close()
                return {"name": name, "station": station, "stopped": False, "message": "Not running"}
        try:
            if os.name == "nt":
                proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                proc.terminate()
            proc.wait(timeout=10)
        except Exception:
            if os.name == "nt" and name in ("spark", "spark_shared"):
                subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                               capture_output=True, timeout=10, check=False)
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=5)
        with self.lock:
            if self.processes.get(key) is proc:
                handle = self.log_handles.pop(key, None)
                if handle:
                    handle.close()
        return {"name": name, "station": station, "stopped": True}

    def _shared_query_status(self, process: dict | None = None) -> dict:
        directory = self._spark_dir
        if directory is None:
            return {}
        if process is None:
            with self.lock:
                child = self.processes.get("spark_shared")
                if child and child.poll() is None:
                    process = {"running": True, "state": "running", "pid": child.pid}
                else:
                    process = {"running": False, "state": "stopped", "pid": None}
        try:
            report = read_control(directory / "status.json") if process["running"] else {}
        except OSError:
            report = {}
        reported = report.get("stations", {}) if report.get("pid") == process.get("pid") else {}
        requested = self._spark_requests.copy()
        states = {}
        for station in requested.keys() | reported.keys():
            item = reported.get(station, {})
            state = "stopping" if station not in requested else (
                item["state"] if item.get("token") == requested[station] else
                ("starting" if process["running"] else process["state"])
            )
            states[station] = {
                "running": state == "running", "state": state, "pid": process.get("pid"),
                "return_code": process.get("return_code"), "shared_worker": True,
            }
            if item.get("error") and item.get("token") == requested.get(station):
                states[station]["message"] = item["error"]
        return states

    def _start_shared_query(self, station: str) -> bool:
        if self._spark_dir is None:
            self._spark_dir = Path(tempfile.mkdtemp(prefix="electricity-spark-"))
        directory = self._spark_dir
        new = station not in self._spark_requests
        if new:
            self._spark_requests[station] = uuid.uuid4().hex
        command = [sys.executable, "-u", "-m", "src.spark.streaming", "--control-dir", str(directory)]
        try:
            write_control(directory / "desired.json", self._spark_requests)
            worker = self.start("spark_shared", command)
            if worker["started"]:
                self._supervise("spark_shared", command)
            return new or worker["started"]
        except Exception:
            if new:
                self._spark_requests.pop(station, None)
                write_control(directory / "desired.json", self._spark_requests)
            raise

    def _stop_shared_query(self, station: str) -> dict:
        directory = self._spark_dir
        previous = self._spark_requests.pop(station, None)
        try:
            write_control(directory / "desired.json", self._spark_requests)
        except OSError:
            if previous is not None:
                self._spark_requests[station] = previous
            raise
        if not self._spark_requests:
            self.stop("spark_shared")
        else:
            deadline = time.monotonic() + 12
            while station in self._shared_query_status() and time.monotonic() < deadline:
                time.sleep(0.1)
        pending = station in self._shared_query_status()
        return {"name": "spark", "station": station, "stopped": not pending,
                "message": "Shared Spark query is stopping" if pending else "Shared Spark query stopped"}

    def ensure_station_topics(self, station: str) -> dict[str, Any]:
        """Create missing station topics. Never deletes or truncates topics."""
        config = read_station_config()
        stations = config.get("stations", {})
        kafka_cfg = read_kafka_config()
        if station not in stations:
            raise ValueError(f"Unknown station '{station}'.")
        entry = stations[station]
        kafka_cfg = read_kafka_config()
        topic_names = [entry.get("raw_topic"), entry.get("featured_topic"),
                       kafka_cfg.get("dead_letter_topic", "electricity-dlq")]
        if not all(topic_names):
            raise ValueError(f"Station '{station}' must define raw_topic and featured_topic.")
        if not self.kafka_available():
            raise RuntimeError(
                f"Kafka bootstrap servers are not reachable: {kafka_cfg.get('bootstrap_servers')}. Start Kafka first."
            )
        try:
            from kafka.admin import KafkaAdminClient, NewTopic
        except ImportError as exc:
            raise RuntimeError("kafka-python is required for topic management. Install requirements.txt.") from exc
        kafka_cfg = read_kafka_config()
        admin = KafkaAdminClient(
            bootstrap_servers=kafka_cfg.get("bootstrap_servers", "127.0.0.1:9092"),
            client_id=f"station-manager-{station}",
            request_timeout_ms=15000,
        )
        try:
            existing = set(admin.list_topics())
            missing = [name for name in topic_names if name not in existing]
            created: list[str] = []
            if missing:
                partitions = max(1, int(kafka_cfg.get("topic_partitions", 1)))
                replication_factor = max(1, int(kafka_cfg.get("replication_factor", 1)))
                try:
                    admin.create_topics(
                        new_topics=[NewTopic(name, num_partitions=partitions, replication_factor=replication_factor)
                                    for name in missing],
                        validate_only=False,
                    )
                    created.extend(missing)
                except Exception as exc:
                    # When another request created the same topics, they are safe to use.
                    available = set(admin.list_topics())
                    absent = [name for name in missing if name not in available]
                    if absent:
                        raise RuntimeError(f"Could not create Kafka topics {absent}: {exc}") from exc
            refreshed = set(admin.list_topics())
            return {
                "station": station,
                "created": created,
                "existing": [name for name in topic_names if name in existing],
                "topics": {name: (name in refreshed) for name in topic_names},
            }
        finally:
            admin.close()

    def training_running(self) -> bool:
        with self.lock:
            return any(key.endswith(":training") and process.poll() is None
                       for key, process in self.processes.items())

    def start_model_experiment(
        self,
        station: str,
        algorithm: str,
        parameters: dict[str, Any],
        training_options: dict[str, Any] | None = None,
        component: str = "demand_forecast",
        training_data_path: str | None = None,
        training_data_id: str | None = None,
    ) -> dict[str, Any]:
        """Launch a versioned candidate using a registered or validated uploaded dataset."""
        stations = read_station_config().get("stations", {})
        if station not in stations:
            raise ValueError(f"Unknown station '{station}'.")
        entry = stations[station]
        from backend.training_data import station_component_state, registered_source, digest_file
        if not station_component_state(entry).get(component, False):
            raise ValueError(f"Component {component} is not enabled for station {station}")
        training_file = training_data_path or registered_source(entry, component)
        if not training_file or not resolve_root_path(training_file).is_file():
            raise FileNotFoundError(
                f"No training CSV exists for '{station}'/{component}: {training_file}. "
                "Upload a complete historical dataset from Model Management."
            )
        source_file = resolve_root_path(training_file)
        source_digest = digest_file(source_file)

        if self.training_running():
            raise RuntimeError("A training job is already active. Finish it before starting another.")

        experiment_id = f"{station}-{component}-{algorithm}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        experiment_dir = ROOT / "models" / station / "experiments" / experiment_id
        experiment_dir.mkdir(parents=True, exist_ok=True)
        config_path = experiment_dir / "experiment.json"
        experiment_config = {
            "experiment_id": experiment_id,
            "station": station,
            "algorithm": algorithm,
            "component": component,
            "parameters": parameters,
            "training_options": training_options or {},
            "training_data_path": training_file,
            "training_data_id": training_data_id,
            "training_data_sha256": source_digest,
            "created_at": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        }
        temporary = config_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(experiment_config, indent=2), encoding="utf-8")
        temporary.replace(config_path)

        try:
            from src.models.model_registry import upsert_experiment
            upsert_experiment({
                "experiment_id": experiment_id,
                "station": station,
                "algorithm": algorithm,
                "component": component,
                "status": "queued",
                "parameters": parameters,
                "training_data_path": training_file,
                "training_data_id": training_data_id,
                "training_data_sha256": source_digest,
                "config_path": config_path.relative_to(ROOT).as_posix(),
                "created_at": experiment_config["created_at"],
            })
        except Exception:
            config_path.unlink(missing_ok=True)
            raise

        command = [sys.executable, "-u", "-m", "src.models.run_experiment", "--config", str(config_path)]
        try:
            result = self.start("training", command, station)
        except Exception as exc:
            upsert_experiment({"experiment_id": experiment_id, "status": "failed", "error": str(exc)})
            raise
        return {
            "station": station,
            "experiment_id": experiment_id,
            "algorithm": algorithm,
            "component": component,
            "training_data_id": training_data_id,
            "training_data_sha256": source_digest,
            "started": bool(result.get("started")),
            "pid": result.get("pid"),
            "log": result.get("log"),
            "status": "queued",
        }

    def start_training(self, station: str) -> dict[str, Any]:
        if self.training_running():
            raise RuntimeError("A training job is already active. Finish it before starting another.")
        stations = read_station_config().get("stations", {})
        if station not in stations:
            raise ValueError(f"Unknown station '{station}'.")
        entry = stations[station]
        training_file = entry.get("training_file") or entry.get("input_file")
        if not training_file or not resolve_root_path(training_file).is_file():
            raise FileNotFoundError(f"Training CSV not found for '{station}': {training_file}")
        checkpoint = entry.get("model_checkpoint")
        if not checkpoint:
            raise ValueError(f"Station '{station}' has no model_checkpoint configured.")
        python = sys.executable
        command = [python, "-u", "-m", "src.models.train", "--station", station,
                   "--csv", str(resolve_root_path(training_file)),
                   "--output", str(resolve_root_path(checkpoint))]
        return self.start("training", command, station)

    def start_pipeline(self, station: str | None = None) -> dict[str, Any]:
        with self.lifecycle_lock:
            return self._start_pipeline(station)

    def _start_pipeline(self, station: str | None = None) -> dict[str, Any]:
        station = station or read_station_config().get("active_station")
        commands, topics = self._prepare_pipeline(station)
        started: list[str] = []
        quality_command = self._quality_command()
        try:
            if quality_command and self.start("quality", quality_command).get("started"):
                started.append("quality")
            self._launch_services(station, commands, started)
            self._activate_pipelines({station: commands}, quality_command)
            return {"station": station, "started": started, "topics": topics, "status": self.station_status(station)}
        except Exception:
            self._stop_launched_services(station, started)
            raise

    @staticmethod
    def _quality_command() -> list[str] | None:
        if os.getenv("QUALITY_CONSUMER_EXTERNAL", "false").lower() == "true":
            return None
        return [sys.executable, "-u", "-m", "src.quality.dlq_consumer"]

    def _prepare_pipeline(self, station: str | None) -> tuple[list[tuple[str, list[str]]], dict[str, Any]]:
        config = read_station_config()
        stations = config.get("stations", {})
        if station not in stations:
            raise ValueError(f"Station '{station}' is not configured.")
        if not self.kafka_available():
            kafka_cfg = read_kafka_config()
            raise RuntimeError(
                f"Kafka bootstrap servers are not reachable: {kafka_cfg.get('bootstrap_servers')}. Start Kafka first."
            )

        entry = stations[station]
        from backend.onboarding_training import training_pending
        if training_pending(entry):
            raise RuntimeError("Initial model training is still queued or running for this station.")
        components = entry.get("model_components")
        if components is not None and not components.get("demand_forecast", {}).get("enabled", False):
            raise ValueError(
                "Streaming activation currently requires the demand_forecast component. "
                "Solar/wind-only stations can be registered and trained but require a separate stream driver."
            )
        # Isolate station outputs so one station can be stopped/reset while
        # other station predictors continue writing their own result files.
        registry_changed = False
        if not entry.get("actuals_file"):
            entry["actuals_file"] = f"data/processed/stations/{station}/actuals.csv"
            registry_changed = True
        if not entry.get("forecasts_file"):
            entry["forecasts_file"] = f"data/processed/stations/{station}/forecasts.csv"
            registry_changed = True
        if registry_changed:
            write_station_config(config)
        training_proc = self.processes.get(self.process_key("training", station))
        if training_proc and training_proc.poll() is None:
            raise RuntimeError(f"Station '{station}' is still training. Wait for training to finish before activation.")
        stream_file = resolve_root_path(entry["input_file"])
        history_value = entry.get("history_file")
        if history_value:
            history_file = resolve_root_path(history_value)
        else:
            history_file = stream_file.with_name(f"{station}_history.csv")
        active_model = entry.get("active_model") or {}
        checkpoint_value = active_model.get("artifact_path") or entry.get("model_checkpoint")
        if not checkpoint_value:
            raise ValueError(f"Station '{station}' has no active model or legacy model_checkpoint configured.")
        checkpoint = resolve_root_path(checkpoint_value)
        use_csv_producer = entry.get("stream_input", "csv_replay") != "kafka"
        required_files = [("history CSV", history_file), ("model artifact", checkpoint)]
        if use_csv_producer:
            required_files.append(("stream CSV", stream_file))
        for label, path in required_files:
            if not path.is_file():
                raise FileNotFoundError(f"Station '{station}' {label} not found: {path}")

        self.get_runtime_environment()
        topic_result = self.ensure_station_topics(station)
        python = sys.executable
        history_arg = ["--history-file", str(history_file)]
        producer_checkpoint_every = int(entry.get("producer_checkpoint_every", 1))
        if producer_checkpoint_every <= 0:
            raise ValueError("producer_checkpoint_every must be a positive integer.")
        commands = [
            ("spark", [python, "-u", "-m", "src.spark.streaming", "--station", station]),
            ("predictor", [python, "-u", "-m", "src.models.streaming_predictor", "--station", station,
                           "--model", str(checkpoint), *history_arg,
                           "--output", str(resolve_root_path(entry["forecasts_file"])),
                           "--actual-output", str(resolve_root_path(entry["actuals_file"]))]),
        ]
        if entry.get("live_actuals_enabled", bool(entry.get("hierarchy_path"))) and int(
            read_kafka_config().get("topic_partitions", 1)
        ) == 1:
            commands.append(("live_actuals", [python, "-u", "-m", "src.ingestion.live_actuals_consumer", "--station", station]))
        if use_csv_producer:
            commands.append((
                "producer", [python, "-u", "-m", "src.producers.csv_producer", "--station", station, *history_arg,
                             "--delay", str(float(entry.get("stream_delay_seconds", 1.0))),
                             "--checkpoint-every", str(producer_checkpoint_every)],
            ))
        return commands, topic_result

    def _launch_services(
        self, station: str, commands: list[tuple[str, list[str]]], started: list[str], *,
        ready_file: Path | None = None,
    ) -> None:
        for name, command in commands:
            if name == "spark" and "src.spark.streaming" in command and read_spark_config().get("shared_worker", True):
                if self._start_shared_query(station):
                    started.append(name)
                continue
            with self.lock:
                key = self.process_key(name, station)
                watched = self._supervised.get(key)
                if watched is not None and watched.process is not None:
                    code = watched.process.poll()
                    if code is None or (name == "producer" and code == 0):
                        continue
                # An explicit activation gives a failed service a fresh retry budget.
                previous = self._supervised.pop(key, None)
            launch_command = command
            if name == "producer" and ready_file is not None:
                launch_command = [*command, "--group-ready-file", str(ready_file)]
            try:
                result = self.start(name, launch_command, station)
            except Exception:
                if previous is not None:
                    with self.lock:
                        self._supervised[key] = previous
                raise
            if result.get("started"):
                started.append(name)

    def _activate_pipelines(self, launches: dict[str, list[tuple[str, list[str]]]],
                            quality_command: list[str] | None) -> None:
        if not launches:
            return
        registry = read_station_config()
        active_stations = set(registry.get("active_stations") or [])
        for station in launches:
            active_stations.add(station)
            registry["stations"][station].update(enabled=True, status="active", topics_ready=True)
        if len(launches) == 1 or registry.get("active_station") not in active_stations:
            registry["active_station"] = next(iter(launches))
        registry["active_stations"] = sorted(active_stations)
        write_station_config(registry)
        for station, commands in launches.items():
            for name, command in commands:
                # A later restart resumes directly from its durable cursor.
                if name != "spark" or station not in self._spark_requests:
                    self._supervise(name, command, station)
        if quality_command:
            self._supervise("quality", quality_command)

    def _stop_launched_services(self, station: str, started: list[str]) -> None:
        for name in reversed(started):
            self.stop(name, None if name == "quality" else station)

    def _wait_for_replay_group(self, ready_files: dict[str, Path], *, released: bool = False) -> dict[str, str]:
        pending = dict(ready_files)
        errors = {}
        deadline = time.monotonic() + (15 if released else 75)
        while pending:
            with self.lock:
                for station, ready_file in list(pending.items()):
                    process = self.processes.get(self.process_key("producer", station))
                    code = process.poll() if process is not None else -1
                    if code is not None:
                        pending.pop(station)
                        if code != 0:
                            errors[station] = f"CSV producer exited with code {code}. Check {self.log_path('producer', station).name}."
                    elif ready_file.is_file() != released:
                        pending.pop(station)
            if not pending or time.monotonic() >= deadline:
                break
            time.sleep(0.05)
        phase = "acknowledge the group start" if released else "finish CSV and Kafka startup"
        errors.update({station: f"CSV producer did not {phase} before the timeout. Check the producer log."
                       for station in pending})
        return errors

    def start_all_pipelines(self) -> dict[str, dict]:
        """Activate registered stations and release their new CSV producers together."""
        with self.lifecycle_lock, tempfile.TemporaryDirectory(prefix="electricity-start-") as location:
            results: dict[str, dict] = {}
            launches: dict[str, list[tuple[str, list[str]]]] = {}
            started_by_station: dict[str, list[str]] = {}
            ready_files: dict[str, Path] = {}
            gate = Path(location)
            quality_command = self._quality_command()
            quality_started = False
            completed = False
            try:
                stations = read_station_config().get("stations", {})
                for slot, station in enumerate(sorted(stations)):
                    started: list[str] = []
                    try:
                        commands, _ = self._prepare_pipeline(station)
                        if quality_command and not launches:
                            quality_started |= self.start("quality", quality_command).get("started", False)
                        ready_file = gate / f"{slot}.ready"
                        self._launch_services(station, commands, started, ready_file=ready_file)
                    except Exception as exc:
                        self._stop_launched_services(station, started)
                        results[station] = {"status": "failed", "message": str(exc)}
                        continue
                    launches[station] = commands
                    started_by_station[station] = started
                    if "producer" in started:
                        ready_files[station] = ready_file

                errors = self._wait_for_replay_group(ready_files)
                with self.lock:
                    for station in launches:
                        for service in ("spark", "predictor"):
                            worker = self.processes.get(self.process_key(service, station))
                            if worker is not None and (code := worker.poll()) is not None:
                                errors.setdefault(station, f"{service.title()} exited with code {code}. "
                                                  f"Check {self.log_path(service, station).name}.")
                for station in errors:
                    self._stop_launched_services(station, started_by_station.pop(station))
                    launches.pop(station)
                    ready_files.pop(station, None)
                if ready_files:
                    # One release timestamp is read by each waiting producer.
                    marker = gate / "release.tmp"
                    marker.write_text(str(time.time() + 0.3), encoding="ascii")
                    marker.replace(gate / "release")
                    release_errors = self._wait_for_replay_group(ready_files, released=True)
                    for station in release_errors:
                        self._stop_launched_services(station, started_by_station.pop(station))
                        launches.pop(station)
                    errors.update(release_errors)
                results.update({station: {"status": "failed", "message": error} for station, error in errors.items()})
                self._activate_pipelines(launches, quality_command)
                results.update({
                    station: {"status": "started" if started else "already_running", "services": started}
                    for station, started in started_by_station.items()
                })
                completed = True
                return {"stations": results}
            finally:
                if not completed:
                    for station, started in started_by_station.items():
                        self._stop_launched_services(station, started)
                if quality_started and (not completed or not launches):
                    self.stop("quality")

    def stop_pipeline(self, station: str | None = None) -> dict[str, Any]:
        with self.lifecycle_lock:
            return self._stop_pipeline(station)

    def _stop_pipeline(self, station: str | None = None) -> dict[str, Any]:
        with self.lock:
            if station is None:
                self._supervised.clear()
            else:
                for service in self.SERVICE_NAMES:
                    self._supervised.pop(self.process_key(service, station), None)
        results: dict[str, Any] = {}
        if station:
            for name in ("producer", "predictor", "spark", "training", "live_actuals"):
                results[name] = self.stop(name, station)
            registry = read_station_config()
            if station in registry.get("stations", {}):
                registry["stations"][station]["enabled"] = False
                registry["stations"][station]["status"] = "stopped"
                active_stations = set(registry.get("active_stations", []) or [])
                active_stations.discard(station)
                registry["active_stations"] = sorted(active_stations)
                if registry.get("active_station") == station:
                    registry["active_station"] = next(iter(sorted(active_stations)), None)
                write_station_config(registry)
                if not active_stations:
                    results["quality"] = self.stop("quality")
            return {"station": station, "results": results}

        # Global shutdown owns the shared worker as well as station processes.
        if self._spark_dir is not None:
            self._spark_requests.clear()
            self.stop("spark_shared")
            shutil.rmtree(self._spark_dir, ignore_errors=True)
            self._spark_dir = None
        with self.lock:
            keys = list(self.processes.keys())
        for key in keys:
            if ":" in key:
                station_id, name = key.split(":", 1)
                results[key] = self.stop(name, station_id)
            else:
                results[key] = self.stop(key)
        return results

    # --------------------------------------------------------
    # STATION-SCOPED FULL RESET + REPLAY
    # --------------------------------------------------------

    def _assert_station_stopped(self, station: str) -> None:
        """Refuse destructive reset while this station has streaming services alive."""
        active = []
        if station in self._spark_requests or station in self._shared_query_status():
            active.append("shared Spark query")
        with self.lock:
            for service in self.SERVICE_NAMES:
                proc = self.processes.get(self.process_key(service, station))
                supervisor = self._supervised.get(self.process_key(service, station))
                if (proc and proc.poll() is None) or (supervisor is not None
                    and not (service == "producer" and proc and proc.poll() == 0)):
                    active.append(service)
        if active:
            raise RuntimeError(
                f"Stop {station} first. These managed services are still running: "
                + ", ".join(active)
            )

        # Also detect processes started outside FastAPI (for example from an old BAT).
        if os.name == "nt":
            escaped = re.escape(station)
            script = (
                "$mods='src\\.producers\\.csv_producer|src\\.spark\\.streaming|src\\.models\\.streaming_predictor|src\\.ingestion\\.live_actuals_consumer'; "
                f"$needle='--station\\s+{escaped}(\\s|$)'; "
                "$p=Get-CimInstance Win32_Process | Where-Object { "
                "$_.CommandLine -match $mods -and $_.CommandLine -match $needle }; "
                "if($p){$p | Select-Object ProcessId,CommandLine | Format-List | Out-String | Write-Output; exit 3}; exit 0"
            )
            result = subprocess.run(
                ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode == 3:
                raise RuntimeError(
                    f"Streaming process(es) for {station} are still running outside the dashboard. "
                    "Stop that station's producer, predictor, Spark and training processes, then retry.\n"
                    + result.stdout.strip()
                )
            if result.returncode != 0:
                raise RuntimeError(
                    "Could not verify station processes with PowerShell: "
                    + (result.stderr or result.stdout).strip()
                )

    def _assert_other_predictors_stopped(self, selected_station: str) -> None:
        """Avoid rewriting shared CSV result files while another predictor appends to them."""
        active: list[str] = []
        with self.lock:
            for key, proc in self.processes.items():
                if not key.endswith(":predictor") or proc.poll() is not None:
                    continue
                station_id = key.rsplit(":", 1)[0]
                if station_id != selected_station:
                    active.append(station_id)
        if active:
            raise RuntimeError(
                "Stop the other station predictor(s) before Reset & Replay because all stations "
                "currently append to shared actuals.csv and forecasts.csv: "
                + ", ".join(sorted(set(active)))
            )

        # Detect predictor processes that were started outside FastAPI.
        if os.name == "nt":
            escaped = re.escape(selected_station)
            script = (
                "$p=Get-CimInstance Win32_Process | Where-Object { "
                "$_.CommandLine -match 'src\\.models\\.streaming_predictor' }; "
                f"$p=$p | Where-Object {{ $_.CommandLine -notmatch '--station\\s+{escaped}(\\s|$)' }}; "
                "if($p){$p | Select-Object ProcessId,CommandLine | Format-List | Out-String | Write-Output; exit 3}; exit 0"
            )
            result = subprocess.run(
                ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode == 3:
                raise RuntimeError(
                    "Stop predictor process(es) for other stations before Reset & Replay; "
                    "they may be writing to the shared result CSV files.\n"
                    + result.stdout.strip()
                )
            if result.returncode != 0:
                raise RuntimeError(
                    "Could not verify other station predictor processes with PowerShell: "
                    + (result.stderr or result.stdout).strip()
                )

    _KAFKA_TOOL_NAMES = {
        "kafka-topics": ("kafka-topics.bat", "kafka-topics.sh", "kafka-topics"),
        "kafka-delete-records": ("kafka-delete-records.bat", "kafka-delete-records.sh", "kafka-delete-records"),
    }

    @classmethod
    def _kafka_tool_order(cls, tool: str) -> tuple[str, ...]:
        names = cls._KAFKA_TOOL_NAMES[tool]
        if os.name == "nt":
            return names
        return tuple(name for name in names if not name.endswith(".bat"))

    @classmethod
    def _directory_has_kafka_tools(cls, directory: Path) -> bool:
        return all(
            any((directory / name).is_file() for name in cls._kafka_tool_order(tool))
            for tool in cls._KAFKA_TOOL_NAMES
        )

    @staticmethod
    def _kafka_directory_options(base: Path) -> list[Path]:
        options = [base]
        if base.name.lower() != "windows":
            options.extend((base / "bin" / "windows", base / "windows", base / "bin"))
        return options

    def _kafka_cli_dir(self) -> Path:
        kafka_cfg = read_kafka_config()
        configured = kafka_cfg.get("kafka_cli_dir") or os.environ.get("KAFKA_HOME")
        candidates: list[Path] = []
        if configured:
            base = Path(str(configured)).expanduser()
            if not base.is_absolute():
                base = ROOT / base
            candidates.extend(self._kafka_directory_options(base))
        elif os.name == "nt":
            candidates.append(Path(r"C:\kafka_2.13-4.3.1\bin\windows"))
        else:
            for raw in (
                "/opt/homebrew/opt/kafka/bin",
                "/usr/local/opt/kafka/bin",
                "/opt/kafka/bin",
            ):
                candidates.append(Path(raw))
            for tool in ("kafka-topics", "kafka-topics.sh"):
                found = shutil.which(tool)
                if found:
                    candidates.append(Path(found).resolve().parent)
        searched: list[Path] = []
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
            except OSError:
                continue
            if resolved in searched:
                continue
            searched.append(resolved)
            if self._directory_has_kafka_tools(resolved):
                return resolved
        expected = ", ".join(str(path) for path in searched) or "(no candidate directories)"
        required = ", ".join(
            " or ".join(self._kafka_tool_order(tool)) for tool in self._KAFKA_TOOL_NAMES
        )
        raise FileNotFoundError(
            "Kafka command-line tools were not found. Configure kafka_cli_dir "
            "in config/kafka.yaml or set KAFKA_HOME. Expected folder: "
            f"{expected}; missing: {required}"
        )

    def _kafka_script(self, tool: str) -> Path:
        directory = self._kafka_cli_dir()
        for name in self._kafka_tool_order(tool):
            path = directory / name
            if path.is_file():
                return path
        raise FileNotFoundError(f"Kafka tool {tool} was not found in {directory}.")

    @staticmethod
    def _run_bat(path: Path, args: list[str], timeout: int = 60) -> str:
        if os.name == "nt":
            command: str | list[str] = subprocess.list2cmdline([str(path), *args])
            shell = True
        else:
            # Run the script directly so topic names are arguments, not shell text.
            command = [str(path), *args]
            shell = False
        result = subprocess.run(
            command,
            cwd=ROOT,
            shell=shell,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()
        if result.returncode != 0:
            raise RuntimeError(
                f"Kafka command failed ({path.name}, exit {result.returncode}).\n{output}"
            )
        return output

    def _ensure_reset_topics(self, station: str, raw_topic: str, featured_topic: str) -> dict[str, Any]:
        """Create missing topics and refuse partition layouts the reset cannot clear safely."""
        kafka_cfg = read_kafka_config()
        bootstrap = str(kafka_cfg.get("bootstrap_servers", "127.0.0.1:9092"))
        topics_cli = self._kafka_script("kafka-topics")
        listing = self._run_bat(topics_cli, ["--bootstrap-server", bootstrap, "--list"])
        existing = {line.strip() for line in listing.splitlines() if line.strip()}
        created: list[str] = []
        for topic in (raw_topic, featured_topic):
            if topic not in existing:
                self._run_bat(
                    topics_cli,
                    ["--bootstrap-server", bootstrap, "--create", "--topic", topic,
                     "--partitions", "1", "--replication-factor", "1"],
                )
                created.append(topic)
            description = self._run_bat(
                topics_cli,
                ["--bootstrap-server", bootstrap, "--describe", "--topic", topic],
            )
            match = re.search(r"PartitionCount:\s*(\d+)", description)
            if not match:
                raise RuntimeError(
                    f"Could not verify partition count for topic '{topic}'. Refusing reset.\n{description}"
                )
            partitions = int(match.group(1))
            if partitions != 1:
                raise RuntimeError(
                    f"Topic '{topic}' has {partitions} partitions. This reset operation is configured "
                    "to clear partition 0 only, so no records were deleted. Use one partition per station topic."
                )
        return {"station": station, "created": created, "topics": [raw_topic, featured_topic]}

    @staticmethod
    def _validate_delete_records_output(
        output: str, raw_topic: str, featured_topic: str
    ) -> dict[str, int]:
        """Validate Kafka 4.x text output or structured JSON delete results.

        Kafka 4.3.1 commonly prints:
          Records delete operation completed
          partition: 50hertz-raw-0    low_watermark: 3457
          partition: 50hertz-featured-0    low_watermark: 3394

        Kafka may also print a Log4j reconfiguration warning on stderr after
        the successful result. That warning is not treated as a delete failure
        when the CLI exits successfully and both requested partition results
        are present and valid.
        """
        expected = {f"{raw_topic}-0", f"{featured_topic}-0"}
        results: dict[str, int] = {}

        # First try to parse a structured JSON result, if this Kafka build emits one.
        decoder = json.JSONDecoder()
        decoded = None
        for position, character in enumerate(output):
            if character != "{":
                continue
            try:
                candidate, _ = decoder.raw_decode(output[position:])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict) and isinstance(candidate.get("partitions"), list):
                decoded = candidate
                break

        if decoded is not None:
            for item in decoded["partitions"]:
                if not isinstance(item, dict):
                    continue
                topic = item.get("topic")
                try:
                    partition_id = int(item.get("partition", -1))
                except (TypeError, ValueError):
                    continue
                if topic is None or partition_id != 0:
                    continue
                partition_name = f"{topic}-{partition_id}"
                if partition_name not in expected:
                    continue
                error = item.get("error")
                if error not in (None, "NONE", 0, "0"):
                    raise RuntimeError(
                        f"Kafka could not clear {partition_name}: {error}.\n{output[-3000:]}"
                    )
                watermark = item.get("low_watermark", item.get("lowWatermark"))
                if watermark is not None:
                    try:
                        results[partition_name] = int(watermark)
                    except (TypeError, ValueError):
                        pass

        # Kafka 4.3.1 text format: parse the per-partition result lines.
        if not results:
            for partition, watermark in re.findall(
                r"partition:\s*([^\s]+)\s+low_watermark:\s*(-?\d+)",
                output,
                flags=re.IGNORECASE,
            ):
                results[partition.strip()] = int(watermark)

        missing = sorted(expected - set(results))
        if missing:
            raise RuntimeError(
                "Kafka delete-records output did not contain a valid low_watermark "
                f"for every requested partition. Missing: {', '.join(missing)}. "
                "No output CSV or Spark checkpoint cleanup was performed.\n"
                + (output[-3000:] if output else "(Kafka returned no output)")
            )

        invalid = {name: value for name, value in results.items() if name in expected and value < 0}
        if invalid:
            raise RuntimeError(
                f"Kafka returned invalid low_watermark values: {invalid}.\n{output[-3000:]}"
            )

        # For text output require the operation-completed marker. Structured JSON
        # must contain both partition results and passed the per-partition error checks.
        if decoded is None and not re.search(
            r"records delete operation completed", output, flags=re.IGNORECASE
        ):
            raise RuntimeError(
                "Kafka returned partition watermarks but no successful completion marker. "
                "No output CSV or Spark checkpoint cleanup was performed.\n"
                + output[-3000:]
            )

        return {name: results[name] for name in sorted(expected)}

    def _clear_topic_records(self, raw_topic: str, featured_topic: str) -> str:
        """Advance both selected topic log-start offsets to their current high watermarks."""
        bootstrap = str(read_kafka_config().get("bootstrap_servers", "127.0.0.1:9092"))
        payload = {
            "partitions": [
                {"topic": raw_topic, "partition": 0, "offset": -1},
                {"topic": featured_topic, "partition": 0, "offset": -1},
            ],
            "version": 1,
        }
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", suffix=".json", prefix="station_reset_",
                dir=os.environ.get("TEMP") or None, delete=False,
            ) as handle:
                json.dump(payload, handle, indent=2)
                temp_path = Path(handle.name)
            output = self._run_bat(
                self._kafka_script("kafka-delete-records"),
                ["--bootstrap-server", bootstrap, "--offset-json-file", str(temp_path)],
                timeout=120,
            )
            validated = self._validate_delete_records_output(
                output, raw_topic, featured_topic
            )
            validation_note = "\nValidated Kafka partition low_watermarks: " + ", ".join(
                f"{partition}={watermark}" for partition, watermark in validated.items()
            )
            return output + validation_note
        finally:
            if temp_path:
                temp_path.unlink(missing_ok=True)

    def reset_station_pipeline(self, station: str) -> dict[str, Any]:
        with self.lifecycle_lock:
            return self._reset_station_pipeline(station)

    def _reset_station_pipeline(self, station: str) -> dict[str, Any]:
        """Reset one station's Kafka/Spark/output state without starting services.

        The station's producer, predictor, Spark and training processes must be
        stopped before this method is called. After reset, the user can choose
        Start selected station separately from the dashboard.
        """
        registry = read_station_config()
        stations = registry.get("stations", {})
        if station not in stations:
            raise ValueError(f"Unknown station '{station}'.")
        if not self.kafka_available():
            raise RuntimeError(
                f"Kafka bootstrap servers are not reachable: {read_kafka_config().get('bootstrap_servers')}. Start Kafka first."
            )

        entry = stations[station]
        raw_topic = str(entry.get("raw_topic") or "").strip()
        featured_topic = str(entry.get("featured_topic") or "").strip()
        if not raw_topic or not featured_topic:
            raise ValueError(
                f"Station '{station}' must define raw_topic and featured_topic in config/stations.yaml."
            )
        if raw_topic == featured_topic:
            raise ValueError("A station's raw_topic and featured_topic must be different.")

        self._assert_station_stopped(station)
        # New station registrations use isolated result files. This permits
        # independent stop/reset while other stations continue forecasting.
        # Legacy stations without isolated paths retain the original safety gate.
        isolated_outputs = bool(stations[station].get("actuals_file") and stations[station].get("forecasts_file"))
        if not isolated_outputs:
            self._assert_other_predictors_stopped(station)

        # Do not truncate a topic that is also used by another registered station.
        for other_id, other in stations.items():
            if other_id == station:
                continue
            other_topics = {other.get("raw_topic"), other.get("featured_topic")}
            if raw_topic in other_topics or featured_topic in other_topics:
                raise RuntimeError(
                    f"Reset blocked: {station!r} shares a Kafka topic with {other_id!r}. "
                    "Give each station unique raw and featured topics before resetting."
                )

        # Preflight storage before deleting Kafka records. Never silently leave
        # Cassandra station partitions behind while claiming reset succeeded.
        from src.storage.forecast_store import storage_backend
        using_cassandra = storage_backend() == "cassandra"
        if using_cassandra:
            from src.storage.cassandra_store import clear_station_operational_data
            clear_station_operational_data(station, dry_run=True)
        elif not isolated_outputs:
            validate_station_output_files(station)
        spark_cfg = read_spark_config()
        checkpoint_base = str(
            spark_cfg.get("checkpoint_location", "checkpoints/spark_features")
        )
        checkpoint_path = resolve_root_path(f"{checkpoint_base}_{station}")
        if (
            not checkpoint_path.is_relative_to(ROOT.resolve())
            or checkpoint_path == ROOT.resolve()
        ):
            raise RuntimeError(
                f"Refusing to remove a Spark checkpoint outside the POC folder: {checkpoint_path}"
            )
        if checkpoint_path.exists() and not checkpoint_path.is_dir():
            raise RuntimeError(
                f"Spark checkpoint path is not a directory: {checkpoint_path}"
            )
        # The DLQ output query has its own checkpoint. It must be reset with
        # the featured query after Kafka topic records are truncated.
        dlq_checkpoint_path = resolve_root_path(f"{checkpoint_base}_{station}_dlq")
        for label, candidate in (("featured", checkpoint_path), ("DLQ", dlq_checkpoint_path)):
            if (
                not candidate.is_relative_to(ROOT.resolve())
                or candidate == ROOT.resolve()
            ):
                raise RuntimeError(
                    f"Refusing to remove the {label} Spark checkpoint outside the POC folder: {candidate}"
                )
            if candidate.exists() and not candidate.is_dir():
                raise RuntimeError(
                    f"Spark {label} checkpoint path is not a directory: {candidate}"
                )

        from src.models.predictor_recovery import predictor_state_path
        predictor_state = predictor_state_path(ROOT, station)
        recovery_files = [predictor_state, predictor_state.with_name(predictor_state.name + "-wal"),
                          predictor_state.with_name(predictor_state.name + "-shm")]
        from src.storage.live_actuals import live_actuals_path
        live_state = live_actuals_path(ROOT, station)
        live_files = [live_state, live_state.with_name(live_state.name + "-wal"),
                      live_state.with_name(live_state.name + "-shm")]
        for path in recovery_files + live_files:
            if not path.resolve().is_relative_to(ROOT.resolve()) or path.is_symlink():
                raise RuntimeError(f"Refusing to reset unsafe recovery path: {path}")
            if path.exists() and not path.is_file():
                raise RuntimeError(f"Recovery path is not a file: {path}")

        topic_result = self._ensure_reset_topics(
            station, raw_topic, featured_topic
        )

        # Kafka topics are retained; only their existing records are expired.
        delete_output = self._clear_topic_records(raw_topic, featured_topic)

        # Back up and clear this station's isolated result files. Legacy
        # stations continue using the original shared-CSV filtering behavior.
        output_backup = None
        if using_cassandra:
            # CSV files remain untouched when switching between backends.
            # Only this station's registered Cassandra partitions are cleared.
            clear_station_operational_data(station, dry_run=False)
        elif isolated_outputs:
            import datetime as _dt
            backup_dir = ROOT / "data" / "processed" / "reset_backups" / f"{station}_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            output_files = [stations[station].get(field) for field in (
                "actuals_file", "forecasts_file", "renewable_forecasts_file"
            )]
            output_files.append(f"data/processed/stations/{station}/renewable_actuals.csv")
            seen_outputs: set[Path] = set()
            for value in output_files:
                if not value:
                    continue
                output_path = resolve_root_path(value)
                if output_path in seen_outputs:
                    continue
                seen_outputs.add(output_path)
                if not output_path.is_relative_to(ROOT.resolve()):
                    raise RuntimeError(f"Refusing to reset output outside POC folder: {output_path}")
                if output_path.exists():
                    backup_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(output_path, backup_dir / output_path.name)
                    output_path.unlink()
            output_backup = backup_dir if backup_dir.exists() else None
        else:
            output_backup = prepare_station_outputs(station)

        # Clear only this station's Spark checkpoint. The trained model and
        # source CSVs are intentionally preserved.
        checkpoint_removed = checkpoint_path.exists()
        if checkpoint_removed:
            shutil.rmtree(checkpoint_path)
        dlq_checkpoint_removed = dlq_checkpoint_path.exists()
        if dlq_checkpoint_removed:
            shutil.rmtree(dlq_checkpoint_path)

        # Kafka records and result files have been reset, so the producer's
        # persisted source cursor must also be cleared. Otherwise the next
        # normal start would correctly resume *past* the rows that this reset
        # deliberately removed from Kafka.
        producer_progress_path = resolve_root_path(
            f"data/state/{station}/producer_progress.json"
        )
        if (
            not producer_progress_path.is_relative_to(ROOT.resolve())
            or producer_progress_path == ROOT.resolve()
            or producer_progress_path.is_symlink()
        ):
            raise RuntimeError(
                f"Refusing to remove unsafe producer progress path: {producer_progress_path}"
            )
        if producer_progress_path.exists() and not producer_progress_path.is_file():
            raise RuntimeError(
                f"Producer progress path is not a file: {producer_progress_path}"
            )
        producer_progress_removed = producer_progress_path.exists()
        if producer_progress_removed:
            producer_progress_path.unlink()

        predictor_recovery_removed = any(path.exists() for path in recovery_files)
        for path in recovery_files + live_files:
            path.unlink(missing_ok=True)

        # Record the reset state but keep every service stopped. Starting the
        # station is a separate, explicit dashboard action.
        updated_registry = read_station_config()
        if station in updated_registry.get("stations", {}):
            updated_registry["stations"][station]["enabled"] = False
            updated_registry["stations"][station]["status"] = "reset"
            write_station_config(updated_registry)

        return {
            "station": station,
            "mode": "reset_only",
            "topics": topic_result,
            "topic_records_cleared": True,
            "kafka_delete_records_output": delete_output[-2000:],
            "spark_checkpoint": str(checkpoint_path),
            "spark_checkpoint_removed": checkpoint_removed,
            "spark_dlq_checkpoint": str(dlq_checkpoint_path),
            "spark_dlq_checkpoint_removed": dlq_checkpoint_removed,
            "producer_progress_file": str(producer_progress_path),
            "predictor_recovery_removed": predictor_recovery_removed,
            "producer_progress_removed": producer_progress_removed,
            "processed_output_backup": str(output_backup) if output_backup else None,
            "pipeline_started": False,
            "services_remain_stopped": True,
            "message": (
                f"{station} reset completed. Kafka records, station output rows, "
                "Spark checkpoints and the producer replay cursor were reset. "
                "No services were started. Use Start "
                "selected station when you are ready."
            ),
        }

    def _station_delete_paths(
        self,
        station: str,
        entry: dict[str, Any],
        stations: dict[str, Any],
    ) -> tuple[list[Path], list[Path]]:
        # Preflight configured files and station-owned directories.
        root = ROOT.resolve()
        file_fields = (
            "input_file", "history_file", "training_file", "model_checkpoint", "model_metadata",
            "actuals_file", "forecasts_file",
        )
        own_files: dict[Path, str] = {}
        other_files: dict[Path, str] = {}

        for field in file_fields:
            value = entry.get(field)
            if value:
                raw_path = Path(value).expanduser()
                if not raw_path.is_absolute():
                    raw_path = ROOT / raw_path
                if raw_path.is_symlink():
                    raise RuntimeError(f"Refusing to delete a symbolic-link station file: {raw_path}")
                path = raw_path.resolve()
                if path == root or not path.is_relative_to(root):
                    raise RuntimeError(f"Refusing to delete {field} outside the POC folder: {path}")
                own_files[path] = field

        for other_id, other_entry in stations.items():
            if other_id == station or not isinstance(other_entry, dict):
                continue
            for field in file_fields:
                value = other_entry.get(field)
                if value:
                    other_files[resolve_root_path(value)] = f"{other_id}.{field}"

        shared = sorted(set(own_files).intersection(other_files))
        if shared:
            details = ", ".join(
                f"{path} ({own_files[path]} also used by {other_files[path]})" for path in shared
            )
            raise RuntimeError("Station deletion blocked because files are shared: " + details)

        for path in own_files:
            if path.exists() and not path.is_file():
                raise RuntimeError(f"Refusing to delete configured path because it is not a file: {path}")

        spark_cfg = read_spark_config()
        checkpoint_base = str(spark_cfg.get("checkpoint_location", "checkpoints/spark_features"))
        checkpoint_path = resolve_root_path(f"{checkpoint_base}_{station}")
        fixed_station_dirs = {
            (ROOT / "data" / "raw" / "stations" / station).resolve(),
            (ROOT / "data" / "training_uploads" / station).resolve(),
            (ROOT / "data" / "state" / station).resolve(),
            (ROOT / "data" / "processed" / "stations" / station).resolve(),
            (ROOT / "models" / station).resolve(),
        }
        candidate_dirs = [*fixed_station_dirs, checkpoint_path]
        # Also remove the station-specific top-level model directory that contains
        # the configured checkpoint (for example models/<station>/v1 or a legacy
        # models/<station>_seq2seq directory), provided no other station references it.
        model_value = entry.get("model_checkpoint")
        if model_value:
            configured_model = resolve_root_path(model_value)
            models_root = (ROOT / "models").resolve()
            try:
                relative_model = configured_model.relative_to(models_root)
            except ValueError:
                relative_model = None
            if relative_model is not None and len(relative_model.parts) >= 2:
                candidate_dirs.append(models_root / relative_model.parts[0])
        own_dirs: list[Path] = []
        for raw_directory in candidate_dirs:
            if raw_directory.is_symlink():
                raise RuntimeError(f"Refusing to remove a symbolic-link directory: {raw_directory}")
            directory = raw_directory.resolve()
            if directory == root or not directory.is_relative_to(root):
                raise RuntimeError(f"Refusing to remove a directory outside the POC folder: {directory}")
            if directory.exists() and not directory.is_dir():
                raise RuntimeError(f"Expected a station directory but found a file: {directory}")
            if directory.exists():
                station_model_root = False
                if model_value:
                    configured_model = resolve_root_path(model_value)
                    models_root = (ROOT / "models").resolve()
                    try:
                        relative_model = configured_model.relative_to(models_root)
                        station_model_root = (
                            len(relative_model.parts) >= 2
                            and directory == (models_root / relative_model.parts[0]).resolve()
                        )
                    except ValueError:
                        station_model_root = False
                if directory not in fixed_station_dirs and directory != checkpoint_path and not station_model_root:
                    raise RuntimeError(f"Refusing to remove a directory that is not clearly station-specific: {directory}")
                own_dirs.append(directory)

        for directory in own_dirs:
            for other_path, owner in other_files.items():
                if other_path == directory or other_path.is_relative_to(directory):
                    raise RuntimeError(
                        f"Deletion blocked: {directory} contains a configured file belonging to {owner}: {other_path}"
                    )

        log_root = (ROOT / "logs").resolve()
        for service in self.SERVICE_NAMES:
            log_path = self.log_path(service, station)
            if log_path.exists():
                resolved_log = log_path.resolve()
                if log_path.is_symlink() or not resolved_log.is_relative_to(log_root) or not resolved_log.is_file():
                    raise RuntimeError(f"Refusing to delete unsafe station log path: {log_path}")
                own_files[resolved_log] = "station_process_log"

        return sorted((path for path in own_files if path.exists()), key=str), own_dirs

    def _assert_station_training_stopped(self, station: str) -> None:
        # _assert_station_stopped checks the managed process table. This also catches manual CMD training sessions.
        if os.name != "nt":
            return
        escaped = re.escape(station)
        script = (
            "$p=Get-CimInstance Win32_Process | Where-Object { "
            "$_.CommandLine -match 'src\\.models\\.train' -and "
            f"$_.CommandLine -match '--station\\s+{escaped}(\\s|$)' "
            "}; if($p){$p | Select-Object ProcessId,CommandLine | Format-List | Out-String | Write-Output; exit 3}; exit 0"
        )
        result = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
            cwd=ROOT, capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 3:
            raise RuntimeError(
                f"Training process for {station} is still running outside the dashboard. Stop it before deleting the station.\n"
                + result.stdout.strip()
            )
        if result.returncode != 0:
            raise RuntimeError("Could not verify external training processes: " + (result.stderr or result.stdout).strip())

    def delete_station(self, station: str, confirm_station_id: str) -> dict[str, Any]:
        with self.lifecycle_lock:
            return self._delete_station(station, confirm_station_id)

    def _delete_station(self, station: str, confirm_station_id: str) -> dict[str, Any]:
        # Permanently delete one station's registry, data, model, outputs and retained Kafka records.
        if confirm_station_id.strip().lower() != station.strip().lower():
            raise ValueError("Typed confirmation does not match the selected station ID.")
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{1,31}", station):
            raise ValueError("Station ID is not a valid registered station key.")

        registry = read_station_config()
        stations = registry.get("stations", {})
        if station not in stations:
            raise ValueError(f"Unknown station '{station}'.")
        if not self.kafka_available():
            raise RuntimeError("Kafka broker is not reachable. Start Kafka before deleting a station so its topic records can be cleared safely.")

        self._assert_station_stopped(station)
        self._assert_station_training_stopped(station)
        # Result CSV files are shared; another predictor must not append while rows are filtered.
        self._assert_other_predictors_stopped(station)

        entry = stations[station]
        raw_topic = entry.get("raw_topic")
        featured_topic = entry.get("featured_topic")
        if not raw_topic or not featured_topic:
            raise RuntimeError(f"Station '{station}' must define raw_topic and featured_topic before deletion.")

        # Protect every other registered station from a shared or cross-assigned topic.
        for other_id, other in stations.items():
            if other_id == station or not isinstance(other, dict):
                continue
            other_topics = {other.get("raw_topic"), other.get("featured_topic")}
            if raw_topic in other_topics or featured_topic in other_topics:
                raise RuntimeError(
                    f"Deletion blocked: {station!r} shares Kafka topic(s) with {other_id!r}. Assign unique topics before deleting."
                )

        # Preflight the active backend before mutating Kafka or station data.
        from src.storage.forecast_store import storage_backend
        using_cassandra = storage_backend() == "cassandra"
        if using_cassandra:
            from src.storage.cassandra_store import clear_station_operational_data
            clear_station_operational_data(station, dry_run=True)
        else:
            validate_station_output_files(station)
        files_to_delete, dirs_to_delete = self._station_delete_paths(station, entry, stations)

        # Clear only this station's Kafka records. Keep empty topic definitions to avoid
        # Kafka KRaft topic-directory rename failures seen on this Windows installation.
        topic_result = self._ensure_reset_topics(station, raw_topic, featured_topic)
        delete_output = self._clear_topic_records(raw_topic, featured_topic)

        # Remove only station-scoped operational Cassandra partitions. Local
        # station file removal below is still performed for a full deletion.
        if using_cassandra:
            clear_station_operational_data(station, dry_run=False)
        removed_output_rows = delete_station_output_rows(station)

        # Close any log handles retained by exited managed processes before unlinking logs.
        with self.lock:
            for service in self.SERVICE_NAMES:
                key = self.process_key(service, station)
                proc = self.processes.get(key)
                if proc and proc.poll() is None:
                    raise RuntimeError(f"Process {key} restarted during deletion; stop it and retry.")
                self.processes.pop(key, None)
                handle = self.log_handles.pop(key, None)
                if handle:
                    handle.close()

        deleted_files: list[str] = []
        for path in files_to_delete:
            if path.exists():
                path.unlink()
                deleted_files.append(str(path.relative_to(ROOT)))

        deleted_directories: list[str] = []
        for directory in sorted(dirs_to_delete, key=lambda item: len(item.parts), reverse=True):
            if directory.exists():
                shutil.rmtree(directory)
                deleted_directories.append(str(directory.relative_to(ROOT)))

        # Remove only empty parent directories of configured station files.
        protected_roots = {
            ROOT.resolve(), (ROOT / "data").resolve(), (ROOT / "data" / "raw").resolve(),
            (ROOT / "models").resolve(), (ROOT / "logs").resolve(),
        }
        for deleted in files_to_delete:
            parent = deleted.parent
            while parent.resolve() not in protected_roots and parent.resolve().is_relative_to(ROOT.resolve()):
                try:
                    parent.rmdir()
                except OSError:
                    break
                parent = parent.parent

        # Remove registry entry last. The registry remains if any previous step failed.
        updated = read_station_config()
        updated_stations = updated.get("stations", {})
        updated_stations.pop(station, None)
        if updated.get("active_station") == station:
            updated["active_station"] = next(iter(updated_stations), None)
        updated["stations"] = updated_stations
        write_station_config(updated)

        return {
            "station": station,
            "deleted": True,
            "removed_output_rows": removed_output_rows,
            "deleted_files": deleted_files,
            "deleted_directories": deleted_directories,
            "kafka_topics": {
                "records_cleared": True,
                "preserved_empty": [raw_topic, featured_topic],
                "low_watermarks": delete_output[-1200:],
                "created_during_delete": topic_result.get("created", []),
            },
            "remaining_stations": list(updated_stations),
            "active_station": updated.get("active_station"),
            "message": (
                f"Station '{station}' and its local data/model/results were deleted. "
                "Kafka topic definitions were retained empty for Windows KRaft safety."
            ),
        }

    def read_log(self, name: str, station: str | None = None, lines: int = 100) -> str:
        if name not in self.SERVICE_NAMES:
            raise ValueError("Unknown process log. Use spark, predictor, producer, training or live_actuals.")
        if name == "spark" and station and read_spark_config().get("shared_worker", True):
            shared_log = self.log_path("spark_shared")
            if shared_log.exists():
                return _tail_log(shared_log, lines)
        return _tail_log(self.log_path(name, station), lines)


manager = PipelineManager()
