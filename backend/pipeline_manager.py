"""Station-aware local process manager for the electricity forecasting POC.

Run FastAPI with one worker and without --reload. Kafka broker is managed
separately; station topic creation is performed through Kafka AdminClient.
"""
from __future__ import annotations

import json
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
from pathlib import Path
from threading import Lock
from typing import Any

import yaml

from scripts.station_replay_helper import (
    prepare_station_outputs,
    validate_station_output_files,
    delete_station_output_rows,
)

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
SPARK_CONFIG = ROOT / "config" / "spark.yaml"
STATION_CONFIG = ROOT / "config" / "stations.yaml"
KAFKA_CONFIG = ROOT / "config" / "kafka.yaml"


def read_station_config() -> dict[str, Any]:
    with STATION_CONFIG.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


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
        return {"bootstrap_servers": "127.0.0.1:9092"}
    with KAFKA_CONFIG.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_root_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


class PipelineManager:
    """Own independent local processes for every station and service."""

    SERVICE_NAMES = ("spark", "predictor", "producer", "training")

    def __init__(self) -> None:
        self.processes: dict[str, subprocess.Popen] = {}
        self.log_handles: dict[str, Any] = {}
        self.lock = Lock()

    def get_runtime_environment(self) -> dict[str, Any]:
        env = os.environ.copy()
        python_exe = Path(sys.executable).resolve()
        if not python_exe.exists():
            raise RuntimeError(f"Python executable not found: {python_exe}")
        env["PYSPARK_PYTHON"] = str(python_exe)
        env["PYSPARK_DRIVER_PYTHON"] = str(python_exe)
        configured_java = read_spark_config().get("java_home")
        java_candidates = [configured_java, env.get("JAVA_HOME")]
        java_name = "java.exe" if os.name == "nt" else "java"
        java_home_path = next(
            (
                Path(value).expanduser().resolve()
                for value in java_candidates
                if value
                and (Path(value).expanduser() / "bin" / java_name).is_file()
            ),
            None,
        )
        if java_home_path is None:
            java_on_path = shutil.which(java_name, path=env.get("PATH"))
            if not java_on_path:
                raise RuntimeError("Java was not found. Set JAVA_HOME or configure java_home in config/spark.yaml.")
            java_home_path = Path(java_on_path).resolve().parent.parent
        java_exe = java_home_path / "bin" / java_name
        env["JAVA_HOME"] = str(java_home_path)
        env["PATH"] = str(java_home_path / "bin") + os.pathsep + env.get("PATH", "")
        result = subprocess.run([str(java_exe), "-version"], capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            raise RuntimeError("Java version check failed: " + result.stderr)
        return {
            "env": env,
            "python_executable": str(python_exe),
            "java_home": str(java_home_path),
            "java_executable": str(java_exe),
            "java_version": result.stderr + result.stdout,
        }

    def kafka_available(self, host: str = "127.0.0.1", port: int = 9092) -> bool:
        try:
            with socket.create_connection((host, port), timeout=2):
                return True
        except OSError:
            return False

    @staticmethod
    def process_key(name: str, station: str | None = None) -> str:
        return f"{station}:{name}" if station else name

    @staticmethod
    def log_path(name: str, station: str | None = None) -> Path:
        return LOG_DIR / (f"{station}_{name}.log" if station else f"{name}.log")

    def status(self) -> dict[str, Any]:
        with self.lock:
            processes = {}
            for key, proc in self.processes.items():
                processes[key] = {
                    "running": proc.poll() is None,
                    "pid": proc.pid,
                    "return_code": proc.poll(),
                }
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
        return {
            "kafka_tcp_available": self.kafka_available(),
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
            "services": selected,
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

            runtime = self.get_runtime_environment()
            log_path = self.log_path(name, station)
            log = log_path.open("a", encoding="utf-8")
            kwargs: dict[str, Any] = {}
            if os.name == "nt":
                kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            try:
                proc = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=runtime["env"],
                    **kwargs,
                )
            except Exception:
                log.close()
                raise

            old_log = self.log_handles.get(key)
            if old_log:
                old_log.close()
            self.processes[key] = proc
            self.log_handles[key] = log
            return {"name": name, "station": station, "started": True, "pid": proc.pid, "log": str(log_path)}

    def stop(self, name: str, station: str | None = None) -> dict[str, Any]:
        key = self.process_key(name, station)
        with self.lock:
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
                proc.kill()
                proc.wait(timeout=5)
            handle = self.log_handles.pop(key, None)
            if handle:
                handle.close()
            return {"name": name, "station": station, "stopped": True}

    def ensure_station_topics(self, station: str) -> dict[str, Any]:
        """Create missing station topics. Never deletes or truncates topics."""
        config = read_station_config()
        stations = config.get("stations", {})
        if station not in stations:
            raise ValueError(f"Unknown station '{station}'.")
        entry = stations[station]
        topic_names = [entry.get("raw_topic"), entry.get("featured_topic")]
        if not all(topic_names):
            raise ValueError(f"Station '{station}' must define raw_topic and featured_topic.")
        if not self.kafka_available():
            raise RuntimeError("Kafka broker is not reachable at 127.0.0.1:9092. Start Kafka first.")
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
                futures = admin.create_topics(
                    new_topics=[NewTopic(name, num_partitions=1, replication_factor=1) for name in missing],
                    validate_only=False,
                )
                for topic, future in futures.items():
                    try:
                        future.get(timeout=20)
                        created.append(topic)
                    except Exception as exc:
                        # A concurrent create is acceptable only if the topic now exists.
                        refreshed = set(admin.list_topics())
                        if topic not in refreshed:
                            raise RuntimeError(f"Could not create Kafka topic '{topic}': {exc}") from exc
            refreshed = set(admin.list_topics())
            return {
                "station": station,
                "created": created,
                "existing": [name for name in topic_names if name in existing],
                "topics": {name: (name in refreshed) for name in topic_names},
            }
        finally:
            admin.close()

    def start_model_experiment(
        self,
        station: str,
        algorithm: str,
        parameters: dict[str, Any],
        training_options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Launch an immutable, versioned model training experiment."""
        stations = read_station_config().get("stations", {})
        if station not in stations:
            raise ValueError(f"Unknown station '{station}'.")
        entry = stations[station]
        training_file = entry.get("training_file") or entry.get("input_file")
        if not training_file or not resolve_root_path(training_file).is_file():
            raise FileNotFoundError(f"Training CSV not found for '{station}': {training_file}")

        key = self.process_key("training", station)
        existing = self.processes.get(key)
        if existing and existing.poll() is None:
            raise RuntimeError(f"A training job is already running for station '{station}'.")

        experiment_id = f"{station}-{algorithm}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        experiment_dir = ROOT / "models" / station / "experiments" / experiment_id
        experiment_dir.mkdir(parents=True, exist_ok=True)
        config_path = experiment_dir / "experiment.json"
        experiment_config = {
            "experiment_id": experiment_id,
            "station": station,
            "algorithm": algorithm,
            "parameters": parameters,
            "training_options": training_options or {},
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
                "status": "queued",
                "parameters": parameters,
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
            config_path.unlink(missing_ok=True)
            upsert_experiment({
                "experiment_id": experiment_id,
                "station": station,
                "algorithm": algorithm,
                "status": "failed",
                "parameters": parameters,
                "config_path": config_path.relative_to(ROOT).as_posix(),
                "created_at": experiment_config["created_at"],
                "error": f"Training process could not be started: {exc}",
                "completed_at": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            })
            raise
        return {
            "station": station,
            "experiment_id": experiment_id,
            "algorithm": algorithm,
            "started": bool(result.get("started")),
            "pid": result.get("pid"),
            "log": result.get("log"),
            "status": "queued",
        }

    def start_training(self, station: str) -> dict[str, Any]:
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
        config = read_station_config()
        station = station or config.get("active_station")
        stations = config.get("stations", {})
        if station not in stations:
            raise ValueError(f"Station '{station}' is not configured.")
        if not self.kafka_available():
            raise RuntimeError("Kafka broker is not reachable at 127.0.0.1:9092. Start Kafka first.")

        entry = stations[station]
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
        for label, path in (("stream CSV", stream_file), ("history CSV", history_file), ("model artifact", checkpoint)):
            if not path.is_file():
                raise FileNotFoundError(f"Station '{station}' {label} not found: {path}")

        self.get_runtime_environment()
        topic_result = self.ensure_station_topics(station)
        python = sys.executable
        history_arg = ["--history-file", str(history_file)]
        commands = [
            ("spark", [python, "-u", "-m", "src.spark.streaming", "--station", station]),
            ("predictor", [python, "-u", "-m", "src.models.streaming_predictor", "--station", station,
                           "--model", str(checkpoint), *history_arg,
                           "--output", str(resolve_root_path(entry["forecasts_file"])),
                           "--actual-output", str(resolve_root_path(entry["actuals_file"]))]),
            ("producer", [python, "-u", "-m", "src.producers.csv_producer", "--station", station, *history_arg,
                           "--delay", str(float(entry.get("stream_delay_seconds", 1.0)))]),
        ]
        started: list[str] = []
        try:
            for name, command in commands:
                result = self.start(name, command, station)
                if result.get("started"):
                    started.append(name)
                # Give Spark more time to initialize before predictor and producer start.
                startup_wait = {"spark": 15, "predictor": 5, "producer": 2}.get(name, 2)
                time.sleep(startup_wait)
                proc = self.processes.get(self.process_key(name, station))
                if proc and proc.poll() is not None:
                    raise RuntimeError(f"{station} {name} exited during startup with code {proc.returncode}. Check logs/{station}_{name}.log")
            registry = read_station_config()
            registry["active_station"] = station
            active_stations = set(registry.get("active_stations", []) or [])
            active_stations.add(station)
            registry["active_stations"] = sorted(active_stations)
            registry["stations"][station]["enabled"] = True
            registry["stations"][station]["status"] = "active"
            registry["stations"][station]["topics_ready"] = True
            write_station_config(registry)
            return {"station": station, "started": started, "topics": topic_result, "status": self.station_status(station)}
        except Exception:
            for name in reversed(started):
                self.stop(name, station)
            raise

    def stop_pipeline(self, station: str | None = None) -> dict[str, Any]:
        results: dict[str, Any] = {}
        if station:
            for name in ("producer", "predictor", "spark", "training"):
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
            return {"station": station, "results": results}

        # Backwards-compatible global stop: stop all managed station processes.
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
        with self.lock:
            for service in ("spark", "predictor", "producer", "training"):
                proc = self.processes.get(self.process_key(service, station))
                if proc and proc.poll() is None:
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
                "$mods='src\\.producers\\.csv_producer|src\\.spark\\.streaming|src\\.models\\.streaming_predictor'; "
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

    def _kafka_cli_dir(self) -> Path:
        kafka_cfg = read_kafka_config()
        configured = kafka_cfg.get("kafka_cli_dir") or os.environ.get("KAFKA_HOME")
        if configured:
            base = Path(configured).expanduser()
            if base.name.lower() != "windows" and (base / "bin" / "windows").is_dir():
                base = base / "bin" / "windows"
            elif base.name.lower() != "windows" and (base / "windows").is_dir():
                base = base / "windows"
        else:
            base = Path(r"C:\kafka_2.13-4.3.1\bin\windows")
        if not base.is_absolute():
            base = ROOT / base
        base = base.resolve()
        required = ("kafka-topics.bat", "kafka-delete-records.bat")
        missing = [name for name in required if not (base / name).is_file()]
        if missing:
            raise FileNotFoundError(
                "Kafka command-line tools were not found. Configure kafka_cli_dir "
                "in config/kafka.yaml or set KAFKA_HOME. Expected folder: "
                f"{base}; missing: {', '.join(missing)}"
            )
        if os.name != "nt":
            raise RuntimeError("Dashboard Reset & Replay currently requires the Windows Kafka .bat tools.")
        return base

    @staticmethod
    def _run_bat(path: Path, args: list[str], timeout: int = 60) -> str:
        command = subprocess.list2cmdline([str(path), *args])
        result = subprocess.run(
            command,
            cwd=ROOT,
            shell=True,
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
        cli = self._kafka_cli_dir()
        kafka_cfg = read_kafka_config()
        bootstrap = str(kafka_cfg.get("bootstrap_servers", "127.0.0.1:9092"))
        topics_cli = cli / "kafka-topics.bat"
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
        cli = self._kafka_cli_dir()
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
                cli / "kafka-delete-records.bat",
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
                "Kafka broker is not reachable at 127.0.0.1:9092. Start Kafka first."
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

        # Preflight output schema and checkpoint path before changing Kafka.
        if not isolated_outputs:
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

        topic_result = self._ensure_reset_topics(
            station, raw_topic, featured_topic
        )

        # Kafka topics are retained; only their existing records are expired.
        delete_output = self._clear_topic_records(raw_topic, featured_topic)

        # Back up and clear this station's isolated result files. Legacy
        # stations continue using the original shared-CSV filtering behavior.
        output_backup = None
        if isolated_outputs:
            import datetime as _dt
            backup_dir = ROOT / "data" / "processed" / "reset_backups" / f"{station}_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            for field in ("actuals_file", "forecasts_file"):
                output_path = resolve_root_path(stations[station][field])
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
            "processed_output_backup": str(output_backup) if output_backup else None,
            "pipeline_started": False,
            "services_remain_stopped": True,
            "message": (
                f"{station} reset completed. Kafka records, station output rows and "
                "Spark checkpoint were reset. No services were started. Use Start "
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

        # Complete local preflight checks before mutating Kafka or filesystem records.
        validate_station_output_files(station)
        files_to_delete, dirs_to_delete = self._station_delete_paths(station, entry, stations)

        # Clear only this station's Kafka records. Keep empty topic definitions to avoid
        # Kafka KRaft topic-directory rename failures seen on this Windows installation.
        topic_result = self._ensure_reset_topics(station, raw_topic, featured_topic)
        delete_output = self._clear_topic_records(raw_topic, featured_topic)

        # Remove the selected station's rows; all other station rows remain.
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
            raise ValueError("Unknown process log. Use spark, predictor, producer or training.")
        path = self.log_path(name, station)
        if not path.exists():
            return ""
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


manager = PipelineManager()
