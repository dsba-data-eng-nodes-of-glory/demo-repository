"""Manage the optional distributed infrastructure with Docker Compose.

Usage:
    python -m scripts.cluster_control up
    python -m scripts.cluster_control status
    python -m scripts.cluster_control down
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPOSE = ROOT / "docker-compose.scale.yml"

def main() -> None:
    parser = argparse.ArgumentParser(description="Control the electricity scale infrastructure")
    parser.add_argument("action", choices=("up", "down", "status", "logs"))
    parser.add_argument("--build", action="store_true", help="Build the application image during up")
    args = parser.parse_args()
    if not shutil.which("docker"):
        raise SystemExit("Docker CLI was not found. Install Docker Desktop and enable Compose v2.")
    command = ["docker", "compose", "-f", str(COMPOSE)]
    if args.action == "up":
        command += ["up", "-d"]
        if args.build:
            command.append("--build")
    elif args.action == "down":
        command += ["down"]
    elif args.action == "status":
        command += ["ps"]
    else:
        command += ["logs", "--tail", "100", "-f"]
    subprocess.run(command, cwd=ROOT, check=True)

if __name__ == "__main__":
    main()
