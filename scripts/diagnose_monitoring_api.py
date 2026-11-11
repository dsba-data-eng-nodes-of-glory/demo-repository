"""Probe the FastAPI endpoints used by Streamlit monitoring."""
from __future__ import annotations
import argparse
import time
import requests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--station", default="amprion")
    parser.add_argument("--end-time", default="2026-10-02T11:00:00Z")
    args = parser.parse_args()
    checks = [
        ("/health/live", {}),
        ("/stations", {}),
        ("/results/renewables", {"station": args.station, "limit": 5000, "end_time": args.end_time}),
    ]
    for path, params in checks:
        started = time.perf_counter()
        try:
            response = requests.get(args.api.rstrip("/") + path, params=params, timeout=(3, 30))
            elapsed = time.perf_counter() - started
            print(f"{path}: HTTP {response.status_code} in {elapsed:.2f}s")
            response.raise_for_status()
            payload = response.json()
            if path.endswith("/renewables"):
                print("  diagnostics:", payload.get("diagnostics", {}))
                print("  actual timestamps:", len(payload.get("actuals", [])))
            elif path == "/stations":
                print("  stations:", list((payload.get("stations") or {}).keys()))
        except requests.RequestException as exc:
            print(f"{path}: FAILED after {time.perf_counter() - started:.2f}s: {exc}")


if __name__ == "__main__":
    main()
