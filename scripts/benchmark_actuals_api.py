"""Measure repeated FastAPI actuals requests to confirm cache warm-up behaviour."""
from __future__ import annotations

import argparse
import statistics
import time

import requests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--station", default="amprion")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument(
        "--end-time",
        default=None,
        help="Optional forecast origin/cutoff, for example 2026-10-09T20:45:00Z",
    )
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    params = {"station": args.station, "limit": args.limit}
    if args.end_time:
        params["end_time"] = args.end_time

    timings = []
    url = f"{args.api.rstrip('/')}/results/actuals"
    for index in range(max(1, args.runs)):
        started = time.perf_counter()
        try:
            response = requests.get(url, params=params, timeout=(3, 60))
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            elapsed = time.perf_counter() - started
            print(f"Run {index + 1}: FAILED after {elapsed:.2f}s: {exc}")
            raise SystemExit(1) from exc

        elapsed = time.perf_counter() - started
        timings.append(elapsed)
        records = payload.get("records", [])
        print(
            f"Run {index + 1}: HTTP {response.status_code}, "
            f"{elapsed:.3f}s, {len(records):,} records"
        )

    if len(timings) > 1:
        print(f"Warm-call median: {statistics.median(timings[1:]):.3f}s")
    print(f"All-run median: {statistics.median(timings):.3f}s")


if __name__ == "__main__":
    main()
