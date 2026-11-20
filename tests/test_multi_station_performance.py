"""Regressions for shared Spark status and overlapping hierarchy polls."""

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

from backend import pipeline_manager, results_api
from src.spark.control import write_control


def test_hierarchy_coalesces_same_view_and_keeps_other_views_separate(monkeypatch):
    monkeypatch.setattr(results_api, "_GROUP_RESULTS", OrderedDict())
    monkeypatch.setattr(results_api, "_GROUP_PENDING", {})
    requests = []
    first = {"forecasts": [96]}

    def compute(node, window, include_forecasts):
        requests.append((node, window, include_forecasts))
        return first if include_forecasts else {"forecasts": []}

    monkeypatch.setattr(results_api, "hierarchy_results", compute)
    gate = Barrier(4)

    def poll():
        gate.wait(timeout=2)
        return results_api.cached_hierarchy_results("germany", 288, True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(poll) for _ in range(4)]
        assert [future.result(timeout=3) for future in futures] == [first] * 4
    assert requests == [("germany", 288, True)]

    actuals_only = results_api.cached_hierarchy_results("germany", 288, False)
    assert actuals_only == {"forecasts": []}
    assert requests[-1] == ("germany", 288, False)


def test_shared_spark_status_requires_matching_worker_and_station_launch(tmp_path):
    manager = pipeline_manager.PipelineManager()
    manager._spark_dir = tmp_path
    manager._spark_requests.update(amprion="latest", hertz="new-token")
    write_control(tmp_path / "status.json", {
        "pid": 123, "stations": {
            "amprion": {"token": "latest", "state": "running"},
            "hertz": {"token": "old-token", "state": "running"},
            "pending_stop": {"token": "removed", "state": "running"},
        },
    })
    live = {"pid": 123, "state": "running", "running": True}
    states = manager._shared_query_status(live)
    assert states["amprion"]["running"]
    assert states["hertz"]["state"] == "starting"
    assert states["pending_stop"]["state"] == "stopping"
    assert manager._shared_query_status(dict(live, pid=777))["amprion"]["state"] == "starting"
