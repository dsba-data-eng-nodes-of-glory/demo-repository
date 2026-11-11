from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_monitoring_uses_fragment_refresh_not_full_page_autorefresh():
    source = (ROOT / "dashboard" / "pages" / "monitoring.py").read_text(encoding="utf-8")
    assert "st.fragment(" in source
    assert "run_every=float(refresh_seconds) if auto_refresh else None" in source
    assert "st_autorefresh" not in source
    assert "fragment_runner(api, station, stations" in source


def test_monitoring_caches_slow_operational_endpoints_separately():
    source = (ROOT / "dashboard" / "pages" / "monitoring.py").read_text(encoding="utf-8")
    assert "@st.cache_data(ttl=60, show_spinner=False)\ndef api_get_station_registry" in source
    assert "@st.cache_data(ttl=15, show_spinner=False)\ndef api_get_readiness" in source
    assert "@st.cache_data(ttl=30, show_spinner=False)\ndef api_get_quality_events" in source


def test_monitoring_keeps_two_second_refresh_option():
    source = (ROOT / "dashboard" / "pages" / "monitoring.py").read_text(encoding="utf-8")
    assert "[2, 5, 10, 15]" in source
    assert 'key="electricity_poc_monitoring_refresh_v4"' in source


def test_demand_actual_fetch_keeps_72_hour_chart_and_24_hour_context_buffer():
    source = (ROOT / "dashboard" / "pages" / "monitoring.py").read_text(encoding="utf-8")
    assert 'actual_params = {"station": station, "limit": 600}' in source
