"""Guard 2-second live-only refresh defaults across all customer dashboard pages."""
from pathlib import Path
import ast

ROOT=Path(__file__).resolve().parents[1]/'dashboard'

def source(path):
    text=(ROOT/path).read_text(encoding='utf-8')
    ast.parse(text)
    return text

def test_monitoring_default_2s_and_fragment_only():
    s=source('pages/monitoring.py')
    assert '[2, 5, 10, 15],\n        index=0' in s
    assert 'run_every=float(refresh_seconds) if auto_refresh else None' in s
    assert '_monitoring_default_2s_applied' in s
    assert 'st_autorefresh' not in s

def test_model_analysis_defaults_on_two_seconds_and_fragment_only():
    s=source('pages/model_management.py')
    assert 'st.checkbox("Auto-refresh model analysis", value=True' in s
    assert 'st.selectbox("Refresh every", [2, 5, 10, 15, 30], index=0' in s
    assert 'analysis_panel = st.fragment(run_every=float(refresh_seconds) if auto_refresh else None)' in s
    assert 'analysis_panel(station, selected_component, evaluation_window)' in s
    assert 'st_autorefresh' not in s
    assert '@st.cache_data(ttl=2, show_spinner=False)' in s
    assert 'if active_section == "Configure & Retrain"' in s

def test_pipeline_status_fragment_2s_no_autoposts():
    s=source('pages/pipeline_control.py')
    assert '@st.fragment(run_every=2.0)' in s
    assert 'render_live_pipeline_status(station)' in s
    frag=s.split('def render_live_pipeline_status',1)[1].split('render_live_pipeline_status(station)',1)[0]
    assert 'get_json(f"/stations/{selected_station}/status")' in frag
    assert 'post_json(' not in frag

def test_station_forms_are_not_autorefreshed():
    s=source('pages/station_management.py')
    assert '@st.fragment(run_every=2.0)' in s
    frag=s.split('def render_live_station_status',1)[1].split('render_live_station_status(selected, entry, model)',1)[0]
    assert 'api_get(f"/stations/{station_id}/status")' in frag
    assert 'api_post(' not in frag
    assert 'api_delete(' not in frag

def test_home_metrics_one_batched_call_only():
    s=source('app.py')
    assert '@st.fragment(run_every=2.0)' in s
    frag=s.split('def render_live_overview_metrics',1)[1].split('render_live_overview_metrics()',1)[0]
    assert 'api_get("/stations/overview", timeout=5)' in frag
    assert 'requests.post(' not in frag
