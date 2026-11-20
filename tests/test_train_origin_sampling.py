import numpy as np

from src.models.train_tabular import sample_origins


def test_zero_maximum_keeps_all_origins():
    origins = np.arange(10, dtype=np.int64)
    assert np.array_equal(sample_origins(origins, 0), origins)


def test_positive_maximum_samples_across_full_time_range():
    origins = np.arange(100, dtype=np.int64)
    sampled = sample_origins(origins, 10)
    assert len(sampled) == 10
    assert sampled[0] == 0
    assert sampled[-1] == 99


def test_negative_maximum_is_rejected():
    try:
        sample_origins(np.arange(5), -1)
    except ValueError as exc:
        assert "cannot be negative" in str(exc)
    else:
        raise AssertionError("Expected a negative sample limit to fail")


def test_weather_training_retains_later_periods_without_crossing_missing_weather_or_actuals():
    import numpy as np
    import pandas as pd
    from src.models.evaluation_splits import archived_weather_history, forecast_split_origins

    days = pd.date_range("2026-01-01", periods=100, freq="15min", tz="UTC").delete(32)
    frame = pd.DataFrame({"utc_timestamp": days, "forecast_temperature": 12.})
    frame.loc[:8, "forecast_temperature"] = np.nan
    frame.loc[38:45, "forecast_temperature"] = np.nan
    eligible = archived_weather_history(frame, ["forecast_temperature"], history=4)
    groups, train_end, val_end = forecast_split_origins(eligible, 4, 3, 15, 0.15, True, ["forecast_temperature"])
    assert len(groups) == 3 and all(len(group) for group in groups)
    assert eligible.utc_timestamp.iloc[0] == frame.utc_timestamp.iloc[5]
    assert groups[0][0] == 4
    assert eligible.utc_timestamp.iloc[groups[0][-1]] > frame.utc_timestamp.iloc[45]
    assert groups[0][-1] + 3 <= train_end and groups[1][-1] + 3 <= val_end
    for start in np.concatenate(groups):
        assert eligible.forecast_temperature.iloc[start:start+3].notna().all()
        assert eligible.utc_timestamp.iloc[start-4:start+3].diff().dropna().eq(pd.Timedelta(minutes=15)).all()
