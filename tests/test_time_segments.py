import numpy as np
import pandas as pd

from src.models.time_segments import (
    continuous_segment_bounds, sample_origins_by_segment, split_forecast_origins,
)


def test_windows_and_holdout_splits_never_cross_gaps():
    first = pd.date_range("2026-01-01T00:00:00Z", periods=20, freq="15min")
    second = pd.date_range("2026-01-01T08:00:00Z", periods=20, freq="15min")
    timestamps = first.append(second)

    bounds = continuous_segment_bounds(timestamps, 15)
    train_origins, validation_origins = split_forecast_origins(
        timestamps, history_steps=4, horizon_steps=2,
        interval_minutes=15, validation_fraction=0.2,
    )

    assert bounds == [(0, 20), (20, 40)]
    assert len(train_origins) > 0
    assert len(validation_origins) > 0
    for origin in np.concatenate((train_origins, validation_origins)):
        window = timestamps[origin - 4:origin + 2]
        assert len(window) == 6
        assert np.all(window[1:] - window[:-1] == pd.Timedelta(minutes=15))


def test_origin_sampling_keeps_every_eligible_segment_represented():
    segments = [
        pd.date_range("2026-01-01T00:00:00Z", periods=40, freq="15min"),
        pd.date_range("2026-01-02T00:00:00Z", periods=40, freq="15min"),
        pd.date_range("2026-01-03T00:00:00Z", periods=40, freq="15min"),
    ]
    timestamps = segments[0].append(segments[1]).append(segments[2])
    train, validation = split_forecast_origins(
        timestamps, history_steps=4, horizon_steps=2,
        interval_minutes=15, validation_fraction=0.2,
    )

    sampled_train = sample_origins_by_segment(train, timestamps, 15, maximum=6)
    sampled_validation = sample_origins_by_segment(validation, timestamps, 15, maximum=3)

    assert len(sampled_train) == 6
    assert len(sampled_validation) == 3
    assert {int(value // 40) for value in sampled_train} == {0, 1, 2}
    assert {int(value // 40) for value in sampled_validation} == {0, 1, 2}
