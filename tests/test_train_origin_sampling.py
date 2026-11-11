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
