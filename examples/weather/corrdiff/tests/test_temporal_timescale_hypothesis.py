import numpy as np
import pandas as pd
from scipy.ndimage import shift

from scripts.analysis.analyze_temporal_timescale_hypothesis import (
    aggregate_storm_days,
    field_correlation,
    normalized_change,
    phase_motion_and_deformation,
)


def test_regime_metrics_for_identical_fields():
    field = np.arange(25, dtype=np.float32).reshape(5, 5)
    assert np.isclose(field_correlation(field, field), 1.0)
    assert normalized_change(field, field) == 0.0


def test_normalized_change_is_scale_invariant():
    first = np.ones((4, 4), dtype=np.float32)
    second = np.full((4, 4), 2.0, dtype=np.float32)
    assert np.isclose(
        normalized_change(first, second), normalized_change(first * 10, second * 10)
    )


def test_daily_aggregation_uses_active_hours_and_minimum_count():
    times = pd.date_range("2021-01-01", periods=14, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "storm": ["Example"] * 14,
            "time": times,
            "active_area_ge_20dbz_km2": [6_000.0] * 8 + [0.0] * 6,
            "rmse_3h_minus_1h": np.arange(14, dtype=float),
        }
    )
    daily = aggregate_storm_days(frame)
    assert len(daily) == 1
    assert daily.loc[0, "valid_hours"] == 8
    assert np.isclose(daily.loc[0, "rmse_3h_minus_1h"], 3.5)


def test_phase_motion_recovers_translation_and_removes_deformation():
    first = np.zeros((128, 128), dtype=np.float32)
    first[50:70, 55:80] = 1.0
    second = shift(first, shift=(3, 5), order=1)
    motion_km, aligned_change, response = phase_motion_and_deformation(first, second)
    assert np.isclose(motion_km, np.hypot(3, 5) * 3.0, atol=0.2)
    assert aligned_change < 0.01
    assert response > 0.9
