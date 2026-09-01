from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analysis.analyze_europa_extreme_cases import (
    case_metrics,
    select_separated_cases,
    top_tail_mean,
)


def test_top_tail_mean_uses_requested_fraction():
    field = np.arange(100, dtype=np.float32).reshape(10, 10)
    assert top_tail_mean(field, 0.02) == 98.5


def test_select_separated_cases_keeps_strongest_independent_peaks():
    catalogue = pd.DataFrame(
        {
            "index": [0, 1, 2, 3],
            "time": pd.to_datetime(
                ["2021-01-01 00:00", "2021-01-01 06:00", "2021-01-02 06:00", "2021-01-04 00:00"]
            ),
            "tail_mean_mm_h": [8.0, 10.0, 9.0, 7.0],
        }
    )
    selected = select_separated_cases(catalogue, count=3, separation_hours=24)
    assert set(selected["index"]) == {1, 2, 3}
    assert list(selected["case"]) == ["E1", "E2", "E3"]


def test_case_metrics_perfect_ensemble():
    truth = np.array([[0.0, 1.0], [5.0, 10.0]], dtype=np.float32)
    prediction = np.stack([truth, truth])
    result = case_metrics("E1", pd.Timestamp("2021-01-01"), "model", prediction, truth)
    assert result["rmse"] == 0.0
    assert result["mae"] == 0.0
    assert result["crps"] == 0.0
    assert result["spatial_correlation"] == 1.0
    assert result["csi_1mm"] == 1.0
    assert result["csi_5mm"] == 1.0
