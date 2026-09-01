import numpy as np
import pandas as pd

from scripts.analysis.analyze_annual_performance import (
    align_frames,
    chunk_metrics,
    empirical_crps,
    summarize_frame,
)


def test_two_member_empirical_crps_matches_closed_form():
    truth = np.array([[[1.0, 2.0]]])
    prediction = np.array([[[[0.0, 4.0]]], [[[2.0, 2.0]]]])

    actual = empirical_crps(prediction, truth)
    expected = np.mean(np.abs(prediction - truth[None]), axis=(0, 2, 3))
    expected -= np.mean(np.abs(prediction[0] - prediction[1]), axis=(1, 2)) / 4.0

    np.testing.assert_allclose(actual, expected)


def test_chunk_metrics_use_ensemble_mean_for_deterministic_errors():
    truth = np.zeros((2, 1, 2), dtype=np.float32)
    prediction = np.array(
        [
            [[[1.0, 1.0]], [[2.0, 2.0]]],
            [[[3.0, 3.0]], [[4.0, 4.0]]],
        ],
        dtype=np.float32,
    )

    metrics = chunk_metrics(prediction, truth)

    np.testing.assert_allclose(metrics["rmse"], [2.0, 3.0])
    np.testing.assert_allclose(metrics["mae"], [2.0, 3.0])
    np.testing.assert_allclose(metrics["bias"], [2.0, 3.0])


def test_intersection_alignment_and_summary():
    first = pd.DataFrame(
        {
            "time": pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03"]),
            "model": "first",
            "rmse": [1.0, 2.0, 3.0],
            "mae": [1.0, 2.0, 3.0],
            "crps": [1.0, 2.0, 3.0],
            "bias": [1.0, 2.0, 3.0],
        }
    )
    second = first.iloc[1:].copy()
    second["model"] = "second"

    aligned = align_frames([first, second], "intersection")
    summary = summarize_frame(aligned[0])

    assert [len(frame) for frame in aligned] == [2, 2]
    assert summary["timesteps"] == 2
    np.testing.assert_allclose(summary["aggregate_rmse"], np.sqrt(6.5))
