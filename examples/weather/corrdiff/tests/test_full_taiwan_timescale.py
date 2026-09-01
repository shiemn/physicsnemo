import numpy as np

from scripts.analysis.analyze_full_taiwan_timescale import member_mean_fss


def test_member_mean_fss_is_one_for_identical_ensemble():
    truth = np.zeros((21, 21), dtype=np.float32)
    truth[5:15, 7:17] = 40.0
    prediction = np.stack([truth, truth])
    assert np.isclose(member_mean_fss(prediction, truth), 1.0)


def test_member_mean_fss_averages_members():
    truth = np.zeros((21, 21), dtype=np.float32)
    truth[10, 10] = 40.0
    exact = truth.copy()
    displaced = np.zeros_like(truth)
    displaced[10, 14] = 40.0
    ensemble_score = member_mean_fss(np.stack([exact, displaced]), truth)
    assert 0.0 < ensemble_score < 1.0
