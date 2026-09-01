import numpy as np

from scripts.analysis.analyze_norway_reuse_tests import bh_qvalues, fss_grid_for_fields


def test_fss_grid_is_one_for_identical_members():
    truth = np.zeros((61, 61), dtype=np.float32)
    truth[15:45, 18:48] = 7.0
    predictions = {"Baseline": np.stack([truth, truth])}
    scores = fss_grid_for_fields(predictions, truth)
    assert scores
    assert all(np.isclose(value, 1.0) for value in scores.values())


def test_bh_qvalues_preserve_order_and_bound_values():
    p = np.array([0.04, 0.001, 0.02, 0.8])
    q = bh_qvalues(p)
    assert np.all((q >= 0) & (q <= 1))
    assert list(np.argsort(p)) == list(np.argsort(q))
