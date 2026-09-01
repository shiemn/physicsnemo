import numpy as np

from scripts.analysis.analyze_taiwan_typhoons import (
    fss,
    orientation_difference_deg,
    weighted_morphology,
)


def test_fss_is_one_for_identical_fields():
    field = np.zeros((9, 9), dtype=np.float32)
    field[2:6, 3:8] = 30.0
    assert fss(field, field, threshold=20.0, scale_px=3) == 1.0


def test_fss_rewards_neighborhood_match_at_larger_scale():
    truth = np.zeros((15, 15), dtype=np.float32)
    prediction = np.zeros_like(truth)
    truth[7, 7] = 30.0
    prediction[7, 8] = 30.0
    assert fss(prediction, truth, 20.0, 5) > fss(prediction, truth, 20.0, 1)


def test_orientation_difference_is_axis_symmetric():
    assert orientation_difference_deg(5.0, 175.0) == 10.0
    assert orientation_difference_deg(10.0, 100.0) == 90.0


def test_weighted_morphology_recovers_horizontal_axis():
    x, y = np.meshgrid(np.arange(-5.0, 6.0), np.arange(-5.0, 6.0))
    field = np.zeros_like(x)
    field[5, 2:9] = 40.0
    shape = weighted_morphology(field, x, y, np.hypot(x, y), threshold=20.0)
    assert min(shape["orientation_deg"], 180.0 - shape["orientation_deg"]) < 1e-6
    assert shape["anisotropy"] > 0.9
