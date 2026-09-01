import numpy as np
from scipy.ndimage import shift

from scripts.analysis.analyze_full_norway_timescale import (
    member_mean_fss,
    phase_motion_and_deformation,
)


def test_member_mean_fss_is_one_for_identical_ensemble():
    truth = np.zeros((31, 31), dtype=np.float32)
    truth[8:22, 10:24] = 2.0
    assert np.isclose(member_mean_fss(np.stack([truth, truth]), truth), 1.0)


def test_phase_motion_reports_speed_and_removes_translation():
    first = np.zeros((128, 128), dtype=np.float32)
    first[35:55, 48:75] = 1.0
    second = shift(first, shift=(5, 3), order=1, mode="constant")
    speed, aligned_change, response = phase_motion_and_deformation(first, second)
    assert response > 0.02
    assert np.isclose(speed, np.hypot(3, 5) * 2.0 / 3.0, atol=0.12)
    assert aligned_change < 0.03
