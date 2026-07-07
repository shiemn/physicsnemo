import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import cwb


class DummyGroup(dict):
    pass


def test_v3_europa_normalization_round_trips_log_precipitation():
    group = DummyGroup(
        cwb_variable=np.array(
            [
                "temperature_2m",
                "eastward_wind_10m",
                "northward_wind_10m",
                "precipitation_amount_1hr",
            ]
        ),
        cwb_center=np.array([280.0, 1.5, -2.0, 0.4], dtype=np.float32),
        cwb_scale=np.array([8.0, 3.0, 4.0, 0.7], dtype=np.float32),
    )

    center, scale, fwd, inv = cwb.get_target_normalizations_v3_europa(group)

    np.testing.assert_allclose(center, [280.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(scale, [8.0, 3.0, 4.0, 1.0])
    assert fwd[:3] == [None, None, None]
    assert inv[:3] == [None, None, None]

    precip = np.array([0.0, 1.0, 5.0, 50.0], dtype=np.float32)
    transformed = fwd[3](precip)
    restored = inv[3](transformed)

    np.testing.assert_allclose(transformed, np.log1p(precip))
    np.testing.assert_allclose(restored, precip, rtol=1e-6)


def test_linear_europa_normalization_keeps_precipitation_in_physical_units():
    group = DummyGroup(
        cwb_variable=np.array(
            [
                "temperature_2m",
                "eastward_wind_10m",
                "northward_wind_10m",
                "precipitation_amount_1hr",
            ]
        ),
        cwb_center=np.array([280.0, 1.5, -2.0, 0.4], dtype=np.float32),
        cwb_scale=np.array([8.0, 3.0, 4.0, 0.7], dtype=np.float32),
    )

    center, scale = cwb.get_target_normalizations_europa(group)

    np.testing.assert_allclose(center, [280.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(scale, [8.0, 3.0, 4.0, 5.0])
