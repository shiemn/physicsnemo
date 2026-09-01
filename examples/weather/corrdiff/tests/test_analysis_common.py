import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.common import (
    member_mean_fss,
    parse_model,
    read_times,
    season_for_month,
    time_indices,
)


class _TimeVariable:
    units = "hours since 2021-01-01 00:00:00"
    calendar = "standard"

    def __getitem__(self, key):
        return np.array([0, 3])


class _Dataset:
    variables = {"time": _TimeVariable()}


def test_parse_model_requires_label_and_path():
    source = parse_model("past3h=/tmp/predictions.nc")
    assert source.label == "past3h"
    assert source.path == Path("/tmp/predictions.nc")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_model("missing-separator")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_model("=/tmp/predictions.nc")


def test_read_times_preserves_requested_timezone_policy():
    utc = read_times(_Dataset())
    naive = read_times(_Dataset(), utc=False)
    assert str(utc.tz) == "UTC"
    assert naive.tz is None
    assert list(naive) == list(
        pd.to_datetime(["2021-01-01 00:00:00", "2021-01-01 03:00:00"])
    )


def test_time_indices_rejects_missing_timestamps():
    native = pd.DatetimeIndex(pd.to_datetime(["2021-01-01", "2021-01-02"]))
    selected = pd.DatetimeIndex(pd.to_datetime(["2021-01-02"]))
    np.testing.assert_array_equal(time_indices(native, selected), [1])
    with pytest.raises(ValueError, match="not present"):
        time_indices(
            native, pd.DatetimeIndex(pd.to_datetime(["2021-01-03"]))
        )


def test_member_mean_fss_handles_matches_and_empty_events():
    truth = np.zeros((9, 9), dtype=np.float32)
    truth[3:6, 3:6] = 2.0
    prediction = np.stack([truth, truth])
    assert member_mean_fss(
        prediction, truth, threshold=1.0, scale_px=3
    ) == pytest.approx(1.0)
    assert np.isnan(
        member_mean_fss(
            np.zeros_like(prediction),
            np.zeros_like(truth),
            threshold=1.0,
            scale_px=3,
        )
    )


@pytest.mark.parametrize(
    ("month", "season"),
    [(1, "DJF"), (3, "MAM"), (6, "JJA"), (9, "SON"), (12, "DJF")],
)
def test_season_for_month(month, season):
    assert season_for_month(month) == season
