"""The offline evaluation path must recover real timestamps from predictions.nc.

evaluate.py silently switches to _evaluate_from_file when the cached
predictions file exists (eval.predictions_file defaults to "auto"). That path
used to label its example-event figures with a bare time index while the
online path used the real timestamp. This pins the decoding.
"""

from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from physicsnemo.utils.corrdiff.utils import NetCDFWriter

evaluate = pytest.importorskip("evaluate")
nc = pytest.importorskip("netCDF4")


def _channel(name):
    return SimpleNamespace(name=name, level="")


def _write_predictions(path, times):
    """Write a minimal but structurally real predictions.nc."""
    lat = np.linspace(59.0, 60.0, 4).reshape(2, 2)
    lon = np.linspace(10.0, 11.0, 4).reshape(2, 2)

    with nc.Dataset(path, "w") as f:
        writer = NetCDFWriter(
            f,
            lat=lat,
            lon=lon,
            input_channels=[_channel("hus500")],
            output_channels=[_channel("precipitation")],
            has_lead_time=False,
        )
        for index, moment in enumerate(times):
            writer.write_time(index, moment)
            writer.write_truth("precipitation", index, np.full((2, 2), float(index)))
            writer.write_prediction(
                "precipitation", 0, index, np.full((2, 2), float(index) + 0.5)
            )
            writer.write_input("hus500", index, np.zeros((2, 2)))


def test_load_predictions_netcdf_decodes_times(tmp_path):
    times = [datetime(2005, 6, 1, 0) + timedelta(hours=3 * i) for i in range(3)]
    path = tmp_path / "predictions.nc"
    _write_predictions(path, times)

    data = evaluate._load_predictions_netcdf(str(path))

    assert data["n_times"] == 3
    assert len(data["times"]) == 3
    # cftime objects compare on their fields; check the formatted labels, which
    # is what the event-plot caption actually uses.
    assert [str(t)[:13] for t in data["times"]] == [
        "2005-06-01 00",
        "2005-06-01 03",
        "2005-06-01 06",
    ]


def test_load_predictions_netcdf_survives_missing_time_variable(tmp_path):
    """A file without usable times must still evaluate, just without labels."""
    path = tmp_path / "no_times.nc"
    _write_predictions(path, [datetime(2005, 6, 1, 0)])

    # Blank the units so decoding cannot succeed.
    with nc.Dataset(path, "a") as f:
        f.variables["time"].units = "not a udunits string"

    data = evaluate._load_predictions_netcdf(str(path))

    assert data["n_times"] == 1
    assert data["times"] == []
