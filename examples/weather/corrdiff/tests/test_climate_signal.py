from datetime import datetime, timedelta
from pathlib import Path
import sys

import netCDF4 as nc
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers.climate_signal import ClimateAccumulator, timestamp_seeds


def _day(start: datetime):
    return [start + timedelta(hours=3 * index) for index in range(8)]


def test_timestamp_seeds_are_reproducible_and_vary_by_time_and_member():
    timestamp = datetime(2081, 7, 3, 12)
    first = timestamp_seeds(timestamp, n_members=2, base_seed=17)
    np.testing.assert_array_equal(first, timestamp_seeds(timestamp, 2, 17))
    assert first[0] != first[1]
    assert first[0] != timestamp_seeds(timestamp + timedelta(hours=3), 1, 17)[0]
    assert first[0] != timestamp_seeds(timestamp, 1, 18)[0]
    assert np.all((first >= 0) & (first < 2**32))


def test_complete_days_accumulate_mean_sdii_and_extremes(tmp_path):
    accumulator = ClimateAccumulator(shape=(2, 2), wet_day_threshold_mm=1.0)
    start = datetime(2004, 2, 28)
    # Daily totals are 0.8 mm (dry) and 1.6 mm (wet), respectively.
    for timestamp in _day(start):
        accumulator.update(
            timestamp,
            prediction=np.full((2, 2), 0.1),
            target=np.full((2, 2), 0.2),
        )
    for timestamp in _day(start + timedelta(days=1)):  # leap day
        accumulator.update(
            timestamp,
            prediction=np.full((2, 2), 0.3),
            target=np.full((2, 2), 0.4),
        )

    output = accumulator.write_netcdf(
        tmp_path / "chunk.nc",
        latitude=np.ones((2, 2)),
        longitude=np.ones((2, 2)) * 10,
        metadata={"model": "test"},
    )
    assert output.exists()
    assert not (tmp_path / "chunk.nc.tmp").exists()

    with nc.Dataset(output) as ds:
        assert ds.complete == 1
        assert ds.completed_timesteps == 16
        assert ds.model == "test"
        assert ds.variables["complete_days"][:] == 2
        assert ds.variables["incomplete_days"][:] == 0
        np.testing.assert_allclose(ds.variables["prediction_step_sum"][:], 3.2)
        np.testing.assert_allclose(ds.variables["prediction_daily_sum"][:], 3.2)
        np.testing.assert_allclose(ds.variables["prediction_daily_count"][:], 2)
        np.testing.assert_allclose(ds.variables["prediction_wet_day_sum"][:], 2.4)
        np.testing.assert_allclose(ds.variables["prediction_wet_day_count"][:], 1)
        np.testing.assert_allclose(ds.variables["prediction_rx1day"][:], 2.4)
        np.testing.assert_allclose(ds.variables["prediction_rx3h"][:], 0.3)
        np.testing.assert_allclose(ds.variables["target_rx1day"][:], 3.2)


def test_incomplete_day_is_discarded_only_from_daily_statistics(tmp_path):
    accumulator = ClimateAccumulator(shape=(1, 1))
    timestamps = _day(datetime(2005, 1, 2))[:-1]
    for timestamp in timestamps:
        accumulator.update(timestamp, np.ones((1, 1)), np.ones((1, 1)) * 2)
    output = accumulator.write_netcdf(tmp_path / "incomplete.nc")

    with nc.Dataset(output) as ds:
        assert ds.variables["timesteps"][:] == 7
        assert ds.variables["complete_days"][:] == 0
        assert ds.variables["incomplete_days"][:] == 1
        assert ds.variables["prediction_step_count"][:] == 7
        assert ds.variables["prediction_daily_count"][:] == 0
        np.testing.assert_allclose(ds.variables["prediction_rx3h"][:], 1.0)
        assert np.ma.getmaskarray(ds.variables["prediction_rx1day"][:]).all()


def test_missing_pixel_value_excludes_that_pixel_from_daily_statistic(tmp_path):
    accumulator = ClimateAccumulator(shape=(1, 2))
    for index, timestamp in enumerate(_day(datetime(2005, 1, 2))):
        prediction = np.ones((1, 2))
        if index == 3:
            prediction[0, 1] = np.nan
        accumulator.update(timestamp, prediction, np.ones((1, 2)))
    output = accumulator.write_netcdf(tmp_path / "missing.nc")

    with nc.Dataset(output) as ds:
        np.testing.assert_array_equal(
            ds.variables["prediction_daily_count"][:], [[[1, 0]]]
        )
        np.testing.assert_array_equal(
            ds.variables["prediction_step_count"][:], [[[8, 7]]]
        )


def test_rejects_nonmonotonic_timestamps_and_existing_output(tmp_path):
    accumulator = ClimateAccumulator(shape=(1, 1))
    timestamp = datetime(2005, 1, 2)
    accumulator.update(timestamp, np.ones((1, 1)), np.ones((1, 1)))
    with pytest.raises(ValueError, match="strictly increasing"):
        accumulator.update(timestamp, np.ones((1, 1)), np.ones((1, 1)))

    output = accumulator.write_netcdf(tmp_path / "chunk.nc")
    assert output.exists()
    with pytest.raises(FileExistsError, match="already exists"):
        accumulator.write_netcdf(output)
