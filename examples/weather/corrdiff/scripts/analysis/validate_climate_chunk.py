#!/usr/bin/env python3
"""Validate a completed climate-only CorrDiff chunk."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import netCDF4 as nc
import numpy as np


GRID_VARIABLES = (
    "prediction_step_sum",
    "prediction_step_count",
    "prediction_daily_sum",
    "prediction_daily_count",
    "prediction_wet_day_sum",
    "prediction_wet_day_count",
    "prediction_rx1day",
    "prediction_rx3h",
    "target_step_sum",
    "target_step_count",
    "target_daily_sum",
    "target_daily_count",
    "target_wet_day_sum",
    "target_wet_day_count",
    "target_rx1day",
    "target_rx3h",
)


def _plain(variable) -> np.ndarray:
    values = variable[:]
    if np.ma.isMaskedArray(values):
        values = values.filled(np.nan)
    return np.asarray(values)


def validate_chunk(
    path: Path,
    expected_years: list[int],
    expected_days_per_year: list[int],
) -> dict:
    if len(expected_days_per_year) != len(expected_years):
        raise ValueError("Provide one expected day count per expected year")
    expected_steps_per_year = [days * 8 for days in expected_days_per_year]
    expected_timesteps = sum(expected_steps_per_year)
    if not path.is_file():
        raise AssertionError(f"Missing climate chunk: {path}")
    if Path(f"{path}.tmp").exists():
        raise AssertionError(f"Temporary output still exists: {path}.tmp")
    if (path.parent / "predictions.nc").exists():
        raise AssertionError("Climate-only run unexpectedly wrote predictions.nc")

    with nc.Dataset(path) as ds:
        assert int(ds.getncattr("complete")) == 1
        assert int(ds.getncattr("completed_timesteps")) == expected_timesteps
        years = _plain(ds.variables["year"]).astype(int).tolist()
        assert years == expected_years, (years, expected_years)
        assert (
            _plain(ds.variables["complete_days"]).astype(int).tolist()
            == expected_days_per_year
        )
        assert _plain(ds.variables["incomplete_days"]).astype(int).tolist() == [
            0
        ] * len(expected_years)
        assert _plain(ds.variables["timesteps"]).astype(int).sum() == expected_timesteps

        expected_shape = (len(expected_years), 512, 512)
        for name in GRID_VARIABLES:
            assert name in ds.variables, f"Missing variable {name}"
            values = _plain(ds.variables[name])
            assert values.shape == expected_shape, (name, values.shape)
            assert np.isfinite(values).all(), f"Non-finite values in {name}"

        for year_idx, (expected_steps, expected_days) in enumerate(
            zip(expected_steps_per_year, expected_days_per_year)
        ):
            assert np.all(
                _plain(ds.variables["prediction_step_count"])[year_idx]
                == expected_steps
            )
            assert np.all(
                _plain(ds.variables["target_step_count"])[year_idx]
                == expected_steps
            )
            assert np.all(
                _plain(ds.variables["prediction_daily_count"])[year_idx]
                == expected_days
            )
            assert np.all(
                _plain(ds.variables["target_daily_count"])[year_idx]
                == expected_days
            )

        required_attrs = (
            "model_regression_checkpoint",
            "model_diffusion_checkpoint",
            "seed_mode",
            "seed_base",
            "num_ensembles",
            "temporal_offsets_hours",
            "requested_first_time",
            "requested_last_time",
        )
        missing_attrs = [name for name in required_attrs if name not in ds.ncattrs()]
        assert not missing_attrs, f"Missing provenance attributes: {missing_attrs}"
        assert ds.getncattr("seed_mode") == "timestamp"
        assert int(ds.getncattr("num_ensembles")) == 1

        return {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "years": years,
            "completed_timesteps": int(ds.getncattr("completed_timesteps")),
            "complete_days": _plain(ds.variables["complete_days"]).astype(int).tolist(),
            "incomplete_days": _plain(ds.variables["incomplete_days"]).astype(int).tolist(),
            "grid_shape": list(expected_shape[1:]),
            "seed_mode": ds.getncattr("seed_mode"),
            "num_ensembles": int(ds.getncattr("num_ensembles")),
            "temporal_offsets_hours": json.loads(
                ds.getncattr("temporal_offsets_hours")
            ),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--expected-years", type=int, nargs="+", required=True)
    parser.add_argument(
        "--expected-days-per-year", type=int, nargs="+", required=True
    )
    args = parser.parse_args()
    result = validate_chunk(
        args.path,
        expected_years=args.expected_years,
        expected_days_per_year=args.expected_days_per_year,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
