"""Streaming precipitation statistics for climate-signal evaluation.

The accumulator deliberately stores sufficient statistics rather than the
full prediction time series.  Chunk files can therefore be merged exactly for
means and SDII, while annual maxima remain available for Rx1day and Rx3h.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import netCDF4 as nc
import numpy as np


UINT32_MODULUS = 2**32
DEFAULT_EXPECTED_HOURS = (0, 3, 6, 9, 12, 15, 18, 21)


def timestamp_seeds(timestamp, n_members: int = 1, base_seed: int = 0) -> np.ndarray:
    """Return reproducible, timestamp-varying uint32-compatible seeds."""
    if n_members < 1:
        raise ValueError("n_members must be at least 1")
    stamp = (
        int(timestamp.year) * 1_000_000
        + int(timestamp.month) * 10_000
        + int(timestamp.day) * 100
        + int(timestamp.hour)
    )
    return np.asarray(
        [
            (int(base_seed) + stamp + member * 1_000_003) % UINT32_MODULUS
            for member in range(n_members)
        ],
        dtype=np.int64,
    )


@dataclass
class _SourceStats:
    step_sum: np.ndarray
    step_count: np.ndarray
    daily_sum: np.ndarray
    daily_count: np.ndarray
    wet_day_sum: np.ndarray
    wet_day_count: np.ndarray
    rx1day: np.ndarray
    rx3h: np.ndarray


@dataclass
class _YearStats:
    prediction: _SourceStats
    target: _SourceStats
    complete_days: int = 0
    incomplete_days: int = 0
    timesteps: int = 0


@dataclass
class _DailyBuffer:
    date_key: tuple[int, int, int]
    hours: list[int] = field(default_factory=list)
    prediction_sum: np.ndarray | None = None
    prediction_count: np.ndarray | None = None
    target_sum: np.ndarray | None = None
    target_count: np.ndarray | None = None


def _empty_source(shape: tuple[int, int]) -> _SourceStats:
    zeros = lambda dtype: np.zeros(shape, dtype=dtype)
    return _SourceStats(
        step_sum=zeros(np.float64),
        step_count=zeros(np.uint32),
        daily_sum=zeros(np.float64),
        daily_count=zeros(np.uint32),
        wet_day_sum=zeros(np.float64),
        wet_day_count=zeros(np.uint32),
        rx1day=np.full(shape, np.nan, dtype=np.float64),
        rx3h=np.full(shape, np.nan, dtype=np.float64),
    )


def _update_sum_count(total, count, values) -> None:
    finite = np.isfinite(values)
    total += np.where(finite, values, 0.0)
    count += finite.astype(np.uint32)


def _update_max(current, values) -> None:
    finite = np.isfinite(values)
    current[:] = np.where(
        finite,
        np.where(np.isfinite(current), np.maximum(current, values), values),
        current,
    )


class ClimateAccumulator:
    """Accumulate annual precipitation statistics from ordered 3-hour fields."""

    def __init__(
        self,
        shape: Sequence[int],
        wet_day_threshold_mm: float = 1.0,
        expected_hours: Sequence[int] = DEFAULT_EXPECTED_HOURS,
    ) -> None:
        if len(shape) != 2:
            raise ValueError(f"Climate fields must be 2-D, got shape={tuple(shape)}")
        self.shape = (int(shape[0]), int(shape[1]))
        self.wet_day_threshold_mm = float(wet_day_threshold_mm)
        self.expected_hours = tuple(int(hour) for hour in expected_hours)
        if len(set(self.expected_hours)) != len(self.expected_hours):
            raise ValueError("expected_hours must not contain duplicates")
        self._years: dict[int, _YearStats] = {}
        self._day: _DailyBuffer | None = None
        self._last_timestamp_key: tuple[int, int, int, int, int, int] | None = None
        self.completed_timesteps = 0
        self._finalized = False

    @property
    def years(self) -> tuple[int, ...]:
        return tuple(sorted(self._years))

    def _year(self, year: int) -> _YearStats:
        if year not in self._years:
            self._years[year] = _YearStats(
                prediction=_empty_source(self.shape),
                target=_empty_source(self.shape),
            )
        return self._years[year]

    def update(self, timestamp, prediction: np.ndarray, target: np.ndarray) -> None:
        if self._finalized:
            raise RuntimeError("Cannot update a finalized ClimateAccumulator")
        prediction = np.asarray(prediction, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        if prediction.shape != self.shape or target.shape != self.shape:
            raise ValueError(
                f"Expected prediction and target shape {self.shape}, got "
                f"{prediction.shape} and {target.shape}"
            )

        timestamp_key = tuple(
            int(getattr(timestamp, part, 0))
            for part in ("year", "month", "day", "hour", "minute", "second")
        )
        if self._last_timestamp_key is not None and timestamp_key <= self._last_timestamp_key:
            raise ValueError("Climate timestamps must be unique and strictly increasing")
        self._last_timestamp_key = timestamp_key

        year, month, day, hour, _, _ = timestamp_key
        date_key = (year, month, day)
        if self._day is not None and date_key != self._day.date_key:
            self._finalize_day()
        if self._day is None:
            self._day = _DailyBuffer(
                date_key=date_key,
                prediction_sum=np.zeros(self.shape, dtype=np.float64),
                prediction_count=np.zeros(self.shape, dtype=np.uint8),
                target_sum=np.zeros(self.shape, dtype=np.float64),
                target_count=np.zeros(self.shape, dtype=np.uint8),
            )
        if hour in self._day.hours:
            raise ValueError(f"Duplicate hour {hour} for {date_key}")
        self._day.hours.append(hour)

        stats = self._year(year)
        _update_sum_count(stats.prediction.step_sum, stats.prediction.step_count, prediction)
        _update_sum_count(stats.target.step_sum, stats.target.step_count, target)
        _update_max(stats.prediction.rx3h, prediction)
        _update_max(stats.target.rx3h, target)
        _update_sum_count(self._day.prediction_sum, self._day.prediction_count, prediction)
        _update_sum_count(self._day.target_sum, self._day.target_count, target)
        stats.timesteps += 1
        self.completed_timesteps += 1

    def _finalize_day(self) -> None:
        if self._day is None:
            return
        stats = self._year(self._day.date_key[0])
        if tuple(sorted(self._day.hours)) != tuple(sorted(self.expected_hours)):
            stats.incomplete_days += 1
            self._day = None
            return

        stats.complete_days += 1
        for source_stats, daily_sum, daily_count in (
            (stats.prediction, self._day.prediction_sum, self._day.prediction_count),
            (stats.target, self._day.target_sum, self._day.target_count),
        ):
            complete = daily_count == len(self.expected_hours)
            daily = np.where(complete, daily_sum, np.nan)
            _update_sum_count(source_stats.daily_sum, source_stats.daily_count, daily)
            wet = np.where(daily >= self.wet_day_threshold_mm, daily, np.nan)
            _update_sum_count(source_stats.wet_day_sum, source_stats.wet_day_count, wet)
            _update_max(source_stats.rx1day, daily)
        self._day = None

    def finalize(self) -> None:
        if not self._finalized:
            self._finalize_day()
            self._finalized = True

    def write_netcdf(
        self,
        path: str | os.PathLike,
        latitude: np.ndarray | None = None,
        longitude: np.ndarray | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> Path:
        """Atomically write a compressed, mergeable chunk file."""
        self.finalize()
        if not self._years:
            raise ValueError("Cannot write an empty ClimateAccumulator")
        final_path = Path(path)
        tmp_path = Path(f"{final_path}.tmp")
        final_path.parent.mkdir(parents=True, exist_ok=True)
        if final_path.exists():
            raise FileExistsError(f"Climate output already exists: {final_path}")
        if tmp_path.exists():
            raise FileExistsError(
                f"Incomplete climate output already exists: {tmp_path}; inspect or move it first"
            )

        years = self.years
        with nc.Dataset(tmp_path, "w") as ds:
            ds.createDimension("year", len(years))
            ds.createDimension("y", self.shape[0])
            ds.createDimension("x", self.shape[1])
            ds.createVariable("year", "i4", ("year",))[:] = years
            if latitude is not None:
                lat = np.asarray(latitude)
                if lat.shape != self.shape:
                    raise ValueError(f"Latitude shape {lat.shape} does not match {self.shape}")
                ds.createVariable("latitude", "f4", ("y", "x"), zlib=True)[:] = lat
            if longitude is not None:
                lon = np.asarray(longitude)
                if lon.shape != self.shape:
                    raise ValueError(f"Longitude shape {lon.shape} does not match {self.shape}")
                ds.createVariable("longitude", "f4", ("y", "x"), zlib=True)[:] = lon

            for source_name in ("prediction", "target"):
                for field_name, dtype in (
                    ("step_sum", "f8"),
                    ("step_count", "u4"),
                    ("daily_sum", "f8"),
                    ("daily_count", "u4"),
                    ("wet_day_sum", "f8"),
                    ("wet_day_count", "u4"),
                    ("rx1day", "f4"),
                    ("rx3h", "f4"),
                ):
                    values = np.stack(
                        [getattr(getattr(self._years[year], source_name), field_name) for year in years]
                    )
                    ds.createVariable(
                        f"{source_name}_{field_name}",
                        dtype,
                        ("year", "y", "x"),
                        zlib=True,
                        complevel=4,
                        fill_value=np.nan if dtype.startswith("f") else None,
                    )[:] = values

            for name in ("complete_days", "incomplete_days", "timesteps"):
                ds.createVariable(name, "u4", ("year",))[:] = [
                    getattr(self._years[year], name) for year in years
                ]

            ds.setncattr("complete", 1)
            ds.setncattr("completed_timesteps", self.completed_timesteps)
            ds.setncattr("precipitation_units", "mm per 3 hours")
            ds.setncattr("wet_day_threshold_mm", self.wet_day_threshold_mm)
            ds.setncattr("expected_utc_hours", json.dumps(self.expected_hours))
            ds.setncattr("daily_completeness_policy", "exact expected UTC hours; incomplete days discarded")
            for key, value in (metadata or {}).items():
                if value is None:
                    continue
                if isinstance(value, (str, int, float, np.number)):
                    ds.setncattr(str(key), value)
                else:
                    ds.setncattr(str(key), json.dumps(value, sort_keys=True))
        os.replace(tmp_path, final_path)
        return final_path
