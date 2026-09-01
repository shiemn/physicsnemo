"""Shared utilities for CorrDiff process-oriented analysis scripts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from netCDF4 import Dataset, num2date
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter


@dataclass(frozen=True)
class ModelSource:
    """Labelled prediction artifact supplied on an analysis command line."""

    label: str
    path: Path


def parse_model(value: str) -> ModelSource:
    """Parse ``LABEL=/path/predictions.nc`` into a model source."""

    try:
        label, path = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "models must use LABEL=/path/predictions.nc"
        ) from exc
    if not label or not path:
        raise argparse.ArgumentTypeError(
            "models must use LABEL=/path/predictions.nc"
        )
    return ModelSource(label, Path(path))


def read_times(ds: Dataset, *, utc: bool = True) -> pd.DatetimeIndex:
    """Decode a NetCDF time coordinate into a pandas datetime index."""

    variable = ds.variables["time"]
    decoded = num2date(
        variable[:],
        units=variable.units,
        calendar=getattr(variable, "calendar", "standard"),
        only_use_cftime_datetimes=False,
    )
    return pd.DatetimeIndex(
        pd.to_datetime([str(item) for item in decoded], utc=utc)
    )


def common_times(
    models: list[ModelSource],
) -> tuple[pd.DatetimeIndex, dict[str, pd.DatetimeIndex]]:
    """Return sorted timestamps common to every model and each native axis."""

    if not models:
        raise ValueError("at least one model source is required")
    native: dict[str, pd.DatetimeIndex] = {}
    common: set[pd.Timestamp] | None = None
    for model in models:
        with Dataset(model.path) as ds:
            times = read_times(ds)
        native[model.label] = times
        common = set(times) if common is None else common.intersection(times)
    assert common is not None
    return pd.DatetimeIndex(sorted(common)), native


def time_indices(
    native: pd.DatetimeIndex, selected: pd.DatetimeIndex
) -> np.ndarray:
    """Locate selected timestamps on a model's native time axis."""

    indices = native.get_indexer(selected)
    if np.any(indices < 0):
        raise ValueError("selected timestamps are not present in a model source")
    return indices


def member_mean_fss(
    prediction: np.ndarray,
    truth: np.ndarray,
    *,
    threshold: float,
    scale_px: int,
) -> float:
    """Return deterministic fractions skill score averaged over members."""

    truth_fraction = uniform_filter(
        (truth >= threshold).astype(np.float32), size=scale_px, mode="constant"
    )
    scores: list[float] = []
    for member in prediction:
        predicted_fraction = uniform_filter(
            (member >= threshold).astype(np.float32),
            size=scale_px,
            mode="constant",
        )
        denominator = float(
            np.mean(predicted_fraction**2) + np.mean(truth_fraction**2)
        )
        if denominator > 1e-12:
            scores.append(
                1.0
                - float(np.mean((predicted_fraction - truth_fraction) ** 2))
                / denominator
            )
    return float(np.mean(scores)) if scores else np.nan


def season_for_month(month: int) -> str:
    """Map a calendar month to its meteorological season."""

    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"
