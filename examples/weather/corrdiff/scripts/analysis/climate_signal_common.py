"""Shared climate-signal loading and statistical calculations."""

from __future__ import annotations

from pathlib import Path

import netCDF4 as nc
import numpy as np


METRICS = {
    "mean_daily": ("daily_sum", "daily_count", 1.0, "mm day-1"),
    "sdii": ("wet_day_sum", "wet_day_count", 1.0, "mm wet-day-1"),
    "wet_day_fraction": ("wet_day_count", "daily_count", 100.0, "%"),
    "rx1day": ("rx1day", None, 1.0, "mm day-1"),
    "rx3h": ("rx3h", None, 1.0, "mm 3h-1"),
}

AGGREGATE_SUFFIXES = (
    {spec[0] for spec in METRICS.values()}
    | {spec[1] for spec in METRICS.values() if spec[1]}
    | {"step_sum", "step_count"}
)


def _plain(variable: nc.Variable) -> np.ndarray:
    values = variable[:]
    if np.ma.isMaskedArray(values):
        values = values.filled(np.nan)
    return np.asarray(values, dtype=np.float64)


def _load_epoch(paths: list[Path]) -> dict:
    years: list[np.ndarray] = []
    fields: dict[str, list[np.ndarray]] = {}
    latitude = longitude = None

    for path in paths:
        with nc.Dataset(path) as ds:
            if int(ds.getncattr("complete")) != 1:
                raise ValueError(f"Incomplete chunk: {path}")
            current_lat = _plain(ds.variables["latitude"])
            current_lon = _plain(ds.variables["longitude"])
            if latitude is None:
                latitude, longitude = current_lat, current_lon
            elif not (
                np.allclose(latitude, current_lat, equal_nan=True)
                and np.allclose(longitude, current_lon, equal_nan=True)
            ):
                raise ValueError(f"Grid differs in {path}")
            years.append(_plain(ds.variables["year"]).astype(int))
            for source in ("prediction", "target"):
                for suffix in AGGREGATE_SUFFIXES:
                    name = f"{source}_{suffix}"
                    fields.setdefault(name, []).append(_plain(ds.variables[name]))

    all_years = np.concatenate(years)
    order = np.argsort(all_years)
    all_years = all_years[order]
    if len(np.unique(all_years)) != len(all_years):
        raise ValueError(f"Duplicate years in chunks: {all_years.tolist()}")
    result = {
        "years": all_years,
        "latitude": latitude,
        "longitude": longitude,
    }
    for name, chunks in fields.items():
        result[name] = np.concatenate(chunks, axis=0)[order]
    return result


def _annual_and_climatology(epoch: dict, source: str, metric: str):
    numerator, denominator, scale, _ = METRICS[metric]
    num = epoch[f"{source}_{numerator}"]
    if denominator is None:
        annual = num * scale
        climatology = np.nanmean(annual, axis=0)
    else:
        den = epoch[f"{source}_{denominator}"]
        annual = np.divide(
            num,
            den,
            out=np.full_like(num, np.nan),
            where=den > 0,
        ) * scale
        climatology = np.divide(
            np.nansum(num, axis=0),
            np.nansum(den, axis=0),
            out=np.full(num.shape[1:], np.nan),
            where=np.nansum(den, axis=0) > 0,
        ) * scale
    annual_domain = np.nanmean(annual, axis=(1, 2))
    return annual_domain, climatology


def _ci(values: np.ndarray) -> list[float]:
    return [float(x) for x in np.percentile(values, [2.5, 97.5])]


def _bootstrap(
    hist_prediction: np.ndarray,
    hist_target: np.ndarray,
    future_prediction: np.ndarray,
    future_target: np.ndarray,
    samples: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    hist_idx = rng.integers(0, len(hist_target), size=(samples, len(hist_target)))
    future_idx = rng.integers(
        0, len(future_target), size=(samples, len(future_target))
    )
    target_change = future_target[future_idx].mean(axis=1) - hist_target[
        hist_idx
    ].mean(axis=1)
    prediction_change = future_prediction[future_idx].mean(axis=1) - hist_prediction[
        hist_idx
    ].mean(axis=1)
    error = prediction_change - target_change
    return {
        "target_change_95ci": _ci(target_change),
        "prediction_change_95ci": _ci(prediction_change),
        "signal_error_95ci": _ci(error),
        "target_change_positive_probability": float(np.mean(target_change > 0)),
        "prediction_change_positive_probability": float(
            np.mean(prediction_change > 0)
        ),
    }


def _spatial_summary(prediction: np.ndarray, target: np.ndarray) -> dict:
    mask = np.isfinite(prediction) & np.isfinite(target)
    prediction = prediction[mask]
    target = target[mask]
    error = prediction - target
    corr = np.corrcoef(prediction, target)[0, 1]
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "mean_error": float(np.mean(error)),
        "pattern_correlation": float(corr),
        "sign_agreement_fraction": float(np.mean(np.sign(prediction) == np.sign(target))),
        "target_change_quantiles_05_50_95": [
            float(x) for x in np.percentile(target, [5, 50, 95])
        ],
        "prediction_change_quantiles_05_50_95": [
            float(x) for x in np.percentile(prediction, [5, 50, 95])
        ],
        "error_quantiles_05_50_95": [
            float(x) for x in np.percentile(error, [5, 50, 95])
        ],
    }
