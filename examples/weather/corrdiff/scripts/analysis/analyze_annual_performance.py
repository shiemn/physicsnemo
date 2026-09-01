#!/usr/bin/env python3
"""Stream annual CorrDiff NetCDFs into error time series and summary plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np
import pandas as pd

if __package__:
    from .common import read_times as _read_times
else:
    from common import read_times as _read_times


METRICS = ("rmse", "mae", "crps", "bias")
COLORS = ("#4c566a", "#0072b2", "#d55e00", "#009e73", "#cc79a7")


def parse_model(value: str) -> tuple[str, Path]:
    try:
        label, path = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("models must use LABEL=/path/predictions.nc") from exc
    if not label or not path:
        raise argparse.ArgumentTypeError("models must use LABEL=/path/predictions.nc")
    return label, Path(path)


def empirical_crps(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Return spatial-mean empirical ensemble CRPS for each time.

    ``prediction`` has shape ``(ensemble, time, y, x)`` and ``truth`` has
    shape ``(time, y, x)``. NaNs are ignored in the spatial means.
    """

    first = np.nanmean(np.abs(prediction - truth[None]), axis=(0, 2, 3))
    ensemble = prediction.shape[0]
    pairwise = np.zeros(prediction.shape[1], dtype=np.float64)
    for i in range(ensemble):
        for j in range(ensemble):
            pairwise += np.nanmean(
                np.abs(prediction[i] - prediction[j]), axis=(1, 2)
            )
    return first - pairwise / (2.0 * ensemble * ensemble)


def chunk_metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, np.ndarray]:
    ensemble_mean = np.nanmean(prediction, axis=0)
    error = ensemble_mean - truth
    return {
        "rmse": np.sqrt(np.nanmean(error * error, axis=(1, 2))),
        "mae": np.nanmean(np.abs(error), axis=(1, 2)),
        "bias": np.nanmean(error, axis=(1, 2)),
        "crps": empirical_crps(prediction, truth),
    }


def read_times(ds: Dataset) -> pd.DatetimeIndex:
    return _read_times(ds, utc=False)


def analyze_file(
    label: str, path: Path, channel: str, chunk_size: int
) -> tuple[pd.DataFrame, dict[str, float], dict]:
    if not path.is_file():
        raise FileNotFoundError(path)

    rows: list[pd.DataFrame] = []
    squared_error_sum = 0.0
    valid_error_count = 0
    with Dataset(path) as ds:
        if channel not in ds.groups["truth"].variables:
            raise KeyError(f"{channel!r} is absent from truth group in {path}")
        if channel not in ds.groups["prediction"].variables:
            raise KeyError(f"{channel!r} is absent from prediction group in {path}")
        truth_var = ds.groups["truth"].variables[channel]
        prediction_var = ds.groups["prediction"].variables[channel]
        times = read_times(ds)
        if prediction_var.shape[1] != len(times) or truth_var.shape[0] != len(times):
            raise ValueError(f"inconsistent time dimensions in {path}")

        for start in range(0, len(times), chunk_size):
            stop = min(start + chunk_size, len(times))
            truth = np.asarray(truth_var[start:stop], dtype=np.float32)
            prediction = np.asarray(prediction_var[:, start:stop], dtype=np.float32)
            metrics = chunk_metrics(prediction, truth)
            error = np.nanmean(prediction, axis=0) - truth
            finite = np.isfinite(error)
            squared_error_sum += float(np.sum(np.square(error[finite]), dtype=np.float64))
            valid_error_count += int(finite.sum())
            rows.append(
                pd.DataFrame(
                    {
                        "time": times[start:stop],
                        "model": label,
                        **metrics,
                    }
                )
            )
            print(f"{label}: {stop:,}/{len(times):,}", flush=True)

        metadata = {
            "path": str(path),
            "timesteps": len(times),
            "ensemble_members": prediction_var.shape[0],
            "shape": list(prediction_var.shape),
            "first_time": times[0].isoformat(),
            "last_time": times[-1].isoformat(),
        }

    frame = pd.concat(rows, ignore_index=True)
    summary = {
        "model": label,
        "timesteps": len(frame),
        "aggregate_rmse": np.sqrt(squared_error_sum / valid_error_count),
        "mean_timestep_rmse": frame["rmse"].mean(),
        "mae": frame["mae"].mean(),
        "bias": frame["bias"].mean(),
        "crps": frame["crps"].mean(),
        "max_timestep_rmse": frame["rmse"].max(),
    }
    return frame, summary, metadata


def rolling_metrics(frame: pd.DataFrame, days: int) -> pd.DataFrame:
    outputs = []
    for label, group in frame.groupby("model", sort=False):
        ordered = group.sort_values("time").set_index("time")
        values = ordered[list(METRICS)].rolling(
            f"{days}D", center=True, min_periods=24
        ).mean()
        values["model"] = label
        outputs.append(values.reset_index())
    return pd.concat(outputs, ignore_index=True)


def monthly_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["month"] = result["time"].dt.to_period("M").astype(str)
    return (
        result.groupby(["month", "model"], sort=True)[list(METRICS)]
        .mean()
        .reset_index()
    )


def align_frames(frames: list[pd.DataFrame], mode: str) -> list[pd.DataFrame]:
    if mode == "native":
        return frames
    if mode != "intersection":
        raise ValueError(f"unknown time alignment mode: {mode}")
    common_times = set(frames[0]["time"])
    for frame in frames[1:]:
        common_times.intersection_update(frame["time"])
    if not common_times:
        raise ValueError("models have no timestamps in common")
    return [
        frame[frame["time"].isin(common_times)].sort_values("time").reset_index(drop=True)
        for frame in frames
    ]


def summarize_frame(frame: pd.DataFrame) -> dict[str, float]:
    labels = frame["model"].unique()
    if len(labels) != 1:
        raise ValueError("summary frames must contain exactly one model")
    return {
        "model": labels[0],
        "timesteps": len(frame),
        "aggregate_rmse": np.sqrt(np.mean(np.square(frame["rmse"]))),
        "mean_timestep_rmse": frame["rmse"].mean(),
        "mae": frame["mae"].mean(),
        "bias": frame["bias"].mean(),
        "crps": frame["crps"].mean(),
        "max_timestep_rmse": frame["rmse"].max(),
    }


def plot_metrics(
    raw: pd.DataFrame,
    rolling: pd.DataFrame,
    title: str,
    units: str,
    rolling_days: int,
    output_dir: Path,
) -> None:
    labels = list(raw["model"].drop_duplicates())
    color_by_label = {label: COLORS[i % len(COLORS)] for i, label in enumerate(labels)}
    fig, axes = plt.subplots(4, 1, figsize=(16, 13), sharex=True)
    fig.patch.set_facecolor("white")
    for axis in axes:
        axis.set_facecolor("white")
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.07, top=0.86, hspace=0.06)
    wrapped_title = textwrap.fill(title, width=92)
    fig.suptitle(
        f"{wrapped_title}\nFaint: hourly errors; solid: centered {rolling_days}-day mean",
        fontsize=14,
        y=0.98,
    )

    for axis, metric in zip(axes, METRICS):
        for label in labels:
            color = color_by_label[label]
            data = raw[raw["model"] == label].sort_values("time").copy()
            trend = rolling[rolling["model"] == label].sort_values("time").copy()
            for _, segment in data.groupby(
                data["time"].diff().gt(pd.Timedelta(hours=1.5)).cumsum()
            ):
                axis.plot(
                    segment["time"], segment[metric], color=color, alpha=0.07, linewidth=0.45
                )
            first_segment = True
            for _, segment in trend.groupby(
                trend["time"].diff().gt(pd.Timedelta(hours=1.5)).cumsum()
            ):
                axis.plot(
                    segment["time"],
                    segment[metric],
                    color=color,
                    linewidth=1.8,
                    label=label if first_segment else None,
                )
                first_segment = False
        if metric == "bias":
            axis.axhline(0.0, color="0.35", linewidth=0.7)
        axis.set_ylabel(f"{metric.upper()} ({units})")
        axis.grid(True, color="0.85", alpha=0.55, linewidth=0.5)

    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=len(labels))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[-1].set_xlabel(str(raw["time"].dt.year.mode().iat[0]))
    for label in ("annual_performance_over_time.png", "annual_performance_over_time.pdf"):
        fig.savefig(
            output_dir / label,
            dpi=180 if label.endswith(".png") else None,
            facecolor="white",
        )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", type=parse_model, required=True)
    parser.add_argument("--channel", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--units", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rolling-days", type=int, default=7)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument(
        "--time-alignment", choices=("native", "intersection"), default="native"
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    metadata = {
        "channel": args.channel,
        "time_alignment": args.time_alignment,
        "models": {},
    }
    for label, path in args.model:
        frame, _, model_metadata = analyze_file(
            label, path, args.channel, args.chunk_size
        )
        frames.append(frame)
        metadata["models"][label] = model_metadata

    frames = align_frames(frames, args.time_alignment)
    for frame in frames:
        metadata["models"][frame["model"].iat[0]]["analyzed_timesteps"] = len(frame)
    raw = pd.concat(frames, ignore_index=True)
    rolling = rolling_metrics(raw, args.rolling_days)
    monthly = monthly_metrics(raw)
    summary = pd.DataFrame([summarize_frame(frame) for frame in frames])
    raw.to_csv(args.output_dir / "errors_hourly.csv", index=False, float_format="%.7g")
    rolling.to_csv(
        args.output_dir / "errors_7day_rolling.csv", index=False, float_format="%.7g"
    )
    monthly.to_csv(
        args.output_dir / "monthly_mean_errors.csv", index=False, float_format="%.7g"
    )
    summary.to_csv(
        args.output_dir / "annual_metrics.csv", index=False, float_format="%.7g"
    )
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    plot_metrics(raw, rolling, args.title, args.units, args.rolling_days, args.output_dir)
    print(summary.to_string(index=False), flush=True)
    print(f"Wrote analysis to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
