#!/usr/bin/env python3
"""Test whether the preferred temporal lag depends on system timescale."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from netCDF4 import Dataset
import numpy as np
import pandas as pd
from scipy.ndimage import shift as spatial_shift
from scipy.ndimage import uniform_filter
from scipy.stats import spearmanr

if __package__:
    from .common import read_times as _read_times
else:
    from common import read_times as _read_times


CHANNEL = "maximum_radar_reflectivity"
MODEL_BASELINE = "Baseline"
MODEL_1H = "Symmetric 1 h"
MODEL_3H = "Symmetric 3 h"
GRID_SPACING_KM = 3.0
SMOOTHING_SCALE_PX = 9
MIN_ACTIVE_AREA_KM2 = 5_000.0
PRIMARY_FSS_THRESHOLD_DBZ = 30.0
PRIMARY_FSS_SCALE_KM = 27.0
STORM_COLORS = {"Chanthu": "#7b3294", "Lupit": "#008837", "In-fa": "#d95f02"}


def read_times(ds: Dataset) -> pd.DatetimeIndex:
    return _read_times(ds)


def field_correlation(first: np.ndarray, second: np.ndarray) -> float:
    first_flat = first.ravel().astype(np.float64)
    second_flat = second.ravel().astype(np.float64)
    first_flat -= first_flat.mean()
    second_flat -= second_flat.mean()
    denominator = np.sqrt(np.sum(first_flat**2) * np.sum(second_flat**2))
    if denominator <= 1e-12:
        return np.nan
    return float(np.sum(first_flat * second_flat) / denominator)


def normalized_change(first: np.ndarray, second: np.ndarray) -> float:
    numerator = np.sqrt(np.mean((second - first) ** 2, dtype=np.float64))
    denominator = np.sqrt(
        0.5
        * (
            np.mean(first**2, dtype=np.float64)
            + np.mean(second**2, dtype=np.float64)
        )
    )
    if denominator <= 1e-12:
        return np.nan
    return float(numerator / denominator)


def phase_motion_and_deformation(
    previous: np.ndarray, current: np.ndarray
) -> tuple[float, float, float]:
    """Return motion (km), aligned change, and phase-correlation response."""
    window = cv2.createHanningWindow(
        (previous.shape[1], previous.shape[0]), cv2.CV_32F
    )
    (dx, dy), response = cv2.phaseCorrelate(
        previous.astype(np.float32), current.astype(np.float32), window
    )
    if response < 0.02 or abs(dx) > 100 or abs(dy) > 100:
        return np.nan, np.nan, float(response)
    aligned = spatial_shift(
        previous,
        shift=(dy, dx),
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    return (
        float(np.hypot(dx, dy) * GRID_SPACING_KM),
        normalized_change(aligned, current),
        float(response),
    )


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    if not all(np.isfinite([lat1, lon1, lat2, lon2])):
        return np.nan
    lat1_rad, lat2_rad = np.deg2rad([lat1, lat2])
    dlat = lat2_rad - lat1_rad
    dlon = np.deg2rad(lon2 - lon1)
    value = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    )
    return float(6371.0 * 2.0 * np.arcsin(np.sqrt(np.clip(value, 0.0, 1.0))))


def compute_regimes(dataset_path: Path) -> pd.DataFrame:
    rows: list[dict] = []
    with Dataset(dataset_path) as ds:
        times = read_times(ds)
        storm_names = np.asarray(ds.variables["storm"][:], dtype=str)
        storm_indices = np.asarray(ds.variables["storm_index"][:], dtype=int)
        truth = ds.variables[CHANNEL + "_truth"]
        for storm_index, storm_name in enumerate(storm_names):
            indices = np.flatnonzero(storm_indices == storm_index)
            indices = indices[np.argsort(times[indices].asi8)]
            fields: dict[pd.Timestamp, np.ndarray] = {}
            for source_index in indices:
                timestamp = times[source_index]
                field = np.asarray(truth[source_index], dtype=np.float32)
                smoothed = uniform_filter(
                    field, size=SMOOTHING_SCALE_PX, mode="nearest"
                ).astype(np.float32)
                row = {
                    "storm": storm_name,
                    "time": timestamp,
                    "active_area_ge_20dbz_km2": float(
                        np.sum(field >= 20.0) * GRID_SPACING_KM**2
                    ),
                }
                for lag_hours in (1, 3):
                    previous_time = timestamp - pd.Timedelta(hours=lag_hours)
                    previous = fields.get(previous_time)
                    if previous is None:
                        row[f"radar_corr_{lag_hours}h"] = np.nan
                        row[f"radar_change_{lag_hours}h"] = np.nan
                    else:
                        row[f"radar_corr_{lag_hours}h"] = field_correlation(
                            previous, smoothed
                        )
                        row[f"radar_change_{lag_hours}h"] = normalized_change(
                            previous, smoothed
                        )
                previous = fields.get(timestamp - pd.Timedelta(hours=1))
                if previous is None:
                    row.update(
                        {
                            "radar_motion_kmh": np.nan,
                            "radar_aligned_change_1h": np.nan,
                            "phase_response": np.nan,
                        }
                    )
                else:
                    motion, deformation, response = phase_motion_and_deformation(
                        previous, smoothed
                    )
                    row.update(
                        {
                            "radar_motion_kmh": motion,
                            "radar_aligned_change_1h": deformation,
                            "phase_response": response,
                        }
                    )
                fields[timestamp] = smoothed
                for old_time in list(fields):
                    if timestamp - old_time > pd.Timedelta(hours=3):
                        del fields[old_time]
                rows.append(row)
            print(f"regimes: {storm_name} ({len(indices)} hours)", flush=True)
    return pd.DataFrame(rows)


def load_track_speed(analysis_root: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for storm_dir in sorted(path for path in analysis_root.iterdir() if path.is_dir()):
        track_path = storm_dir / "interpolated_jma_track.csv"
        if not track_path.is_file():
            continue
        track = pd.read_csv(track_path)
        track["time"] = pd.to_datetime(track["time"], utc=True)
        speed = np.full(len(track), np.nan)
        for index in range(1, len(track)):
            elapsed_hours = (
                track.loc[index, "time"] - track.loc[index - 1, "time"]
            ).total_seconds() / 3600.0
            if elapsed_hours <= 1.5:
                distance = haversine_km(
                    track.loc[index - 1, "center_latitude"],
                    track.loc[index - 1, "center_longitude"],
                    track.loc[index, "center_latitude"],
                    track.loc[index, "center_longitude"],
                )
                speed[index] = distance / elapsed_hours
        rows.append(
            pd.DataFrame(
                {
                    "storm": storm_dir.name.replace("in-fa", "In-fa").title().replace("In-Fa", "In-fa"),
                    "time": track["time"],
                    "jma_speed_kmh": speed,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def load_skill(analysis_root: Path) -> pd.DataFrame:
    frame_parts, fss_parts = [], []
    for storm_dir in sorted(path for path in analysis_root.iterdir() if path.is_dir()):
        frame_parts.append(pd.read_csv(storm_dir / "frame_metrics.csv"))
        fss = pd.read_csv(storm_dir / "fss.csv")
        fss_parts.append(
            fss[
                (fss["threshold_dbz"] == PRIMARY_FSS_THRESHOLD_DBZ)
                & (fss["scale_km"] == PRIMARY_FSS_SCALE_KM)
            ]
        )
    frames = pd.concat(frame_parts, ignore_index=True)
    fss = pd.concat(fss_parts, ignore_index=True)
    for frame in (frames, fss):
        frame["time"] = pd.to_datetime(frame["time"], utc=True)
    rmse = frames.pivot(index=["storm", "time"], columns="model", values="rmse_dbz")
    fss_pivot = fss.pivot(index=["storm", "time"], columns="model", values="fss")
    required = [MODEL_BASELINE, MODEL_1H, MODEL_3H]
    if any(column not in rmse or column not in fss_pivot for column in required):
        raise ValueError("expected Baseline, Symmetric 1 h, and Symmetric 3 h")
    output = pd.DataFrame(index=rmse.index)
    output["rmse_gain_1h"] = rmse[MODEL_BASELINE] - rmse[MODEL_1H]
    output["rmse_gain_3h"] = rmse[MODEL_BASELINE] - rmse[MODEL_3H]
    output["rmse_3h_minus_1h"] = rmse[MODEL_1H] - rmse[MODEL_3H]
    output["fss_gain_1h"] = fss_pivot[MODEL_1H] - fss_pivot[MODEL_BASELINE]
    output["fss_gain_3h"] = fss_pivot[MODEL_3H] - fss_pivot[MODEL_BASELINE]
    output["fss_3h_minus_1h"] = fss_pivot[MODEL_3H] - fss_pivot[MODEL_1H]
    return output.reset_index()


def aggregate_storm_days(frame: pd.DataFrame) -> pd.DataFrame:
    active = frame[frame["active_area_ge_20dbz_km2"] >= MIN_ACTIVE_AREA_KM2].copy()
    active["date"] = active["time"].dt.floor("D")
    numeric = active.select_dtypes(include=[np.number]).columns.tolist()
    daily = active.groupby(["storm", "date"], as_index=False)[numeric].mean()
    counts = active.groupby(["storm", "date"], as_index=False).size().rename(columns={"size": "valid_hours"})
    daily = daily.merge(counts, on=["storm", "date"])
    return daily[daily["valid_hours"] >= 6].reset_index(drop=True)


def quantile_edges(values: pd.Series, bins: int = 4) -> np.ndarray:
    finite = values[np.isfinite(values)]
    edges = np.quantile(finite, np.linspace(0.0, 1.0, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    return np.unique(edges)


def block_bootstrap_bins(
    daily: pd.DataFrame,
    predictor: str,
    outcome: str,
    bins: int = 4,
    replicates: int = 2_000,
    seed: int = 20210714,
) -> pd.DataFrame:
    valid = daily[np.isfinite(daily[predictor]) & np.isfinite(daily[outcome])].copy()
    edges = quantile_edges(valid[predictor], bins)
    valid["bin"] = pd.cut(valid[predictor], edges, labels=False, include_lowest=True)
    observed = valid.groupby("bin").agg(
        predictor_mean=(predictor, "mean"),
        outcome_mean=(outcome, "mean"),
        storm_days=(outcome, "size"),
    )
    rng = np.random.default_rng(seed)
    bootstrap: dict[int, list[float]] = {int(index): [] for index in observed.index}
    storm_groups = [group for _, group in valid.groupby("storm")]
    for _ in range(replicates):
        sampled = pd.concat(
            [group.iloc[rng.integers(0, len(group), len(group))] for group in storm_groups],
            ignore_index=True,
        )
        sampled["bin"] = pd.cut(sampled[predictor], edges, labels=False, include_lowest=True)
        means = sampled.groupby("bin")[outcome].mean()
        for index in bootstrap:
            if index in means:
                bootstrap[index].append(float(means[index]))
    result = observed.reset_index()
    result["ci_low"] = [np.quantile(bootstrap[int(index)], 0.025) for index in result["bin"]]
    result["ci_high"] = [np.quantile(bootstrap[int(index)], 0.975) for index in result["bin"]]
    result["predictor"] = predictor
    result["outcome"] = outcome
    return result


def within_storm_spearman_test(
    daily: pd.DataFrame,
    predictor: str,
    outcome: str,
    permutations: int = 5_000,
    seed: int = 20210714,
) -> dict:
    valid = (
        daily[np.isfinite(daily[predictor]) & np.isfinite(daily[outcome])]
        .sort_values(["storm", "date"])
        .reset_index(drop=True)
        .copy()
    )
    for column in (predictor, outcome):
        valid[column + "_within"] = valid[column] - valid.groupby("storm")[column].transform("mean")
    observed = float(spearmanr(valid[predictor + "_within"], valid[outcome + "_within"]).statistic)
    rng = np.random.default_rng(seed)
    null = np.empty(permutations)
    groups = [group.index.to_numpy() for _, group in valid.groupby("storm", sort=False)]
    original = valid[outcome + "_within"].to_numpy()
    x = valid[predictor + "_within"].to_numpy()
    for permutation in range(permutations):
        shuffled = original.copy()
        for positions in groups:
            if len(positions) > 1:
                offset = int(rng.integers(0, len(positions)))
                shuffled[positions] = np.roll(shuffled[positions], offset)
        null[permutation] = spearmanr(x, shuffled).statistic
    p_value = float((1 + np.sum(np.abs(null) >= abs(observed))) / (permutations + 1))
    return {
        "predictor": predictor,
        "outcome": outcome,
        "storm_days": len(valid),
        "within_storm_spearman": observed,
        "circular_shift_p_two_sided": p_value,
    }


def build_bin_summaries(daily: pd.DataFrame) -> pd.DataFrame:
    predictors = [
        "radar_corr_3h",
        "radar_change_3h",
        "radar_aligned_change_1h",
        "jma_speed_kmh",
    ]
    outcomes = [
        "rmse_3h_minus_1h",
        "fss_3h_minus_1h",
        "rmse_gain_1h",
        "rmse_gain_3h",
        "fss_gain_1h",
        "fss_gain_3h",
    ]
    outputs = []
    for predictor in predictors:
        for outcome in outcomes:
            outputs.append(block_bootstrap_bins(daily, predictor, outcome))
    return pd.concat(outputs, ignore_index=True)


def per_storm_spearman_tests(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for predictor in (
        "radar_corr_3h",
        "radar_change_3h",
        "radar_aligned_change_1h",
        "jma_speed_kmh",
    ):
        for outcome in ("rmse_3h_minus_1h", "fss_3h_minus_1h"):
            for storm, group in daily.groupby("storm"):
                valid = group[np.isfinite(group[predictor]) & np.isfinite(group[outcome])]
                statistic, p_value = spearmanr(valid[predictor], valid[outcome])
                rows.append(
                    {
                        "storm": storm,
                        "predictor": predictor,
                        "outcome": outcome,
                        "storm_days": len(valid),
                        "spearman": statistic,
                        "p_two_sided": p_value,
                    }
                )
    return pd.DataFrame(rows)


def plot_timescale_test(daily: pd.DataFrame, bins: pd.DataFrame, output_dir: Path) -> None:
    predictors = [
        ("radar_corr_3h", "3 h radar persistence"),
        ("radar_change_3h", "3 h normalized change"),
        ("jma_speed_kmh", "JMA centre speed (km h⁻¹)"),
    ]
    outcomes = [
        ("rmse_3h_minus_1h", "RMSE advantage of ±3 h (dBZ)"),
        ("fss_3h_minus_1h", "FSS advantage of ±3 h"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for row, (outcome, ylabel) in enumerate(outcomes):
        for column, (predictor, xlabel) in enumerate(predictors):
            axis = axes[row, column]
            for storm, group in daily.groupby("storm"):
                axis.scatter(
                    group[predictor],
                    group[outcome],
                    s=24,
                    alpha=0.48,
                    color=STORM_COLORS.get(storm, "0.4"),
                    label=storm if row == 0 and column == 0 else None,
                )
            summary = bins[(bins["predictor"] == predictor) & (bins["outcome"] == outcome)]
            axis.errorbar(
                summary["predictor_mean"],
                summary["outcome_mean"],
                yerr=np.vstack(
                    [
                        summary["outcome_mean"] - summary["ci_low"],
                        summary["ci_high"] - summary["outcome_mean"],
                    ]
                ),
                color="black",
                marker="o",
                markersize=5,
                linewidth=1.5,
                capsize=3,
                label="Storm-day bin mean" if row == 0 and column == 0 else None,
            )
            axis.axhline(0.0, color="0.3", linewidth=0.8)
            axis.set_xlabel(xlabel)
            axis.set_ylabel(ylabel if column == 0 else "")
            axis.grid(True, color="0.88", linewidth=0.6)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle(
        "Does preferred temporal context depend on system timescale?\n"
        "Points are storm-days; black intervals are storm-stratified bootstrap 95% CIs",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_dir / "timescale_skill_test.png", dpi=190)
    fig.savefig(output_dir / "timescale_skill_test.pdf")
    plt.close(fig)


def plot_phase_diagram(daily: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    outcomes = [
        ("rmse_3h_minus_1h", "RMSE advantage of ±3 h (dBZ)"),
        ("fss_3h_minus_1h", "FSS advantage of ±3 h"),
    ]
    for axis, (outcome, title) in zip(axes, outcomes):
        valid = daily[
            np.isfinite(daily["radar_corr_3h"])
            & np.isfinite(daily["radar_change_3h"])
            & np.isfinite(daily[outcome])
        ]
        limit = max(float(np.nanmax(np.abs(valid[outcome]))), 1e-6)
        scatter = axis.scatter(
            valid["radar_corr_3h"],
            valid["radar_change_3h"],
            c=valid[outcome],
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
            s=65,
            edgecolor="white",
            linewidth=0.5,
        )
        axis.set(
            xlabel="3 h radar persistence",
            ylabel="3 h normalized structural change",
            title=title,
        )
        axis.grid(True, color="0.9", linewidth=0.6)
        fig.colorbar(scatter, ax=axis, shrink=0.86)
    fig.suptitle("Persistence–evolution phase diagram (one point per storm-day)")
    fig.tight_layout()
    fig.savefig(output_dir / "persistence_evolution_phase.png", dpi=190)
    plt.close(fig)


def plot_gains_over_baseline(bins: pd.DataFrame, output_dir: Path) -> None:
    predictors = [
        ("radar_corr_3h", "3 h radar persistence"),
        ("radar_change_3h", "3 h normalized change"),
        ("jma_speed_kmh", "JMA centre speed (km h⁻¹)"),
    ]
    metrics = [
        ("rmse", "RMSE gain over baseline (dBZ)"),
        ("fss", "FSS gain over baseline"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    for row, (metric, ylabel) in enumerate(metrics):
        for column, (predictor, xlabel) in enumerate(predictors):
            axis = axes[row, column]
            for lag, color in (("1h", "#0072b2"), ("3h", "#d55e00")):
                outcome = f"{metric}_gain_{lag}"
                summary = bins[
                    (bins["predictor"] == predictor) & (bins["outcome"] == outcome)
                ]
                axis.errorbar(
                    summary["predictor_mean"],
                    summary["outcome_mean"],
                    yerr=np.vstack(
                        [
                            summary["outcome_mean"] - summary["ci_low"],
                            summary["ci_high"] - summary["outcome_mean"],
                        ]
                    ),
                    color=color,
                    marker="o",
                    markersize=5,
                    linewidth=1.6,
                    capsize=3,
                    label=f"Symmetric ±{lag[0]} h" if row == 0 and column == 0 else None,
                )
            axis.axhline(0.0, color="0.3", linewidth=0.8)
            axis.set_xlabel(xlabel)
            axis.set_ylabel(ylabel if column == 0 else "")
            axis.grid(True, color="0.88", linewidth=0.6)
    axes[0, 0].legend()
    fig.suptitle(
        "Temporal-model gain over the no-temporal baseline\n"
        "Storm-day bin means with storm-stratified bootstrap 95% CIs",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_dir / "temporal_gain_over_baseline.png", dpi=190)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("/Users/simon/Datasets/Outputs/taiwan_2021_typhoon_cases_2021.nc"),
    )
    parser.add_argument(
        "--analysis-root", type=Path, default=Path("outputs/taiwan_typhoon_analysis")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/taiwan_timescale_hypothesis")
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    regimes = compute_regimes(args.dataset)
    skill = load_skill(args.analysis_root)
    tracks = load_track_speed(args.analysis_root)
    hourly = skill.merge(regimes, on=["storm", "time"], how="inner")
    hourly = hourly.merge(tracks, on=["storm", "time"], how="left")
    hourly.to_csv(args.output_dir / "hourly_regime_skill.csv", index=False)
    daily = aggregate_storm_days(hourly)
    daily.to_csv(args.output_dir / "storm_day_regime_skill.csv", index=False)
    bins = build_bin_summaries(daily)
    bins.to_csv(args.output_dir / "binned_skill_with_block_ci.csv", index=False)
    tests = []
    for predictor in (
        "radar_corr_3h",
        "radar_change_3h",
        "radar_aligned_change_1h",
        "jma_speed_kmh",
    ):
        for outcome in ("rmse_3h_minus_1h", "fss_3h_minus_1h"):
            tests.append(within_storm_spearman_test(daily, predictor, outcome))
    tests_frame = pd.DataFrame(tests)
    tests_frame.to_csv(args.output_dir / "within_storm_tests.csv", index=False)
    per_storm_spearman_tests(daily).to_csv(
        args.output_dir / "per_storm_tests.csv", index=False
    )
    plot_timescale_test(daily, bins, args.output_dir)
    plot_phase_diagram(daily, args.output_dir)
    plot_gains_over_baseline(bins, args.output_dir)
    metadata = {
        "hypothesis": "The preferred temporal lag depends on system persistence: ±3 h should beat ±1 h for persistent but measurably evolving systems, while ±1 h should win after 3 h correspondence is lost; both should converge for stationary systems.",
        "primary_outcomes": {
            "rmse": "ensemble-mean field RMSE; contrast is RMSE_1h - RMSE_3h",
            "fss": "member-mean FSS at 30 dBZ and 27 km; contrast is FSS_3h - FSS_1h",
        },
        "positive_contrast": "Symmetric ±3 h is better than symmetric ±1 h",
        "smoothing_scale_km": SMOOTHING_SCALE_PX * GRID_SPACING_KM,
        "minimum_active_area_km2": MIN_ACTIVE_AREA_KM2,
        "hourly_rows": len(hourly),
        "storm_day_rows": len(daily),
        "uncertainty": "storm-stratified bootstrap of UTC storm-days",
        "limitation": "Radar-derived regimes are diagnostic proxies; confirmation should estimate persistence from conditioning predictors and validate on independent annual episodes.",
    }
    with (args.output_dir / "metadata.json").open("w") as stream:
        json.dump(metadata, stream, indent=2)
    print(tests_frame.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
