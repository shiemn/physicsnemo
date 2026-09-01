#!/usr/bin/env python3
"""Held-out full-series validation of the Taiwan temporal-timescale hypothesis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter

if __package__:
    from .common import (
        ModelSource,
        common_times,
        member_mean_fss as _member_mean_fss,
        parse_model,
        season_for_month,
        time_indices,
    )
    from .analyze_temporal_timescale_hypothesis import (
        CHANNEL,
        GRID_SPACING_KM,
        MIN_ACTIVE_AREA_KM2,
        MODEL_1H,
        MODEL_3H,
        MODEL_BASELINE,
        PRIMARY_FSS_SCALE_KM,
        PRIMARY_FSS_THRESHOLD_DBZ,
        SMOOTHING_SCALE_PX,
        block_bootstrap_bins,
        field_correlation,
        normalized_change,
        phase_motion_and_deformation,
        within_storm_spearman_test,
    )
else:
    from common import (
        ModelSource,
        common_times,
        member_mean_fss as _member_mean_fss,
        parse_model,
        season_for_month,
        time_indices,
    )
    from analyze_temporal_timescale_hypothesis import (
        CHANNEL,
        GRID_SPACING_KM,
        MIN_ACTIVE_AREA_KM2,
        MODEL_1H,
        MODEL_3H,
        MODEL_BASELINE,
        PRIMARY_FSS_SCALE_KM,
        PRIMARY_FSS_THRESHOLD_DBZ,
        SMOOTHING_SCALE_PX,
        block_bootstrap_bins,
        field_correlation,
        normalized_change,
        phase_motion_and_deformation,
        within_storm_spearman_test,
    )


DISCOVERY_WINDOWS = (
    (pd.Timestamp("2021-07-17T00:00:00Z"), pd.Timestamp("2021-07-28T00:00:00Z")),
    (pd.Timestamp("2021-08-02T00:00:00Z"), pd.Timestamp("2021-08-10T00:00:00Z")),
    (pd.Timestamp("2021-09-06T00:00:00Z"), pd.Timestamp("2021-09-19T00:00:00Z")),
)
SEASON_COLORS = {
    "DJF": "#4575b4",
    "MAM": "#66bd63",
    "JJA": "#fdae61",
    "SON": "#d73027",
}


def member_mean_fss(
    prediction: np.ndarray,
    truth: np.ndarray,
    threshold: float = PRIMARY_FSS_THRESHOLD_DBZ,
    scale_px: int = int(PRIMARY_FSS_SCALE_KM / GRID_SPACING_KM),
) -> float:
    return _member_mean_fss(
        prediction, truth, threshold=threshold, scale_px=scale_px
    )


def is_discovery_time(timestamp: pd.Timestamp) -> bool:
    return any(start <= timestamp < stop for start, stop in DISCOVERY_WINDOWS)


def compute_hourly(models: list[ModelSource]) -> pd.DataFrame:
    common, native = common_times(models)
    if set(model.label for model in models) != {MODEL_BASELINE, MODEL_1H, MODEL_3H}:
        raise ValueError("models must be labelled Baseline, Symmetric 1 h, and Symmetric 3 h")
    sources = {model.label: Dataset(model.path) for model in models}
    indices = {
        model.label: time_indices(native[model.label], common) for model in models
    }
    rows: list[dict] = []
    fields: dict[pd.Timestamp, np.ndarray] = {}
    try:
        baseline_truth = sources[MODEL_BASELINE].groups["truth"].variables[CHANNEL]
        prediction_variables = {
            label: source.groups["prediction"].variables[CHANNEL]
            for label, source in sources.items()
        }
        for time_index, timestamp in enumerate(common):
            truth = np.asarray(
                baseline_truth[indices[MODEL_BASELINE][time_index]], dtype=np.float32
            )
            smoothed = uniform_filter(
                truth, size=SMOOTHING_SCALE_PX, mode="nearest"
            ).astype(np.float32)
            row = {
                "time": timestamp,
                "discovery_case": is_discovery_time(timestamp),
                "active_area_ge_20dbz_km2": float(
                    np.sum(truth >= 20.0) * GRID_SPACING_KM**2
                ),
            }
            for lag_hours in (1, 3):
                previous = fields.get(timestamp - pd.Timedelta(hours=lag_hours))
                if previous is None:
                    row[f"radar_corr_{lag_hours}h"] = np.nan
                    row[f"radar_change_{lag_hours}h"] = np.nan
                else:
                    row[f"radar_corr_{lag_hours}h"] = field_correlation(previous, smoothed)
                    row[f"radar_change_{lag_hours}h"] = normalized_change(previous, smoothed)
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
                motion, aligned_change, response = phase_motion_and_deformation(
                    previous, smoothed
                )
                row.update(
                    {
                        "radar_motion_kmh": motion,
                        "radar_aligned_change_1h": aligned_change,
                        "phase_response": response,
                    }
                )
            fields[timestamp] = smoothed
            for old_time in list(fields):
                if timestamp - old_time > pd.Timedelta(hours=3):
                    del fields[old_time]

            rmse, fss = {}, {}
            for label in (MODEL_BASELINE, MODEL_1H, MODEL_3H):
                prediction = np.asarray(
                    prediction_variables[label][:, indices[label][time_index]],
                    dtype=np.float32,
                )
                ensemble_mean = np.mean(prediction, axis=0)
                rmse[label] = float(np.sqrt(np.mean((ensemble_mean - truth) ** 2)))
                fss[label] = member_mean_fss(prediction, truth)
            row.update(
                {
                    "rmse_gain_1h": rmse[MODEL_BASELINE] - rmse[MODEL_1H],
                    "rmse_gain_3h": rmse[MODEL_BASELINE] - rmse[MODEL_3H],
                    "rmse_3h_minus_1h": rmse[MODEL_1H] - rmse[MODEL_3H],
                    "fss_gain_1h": fss[MODEL_1H] - fss[MODEL_BASELINE],
                    "fss_gain_3h": fss[MODEL_3H] - fss[MODEL_BASELINE],
                    "fss_3h_minus_1h": fss[MODEL_3H] - fss[MODEL_1H],
                }
            )
            rows.append(row)
            if (time_index + 1) % 100 == 0 or time_index + 1 == len(common):
                print(f"annual: {time_index + 1:,}/{len(common):,}", flush=True)
    finally:
        for source in sources.values():
            source.close()
    return pd.DataFrame(rows)


def aggregate_active_days(hourly: pd.DataFrame) -> pd.DataFrame:
    active = hourly[
        hourly["active_area_ge_20dbz_km2"] >= MIN_ACTIVE_AREA_KM2
    ].copy()
    active["date"] = active["time"].dt.floor("D")
    numeric = active.select_dtypes(include=[np.number, bool]).columns.tolist()
    daily = active.groupby("date", as_index=False)[numeric].mean()
    counts = active.groupby("date", as_index=False).size().rename(
        columns={"size": "valid_hours"}
    )
    daily = daily.merge(counts, on="date")
    daily = daily[daily["valid_hours"] >= 6].copy()
    daily["discovery_case"] = daily["discovery_case"] > 0.0
    daily["month_group"] = daily["date"].dt.strftime("%Y-%m")
    daily["season"] = daily["date"].dt.month.map(season_for_month)
    return daily.reset_index(drop=True)


def prepare_month_strata(daily: pd.DataFrame) -> pd.DataFrame:
    result = daily.copy()
    result["storm"] = result["month_group"]
    return result


def build_annual_bins(validation: pd.DataFrame) -> pd.DataFrame:
    predictors = [
        "radar_corr_3h",
        "radar_change_3h",
        "radar_motion_kmh",
        "radar_aligned_change_1h",
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
    stratified = prepare_month_strata(validation)
    for predictor in predictors:
        for outcome in outcomes:
            outputs.append(
                block_bootstrap_bins(
                    stratified,
                    predictor,
                    outcome,
                    bins=5,
                    replicates=2_000,
                    seed=20210714,
                )
            )
    return pd.concat(outputs, ignore_index=True)


def build_annual_tests(validation: pd.DataFrame) -> pd.DataFrame:
    outputs = []
    stratified = prepare_month_strata(validation)
    for predictor in (
        "radar_corr_3h",
        "radar_change_3h",
        "radar_motion_kmh",
        "radar_aligned_change_1h",
    ):
        for outcome in ("rmse_3h_minus_1h", "fss_3h_minus_1h"):
            result = within_storm_spearman_test(
                stratified,
                predictor,
                outcome,
                permutations=5_000,
                seed=20210714,
            )
            result["within_group"] = "calendar month"
            result["within_month_spearman"] = result.pop("within_storm_spearman")
            outputs.append(result)
    return pd.DataFrame(outputs)


def bootstrap_overall_contrasts(
    validation: pd.DataFrame,
    replicates: int = 5_000,
    seed: int = 20210714,
) -> pd.DataFrame:
    outcomes = [
        "rmse_gain_1h",
        "rmse_gain_3h",
        "rmse_3h_minus_1h",
        "fss_gain_1h",
        "fss_gain_3h",
        "fss_3h_minus_1h",
    ]
    groups = [group for _, group in validation.groupby("month_group")]
    rng = np.random.default_rng(seed)
    samples = {outcome: np.empty(replicates) for outcome in outcomes}
    for replicate in range(replicates):
        sampled = pd.concat(
            [group.iloc[rng.integers(0, len(group), len(group))] for group in groups],
            ignore_index=True,
        )
        for outcome in outcomes:
            samples[outcome][replicate] = sampled[outcome].mean()
    rows = []
    for outcome in outcomes:
        values = samples[outcome]
        rows.append(
            {
                "outcome": outcome,
                "active_days": len(validation),
                "mean": validation[outcome].mean(),
                "ci_low": np.quantile(values, 0.025),
                "ci_high": np.quantile(values, 0.975),
                "positive_fraction": float((validation[outcome] > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def plot_annual_test(validation: pd.DataFrame, bins: pd.DataFrame, output_dir: Path) -> None:
    predictors = [
        ("radar_corr_3h", "3 h radar persistence"),
        ("radar_change_3h", "3 h normalized change"),
        ("radar_motion_kmh", "Radar translation (km h⁻¹)"),
    ]
    outcomes = [
        ("rmse_3h_minus_1h", "RMSE advantage of ±3 h (dBZ)"),
        ("fss_3h_minus_1h", "FSS advantage of ±3 h"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for row, (outcome, ylabel) in enumerate(outcomes):
        for column, (predictor, xlabel) in enumerate(predictors):
            axis = axes[row, column]
            for season, group in validation.groupby("season"):
                axis.scatter(
                    group[predictor],
                    group[outcome],
                    s=15,
                    alpha=0.32,
                    color=SEASON_COLORS[season],
                    label=season if row == 0 and column == 0 else None,
                )
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
                color="black",
                marker="o",
                linewidth=1.5,
                markersize=5,
                capsize=3,
                label="Active-day bin mean" if row == 0 and column == 0 else None,
            )
            axis.axhline(0.0, color="0.3", linewidth=0.8)
            axis.set_xlabel(xlabel)
            axis.set_ylabel(ylabel if column == 0 else "")
            axis.grid(True, color="0.89", linewidth=0.6)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle(
        "Held-out annual validation of the temporal-timescale hypothesis\n"
        "Three discovery typhoon windows excluded; points are active UTC days",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_dir / "annual_timescale_skill_test.png", dpi=190)
    fig.savefig(output_dir / "annual_timescale_skill_test.pdf")
    plt.close(fig)


def plot_annual_gains(bins: pd.DataFrame, output_dir: Path) -> None:
    predictors = [
        ("radar_corr_3h", "3 h radar persistence"),
        ("radar_change_3h", "3 h normalized change"),
        ("radar_motion_kmh", "Radar translation (km h⁻¹)"),
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
                    linewidth=1.6,
                    markersize=5,
                    capsize=3,
                    label=f"Symmetric ±{lag[0]} h" if row == 0 and column == 0 else None,
                )
            axis.axhline(0.0, color="0.3", linewidth=0.8)
            axis.set_xlabel(xlabel)
            axis.set_ylabel(ylabel if column == 0 else "")
            axis.grid(True, color="0.89", linewidth=0.6)
    axes[0, 0].legend()
    fig.suptitle("Held-out annual temporal-model gains over baseline", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_dir / "annual_temporal_gain_over_baseline.png", dpi=190)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", type=parse_model, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/taiwan_timescale_full_2021")
    )
    parser.add_argument(
        "--reuse-hourly",
        action="store_true",
        help="Reuse output-dir/hourly_regime_skill.csv instead of reading NetCDFs.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    hourly_path = args.output_dir / "hourly_regime_skill.csv"
    if args.reuse_hourly and hourly_path.is_file():
        hourly = pd.read_csv(hourly_path)
        hourly["time"] = pd.to_datetime(hourly["time"], utc=True)
    else:
        hourly = compute_hourly(args.model)
        hourly.to_csv(hourly_path, index=False)
    daily = aggregate_active_days(hourly)
    daily.to_csv(args.output_dir / "active_day_regime_skill.csv", index=False)
    validation = daily[~daily["discovery_case"]].copy()
    validation.to_csv(args.output_dir / "heldout_active_days.csv", index=False)
    bins = build_annual_bins(validation)
    bins.to_csv(args.output_dir / "heldout_binned_skill_with_ci.csv", index=False)
    tests = build_annual_tests(validation)
    tests.to_csv(args.output_dir / "heldout_within_month_tests.csv", index=False)
    overall = bootstrap_overall_contrasts(validation)
    overall.to_csv(args.output_dir / "heldout_overall_contrasts.csv", index=False)
    plot_annual_test(validation, bins, args.output_dir)
    plot_annual_gains(bins, args.output_dir)
    metadata = {
        "common_hours": len(hourly),
        "active_days_all": len(daily),
        "heldout_active_days": len(validation),
        "discovery_windows_excluded": [
            [start.isoformat(), stop.isoformat()] for start, stop in DISCOVERY_WINDOWS
        ],
        "primary_metrics": {
            "rmse": "ensemble-mean RMSE",
            "fss": f"member-mean FSS at {PRIMARY_FSS_THRESHOLD_DBZ:g} dBZ and {PRIMARY_FSS_SCALE_KM:g} km",
        },
        "positive_head_to_head_contrast": "Symmetric ±3 h is better than symmetric ±1 h",
        "active_area_threshold_km2": MIN_ACTIVE_AREA_KM2,
        "minimum_active_hours_per_day": 6,
        "seasonal_control": "within-calendar-month demeaning and circular shifts",
        "regime_source": "27 km-smoothed radar truth; diagnostic proxy, not conditioning predictors",
    }
    with (args.output_dir / "metadata.json").open("w") as stream:
        json.dump(metadata, stream, indent=2)
    print(tests.to_string(index=False), flush=True)
    print(overall.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
