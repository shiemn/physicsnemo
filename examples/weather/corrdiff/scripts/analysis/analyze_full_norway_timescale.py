#!/usr/bin/env python3
"""Test the temporal-timescale hypothesis on the full Norway 2005 series."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np
import pandas as pd
from scipy.ndimage import shift as spatial_shift
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
        block_bootstrap_bins,
        field_correlation,
        normalized_change,
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
        block_bootstrap_bins,
        field_correlation,
        normalized_change,
        within_storm_spearman_test,
    )


CHANNEL = "precipitation"
MODEL_PAST3 = "Past 3h"
MODEL_PAST12 = "Past 12h"
GRID_SPACING_KM = 2.0
STEP_HOURS = 3.0
SMOOTHING_SCALE_PX = 13  # 26 km
FSS_THRESHOLD = 1.0  # mm per 3 h
FSS_SCALE_PX = 13  # 26 km
MIN_ACTIVE_AREA_KM2 = 1_000.0
MIN_ACTIVE_STEPS_PER_DAY = 4
SEASON_COLORS = {"DJF": "#4575b4", "MAM": "#66bd63", "JJA": "#fdae61", "SON": "#d73027"}


def member_mean_fss(
    prediction: np.ndarray,
    truth: np.ndarray,
    threshold: float = FSS_THRESHOLD,
    scale_px: int = FSS_SCALE_PX,
) -> float:
    """Mean deterministic FSS across ensemble members."""
    return _member_mean_fss(
        prediction, truth, threshold=threshold, scale_px=scale_px
    )


def phase_motion_and_deformation(
    previous: np.ndarray, current: np.ndarray
) -> tuple[float, float, float]:
    """Return translation speed (km/h), residual change, and phase response."""
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
        float(np.hypot(dx, dy) * GRID_SPACING_KM / STEP_HOURS),
        normalized_change(aligned, current),
        float(response),
    )


def load_rmse(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    pivot = frame.pivot(index="time", columns="model", values="rmse")
    required = ["Baseline", MODEL_PAST3, MODEL_PAST12, "Symmetric 3h"]
    if any(label not in pivot for label in required):
        raise ValueError(f"RMSE table must contain {required}")
    return pivot


def compute_steps(models: list[ModelSource], rmse_path: Path) -> pd.DataFrame:
    if set(model.label for model in models) != {MODEL_PAST3, MODEL_PAST12}:
        raise ValueError("models must be labelled Past 3h and Past 12h")
    common, native = common_times(models)
    rmse = load_rmse(rmse_path)
    common = common.intersection(rmse.index)
    sources = {model.label: Dataset(model.path) for model in models}
    indices = {model.label: time_indices(native[model.label], common) for model in models}
    truth_variable = sources[MODEL_PAST3].groups["truth"].variables[CHANNEL]
    prediction_variables = {
        label: source.groups["prediction"].variables[CHANNEL]
        for label, source in sources.items()
    }
    fields: dict[pd.Timestamp, np.ndarray] = {}
    rows: list[dict] = []
    try:
        for position, timestamp in enumerate(common):
            truth = np.asarray(
                truth_variable[indices[MODEL_PAST3][position]], dtype=np.float32
            )
            smooth = uniform_filter(
                truth, size=SMOOTHING_SCALE_PX, mode="nearest"
            ).astype(np.float32)
            row = {
                "time": timestamp,
                "active_area_ge_1mm_km2": float(
                    np.sum(truth >= FSS_THRESHOLD) * GRID_SPACING_KM**2
                ),
            }
            for lag_hours in (3, 12):
                previous = fields.get(timestamp - pd.Timedelta(hours=lag_hours))
                if previous is None:
                    row[f"precip_corr_{lag_hours}h"] = np.nan
                    row[f"precip_change_{lag_hours}h"] = np.nan
                else:
                    row[f"precip_corr_{lag_hours}h"] = field_correlation(previous, smooth)
                    row[f"precip_change_{lag_hours}h"] = normalized_change(previous, smooth)
            previous = fields.get(timestamp - pd.Timedelta(hours=3))
            if previous is None:
                motion, aligned_change, response = np.nan, np.nan, np.nan
            else:
                motion, aligned_change, response = phase_motion_and_deformation(previous, smooth)
            row.update(
                {
                    "precip_motion_kmh": motion,
                    "precip_aligned_change_3h": aligned_change,
                    "phase_response": response,
                }
            )
            fields[timestamp] = smooth
            for old_time in list(fields):
                if timestamp - old_time > pd.Timedelta(hours=12):
                    del fields[old_time]

            fss = {}
            for label in (MODEL_PAST3, MODEL_PAST12):
                prediction = np.asarray(
                    prediction_variables[label][:, indices[label][position]],
                    dtype=np.float32,
                )
                fss[label] = member_mean_fss(prediction, truth)
            scores = rmse.loc[timestamp]
            row.update(
                {
                    # Positive values always favor the longer, past-12h context.
                    "rmse_past12_minus_past3": float(scores[MODEL_PAST3] - scores[MODEL_PAST12]),
                    "fss_past12_minus_past3": fss[MODEL_PAST12] - fss[MODEL_PAST3],
                    "rmse_gain_past3": float(scores["Baseline"] - scores[MODEL_PAST3]),
                    "rmse_gain_past12": float(scores["Baseline"] - scores[MODEL_PAST12]),
                    # Same lag, different information direction: future-context diagnostic.
                    "rmse_symmetric_minus_past3": float(scores[MODEL_PAST3] - scores["Symmetric 3h"]),
                }
            )
            rows.append(row)
            if (position + 1) % 100 == 0 or position + 1 == len(common):
                print(f"Norway: {position + 1:,}/{len(common):,}", flush=True)
    finally:
        for source in sources.values():
            source.close()
    return pd.DataFrame(rows)


def aggregate_active_days(steps: pd.DataFrame) -> pd.DataFrame:
    active = steps[steps["active_area_ge_1mm_km2"] >= MIN_ACTIVE_AREA_KM2].copy()
    active["date"] = active["time"].dt.floor("D")
    numeric = active.select_dtypes(include=np.number).columns.tolist()
    daily = active.groupby("date", as_index=False)[numeric].mean()
    counts = active.groupby("date", as_index=False).size().rename(columns={"size": "valid_steps"})
    daily = daily.merge(counts, on="date")
    daily = daily[daily["valid_steps"] >= MIN_ACTIVE_STEPS_PER_DAY].copy()
    daily["month_group"] = daily["date"].dt.strftime("%Y-%m")
    daily["storm"] = daily["month_group"]  # expected grouping name in shared statistics
    daily["season"] = daily["date"].dt.month.map(season_for_month)
    return daily.reset_index(drop=True)


PREDICTORS = (
    "precip_corr_3h",
    "precip_change_3h",
    "precip_corr_12h",
    "precip_change_12h",
    "precip_motion_kmh",
    "precip_aligned_change_3h",
)
OUTCOMES = ("rmse_past12_minus_past3", "fss_past12_minus_past3")


def build_bins(daily: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(
        [
            block_bootstrap_bins(daily, predictor, outcome, bins=5, replicates=2_000)
            for predictor in PREDICTORS
            for outcome in OUTCOMES
        ],
        ignore_index=True,
    )


def build_tests(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for predictor in PREDICTORS:
        for outcome in OUTCOMES:
            result = within_storm_spearman_test(
                daily, predictor, outcome, permutations=5_000
            )
            result["within_group"] = "calendar month"
            result["within_month_spearman"] = result.pop("within_storm_spearman")
            rows.append(result)
    return pd.DataFrame(rows)


def bootstrap_contrasts(
    daily: pd.DataFrame, replicates: int = 5_000, seed: int = 20210714
) -> pd.DataFrame:
    outcomes = [
        *OUTCOMES,
        "rmse_gain_past3",
        "rmse_gain_past12",
        "rmse_symmetric_minus_past3",
    ]
    groups = [group for _, group in daily.groupby("month_group")]
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
                "active_days": len(daily),
                "mean": daily[outcome].mean(),
                "ci_low": np.quantile(values, 0.025),
                "ci_high": np.quantile(values, 0.975),
                "positive_fraction": float((daily[outcome] > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def plot_test(daily: pd.DataFrame, bins: pd.DataFrame, output_dir: Path) -> None:
    predictors = [
        ("precip_corr_3h", "Adjacent-frame (3 h) persistence"),
        ("precip_change_3h", "Adjacent-frame (3 h) normalized change"),
        ("precip_motion_kmh", "3 h translation estimate (km h⁻¹)"),
    ]
    outcomes = [
        ("rmse_past12_minus_past3", "RMSE advantage of past 12 h"),
        ("fss_past12_minus_past3", "FSS advantage of past 12 h"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for row, (outcome, ylabel) in enumerate(outcomes):
        for column, (predictor, xlabel) in enumerate(predictors):
            axis = axes[row, column]
            for season, group in daily.groupby("season"):
                axis.scatter(
                    group[predictor], group[outcome], s=16, alpha=0.3,
                    color=SEASON_COLORS[season],
                    label=season if row == 0 and column == 0 else None,
                )
            summary = bins[(bins["predictor"] == predictor) & (bins["outcome"] == outcome)]
            axis.errorbar(
                summary["predictor_mean"], summary["outcome_mean"],
                yerr=np.vstack((summary["outcome_mean"] - summary["ci_low"], summary["ci_high"] - summary["outcome_mean"])),
                color="black", marker="o", linewidth=1.5, markersize=5, capsize=3,
                label="Active-day bin mean" if row == 0 and column == 0 else None,
            )
            axis.axhline(0, color="0.3", linewidth=0.8)
            axis.set_xlabel(xlabel)
            axis.set_ylabel(ylabel if column == 0 else "")
            axis.grid(True, color="0.89", linewidth=0.6)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle(
        "Full-year Norway test: does slower evolution favor longer past context?\n"
        "Positive skill means past 12 h beats past 3 h; points are active UTC days",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_dir / "norway_timescale_skill_test.png", dpi=190)
    fig.savefig(output_dir / "norway_timescale_skill_test.pdf")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", type=parse_model, required=True)
    parser.add_argument("--rmse-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/norway_timescale_full_2005"))
    parser.add_argument("--reuse-steps", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    steps_path = args.output_dir / "step_regime_skill.csv"
    if args.reuse_steps and steps_path.is_file():
        steps = pd.read_csv(steps_path)
        steps["time"] = pd.to_datetime(steps["time"], utc=True)
    else:
        steps = compute_steps(args.model, args.rmse_csv)
        steps.to_csv(steps_path, index=False)
    daily = aggregate_active_days(steps)
    daily.to_csv(args.output_dir / "active_day_regime_skill.csv", index=False)
    bins = build_bins(daily)
    bins.to_csv(args.output_dir / "binned_skill_with_ci.csv", index=False)
    tests = build_tests(daily)
    tests.to_csv(args.output_dir / "within_month_tests.csv", index=False)
    contrasts = bootstrap_contrasts(daily)
    contrasts.to_csv(args.output_dir / "overall_contrasts.csv", index=False)
    plot_test(daily, bins, args.output_dir)
    metadata = {
        "domain": "Norway",
        "period": [steps["time"].min().isoformat(), steps["time"].max().isoformat()],
        "common_three_hour_steps": len(steps),
        "active_days": len(daily),
        "comparison": "Past 12h versus Past 3h",
        "positive_head_to_head_contrast": "Past 12h is better than Past 3h",
        "primary_metrics": {
            "rmse": "precomputed ensemble-mean precipitation RMSE",
            "fss": "member-mean FSS at 1 mm/3h and 26 km",
        },
        "regime_source": "26 km-smoothed precipitation truth; primary evolution proxies compare adjacent 3h frames",
        "active_day_rule": f">={MIN_ACTIVE_AREA_KM2:g} km2 at >=1 mm/3h for >={MIN_ACTIVE_STEPS_PER_DAY} steps",
        "seasonal_control": "within-calendar-month demeaning and circular shifts",
        "future_context_diagnostic": "Symmetric 3h versus Past 3h RMSE only; not interpreted as a timescale contrast",
    }
    with (args.output_dir / "metadata.json").open("w") as stream:
        json.dump(metadata, stream, indent=2)
    print(tests.to_string(index=False), flush=True)
    print(contrasts.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
