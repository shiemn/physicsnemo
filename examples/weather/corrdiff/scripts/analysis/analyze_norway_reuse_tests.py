#!/usr/bin/env python3
"""Reuse-only tests of the temporal observability theory on Norway 2005."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from netCDF4 import Dataset
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch

if __package__:
    from .common import ModelSource, common_times, parse_model, time_indices
else:
    from common import ModelSource, common_times, parse_model, time_indices


CHANNEL = "precipitation"
MODELS = ("Baseline", "Past 3h", "Past 12h", "Symmetric 3h")
TEMPORAL_MODELS = MODELS[1:]
THRESHOLDS = (0.1, 1.0, 5.0)
SCALES_PX = (5, 13, 25, 51)
GRID_KM = 2.0
PRIMARY_THRESHOLD = 1.0
PRIMARY_SCALE_PX = 13
ACTIVE_AREA_KM2 = 1_000.0
MIN_ACTIVE_STEPS = 4
CORE_PREDICTORS = ("precip_corr_3h", "precip_change_3h")
EXPLORATORY_PREDICTORS = ("precip_motion_kmh", "precip_aligned_change_3h")
COARSE_PREDICTORS = (
    "coarse_state_change_3h",
    "coarse_humidity_change_3h",
    "coarse_wind_change_3h",
    "coarse_temperature_change_3h",
)
MODEL_COLORS = {
    "Past 3h": "#0072b2",
    "Past 12h": "#d55e00",
    "Symmetric 3h": "#009e73",
}


def fss_grid_for_fields(
    predictions: dict[str, np.ndarray], truth: np.ndarray
) -> dict[tuple[str, float, int], float]:
    """Calculate member-mean FSS efficiently for all requested thresholds/scales."""
    truth_binary = np.stack(
        [(truth >= threshold).astype(np.float32) for threshold in THRESHOLDS], axis=-1
    )
    prediction_binary = {}
    for model, values in predictions.items():
        # OpenCV treats the last dimension as channels: threshold x member.
        prediction_binary[model] = np.concatenate(
            [
                np.stack(
                    [(member >= threshold).astype(np.float32) for threshold in THRESHOLDS],
                    axis=-1,
                )
                for member in values
            ],
            axis=-1,
        )
    output: dict[tuple[str, float, int], float] = {}
    for scale_px in SCALES_PX:
        truth_fraction = cv2.boxFilter(
            truth_binary,
            ddepth=-1,
            ksize=(scale_px, scale_px),
            normalize=True,
            borderType=cv2.BORDER_CONSTANT,
        )
        truth_energy = np.mean(truth_fraction**2, axis=(0, 1))
        for model, binary in prediction_binary.items():
            predicted_fraction = cv2.boxFilter(
                binary,
                ddepth=-1,
                ksize=(scale_px, scale_px),
                normalize=True,
                borderType=cv2.BORDER_CONSTANT,
            )
            member_scores = []
            for member in range(predictions[model].shape[0]):
                start = member * len(THRESHOLDS)
                stop = start + len(THRESHOLDS)
                fraction = predicted_fraction[..., start:stop]
                denominator = np.mean(fraction**2, axis=(0, 1)) + truth_energy
                numerator = np.mean((fraction - truth_fraction) ** 2, axis=(0, 1))
                member_scores.append(
                    np.where(denominator > 1e-12, 1.0 - numerator / denominator, np.nan)
                )
            scores = np.nanmean(np.stack(member_scores), axis=0)
            for index, threshold in enumerate(THRESHOLDS):
                output[(model, threshold, scale_px)] = float(scores[index])
    return output


def compute_fss_grid(models: list[ModelSource]) -> pd.DataFrame:
    if set(model.label for model in models) != set(MODELS):
        raise ValueError(f"models must be labelled {MODELS}")
    common, native = common_times(models)
    sources = {model.label: Dataset(model.path) for model in models}
    indices = {model.label: time_indices(native[model.label], common) for model in models}
    truth_variable = sources["Baseline"].groups["truth"].variables[CHANNEL]
    prediction_variables = {
        label: source.groups["prediction"].variables[CHANNEL]
        for label, source in sources.items()
    }
    rows: list[dict] = []
    try:
        for position, timestamp in enumerate(common):
            truth = np.asarray(truth_variable[indices["Baseline"][position]], dtype=np.float32)
            predictions = {
                label: np.asarray(variable[:, indices[label][position]], dtype=np.float32)
                for label, variable in prediction_variables.items()
            }
            scores = fss_grid_for_fields(predictions, truth)
            active_area = float(np.sum(truth >= PRIMARY_THRESHOLD) * GRID_KM**2)
            for (model, threshold, scale_px), score in scores.items():
                rows.append(
                    {
                        "time": timestamp,
                        "model": model,
                        "threshold_mm_per_3h": threshold,
                        "scale_px": scale_px,
                        "scale_km": scale_px * GRID_KM,
                        "fss": score,
                        "active_area_ge_1mm_km2": active_area,
                    }
                )
            if (position + 1) % 100 == 0 or position + 1 == len(common):
                print(f"FSS grid: {position + 1:,}/{len(common):,}", flush=True)
    finally:
        for source in sources.values():
            source.close()
    return pd.DataFrame(rows)


def normalized_vector_change(first: np.ndarray, second: np.ndarray) -> float:
    numerator = np.sqrt(np.mean((second - first) ** 2, dtype=np.float64))
    denominator = np.sqrt(
        0.5
        * (
            np.mean(first**2, dtype=np.float64)
            + np.mean(second**2, dtype=np.float64)
        )
    )
    return float(numerator / denominator) if denominator > 1e-12 else np.nan


def compute_coarse_regimes(
    predictor_h5: Path, predictor_stats: Path, times: pd.DatetimeIndex
) -> pd.DataFrame:
    """Inference-available evolution proxies from normalized coarse predictors."""
    mean, std = torch.load(predictor_stats, weights_only=False)
    mean = np.asarray(mean, dtype=np.float32).reshape(-1, 1, 1)
    std = np.asarray(std, dtype=np.float32).reshape(-1, 1, 1)
    start = pd.Timestamp("2005-01-01T00:00:00Z")
    rows = []
    previous = None
    previous_index = None
    with h5py.File(predictor_h5, "r") as source:
        variable = source["predictors"]
        for timestamp in times:
            source_index = int((timestamp - start).total_seconds() / (3 * 3600))
            if source_index < 0 or source_index >= len(variable):
                raise IndexError(f"{timestamp} maps outside {predictor_h5}")
            current = (np.asarray(variable[source_index], dtype=np.float32) - mean) / std
            row = {"time": timestamp}
            if previous is None or previous_index != source_index - 1:
                for predictor in COARSE_PREDICTORS:
                    row[predictor] = np.nan
            else:
                row["coarse_state_change_3h"] = normalized_vector_change(previous, current)
                row["coarse_humidity_change_3h"] = normalized_vector_change(
                    previous[0:2], current[0:2]
                )
                row["coarse_wind_change_3h"] = normalized_vector_change(
                    previous[2:6], current[2:6]
                )
                row["coarse_temperature_change_3h"] = normalized_vector_change(
                    previous[6:8], current[6:8]
                )
            rows.append(row)
            previous, previous_index = current, source_index
    return pd.DataFrame(rows)


def valid_active_times(step: pd.DataFrame) -> pd.DatetimeIndex:
    active = step[step["active_area_ge_1mm_km2"] >= ACTIVE_AREA_KM2].copy()
    active["date"] = active["time"].dt.floor("D")
    counts = active.groupby("date").size()
    valid_dates = counts[counts >= MIN_ACTIVE_STEPS].index
    return pd.DatetimeIndex(active.loc[active["date"].isin(valid_dates), "time"])


def prepare_daily(
    fss: pd.DataFrame, step: pd.DataFrame, errors: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    active_times = valid_active_times(step)
    active_step = step[step["time"].isin(active_times)].copy()
    active_step["date"] = active_step["time"].dt.floor("D")
    numeric = active_step.select_dtypes(include=np.number).columns.tolist()
    regimes = active_step.groupby("date", as_index=False)[numeric].mean()
    regimes["month_group"] = regimes["date"].dt.strftime("%Y-%m")

    active_errors = errors[errors["time"].isin(active_times)].copy()
    active_errors["date"] = active_errors["time"].dt.floor("D")
    daily_errors = active_errors.groupby(["date", "model"], as_index=False)[
        ["rmse", "mae", "bias", "crps"]
    ].mean()
    error_wide = daily_errors.pivot(index="date", columns="model")
    error_daily = pd.DataFrame(index=error_wide.index)
    for metric in ("rmse", "mae", "bias", "crps"):
        for model in MODELS:
            error_daily[f"{metric}_{model}"] = error_wide[(metric, model)]
    error_daily = error_daily.reset_index()

    active_fss = fss[fss["time"].isin(active_times)].copy()
    active_fss["date"] = active_fss["time"].dt.floor("D")
    daily_fss = active_fss.groupby(
        ["date", "model", "threshold_mm_per_3h", "scale_px", "scale_km"],
        as_index=False,
    )["fss"].mean()
    pivot = daily_fss.pivot(
        index=["date", "threshold_mm_per_3h", "scale_px", "scale_km"],
        columns="model",
        values="fss",
    ).reset_index()
    pivot["fss_past12_minus_past3"] = pivot["Past 12h"] - pivot["Past 3h"]
    for model in TEMPORAL_MODELS:
        pivot[f"fss_gain_{model}"] = pivot[model] - pivot["Baseline"]

    daily = regimes.merge(error_daily, on="date", how="inner")
    daily["rmse_past12_minus_past3"] = daily["rmse_Past 3h"] - daily["rmse_Past 12h"]
    for model in TEMPORAL_MODELS:
        daily[f"rmse_gain_{model}"] = daily["rmse_Baseline"] - daily[f"rmse_{model}"]
        daily[f"crps_gain_{model}"] = daily["crps_Baseline"] - daily[f"crps_{model}"]
    return daily, pivot


def within_month_spearman(frame: pd.DataFrame, predictor: str, outcome: str) -> float:
    valid = frame[np.isfinite(frame[predictor]) & np.isfinite(frame[outcome])].copy()
    x = valid[predictor] - valid.groupby("month_group")[predictor].transform("mean")
    y = valid[outcome] - valid.groupby("month_group")[outcome].transform("mean")
    return float(spearmanr(x, y).statistic)


def circular_shift_p(
    frame: pd.DataFrame,
    predictor: str,
    outcome: str,
    permutations: int = 2_000,
    seed: int = 20210715,
) -> tuple[float, float]:
    valid = (
        frame[np.isfinite(frame[predictor]) & np.isfinite(frame[outcome])]
        .sort_values(["month_group", "date"])
        .reset_index(drop=True)
        .copy()
    )
    x = valid[predictor] - valid.groupby("month_group")[predictor].transform("mean")
    y = valid[outcome] - valid.groupby("month_group")[outcome].transform("mean")
    x_values, y_values = x.to_numpy(), y.to_numpy()
    observed = float(spearmanr(x_values, y_values).statistic)
    groups = [group.index.to_numpy() for _, group in valid.groupby("month_group", sort=False)]
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(permutations):
        shifted = y_values.copy()
        for positions in groups:
            shifted[positions] = np.roll(
                shifted[positions], int(rng.integers(0, len(positions)))
            )
        exceed += abs(spearmanr(x_values, shifted).statistic) >= abs(observed)
    return observed, float((exceed + 1) / (permutations + 1))


def bh_qvalues(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.clip(adjusted, 0.0, 1.0)
    return output


def build_scale_tests(daily: pd.DataFrame, daily_fss: pd.DataFrame) -> pd.DataFrame:
    merged = daily_fss.merge(
        daily[["date", "month_group", *CORE_PREDICTORS]], on="date", how="inner"
    )
    rows = []
    for predictor in CORE_PREDICTORS:
        for (threshold, scale_px, scale_km), group in merged.groupby(
            ["threshold_mm_per_3h", "scale_px", "scale_km"]
        ):
            rho, p_value = circular_shift_p(
                group, predictor, "fss_past12_minus_past3"
            )
            rows.append(
                {
                    "predictor": predictor,
                    "threshold_mm_per_3h": threshold,
                    "scale_px": scale_px,
                    "scale_km": scale_km,
                    "active_days": len(group),
                    "within_month_spearman": rho,
                    "circular_shift_p": p_value,
                }
            )
    result = pd.DataFrame(rows)
    result["fdr_q_across_grid"] = bh_qvalues(result["circular_shift_p"].to_numpy())
    return result


def circular_block_sample_indices(
    frame: pd.DataFrame, block_days: int, rng: np.random.Generator
) -> np.ndarray:
    sampled = []
    for _, group in frame.groupby("month_group", sort=False):
        indices = group.index.to_numpy()
        chosen = []
        while len(chosen) < len(indices):
            start = int(rng.integers(0, len(indices)))
            chosen.extend(indices[(start + np.arange(block_days)) % len(indices)].tolist())
        sampled.extend(chosen[: len(indices)])
    return np.asarray(sampled, dtype=int)


def bootstrap_mean(
    frame: pd.DataFrame,
    column: str,
    block_days: int = 3,
    replicates: int = 2_000,
    seed: int = 20210715,
) -> tuple[float, float, float]:
    ordered = frame.sort_values(["month_group", "date"]).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates)
    for index in range(replicates):
        positions = circular_block_sample_indices(ordered, block_days, rng)
        samples[index] = ordered.iloc[positions][column].mean()
    return (
        float(ordered[column].mean()),
        float(np.quantile(samples, 0.025)),
        float(np.quantile(samples, 0.975)),
    )


def build_overall_contrasts(daily: pd.DataFrame, daily_fss: pd.DataFrame) -> pd.DataFrame:
    primary = daily_fss[
        (daily_fss["threshold_mm_per_3h"] == PRIMARY_THRESHOLD)
        & (daily_fss["scale_px"] == PRIMARY_SCALE_PX)
    ]
    merged = daily.merge(primary[["date", *[f"fss_gain_{m}" for m in TEMPORAL_MODELS]]], on="date")
    rows = []
    for metric in ("rmse", "crps", "fss"):
        for model in TEMPORAL_MODELS:
            column = f"{metric}_gain_{model}"
            mean, low, high = bootstrap_mean(merged, column)
            rows.append(
                {
                    "metric": metric,
                    "model": model,
                    "positive_means": "temporal model better than baseline",
                    "active_days": len(merged),
                    "mean_gain": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "positive_day_fraction": float((merged[column] > 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def bootstrap_spearman(
    frame: pd.DataFrame,
    predictor: str,
    outcome: str,
    block_days: int,
    replicates: int = 2_000,
    seed: int = 20210715,
) -> tuple[float, float, float]:
    valid = (
        frame[np.isfinite(frame[predictor]) & np.isfinite(frame[outcome])]
        .sort_values(["month_group", "date"])
        .reset_index(drop=True)
    )
    observed = within_month_spearman(valid, predictor, outcome)
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates)
    for index in range(replicates):
        positions = circular_block_sample_indices(valid, block_days, rng)
        sample = valid.iloc[positions].copy()
        samples[index] = within_month_spearman(sample, predictor, outcome)
    return observed, float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def build_block_sensitivity(daily: pd.DataFrame, daily_fss: pd.DataFrame) -> pd.DataFrame:
    primary = daily_fss[
        (daily_fss["threshold_mm_per_3h"] == PRIMARY_THRESHOLD)
        & (daily_fss["scale_px"] == PRIMARY_SCALE_PX)
    ][["date", "fss_past12_minus_past3"]]
    merged = daily.drop(columns=["fss_past12_minus_past3"], errors="ignore").merge(
        primary, on="date"
    )
    rows = []
    for predictor in (*CORE_PREDICTORS, *EXPLORATORY_PREDICTORS, *COARSE_PREDICTORS):
        for metric, outcome in (
            ("RMSE", "rmse_past12_minus_past3"),
            ("FSS", "fss_past12_minus_past3"),
        ):
            for block_days in (1, 3, 7):
                rho, low, high = bootstrap_spearman(
                    merged, predictor, outcome, block_days
                )
                rows.append(
                    {
                        "predictor": predictor,
                        "metric": metric,
                        "block_days": block_days,
                        "within_month_spearman": rho,
                        "ci_low": low,
                        "ci_high": high,
                        "active_days": len(merged),
                    }
                )
    return pd.DataFrame(rows)


def build_coarse_primary_tests(
    daily: pd.DataFrame, daily_fss: pd.DataFrame
) -> pd.DataFrame:
    primary = daily_fss[
        (daily_fss["threshold_mm_per_3h"] == PRIMARY_THRESHOLD)
        & (daily_fss["scale_px"] == PRIMARY_SCALE_PX)
    ][["date", "fss_past12_minus_past3"]]
    merged = daily.drop(columns=["fss_past12_minus_past3"], errors="ignore").merge(
        primary, on="date"
    )
    rows = []
    for predictor in COARSE_PREDICTORS:
        for metric, outcome in (
            ("RMSE", "rmse_past12_minus_past3"),
            ("FSS", "fss_past12_minus_past3"),
        ):
            rho, p_value = circular_shift_p(merged, predictor, outcome)
            rows.append(
                {
                    "predictor": predictor,
                    "metric": metric,
                    "active_days": len(merged),
                    "within_month_spearman": rho,
                    "circular_shift_p": p_value,
                    "expected_for_fss": "negative: larger input-state change favors past 3h",
                }
            )
    result = pd.DataFrame(rows)
    result["fdr_q_across_8_tests"] = bh_qvalues(result["circular_shift_p"].to_numpy())
    return result


def build_metric_interactions(daily: pd.DataFrame, daily_fss: pd.DataFrame) -> pd.DataFrame:
    primary = daily_fss[
        (daily_fss["threshold_mm_per_3h"] == PRIMARY_THRESHOLD)
        & (daily_fss["scale_px"] == PRIMARY_SCALE_PX)
    ][["date", "fss_past12_minus_past3"]]
    merged = (
        daily.drop(columns=["fss_past12_minus_past3"], errors="ignore")
        .merge(primary, on="date")
        .sort_values(["month_group", "date"])
        .reset_index(drop=True)
    )
    rows = []
    rng = np.random.default_rng(20210715)
    overall_rmse_gain = float(
        0.5 * (merged["rmse_gain_Past 3h"].mean() + merged["rmse_gain_Past 12h"].mean())
    )
    for predictor in CORE_PREDICTORS:
        rho_rmse = within_month_spearman(merged, predictor, "rmse_past12_minus_past3")
        rho_fss = within_month_spearman(merged, predictor, "fss_past12_minus_past3")
        valid = merged[
            np.isfinite(merged[predictor])
            & np.isfinite(merged["rmse_past12_minus_past3"])
            & np.isfinite(merged["fss_past12_minus_past3"])
        ].copy().reset_index(drop=True)
        differences = np.empty(2_000)
        ratios = np.empty(2_000)
        iqr = float(valid[predictor].quantile(0.75) - valid[predictor].quantile(0.25))
        for replicate in range(2_000):
            positions = circular_block_sample_indices(valid, 3, rng)
            sample = valid.iloc[positions].copy()
            rmse_rho = within_month_spearman(sample, predictor, "rmse_past12_minus_past3")
            fss_rho = within_month_spearman(sample, predictor, "fss_past12_minus_past3")
            differences[replicate] = fss_rho - rmse_rho
            x = sample[predictor] - sample.groupby("month_group")[predictor].transform("mean")
            y = sample["rmse_past12_minus_past3"] - sample.groupby("month_group")[
                "rmse_past12_minus_past3"
            ].transform("mean")
            slope = float(np.sum(x * y) / max(np.sum(x * x), 1e-12))
            ratios[replicate] = slope * iqr / overall_rmse_gain
        x = valid[predictor] - valid.groupby("month_group")[predictor].transform("mean")
        y = valid["rmse_past12_minus_past3"] - valid.groupby("month_group")[
            "rmse_past12_minus_past3"
        ].transform("mean")
        observed_slope = float(np.sum(x * y) / max(np.sum(x * x), 1e-12))
        ratio = observed_slope * iqr / overall_rmse_gain
        rows.append(
            {
                "predictor": predictor,
                "rho_rmse": rho_rmse,
                "rho_fss": rho_fss,
                "rho_fss_minus_rmse": rho_fss - rho_rmse,
                "difference_ci_low": np.quantile(differences, 0.025),
                "difference_ci_high": np.quantile(differences, 0.975),
                "rmse_iqr_effect_fraction_of_temporal_gain": ratio,
                "rmse_fraction_ci_low": np.quantile(ratios, 0.025),
                "rmse_fraction_ci_high": np.quantile(ratios, 0.975),
                "equivalent_within_10pct": bool(
                    np.quantile(ratios, 0.025) > -0.1
                    and np.quantile(ratios, 0.975) < 0.1
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_overall(contrasts: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    titles = {"rmse": "RMSE gain", "crps": "CRPS gain", "fss": "FSS gain (1 mm, 26 km)"}
    for axis, metric in zip(axes, ("rmse", "crps", "fss")):
        part = contrasts[contrasts["metric"] == metric].set_index("model").loc[list(TEMPORAL_MODELS)]
        y = np.arange(len(part))
        for index, (model, row) in enumerate(part.iterrows()):
            axis.errorbar(
                row["mean_gain"], index,
                xerr=[[row["mean_gain"] - row["ci_low"]], [row["ci_high"] - row["mean_gain"]]],
                fmt="o", color=MODEL_COLORS[model], capsize=4, markersize=7,
            )
        axis.axvline(0, color="0.35", linewidth=0.9)
        axis.set_yticks(y, part.index if axis is axes[0] else [""] * len(part))
        axis.set_title(titles[metric])
        axis.set_xlabel("Gain over baseline (positive is better)")
        axis.grid(True, axis="x", color="0.9", linewidth=0.7)
    fig.suptitle("Norway 2005: paired temporal-model gains on 360 active days")
    fig.tight_layout()
    fig.savefig(output_dir / "overall_temporal_gains.png", dpi=190)
    fig.savefig(output_dir / "overall_temporal_gains.pdf")
    plt.close(fig)


def plot_scale_heatmap(tests: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    labels = {
        "precip_corr_3h": "Adjacent-frame persistence",
        "precip_change_3h": "Adjacent-frame normalized change",
    }
    limit = max(abs(tests["within_month_spearman"].min()), abs(tests["within_month_spearman"].max()))
    image = None
    for axis, predictor in zip(axes, CORE_PREDICTORS):
        part = tests[tests["predictor"] == predictor]
        matrix = part.pivot(index="threshold_mm_per_3h", columns="scale_km", values="within_month_spearman")
        qmatrix = part.pivot(index="threshold_mm_per_3h", columns="scale_km", values="fdr_q_across_grid")
        image = axis.imshow(matrix.to_numpy(), cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                star = "*" if qmatrix.iloc[row, column] < 0.05 else ""
                axis.text(column, row, f"{matrix.iloc[row, column]:+.2f}{star}", ha="center", va="center", fontsize=9)
        axis.set_xticks(np.arange(matrix.shape[1]), [f"{value:g}" for value in matrix.columns])
        axis.set_yticks(np.arange(matrix.shape[0]), [f"{value:g}" for value in matrix.index])
        axis.set_xlabel("FSS neighborhood scale (km)")
        axis.set_title(labels[predictor])
    axes[0].set_ylabel("Precipitation threshold (mm/3 h)")
    color_axis = fig.add_axes([0.91, 0.18, 0.018, 0.58])
    fig.colorbar(image, cax=color_axis, label="Within-month Spearman rho")
    fig.suptitle("Past 12 h versus past 3 h: regime dependence across the FSS grid\n* FDR q < 0.05 across 24 tests")
    fig.subplots_adjust(left=0.08, right=0.88, bottom=0.14, top=0.78, wspace=0.18)
    fig.savefig(output_dir / "fss_regime_scale_heatmap.png", dpi=190)
    fig.savefig(output_dir / "fss_regime_scale_heatmap.pdf")
    plt.close(fig)


def _plot_threshold_panel(
    axis: plt.Axes, part: pd.DataFrame, threshold: float, show_legend: bool = False
) -> None:
    styles = {
        "precip_corr_3h": ("Persistence", "#d55e00", "o"),
        "precip_change_3h": ("Normalized change", "#0072b2", "s"),
    }
    scales = sorted(part["scale_km"].unique())
    positions = np.arange(len(scales))
    for predictor, (label, color, marker) in styles.items():
        selected = part[part["predictor"] == predictor].set_index("scale_km").loc[scales]
        values = selected["within_month_spearman"].to_numpy()
        significant = selected["fdr_q_across_grid"].to_numpy() < 0.05
        axis.plot(positions, values, color=color, linewidth=2.0, label=label)
        for position, value, is_significant in zip(positions, values, significant):
            axis.scatter(
                position,
                value,
                marker=marker,
                s=62,
                facecolor=color if is_significant else "white",
                edgecolor=color,
                linewidth=1.8,
                zorder=3,
            )
            vertical = 0.018 if value >= 0 else -0.025
            axis.text(
                position,
                value + vertical,
                f"{value:+.2f}{'*' if is_significant else ''}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=9,
            )
    axis.axhline(0.0, color="0.3", linewidth=0.9)
    axis.set_xticks(positions, [f"{scale:g}" for scale in scales])
    axis.set_ylim(-0.27, 0.27)
    axis.set_title(f"Threshold: {threshold:g} mm / 3 h")
    axis.set_xlabel("FSS neighborhood scale (km)")
    axis.grid(True, axis="y", color="0.9", linewidth=0.7)
    if show_legend:
        handles = [
            Line2D([0], [0], color=styles[key][1], marker=styles[key][2], linewidth=2, label=styles[key][0])
            for key in styles
        ]
        handles.extend(
            [
                Line2D([0], [0], color="0.35", marker="o", markerfacecolor="0.35", linewidth=0, label="FDR q < 0.05"),
                Line2D([0], [0], color="0.35", marker="o", markerfacecolor="white", linewidth=0, label="Not FDR-significant"),
            ]
        )
        axis.legend(handles=handles, fontsize=9, loc="best")


def plot_threshold_correlations(tests: pd.DataFrame, output_dir: Path) -> None:
    """Write combined and threshold-specific PNG plots with an explicit rho definition."""
    definition = (
        "ρ = Spearman(R_day − monthly mean R, A_day − monthly mean A),  "
        "A_day = daily mean(FSS_past12h − FSS_past3h);  n = 360 active days"
    )
    thresholds = sorted(tests["threshold_mm_per_3h"].unique())
    fig, axes = plt.subplots(1, len(thresholds), figsize=(16, 5.6), sharey=True)
    for index, (axis, threshold) in enumerate(zip(axes, thresholds)):
        part = tests[tests["threshold_mm_per_3h"] == threshold]
        _plot_threshold_panel(axis, part, threshold, show_legend=index == 0)
    axes[0].set_ylabel(
        "Within-month Spearman ρ\nregime vs daily FSS difference (past 12 h − past 3 h)"
    )
    fig.suptitle("Norway 2005: temporal-lag preference by precipitation threshold", y=0.98)
    fig.text(0.5, 0.015, definition, ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(output_dir / "fss_regime_correlations_all_thresholds.png", dpi=200)
    fig.savefig(output_dir / "fss_regime_correlations_all_thresholds.pdf")
    plt.close(fig)

    for threshold in thresholds:
        part = tests[tests["threshold_mm_per_3h"] == threshold]
        fig, axis = plt.subplots(figsize=(8.6, 6.0))
        _plot_threshold_panel(axis, part, threshold, show_legend=True)
        axis.set_ylabel(
            "Within-month Spearman ρ\nregime vs daily FSS difference (past 12 h − past 3 h)"
        )
        fig.suptitle("Norway 2005: temporal-lag preference", y=0.98)
        fig.text(0.5, 0.015, definition, ha="center", fontsize=8.5)
        fig.tight_layout(rect=(0, 0.065, 1, 0.94))
        threshold_name = str(threshold).replace(".", "p")
        fig.savefig(
            output_dir / f"fss_regime_correlation_threshold_{threshold_name}mm.png",
            dpi=200,
        )
        fig.savefig(
            output_dir / f"fss_regime_correlation_threshold_{threshold_name}mm.pdf"
        )
        plt.close(fig)


def plot_block_sensitivity(sensitivity: pd.DataFrame, output_dir: Path) -> None:
    core = sensitivity[sensitivity["predictor"].isin(CORE_PREDICTORS)]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    labels = {
        "precip_corr_3h": "Adjacent-frame persistence",
        "precip_change_3h": "Adjacent-frame normalized change",
    }
    colors = {"RMSE": "#7f7f7f", "FSS": "#0072b2"}
    offsets = {"RMSE": -0.08, "FSS": 0.08}
    for axis, predictor in zip(axes, CORE_PREDICTORS):
        part = core[core["predictor"] == predictor]
        for metric in ("RMSE", "FSS"):
            selected = part[part["metric"] == metric].sort_values("block_days")
            x = np.arange(len(selected)) + offsets[metric]
            axis.errorbar(
                x, selected["within_month_spearman"],
                yerr=np.vstack((selected["within_month_spearman"] - selected["ci_low"], selected["ci_high"] - selected["within_month_spearman"])),
                fmt="o-", color=colors[metric], capsize=4, label=metric,
            )
        axis.axhline(0, color="0.35", linewidth=0.9)
        axis.set_xticks(np.arange(3), ["1 day", "3 days", "7 days"])
        axis.set_xlabel("Circular moving-block bootstrap")
        axis.set_title(labels[predictor])
        axis.grid(True, axis="y", color="0.9", linewidth=0.7)
    axes[0].set_ylabel("Within-month Spearman rho")
    axes[0].legend()
    fig.suptitle("Serial-dependence sensitivity of the primary 1 mm / 26 km result")
    fig.tight_layout()
    fig.savefig(output_dir / "block_length_sensitivity.png", dpi=190)
    fig.savefig(output_dir / "block_length_sensitivity.pdf")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", type=parse_model, required=True)
    parser.add_argument("--step-csv", type=Path, required=True)
    parser.add_argument("--error-csv", type=Path, required=True)
    parser.add_argument("--predictor-h5", type=Path, required=True)
    parser.add_argument("--predictor-stats", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/norway_reuse_tests_2005"))
    parser.add_argument("--reuse-fss", action="store_true")
    parser.add_argument("--reuse-coarse", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fss_path = args.output_dir / "step_fss_grid.csv"
    if args.reuse_fss and fss_path.is_file():
        fss = pd.read_csv(fss_path)
        fss["time"] = pd.to_datetime(fss["time"], utc=True)
    else:
        fss = compute_fss_grid(args.model)
        fss.to_csv(fss_path, index=False)
    step = pd.read_csv(args.step_csv)
    step["time"] = pd.to_datetime(step["time"], utc=True)
    coarse_path = args.output_dir / "step_coarse_regimes.csv"
    if args.reuse_coarse and coarse_path.is_file():
        coarse = pd.read_csv(coarse_path)
        coarse["time"] = pd.to_datetime(coarse["time"], utc=True)
    else:
        coarse = compute_coarse_regimes(
            args.predictor_h5, args.predictor_stats, pd.DatetimeIndex(step["time"])
        )
        coarse.to_csv(coarse_path, index=False)
    step = step.drop(columns=list(COARSE_PREDICTORS), errors="ignore").merge(
        coarse, on="time", how="left"
    )
    errors = pd.read_csv(args.error_csv)
    errors["time"] = pd.to_datetime(errors["time"], utc=True)
    daily, daily_fss = prepare_daily(fss, step, errors)
    daily.to_csv(args.output_dir / "daily_regimes_and_errors.csv", index=False)
    daily_fss.to_csv(args.output_dir / "daily_fss_grid.csv", index=False)
    scale_tests = build_scale_tests(daily, daily_fss)
    scale_tests.to_csv(args.output_dir / "fss_grid_regime_tests.csv", index=False)
    contrasts = build_overall_contrasts(daily, daily_fss)
    contrasts.to_csv(args.output_dir / "overall_temporal_contrasts.csv", index=False)
    sensitivity = build_block_sensitivity(daily, daily_fss)
    sensitivity.to_csv(args.output_dir / "block_length_sensitivity.csv", index=False)
    interactions = build_metric_interactions(daily, daily_fss)
    interactions.to_csv(args.output_dir / "metric_regime_interactions.csv", index=False)
    coarse_tests = build_coarse_primary_tests(daily, daily_fss)
    coarse_tests.to_csv(args.output_dir / "coarse_predictor_regime_tests.csv", index=False)
    plot_overall(contrasts, args.output_dir)
    plot_scale_heatmap(scale_tests, args.output_dir)
    plot_threshold_correlations(scale_tests, args.output_dir)
    plot_block_sensitivity(sensitivity, args.output_dir)
    metadata = {
        "analysis": "reuse-only Norway temporal theory tests",
        "common_steps": int(fss["time"].nunique()),
        "active_days": int(daily["date"].nunique()),
        "models": list(MODELS),
        "thresholds_mm_per_3h": list(THRESHOLDS),
        "scales_km": [scale * GRID_KM for scale in SCALES_PX],
        "primary_fss": {"threshold_mm_per_3h": PRIMARY_THRESHOLD, "scale_km": PRIMARY_SCALE_PX * GRID_KM},
        "resampling": "calendar-month-stratified circular moving blocks",
        "limitations": [
            "two ensemble members do not identify conditional-distribution narrowing",
            "no duplicate-current or shuffled-context controls",
            "two past-only lags cannot identify curvature or an interior optimal lag",
        ],
    }
    with (args.output_dir / "metadata.json").open("w") as stream:
        json.dump(metadata, stream, indent=2)
    print("OVERALL\n", contrasts.to_string(index=False), flush=True)
    print("\nINTERACTIONS\n", interactions.to_string(index=False), flush=True)
    print("\nFSS GRID\n", scale_tests.to_string(index=False), flush=True)
    print("\nCOARSE INPUT REGIMES\n", coarse_tests.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
