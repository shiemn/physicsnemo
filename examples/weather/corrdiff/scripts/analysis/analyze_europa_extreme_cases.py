#!/usr/bin/env python3
"""Select and compare extreme Europe precipitation cases from annual NetCDFs."""

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

if __package__:
    from .common import read_times as _read_times
else:
    from common import read_times as _read_times


CHANNEL = "precipitation_amount_1hr"
MODEL_COLORS = {
    "Baseline": "#4c566a",
    "Symmetric 1h": "#0072b2",
    "Symmetric 3h": "#d55e00",
}


def parse_model(value: str) -> tuple[str, Path]:
    try:
        label, path = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("models must use LABEL=/path/predictions.nc") from exc
    return label, Path(path)


def read_times(ds: Dataset) -> pd.DatetimeIndex:
    return _read_times(ds, utc=False)


def top_tail_mean(field: np.ndarray, fraction: float) -> float:
    finite = np.asarray(field, dtype=np.float32).ravel()
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    count = max(1, int(np.ceil(finite.size * fraction)))
    return float(np.mean(np.partition(finite, finite.size - count)[-count:]))


def scan_truth(path: Path, fraction: float, chunk_size: int) -> tuple[pd.DatetimeIndex, pd.DataFrame]:
    rows: list[dict] = []
    with Dataset(path) as ds:
        times = read_times(ds)
        variable = ds.groups["truth"].variables[CHANNEL]
        for start in range(0, len(times), chunk_size):
            stop = min(len(times), start + chunk_size)
            fields = np.asarray(variable[start:stop], dtype=np.float32)
            for offset, field in enumerate(fields):
                positive = np.maximum(field, 0.0)
                rows.append(
                    {
                        "index": start + offset,
                        "time": times[start + offset],
                        "tail_mean_mm_h": top_tail_mean(positive, fraction),
                        "maximum_mm_h": float(np.nanmax(positive)),
                        "domain_mean_mm_h": float(np.nanmean(positive)),
                        "wet_fraction_1mm": float(np.nanmean(positive >= 1.0)),
                        "heavy_fraction_5mm": float(np.nanmean(positive >= 5.0)),
                    }
                )
            print(f"truth scan: {stop:,}/{len(times):,}", flush=True)
    return times, pd.DataFrame(rows)


def select_separated_cases(catalogue: pd.DataFrame, count: int, separation_hours: int) -> pd.DataFrame:
    selected: list[pd.Series] = []
    for _, candidate in catalogue.sort_values("tail_mean_mm_h", ascending=False).iterrows():
        time = pd.Timestamp(candidate.time)
        if all(abs(time - pd.Timestamp(item.time)) >= pd.Timedelta(hours=separation_hours) for item in selected):
            selected.append(candidate)
            if len(selected) == count:
                break
    if len(selected) < count:
        raise RuntimeError(f"only found {len(selected)} cases with requested separation")
    result = pd.DataFrame(selected).sort_values("time").reset_index(drop=True)
    result.insert(0, "case", [f"E{i + 1}" for i in range(len(result))])
    return result


def empirical_crps(prediction: np.ndarray, truth: np.ndarray) -> float:
    first = np.nanmean(np.abs(prediction - truth[None]))
    pairwise = 0.0
    ensemble = prediction.shape[0]
    for i in range(ensemble):
        for j in range(ensemble):
            pairwise += float(np.nanmean(np.abs(prediction[i] - prediction[j])))
    return float(first - pairwise / (2.0 * ensemble * ensemble))


def spatial_correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 2 or np.std(left[valid]) == 0 or np.std(right[valid]) == 0:
        return float("nan")
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def categorical_scores(prediction: np.ndarray, truth: np.ndarray, threshold: float) -> tuple[float, float]:
    predicted = prediction >= threshold
    observed = truth >= threshold
    hits = np.count_nonzero(predicted & observed)
    misses = np.count_nonzero(~predicted & observed)
    false_alarms = np.count_nonzero(predicted & ~observed)
    csi_denominator = hits + misses + false_alarms
    observed_count = hits + misses
    return (
        float(hits / csi_denominator) if csi_denominator else float("nan"),
        float(hits / observed_count) if observed_count else float("nan"),
    )


def case_metrics(case: str, time: pd.Timestamp, label: str, prediction: np.ndarray, truth: np.ndarray) -> dict:
    mean_prediction = np.nanmean(prediction, axis=0)
    error = mean_prediction - truth
    csi_1, pod_1 = categorical_scores(mean_prediction, truth, 1.0)
    csi_5, pod_5 = categorical_scores(mean_prediction, truth, 5.0)
    return {
        "case": case,
        "time": time,
        "model": label,
        "rmse": float(np.sqrt(np.nanmean(np.square(error)))),
        "mae": float(np.nanmean(np.abs(error))),
        "bias": float(np.nanmean(error)),
        "crps": empirical_crps(prediction, truth),
        "spatial_correlation": spatial_correlation(mean_prediction, truth),
        "truth_max_mm_h": float(np.nanmax(truth)),
        "prediction_max_mm_h": float(np.nanmax(mean_prediction)),
        "truth_tail_mean_mm_h": top_tail_mean(truth, 0.001),
        "prediction_tail_mean_mm_h": top_tail_mean(mean_prediction, 0.001),
        "truth_wet_fraction_1mm": float(np.nanmean(truth >= 1.0)),
        "prediction_wet_fraction_1mm": float(np.nanmean(mean_prediction >= 1.0)),
        "csi_1mm": csi_1,
        "pod_1mm": pod_1,
        "csi_5mm": csi_5,
        "pod_5mm": pod_5,
    }


def load_selected(
    models: list[tuple[str, Path]], times: pd.DatetimeIndex, cases: pd.DataFrame
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    center_indices = cases["index"].astype(int).to_numpy()
    sequence_indices = np.stack([center_indices - 3, center_indices, center_indices + 3], axis=1)
    if sequence_indices.min() < 0 or sequence_indices.max() >= len(times):
        raise ValueError("selected sequence reaches beyond the common time axis")
    truth_by_case: dict[str, np.ndarray] = {}
    prediction_by_model: dict[str, np.ndarray] = {}
    reference_truth = None
    for label, path in models:
        with Dataset(path) as ds:
            model_times = read_times(ds)
            if not model_times.equals(times):
                raise ValueError(f"{label} does not use the reference time axis")
            truth_variable = ds.groups["truth"].variables[CHANNEL]
            prediction_variable = ds.groups["prediction"].variables[CHANNEL]
            truth = np.stack(
                [np.asarray(truth_variable[row], dtype=np.float32) for row in sequence_indices]
            )
            prediction = np.stack(
                [np.asarray(prediction_variable[:, row], dtype=np.float32) for row in sequence_indices]
            )
        if reference_truth is None:
            reference_truth = truth
        elif not np.allclose(reference_truth, truth, equal_nan=True):
            raise ValueError(f"truth differs for {label}")
        prediction_by_model[label] = prediction
    assert reference_truth is not None
    for case_index, case in enumerate(cases["case"]):
        truth_by_case[case] = reference_truth[case_index]
    return truth_by_case, prediction_by_model


def plot_overview(
    cases: pd.DataFrame,
    truth_by_case: dict[str, np.ndarray],
    predictions: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    labels = list(predictions)
    fig, axes = plt.subplots(len(cases), len(labels) + 1, figsize=(3.2 * (len(labels) + 1), 2.8 * len(cases)))
    for row, case in enumerate(cases.itertuples(index=False)):
        truth = truth_by_case[case.case][1]
        fields = [("Truth", truth)] + [
            (label, np.nanmean(predictions[label][row, :, 1], axis=0)) for label in labels
        ]
        vmax = max(5.0, float(np.nanquantile(np.concatenate([field.ravel() for _, field in fields]), 0.999)))
        for column, (label, field) in enumerate(fields):
            axis = axes[row, column]
            image = axis.imshow(field, origin="lower", cmap="Blues", vmin=0.0, vmax=vmax)
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(label)
            if column == 0:
                axis.set_ylabel(f"{case.case}\n{pd.Timestamp(case.time):%d %b %HZ}")
        fig.colorbar(image, ax=axes[row, :], fraction=0.012, pad=0.008, label="mm h$^{-1}$")
    fig.suptitle("Europe 2021: spatially coherent extreme precipitation peaks", y=0.998)
    fig.savefig(output_dir / "extreme_cases_overview.png", dpi=180, bbox_inches="tight")
    fig.savefig(output_dir / "extreme_cases_overview.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_sequences(
    cases: pd.DataFrame,
    truth_by_case: dict[str, np.ndarray],
    predictions: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    labels = list(predictions)
    for row, case in enumerate(cases.itertuples(index=False)):
        truth = truth_by_case[case.case]
        fields = [("Truth", truth)] + [
            (label, np.nanmean(predictions[label][row], axis=0)) for label in labels
        ]
        vmax = max(5.0, float(np.nanquantile(np.concatenate([field.ravel() for _, field in fields]), 0.999)))
        fig, axes = plt.subplots(len(fields), 3, figsize=(10.5, 2.7 * len(fields)), constrained_layout=True)
        for field_row, (label, sequence) in enumerate(fields):
            for column, offset in enumerate((-3, 0, 3)):
                axis = axes[field_row, column]
                image = axis.imshow(sequence[column], origin="lower", cmap="Blues", vmin=0.0, vmax=vmax)
                axis.set_xticks([])
                axis.set_yticks([])
                if column == 0:
                    axis.set_ylabel(label)
                if field_row == 0:
                    axis.set_title(f"{offset:+d} h")
        fig.colorbar(image, ax=axes, shrink=0.7, label="Precipitation (mm h$^{-1}$)")
        fig.suptitle(f"{case.case}: {pd.Timestamp(case.time):%Y-%m-%d %H:%M UTC}")
        fig.savefig(output_dir / f"{case.case.lower()}_sequence.png", dpi=180, bbox_inches="tight")
        fig.savefig(output_dir / f"{case.case.lower()}_sequence.pdf", bbox_inches="tight")
        plt.close(fig)


def plot_metric_summary(metrics: pd.DataFrame, output_dir: Path) -> None:
    metrics = metrics.copy()
    baseline = metrics[metrics.model == "Baseline"].set_index("case")["rmse"]
    metrics["rmse_change_percent"] = [
        100.0 * (row.rmse / baseline.loc[row.case] - 1.0) for row in metrics.itertuples()
    ]
    cases = list(metrics["case"].drop_duplicates())
    labels = list(metrics["model"].drop_duplicates())
    x = np.arange(len(cases))
    width = 0.24
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True, constrained_layout=True)
    for index, label in enumerate(labels):
        subset = metrics[metrics.model == label].set_index("case").loc[cases]
        offset = (index - (len(labels) - 1) / 2) * width
        axes[0].bar(x + offset, subset.rmse, width, label=label, color=MODEL_COLORS.get(label))
        axes[1].bar(x + offset, subset.rmse_change_percent, width, color=MODEL_COLORS.get(label))
    axes[0].set_ylabel("RMSE (mm h$^{-1}$)")
    axes[0].legend(ncol=len(labels))
    axes[1].axhline(0.0, color="0.35", linewidth=0.8)
    axes[1].set_ylabel("RMSE change vs baseline (%)")
    axes[1].set_xticks(x, cases)
    axes[1].set_xlabel("Extreme case")
    for axis in axes:
        axis.grid(axis="y", alpha=0.3)
    fig.savefig(output_dir / "extreme_case_rmse_summary.png", dpi=180)
    fig.savefig(output_dir / "extreme_case_rmse_summary.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", type=parse_model, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-count", type=int, default=6)
    parser.add_argument("--separation-hours", type=int, default=24)
    parser.add_argument("--tail-fraction", type=float, default=0.001)
    parser.add_argument("--chunk-size", type=int, default=16)
    args = parser.parse_args()
    if not 0.0 < args.tail_fraction <= 1.0:
        parser.error("--tail-fraction must be in (0, 1]")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reference_path = args.model[0][1]
    times, catalogue = scan_truth(reference_path, args.tail_fraction, args.chunk_size)
    cases = select_separated_cases(catalogue, args.case_count, args.separation_hours)
    truth_by_case, predictions = load_selected(args.model, times, cases)

    metric_rows = []
    for row, case in enumerate(cases.itertuples(index=False)):
        truth = truth_by_case[case.case][1]
        for label in predictions:
            metric_rows.append(
                case_metrics(case.case, pd.Timestamp(case.time), label, predictions[label][row, :, 1], truth)
            )
    metrics = pd.DataFrame(metric_rows)
    baseline_rmse = metrics[metrics.model == "Baseline"].set_index("case")["rmse"]
    metrics["rmse_change_percent_vs_baseline"] = [
        100.0 * (row.rmse / baseline_rmse.loc[row.case] - 1.0) for row in metrics.itertuples()
    ]
    metrics["rmse_rank"] = metrics.groupby("case")["rmse"].rank(method="min").astype(int)
    cases.to_csv(args.output_dir / "selected_extreme_cases.csv", index=False, float_format="%.7g")
    catalogue.to_csv(args.output_dir / "extreme_score_catalogue.csv", index=False, float_format="%.7g")
    metrics.to_csv(args.output_dir / "extreme_case_metrics.csv", index=False, float_format="%.7g")
    summary = (
        metrics.groupby("model", sort=False)
        .agg(
            cases=("case", "size"),
            mean_rmse=("rmse", "mean"),
            median_rmse=("rmse", "median"),
            mean_mae=("mae", "mean"),
            mean_crps=("crps", "mean"),
            mean_bias=("bias", "mean"),
            mean_spatial_correlation=("spatial_correlation", "mean"),
            mean_csi_5mm=("csi_5mm", "mean"),
            rmse_wins=("rmse_rank", lambda values: int(np.sum(values == 1))),
        )
        .reset_index()
    )
    summary.to_csv(args.output_dir / "extreme_case_summary.csv", index=False, float_format="%.7g")
    plot_overview(cases, truth_by_case, predictions, args.output_dir)
    plot_sequences(cases, truth_by_case, predictions, args.output_dir)
    plot_metric_summary(metrics, args.output_dir)
    metadata = {
        "selection": "top spatial tail-mean truth precipitation with temporal separation",
        "channel": CHANNEL,
        "tail_fraction": args.tail_fraction,
        "case_count": args.case_count,
        "separation_hours": args.separation_hours,
        "models": {label: str(path) for label, path in args.model},
        "timesteps_scanned": len(times),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(cases.to_string(index=False), flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
