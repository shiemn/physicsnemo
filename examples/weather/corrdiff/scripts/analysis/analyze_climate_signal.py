#!/usr/bin/env python3
"""Analyze historical, mid-century, and end-century climate chunks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from climate_signal_common import (
    METRICS,
    _annual_and_climatology,
    _bootstrap,
    _load_epoch,
    _spatial_summary,
)


EPOCHS = ("historical", "mid_century", "end_century")
EPOCH_CENTERS = {
    "historical": 1995.5,
    "mid_century": 2050.5,
    "end_century": 2090.5,
}
COMPARISONS = {
    "historical_to_mid": ("historical", "mid_century"),
    "historical_to_end": ("historical", "end_century"),
    "mid_to_end": ("mid_century", "end_century"),
}


def _metric_epoch_summary(epoch: dict, metric: str) -> dict:
    prediction_annual, prediction_grid = _annual_and_climatology(
        epoch, "prediction", metric
    )
    target_annual, target_grid = _annual_and_climatology(epoch, "target", metric)
    prediction = float(np.nanmean(prediction_grid))
    target = float(np.nanmean(target_grid))
    return {
        "prediction": prediction,
        "target": target,
        "bias": prediction - target,
        "prediction_annual": prediction_annual,
        "target_annual": target_annual,
        "prediction_grid": prediction_grid,
        "target_grid": target_grid,
    }


def _comparison(
    earlier: dict,
    later: dict,
    bootstrap_samples: int,
    seed: int,
) -> dict:
    prediction_change = later["prediction"] - earlier["prediction"]
    target_change = later["target"] - earlier["target"]
    return {
        "target_change": target_change,
        "target_change_percent": 100.0 * target_change / earlier["target"],
        "prediction_change": prediction_change,
        "prediction_change_percent": 100.0
        * prediction_change
        / earlier["prediction"],
        "signal_error": prediction_change - target_change,
        "captured_signal_percent": 100.0 * prediction_change / target_change,
        "spatial_signal": _spatial_summary(
            later["prediction_grid"] - earlier["prediction_grid"],
            later["target_grid"] - earlier["target_grid"],
        ),
        "annual_bootstrap": _bootstrap(
            earlier["prediction_annual"],
            earlier["target_annual"],
            later["prediction_annual"],
            later["target_annual"],
            bootstrap_samples,
            seed,
        ),
    }


def analyze_epochs(
    paths: dict[str, list[Path]],
    bootstrap_samples: int,
    seed: int,
) -> tuple[dict, dict]:
    epoch_data = {name: _load_epoch(paths[name]) for name in EPOCHS}
    latitude = epoch_data["historical"]["latitude"]
    longitude = epoch_data["historical"]["longitude"]
    for name in EPOCHS[1:]:
        if not (
            np.allclose(latitude, epoch_data[name]["latitude"], equal_nan=True)
            and np.allclose(
                longitude, epoch_data[name]["longitude"], equal_nan=True
            )
        ):
            raise ValueError(f"Grid differs in {name}")

    internal: dict[str, dict[str, dict]] = {}
    result = {
        "domain_average": "unweighted mean over the 512x512 grid",
        "bootstrap_samples": bootstrap_samples,
        "epochs": {
            name: {
                "years": epoch_data[name]["years"].tolist(),
                "center_year": EPOCH_CENTERS[name],
            }
            for name in EPOCHS
        },
        "metrics": {},
    }

    for metric, (*_, units) in METRICS.items():
        internal[metric] = {
            name: _metric_epoch_summary(epoch_data[name], metric) for name in EPOCHS
        }
        metric_result = {
            "units": units,
            "climatologies": {
                name: {
                    key: internal[metric][name][key]
                    for key in ("target", "prediction", "bias")
                }
                for name in EPOCHS
            },
            "comparisons": {},
        }
        for comparison_name, (earlier_name, later_name) in COMPARISONS.items():
            metric_result["comparisons"][comparison_name] = _comparison(
                internal[metric][earlier_name],
                internal[metric][later_name],
                bootstrap_samples,
                seed,
            )

        hist_to_mid = metric_result["comparisons"]["historical_to_mid"]
        hist_to_end = metric_result["comparisons"]["historical_to_end"]
        mid_to_end = metric_result["comparisons"]["mid_to_end"]
        target_early_rate = hist_to_mid["target_change"] / (
            EPOCH_CENTERS["mid_century"] - EPOCH_CENTERS["historical"]
        )
        target_late_rate = mid_to_end["target_change"] / (
            EPOCH_CENTERS["end_century"] - EPOCH_CENTERS["mid_century"]
        )
        prediction_early_rate = hist_to_mid["prediction_change"] / (
            EPOCH_CENTERS["mid_century"] - EPOCH_CENTERS["historical"]
        )
        prediction_late_rate = mid_to_end["prediction_change"] / (
            EPOCH_CENTERS["end_century"] - EPOCH_CENTERS["mid_century"]
        )
        metric_result["progression"] = {
            "target_mid_fraction_of_end_signal": hist_to_mid["target_change"]
            / hist_to_end["target_change"],
            "prediction_mid_fraction_of_end_signal": hist_to_mid[
                "prediction_change"
            ]
            / hist_to_end["prediction_change"],
            "target_early_rate_per_year": target_early_rate,
            "target_late_rate_per_year": target_late_rate,
            "target_late_to_early_rate_ratio": target_late_rate
            / target_early_rate,
            "prediction_early_rate_per_year": prediction_early_rate,
            "prediction_late_rate_per_year": prediction_late_rate,
            "prediction_late_to_early_rate_ratio": prediction_late_rate
            / prediction_early_rate,
        }
        result["metrics"][metric] = metric_result

    return result, {
        "metrics": internal,
        "latitude": latitude,
        "longitude": longitude,
    }


def _write_climatology_csv(result: dict, path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("metric", "units", "epoch", "target", "prediction", "bias"),
        )
        writer.writeheader()
        for metric, values in result["metrics"].items():
            for epoch, climatology in values["climatologies"].items():
                writer.writerow(
                    {
                        "metric": metric,
                        "units": values["units"],
                        "epoch": epoch,
                        **climatology,
                    }
                )


def _write_comparison_csv(result: dict, path: Path) -> None:
    fields = (
        "metric",
        "units",
        "comparison",
        "target_change",
        "target_change_percent",
        "prediction_change",
        "prediction_change_percent",
        "signal_error",
        "signal_error_ci_low",
        "signal_error_ci_high",
        "captured_signal_percent",
        "spatial_rmse",
        "spatial_pattern_correlation",
        "spatial_sign_agreement_fraction",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for metric, values in result["metrics"].items():
            for comparison_name, comparison in values["comparisons"].items():
                ci = comparison["annual_bootstrap"]["signal_error_95ci"]
                spatial = comparison["spatial_signal"]
                writer.writerow(
                    {
                        "metric": metric,
                        "units": values["units"],
                        "comparison": comparison_name,
                        "target_change": comparison["target_change"],
                        "target_change_percent": comparison[
                            "target_change_percent"
                        ],
                        "prediction_change": comparison["prediction_change"],
                        "prediction_change_percent": comparison[
                            "prediction_change_percent"
                        ],
                        "signal_error": comparison["signal_error"],
                        "signal_error_ci_low": ci[0],
                        "signal_error_ci_high": ci[1],
                        "captured_signal_percent": comparison[
                            "captured_signal_percent"
                        ],
                        "spatial_rmse": spatial["rmse"],
                        "spatial_pattern_correlation": spatial[
                            "pattern_correlation"
                        ],
                        "spatial_sign_agreement_fraction": spatial[
                            "sign_agreement_fraction"
                        ],
                    }
                )


def _plot_trajectories(result: dict, path: Path, model_label: str) -> None:
    labels = {
        "mean_daily": "Mean daily precipitation",
        "sdii": "SDII",
        "rx1day": "Rx1day",
        "rx3h": "Rx3h",
        "wet_day_fraction": "Wet-day fraction",
    }
    years = [EPOCH_CENTERS[name] for name in EPOCHS]
    tick_labels = ["1986–2005", "2041–2060", "2081–2100"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    for axis, metric in zip(axes.flat, labels):
        values = result["metrics"][metric]
        target = [values["climatologies"][name]["target"] for name in EPOCHS]
        prediction = [
            values["climatologies"][name]["prediction"] for name in EPOCHS
        ]
        axis.plot(years, target, marker="o", linewidth=2, label="HCLIM target")
        axis.plot(years, prediction, marker="s", linewidth=2, label=model_label)
        axis.set_title(labels[metric])
        axis.set_ylabel(values["units"])
        axis.set_xticks(years, tick_labels, rotation=15)
        axis.grid(alpha=0.25)
    axes.flat[-1].axis("off")
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower right", bbox_to_anchor=(0.96, 0.08))
    fig.suptitle(f"{model_label} climatologies across the three 20-year epochs")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_signal_maps(
    internal: dict, comparison: str, path: Path, model_label: str
) -> None:
    earlier_name, later_name = COMPARISONS[comparison]
    metrics = ("mean_daily", "sdii", "rx1day", "rx3h")
    labels = {
        "mean_daily": "Mean daily",
        "sdii": "SDII",
        "rx1day": "Rx1day",
        "rx3h": "Rx3h",
    }
    longitude = internal["longitude"]
    latitude = internal["latitude"]
    fig, axes = plt.subplots(4, 3, figsize=(14, 15), constrained_layout=True)
    for row, metric in enumerate(metrics):
        earlier = internal["metrics"][metric][earlier_name]
        later = internal["metrics"][metric][later_name]
        target = later["target_grid"] - earlier["target_grid"]
        prediction = later["prediction_grid"] - earlier["prediction_grid"]
        error = prediction - target
        signal_limit = float(
            np.nanpercentile(np.abs(np.concatenate([target.ravel(), prediction.ravel()])), 99)
        )
        error_limit = float(np.nanpercentile(np.abs(error), 99))
        for col, (field, title, limit) in enumerate(
            (
                (target, "HCLIM target", signal_limit),
                (prediction, model_label, signal_limit),
                (error, "Model − target", error_limit),
            )
        ):
            mesh = axes[row, col].pcolormesh(
                longitude,
                latitude,
                field,
                shading="auto",
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
                rasterized=True,
            )
            axes[row, col].set_title(f"{labels[metric]} — {title}")
            axes[row, col].set_xlabel("Longitude")
            axes[row, col].set_ylabel("Latitude")
            fig.colorbar(mesh, ax=axes[row, col], shrink=0.75)
    fig.suptitle(comparison.replace("_", " ").title() + " climate signal")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical", type=Path, nargs="+", required=True)
    parser.add_argument("--mid-century", type=Path, nargs="+", required=True)
    parser.add_argument("--end-century", type=Path, nargs="+", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--model-key", default="baseline")
    parser.add_argument("--model-label", default="Baseline")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result, internal = analyze_epochs(
        {
            "historical": args.historical,
            "mid_century": args.mid_century,
            "end_century": args.end_century,
        },
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    result["model_key"] = args.model_key
    result["model_label"] = args.model_label
    prefix = args.model_key.replace("/", "_").replace(" ", "_")
    (args.output_dir / f"{prefix}_three_epoch_analysis.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    _write_climatology_csv(
        result, args.output_dir / f"{prefix}_epoch_climatologies.csv"
    )
    _write_comparison_csv(
        result, args.output_dir / f"{prefix}_signal_comparisons.csv"
    )
    _plot_trajectories(
        result,
        args.output_dir / f"{prefix}_epoch_trajectories.png",
        args.model_label,
    )
    _plot_signal_maps(
        internal,
        "historical_to_mid",
        args.output_dir / f"{prefix}_signal_maps_historical_to_mid.png",
        args.model_label,
    )
    _plot_signal_maps(
        internal,
        "historical_to_end",
        args.output_dir / f"{prefix}_signal_maps_historical_to_end.png",
        args.model_label,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
