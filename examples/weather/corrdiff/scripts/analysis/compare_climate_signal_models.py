#!/usr/bin/env python3
"""Compare two completed three-epoch climate-signal model suites."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analyze_climate_signal import COMPARISONS, EPOCHS, analyze_epochs
from climate_signal_common import METRICS, _load_epoch


CORE_METRICS = ("mean_daily", "sdii", "rx1day", "rx3h")
TARGET_SUFFIXES = sorted(
    {spec[0] for spec in METRICS.values()}
    | {spec[1] for spec in METRICS.values() if spec[1]}
    | {"step_sum", "step_count"}
)


def _verify_common_targets(
    reference_epochs: dict[str, dict], candidate_epochs: dict[str, dict]
) -> dict:
    checked: list[str] = []
    for epoch in EPOCHS:
        reference = reference_epochs[epoch]
        candidate = candidate_epochs[epoch]
        if not np.array_equal(reference["years"], candidate["years"]):
            raise AssertionError(f"Year mismatch for {epoch}")
        for coordinate in ("latitude", "longitude"):
            if not np.array_equal(
                reference[coordinate], candidate[coordinate], equal_nan=True
            ):
                raise AssertionError(f"{coordinate} mismatch for {epoch}")
            checked.append(f"{epoch}:{coordinate}")
        for suffix in TARGET_SUFFIXES:
            name = f"target_{suffix}"
            if not np.array_equal(
                reference[name], candidate[name], equal_nan=True
            ):
                difference = np.nanmax(np.abs(reference[name] - candidate[name]))
                raise AssertionError(
                    f"Target mismatch for {epoch}:{name}; max difference={difference}"
                )
            checked.append(f"{epoch}:{name}")
    return {"identical": True, "checked_arrays": checked}


def _ci(values: np.ndarray) -> list[float]:
    return [float(x) for x in np.percentile(values, [2.5, 97.5])]


def _paired_bootstrap(
    reference_internal: dict,
    candidate_internal: dict,
    metric: str,
    comparison_name: str,
    samples: int,
    seed: int,
) -> dict:
    earlier_name, later_name = COMPARISONS[comparison_name]
    ref_earlier = reference_internal["metrics"][metric][earlier_name]
    ref_later = reference_internal["metrics"][metric][later_name]
    cand_earlier = candidate_internal["metrics"][metric][earlier_name]
    cand_later = candidate_internal["metrics"][metric][later_name]

    rng = np.random.default_rng(seed)
    earlier_idx = rng.integers(
        0,
        len(ref_earlier["target_annual"]),
        size=(samples, len(ref_earlier["target_annual"])),
    )
    later_idx = rng.integers(
        0,
        len(ref_later["target_annual"]),
        size=(samples, len(ref_later["target_annual"])),
    )

    target_change = ref_later["target_annual"][later_idx].mean(
        axis=1
    ) - ref_earlier["target_annual"][earlier_idx].mean(axis=1)
    reference_change = ref_later["prediction_annual"][later_idx].mean(
        axis=1
    ) - ref_earlier["prediction_annual"][earlier_idx].mean(axis=1)
    candidate_change = cand_later["prediction_annual"][later_idx].mean(
        axis=1
    ) - cand_earlier["prediction_annual"][earlier_idx].mean(axis=1)
    reference_error = reference_change - target_change
    candidate_error = candidate_change - target_change
    signed_error_difference = candidate_error - reference_error
    absolute_error_improvement = np.abs(reference_error) - np.abs(candidate_error)
    return {
        "signed_error_difference_95ci": _ci(signed_error_difference),
        "absolute_error_improvement_95ci": _ci(absolute_error_improvement),
        "candidate_lower_absolute_error_probability": float(
            np.mean(np.abs(candidate_error) < np.abs(reference_error))
        ),
    }


def compare_models(
    reference_paths: dict[str, list[Path]],
    candidate_paths: dict[str, list[Path]],
    reference_label: str,
    candidate_label: str,
    bootstrap_samples: int,
    seed: int,
) -> tuple[dict, dict, dict]:
    reference_epochs = {name: _load_epoch(reference_paths[name]) for name in EPOCHS}
    candidate_epochs = {name: _load_epoch(candidate_paths[name]) for name in EPOCHS}
    target_check = _verify_common_targets(reference_epochs, candidate_epochs)
    reference_result, reference_internal = analyze_epochs(
        reference_paths, bootstrap_samples, seed
    )
    candidate_result, candidate_internal = analyze_epochs(
        candidate_paths, bootstrap_samples, seed
    )

    result = {
        "reference_model": reference_label,
        "candidate_model": candidate_label,
        "bootstrap_samples": bootstrap_samples,
        "target_check": target_check,
        "metrics": {},
    }
    for metric, (*_, units) in METRICS.items():
        metric_result = {
            "units": units,
            "climatology_bias": {},
            "comparisons": {},
        }
        for epoch in EPOCHS:
            ref_bias = reference_result["metrics"][metric]["climatologies"][epoch][
                "bias"
            ]
            cand_bias = candidate_result["metrics"][metric]["climatologies"][epoch][
                "bias"
            ]
            metric_result["climatology_bias"][epoch] = {
                "reference": ref_bias,
                "candidate": cand_bias,
                "absolute_bias_improvement": abs(ref_bias) - abs(cand_bias),
            }

        for comparison_name in COMPARISONS:
            reference = reference_result["metrics"][metric]["comparisons"][
                comparison_name
            ]
            candidate = candidate_result["metrics"][metric]["comparisons"][
                comparison_name
            ]
            reference_error = reference["signal_error"]
            candidate_error = candidate["signal_error"]
            reference_spatial = reference["spatial_signal"]
            candidate_spatial = candidate["spatial_signal"]
            metric_result["comparisons"][comparison_name] = {
                "target_change": reference["target_change"],
                "reference": {
                    "prediction_change": reference["prediction_change"],
                    "prediction_change_percent": reference[
                        "prediction_change_percent"
                    ],
                    "signal_error": reference_error,
                    "captured_signal_percent": reference["captured_signal_percent"],
                    "spatial_rmse": reference_spatial["rmse"],
                    "spatial_pattern_correlation": reference_spatial[
                        "pattern_correlation"
                    ],
                    "spatial_sign_agreement_fraction": reference_spatial[
                        "sign_agreement_fraction"
                    ],
                },
                "candidate": {
                    "prediction_change": candidate["prediction_change"],
                    "prediction_change_percent": candidate[
                        "prediction_change_percent"
                    ],
                    "signal_error": candidate_error,
                    "captured_signal_percent": candidate["captured_signal_percent"],
                    "spatial_rmse": candidate_spatial["rmse"],
                    "spatial_pattern_correlation": candidate_spatial[
                        "pattern_correlation"
                    ],
                    "spatial_sign_agreement_fraction": candidate_spatial[
                        "sign_agreement_fraction"
                    ],
                },
                "absolute_signal_error_improvement": abs(reference_error)
                - abs(candidate_error),
                "spatial_rmse_improvement": reference_spatial["rmse"]
                - candidate_spatial["rmse"],
                "spatial_correlation_change": candidate_spatial[
                    "pattern_correlation"
                ]
                - reference_spatial["pattern_correlation"],
                "paired_bootstrap": _paired_bootstrap(
                    reference_internal,
                    candidate_internal,
                    metric,
                    comparison_name,
                    bootstrap_samples,
                    seed,
                ),
            }
        result["metrics"][metric] = metric_result
    return result, reference_internal, candidate_internal


def _write_csv(result: dict, path: Path) -> None:
    fields = (
        "metric",
        "units",
        "comparison",
        "target_change",
        "reference_change",
        "candidate_change",
        "reference_signal_error",
        "candidate_signal_error",
        "absolute_signal_error_improvement",
        "absolute_error_improvement_ci_low",
        "absolute_error_improvement_ci_high",
        "candidate_lower_absolute_error_probability",
        "reference_captured_signal_percent",
        "candidate_captured_signal_percent",
        "reference_spatial_rmse",
        "candidate_spatial_rmse",
        "spatial_rmse_improvement",
        "reference_spatial_correlation",
        "candidate_spatial_correlation",
        "spatial_correlation_change",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for metric, metric_values in result["metrics"].items():
            for comparison_name, comparison in metric_values["comparisons"].items():
                reference = comparison["reference"]
                candidate = comparison["candidate"]
                bootstrap = comparison["paired_bootstrap"]
                ci = bootstrap["absolute_error_improvement_95ci"]
                writer.writerow(
                    {
                        "metric": metric,
                        "units": metric_values["units"],
                        "comparison": comparison_name,
                        "target_change": comparison["target_change"],
                        "reference_change": reference["prediction_change"],
                        "candidate_change": candidate["prediction_change"],
                        "reference_signal_error": reference["signal_error"],
                        "candidate_signal_error": candidate["signal_error"],
                        "absolute_signal_error_improvement": comparison[
                            "absolute_signal_error_improvement"
                        ],
                        "absolute_error_improvement_ci_low": ci[0],
                        "absolute_error_improvement_ci_high": ci[1],
                        "candidate_lower_absolute_error_probability": bootstrap[
                            "candidate_lower_absolute_error_probability"
                        ],
                        "reference_captured_signal_percent": reference[
                            "captured_signal_percent"
                        ],
                        "candidate_captured_signal_percent": candidate[
                            "captured_signal_percent"
                        ],
                        "reference_spatial_rmse": reference["spatial_rmse"],
                        "candidate_spatial_rmse": candidate["spatial_rmse"],
                        "spatial_rmse_improvement": comparison[
                            "spatial_rmse_improvement"
                        ],
                        "reference_spatial_correlation": reference[
                            "spatial_pattern_correlation"
                        ],
                        "candidate_spatial_correlation": candidate[
                            "spatial_pattern_correlation"
                        ],
                        "spatial_correlation_change": comparison[
                            "spatial_correlation_change"
                        ],
                    }
                )


def _plot_comparison(result: dict, path: Path) -> None:
    labels = {
        "mean_daily": "Mean daily",
        "sdii": "SDII",
        "rx1day": "Rx1day",
        "rx3h": "Rx3h",
    }
    comparison_labels = {
        "historical_to_mid": "Historical → mid-century",
        "historical_to_end": "Historical → end-century",
    }
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    x = np.arange(len(CORE_METRICS))
    width = 0.36
    for row, comparison_name in enumerate(comparison_labels):
        reference_capture = [
            result["metrics"][metric]["comparisons"][comparison_name]["reference"][
                "captured_signal_percent"
            ]
            for metric in CORE_METRICS
        ]
        candidate_capture = [
            result["metrics"][metric]["comparisons"][comparison_name]["candidate"][
                "captured_signal_percent"
            ]
            for metric in CORE_METRICS
        ]
        axes[row, 0].bar(
            x - width / 2,
            reference_capture,
            width,
            label=result["reference_model"],
        )
        axes[row, 0].bar(
            x + width / 2,
            candidate_capture,
            width,
            label=result["candidate_model"],
        )
        axes[row, 0].axhline(100, color="0.4", linewidth=1)
        axes[row, 0].set_ylabel("Absolute signal captured (%)")
        axes[row, 0].set_title(comparison_labels[comparison_name])
        axes[row, 0].set_xticks(x, [labels[m] for m in CORE_METRICS])
        axes[row, 0].grid(axis="y", alpha=0.25)

        reference_corr = [
            result["metrics"][metric]["comparisons"][comparison_name]["reference"][
                "spatial_pattern_correlation"
            ]
            for metric in CORE_METRICS
        ]
        candidate_corr = [
            result["metrics"][metric]["comparisons"][comparison_name]["candidate"][
                "spatial_pattern_correlation"
            ]
            for metric in CORE_METRICS
        ]
        axes[row, 1].bar(x - width / 2, reference_corr, width)
        axes[row, 1].bar(x + width / 2, candidate_corr, width)
        axes[row, 1].set_ylim(0, 1)
        axes[row, 1].set_ylabel("Spatial signal correlation")
        axes[row, 1].set_title(comparison_labels[comparison_name])
        axes[row, 1].set_xticks(x, [labels[m] for m in CORE_METRICS])
        axes[row, 1].grid(axis="y", alpha=0.25)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=2,
    )
    fig.suptitle("Climate-signal preservation: model comparison", y=1.02)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_error_maps(
    result: dict,
    reference_internal: dict,
    candidate_internal: dict,
    comparison_name: str,
    path: Path,
) -> None:
    earlier_name, later_name = COMPARISONS[comparison_name]
    longitude = reference_internal["longitude"]
    latitude = reference_internal["latitude"]
    fig, axes = plt.subplots(4, 3, figsize=(14, 15), constrained_layout=True)
    for row, metric in enumerate(CORE_METRICS):
        ref_earlier = reference_internal["metrics"][metric][earlier_name]
        ref_later = reference_internal["metrics"][metric][later_name]
        cand_earlier = candidate_internal["metrics"][metric][earlier_name]
        cand_later = candidate_internal["metrics"][metric][later_name]
        target_change = ref_later["target_grid"] - ref_earlier["target_grid"]
        ref_error = (
            ref_later["prediction_grid"] - ref_earlier["prediction_grid"]
        ) - target_change
        cand_error = (
            cand_later["prediction_grid"] - cand_earlier["prediction_grid"]
        ) - target_change
        improvement = np.abs(ref_error) - np.abs(cand_error)
        error_limit = float(
            np.nanpercentile(
                np.abs(np.concatenate([ref_error.ravel(), cand_error.ravel()])), 99
            )
        )
        improvement_limit = float(np.nanpercentile(np.abs(improvement), 99))
        fields = (
            (ref_error, f"{result['reference_model']} error", error_limit),
            (cand_error, f"{result['candidate_model']} error", error_limit),
            (improvement, "Absolute-error improvement", improvement_limit),
        )
        for col, (field, title, limit) in enumerate(fields):
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
            axes[row, col].set_title(f"{metric} — {title}")
            axes[row, col].set_xlabel("Longitude")
            axes[row, col].set_ylabel("Latitude")
            fig.colorbar(mesh, ax=axes[row, col], shrink=0.75)
    fig.suptitle(comparison_name.replace("_", " ").title())
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    for prefix in ("reference", "candidate"):
        parser.add_argument(f"--{prefix}-historical", type=Path, nargs="+", required=True)
        parser.add_argument(f"--{prefix}-mid-century", type=Path, nargs="+", required=True)
        parser.add_argument(f"--{prefix}-end-century", type=Path, nargs="+", required=True)
    parser.add_argument("--reference-label", default="Baseline")
    parser.add_argument("--candidate-label", default="Symmetric 3 h")
    parser.add_argument("--output-prefix", default="baseline_vs_sym3h")
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    reference_paths = {
        "historical": args.reference_historical,
        "mid_century": args.reference_mid_century,
        "end_century": args.reference_end_century,
    }
    candidate_paths = {
        "historical": args.candidate_historical,
        "mid_century": args.candidate_mid_century,
        "end_century": args.candidate_end_century,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result, reference_internal, candidate_internal = compare_models(
        reference_paths,
        candidate_paths,
        args.reference_label,
        args.candidate_label,
        args.bootstrap_samples,
        args.seed,
    )
    prefix = args.output_prefix.replace("/", "_").replace(" ", "_")
    (args.output_dir / f"{prefix}_comparison.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    _write_csv(result, args.output_dir / f"{prefix}_comparison.csv")
    _plot_comparison(result, args.output_dir / f"{prefix}_comparison.png")
    for comparison_name in ("historical_to_mid", "historical_to_end"):
        _plot_error_maps(
            result,
            reference_internal,
            candidate_internal,
            comparison_name,
            args.output_dir / f"{prefix}_error_maps_{comparison_name}.png",
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
