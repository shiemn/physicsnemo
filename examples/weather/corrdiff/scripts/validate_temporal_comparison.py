#!/usr/bin/env python3
"""Validate a completed temporal-comparison NetCDF/JSON artifact pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import netCDF4 as nc
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--expected-times", type=int, required=True)
    parser.add_argument("--expected-members", type=int, default=10)
    parser.add_argument("--chunk-times", type=int, default=8)
    args = parser.parse_args()

    results = json.loads(Path(args.results).read_text())
    report: dict[str, object] = {
        "predictions": args.predictions,
        "results": args.results,
        "expected_times": args.expected_times,
        "expected_members": args.expected_members,
    }

    with nc.Dataset(args.predictions, "r") as dataset:
        if "truth" not in dataset.groups or "prediction" not in dataset.groups:
            raise RuntimeError("predictions.nc lacks truth/prediction groups")

        truth_group = dataset.groups["truth"]
        prediction_group = dataset.groups["prediction"]
        channels = list(truth_group.variables)
        if channels != list(prediction_group.variables):
            raise RuntimeError("truth and prediction channel lists differ")

        pairwise_distinct = np.zeros(
            (args.expected_members, args.expected_members), dtype=bool
        )
        channel_report: dict[str, object] = {}

        for channel in channels:
            truth_var = truth_group.variables[channel]
            prediction_var = prediction_group.variables[channel]
            if truth_var.shape[0] != args.expected_times:
                raise RuntimeError(f"{channel}: truth time count {truth_var.shape[0]}")
            if prediction_var.shape[:2] != (
                args.expected_members,
                args.expected_times,
            ):
                raise RuntimeError(
                    f"{channel}: prediction leading shape {prediction_var.shape[:2]}"
                )

            pred_min = np.inf
            pred_max = -np.inf
            truth_min = np.inf
            truth_max = -np.inf
            for start in range(0, args.expected_times, args.chunk_times):
                stop = min(start + args.chunk_times, args.expected_times)
                truth = np.asarray(truth_var[start:stop])
                prediction = np.asarray(prediction_var[:, start:stop])
                if not np.isfinite(truth).all():
                    raise RuntimeError(f"{channel}: non-finite truth in {start}:{stop}")
                if not np.isfinite(prediction).all():
                    raise RuntimeError(
                        f"{channel}: non-finite prediction in {start}:{stop}"
                    )
                truth_min = min(truth_min, float(truth.min()))
                truth_max = max(truth_max, float(truth.max()))
                pred_min = min(pred_min, float(prediction.min()))
                pred_max = max(pred_max, float(prediction.max()))
                for first in range(args.expected_members):
                    for second in range(first + 1, args.expected_members):
                        if not pairwise_distinct[first, second] and np.any(
                            prediction[first] != prediction[second]
                        ):
                            pairwise_distinct[first, second] = True

            channel_report[channel] = {
                "truth_shape": list(truth_var.shape),
                "prediction_shape": list(prediction_var.shape),
                "truth_min": truth_min,
                "truth_max": truth_max,
                "prediction_min": pred_min,
                "prediction_max": pred_max,
                "finite": True,
            }

        missing_pairs = [
            [first, second]
            for first in range(args.expected_members)
            for second in range(first + 1, args.expected_members)
            if not pairwise_distinct[first, second]
        ]
        if missing_pairs:
            raise RuntimeError(f"identical ensemble-member pairs: {missing_pairs}")

        # Physical Europe T2M is expressed in Kelvin and is naturally positive,
        # so negativity is not a valid clamp test for that channel. Winds are
        # signed in physical units and must retain negative predictions.
        signed_channels = [
            channel
            for channel in channels
            if channel
            in {"eastward_wind_10m", "northward_wind_10m"}
        ]
        for channel in signed_channels:
            if channel_report[channel]["prediction_min"] >= 0:
                raise RuntimeError(f"{channel}: no negative predictions; possible clamp")

        # The NetCDF intentionally preserves raw denormalized samples. The
        # online and offline metric paths clamp only designated nonnegative
        # channels before scoring, so raw precipitation may remain negative.

        report["channels"] = channel_report
        report["all_member_pairs_distinct"] = True
        report["signed_winds_have_negative_predictions"] = True

    n_sample_keys = [key for key in results if key.endswith("/n_samples")]
    if not n_sample_keys or any(results[key] != args.expected_times for key in n_sample_keys):
        raise RuntimeError("eval_results.json has inconsistent n_samples")
    crps_keys = [key for key in results if key.endswith("/crps")]
    if not crps_keys or not all(np.isfinite(results[key]) for key in crps_keys):
        raise RuntimeError("eval_results.json has missing/non-finite CRPS")
    report["json_n_samples"] = {key: results[key] for key in n_sample_keys}
    report["json_crps"] = {key: results[key] for key in crps_keys}
    report["status"] = "PASS"
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
