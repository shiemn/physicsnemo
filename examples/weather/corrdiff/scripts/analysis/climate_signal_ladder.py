#!/usr/bin/env python3
"""Validate and analyze the matched diffusion climate-signal ladder."""

from __future__ import annotations

import argparse
import calendar
import json
import subprocess
import sys
from pathlib import Path

from validate_climate_chunk import validate_chunk


CLIMATE_ROOT = Path("/outputs/climate_signal")
CHUNKS = (
    ("hist_1986_1995", 1986, 1995),
    ("hist_1996_2005", 1996, 2005),
    ("mid_2041_2050", 2041, 2050),
    ("mid_2051_2060", 2051, 2060),
    ("end_2081_2090", 2081, 2090),
    ("end_2091_2100", 2091, 2100),
)


def _paths(prefix: str) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {
        "historical": [],
        "mid_century": [],
        "end_century": [],
    }
    for chunk, _, _ in CHUNKS:
        epoch = (
            "historical"
            if chunk.startswith("hist_")
            else "mid_century"
            if chunk.startswith("mid_")
            else "end_century"
        )
        result[epoch].append(CLIMATE_ROOT / f"{prefix}_{chunk}" / "climate_chunk.nc")
    return result


def _validate(prefix: str) -> list[dict]:
    reports = []
    for chunk, start, end in CHUNKS:
        years = list(range(start, end + 1))
        days = [364 if calendar.isleap(year) else 363 for year in years]
        reports.append(
            validate_chunk(
                CLIMATE_ROOT / f"{prefix}_{chunk}" / "climate_chunk.nc",
                expected_years=years,
                expected_days_per_year=days,
            )
        )
    return reports


def _run(command: list[str]) -> None:
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def analyze_model(model_key: str, model_label: str, prefix: str) -> None:
    print(json.dumps({"validation": _validate(prefix)}, indent=2, sort_keys=True))
    paths = _paths(prefix)
    _run(
        [
            sys.executable,
            "scripts/analysis/analyze_climate_signal.py",
            "--historical",
            *(str(path) for path in paths["historical"]),
            "--mid-century",
            *(str(path) for path in paths["mid_century"]),
            "--end-century",
            *(str(path) for path in paths["end_century"]),
            "--bootstrap-samples",
            "20000",
            "--seed",
            "20260812",
            "--model-key",
            model_key,
            "--model-label",
            model_label,
            "--output-dir",
            str(CLIMATE_ROOT / "analysis" / f"{model_key}_three_epoch"),
        ]
    )


def compare_ladder() -> None:
    reference = _paths("climate_signal_t0_2m")
    candidates = (
        ("sym3h", "Symmetric 3 h", "climate_signal_sym3h"),
        ("past3h", "Past 3 h", "climate_signal_past3h"),
        ("past12h", "Past 12 h", "climate_signal_past12h"),
    )
    for key, label, prefix in candidates:
        candidate = _paths(prefix)
        _run(
            [
                sys.executable,
                "scripts/analysis/compare_climate_signal_models.py",
                "--reference-historical",
                *(str(path) for path in reference["historical"]),
                "--reference-mid-century",
                *(str(path) for path in reference["mid_century"]),
                "--reference-end-century",
                *(str(path) for path in reference["end_century"]),
                "--candidate-historical",
                *(str(path) for path in candidate["historical"]),
                "--candidate-mid-century",
                *(str(path) for path in candidate["mid_century"]),
                "--candidate-end-century",
                *(str(path) for path in candidate["end_century"]),
                "--reference-label",
                "t0 matched 2M",
                "--candidate-label",
                label,
                "--output-prefix",
                f"t0_2m_vs_{key}",
                "--bootstrap-samples",
                "20000",
                "--seed",
                "20260812",
                "--output-dir",
                str(CLIMATE_ROOT / "analysis" / f"t0_2m_vs_{key}"),
            ]
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    model = subparsers.add_parser("model")
    model.add_argument("--model-key", required=True)
    model.add_argument("--model-label", required=True)
    model.add_argument("--tag-prefix", required=True)
    subparsers.add_parser("compare-ladder")
    args = parser.parse_args()
    if args.command == "model":
        analyze_model(args.model_key, args.model_label, args.tag_prefix)
    else:
        compare_ladder()


if __name__ == "__main__":
    main()
