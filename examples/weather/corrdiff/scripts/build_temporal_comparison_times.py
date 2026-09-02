#!/usr/bin/env python3
"""Build an explicit common eval-time axis for temporal-context comparisons."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from omegaconf import OmegaConf
import yaml

from datasets import cwb
from datasets.dataset import TemporalInputDataset


REQUIRED_OFFSETS = [-12, -6, -3, 0, 3]


def timestamp_key(value) -> str:
    """Return the second-resolution ISO key used by CorrDiff time configs."""

    if isinstance(value, str):
        return value
    return (
        f"{int(value.year):04d}-{int(value.month):02d}-{int(value.day):02d}T"
        f"{int(value.hour):02d}:{int(value.minute):02d}:{int(value.second):02d}"
    )


def select_common_times(requested: Iterable, valid_centers: Iterable) -> list[str]:
    """Preserve requested order while intersecting with union-valid centers."""

    valid = {timestamp_key(value) for value in valid_centers}
    return [key for value in requested if (key := timestamp_key(value)) in valid]


def load_requested_times(path: Path) -> list[str]:
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict) or not isinstance(payload.get("times"), list):
        raise ValueError(f"Expected a YAML mapping with a times list: {path}")
    return [timestamp_key(value) for value in payload["times"]]


def write_time_config(path: Path, times: list[str], source: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# @package generation",
        "# Exact temporal-comparison centers valid at -12, -6, -3, 0, and +3 h.",
        f"# Requested base: {source}",
        f"# Retained centers: {len(times)}",
        "times:",
        *[f"- {value}" for value in times],
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--base-times", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dataset_cfg = OmegaConf.to_container(OmegaConf.load(args.dataset_config), resolve=True)
    if not isinstance(dataset_cfg, dict):
        raise ValueError(f"Expected mapping in {args.dataset_config}")
    dataset_cfg.pop("type", None)
    dataset_cfg["data_path"] = args.data_path
    dataset_cfg["all_times"] = True
    dataset_cfg["train"] = False

    base_dataset = cwb.get_zarr_dataset(**dataset_cfg)
    union_dataset = TemporalInputDataset(
        base_dataset,
        offsets=REQUIRED_OFFSETS,
        boundary="drop",
        strict_time_step_hours=1,
    )
    requested = load_requested_times(args.base_times)
    selected = select_common_times(requested, union_dataset.time())
    if not selected:
        raise RuntimeError("No requested timestamps survive the required-offset intersection")
    write_time_config(args.output, selected, args.base_times)
    print(
        f"retained={len(selected)} requested={len(requested)} "
        f"first={selected[0]} last={selected[-1]} output={args.output}"
    )


if __name__ == "__main__":
    main()
