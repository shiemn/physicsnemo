#!/usr/bin/env python3
"""Generate a reproducible held-out Taiwan regression-evaluation time config.

Run this in an Apptainer CPU job on Helma, where the CWB Zarr metadata is
available. It samples only timestamps for which CWB and every required ERA5
channel are valid, then excludes timestamps listed in an existing time config.
The generated YAML is an explicit evaluation protocol and should be reviewed
and committed before it is used by Hydra.

From ``~/corrdiffProjektSimon/code/examples/weather/corrdiff`` on Helma, a
short CPU job can write a proposed config outside the checkout:

    sbatch --job-name=taiwan-times --partition=preempt_cpu --nodes=1 \\
      --ntasks=1 --cpus-per-task=2 --time=00:15:00 \\
      --output=/hnvme/workspace/b214cb11-helma-ecodata/downscaling/daniel/outputs/ablation_1_and_2_models/slurm/taiwan-times-%j.out \\
      --wrap="srun --ntasks=1 --cpus-per-task=2 apptainer exec \\
        --bind $HOME/corrdiffProjektSimon/code/examples/weather/corrdiff:/workspace/corrdiff \\
        --bind /hnvme/workspace/b214cb11-helma-ecodata/downscaling/CorrDiff:/data/CorrDiff \\
        --bind /hnvme/workspace/b214cb11-helma-ecodata/downscaling/daniel/outputs/ablation_1_and_2_models:/outputs \\
        /hnvme/workspace/b214cb11-helma-ecodata/downscaling/apptainer/corrdiff_ngc.sif \\
        python /workspace/corrdiff/scripts/generate_taiwan_holdout_times.py \\
          --dataset-config /workspace/corrdiff/conf/base/dataset/cwb.yaml \\
          --data-path /data/CorrDiff/cwa_dataset.zarr.zip \\
          --exclude-times /workspace/corrdiff/conf/base/times/taiwan256.yaml \\
          --output /outputs/analysis/variance_inequality/taiwan1024_seed20260930_holdout.yaml"

Copy the proposed YAML to the local checkout, inspect it, then commit it as
``conf/base/times/taiwan1024_seed20260930_holdout.yaml``. Do not use the
uncommitted HNVME copy directly for an evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

from omegaconf import OmegaConf
import yaml


_CORRDIFF_ROOT = Path(__file__).resolve().parents[1]
if str(_CORRDIFF_ROOT) not in sys.path:
    sys.path.insert(0, str(_CORRDIFF_ROOT))

from datasets import cwb


def timestamp_key(value) -> str:
    """Return the second-resolution ISO representation used by time configs."""

    if isinstance(value, str):
        return value
    return (
        f"{int(value.year):04d}-{int(value.month):02d}-{int(value.day):02d}T"
        f"{int(value.hour):02d}:{int(value.minute):02d}:{int(value.second):02d}"
    )


def load_times(path: Path) -> set[str]:
    """Load the explicit timestamp list from a Hydra time config."""

    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict) or not isinstance(payload.get("times"), list):
        raise ValueError(f"Expected a YAML mapping with a times list: {path}")
    times = [timestamp_key(value) for value in payload["times"]]
    if len(times) != len(set(times)):
        raise ValueError(f"Time config contains duplicate timestamps: {path}")
    return set(times)


def load_valid_times(dataset_config: Path, data_path: str, year: int) -> list[str]:
    """Read valid CWB plus all-ERA5 timestamps for one calendar year."""

    dataset_cfg = OmegaConf.to_container(OmegaConf.load(dataset_config), resolve=True)
    if not isinstance(dataset_cfg, dict):
        raise ValueError(f"Expected a mapping in dataset config: {dataset_config}")
    dataset_cfg.pop("type", None)
    dataset_cfg["data_path"] = data_path
    dataset_cfg["all_times"] = True
    dataset_cfg["train"] = False

    dataset = cwb.get_zarr_dataset(**dataset_cfg)
    return sorted(
        timestamp
        for value in dataset.time()
        if (timestamp := timestamp_key(value)).startswith(f"{year:04d}-")
    )


def write_config(
    path: Path,
    selected: list[str],
    *,
    year: int,
    seed: int,
    candidate_count: int,
    excluded_label: str,
    excluded_count: int,
) -> None:
    """Write a fixed Hydra generation.times config and provenance header."""

    lines = [
        "# @package generation",
        f"# {len(selected)} random hourly timesteps from {year} valid CWB + ERA5 data (seed={seed}).",
        f"# Sampled without replacement from {candidate_count} valid hours after excluding {excluded_count} times.",
        f"# Excluded config: {excluded_label}",
        "times:",
        *[f"- {timestamp}" for timestamp in selected],
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--exclude-times", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--year", type=int, default=2021)
    parser.add_argument("--exclude-label", default="conf/base/times/taiwan256.yaml")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.count < 1:
        raise ValueError("count must be positive")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing output: {args.output}. Use --overwrite explicitly."
        )

    valid_times = load_valid_times(args.dataset_config, args.data_path, args.year)
    valid_set = set(valid_times)
    excluded = load_times(args.exclude_times)
    missing = excluded - valid_set
    if missing:
        example = min(missing)
        raise ValueError(f"Excluded config has timestamps absent from valid data, e.g. {example}")

    candidates = sorted(valid_set - excluded)
    if len(candidates) < args.count:
        raise ValueError(
            f"Need {args.count} unobserved candidates, found only {len(candidates)}"
        )
    selected = sorted(random.Random(args.seed).sample(candidates, args.count))
    write_config(
        args.output,
        selected,
        year=args.year,
        seed=args.seed,
        candidate_count=len(candidates),
        excluded_label=args.exclude_label,
        excluded_count=len(excluded),
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "year": args.year,
                "seed": args.seed,
                "valid_times": len(valid_times),
                "excluded_times": len(excluded),
                "candidate_times": len(candidates),
                "selected_times": len(selected),
                "first": selected[0],
                "last": selected[-1],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
