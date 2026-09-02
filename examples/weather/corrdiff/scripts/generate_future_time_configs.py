#!/usr/bin/env python3
"""Generate Norway future-period eval timestamp configs.

The generated samples are valid for temporal contexts up to +/-24h on a
3-hourly dataset. They intentionally use common 512-sample sets per period so
t0 and temporal models can be compared on identical timestamps. The original
seed-42 configs keep their historical filenames; extra seed-specific configs
support confidence estimates over repeated random draws.

Writes conf/base/times/future_confidence/*.yaml (30 files). That output is
committed -- it is the eval protocol and is worth pinning -- so this script is
the source of truth for it. tests/test_future_time_configs.py asserts the two
stay in sync; re-run this script and commit the result if you change a
parameter below.
"""

from __future__ import annotations

import datetime as dt
import random
from pathlib import Path


N_SAMPLES = 512
STEP_HOURS = 3
CONTEXT_HOURS = 24
CONFIGS = [
    (
        "random512_current_2005_24h_compatible.yaml",
        "2005 current/reference period",
        [2005],
    ),
    (
        "random512_current_2004_2005_24h_compatible.yaml",
        "2004-2005 current/reference period",
        [2004, 2005],
    ),
    (
        "random512_midcentury_start_2041_2042_24h_compatible.yaml",
        "2041-2042 mid-century start",
        [2041, 2042],
    ),
    (
        "random512_midcentury_end_2059_2060_24h_compatible.yaml",
        "2059-2060 mid-century end",
        [2059, 2060],
    ),
    (
        "random512_endcentury_start_2081_2082_24h_compatible.yaml",
        "2081-2082 end-century start",
        [2081, 2082],
    ),
    (
        "random512_endcentury_end_2099_2100_24h_compatible.yaml",
        "2099-2100 end-century end",
        [2099, 2100],
    ),
]
EXTRA_SEEDS = [43, 44, 45, 46]


def year_times(year: int) -> list[dt.datetime]:
    start = dt.datetime(year, 1, 1)
    end = dt.datetime(year + 1, 1, 1) - dt.timedelta(hours=STEP_HOURS)
    out = []
    cur = start
    step = dt.timedelta(hours=STEP_HOURS)
    while cur < end:
        out.append(cur)
        cur += step
    return out


def compatible_sample(years: list[int], seed: int) -> list[dt.datetime]:
    times = [time for year in years for time in year_times(year)]
    available = set(times)
    offsets = range(-CONTEXT_HOURS, CONTEXT_HOURS + STEP_HOURS, STEP_HOURS)
    candidates = [
        time
        for time in times
        if all(time + dt.timedelta(hours=offset) in available for offset in offsets)
    ]
    if len(candidates) < N_SAMPLES:
        raise ValueError(f"Need {N_SAMPLES} candidates for {years}, got {len(candidates)}")
    rng = random.Random(seed)
    return sorted(rng.sample(candidates, N_SAMPLES))


def write_config(path: Path, label: str, years: list[int], seed: int) -> None:
    sample = compatible_sample(years, seed)
    first_context = sample[0] - dt.timedelta(hours=CONTEXT_HOURS)
    last_context = sample[-1] + dt.timedelta(hours=CONTEXT_HOURS)
    lines = [
        "# @package generation",
        f"# {N_SAMPLES} random 3-hourly center timesteps from {label} (seed={seed}).",
        f"# Valid for temporal contexts up to +/-{CONTEXT_HOURS}h.",
        f"# Context span covered by selected centers: {first_context:%Y-%m-%dT%H:%M:%S} to {last_context:%Y-%m-%dT%H:%M:%S}.",
        "times:",
    ]
    lines.extend(f"- {time:%Y-%m-%dT%H:%M:%S}" for time in sample)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    out_dir = (
        Path(__file__).resolve().parents[1]
        / "conf"
        / "base"
        / "times"
        / "future_confidence"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    for filename, label, years in CONFIGS:
        write_config(out_dir / filename, label, years, seed=42)
        stem = filename.removesuffix("_24h_compatible.yaml")
        for seed in EXTRA_SEEDS:
            write_config(
                out_dir / f"{stem}_seed{seed}_24h_compatible.yaml",
                label,
                years,
                seed=seed,
            )


if __name__ == "__main__":
    main()
