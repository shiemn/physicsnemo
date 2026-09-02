"""Tests for generation timestamp resolution.

Covers helpers.generate_helpers.resolve_times and pins the compacted Taiwan
eval set. That config was a 5,969-entry timestamp dump; it is now the dense
hourly range minus 36 data-gap windows, so the resolved list must be checked
against the enumeration it replaced or the eval set could silently change.
"""

import hashlib
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from helpers.generate_helpers import resolve_times

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
TAIWAN_TIMES = EXAMPLE_ROOT / "conf" / "base" / "times" / "taiwan" / "full_2021_common3h.yaml"

# Digest of "\n".join(times) for the original 5,969-entry enumeration, taken
# from the config before it was rewritten as range + exclusions.
TAIWAN_N = 5969
TAIWAN_SHA256 = "ae9c0ebd167573635f07208af1dc76b4b741f06d33dd932bfb7e1062fc9af6ee"


def test_taiwan_range_matches_original_enumeration():
    times = resolve_times(OmegaConf.load(TAIWAN_TIMES))

    assert len(times) == TAIWAN_N
    assert times == sorted(times)
    assert times[0] == "2021-02-01T03:00:00"
    assert times[-1] == "2021-12-31T20:00:00"
    assert hashlib.sha256("\n".join(times).encode()).hexdigest() == TAIWAN_SHA256


def test_times_range_expands_with_step():
    cfg = OmegaConf.create(
        {"times_range": ["2021-01-01T00:00:00", "2021-01-01T06:00:00", 3]}
    )
    assert resolve_times(cfg) == [
        "2021-01-01T00:00:00",
        "2021-01-01T03:00:00",
        "2021-01-01T06:00:00",
    ]


def test_explicit_times_pass_through():
    stamps = ["2021-01-01T00:00:00", "2021-01-02T00:00:00"]
    assert resolve_times(OmegaConf.create({"times": stamps})) == stamps


def test_exclude_window_is_inclusive_on_both_ends():
    cfg = OmegaConf.create(
        {
            "times_range": ["2021-01-01T00:00:00", "2021-01-01T05:00:00", 1],
            "times_exclude": [["2021-01-01T01:00:00", "2021-01-01T03:00:00"]],
        }
    )
    assert resolve_times(cfg) == [
        "2021-01-01T00:00:00",
        "2021-01-01T04:00:00",
        "2021-01-01T05:00:00",
    ]


def test_exclude_applies_to_explicit_times_too():
    cfg = OmegaConf.create(
        {
            "times": ["2021-01-01T00:00:00", "2021-01-01T01:00:00", "2021-01-01T02:00:00"],
            "times_exclude": [["2021-01-01T01:00:00", "2021-01-01T01:00:00"]],
        }
    )
    assert resolve_times(cfg) == ["2021-01-01T00:00:00", "2021-01-01T02:00:00"]


def test_times_and_times_range_are_mutually_exclusive():
    cfg = OmegaConf.create(
        {
            "times": ["2021-01-01T00:00:00"],
            "times_range": ["2021-01-01T00:00:00", "2021-01-01T01:00:00", 1],
        }
    )
    with pytest.raises(ValueError, match="not both"):
        resolve_times(cfg)


def test_missing_both_sources_raises():
    with pytest.raises(ValueError, match="must be set"):
        resolve_times(OmegaConf.create({}))


def test_malformed_exclude_window_raises():
    cfg = OmegaConf.create(
        {
            "times_range": ["2021-01-01T00:00:00", "2021-01-01T02:00:00", 1],
            "times_exclude": [["2021-01-01T00:00:00"]],
        }
    )
    with pytest.raises(ValueError, match=r"\[start, end\]"):
        resolve_times(cfg)


def test_reversed_exclude_window_raises():
    cfg = OmegaConf.create(
        {
            "times_range": ["2021-01-01T00:00:00", "2021-01-01T02:00:00", 1],
            "times_exclude": [["2021-01-01T02:00:00", "2021-01-01T00:00:00"]],
        }
    )
    with pytest.raises(ValueError, match="start after end"):
        resolve_times(cfg)


def test_excluding_everything_raises():
    cfg = OmegaConf.create(
        {
            "times_range": ["2021-01-01T00:00:00", "2021-01-01T02:00:00", 1],
            "times_exclude": [["2020-01-01T00:00:00", "2022-01-01T00:00:00"]],
        }
    )
    with pytest.raises(ValueError, match="removed every"):
        resolve_times(cfg)
