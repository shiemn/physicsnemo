# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Input/config helpers for the CorrDiff paper-protocol evaluation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import netCDF4 as nc
import numpy as np
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf


@dataclass(frozen=True)
class PredictionFile:
    """One prediction NetCDF source file."""

    path: str
    label: str
    epoch: str
    model: str | None = None


@dataclass(frozen=True)
class TargetSpec:
    """One selected target timestep."""

    label: str
    epoch: str = ""
    kind: str = "target"
    timestamp: str | None = None
    time_idx: int | None = None
    name: str = ""


@dataclass(frozen=True)
class EvalOutputs:
    """Normalized local output layout."""

    root: Path
    json_path: Path
    manifest_path: Path
    tables_dir: Path
    figures_dir: Path
    data_dir: Path


def as_plain_container(value: Any) -> Any:
    if value is None:
        return None
    return OmegaConf.to_container(value, resolve=True)


def format_timestamp(value) -> str:
    """Format netCDF/cftime/datetime timestamps for stable config matching."""
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    text = str(value)
    if "T" in text:
        text = text.replace("T", " ")
    return text.split(".")[0]


def normalize_timestamp(value) -> str:
    text = str(value).strip().replace("T", " ")
    if len(text) == 16:
        text += ":00"
    return text.split(".")[0]


def resolve_channel(channel_names: list[str], selection) -> int:
    """Resolve the precipitation channel index from a name, int, or 'auto'."""
    if isinstance(selection, int):
        return selection
    if isinstance(selection, str) and selection.isdigit():
        return int(selection)
    if isinstance(selection, str) and selection != "auto":
        for i, name in enumerate(channel_names):
            if name == selection:
                return i
        raise ValueError(f"channel {selection!r} not found in {channel_names}")
    for i, name in enumerate(channel_names):
        norm = "".join(c for c in name.lower() if c.isalnum())
        if any(tok in norm for tok in ["precip", "rain", "tp", "rr"]):
            return i
    return 0


class NetCDFStream:
    """Streaming reader for the prediction NetCDF written by evaluate.py."""

    def __init__(self, path: str, channel_selection):
        self.path = path
        self.f = nc.Dataset(path, "r")
        if "truth" not in self.f.groups or "prediction" not in self.f.groups:
            self.f.close()
            raise ValueError(
                f"{path} is missing required groups (truth/prediction); "
                "it may be corrupt or from a crashed run."
            )
        self.truth = self.f.groups["truth"]
        self.pred = self.f.groups["prediction"]
        self.channel_names = list(self.truth.variables.keys())
        self.ch = resolve_channel(self.channel_names, channel_selection)
        self.channel_name = self.channel_names[self.ch]
        var = self.truth[self.channel_name]
        self.n_times = var.shape[0]
        self.img_shape = (var.shape[-2], var.shape[-1])
        pvar = self.pred[self.channel_name]
        self.n_ensemble = pvar.shape[0]
        try:
            self.lat = np.array(self.f["lat"][:])
            self.lon = np.array(self.f["lon"][:])
        except (IndexError, KeyError):
            self.lat = None
            self.lon = None
        self.times = self._read_times()

    def _read_times(self) -> list[str]:
        try:
            tvar = self.f["time"]
        except (IndexError, KeyError):
            return [str(i) for i in range(self.n_times)]
        units = getattr(tvar, "units", None)
        calendar = getattr(tvar, "calendar", "standard")
        if units is None:
            return [str(x) for x in np.asarray(tvar[:])]
        vals = nc.num2date(
            tvar[:],
            units=units,
            calendar=calendar,
            only_use_cftime_datetimes=False,
            only_use_python_datetimes=False,
        )
        return [format_timestamp(v) for v in vals]

    def target(self, t: int) -> np.ndarray:
        return np.clip(np.asarray(self.truth[self.channel_name][t]), 0.0, None)

    def pred_ens(self, t: int) -> np.ndarray:
        return np.clip(np.asarray(self.pred[self.channel_name][:, t]), 0.0, None)

    def close(self) -> None:
        self.f.close()


def _read_target_set(path: str | None) -> list[dict]:
    if not path:
        return []
    abs_path = Path(to_absolute_path(str(path)))
    if not abs_path.is_file():
        raise FileNotFoundError(f"eval.target_set not found: {abs_path}")
    data = OmegaConf.load(abs_path)
    plain = as_plain_container(data)
    if isinstance(plain, dict):
        plain = plain.get("targets", [])
    if not isinstance(plain, list):
        raise ValueError("target_set must be a list or mapping with a 'targets' list")
    return plain


def prediction_file_entries(cfg) -> list[dict]:
    """Return normalized prediction file entries.

    Supports new ``eval.inputs`` and legacy ``eval.prediction_files``.
    """

    entries_cfg = as_plain_container(cfg.eval.get("inputs", None))
    if not entries_cfg:
        entries_cfg = as_plain_container(cfg.eval.get("prediction_files", None))
    if not entries_cfg:
        return []
    if not isinstance(entries_cfg, list):
        raise ValueError("eval.inputs/eval.prediction_files must be a list of mappings")
    entries = []
    for i, item in enumerate(entries_cfg):
        if not isinstance(item, dict):
            raise ValueError(f"prediction input {i} must be a mapping")
        path = item.get("path")
        if not path:
            raise ValueError(f"prediction input {i} is missing 'path'")
        label = str(item.get("label") or Path(str(path)).parent.name or f"file_{i}")
        epoch = str(item.get("epoch") or label)
        model = item.get("model")
        abs_path = to_absolute_path(str(path))
        if not os.path.isfile(abs_path):
            raise FileNotFoundError(f"prediction input {i} path not found: {abs_path}")
        out = {"epoch": epoch, "label": label, "path": abs_path}
        if model is not None:
            out["model"] = str(model)
        entries.append(out)
    return entries


def target_entries(cfg) -> list[dict]:
    """Return normalized target entries from new target_set and legacy eval.targets."""

    targets_cfg = []
    target_set = cfg.eval.get("target_set", None)
    targets_cfg.extend(_read_target_set(target_set))
    inline = as_plain_container(cfg.eval.get("targets", None))
    if inline:
        if not isinstance(inline, list):
            raise ValueError("eval.targets must be a list of mappings")
        targets_cfg.extend(inline)
    if not targets_cfg:
        return []
    targets = []
    for i, item in enumerate(targets_cfg):
        if not isinstance(item, dict):
            raise ValueError(f"target {i} must be a mapping")
        label = item.get("label")
        if not label:
            raise ValueError(f"target {i} is missing 'label'")
        if "time_idx" not in item and "timestamp" not in item:
            raise ValueError(f"target {i} must provide time_idx or timestamp")
        target = {
            "epoch": str(item.get("epoch") or ""),
            "kind": str(item.get("kind") or "target"),
            "label": str(label),
            "name": str(item.get("name") or ""),
        }
        if "time_idx" in item and item["time_idx"] is not None:
            target["time_idx"] = int(item["time_idx"])
        if "timestamp" in item and item["timestamp"] is not None:
            target["timestamp"] = normalize_timestamp(item["timestamp"])
        targets.append(target)
    return targets


def stream_target_index(stream: NetCDFStream, target: dict) -> int:
    """Resolve a target entry against an open stream."""
    if "time_idx" in target:
        idx = int(target["time_idx"])
        if idx < 0 or idx >= stream.n_times:
            raise IndexError(
                f"target {target['label']} time_idx={idx} outside [0, {stream.n_times})"
            )
        if "timestamp" in target:
            expected = normalize_timestamp(target["timestamp"])
            actual = normalize_timestamp(stream.times[idx])
            if actual != expected:
                raise ValueError(
                    f"target {target['label']} time_idx={idx} timestamp mismatch: "
                    f"expected {expected}, file has {actual}"
                )
        return idx
    expected = normalize_timestamp(target["timestamp"])
    matches = [i for i, ts in enumerate(stream.times) if normalize_timestamp(ts) == expected]
    if not matches:
        raise ValueError(f"target timestamp {expected} not found in {target['label']}")
    if len(matches) > 1:
        raise ValueError(f"target timestamp {expected} is ambiguous in {target['label']}")
    return matches[0]


def eval_outputs(cfg, run_tag: str) -> EvalOutputs:
    outputs_cfg = as_plain_container(cfg.eval.get("outputs", None)) or {}
    root = outputs_cfg.get("root")
    legacy_json = cfg.eval.get("output_json", None)
    if root is None:
        if legacy_json and str(legacy_json) != "paper_eval_results.json":
            root = str(Path(to_absolute_path(str(legacy_json))).parent)
        else:
            root = f"/outputs/paper_eval/{run_tag}" if str(run_tag) else "paper_eval"
    root_path = Path(to_absolute_path(str(root)))
    json_name = outputs_cfg.get("json")
    if json_name is None and legacy_json and str(legacy_json) != "paper_eval_results.json":
        json_name = Path(str(legacy_json)).name
    if json_name is None:
        json_name = "paper_eval_results.json"
    manifest_name = outputs_cfg.get("manifest", "manifest.json")
    return EvalOutputs(
        root=root_path,
        json_path=root_path / str(json_name),
        manifest_path=root_path / str(manifest_name),
        tables_dir=root_path / str(outputs_cfg.get("tables", "tables")),
        figures_dir=root_path / str(outputs_cfg.get("figures", "figures")),
        data_dir=root_path / str(outputs_cfg.get("data", "data")),
    )
