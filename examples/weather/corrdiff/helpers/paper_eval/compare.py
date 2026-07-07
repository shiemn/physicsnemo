#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare paper-protocol outputs across multiple CorrDiff models.

This is an offline helper. It consumes completed ``evaluate_paper.py`` outputs:

* ``paper_eval_results.json`` for metric-derived comparison tables/figures.
* retained W&B PNG media for spatial/diagnostic contact sheets.

It does not regenerate predictions or re-run the paper evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

from helpers.paper_eval.maps import plot_map_grid
from helpers.paper_eval.styles import get_style


DEFAULT_MODEL_SPECS = [
    (
        "t0",
        "t0",
        "#6f6f6f",
        "/outputs/paper_eval/future_conf_s43_t0_epoch_balanced_paper",
    ),
    (
        "past12h",
        "past12h",
        "#2878b5",
        "/outputs/paper_eval/future_conf_s43_past12h_epoch_balanced_paper",
    ),
    (
        "sym3h",
        "sym3h",
        "#2a9d55",
        "/outputs/paper_eval/future_conf_s43_sym3h_epoch_balanced_paper",
    ),
]

EPOCH_ORDER = ["current", "mid", "end"]
CLIM_GROUPS = ["all_periods", "current", "mid", "end"]
CLIM_STATS = ["dry_pct", "mean", "sd", "median", "p99"]
CLIM_WINDOWS = ["raw", "3x3"]

CONTACT_SHEET_KEYS = [
    "climatology/all_periods/bias_mean",
    "climatology/all_periods/bias_sd",
    "climatology/all_periods/bias_dry_pct",
    "climatology/all_periods/rmse_map",
    "climatology/all_periods/qq",
    "climatology/all_periods/rapsd",
    "climatology/current/bias_mean",
    "climatology/current/bias_sd",
    "climatology/current/bias_dry_pct",
    "climatology/current/rmse_map",
    "climatology/current/qq",
    "climatology/current/rapsd",
    "climatology/mid/bias_mean",
    "climatology/mid/bias_sd",
    "climatology/mid/bias_dry_pct",
    "climatology/mid/rmse_map",
    "climatology/mid/qq",
    "climatology/mid/rapsd",
    "climatology/end/bias_mean",
    "climatology/end/bias_sd",
    "climatology/end/bias_dry_pct",
    "climatology/end/rmse_map",
    "climatology/end/qq",
    "climatology/end/rapsd",
    "targets/combined/crps_maps",
    "targets/combined/out_of_envelope",
    "targets/current/crps_maps",
    "targets/current/out_of_envelope",
    "targets/mid/crps_maps",
    "targets/mid/out_of_envelope",
    "targets/end/crps_maps",
    "targets/end/out_of_envelope",
    "targets/sal_scatter_by_epoch",
    "targets/sal_case_grid",
]

MERGEABLE_IMAGE_BASENAMES = {
    "bias_mean",
    "bias_sd",
    "bias_dry_pct",
    "rmse_map",
    "crps_maps",
    "out_of_envelope",
}


@dataclass(frozen=True)
class ModelRun:
    """One completed paper-eval model run."""

    model_id: str
    label: str
    color: str
    result_dir: Path
    image_dir: Path | None = None

    @property
    def json_path(self) -> Path:
        return self.result_dir / "paper_eval_results.json"


def parse_model_spec(raw: str) -> ModelRun:
    """Parse ``id=label=color:result_dir[:image_dir]``.

    The first three separators are structural. ``result_dir`` and ``image_dir``
    may contain additional colons only on platforms that support them poorly;
    this project runs on POSIX paths where that is fine.
    """

    if "=" not in raw or ":" not in raw:
        raise ValueError(
            "Model spec must look like id=label:color:result_dir[:image_dir]"
        )
    model_id, rest = raw.split("=", 1)
    parts = rest.split(":")
    if len(parts) < 3:
        raise ValueError(
            "Model spec must include label, color, and result_dir: "
            "id=label:color:result_dir[:image_dir]"
        )
    label, color, result_dir = parts[:3]
    image_dir = ":".join(parts[3:]) if len(parts) > 3 else None
    return ModelRun(
        model_id=model_id,
        label=label,
        color=color,
        result_dir=Path(result_dir),
        image_dir=Path(image_dir) if image_dir else None,
    )


def default_model_runs() -> list[ModelRun]:
    return [
        ModelRun(model_id=m, label=label, color=color, result_dir=Path(path))
        for m, label, color, path in DEFAULT_MODEL_SPECS
    ]


def model_run_from_mapping(item: dict) -> ModelRun:
    missing = [key for key in ["id", "label", "color", "result_dir"] if key not in item]
    if missing:
        raise ValueError(f"compare model mapping is missing required keys: {missing}")
    return ModelRun(
        model_id=str(item["id"]),
        label=str(item["label"]),
        color=str(item["color"]),
        result_dir=Path(str(item["result_dir"])),
        image_dir=Path(str(item["image_dir"])) if item.get("image_dir") else None,
    )


def _pop_config_name(argv: list[str]) -> tuple[str | None, list[str]]:
    config_name = None
    rest: list[str] = []
    skip_next = False
    for idx, arg in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if arg.startswith("--config-name="):
            config_name = arg.split("=", 1)[1]
        elif arg == "--config-name":
            if idx + 1 >= len(argv):
                raise ValueError("--config-name requires a value")
            config_name = argv[idx + 1]
            skip_next = True
        else:
            rest.append(arg)
    return config_name, rest


def load_compare_config(config_name: str) -> tuple[list[ModelRun], Path]:
    path = Path(config_name)
    if path.suffix not in {".yaml", ".yml"}:
        path = Path("conf") / f"{config_name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"comparison config not found: {path}")
    cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    compare = cfg.get("compare") if isinstance(cfg, dict) else None
    if not isinstance(compare, dict):
        raise ValueError(f"{path} must contain a compare: mapping")
    models_cfg = compare.get("models")
    if not isinstance(models_cfg, list) or not models_cfg:
        raise ValueError(f"{path} compare.models must be a non-empty list")
    output_dir = Path(str(compare.get("output_dir", "/outputs/paper_eval/model_comparison")))
    return [model_run_from_mapping(item) for item in models_cfg], output_dir


def load_results(model: ModelRun) -> dict:
    with model.json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def _candidate_image_roots(model: ModelRun) -> list[Path]:
    roots: list[Path] = []
    if model.image_dir is not None:
        roots.append(model.image_dir)
    roots.append(model.result_dir)
    roots.extend(
        sorted(model.result_dir.glob("wandb/wandb/run-*/files/media/images"))
    )
    roots.extend(sorted(model.result_dir.glob("wandb/run-*/files/media/images")))
    return roots


def semantic_to_filename_prefix(key: str) -> str:
    return key.replace("/", "_")


def find_semantic_png(model: ModelRun, key: str) -> Path | None:
    """Find a W&B PNG whose semantic path matches ``key``.

    Supports both nested W&B media paths
    ``climatology/all_periods/bias_mean_0_hash.png`` and flat exported names
    ``climatology_all_periods_bias_mean.png``.
    """

    basename = key.split("/")[-1]
    flat = semantic_to_filename_prefix(key)
    for root in _candidate_image_roots(model):
        if not root.exists():
            continue
        nested = sorted(root.glob(f"{key}_*.png")) + sorted(root.glob(f"{key}.png"))
        if nested:
            return nested[0]
        flat_matches = sorted(root.rglob(f"{flat}*.png"))
        if flat_matches:
            return flat_matches[0]
        loose = [
            p for p in root.rglob(f"{basename}*.png")
            if semantic_to_filename_prefix(str(p.relative_to(root))).startswith(flat)
            or key in str(p.relative_to(root))
        ]
        if loose:
            return sorted(loose)[0]
    return None


def metric_value(row: dict, key: str) -> float:
    value = row.get(key)
    if isinstance(value, dict):
        value = value.get("mean", value.get("median"))
    if value is None:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def case_id(row: dict, counters: dict[tuple[str, str], int]) -> str:
    existing = row.get("case_id")
    if existing:
        return str(existing)
    epoch = str(row.get("epoch", "target")).upper()[:3]
    kind = str(row.get("kind", "target")).lower()
    letter = "E" if kind == "extreme" else "N"
    key = (epoch, letter)
    counters[key] = counters.get(key, 0) + 1
    return f"{epoch}-{letter}{counters[key]}"


def climatology_rows(models: list[ModelRun], results: dict[str, dict]) -> list[dict]:
    rows: list[dict] = []
    for model in models:
        clim = results[model.model_id].get("climatology", {})
        for group in CLIM_GROUPS:
            group_data = clim.get(group, {})
            for row in group_data.get("table", []):
                rows.append({
                    "model": model.model_id,
                    "label": model.label,
                    "group": group,
                    "n_times": group_data.get("n_times"),
                    "window": row.get("window"),
                    "statistic": row.get("statistic"),
                    "model_value": row.get("model"),
                    "reference": row.get("reference"),
                    "rel_bias_pct": row.get("rel_bias_pct"),
                })
    return rows


def target_rows(models: list[ModelRun], results: dict[str, dict]) -> list[dict]:
    rows: list[dict] = []
    for model in models:
        counters: dict[tuple[str, str], int] = {}
        targets = results[model.model_id].get("targets", {}).get("per_target", [])
        for row in targets:
            cid = case_id(row, counters)
            rows.append({
                "model": model.model_id,
                "label": model.label,
                "case_id": cid,
                "epoch": row.get("epoch"),
                "kind": row.get("kind"),
                "timestamp": row.get("timestamp"),
                "max_precip_mm": row.get("max_precip_mm"),
                "crps_mean": row.get("crps_mean"),
                "S": metric_value(row, "S"),
                "A": metric_value(row, "A"),
                "L": metric_value(row, "L"),
            })
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _bias_lookup(rows: Iterable[dict], model: str, group: str, window: str, stat: str) -> float:
    for row in rows:
        if (
            row["model"] == model
            and row["group"] == group
            and row["window"] == window
            and row["statistic"] == stat
        ):
            return float(row["rel_bias_pct"])
    return math.nan


def plot_climatology_bias_summary(
    models: list[ModelRun],
    rows: list[dict],
    out_dir: Path,
    *,
    include_bars: bool = False,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    x = np.arange(len(CLIM_STATS))
    width = 0.74 / max(len(models), 1)
    for group in CLIM_GROUPS:
        for window in CLIM_WINDOWS:
            if include_bars:
                fig, ax = plt.subplots(figsize=(10, 5.5))
                for i, model in enumerate(models):
                    vals = [
                        _bias_lookup(rows, model.model_id, group, window, stat)
                        for stat in CLIM_STATS
                    ]
                    offset = (i - (len(models) - 1) / 2) * width
                    ax.bar(
                        x + offset, vals, width=width, color=model.color,
                        label=model.label, alpha=0.9,
                    )
                ax.axhline(0.0, color="black", lw=1.0)
                ax.set_xticks(x, CLIM_STATS)
                ax.set_ylabel("Relative bias (%)")
                ax.set_title(
                    f"Future-Confidence Paper Eval: {group.replace('_', ' ').title()} "
                    f"Climatology Bias ({window})",
                    fontweight="bold",
                )
                ax.grid(axis="y", alpha=0.25)
                ax.legend(ncols=len(models), fontsize=9)
                fig.tight_layout()
                path = out_dir / f"climatology_{group}_bias_{window}_models.png"
                fig.savefig(path, dpi=170, bbox_inches="tight")
                plt.close(fig)
                written.append(path)

            fig, ax = plt.subplots(figsize=(10, 5.5))
            for model in models:
                vals = [
                    _bias_lookup(rows, model.model_id, group, window, stat)
                    for stat in CLIM_STATS
                ]
                ax.plot(
                    x,
                    vals,
                    marker="o",
                    color=model.color,
                    label=model.label,
                    linewidth=2.2,
                    markersize=6.0,
                )
            ax.axhline(0.0, color="black", lw=1.0)
            ax.set_xticks(x, CLIM_STATS)
            ax.set_ylabel("Relative bias (%)")
            ax.set_title(
                f"Future-Confidence Paper Eval: {group.replace('_', ' ').title()} "
                f"Climatology Bias Lines ({window})",
                fontweight="bold",
            )
            ax.grid(True, alpha=0.25)
            ax.legend(ncols=len(models), fontsize=9)
            fig.tight_layout()
            path = out_dir / f"climatology_{group}_bias_{window}_lines_models.png"
            fig.savefig(path, dpi=170, bbox_inches="tight")
            plt.close(fig)
            written.append(path)

    for window in CLIM_WINDOWS:
        fig, axes = plt.subplots(
            1, len(CLIM_STATS), figsize=(4.1 * len(CLIM_STATS), 4.6),
            sharex=True, squeeze=False,
        )
        axes = list(axes[0])
        group_x = np.arange(len(CLIM_GROUPS))
        for ax, stat in zip(axes, CLIM_STATS):
            for model in models:
                vals = [
                    _bias_lookup(rows, model.model_id, group, window, stat)
                    for group in CLIM_GROUPS
                ]
                ax.plot(
                    group_x,
                    vals,
                    marker="o",
                    color=model.color,
                    label=model.label,
                    linewidth=2.0,
                    markersize=5.5,
                )
            ax.axhline(0.0, color="black", lw=0.9)
            ax.set_title(stat, fontweight="bold")
            ax.set_xticks(group_x, [g.replace("_", " ") for g in CLIM_GROUPS], rotation=25, ha="right")
            ax.grid(True, alpha=0.25)
        axes[0].set_ylabel("Relative bias (%)")
        axes[-1].legend(loc="best", fontsize=8)
        fig.suptitle(
            f"Future-Confidence Paper Eval: Climatology Bias Across Periods ({window})",
            fontweight="bold",
        )
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        path = out_dir / f"climatology_bias_by_period_{window}_lines_models.png"
        fig.savefig(path, dpi=170, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
    return written


def plot_target_crps(models: list[ModelRun], rows: list[dict], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    case_order = []
    for row in rows:
        if row["case_id"] not in case_order:
            case_order.append(row["case_id"])
    x = np.arange(len(case_order))
    width = 0.74 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(13, 5.8))
    for model in models:
        vals = []
        for cid in case_order:
            match = [
                float(r["crps_mean"]) for r in rows
                if r["model"] == model.model_id and r["case_id"] == cid
            ]
            vals.append(match[0] if match else math.nan)
        ax.plot(
            x, vals, marker="o", color=model.color, label=model.label,
            linewidth=2.2, markersize=6.0,
        )
    ax.set_xticks(x, case_order, rotation=35, ha="right")
    ax.set_ylabel("Mean CRPS")
    ax.set_title(
        "Future-Confidence Paper Eval: Target Case CRPS Lines",
        fontweight="bold",
    )
    ax.grid(True, alpha=0.25)
    ax.legend(ncols=len(models), fontsize=9)
    fig.tight_layout()
    path = out_dir / "targets_crps_by_case_lines_models.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    groups = ["all", "extreme", "normal", *EPOCH_ORDER]
    fig, ax = plt.subplots(figsize=(10, 5.4))
    x = np.arange(len(groups))
    for model in models:
        vals = []
        for group in groups:
            selected = [
                float(r["crps_mean"]) for r in rows
                if r["model"] == model.model_id
                and (
                    group == "all"
                    or r["kind"] == group
                    or r["epoch"] == group
                )
            ]
            vals.append(float(np.mean(selected)) if selected else math.nan)
        ax.plot(
            x, vals, marker="o", color=model.color, label=model.label,
            linewidth=2.2, markersize=6.0,
        )
    ax.set_xticks(x, groups)
    ax.set_ylabel("Mean CRPS")
    ax.set_title(
        "Future-Confidence Paper Eval: Target CRPS Summary Lines",
        fontweight="bold",
    )
    ax.grid(True, alpha=0.25)
    ax.legend(ncols=len(models), fontsize=9)
    fig.tight_layout()
    path = out_dir / "targets_crps_summary_lines_models.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    written.append(path)
    return written


def plot_sal_comparison(models: list[ModelRun], rows: list[dict], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    finite_l = [float(r["L"]) for r in rows if np.isfinite(float(r["L"]))]
    vmax = max(float(np.percentile(finite_l, 95)), 0.05) if finite_l else 1.0
    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    size_min, size_max = 54.0, 165.0
    for model in models:
        model_rows = [r for r in rows if r["model"] == model.model_id]
        for kind, marker in [("extreme", "^"), ("normal", "o")]:
            selected = [r for r in model_rows if r["kind"] == kind]
            if not selected:
                continue
            sizes = [
                size_min + (size_max - size_min) * min(max(float(r["L"]) / vmax, 0.0), 1.0)
                for r in selected
            ]
            ax.scatter(
                [r["S"] for r in selected], [r["A"] for r in selected],
                color=model.color, marker=marker, s=sizes,
                edgecolor="black", linewidth=0.45, alpha=0.78,
                label=f"{model.label} {kind}",
            )
    ax.axvline(0.0, color="black", lw=1.0)
    ax.axhline(0.0, color="black", lw=1.0)
    ax.set_xlim(-2.05, 2.05)
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Structure (S)")
    ax.set_ylabel("Amplitude (A)")
    ax.set_title(
        "Future-Confidence Paper Eval: SAL Targets, All Models In One Axis",
        fontweight="bold",
    )
    ax.legend(loc="best", fontsize=8, ncols=2)
    handles = [
        ax.scatter([], [], s=size_min, facecolors="white", edgecolors="black", label="low L"),
        ax.scatter([], [], s=size_max, facecolors="white", edgecolors="black", label=f"high L (~{vmax:.2f})"),
    ]
    size_legend = ax.legend(handles=handles, loc="lower left", fontsize=8, title="Location")
    ax.add_artist(size_legend)
    ax.legend(loc="upper right", fontsize=8, ncols=2)
    fig.tight_layout()
    path = out_dir / "targets_sal_scatter_models.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_structured_rapsd(
    models: list[ModelRun], results: dict[str, dict], out_dir: Path
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    groups = []
    for result in results.values():
        clim = result.get("climatology", {})
        if isinstance(clim, dict):
            for group, payload in clim.items():
                if isinstance(payload, dict) and payload.get("rapsd") and group not in groups:
                    groups.append(group)
    for group in groups:
        fig, ax = plt.subplots(figsize=(9, 6))
        ref_drawn = False
        for model in models:
            payload = results[model.model_id].get("climatology", {}).get(group, {})
            rapsd = payload.get("rapsd") if isinstance(payload, dict) else None
            if not rapsd:
                continue
            freq = np.asarray(rapsd.get("freq", []), dtype=float)
            ref = np.asarray(rapsd.get("reference_psd", []), dtype=float)
            psd = np.asarray(rapsd.get("model_psd", []), dtype=float)
            valid = (freq > 0) & (psd > 0)
            if valid.any():
                wavelength = 1.0 / freq[valid]
                order = np.argsort(wavelength)
                ax.loglog(
                    wavelength[order],
                    psd[valid][order],
                    color=model.color,
                    lw=2.0,
                    label=model.label,
                )
            if not ref_drawn:
                valid_ref = (freq > 0) & (ref > 0)
                if valid_ref.any():
                    wavelength = 1.0 / freq[valid_ref]
                    order = np.argsort(wavelength)
                    ax.loglog(
                        wavelength[order], ref[valid_ref][order],
                        "k--", lw=1.8, label="Reference",
                    )
                    ref_drawn = True
        if len(ax.lines) <= 1:
            plt.close(fig)
            continue
        ax.invert_xaxis()
        ax.set_xlabel("Wavelength (km)")
        ax.set_ylabel("RAPSD")
        ax.set_title(
            f"Future-Confidence Paper Eval: {group.replace('_', ' ').title()} RAPSD",
            fontweight="bold",
        )
        ax.grid(True, which="both", ls="--", alpha=0.4)
        ax.legend(fontsize=9)
        fig.tight_layout()
        path = out_dir / f"climatology_{group}_rapsd_structured_models.png"
        fig.savefig(path, dpi=170, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
    return written


def plot_structured_qq(
    models: list[ModelRun], results: dict[str, dict], out_dir: Path
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    groups = []
    for result in results.values():
        clim = result.get("climatology", {})
        if isinstance(clim, dict):
            for group, payload in clim.items():
                if isinstance(payload, dict) and payload.get("qq") and group not in groups:
                    groups.append(group)
    for group in groups:
        first = None
        for model in models:
            first = results[model.model_id].get("climatology", {}).get(group, {}).get("qq")
            if first:
                break
        if not first:
            continue
        fig, axes = plt.subplots(1, len(first), figsize=(6 * len(first), 5.5), squeeze=False)
        for panel_idx, ax in enumerate(axes[0]):
            lo, hi = np.inf, -np.inf
            for model in models:
                qq = results[model.model_id].get("climatology", {}).get(group, {}).get("qq", [])
                if panel_idx >= len(qq):
                    continue
                curves = qq[panel_idx].get("curves", [])
                for curve in curves:
                    if str(curve.get("label", "")).lower() == "reference":
                        continue
                    ref = np.asarray(curve.get("ref", []), dtype=float)
                    sim = np.asarray(curve.get("sim", []), dtype=float)
                    if ref.size == 0 or sim.size == 0:
                        continue
                    ax.plot(ref, sim, color=model.color, lw=1.7, label=model.label)
                    lo = min(lo, np.nanmin(ref), np.nanmin(sim))
                    hi = max(hi, np.nanmax(ref), np.nanmax(sim))
            if not np.isfinite(lo):
                lo, hi = 0.0, 1.0
            ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, label="1:1")
            panel = first[panel_idx]
            ax.set_xlabel(panel.get("xlabel") or "Reference")
            ax.set_ylabel(panel.get("ylabel") or "Model")
            ax.set_title(panel.get("title") or f"Q-Q {panel_idx + 1}", fontweight="bold")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        fig.suptitle(
            f"Future-Confidence Paper Eval: {group.replace('_', ' ').title()} Q-Q",
            fontweight="bold",
        )
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        path = out_dir / f"climatology_{group}_qq_structured_models.png"
        fig.savefig(path, dpi=170, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
    return written


def _target_fields_path(model: ModelRun) -> Path:
    return model.result_dir / "data" / "targets_fields.npz"


def plot_structured_target_maps(models: list[ModelRun], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    loaded = []
    for model in models:
        path = _target_fields_path(model)
        if path.is_file():
            loaded.append((model, np.load(path, allow_pickle=True)))
    if not loaded:
        return []
    first_model, first = loaded[0]
    rows = [str(x) for x in first["display_label"]]
    written: list[Path] = []
    for field_name, style_name, out_name, title in [
        ("crps", "crps", "targets_combined_crps_maps_structured_models.png", "Target CRPS Maps"),
        ("out_of_envelope", "ooe", "targets_combined_out_of_envelope_structured_models.png", "Target Out-Of-Envelope Maps"),
    ]:
        columns = ["Reference"] + [model.label for model, _ in loaded]
        fields = {}
        for i, row in enumerate(rows):
            fields[(row, "Reference")] = first["reference"][i]
        for model, data in loaded:
            if field_name not in data or len(data[field_name]) != len(rows):
                continue
            for i, row in enumerate(rows):
                fields[(row, model.label)] = data[field_name][i]
        column_styles = {"Reference": get_style("precip")}
        column_styles.update({model.label: get_style(style_name) for model, _ in loaded})
        fig = plot_map_grid(
            fields,
            rows,
            columns,
            column_styles=column_styles,
            title=f"Future-Confidence Paper Eval: {title}",
            panel_size=2.0,
            annotate=False,
        )
        path = out_dir / out_name
        fig.savefig(path, dpi=170, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
    for _, data in loaded:
        data.close()
    return written


def make_contact_sheet(
    models: list[ModelRun], key: str, out_dir: Path, missing: list[dict]
) -> Path | None:
    images: list[tuple[ModelRun, Path | None]] = [
        (model, find_semantic_png(model, key)) for model in models
    ]
    for model, path in images:
        if path is None:
            missing.append({"model": model.model_id, "key": key})
    if all(path is None for _, path in images):
        return None

    ncols = len(models)
    fig, axes = plt.subplots(
        1, ncols, figsize=(7.2 * max(ncols, 1), 6.2), squeeze=False
    )
    axes = list(axes[0])
    for ax, (model, path) in zip(axes, images):
        ax.axis("off")
        ax.set_title(model.label, fontsize=13, fontweight="bold", color=model.color)
        if path is None:
            ax.text(0.5, 0.5, f"Missing: {key}", ha="center", va="center")
            continue
        img = mpimg.imread(path)
        ax.imshow(img)
    fig.suptitle(
        f"Future-Confidence Paper Eval: {key.replace('/', ' / ').replace('_', ' ').title()}",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91), w_pad=0.25)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{semantic_to_filename_prefix(key)}_models.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _crop_width(crop: tuple[int, int]) -> int:
    return max(crop[1] - crop[0], 1)


def _nonwhite_intervals(img: np.ndarray, y0: int) -> list[tuple[int, int]]:
    rgb = img[y0:, :, :3]
    height = rgb.shape[0]
    nonwhite = np.any(rgb < 0.965, axis=2)
    col_counts = nonwhite.sum(axis=0)
    threshold = max(8, int(height * 0.025))
    mask = col_counts > threshold
    intervals: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(mask):
        if value and start is None:
            start = idx
        if (not value or idx == len(mask) - 1) and start is not None:
            end = idx if not value else idx + 1
            if end - start > 20:
                intervals.append((start, end))
            start = None
    return intervals


def _pad_crop(crop: tuple[int, int], width: int, pad: int = 10) -> tuple[int, int]:
    return max(crop[0] - pad, 0), min(crop[1] + pad, width)


def _image_merge_spec(
    key: str, img: np.ndarray
) -> dict[str, tuple[int, int] | int] | None:
    basename = key.split("/")[-1]
    if basename not in MERGEABLE_IMAGE_BASENAMES:
        return None

    height, width = img.shape[:2]
    if key.startswith("targets/"):
        y0 = int(height * 0.085)
    elif key.startswith("climatology/") and basename == "rmse_map":
        y0 = int(height * 0.24)
    elif key.startswith("climatology/") and basename.startswith("bias_"):
        y0 = int(height * 0.12)
    else:
        y0 = int(height * 0.08)
    intervals = _nonwhite_intervals(img, y0)
    wide_min = max(95, int(width * 0.12))
    wide = [(x0, x1) for x0, x1 in intervals if x1 - x0 >= wide_min]

    if key.startswith("targets/") and basename in {"crps_maps", "out_of_envelope"}:
        if len(wide) >= 2:
            reference = (0, min(wide[0][1] + 12, width))
            model = _pad_crop(wide[1], width, pad=10)
            legend_start = next(
                (x0 for x0, x1 in intervals if x0 > wide[1][1] and x1 - x0 >= 35),
                model[1],
            )
            return {
                "y0": y0,
                "reference": reference,
                "model": model,
                "legend": (max(legend_start - 14, 0), width),
            }
        return {
            "y0": int(height * 0.055),
            "reference": (0, int(width * 0.43)),
            "model": (int(width * 0.43), int(width * 0.69)),
            "legend": (int(width * 0.71), width),
        }
    if key.startswith("climatology/") and basename.startswith("bias_"):
        if len(wide) >= 2:
            reference = (0, min(wide[0][1] + 12, width))
            model = _pad_crop(wide[1], width, pad=10)
            legend_start = next(
                (x0 for x0, x1 in intervals if x0 > wide[1][1] and x1 - x0 >= 35),
                model[1],
            )
            return {
                "y0": y0,
                "reference": reference,
                "model": model,
                "legend": (max(legend_start - 14, 0), width),
            }
        return {
            "y0": int(height * 0.12),
            "reference": (0, int(width * 0.43)),
            "model": (int(width * 0.43), int(width * 0.70)),
            "legend": (int(width * 0.71), width),
        }
    if key.startswith("climatology/") and basename == "rmse_map":
        if intervals:
            model_interval = max(intervals, key=lambda item: item[1] - item[0])
            model = _pad_crop(model_interval, width, pad=10)
            legend_start = next(
                (x0 for x0, x1 in intervals if x0 > model_interval[1] and x1 - x0 >= 35),
                model[1],
            )
            return {
                "y0": y0,
                "model": model,
                "legend": (max(legend_start - 14, 0), width),
            }
        return {
            "y0": int(height * 0.14),
            "model": (0, int(width * 0.70)),
            "legend": (int(width * 0.70), width),
        }
    return None


def make_merged_image_sheet(
    models: list[ModelRun], key: str, out_dir: Path, missing: list[dict]
) -> Path | None:
    """Merge comparable raster diagnostics into one reference/model figure.

    These are already-rendered W&B PNGs, so this function crops the stable
    semantic columns from each image instead of regenerating the underlying map
    data. The result avoids repeating the same reference/legend columns for
    every model.
    """

    image_paths: list[tuple[ModelRun, Path | None]] = [
        (model, find_semantic_png(model, key)) for model in models
    ]
    available = [(model, path) for model, path in image_paths if path is not None]
    if not available:
        return None

    first_img = mpimg.imread(available[0][1])
    height, width = first_img.shape[:2]
    spec = _image_merge_spec(key, first_img)
    if spec is None:
        return None
    for model, path in image_paths:
        if path is None:
            missing.append({"model": model.model_id, "key": key})

    y0 = int(spec["y0"])
    reference_crop = spec.get("reference")
    model_crop = spec["model"]
    legend_crop = spec.get("legend")

    columns: list[tuple[str, str, np.ndarray, float]] = []
    if reference_crop is not None:
        x0, x1 = reference_crop
        columns.append(
            (
                "Reference",
                "black",
                first_img[y0:, x0:x1, :],
                _crop_width(reference_crop),
            )
        )

    for model, path in available:
        img = mpimg.imread(path)
        img_spec = _image_merge_spec(key, img)
        if img_spec is None:
            continue
        img_y0 = int(img_spec["y0"])
        x0, x1 = img_spec["model"]
        columns.append(
            (
                model.label,
                model.color,
                img[img_y0:, x0:x1, :],
                _crop_width(img_spec["model"]),
            )
        )

    if legend_crop is not None:
        x0, x1 = legend_crop
        columns.append(
            (
                "Legend",
                "black",
                first_img[y0:, x0:x1, :],
                _crop_width(legend_crop),
            )
        )

    width_ratios = [max(col[3] / 180.0, 0.55) for col in columns]
    max_body_height = max(col[2].shape[0] for col in columns)
    fig_width = min(max(sum(width_ratios) * 2.15, 8.0), 24.0)
    fig_height = min(max(max_body_height / 170.0, 4.8), 26.0)
    fig, axes = plt.subplots(
        1,
        len(columns),
        figsize=(fig_width, fig_height),
        gridspec_kw={"width_ratios": width_ratios},
        squeeze=False,
    )
    for ax, (title, color, img, _) in zip(axes[0], columns):
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(title, fontsize=12, fontweight="bold", color=color, pad=8)
    fig.suptitle(
        f"Future-Confidence Paper Eval: {key.replace('/', ' / ').replace('_', ' ').title()}",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95), w_pad=0.02)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{semantic_to_filename_prefix(key)}_models.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def write_summary_json(
    path: Path,
    models: list[ModelRun],
    clim_rows: list[dict],
    target_rows_: list[dict],
    written_figures: list[Path],
    missing_images: list[dict],
) -> None:
    def bias(model: str, stat: str) -> float:
        return _bias_lookup(clim_rows, model, "all_periods", "raw", stat)

    target_summary = {}
    for model in models:
        vals = [
            float(r["crps_mean"]) for r in target_rows_
            if r["model"] == model.model_id
        ]
        target_summary[model.model_id] = {
            "mean_target_crps": float(np.mean(vals)) if vals else math.nan
        }

    payload = {
        "models": [
            {
                "id": m.model_id,
                "label": m.label,
                "color": m.color,
                "result_dir": str(m.result_dir),
                "image_dir": str(m.image_dir) if m.image_dir else None,
            }
            for m in models
        ],
        "all_periods_raw_bias_pct": {
            m.model_id: {
                "mean": bias(m.model_id, "mean"),
                "sd": bias(m.model_id, "sd"),
                "p99": bias(m.model_id, "p99"),
                "dry_pct": bias(m.model_id, "dry_pct"),
                "median": bias(m.model_id, "median"),
            }
            for m in models
        },
        "target_summary": target_summary,
        "figures": [str(p) for p in written_figures],
        "missing_images": missing_images,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")


def run_comparison(models: list[ModelRun], output_dir: Path) -> dict:
    results = {model.model_id: load_results(model) for model in models}
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    main_dir = figures_dir / "main"
    contact_dir = figures_dir / "appendix" / "contact_sheets"
    output_dir.mkdir(parents=True, exist_ok=True)

    clim = climatology_rows(models, results)
    targets = target_rows(models, results)
    write_csv(tables_dir / "climatology_bias_comparison.csv", clim)
    write_csv(tables_dir / "target_crps_comparison.csv", targets)
    write_csv(tables_dir / "sal_comparison.csv", targets)

    written: list[Path] = []
    written.extend(plot_climatology_bias_summary(models, clim, main_dir))
    written.extend(plot_target_crps(models, targets, main_dir))
    written.append(plot_sal_comparison(models, targets, main_dir))
    written.extend(plot_structured_rapsd(models, results, main_dir))
    written.extend(plot_structured_qq(models, results, main_dir))
    written.extend(plot_structured_target_maps(models, main_dir))

    missing: list[dict] = []
    for key in CONTACT_SHEET_KEYS:
        path = make_merged_image_sheet(models, key, contact_dir, missing)
        if path is None:
            path = make_contact_sheet(models, key, contact_dir, missing)
        if path is not None:
            written.append(path)

    summary_path = output_dir / "paper_model_comparison_summary.json"
    write_summary_json(summary_path, models, clim, targets, written, missing)
    return {
        "summary": summary_path,
        "tables": tables_dir,
        "figures": figures_dir,
        "n_figures": len(written),
        "missing_images": missing,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Combine completed paper-eval outputs across models."
    )
    parser.add_argument(
        "--config-name",
        default=None,
        help=(
            "Comparison config name under conf/ (without .yaml) or a YAML path. "
            "Example: compare/paper_models_future_s43_all_available."
        ),
    )
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        help=(
            "Model spec: id=label:color:result_dir[:image_dir]. "
            "May be repeated. Defaults to t0/past12h/sym3h under /outputs."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="/outputs/paper_eval/model_comparison/future_conf_s43_t0_past12h_sym3h",
        help="Directory for combined tables and figures.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    config_name, remaining = _pop_config_name(raw_argv)
    args = build_arg_parser().parse_args(remaining)
    config_name = args.config_name or config_name
    if config_name:
        models, output_dir = load_compare_config(config_name)
        if args.model:
            models = [parse_model_spec(raw) for raw in args.model]
        if args.output_dir != build_arg_parser().get_default("output_dir"):
            output_dir = Path(args.output_dir)
    else:
        models = [parse_model_spec(raw) for raw in args.model] if args.model else default_model_runs()
        output_dir = Path(args.output_dir)
    result = run_comparison(models, output_dir)
    print(f"summary: {result['summary']}")
    print(f"tables: {result['tables']}")
    print(f"figures: {result['figures']}")
    print(f"n_figures: {result['n_figures']}")
    if result["missing_images"]:
        print(f"missing_images: {len(result['missing_images'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
