# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Paper-protocol evaluation flow for CorrDiff (separate from evaluate.py).

Reproduces the JAMES-paper Section-2.3 evaluation on top of the ensemble
prediction NetCDF files produced by ``evaluate.py``/``generate.py`` — it does
**not** run any models or modify the existing evaluation.  Two independent
analyses, either of which can be enabled by pointing at a NetCDF file:

  * Climatological (whole period): distributional bias table (Dry%, wet
    Mean/SD/Median/P99) computed raw and after a 3x3 moving average, plus
    per-gridpoint Mean/SD bias maps.   -> ``eval.climatology_predictions_file``
  * Targeted (selected days):  per-target spatial CRPS maps, out-of-envelope
    error maps (raw + 3x3) and SAL diagrams.   -> ``eval.targets_predictions_file``

Predictions are streamed timestep-by-timestep from NetCDF, so 50- or 1000-member
ensembles over a long period never need to fit in RAM.  Works with whatever
ensemble size N is present in the file (start with the usual N=10).

Results are logged to a **new W&B project** (default ``CorrDiff-Paper``) and
saved as a JSON backup.

Usage:
    python evaluate_paper.py --config-name=eval/paper/base run_tag=my_run \\
        eval.climatology_predictions_file=/path/to/eval_clim.nc \\
        eval.targets_predictions_file=/path/to/eval_targets.nc
"""

import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import hydra
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import torch
import wandb
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from physicsnemo.launch.logging import PythonLogger
from physicsnemo.launch.logging.wandb import initialize_wandb

from helpers.climatology import ClimatologyAccumulator, SpatialDistributionAccumulator
from helpers.targets import crps_map, out_of_envelope_map, sal_distribution
from helpers.plots import RAPSDAccumulator
from helpers.paper_eval import io as paper_io
from helpers.paper_eval import results as paper_results
from helpers.paper_eval import run as paper_run
from helpers.paper_eval import (
    rmse_map_figure,
    bias_map_figure,
    crps_map_figure,
    out_of_envelope_figure,
    sal_epoch_figure,
    sal_figure,
    qq_figure,
    rapsd_figure,
)

# Statistic label -> bias_maps() key prefix, in display order.
_BIAS_STATS = [("Dry%", "dry_pct"), ("Mean", "mean"), ("SD", "sd"),
               ("Median", "median"), ("P99", "p99")]

_RUN_OUTPUTS = None
_RUN_MANIFEST = None


def _artifact_filename(key: str, suffix: str) -> str:
    return f"{key.replace('/', '_')}{suffix}"


def _figure_output_path(key: str) -> Path | None:
    if _RUN_OUTPUTS is None:
        return None
    return _RUN_OUTPUTS.figures_dir / f"{key}.png"


def _table_output_path(key: str) -> Path | None:
    if _RUN_OUTPUTS is None:
        return None
    return _RUN_OUTPUTS.tables_dir / _artifact_filename(key, ".csv")


def _data_output_path(name: str) -> Path | None:
    if _RUN_OUTPUTS is None:
        return None
    return _RUN_OUTPUTS.data_dir / name


def _log_image(key: str, fig):
    path = _figure_output_path(key)
    if path is not None:
        paper_results.save_figure(fig, path)
        if _RUN_MANIFEST is not None:
            _RUN_MANIFEST.add(key, path, "figure", logged_to_wandb=True)
    return wandb.Image(fig)


def _write_table_artifact(key: str, rows: list[dict]) -> None:
    path = _table_output_path(key)
    if path is not None:
        paper_results.write_csv(path, rows)
        if _RUN_MANIFEST is not None:
            _RUN_MANIFEST.add(key, path, "table", logged_to_wandb=True)


def _save_target_field_data(rows: list[dict], key: str = "targets/fields") -> None:
    path = _data_output_path(_artifact_filename(key, ".npz"))
    if path is None or not rows:
        return
    paper_results.savez(
        path,
        reference=np.stack([np.asarray(r["reference"]) for r in rows]),
        crps=np.stack([np.asarray(r["crps"]) for r in rows]),
        out_of_envelope=np.stack([np.asarray(r["ooe"]) for r in rows]),
        case_id=np.asarray([str(r.get("case_id", "")) for r in rows]),
        epoch=np.asarray([str(r.get("epoch", "")) for r in rows]),
        kind=np.asarray([str(r.get("kind", "")) for r in rows]),
        label=np.asarray([str(r.get("label", "")) for r in rows]),
        display_label=np.asarray([str(r.get("display_label", "")) for r in rows]),
    )
    if _RUN_MANIFEST is not None:
        _RUN_MANIFEST.add(key, path, "data")


def _figure_context(cfg) -> str:
    return str(
        cfg.eval.get(
            "figure_title_context",
            f"CorrDiff paper evaluation: {cfg.get('run_tag', 'paper-eval')}",
        )
    )


def _clim_title(cfg, group: str, figure: str, detail: str) -> str:
    return f"{_figure_context(cfg)} | Climatology: {group}\n{figure} | {detail}"


def _targets_title(cfg, figure: str, detail: str) -> str:
    return f"{_figure_context(cfg)} | Targeted 12-case set\n{figure} | {detail}"


def _target_display_label(epoch: str, kind: str, timestamp: str, max_precip: float) -> str:
    """Compact figure label for a selected target case."""
    epoch_code = {
        "current": "CUR",
        "mid": "MID",
        "end": "END",
    }.get(str(epoch).lower(), str(epoch).upper()[:3])
    stamp = _normalize_timestamp(timestamp)
    date, time = stamp.split(" ", 1) if " " in stamp else (stamp, "")
    hour = time[:5].replace(":00", "Z") if time else ""
    return f"{epoch_code} {kind}\n{date} {hour}\nmax {max_precip:.0f} mm"


def _target_case_id(epoch: str, kind: str, count: int) -> str:
    epoch_code = {
        "current": "CUR",
        "mid": "MID",
        "end": "END",
    }.get(str(epoch).lower(), str(epoch).upper()[:3])
    kind_code = "E" if str(kind).lower() == "extreme" else "N"
    return f"{epoch_code}-{kind_code}{count}"


def _resolve_channel(channel_names: list[str], selection) -> int:
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
    # auto: first channel whose name looks like precipitation, else channel 0
    for i, name in enumerate(channel_names):
        norm = "".join(c for c in name.lower() if c.isalnum())
        if any(tok in norm for tok in ["precip", "rain", "tp", "rr"]):
            return i
    return 0


class _NetCDFStream:
    """Streaming reader for the prediction NetCDF written by evaluate.py.

    Yields one timestep at a time as physical-unit arrays for a single channel,
    so the full ensemble is never materialised in memory.
    """

    def __init__(self, path: str, channel_selection):
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
        self.ch = _resolve_channel(self.channel_names, channel_selection)
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
        return [_format_timestamp(v) for v in vals]

    def target(self, t: int) -> np.ndarray:
        return np.clip(np.asarray(self.truth[self.channel_name][t]), 0.0, None)

    def pred_ens(self, t: int) -> np.ndarray:
        # prediction layout: (ensemble, time, y, x)
        return np.clip(np.asarray(self.pred[self.channel_name][:, t]), 0.0, None)

    def close(self):
        self.f.close()


def _format_timestamp(value) -> str:
    """Format netCDF/cftime/datetime timestamps for stable config matching."""
    return paper_io.format_timestamp(value)


def _normalize_timestamp(value) -> str:
    return paper_io.normalize_timestamp(value)


def _make_climatology_state(cfg, img_shape):
    rapsd_dx_km = float(cfg.eval.get("rapsd_dx_km", 2.0))
    acc = ClimatologyAccumulator(
        img_shape=img_shape,
        dry_threshold=float(cfg.eval.get("dry_threshold", 1.0)),
        wet_threshold=float(cfg.eval.get("wet_threshold", 1.0)),
        hist_hi=float(cfg.eval.get("hist_max_mm", 300.0)),
        hist_bins=int(cfg.eval.get("hist_bins", 3000)),
        compute_smoothed=bool(cfg.eval.get("compute_smoothed", True)),
        compute_quantile_maps=bool(cfg.eval.get("compute_quantile_maps", False)),
        map_bins=int(cfg.eval.get("map_bins", 100)),
    )
    spatial = SpatialDistributionAccumulator(
        wet_threshold=float(cfg.eval.get("wet_threshold", 1.0))
    )
    rapsd = RAPSDAccumulator(img_shape=img_shape, dx_km=rapsd_dx_km)
    return {
        "acc": acc,
        "spatial": spatial,
        "rapsd": rapsd,
        "n_times": 0,
        "n_ensemble": None,
        "channel": None,
        "lat": None,
        "lon": None,
        "files": [],
    }


def _update_climatology_state(state, stream, pred, tar):
    state["acc"].update(pred, tar)
    state["spatial"].update(pred, tar)
    state["rapsd"].update(
        torch.from_numpy(pred[:, None]),   # (N, 1, H, W)
        torch.from_numpy(tar[None]),        # (1, H, W)
    )
    state["n_times"] += 1
    state["n_ensemble"] = stream.n_ensemble
    state["channel"] = stream.channel_name
    if state["lat"] is None:
        state["lat"] = stream.lat
        state["lon"] = stream.lon


def _finalize_climatology_state(cfg, logger, state, payload_prefix: str, display_name: str) -> tuple[dict, dict]:
    acc = state["acc"]
    spatial = state["spatial"]
    rapsd = state["rapsd"]
    rapsd_dx_km = float(cfg.eval.get("rapsd_dx_km", 2.0))
    acc.reduce()

    rows = acc.to_table()
    json_summary = {
        "channel": state["channel"],
        "n_times": state["n_times"],
        "n_ensemble": state["n_ensemble"],
        "files": state["files"],
        "table": rows,
    }
    logger.info("=" * 70)
    logger.info(f"CLIMATOLOGICAL BIAS TABLE [{display_name}]")
    logger.info(f"  {'window':6s} {'statistic':9s} {'model':>12s} {'reference':>12s} {'rel_bias%':>10s}")
    for r in rows:
        logger.info(
            f"  {r['window']:6s} {r['statistic']:9s} {r['model']:12.4f} "
            f"{r['reference']:12.4f} {r['rel_bias_pct']:10.2f}"
        )

    payload = {}
    table = wandb.Table(columns=["window", "statistic", "model", "reference", "rel_bias_pct"])
    for r in rows:
        table.add_data(r["window"], r["statistic"], r["model"], r["reference"], r["rel_bias_pct"])
    payload[f"{payload_prefix}/bias_table"] = table
    _write_table_artifact(f"{payload_prefix}/bias_table", rows)

    lat, lon = state["lat"], state["lon"]

    # Fig 2 — RMSE map
    rmse = acc.rmse_map()
    if rmse is not None:
        fig = rmse_map_figure(
            rmse,
            lat=lat,
            lon=lon,
            title=_clim_title(
                cfg,
                display_name,
                "Per-gridpoint RMSE",
                "CorrDiff vs reference precipitation; lower is better",
            ),
        )
        payload[f"{payload_prefix}/rmse_map"] = _log_image(f"{payload_prefix}/rmse_map", fig)
        plt.close(fig)

    # Figs 3/4 — per-statistic relative-bias maps (gridpoint + 3x3 rows)
    maps = acc.bias_maps()
    window_label = {"raw": "Gridpoint", "s3x3": "3x3"}
    for label, key in _BIAS_STATS:
        window_maps = {}
        for win_key, row_label in window_label.items():
            md = maps.get(win_key)
            if md and f"{key}_model" in md and f"{key}_ref" in md:
                window_maps[row_label] = {
                    "model": md[f"{key}_model"], "ref": md[f"{key}_ref"]
                }
        if window_maps:
            fig = bias_map_figure(
                label,
                window_maps,
                lat=lat,
                lon=lon,
                title=_clim_title(
                    cfg,
                    display_name,
                    f"{label} precipitation climatology",
                    "Reference column is absolute precipitation; CorrDiff column is relative bias (%)",
                ),
            )
            payload[f"{payload_prefix}/bias_{key}"] = _log_image(f"{payload_prefix}/bias_{key}", fig)
            plt.close(fig)

    # Fig 5 — Q-Q triptych
    qq_panels = spatial.qq_panels()
    json_summary["qq"] = [
        {
            "title": p.get("title"),
            "xlabel": p.get("xlabel"),
            "ylabel": p.get("ylabel"),
            "curves": [
                {
                    "label": c.get("label"),
                    "ref": paper_results.json_array(c.get("ref", [])),
                    "sim": paper_results.json_array(c.get("sim", [])),
                }
                for c in p.get("curves", [])
            ],
        }
        for p in qq_panels
    ]
    fig = qq_figure(
        qq_panels,
        title=_clim_title(
            cfg,
            display_name,
            "Q-Q diagnostics",
            "Reference quantiles on x-axis, CorrDiff quantiles on y-axis; dashed line is perfect",
        ),
    )
    payload[f"{payload_prefix}/qq"] = _log_image(f"{payload_prefix}/qq", fig)
    plt.close(fig)

    # Fig 6 — RAPSD
    if rapsd.n_samples > 0:
        pred_psd = (rapsd.pred_psd_sum / rapsd.n_samples).numpy()
        tar_psd = (rapsd.target_psd_sum / rapsd.n_samples).numpy()
        freq = np.asarray(rapsd.bin_centers)
        json_summary["rapsd"] = {
            "freq": paper_results.json_array(freq),
            "reference_psd": paper_results.json_array(tar_psd),
            "model_psd": paper_results.json_array(pred_psd),
            "dx_km": rapsd_dx_km,
        }
        data_path = _data_output_path(f"climatology_{payload_prefix.replace('/', '_')}_rapsd.npz")
        if data_path is not None:
            paper_results.savez(
                data_path, freq=freq, reference_psd=tar_psd, model_psd=pred_psd
            )
            if _RUN_MANIFEST is not None:
                _RUN_MANIFEST.add(f"{payload_prefix}/rapsd_data", data_path, "data")
        curves = [
            {"label": "Reference", "freq": freq, "psd": tar_psd, "style": "k--"},
            {"label": "CorrDiff", "freq": freq, "psd": pred_psd, "style": "r-"},
        ]
        fig = rapsd_figure(
            curves,
            dx_km=rapsd_dx_km,
            title=_clim_title(
                cfg,
                display_name,
                "Radially averaged power spectral density",
                f"Spatial-scale spectrum of precipitation fields; grid spacing dx={rapsd_dx_km:g} km",
            ),
        )
        payload[f"{payload_prefix}/rapsd"] = _log_image(f"{payload_prefix}/rapsd", fig)
        plt.close(fig)

    return payload, json_summary


def _run_climatology(cfg, logger, path, json_out) -> dict:
    """Stream the climatology NetCDF; return a dict of W&B-loggable artifacts."""
    selection = cfg.eval.get("precip_channel", "auto")
    stream = _NetCDFStream(path, selection)
    logger.info(
        f"[climatology] {stream.n_times} timesteps, {stream.n_ensemble} members, "
        f"channel='{stream.channel_name}', shape {stream.img_shape}"
    )

    state = _make_climatology_state(cfg, stream.img_shape)
    state["files"].append({"label": Path(path).parent.name, "path": path})
    for t in range(stream.n_times):
        pred = stream.pred_ens(t)   # (N, H, W)
        tar = stream.target(t)      # (H, W)
        _update_climatology_state(state, stream, pred, tar)
    stream.close()
    payload, summary = _finalize_climatology_state(
        cfg, logger, state, "climatology", "single_file"
    )
    json_out["climatology"] = summary
    return payload


def _rows_by_epoch(rows: list[dict]) -> OrderedDict[str, list[dict]]:
    order = ["current", "mid", "end"]
    grouped: OrderedDict[str, list[dict]] = OrderedDict()
    for epoch in order:
        subset = [r for r in rows if str(r.get("epoch", "")).lower() == epoch]
        if subset:
            grouped[epoch] = subset
    for r in rows:
        epoch = str(r.get("epoch", "targets")).lower()
        if epoch not in grouped:
            grouped.setdefault(epoch, []).append(r)
    return grouped


def _target_payload(cfg, rows: list[dict], sal_rows: list[dict], lat, lon) -> dict:
    payload = {}
    if not rows:
        return payload

    for epoch, epoch_rows in _rows_by_epoch(rows).items():
        for key, fig in [
            (f"targets/{epoch}/crps_maps", crps_map_figure(
                epoch_rows,
                lat=lat,
                lon=lon,
                title=_targets_title(
                    cfg,
                    f"{epoch.upper()} CRPS maps after 3x3 smoothing",
                    "Rows are selected target cases; columns are reference precipitation and CorrDiff CRPS/MAE",
                ),
            )),
            (f"targets/{epoch}/out_of_envelope", out_of_envelope_figure(
                epoch_rows,
                lat=lat,
                lon=lon,
                title=_targets_title(
                    cfg,
                    f"{epoch.upper()} out-of-envelope error after 3x3 smoothing",
                    "Positive means reference exceeds ensemble max; negative means reference is below ensemble min",
                ),
            )),
        ]:
            payload[key] = _log_image(key, fig)
            plt.close(fig)

    for key, fig in [
        ("targets/combined/crps_maps", crps_map_figure(
            rows,
            lat=lat,
            lon=lon,
            title=_targets_title(
                cfg,
                "Combined CRPS maps after 3x3 smoothing",
                "All selected target cases; use epoch-specific figures for readable inspection",
            ),
        )),
        ("targets/combined/out_of_envelope", out_of_envelope_figure(
            rows,
            lat=lat,
            lon=lon,
            title=_targets_title(
                cfg,
                "Combined out-of-envelope error after 3x3 smoothing",
                "All selected target cases; use epoch-specific figures for readable inspection",
            ),
        )),
        ("targets/sal_scatter_by_epoch", sal_epoch_figure(
            sal_rows,
            title=_targets_title(
                cfg,
                "SAL scatter by epoch",
                "x=Structure, y=Amplitude, color=Location (L), marker=extreme/normal; zero is ideal",
            ),
        )),
        ("targets/sal_case_grid", sal_figure(
            sal_rows,
            title=_targets_title(
                cfg,
                "SAL case grid",
                "x=Structure, y=Amplitude, color=Location (L); each panel is one selected case",
            ),
        )),
    ]:
        payload[key] = _log_image(key, fig)
        plt.close(fig)
    return payload


def _run_targets(cfg, logger, path, json_out) -> dict:
    """Stream the targets NetCDF; return combined per-target figures (rows=targets)."""
    selection = cfg.eval.get("precip_channel", "auto")
    f_factor = float(cfg.eval.get("sal_f_factor", 1.0 / 15.0))
    sal_thr_quantile = float(cfg.eval.get("sal_thr_quantile", 0.95))
    smooth_size = int(cfg.eval.get("smooth_size", 3))
    stream = _NetCDFStream(path, selection)
    logger.info(
        f"[targets] {stream.n_times} target days, {stream.n_ensemble} members, "
        f"channel='{stream.channel_name}', shape {stream.img_shape}"
    )

    def _stats(a):
        a = np.asarray(a, dtype=float)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return {"median": float("nan"), "mean": float("nan")}
        return {"median": float(np.median(a)), "mean": float(np.mean(a))}

    rows = []           # per-target maps/scores for the combined figures
    summaries = []      # JSON
    for t in range(stream.n_times):
        pred = stream.pred_ens(t)
        tar = stream.target(t)

        crps = crps_map(pred, tar, smooth_size=smooth_size)   # Fig 8 (3x3)
        ooe = out_of_envelope_map(pred, tar, smooth_size=smooth_size)  # Fig 7 (3x3)
        sal = sal_distribution(pred, tar, f_factor=f_factor, thr_quantile=sal_thr_quantile)

        max_precip = float(tar.max())
        label = f"t{t} ({max_precip:.0f}mm)"
        epoch = "targets"
        kind = "target"
        display_label = _target_display_label(epoch, kind, str(t), max_precip)
        case_id = f"T{t + 1:02d}"
        rows.append({
            "label": label, "display_label": display_label,
            "case_id": case_id,
            "epoch": epoch, "kind": kind,
            "reference": tar, "crps": crps, "ooe": ooe,
            "S": sal["S"], "A": sal["A"], "L": sal["L"], "reference_sal": None,
        })
        summaries.append({
            "target_idx": t, "case_id": case_id, "max_precip_mm": max_precip,
            "crps_mean": float(np.nanmean(crps)), "sal_threshold": sal["threshold"],
            "S": _stats(sal["S"]), "A": _stats(sal["A"]), "L": _stats(sal["L"]),
        })
        logger.info(
            f"  target {t}: max={max_precip:.1f}mm "
            f"crps_mean={summaries[-1]['crps_mean']:.3f} "
            f"S~{summaries[-1]['S']['median']:.2f} A~{summaries[-1]['A']['median']:.2f} "
            f"L~{summaries[-1]['L']['median']:.2f}"
        )

    json_out["targets"] = {
        "channel": stream.channel_name,
        "n_targets": stream.n_times,
        "n_ensemble": stream.n_ensemble,
        "per_target": summaries,
    }

    lat, lon = stream.lat, stream.lon
    sal_rows = [{"label": r["label"], "display_label": r["display_label"],
                 "case_id": r["case_id"],
                 "epoch": r["epoch"], "kind": r["kind"],
                 "S": r["S"], "A": r["A"], "L": r["L"],
                 "reference": r["reference_sal"]} for r in rows]
    _save_target_field_data(rows)
    payload = _target_payload(cfg, rows, sal_rows, lat, lon)

    stream.close()
    return payload


def _as_plain_container(value):
    if value is None:
        return None
    return OmegaConf.to_container(value, resolve=True)


def _prediction_file_entries(cfg) -> list[dict]:
    """Return normalized multi-file entries from eval.prediction_files."""
    return paper_io.prediction_file_entries(cfg)


def _target_entries(cfg) -> list[dict]:
    """Return normalized target entries from eval.targets."""
    return paper_io.target_entries(cfg)


def _stream_target_index(stream: _NetCDFStream, target: dict) -> int:
    """Resolve a target entry against an open stream."""
    return paper_io.stream_target_index(stream, target)


def _run_climatology_multifile(cfg, logger, entries: list[dict], json_out) -> dict:
    """Stream all prediction files once and accumulate all-period + per-epoch stats."""
    selection = cfg.eval.get("precip_channel", "auto")
    states: OrderedDict[str, dict] = OrderedDict()
    payload = {}
    expected_shape = expected_channel = expected_n_ensemble = None

    for entry in entries:
        stream = _NetCDFStream(entry["path"], selection)
        logger.info(
            f"[climatology:{entry['epoch']}/{entry['label']}] "
            f"{stream.n_times} timesteps, {stream.n_ensemble} members, "
            f"channel='{stream.channel_name}', shape {stream.img_shape}"
        )
        if expected_shape is None:
            expected_shape = stream.img_shape
            expected_channel = stream.channel_name
            expected_n_ensemble = stream.n_ensemble
            states["all_periods"] = _make_climatology_state(cfg, stream.img_shape)
        elif (
            stream.img_shape != expected_shape
            or stream.channel_name != expected_channel
            or stream.n_ensemble != expected_n_ensemble
        ):
            stream.close()
            raise ValueError(
                "All eval.prediction_files must share image shape, precip channel, "
                "and ensemble count for pooled climatology."
            )
        if entry["epoch"] not in states:
            states[entry["epoch"]] = _make_climatology_state(cfg, stream.img_shape)

        file_summary = {
            "epoch": entry["epoch"],
            "label": entry["label"],
            "path": entry["path"],
            "n_times": stream.n_times,
        }
        states["all_periods"]["files"].append(file_summary)
        states[entry["epoch"]]["files"].append(file_summary)

        for t in range(stream.n_times):
            pred = stream.pred_ens(t)
            tar = stream.target(t)
            _update_climatology_state(states["all_periods"], stream, pred, tar)
            _update_climatology_state(states[entry["epoch"]], stream, pred, tar)
        stream.close()

    json_out["climatology"] = {}
    for name, state in states.items():
        group_payload, summary = _finalize_climatology_state(
            cfg, logger, state, f"climatology/{name}", name
        )
        payload.update(group_payload)
        json_out["climatology"][name] = summary
    return payload


def _run_targets_from_entries(cfg, logger, entries: list[dict], targets: list[dict], json_out) -> dict:
    """Run targeted diagnostics by loading only selected timesteps from source files."""
    selection = cfg.eval.get("precip_channel", "auto")
    f_factor = float(cfg.eval.get("sal_f_factor", 1.0 / 15.0))
    sal_thr_quantile = float(cfg.eval.get("sal_thr_quantile", 0.95))
    smooth_size = int(cfg.eval.get("smooth_size", 3))
    entry_by_label = {entry["label"]: entry for entry in entries}
    missing = [target["label"] for target in targets if target["label"] not in entry_by_label]
    if missing:
        raise ValueError(f"eval.targets reference unknown prediction file labels: {missing}")

    def _stats(a):
        a = np.asarray(a, dtype=float)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return {"median": float("nan"), "mean": float("nan")}
        return {"median": float(np.median(a)), "mean": float(np.mean(a))}

    rows = []
    summaries = []
    lat = lon = None
    n_ensemble = None
    channel = None
    case_counts: dict[tuple[str, str], int] = {}

    targets_by_label: OrderedDict[str, list[dict]] = OrderedDict()
    for target in targets:
        targets_by_label.setdefault(target["label"], []).append(target)

    for label, label_targets in targets_by_label.items():
        entry = entry_by_label[label]
        stream = _NetCDFStream(entry["path"], selection)
        if lat is None:
            lat, lon = stream.lat, stream.lon
            n_ensemble = stream.n_ensemble
            channel = stream.channel_name
        elif stream.n_ensemble != n_ensemble or stream.channel_name != channel:
            stream.close()
            raise ValueError("All targeted source files must share channel and ensemble count")
        logger.info(
            f"[targets:{entry['epoch']}/{entry['label']}] resolving "
            f"{len(label_targets)} selected timesteps"
        )
        for target in label_targets:
            t = _stream_target_index(stream, target)
            pred = stream.pred_ens(t)
            tar = stream.target(t)

            crps = crps_map(pred, tar, smooth_size=smooth_size)
            ooe = out_of_envelope_map(pred, tar, smooth_size=smooth_size)
            sal = sal_distribution(pred, tar, f_factor=f_factor, thr_quantile=sal_thr_quantile)

            max_precip = float(tar.max())
            timestamp = _normalize_timestamp(stream.times[t])
            epoch = target.get("epoch") or entry["epoch"]
            kind = target.get("kind", "target")
            row_label = (
                f"{epoch} {kind} {timestamp} ({max_precip:.0f}mm)"
            )
            display_label = _target_display_label(epoch, kind, timestamp, max_precip)
            count_key = (str(epoch).lower(), str(kind).lower())
            case_counts[count_key] = case_counts.get(count_key, 0) + 1
            case_id = _target_case_id(epoch, kind, case_counts[count_key])
            rows.append({
                "label": row_label, "display_label": display_label,
                "case_id": case_id,
                "epoch": epoch, "kind": kind,
                "reference": tar, "crps": crps, "ooe": ooe,
                "S": sal["S"], "A": sal["A"], "L": sal["L"], "reference_sal": None,
            })
            summaries.append({
                "epoch": epoch,
                "kind": kind,
                "label": entry["label"],
                "display_label": display_label,
                "case_id": case_id,
                "target_idx": t,
                "timestamp": timestamp,
                "max_precip_mm": max_precip,
                "crps_mean": float(np.nanmean(crps)),
                "sal_threshold": sal["threshold"],
                "S": _stats(sal["S"]),
                "A": _stats(sal["A"]),
                "L": _stats(sal["L"]),
            })
            logger.info(
                f"  {row_label}: crps_mean={summaries[-1]['crps_mean']:.3f} "
                f"S~{summaries[-1]['S']['median']:.2f} "
                f"A~{summaries[-1]['A']['median']:.2f} "
                f"L~{summaries[-1]['L']['median']:.2f}"
            )
        stream.close()

    json_out["targets"] = {
        "channel": channel,
        "n_targets": len(summaries),
        "n_ensemble": n_ensemble,
        "per_target": summaries,
    }

    sal_rows = [{"label": r["label"], "display_label": r["display_label"],
                 "case_id": r["case_id"],
                 "epoch": r["epoch"], "kind": r["kind"],
                 "S": r["S"], "A": r["A"], "L": r["L"],
                 "reference": r["reference_sal"]} for r in rows]
    _save_target_field_data(rows)
    return _target_payload(cfg, rows, sal_rows, lat, lon)


@hydra.main(version_base="1.2", config_path="conf", config_name="eval/paper/base")
def main(cfg: DictConfig) -> None:
    global _RUN_OUTPUTS, _RUN_MANIFEST
    logger = PythonLogger("evaluate_paper")
    logger.file_logging("evaluate_paper.log")

    run_tag = cfg.get("run_tag", "paper-eval")
    outputs = paper_io.eval_outputs(cfg, str(run_tag))
    paper_results.ensure_output_dirs(outputs)
    _RUN_OUTPUTS = outputs
    _RUN_MANIFEST = paper_results.ArtifactManifest(outputs.root)

    clim_file = cfg.eval.get("climatology_predictions_file", None)
    targets_file = cfg.eval.get("targets_predictions_file", None)
    prediction_files = _prediction_file_entries(cfg)
    selected_targets = _target_entries(cfg)
    products = paper_run.product_selection(cfg)
    run_climatology = products.climatology
    run_targets = products.targets

    if cfg.eval.get("prediction_files", None) is not None or cfg.eval.get("targets", None) is not None:
        logger.info(
            "Legacy eval.prediction_files/eval.targets schema is supported; "
            "prefer eval.inputs plus eval.target_set for reusable configs."
        )

    if not clim_file and not targets_file and not prediction_files:
        raise ValueError(
            "Provide at least one of eval.climatology_predictions_file, "
            "eval.targets_predictions_file, eval.prediction_files, or eval.inputs."
        )

    for line in paper_run.preflight_lines(
        str(run_tag), outputs.root, products, prediction_files, selected_targets
    ):
        logger.info(line)

    initialize_wandb(
        project=cfg.wandb.get("project", "CorrDiff-Paper"),
        entity=cfg.wandb.get("entity", "shiemn"),
        name=f"paper-eval-{run_tag}",
        group="CorrDiff-Paper",
        mode=cfg.wandb.get("mode", "online"),
        config=OmegaConf.to_container(cfg, resolve=True),
        results_dir=cfg.wandb.get("results_dir", "./wandb"),
    )

    wandb_payload: dict = {}
    json_out: dict = {
        "run_tag": run_tag,
        "output_root": str(outputs.root),
        "schema_version": 2,
    }

    if prediction_files:
        if run_climatology:
            wandb_payload.update(_run_climatology_multifile(
                cfg, logger, prediction_files, json_out
            ))
        if selected_targets and run_targets:
            wandb_payload.update(_run_targets_from_entries(
                cfg, logger, prediction_files, selected_targets, json_out
            ))

    if clim_file and run_climatology:
        abs_path = to_absolute_path(str(clim_file))
        if not os.path.isfile(abs_path):
            raise FileNotFoundError(f"climatology_predictions_file not found: {abs_path}")
        wandb_payload.update(_run_climatology(cfg, logger, abs_path, json_out))

    if targets_file and run_targets:
        abs_path = to_absolute_path(str(targets_file))
        if not os.path.isfile(abs_path):
            raise FileNotFoundError(f"targets_predictions_file not found: {abs_path}")
        wandb_payload.update(_run_targets(cfg, logger, abs_path, json_out))

    # Single log of all figures/tables (climatological + targeted).
    if wandb_payload:
        wandb.log(wandb_payload, commit=True)

    paper_results.write_json(outputs.json_path, json_out)
    if _RUN_MANIFEST is not None:
        _RUN_MANIFEST.add("paper_eval_results", outputs.json_path, "json")
        _RUN_MANIFEST.write(
            outputs.manifest_path,
            metadata={
                "run_tag": str(run_tag),
                "n_prediction_inputs": len(prediction_files),
                "n_selected_targets": len(selected_targets),
            },
        )
    logger.info(f"Results saved to: {outputs.json_path}")
    logger.info(f"Manifest saved to: {outputs.manifest_path}")

    wandb.finish()
    logger.info("Paper-protocol evaluation complete.")


if __name__ == "__main__":
    main()
