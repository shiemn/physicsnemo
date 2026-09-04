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

"""Unified evaluation script for CorrDiff models.

Generates predictions for a fixed set of evaluation timesteps and computes
standardised probabilistic metrics (RMSE, CRPS, twCRPS, Spread-Skill).
Results are logged to a fresh W&B run and saved as a JSON backup.

Diagnostic plots (spread-skill reliability, log histogram, RAPSD, georeferenced
event maps) are generated on rank 0 and logged to W&B as images.

Time-step parallelism: each GPU handles a disjoint subset of the evaluation
timesteps.  All ensemble members for a given timestep are generated on the
same GPU.  Metrics are reduced via all_reduce so every rank holds the same
global totals at the end.

Supported inference modes:
    regression   – deterministic ensemble from the regression model only
                   (all N members are the same prediction, spread = 0)
    all          – regression mean + stochastic diffusion residuals

Usage:
    # Single GPU:
    python evaluate.py --config-name=evaluate run_tag=my_experiment \\
        generation.io.reg_ckpt_filename=/path/to/reg.mdlus \\
        generation.io.res_ckpt_filename=/path/to/diff.mdlus

    # Multi-GPU (4 GPUs):
    torchrun --nproc_per_node=4 evaluate.py --config-name=evaluate \\
        run_tag=my_experiment \\
        generation.io.reg_ckpt_filename=/path/to/reg.mdlus \\
        generation.io.res_ckpt_filename=/path/to/diff.mdlus
"""

import json
import os
import sys
import inspect
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

from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.launch.logging.wandb import initialize_wandb
from physicsnemo.utils.corrdiff import regression_step, diffusion_step
from physicsnemo.utils.corrdiff.utils import NetCDFWriter

from datasets.dataset import register_dataset
from helpers.generate_helpers import (
    build_sampler_fn,
    generate_ensemble,
    get_dataset_and_sampler,
    load_model,
    load_models,
    load_timestep_tensors,
    maybe_compile_models,
    resolve_seed_batches,
    resolve_times,
    save_images,
    setup_patching,
)
from helpers.climate_signal import ClimateAccumulator
from helpers.metrics import MetricsAccumulator
from helpers.plots import (
    HistogramAccumulator,
    RAPSDAccumulator,
    plot_diagnostic_panel,
    plot_example_event,
)


def _format_channel_label(channel) -> str:
    """Convert dataset channel metadata to the string labels used in outputs."""
    if isinstance(channel, str):
        return channel

    name = getattr(channel, "name", None)
    if name is None:
        return str(channel)

    level = getattr(channel, "level", "") or ""
    return f"{name}{level}"


def _normalize_channel_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _infer_channel_type(channel_name: str) -> str:
    normalized = _normalize_channel_name(channel_name)
    if normalized in {
        "maximumradarreflectivity",
        "radarreflectivity",
        "reflectivity",
        "dbz",
    }:
        return "reflectivity"
    if normalized in {
        "t2m",
        "temperature2m",
        "airtemperature2m",
        "tas",
        "temperature",
    }:
        return "temperature"
    if normalized in {"u10", "v10"} or "wind" in normalized:
        return "wind"
    if any(token in normalized for token in ["precip", "rain", "tp", "rr"]):
        return "precip"
    return "generic"


def _is_nonnegative_channel(channel_name: str) -> bool:
    """Return True for physical channels that should not go below zero."""
    channel_type = _infer_channel_type(channel_name)
    return channel_type in {"precip", "reflectivity"}


def _nonnegative_channel_indices(channel_names: list[str]) -> list[int]:
    return [
        idx for idx, channel_name in enumerate(channel_names)
        if _is_nonnegative_channel(channel_name)
    ]


def _clamp_nonnegative_channels(
    tensor: torch.Tensor, channel_indices: list[int]
) -> torch.Tensor:
    """Clamp only explicitly nonnegative channels, preserving signed variables."""
    if not channel_indices:
        return tensor
    if tensor.ndim == 4:
        tensor[:, channel_indices] = tensor[:, channel_indices].clamp(min=0.0)
        return tensor
    if tensor.ndim == 3:
        tensor[channel_indices] = tensor[channel_indices].clamp(min=0.0)
        return tensor
    raise ValueError(
        f"Expected tensor with shape (N,C,H,W) or (C,H,W), got {tuple(tensor.shape)}"
    )


def _infer_channel_unit(channel_type: str) -> str | None:
    if channel_type == "reflectivity":
        return "dBZ"
    if channel_type == "temperature":
        return "K"
    if channel_type == "wind":
        return "m/s"
    if channel_type == "precip":
        return "mm"
    return None


def _select_diagnostic_group(cfg: DictConfig, groups: dict, channel_names: list[str]) -> tuple[object, dict]:
    selection = cfg.eval.get("diagnostic_channel", "auto")

    if selection == "auto":
        gname, g = next(iter(groups.items()))
        label = gname if gname is not None else channel_names[0]
        return gname, {
            "label": label,
            "channel_type": _infer_channel_type(label),
            "unit": _infer_channel_unit(_infer_channel_type(label)),
        }

    if isinstance(selection, int):
        selected_index = int(selection)
    elif isinstance(selection, str) and selection.isdigit():
        selected_index = int(selection)
    else:
        selected_index = None

    if selected_index is not None:
        for gname, g in groups.items():
            if g.get("channels") == [selected_index]:
                label = channel_names[selected_index]
                channel_type = _infer_channel_type(label)
                return gname, {
                    "label": label,
                    "channel_type": channel_type,
                    "unit": _infer_channel_unit(channel_type),
                }
        raise ValueError(
            f"diagnostic_channel={selection!r} requires a matching single-channel metric group"
        )

    normalized_selection = _normalize_channel_name(str(selection))
    for gname, g in groups.items():
        candidate_names = []
        if gname is not None:
            candidate_names.append(str(gname))
        channels = g.get("channels")
        if channels is not None:
            candidate_names.extend(channel_names[idx] for idx in channels)
        else:
            candidate_names.extend(channel_names)

        for candidate in candidate_names:
            if _normalize_channel_name(candidate) == normalized_selection:
                channel_type = _infer_channel_type(candidate)
                return gname, {
                    "label": candidate,
                    "channel_type": channel_type,
                    "unit": _infer_channel_unit(channel_type),
                }

    raise ValueError(f"Could not resolve diagnostic_channel={selection!r}")



def _diagnostic_plot_payload(
    acc_label: str,
    metrics_dict: dict,
    hist_acc: "HistogramAccumulator",
    rapsd_acc: "RAPSDAccumulator",
    rapsd_dx_km: float,
    diagnostic_info: dict,
) -> dict:
    """Generate a combined diagnostic panel and return a dict of {key: wandb.Image}.

    Does NOT call wandb.log — the caller collects all payloads into a single log call.
    """
    payload = {}
    fig = plot_diagnostic_panel(
        metrics_dict=metrics_dict,
        acc_label=acc_label,
        hist_acc=hist_acc,
        rapsd_acc=rapsd_acc,
        rapsd_dx_km=rapsd_dx_km,
        diagnostic_info=diagnostic_info,
    )
    if fig is not None:
        payload[f"{acc_label}/diagnostics"] = wandb.Image(fig)
        plt.close(fig)
    return payload


def _load_predictions_netcdf(path: str) -> dict:
    """Load predictions and truth from a NetCDF file produced by generate.py or evaluate.py.

    NetCDF structure (groups created by NetCDFWriter):
        Root: lat(y,x), lon(y,x), time(time)
        /truth/{channel}(time, y, x)                — ground truth (denormalized)
        /prediction/{channel}(ensemble, time, y, x) — ensemble predictions (denormalized)
        /input/{channel}(time, y, x)                — LR inputs (denormalized)

    Returns:
        dict with keys:
            lat:            (H, W) numpy array
            lon:            (H, W) numpy array
            channel_names:  list[str]
            n_times:        int
            n_ensemble:     int
            truth:          (T, C, H, W) numpy — stacked over channels
            prediction:     (N_ens, T, C, H, W) numpy
            times:          list of decoded timestamps, or [] if undecodable
    """
    f = nc.Dataset(path, "r")

    if "truth" not in f.groups or "prediction" not in f.groups:
        f.close()
        raise ValueError(
            f"NetCDF file {path} is missing required groups (truth/prediction). "
            "The file may be corrupt or incomplete from a crashed run."
        )

    lat = f["lat"][:]
    lon = f["lon"][:]

    truth_group = f.groups["truth"]
    pred_group = f.groups["prediction"]
    channel_names = list(truth_group.variables.keys())

    # Stack channels: truth is (time, y, x) per channel -> (T, C, H, W)
    truth_arrays = [truth_group[ch][:] for ch in channel_names]  # list of (T, H, W)
    truth = np.stack(truth_arrays, axis=1)  # (T, C, H, W)

    pred_arrays = [pred_group[ch][:] for ch in channel_names]  # list of (N, T, H, W)
    prediction = np.stack(pred_arrays, axis=2)  # (N, T, C, H, W)

    # Decode timestamps so offline event plots can be labelled with the real
    # time, exactly as the online path does. Lead-time files store strings;
    # everything else stores cftime offsets. Labels are cosmetic, so a file
    # written without usable times degrades to an empty list rather than
    # failing the whole evaluation.
    times = []
    if "time" in f.variables:
        time_v = f.variables["time"]
        try:
            if getattr(time_v, "dtype", None) == str or time_v.dtype.kind in "SU":
                times = [str(t) for t in time_v[:]]
            else:
                times = list(
                    nc.num2date(time_v[:], time_v.units, calendar=time_v.calendar)
                )
        except (AttributeError, ValueError, TypeError):
            times = []

    f.close()
    return {
        "lat": lat,
        "lon": lon,
        "channel_names": channel_names,
        "n_times": truth.shape[0],
        "n_ensemble": prediction.shape[0],
        "truth": truth,
        "prediction": prediction,
        "times": times,
    }


def _evaluate_from_file(cfg: DictConfig, predictions_path: str) -> None:
    """Offline evaluation: load predictions + truth from NetCDF, compute metrics.

    No models, dataset, or GPU needed. Runs on a single process.
    """
    logger = PythonLogger("evaluate")
    logger.file_logging("evaluate.log")

    run_tag = cfg.get("run_tag", "eval")
    precip_threshold = cfg.eval.get("precip_threshold", 1.0)
    output_json = cfg.eval.get("output_json", "eval_results.json")
    twcrps_thresholds = list(cfg.eval.get("twcrps_thresholds", [5.0, 10.0]))
    n_plot_events = cfg.eval.get("n_plot_events", 5)
    plot_events = list(cfg.eval.get("plot_events", None) or [])
    rapsd_dx_km = float(cfg.eval.get("rapsd_dx_km", 2.0))
    spread_skill_bin_mode = cfg.eval.get("spread_skill_bin_mode", "quantile")

    logger.info(f"Loading predictions from: {predictions_path}")
    data = _load_predictions_netcdf(predictions_path)
    n_times = data["n_times"]
    n_ens = data["n_ensemble"]
    channel_names = data["channel_names"]
    lat_np = data["lat"]
    lon_np = data["lon"]
    img_shape = (data["truth"].shape[2], data["truth"].shape[3])
    nonnegative_channels = _nonnegative_channel_indices(channel_names)

    logger.info(f"  Loaded {n_times} timesteps, {n_ens} ensemble members, "
                f"{len(channel_names)} channels, shape {img_shape}")
    if nonnegative_channels:
        logger.info(
            "  Clamping nonnegative channels only: "
            + ", ".join(channel_names[i] for i in nonnegative_channels)
        )

    if n_times == 0:
        logger.warning(
            f"Predictions file {predictions_path} contains 0 timesteps — "
            "it may be corrupt or incomplete from a crashed run. "
            "Delete the file and re-run to regenerate predictions."
        )
        return

    # A one-member diffusion sample is still a diffusion prediction.
    is_regression = cfg.generation.inference_mode == "regression"

    # W&B init
    initialize_wandb(
        project=cfg.wandb.get("project", "evaluation"),
        entity=cfg.wandb.get("entity", "shiemn"),
        name=f"eval-{run_tag}",
        group="CorrDiff-Eval",
        mode=cfg.wandb.get("mode", "online"),
        config=OmegaConf.to_container(cfg, resolve=True),
        results_dir=cfg.wandb.get("results_dir", "./wandb"),
    )

    device = torch.device("cpu")

    # Build metric groups (per-channel or single accumulator)
    per_channel_metrics = cfg.eval.get("per_channel_metrics", False)
    _metric_groups_raw = cfg.eval.get("metric_groups", None)
    metric_groups_cfg = (
        OmegaConf.to_container(_metric_groups_raw, resolve=True)
        if _metric_groups_raw is not None
        else None
    )
    if per_channel_metrics and metric_groups_cfg is None:
        _skip_cond = cfg.eval.get("skip_conditional_metrics", False)
        metric_groups_cfg = {
            name: {"channels": [i], "skip_conditional_metrics": _skip_cond}
            for i, name in enumerate(channel_names)
        }

    def _make_file_acc(gcfg):
        return MetricsAccumulator(
            precip_threshold=gcfg.get("precip_threshold", precip_threshold),
            device=device,
            twcrps_thresholds=gcfg.get("twcrps_thresholds", twcrps_thresholds),
            hrre_threshold=gcfg.get("hrre_threshold", 10.0),
            skip_conditional_metrics=gcfg.get("skip_conditional_metrics", False),
            skip_spread_skill=is_regression or gcfg.get("skip_spread_skill", False),
            bin_mode=spread_skill_bin_mode,
        )

    # Must mirror the online path's fallback (see main()): without this the
    # offline rerun scores every channel while the first, online run scored
    # only eval.metric_channels -- same config, different numbers.
    metric_channels = cfg.eval.get("metric_channels", None)
    if metric_channels is not None:
        metric_channels = list(metric_channels)

    if metric_groups_cfg:
        _groups = {
            gname: {"channels": gcfg.get("channels"), "acc": _make_file_acc(gcfg)}
            for gname, gcfg in metric_groups_cfg.items()
        }
    else:
        _groups = {
            None: {
                "channels": metric_channels,
                "acc": MetricsAccumulator(
                    precip_threshold=precip_threshold,
                    device=device,
                    twcrps_thresholds=twcrps_thresholds,
                    skip_spread_skill=is_regression,
                    bin_mode=spread_skill_bin_mode,
                ),
            }
        }

    diagnostic_group_name, diagnostic_info = _select_diagnostic_group(cfg, _groups, channel_names)
    diagnostic_group = _groups[diagnostic_group_name]
    diagnostic_channels = diagnostic_group["channels"]
    hist_acc = HistogramAccumulator(device=device)
    rapsd_acc = RAPSDAccumulator(img_shape=img_shape, dx_km=rapsd_dx_km, device=device)

    event_candidates = []

    for t in range(n_times):
        pred_t = torch.from_numpy(data["prediction"][:, t])  # (N_ens, C, H, W)
        tar_t = torch.from_numpy(data["truth"][t])          # (C, H, W)
        pred_t = _clamp_nonnegative_channels(pred_t, nonnegative_channels)
        tar_t = _clamp_nonnegative_channels(tar_t, nonnegative_channels)

        for g in _groups.values():
            ch = g["channels"]
            p = pred_t[:, ch] if ch is not None else pred_t
            tg = tar_t[ch] if ch is not None else tar_t
            g["acc"].update(p, tg)

        p_hist = pred_t[:, diagnostic_channels] if diagnostic_channels is not None else pred_t
        tg_hist = tar_t[diagnostic_channels] if diagnostic_channels is not None else tar_t
        hist_acc.update(p_hist, tg_hist)
        rapsd_acc.update(p_hist, tg_hist)
        event_candidates.append((t, float(tar_t.max()), t))

    acc_label = "regression" if is_regression else "diffusion"
    all_metrics = {}
    for gname, g in _groups.items():
        prefix = f"{acc_label}/" if gname is None else f"{acc_label}/{gname}/"
        all_metrics.update(g["acc"].to_dict(prefix=prefix))

    wandb_payload = dict(all_metrics)

    # Console summary
    logger.info("=" * 70)
    logger.info("EVALUATION RESULTS (from file)")
    logger.info("=" * 70)
    for k, v in all_metrics.items():
        if isinstance(v, float):
            logger.info(f"  {k}: {v:.6f}")
        elif not isinstance(v, list):
            logger.info(f"  {k}: {v}")

    # JSON backup
    with open(output_json, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"Results saved to: {output_json}")

    diagnostic_acc_dict = diagnostic_group["acc"].to_dict(prefix=f"{acc_label}/")
    wandb_payload.update(_diagnostic_plot_payload(
        acc_label=acc_label,
        metrics_dict=diagnostic_acc_dict,
        hist_acc=hist_acc,
        rapsd_acc=rapsd_acc,
        rapsd_dx_km=rapsd_dx_km,
        diagnostic_info=diagnostic_info,
    ))

    # Event plots
    file_times = data.get("times") or []
    plot_channels_cfg = cfg.eval.get("plot_channels", None)
    if plot_channels_cfg is not None:
        plot_channels_cfg = list(plot_channels_cfg)
    explicit_set = set(plot_events)
    sorted_cands = sorted(event_candidates, key=lambda x: -x[1])
    auto_set = {c[0] for c in sorted_cands[:n_plot_events]}
    plot_set = explicit_set | auto_set

    # Step 0: diagnostic panels + scalar metrics
    scalar_payload = {k: v for k, v in wandb_payload.items() if not isinstance(v, wandb.Image)}
    wandb.summary.update(scalar_payload)
    wandb.log(wandb_payload, step=0, commit=True)

    # Steps 1..N: one step per event so the slider navigates between events
    if plot_set:
        logger.info(f"Generating {len(plot_set)} example event plot(s)...")
        for event_step, time_idx in enumerate(sorted(plot_set), start=1):
            if time_idx >= n_times:
                continue
            pred_ens_np = data["prediction"][:, time_idx]  # (N_ens, C, H, W)
            target_np = data["truth"][time_idx]             # (C, H, W)
            reg_mean_np = pred_ens_np[0] if is_regression else None
            max_precip = float(target_np.max())

            time_label = (
                str(file_times[time_idx])
                if time_idx < len(file_times)
                else str(time_idx)
            )
            fig = plot_example_event(
                pred_ens_np=pred_ens_np,
                target_np=target_np,
                reg_mean_np=reg_mean_np,
                time_str=time_label,
                channel_names=channel_names,
                lat=lat_np,
                lon=lon_np,
                plot_channels=plot_channels_cfg,
            )
            wandb.log(
                {
                    "event/plot": wandb.Image(
                        fig, caption=f"t{time_idx} {time_label} max={max_precip:.1f}mm"
                    ),
                    "event/time_idx": time_idx,
                    "event/max_precip_mm": max_precip,
                },
                step=event_step,
                commit=True,
            )
            plt.close(fig)

    wandb.finish()
    logger.info("Evaluation from file complete.")


def _run_single_timestep(
    dataset,
    dataset_idx: int,
    net_reg,
    net_res,
    sampler_fn,
    use_dropout_residual: bool,
    img_shape: tuple[int, int],
    img_out_channels: int,
    device,
    hr_mean_conditioning: bool,
    diffusion_kwargs: dict,
    seed_batches,
    nonnegative_channels: list[int] | None = None,
    seed_mode: str = "fixed",
    timestamp=None,
    num_ensembles: int | None = None,
    seed_base: int = 0,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
    """Run regression + diffusion inference for a single timestep on one GPU.

    Seeds and non-negative clamping are resolved exactly as in the main
    evaluation loop, so the ensemble drawn here is the one that was scored.
    Pass ``seed_mode``/``timestamp``/``num_ensembles``/``seed_base`` and
    ``nonnegative_channels`` through from the caller; omitting them reproduces
    the fixed-seed, unclamped behaviour.

    Returns:
        reg_mean_np:  (C, H, W) regression prediction, or None for direct diffusion.
        ens_pred_np:  (N_ens, C, H, W) full ensemble in physical units, or same
                      as reg_mean_np if net_res is None.
        target_np:    (C, H, W) ground truth in physical units (mm).
    """
    image_tar, image_lr, lead_time_label = load_timestep_tensors(
        dataset, dataset_idx, device
    )

    reg_mean = None
    if net_reg is not None:
        with torch.no_grad():
            reg_mean = regression_step(
                net=net_reg,
                img_lr=image_lr,
                latents_shape=(1, img_out_channels, img_shape[0], img_shape[1]),
                lead_time_label=lead_time_label,
            )[0:1]
    elif hr_mean_conditioning:
        raise ValueError("Direct diffusion requires hr_mean_conditioning=false")

    def _to_physical(tensor):
        """Denormalize then clamp, matching the metric path."""
        array = dataset.denormalize_output(tensor.cpu().numpy())
        clamped = _clamp_nonnegative_channels(
            torch.from_numpy(array), nonnegative_channels
        )
        return clamped.numpy()

    reg_mean_np = _to_physical(reg_mean)[0] if reg_mean is not None else None      # (C, H, W)
    target_np = _to_physical(image_tar)[0]       # (C, H, W)

    if net_res is not None and (use_dropout_residual or sampler_fn is not None):
        timestep_seed_batches = resolve_seed_batches(
            seed_batches,
            seed_mode=seed_mode,
            timestamp=timestamp,
            num_ensembles=num_ensembles,
            seed_base=seed_base,
        )
        diffusion_residuals = generate_ensemble(
            net_res=net_res,
            sampler_fn=sampler_fn,
            use_dropout_residual=use_dropout_residual,
            image_lr=image_lr,
            img_shape=img_shape,
            img_out_channels=img_out_channels,
            device=device,
            mean_hr=reg_mean if hr_mean_conditioning else None,
            lead_time_label=lead_time_label,
            seed_batches=timestep_seed_batches,
            diffusion_kwargs=diffusion_kwargs,
        )
        ens_pred = (
            diffusion_residuals if reg_mean is None else reg_mean + diffusion_residuals
        )
        ens_pred_np = _to_physical(ens_pred)                 # (N_ens, C, H, W)
    else:
        ens_pred_np = reg_mean_np[np.newaxis]  # (1, C, H, W)

    return reg_mean_np, ens_pred_np, target_np


@hydra.main(version_base="1.2", config_path="conf", config_name="evaluate")
def main(cfg: DictConfig) -> None:
    """Evaluate a CorrDiff checkpoint on the configured eval timesteps."""

    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device

    logger = PythonLogger("evaluate")
    logger0 = RankZeroLoggingWrapper(logger, dist)
    logger.file_logging("evaluate.log")

    # ------------------------------------------------------------------
    # Config extraction
    # ------------------------------------------------------------------
    run_tag = cfg.get("run_tag", "eval")
    inference_mode = cfg.generation.inference_mode
    if inference_mode not in ("regression", "diffusion", "all"):
        raise ValueError(
            f'Unsupported inference_mode={inference_mode!r}. '
            f'Must be "regression", "diffusion", or "all" (regression + diffusion).'
        )
    has_regression = inference_mode in ("regression", "all")
    has_diffusion = inference_mode in ("diffusion", "all")
    if not has_regression and cfg.generation.hr_mean_conditioning:
        raise ValueError("Direct diffusion requires generation.hr_mean_conditioning=false")
    num_ensembles = cfg.generation.num_ensembles
    seed_batch_size = cfg.generation.seed_batch_size
    climate_cfg_raw = cfg.eval.get("climate", None)
    climate_cfg = (
        OmegaConf.to_container(climate_cfg_raw, resolve=True)
        if climate_cfg_raw is not None
        else {}
    )
    climate_enabled = bool(climate_cfg.get("enabled", False))
    climate_only = bool(cfg.eval.get("climate_only", False))
    if climate_only and not climate_enabled:
        raise ValueError("eval.climate_only=true requires eval.climate.enabled=true")
    if climate_only:
        if dist.world_size != 1:
            raise ValueError("Climate-only aggregation requires exactly one GPU/process")
        if inference_mode != "all" or num_ensembles != 1:
            raise ValueError(
                "Climate-only evaluation requires generation.inference_mode=all "
                "and generation.num_ensembles=1"
            )
        if cfg.generation.get("seed_mode", "fixed") != "timestamp":
            raise ValueError(
                "Climate-only evaluation requires generation.seed_mode=timestamp"
            )
    precip_threshold = cfg.eval.get("precip_threshold", 1.0)
    output_json = cfg.eval.get("output_json", "eval_results.json")
    twcrps_thresholds = list(cfg.eval.get("twcrps_thresholds", [5.0, 10.0]))
    n_plot_events = cfg.eval.get("n_plot_events", 5)
    plot_events = list(cfg.eval.get("plot_events", None) or [])
    rapsd_dx_km = float(cfg.eval.get("rapsd_dx_km", 2.0))
    predictions_file_cfg = cfg.eval.get("predictions_file", None)
    stream_predictions = bool(cfg.eval.get("stream_predictions", False))
    save_prediction_inputs = bool(cfg.eval.get("save_inputs", True))
    prediction_sync_interval = int(cfg.eval.get("prediction_sync_interval", 32))
    if prediction_sync_interval < 1:
        raise ValueError("eval.prediction_sync_interval must be at least 1")
    save_prediction_channels_cfg = cfg.eval.get("save_prediction_channels", None)
    if save_prediction_channels_cfg is not None:
        save_prediction_channels_cfg = list(save_prediction_channels_cfg)
    # metric_groups: optional dict of {group_name: {channels, precip_threshold,
    #   twcrps_thresholds, skip_conditional_metrics, skip_spread_skill, hrre_threshold}}
    # If null, falls back to legacy single-accumulator behaviour using metric_channels.
    _metric_groups_raw = cfg.eval.get("metric_groups", None)
    metric_groups_cfg = (
        OmegaConf.to_container(_metric_groups_raw, resolve=True)
        if _metric_groups_raw is not None
        else None
    )
    per_channel_metrics = cfg.eval.get("per_channel_metrics", False)
    log_regression = has_regression and cfg.eval.get("log_regression", True)
    plot_channels_cfg = cfg.eval.get("plot_channels", None)
    if plot_channels_cfg is not None:
        plot_channels_cfg = list(plot_channels_cfg)

    # Resolve "auto" → derive filename from the primary checkpoint name.
    # Done early so the offline-load check works, but we need the checkpoint
    # path from the config (not the loaded model).
    if predictions_file_cfg == "auto":
        # Use diffusion checkpoint if available, otherwise regression
        if has_diffusion:
            _ckpt = cfg.generation.io.res_ckpt_filename
        else:
            _ckpt = cfg.generation.io.reg_ckpt_filename
        _stem = os.path.splitext(os.path.basename(str(_ckpt)))[0]
        predictions_file = f"eval_{_stem}.nc"
    else:
        predictions_file = predictions_file_cfg  # explicit path or None
    if climate_only and predictions_file is not None:
        raise ValueError("Climate-only evaluation requires eval.predictions_file=null")

    # ------------------------------------------------------------------
    # Offline mode: load predictions from existing NetCDF file
    # ------------------------------------------------------------------
    if predictions_file is not None:
        abs_pred_path = to_absolute_path(predictions_file)
        if os.path.isfile(abs_pred_path):
            logger0.info(f"Predictions file exists — running offline evaluation from: {abs_pred_path}")
            if dist.rank == 0:
                _evaluate_from_file(cfg, abs_pred_path)
            if dist.world_size > 1:
                torch.distributed.barrier()
            return

    logger0.info(f"=== CorrDiff Unified Evaluation ===")
    metric_channels = cfg.eval.get("metric_channels", None)
    if metric_channels is not None:
        metric_channels = list(metric_channels)
    spread_skill_bin_mode = cfg.eval.get("spread_skill_bin_mode", "quantile")

    logger0.info(f"  Run tag:        {run_tag}")
    logger0.info(f"  Inference mode: {inference_mode}")
    logger0.info(f"  Ensembles:      {num_ensembles}")
    logger0.info(f"  Climate only:   {climate_only}")
    logger0.info(f"  Stream output:  {stream_predictions}")
    logger0.info(f"  Save inputs:    {save_prediction_inputs}")
    logger0.info(f"  Threshold:      {precip_threshold} mm")
    logger0.info(f"  twCRPS at:      {twcrps_thresholds} mm")
    logger0.info(f"  Plot events:    top-{n_plot_events} (+ {len(plot_events)} explicit)")
    if metric_groups_cfg:
        logger0.info(f"  Metric groups:  {list(metric_groups_cfg.keys())}")
    else:
        logger0.info(f"  Metric channels:{metric_channels if metric_channels is not None else 'all'}")
    logger0.info(f"  GPUs:           {dist.world_size}")
    residual_model_type = cfg.generation.get("residual_model_type", "diffusion")
    use_dropout_residual = residual_model_type in {"dropout_crps", "dropout_residual"}
    if use_dropout_residual and inference_mode != "all":
        raise ValueError("dropout residual evaluation requires generation.inference_mode=all")

    # ------------------------------------------------------------------
    # W&B initialisation (rank 0 only)
    # ------------------------------------------------------------------
    if dist.rank == 0 and not climate_only:
        guidance_scale = float(cfg.generation.get("guidance_scale", 0.0))
        guidance_schedule_alpha = float(cfg.generation.get("guidance_schedule_alpha", 0.0))
        wandb_name = f"eval-{run_tag}-g{guidance_scale}" if guidance_scale != 0.0 else f"eval-{run_tag}"
        if guidance_schedule_alpha != 0.0:
            wandb_name += f"-a{guidance_schedule_alpha}"
        initialize_wandb(
            project=cfg.wandb.get("project", "evaluation"),
            entity=cfg.wandb.get("entity", "shiemn"),
            name=wandb_name,
            group="CorrDiff-Eval",
            mode=cfg.wandb.get("mode", "online"),
            config=OmegaConf.to_container(cfg, resolve=True),
            results_dir=cfg.wandb.get("results_dir", "./wandb"),
        )

    # ------------------------------------------------------------------
    # Dataset and eval timesteps
    # ------------------------------------------------------------------
    times = resolve_times(cfg.generation)

    dataset_cfg = OmegaConf.to_container(cfg.dataset)
    register_dataset(cfg.dataset.type)

    has_lead_time = cfg.generation.get("has_lead_time", False)
    dataset, sampler = get_dataset_and_sampler(
        dataset_cfg=dataset_cfg, times=times, has_lead_time=has_lead_time
    )
    total_times = len(sampler)

    # Rebuild times as cftime objects (needed by NetCDFWriter.write_time)
    all_dataset_times = dataset.time()
    times = [all_dataset_times[i] for i in sampler] if sampler else all_dataset_times
    excluded_month_days = {
        str(value)
        for value in climate_cfg.get("excluded_month_days", [])
    }
    if climate_enabled and excluded_month_days:
        kept = [
            (dataset_idx, timestamp)
            for dataset_idx, timestamp in zip(sampler, times)
            if f"{int(timestamp.month):02d}-{int(timestamp.day):02d}"
            not in excluded_month_days
        ]
        sampler = [item[0] for item in kept]
        times = [item[1] for item in kept]
        total_times = len(sampler)
        logger0.info(
            "  Excluded month-days: " + ", ".join(sorted(excluded_month_days))
        )

    img_shape = dataset.image_shape()
    output_channels = list(dataset.output_channels())
    img_out_channels = len(output_channels)
    channel_names = [_format_channel_label(channel) for channel in output_channels]
    if save_prediction_channels_cfg is None:
        saved_output_channel_indices = list(range(img_out_channels))
    else:
        if len(set(save_prediction_channels_cfg)) != len(save_prediction_channels_cfg):
            raise ValueError("eval.save_prediction_channels must not contain duplicates")
        unknown_saved_channels = [
            name for name in save_prediction_channels_cfg if name not in channel_names
        ]
        if unknown_saved_channels:
            raise ValueError(
                "Unknown eval.save_prediction_channels value(s): "
                + ", ".join(unknown_saved_channels)
                + f". Available channels: {', '.join(channel_names)}"
            )
        saved_output_channel_indices = [
            channel_names.index(name) for name in save_prediction_channels_cfg
        ]
    saved_output_channels = [
        output_channels[channel_idx] for channel_idx in saved_output_channel_indices
    ]
    logger0.info(
        "  Saved channels:  "
        + ", ".join(channel_names[i] for i in saved_output_channel_indices)
    )
    nonnegative_channels = _nonnegative_channel_indices(channel_names)
    if nonnegative_channels:
        logger0.info(
            "  Clamping nonnegative channels only: "
            + ", ".join(channel_names[i] for i in nonnegative_channels)
        )

    # Auto-expand per_channel_metrics into one group per output channel
    if per_channel_metrics and metric_groups_cfg is None:
        _skip_cond = cfg.eval.get("skip_conditional_metrics", False)
        metric_groups_cfg = {
            name: {"channels": [i], "skip_conditional_metrics": _skip_cond}
            for i, name in enumerate(channel_names)
        }

    # Lat/lon for georeferenced plots (xarray.DataArray -> numpy)
    try:
        lat_np = np.array(dataset.latitude())
        lon_np = np.array(dataset.longitude())
    except AttributeError:
        lat_np = None
        lon_np = None

    climate_accumulator = None
    climate_channel_idx = None
    climate_output_path = None
    if climate_enabled:
        climate_channel = str(climate_cfg.get("channel", "precipitation"))
        normalized_climate_channel = _normalize_channel_name(climate_channel)
        matching_channels = [
            idx
            for idx, name in enumerate(channel_names)
            if _normalize_channel_name(name) == normalized_climate_channel
        ]
        if len(matching_channels) != 1:
            raise ValueError(
                f"Climate channel {climate_channel!r} did not uniquely match "
                f"output channels {channel_names}"
            )
        climate_channel_idx = matching_channels[0]
        climate_output_cfg = climate_cfg.get("output_file")
        if not climate_output_cfg:
            raise ValueError("eval.climate.output_file must be set")
        climate_output_path = to_absolute_path(str(climate_output_cfg))
        climate_accumulator = ClimateAccumulator(
            shape=img_shape,
            wet_day_threshold_mm=float(
                climate_cfg.get("wet_day_threshold_mm", 1.0)
            ),
            expected_hours=climate_cfg.get(
                "expected_hours", [0, 3, 6, 9, 12, 15, 18, 21]
            ),
        )
        logger0.info(
            f"  Climate output: {climate_output_path} "
            f"(channel={channel_names[climate_channel_idx]})"
        )

    logger0.info(f"  Eval timesteps: {total_times} (matched {total_times}/{len(times)})")

    if total_times == 0:
        logger0.error("No matching timesteps found in dataset. Aborting.")
        return

    # ------------------------------------------------------------------
    # Patching
    # ------------------------------------------------------------------
    patching, img_shape = setup_patching(cfg, img_shape)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    # Build EDM2 kwargs if config section is present (needed for .pt checkpoints)
    edm2_kwargs = None
    if cfg.generation.get("edm2") is not None:
        img_in_channels = len(dataset.input_channels())
        if cfg.generation.hr_mean_conditioning:
            img_in_channels += img_out_channels
        edm2_kwargs = {
            "img_resolution": list(img_shape),
            "img_in_channels": img_in_channels,
            "img_out_channels": img_out_channels,
            "use_fp16": cfg.generation.perf.get("use_fp16", False),
            "sigma_data": cfg.generation.edm2.get("sigma_data", 0.5),
        }
        edm2_model_args = cfg.generation.edm2.get("model_args", None)
        if edm2_model_args is not None:
            edm2_kwargs.update(OmegaConf.to_container(edm2_model_args))

    guidance_scale = float(cfg.generation.get("guidance_scale", 0.0))
    guidance_schedule_alpha = float(cfg.generation.get("guidance_schedule_alpha", 0.0))
    guide_ckpt = cfg.generation.io.get("guide_ckpt_filename", None)

    logger0.info("Loading models...")
    net_reg, net_res = load_models(
        cfg, device,
        load_net_reg=has_regression,
        load_net_res=has_diffusion,
        edm2_kwargs=edm2_kwargs,
    )
    net_guide = None
    if guidance_scale != 0.0 and guide_ckpt:
        logger0.info(f"Loading guidance model (scale={guidance_scale}): {guide_ckpt}")
        net_guide = load_model(to_absolute_path(guide_ckpt), device, cfg.generation.perf, edm2_kwargs)
    elif guidance_scale != 0.0:
        logger0.warning("guidance_scale != 0 but no guide_ckpt_filename — running without guidance")
    net_reg, net_res = maybe_compile_models(cfg, net_reg, net_res)

    # ------------------------------------------------------------------
    # Sampler function (diffusion only)
    # ------------------------------------------------------------------
    sampler_fn = (
        None
        if use_dropout_residual or net_res is None
        else build_sampler_fn(
            cfg.sampler,
            patching,
            net_guide=net_guide,
            guidance_scale=guidance_scale,
            guidance_schedule_alpha=guidance_schedule_alpha,
        )
    )

    # ------------------------------------------------------------------
    # Seed batches for ensemble generation
    # ------------------------------------------------------------------
    seeds = list(np.arange(num_ensembles))
    num_seed_batches = (len(seeds) - 1) // seed_batch_size + 1
    seed_batches = np.array_split(seeds, num_seed_batches)
    seed_mode = str(cfg.generation.get("seed_mode", "fixed"))
    seed_base = int(cfg.generation.get("seed_base", 0))
    if seed_mode not in {"fixed", "timestamp"}:
        raise ValueError("generation.seed_mode must be 'fixed' or 'timestamp'")

    # ------------------------------------------------------------------
    # Metric accumulators (one per inference mode × metric group)
    # ------------------------------------------------------------------
    # _groups is an ordered dict: group_name -> {channels, reg_acc, diff_acc}
    # group_name is None for the legacy single-accumulator case, which preserves
    # the existing WandB key structure (regression/rmse, not regression/default/rmse).
    def _make_acc(gcfg, skip_ss):
        return MetricsAccumulator(
            precip_threshold=gcfg.get("precip_threshold", precip_threshold),
            device=device,
            twcrps_thresholds=gcfg.get("twcrps_thresholds", twcrps_thresholds),
            hrre_threshold=gcfg.get("hrre_threshold", 10.0),
            skip_conditional_metrics=gcfg.get("skip_conditional_metrics", False),
            skip_spread_skill=skip_ss or gcfg.get("skip_spread_skill", False),
            bin_mode=spread_skill_bin_mode,
        )

    if metric_groups_cfg:
        _groups = {
            gname: {
                "channels": gcfg.get("channels"),  # list[int] or None = all
                "reg_acc": _make_acc(gcfg, skip_ss=True),
                "diff_acc": _make_acc(gcfg, skip_ss=False) if has_diffusion else None,
            }
            for gname, gcfg in metric_groups_cfg.items()
        }
    else:
        _groups = {
            None: {  # None key → legacy prefix structure (no group sub-key)
                "channels": metric_channels,
                "reg_acc": _make_acc({}, skip_ss=True),
                "diff_acc": _make_acc({}, skip_ss=False) if has_diffusion else None,
            }
        }

    diagnostic_group_name, diagnostic_info = _select_diagnostic_group(cfg, _groups, channel_names)
    diagnostic_group = _groups[diagnostic_group_name]
    diagnostic_channels = diagnostic_group["channels"]

    # Histogram and RAPSD accumulators
    hist_acc_reg = HistogramAccumulator(device=device)
    rapsd_acc_reg = RAPSDAccumulator(img_shape=img_shape, dx_km=rapsd_dx_km, device=device)
    hist_acc_diff = HistogramAccumulator(device=device) if has_diffusion else None
    rapsd_acc_diff = (
        RAPSDAccumulator(img_shape=img_shape, dx_km=rapsd_dx_km, device=device)
        if has_diffusion
        else None
    )

    # Event candidates for auto-selection of plot timesteps: (time_idx, max_precip, dataset_idx)
    local_event_candidates: list[tuple[int, float, int]] = []

    save_predictions = predictions_file is not None
    if stream_predictions and save_predictions and dist.world_size != 1:
        raise ValueError(
            "eval.stream_predictions currently requires a single process/GPU. "
            "Submit one GPU or disable streaming."
        )

    # The legacy output path buffers tensors and writes after evaluation. Annual
    # time series use streaming to avoid retaining many GiB of predictions and
    # upsampled temporal inputs in host memory.
    local_save_data: list[dict] = []
    streaming_nc_file = None
    streaming_nc_writer = None
    streaming_tmp_path = None
    streaming_final_path = None
    if save_predictions and stream_predictions:
        streaming_final_path = to_absolute_path(predictions_file)
        streaming_tmp_path = streaming_final_path + ".tmp"
        os.makedirs(os.path.dirname(streaming_final_path) or ".", exist_ok=True)
        if os.path.exists(streaming_tmp_path):
            raise FileExistsError(
                f"Incomplete streaming output already exists: {streaming_tmp_path}. "
                "Inspect or move it before retrying so diagnostic data is not overwritten."
            )
        logger0.info(f"Streaming predictions to: {streaming_tmp_path}")
        streaming_nc_file = nc.Dataset(streaming_tmp_path, "w")
        streaming_nc_file.cfg = str(cfg)
        streaming_nc_file.setncattr("save_inputs", int(save_prediction_inputs))
        streaming_nc_file.setncattr("streaming_output", 1)
        streaming_nc_writer = NetCDFWriter(
            streaming_nc_file,
            lat=np.array(dataset.latitude()),
            lon=np.array(dataset.longitude()),
            input_channels=(dataset.input_channels() if save_prediction_inputs else []),
            output_channels=saved_output_channels,
            has_lead_time=cfg.generation.get("has_lead_time", False),
        )

    # ------------------------------------------------------------------
    # Distribution kwargs for diffusion_step
    # ------------------------------------------------------------------
    diffusion_kwargs = {}
    for cfg_key, kwarg_key in [
        ("distribution", "distribution"),
        ("student_t_nu", "nu"),
        ("P_mean", "P_mean"),
        ("P_std", "P_std"),
    ]:
        val = cfg.generation.get(cfg_key, None)
        if val is not None:
            diffusion_kwargs[kwarg_key] = val
    diffusion_step_params = inspect.signature(diffusion_step).parameters
    if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in diffusion_step_params.values()):
        diffusion_kwargs = {
            key: val for key, val in diffusion_kwargs.items() if key in diffusion_step_params
        }

    # ------------------------------------------------------------------
    # Main evaluation loop
    # ------------------------------------------------------------------
    if dist.world_size > 1:
        torch.distributed.barrier()

    max_iters = (total_times + dist.world_size - 1) // dist.world_size
    logger0.info("Starting evaluation loop...")

    for iteration in range(max_iters):
        time_idx = dist.rank + iteration * dist.world_size
        if time_idx >= total_times:
            continue

        dataset_idx = sampler[time_idx]
        logger.info(
            f"[GPU {dist.rank}] {iteration + 1}/{max_iters}: time_idx={time_idx}"
        )

        # Load data (normalized)
        image_tar, image_lr, lead_time_label = load_timestep_tensors(
            dataset, dataset_idx, device
        )

        reg_mean = None
        if has_regression:
            # -- Regression forward pass --
            with torch.no_grad():
                try:
                    image_reg = regression_step(
                        net=net_reg,
                        img_lr=image_lr,
                        latents_shape=(1, img_out_channels, img_shape[0], img_shape[1]),
                        lead_time_label=lead_time_label,
                    )
                except (RuntimeError, Exception) as _e:
                    _msg = str(_e)
                    if "channels" in _msg and image_lr is not None:
                        n_got = image_lr.shape[1]
                        raise RuntimeError(
                            f"Regression model received {n_got} input channels but "
                            f"expected a different number. If the model was trained with "
                            f"temporal inputs, add 'dataset.temporal_inputs' to your eval "
                            f"config (e.g. use --config-name=evaluate_temporal_reg). "
                            f"Original error: {_e}"
                        ) from _e
                    raise
            reg_mean = image_reg[0:1]  # (1, C, H, W)
        tar_np = dataset.denormalize_output(image_tar.cpu().numpy())
        tar_t = _clamp_nonnegative_channels(torch.from_numpy(tar_np).to(device), nonnegative_channels)
        reg_pred_t = None
        if reg_mean is not None:
            reg_pred_np = dataset.denormalize_output(reg_mean.cpu().numpy())
            reg_pred_t = _clamp_nonnegative_channels(torch.from_numpy(reg_pred_np).to(device), nonnegative_channels)

        if not climate_only:
            # Regression metrics: update each group's accumulator with its channel subset
            if log_regression:
                for g in _groups.values():
                    ch = g["channels"]
                    r = reg_pred_t[:, ch] if ch is not None else reg_pred_t
                    t = tar_t[:, ch] if ch is not None else tar_t
                    g["reg_acc"].update(r, t)
            # Diffusion diagnostics also use this target subset.
            tar_m = tar_t[:, diagnostic_channels] if diagnostic_channels is not None else tar_t
            if has_regression:
                reg_pred_m = reg_pred_t[:, diagnostic_channels] if diagnostic_channels is not None else reg_pred_t
                hist_acc_reg.update(reg_pred_m, tar_m.squeeze(0))
                rapsd_acc_reg.update(reg_pred_m, tar_m.squeeze(0))

            # Track as event candidate for auto-selection of plot timesteps
            local_event_candidates.append((time_idx, float(tar_t.max()), dataset_idx))

        # -- Direct or residual diffusion forward pass --
        _any_diff_acc = any(g["diff_acc"] is not None for g in _groups.values())
        if net_res is not None and (_any_diff_acc or climate_enabled):
            mean_hr = reg_mean if cfg.generation.hr_mean_conditioning else None
            timestep_seed_batches = resolve_seed_batches(
                seed_batches,
                seed_mode=seed_mode,
                timestamp=times[time_idx],
                num_ensembles=num_ensembles,
                seed_base=seed_base,
            )
            diffusion_residuals = generate_ensemble(
                net_res=net_res,
                sampler_fn=sampler_fn,
                use_dropout_residual=use_dropout_residual,
                image_lr=image_lr,
                img_shape=img_shape,
                img_out_channels=img_out_channels,
                device=device,
                mean_hr=mean_hr,
                lead_time_label=lead_time_label,
                seed_batches=timestep_seed_batches,
                diffusion_kwargs=diffusion_kwargs,
            )

            # Direct samples are already full targets; only residuals need a mean.
            ens_pred = (
                diffusion_residuals if reg_mean is None else reg_mean + diffusion_residuals
            )

            ens_pred_np = dataset.denormalize_output(ens_pred.cpu().numpy())
            ens_pred_t = torch.from_numpy(ens_pred_np).to(device)      # (N_ens, C, H, W)
            ens_pred_t = _clamp_nonnegative_channels(ens_pred_t, nonnegative_channels)

            if climate_enabled:
                climate_accumulator.update(
                    times[time_idx],
                    ens_pred_t[0, climate_channel_idx].detach().cpu().numpy(),
                    tar_t[0, climate_channel_idx].detach().cpu().numpy(),
                )

            if not climate_only:
                for g in _groups.values():
                    if g["diff_acc"] is None:
                        continue
                    ch = g["channels"]
                    e = ens_pred_t[:, ch] if ch is not None else ens_pred_t
                    t = tar_t[:, ch] if ch is not None else tar_t
                    g["diff_acc"].update(e, t)
                ens_pred_m = ens_pred_t[:, diagnostic_channels] if diagnostic_channels is not None else ens_pred_t
                hist_acc_diff.update(ens_pred_m, tar_m.squeeze(0))
                rapsd_acc_diff.update(ens_pred_m, tar_m.squeeze(0))

        # Save or buffer normalized predictions for NetCDF output.
        if save_predictions:
            # Use the diffusion ensemble if available, otherwise regression only
            if net_res is not None and _any_diff_acc:
                image_out = ens_pred.cpu()  # (N_ens, C, H, W), normalized
            else:
                image_out = reg_mean.cpu()  # (1, C, H, W), normalized
            if stream_predictions:
                save_images(
                    writer=streaming_nc_writer,
                    dataset=dataset,
                    times=list(times),
                    image_out=image_out,
                    image_tar=image_tar.cpu(),
                    image_lr=image_lr.cpu() if save_prediction_inputs else None,
                    time_index=time_idx,
                    t_index=time_idx,
                    has_lead_time=cfg.generation.get("has_lead_time", False),
                    save_inputs=save_prediction_inputs,
                    output_channel_indices=saved_output_channel_indices,
                )
                if (time_idx + 1) % prediction_sync_interval == 0:
                    streaming_nc_file.sync()
            else:
                local_save_data.append({
                    "time_idx": time_idx,
                    "dataset_idx": dataset_idx,
                    "image_out": image_out,
                    "image_tar": image_tar.cpu(),  # (1, C, H, W), normalized
                    "image_lr": (
                        image_lr.cpu() if save_prediction_inputs else None
                    ),
                })

    if save_predictions and stream_predictions:
        streaming_nc_file.setncattr("completed_timesteps", total_times)
        streaming_nc_file.close()
        os.rename(streaming_tmp_path, streaming_final_path)
        logger0.info(f"Saved {total_times} streamed timesteps to {streaming_final_path}")

    if climate_only:
        written_path = climate_accumulator.write_netcdf(
            climate_output_path,
            latitude=lat_np,
            longitude=lon_np,
            metadata={
                "run_tag": str(run_tag),
                "model_regression_checkpoint": str(
                    cfg.generation.io.reg_ckpt_filename
                ),
                "model_diffusion_checkpoint": str(
                    cfg.generation.io.res_ckpt_filename
                ),
                "seed_mode": seed_mode,
                "seed_base": seed_base,
                "num_ensembles": int(num_ensembles),
                "requested_first_time": str(times[0]),
                "requested_last_time": str(times[-1]),
                "dataset_years": list(cfg.dataset.years),
                "excluded_month_days": sorted(excluded_month_days),
                "temporal_offsets_hours": list(
                    cfg.dataset.get("temporal_inputs", {}).get("offset_hours", [0])
                ),
            },
        )
        logger0.info(
            f"Climate-only evaluation complete: {climate_accumulator.completed_timesteps} "
            f"timesteps written to {written_path}"
        )
        return

    # ------------------------------------------------------------------
    # Distributed reduce (all_reduce so every rank has global totals)
    # ------------------------------------------------------------------
    for g in _groups.values():
        if log_regression:
            g["reg_acc"].reduce()
        if g["diff_acc"] is not None:
            g["diff_acc"].reduce()
    if has_regression:
        hist_acc_reg.reduce()
        rapsd_acc_reg.reduce()
    if hist_acc_diff is not None:
        hist_acc_diff.reduce()
    if rapsd_acc_diff is not None:
        rapsd_acc_diff.reduce()

    # Gather event candidates from all ranks (plain Python objects, no GPU tensors)
    if dist.world_size > 1:
        all_candidates: list[tuple[int, float, int]] = [None] * dist.world_size
        torch.distributed.all_gather_object(all_candidates, local_event_candidates)
        all_candidates = [item for sublist in all_candidates for item in sublist]
    else:
        all_candidates = local_event_candidates

    # ------------------------------------------------------------------
    # Save predictions to NetCDF (rank 0 gathers from all ranks and writes)
    # ------------------------------------------------------------------
    if save_predictions and not stream_predictions:
        if dist.world_size > 1:
            all_save_data_nested = [None] * dist.world_size
            torch.distributed.all_gather_object(all_save_data_nested, local_save_data)
            all_save_data = sorted(
                [d for rank_data in all_save_data_nested for d in rank_data],
                key=lambda x: x["time_idx"],
            )
        else:
            all_save_data = sorted(local_save_data, key=lambda x: x["time_idx"])

        if dist.rank == 0:
            abs_pred_path = to_absolute_path(predictions_file)
            tmp_path = abs_pred_path + ".tmp"
            logger.info(f"Saving predictions to: {abs_pred_path}")
            nc_file = nc.Dataset(tmp_path, "w")
            nc_file.cfg = str(cfg)
            nc_writer = NetCDFWriter(
                nc_file,
                lat=np.array(dataset.latitude()),
                lon=np.array(dataset.longitude()),
                input_channels=(dataset.input_channels() if save_prediction_inputs else []),
                output_channels=saved_output_channels,
                has_lead_time=cfg.generation.get("has_lead_time", False),
            )
            for write_idx, d in enumerate(all_save_data):
                save_images(
                    writer=nc_writer,
                    dataset=dataset,
                    times=list(times),
                    image_out=d["image_out"],
                    image_tar=d["image_tar"],
                    image_lr=d["image_lr"],
                    time_index=write_idx,
                    t_index=d["time_idx"],
                    has_lead_time=cfg.generation.get("has_lead_time", False),
                    save_inputs=save_prediction_inputs,
                    output_channel_indices=saved_output_channel_indices,
                )
            nc_file.close()
            os.rename(tmp_path, abs_pred_path)
            logger.info(f"Saved {len(all_save_data)} timesteps to {abs_pred_path}")

    # ------------------------------------------------------------------
    # Log metrics and plots (rank 0 only)
    # ------------------------------------------------------------------
    if dist.rank == 0:
        wandb_payload = {}
        all_metrics = {}

        for gname, g in _groups.items():
            # WandB prefix: "regression/" for single group, "regression/{gname}/" for multi
            if gname is None:
                reg_prefix = "regression/"
                diff_prefix = "diffusion/"
            else:
                reg_prefix = f"regression/{gname}/"
                diff_prefix = f"diffusion/{gname}/"

            if log_regression:
                reg_dict = g["reg_acc"].to_dict(prefix=reg_prefix)
                wandb_payload.update(reg_dict)
                all_metrics.update(reg_dict)

            if g["diff_acc"] is not None:
                diff_dict = g["diff_acc"].to_dict(prefix=diff_prefix)
                wandb_payload.update(diff_dict)
                all_metrics.update(diff_dict)

        # Console summary
        logger.info("=" * 70)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 70)
        for k, v in all_metrics.items():
            if isinstance(v, float):
                logger.info(f"  {k}: {v:.6f}")
            elif not isinstance(v, list):
                logger.info(f"  {k}: {v}")

        # JSON backup
        with open(output_json, "w") as f:
            json.dump(all_metrics, f, indent=2)
        logger.info(f"Results saved to: {output_json}")

        # --------------------------------------------------------------
        # Diagnostic plots — use the selected diagnostic group's accumulator.
        # --------------------------------------------------------------
        if log_regression:
            diagnostic_reg_dict = diagnostic_group["reg_acc"].to_dict(prefix="regression/")
            wandb_payload.update(_diagnostic_plot_payload(
                acc_label="regression",
                metrics_dict=diagnostic_reg_dict,
                hist_acc=hist_acc_reg,
                rapsd_acc=rapsd_acc_reg,
                rapsd_dx_km=rapsd_dx_km,
                diagnostic_info=diagnostic_info,
            ))

        if diagnostic_group["diff_acc"] is not None and hist_acc_diff is not None:
            diagnostic_diff_dict = diagnostic_group["diff_acc"].to_dict(prefix="diffusion/")
            wandb_payload.update(_diagnostic_plot_payload(
                acc_label="diffusion",
                metrics_dict=diagnostic_diff_dict,
                hist_acc=hist_acc_diff,
                rapsd_acc=rapsd_acc_diff,
                rapsd_dx_km=rapsd_dx_km,
                diagnostic_info=diagnostic_info,
            ))

        # --------------------------------------------------------------
        # Example event plots (georeferenced, cartopy)
        # --------------------------------------------------------------
        explicit_set = set(plot_events)
        sorted_cands = sorted(all_candidates, key=lambda x: -x[1])
        auto_set = {c[0] for c in sorted_cands[:n_plot_events]}
        plot_set = explicit_set | auto_set

        # Step 0: diagnostic panels + scalar metrics
        scalar_payload = {k: v for k, v in wandb_payload.items() if not isinstance(v, wandb.Image)}
        wandb.summary.update(scalar_payload)
        wandb.log(wandb_payload, step=0, commit=True)

        # Steps 1..N: one step per event so the slider navigates between events
        if plot_set:
            logger.info(f"Generating {len(plot_set)} example event plot(s)...")
            candidate_map = {c[0]: c for c in all_candidates}

            for event_step, time_idx in enumerate(sorted(plot_set), start=1):
                if time_idx not in candidate_map:
                    logger.warning(f"time_idx {time_idx} not in any rank's candidates; skipping.")
                    continue
                _, max_precip, dataset_idx = candidate_map[time_idx]

                reg_mean_np, ens_pred_np, target_np = _run_single_timestep(
                    dataset=dataset,
                    dataset_idx=dataset_idx,
                    net_reg=net_reg,
                    net_res=net_res,
                    sampler_fn=sampler_fn,
                    use_dropout_residual=use_dropout_residual,
                    img_shape=img_shape,
                    img_out_channels=img_out_channels,
                    device=device,
                    hr_mean_conditioning=cfg.generation.hr_mean_conditioning,
                    diffusion_kwargs=diffusion_kwargs,
                    seed_batches=seed_batches,
                    # Match the metric path exactly: same seeds, same clamping,
                    # so the plotted event is the ensemble that was scored.
                    nonnegative_channels=nonnegative_channels,
                    seed_mode=seed_mode,
                    timestamp=times[time_idx] if time_idx < len(times) else None,
                    num_ensembles=num_ensembles,
                    seed_base=seed_base,
                )

                time_label = str(times[time_idx]) if time_idx < len(times) else str(time_idx)
                fig = plot_example_event(
                    pred_ens_np=ens_pred_np,
                    target_np=target_np,
                    reg_mean_np=reg_mean_np,
                    time_str=time_label,
                    channel_names=channel_names,
                    lat=lat_np,
                    lon=lon_np,
                    plot_channels=plot_channels_cfg,
                )
                wandb.log(
                    {
                        "event/plot": wandb.Image(fig, caption=f"t{time_idx} {time_label} max={max_precip:.1f}mm"),
                        "event/time_idx": time_idx,
                        "event/max_precip_mm": max_precip,
                    },
                    step=event_step,
                    commit=True,
                )
                plt.close(fig)
                logger.info(f"  Logged event plot for time_idx={time_idx} ({time_label})")

        wandb.finish()

    logger0.info("Evaluation complete.")


if __name__ == "__main__":
    main()
