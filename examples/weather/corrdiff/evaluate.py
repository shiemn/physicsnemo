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
from physicsnemo.utils.corrdiff import get_time_from_range, regression_step, diffusion_step
from physicsnemo.utils.corrdiff.utils import NetCDFWriter

from datasets.dataset import register_dataset
from helpers.generate_helpers import (
    build_sampler_fn,
    get_dataset_and_sampler,
    load_model,
    load_models,
    maybe_compile_models,
    save_images,
    setup_patching,
)
from helpers.dropout_residual import dropout_residual_step
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

    f.close()
    return {
        "lat": lat,
        "lon": lon,
        "channel_names": channel_names,
        "n_times": truth.shape[0],
        "n_ensemble": prediction.shape[0],
        "truth": truth,
        "prediction": prediction,
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

    # Determine if regression-only (1 member) or ensemble
    is_regression = n_ens == 1

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

    if metric_groups_cfg:
        _groups = {
            gname: {"channels": gcfg.get("channels"), "acc": _make_file_acc(gcfg)}
            for gname, gcfg in metric_groups_cfg.items()
        }
    else:
        _groups = {
            None: {
                "channels": None,
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
            reg_mean_np = pred_ens_np.mean(axis=0)          # (C, H, W)
            max_precip = float(target_np.max())

            fig = plot_example_event(
                pred_ens_np=pred_ens_np,
                target_np=target_np,
                reg_mean_np=reg_mean_np,
                time_str=str(time_idx),
                channel_names=channel_names,
                lat=lat_np,
                lon=lon_np,
            )
            wandb.log(
                {
                    "event/plot": wandb.Image(fig, caption=f"t{time_idx} max={max_precip:.1f}mm"),
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run regression + diffusion inference for a single timestep on one GPU.

    Returns:
        reg_mean_np:  (C, H, W) regression prediction in physical units (mm).
        ens_pred_np:  (N_ens, C, H, W) full ensemble in physical units, or same
                      as reg_mean_np if net_res is None.
        target_np:    (C, H, W) ground truth in physical units (mm).
    """
    image_tar, image_lr, *lead_time_label = dataset[dataset_idx]
    if isinstance(image_tar, np.ndarray):
        image_tar = torch.from_numpy(image_tar)
    if isinstance(image_lr, np.ndarray):
        image_lr = torch.from_numpy(image_lr)

    image_tar = image_tar.unsqueeze(0).to(device=device, dtype=torch.float32)
    image_lr = (
        image_lr.unsqueeze(0)
        .to(device=device, dtype=torch.float32)
        .to(memory_format=torch.channels_last)
    )
    if lead_time_label:
        lt = lead_time_label[0]
        if isinstance(lt, np.ndarray):
            lt = torch.from_numpy(lt)
        lead_time_label = lt.unsqueeze(0).to(device).contiguous()
    else:
        lead_time_label = None

    with torch.no_grad():
        image_reg = regression_step(
            net=net_reg,
            img_lr=image_lr,
            latents_shape=(1, img_out_channels, img_shape[0], img_shape[1]),
            lead_time_label=lead_time_label,
        )
    reg_mean = image_reg[0:1]  # (1, C, H, W)

    reg_mean_np = dataset.denormalize_output(reg_mean.cpu().numpy())[0]  # (C, H, W)
    target_np = dataset.denormalize_output(image_tar.cpu().numpy())[0]   # (C, H, W)

    if net_res is not None and (use_dropout_residual or sampler_fn is not None):
        mean_hr = reg_mean if hr_mean_conditioning else None
        all_residuals = []
        for seed_batch in seed_batches:
            batch_size = len(seed_batch)
            rank_batches = [torch.tensor(seed_batch)]
            with torch.no_grad():
                if use_dropout_residual:
                    res = dropout_residual_step(
                        net=net_res,
                        img_lr=image_lr,
                        latents_shape=(
                            batch_size,
                            img_out_channels,
                            img_shape[0],
                            img_shape[1],
                        ),
                        mean_hr=mean_hr,
                        lead_time_label=lead_time_label,
                        seed=int(seed_batch[0]) if len(seed_batch) else None,
                    )
                else:
                    res = diffusion_step(
                        net=net_res,
                        sampler_fn=sampler_fn,
                        img_shape=img_shape,
                        img_out_channels=img_out_channels,
                        rank_batches=rank_batches,
                        img_lr=image_lr.expand(batch_size, -1, -1, -1).to(
                            memory_format=torch.channels_last
                        ),
                        rank=0,
                        device=device,
                        mean_hr=mean_hr,
                        lead_time_label=lead_time_label,
                        **diffusion_kwargs,
                    )
            all_residuals.append(res)
        diffusion_residuals = torch.cat(all_residuals, dim=0)  # (N_ens, C, H, W)
        ens_pred = reg_mean + diffusion_residuals               # (N_ens, C, H, W)
        ens_pred_np = dataset.denormalize_output(ens_pred.cpu().numpy())  # (N_ens, C, H, W)
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
    if inference_mode not in ("regression", "all"):
        raise ValueError(
            f'Unsupported inference_mode={inference_mode!r}. '
            f'Must be "regression" or "all" (regression + diffusion).'
        )
    num_ensembles = cfg.generation.num_ensembles
    seed_batch_size = cfg.generation.seed_batch_size
    precip_threshold = cfg.eval.get("precip_threshold", 1.0)
    output_json = cfg.eval.get("output_json", "eval_results.json")
    twcrps_thresholds = list(cfg.eval.get("twcrps_thresholds", [5.0, 10.0]))
    n_plot_events = cfg.eval.get("n_plot_events", 5)
    plot_events = list(cfg.eval.get("plot_events", None) or [])
    rapsd_dx_km = float(cfg.eval.get("rapsd_dx_km", 2.0))
    predictions_file_cfg = cfg.eval.get("predictions_file", None)
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
    log_regression = cfg.eval.get("log_regression", True)
    plot_channels_cfg = cfg.eval.get("plot_channels", None)
    if plot_channels_cfg is not None:
        plot_channels_cfg = list(plot_channels_cfg)

    # Resolve "auto" → derive filename from the primary checkpoint name.
    # Done early so the offline-load check works, but we need the checkpoint
    # path from the config (not the loaded model).
    if predictions_file_cfg == "auto":
        # Use diffusion checkpoint if available, otherwise regression
        if inference_mode == "all":
            _ckpt = cfg.generation.io.res_ckpt_filename
        else:
            _ckpt = cfg.generation.io.reg_ckpt_filename
        _stem = os.path.splitext(os.path.basename(str(_ckpt)))[0]
        predictions_file = f"eval_{_stem}.nc"
    else:
        predictions_file = predictions_file_cfg  # explicit path or None

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
    if dist.rank == 0:
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
    has_times_range = (
        hasattr(cfg.generation, "times_range")
        and cfg.generation.times_range is not None
    )
    has_times = (
        hasattr(cfg.generation, "times") and cfg.generation.times is not None
    )
    if has_times_range and has_times:
        raise ValueError("Provide times_range or times, not both.")
    elif has_times_range:
        times = get_time_from_range(cfg.generation.times_range)
    elif has_times:
        times = list(cfg.generation.times)
    else:
        raise ValueError("Either times_range or times must be set in cfg.generation.")

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

    img_shape = dataset.image_shape()
    output_channels = list(dataset.output_channels())
    img_out_channels = len(output_channels)
    channel_names = [_format_channel_label(channel) for channel in output_channels]
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
        load_net_reg=True,
        load_net_res=(inference_mode == "all"),
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
                "diff_acc": _make_acc(gcfg, skip_ss=False) if inference_mode == "all" else None,
            }
            for gname, gcfg in metric_groups_cfg.items()
        }
    else:
        _groups = {
            None: {  # None key → legacy prefix structure (no group sub-key)
                "channels": metric_channels,
                "reg_acc": _make_acc({}, skip_ss=True),
                "diff_acc": _make_acc({}, skip_ss=False) if inference_mode == "all" else None,
            }
        }

    diagnostic_group_name, diagnostic_info = _select_diagnostic_group(cfg, _groups, channel_names)
    diagnostic_group = _groups[diagnostic_group_name]
    diagnostic_channels = diagnostic_group["channels"]

    # Histogram and RAPSD accumulators
    hist_acc_reg = HistogramAccumulator(device=device)
    rapsd_acc_reg = RAPSDAccumulator(img_shape=img_shape, dx_km=rapsd_dx_km, device=device)
    hist_acc_diff = HistogramAccumulator(device=device) if inference_mode == "all" else None
    rapsd_acc_diff = (
        RAPSDAccumulator(img_shape=img_shape, dx_km=rapsd_dx_km, device=device)
        if inference_mode == "all"
        else None
    )

    # Event candidates for auto-selection of plot timesteps: (time_idx, max_precip, dataset_idx)
    local_event_candidates: list[tuple[int, float, int]] = []

    # Buffer for NetCDF save (normalized tensors, CPU)
    local_save_data: list[dict] = [] if predictions_file else []
    save_predictions = predictions_file is not None

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
        image_tar, image_lr, *lead_time_label = dataset[dataset_idx]
        if isinstance(image_tar, np.ndarray):
            image_tar = torch.from_numpy(image_tar)
        if isinstance(image_lr, np.ndarray):
            image_lr = torch.from_numpy(image_lr)

        image_tar = image_tar.unsqueeze(0).to(device=device, dtype=torch.float32)
        image_lr = (
            image_lr.unsqueeze(0)
            .to(device=device, dtype=torch.float32)
            .to(memory_format=torch.channels_last)
        )
        if lead_time_label:
            lt = lead_time_label[0]
            if isinstance(lt, np.ndarray):
                lt = torch.from_numpy(lt)
            lead_time_label = lt.unsqueeze(0).to(device).contiguous()
        else:
            lead_time_label = None

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

        # Denormalize regression prediction and target
        reg_pred_np = dataset.denormalize_output(reg_mean.cpu().numpy())          # (1, C, H, W)
        tar_np = dataset.denormalize_output(image_tar.cpu().numpy())              # (1, C, H, W)
        reg_pred_t = torch.from_numpy(reg_pred_np).to(device)                     # (1, C, H, W)
        tar_t = torch.from_numpy(tar_np).to(device)                               # (1, C, H, W)
        reg_pred_t = _clamp_nonnegative_channels(reg_pred_t, nonnegative_channels)
        tar_t = _clamp_nonnegative_channels(tar_t, nonnegative_channels)

        # Regression metrics: update each group's accumulator with its channel subset
        if log_regression:
            for g in _groups.values():
                ch = g["channels"]
                r = reg_pred_t[:, ch] if ch is not None else reg_pred_t
                t = tar_t[:, ch] if ch is not None else tar_t
                g["reg_acc"].update(r, t)
        # Diagnostic plots use the selected diagnostic group's channel subset.
        reg_pred_m = reg_pred_t[:, diagnostic_channels] if diagnostic_channels is not None else reg_pred_t
        tar_m = tar_t[:, diagnostic_channels] if diagnostic_channels is not None else tar_t
        hist_acc_reg.update(reg_pred_m, tar_m.squeeze(0))
        rapsd_acc_reg.update(reg_pred_m, tar_m.squeeze(0))

        # Track as event candidate for auto-selection of plot timesteps
        local_event_candidates.append((time_idx, float(tar_t.max()), dataset_idx))

        # -- Diffusion forward pass (when mode == "all") --
        _any_diff_acc = any(g["diff_acc"] is not None for g in _groups.values())
        if net_res is not None and _any_diff_acc:
            mean_hr = reg_mean if cfg.generation.hr_mean_conditioning else None
            all_residuals = []
            for seed_batch in seed_batches:
                batch_size = len(seed_batch)
                rank_batches = [torch.tensor(seed_batch)]
                with torch.no_grad():
                    if use_dropout_residual:
                        res = dropout_residual_step(
                            net=net_res,
                            img_lr=image_lr,
                            latents_shape=(
                                batch_size,
                                img_out_channels,
                                img_shape[0],
                                img_shape[1],
                            ),
                            mean_hr=mean_hr,
                            lead_time_label=lead_time_label,
                            seed=int(seed_batch[0]) if len(seed_batch) else None,
                        )
                    else:
                        res = diffusion_step(
                            net=net_res,
                            sampler_fn=sampler_fn,
                            img_shape=img_shape,
                            img_out_channels=img_out_channels,
                            rank_batches=rank_batches,
                            img_lr=image_lr.expand(batch_size, -1, -1, -1).to(
                                memory_format=torch.channels_last
                            ),
                            rank=0,
                            device=device,
                            mean_hr=mean_hr,
                            lead_time_label=lead_time_label,
                            **diffusion_kwargs,
                        )
                all_residuals.append(res)

            # Full ensemble: regression mean + diffusion residuals
            diffusion_residuals = torch.cat(all_residuals, dim=0)   # (N_ens, C, H, W)
            ens_pred = reg_mean + diffusion_residuals                # (N_ens, C, H, W)

            ens_pred_np = dataset.denormalize_output(ens_pred.cpu().numpy())
            ens_pred_t = torch.from_numpy(ens_pred_np).to(device)      # (N_ens, C, H, W)
            ens_pred_t = _clamp_nonnegative_channels(ens_pred_t, nonnegative_channels)

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

        # Buffer normalized predictions for NetCDF save
        if save_predictions:
            # Use the diffusion ensemble if available, otherwise regression only
            if net_res is not None and _any_diff_acc:
                image_out = ens_pred.cpu()  # (N_ens, C, H, W), normalized
            else:
                image_out = reg_mean.cpu()  # (1, C, H, W), normalized
            local_save_data.append({
                "time_idx": time_idx,
                "dataset_idx": dataset_idx,
                "image_out": image_out,
                "image_tar": image_tar.cpu(),  # (1, C, H, W), normalized
                "image_lr": image_lr.cpu(),    # (1, C_lr, H, W), normalized
            })

    # ------------------------------------------------------------------
    # Distributed reduce (all_reduce so every rank has global totals)
    # ------------------------------------------------------------------
    for g in _groups.values():
        if log_regression:
            g["reg_acc"].reduce()
        if g["diff_acc"] is not None:
            g["diff_acc"].reduce()
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
    if save_predictions:
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
                input_channels=dataset.input_channels(),
                output_channels=dataset.output_channels(),
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
