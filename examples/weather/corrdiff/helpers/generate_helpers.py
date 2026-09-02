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

import datetime
from functools import partial

import numpy as np
import torch
import torch._dynamo
from hydra.utils import to_absolute_path

from physicsnemo import Module
from physicsnemo.utils.corrdiff import (
    diffusion_step,
    get_time_from_range,
)
from physicsnemo.utils.diffusion import convert_datetime_to_cftime, deterministic_sampler
from physicsnemo.utils.patching import GridPatching2D

from datasets.dataset import init_dataset_from_config
from datasets.base import DownscalingDataset
from helpers.climate_signal import timestamp_seeds
from helpers.dropout_residual import dropout_residual_step


# ---------------------------------------------------------------------------
# HP evaluation sampler configurations
# ---------------------------------------------------------------------------

HP_EVAL_CONFIGS = [
    {"name": "det_heun_5",    "sampler": {"type": "deterministic", "num_steps": 5,  "solver": "heun"}},
    {"name": "det_euler_5",   "sampler": {"type": "deterministic", "num_steps": 5,  "solver": "euler"}},
    {"name": "det_euler_9",   "sampler": {"type": "deterministic", "num_steps": 9,  "solver": "euler"}},
    {"name": "stoch_18_churn40", "sampler": {"type": "stochastic", "num_steps": 18, "S_churn": 40, "S_min": 0, "S_max": float("inf")}},
    {"name": "stoch_18_churn80", "sampler": {"type": "stochastic", "num_steps": 18, "S_churn": 80, "S_min": 0, "S_max": float("inf")}},
]

HP_EVAL_CONFIGS_HETEROSCEDASTIC = [
    {"name": "uncert_18_churn20_scale1.0", "sampler": {"type": "uncertainty_aware", "num_steps": 18, "S_churn": 20, "S_min": 0, "S_max": float("inf"), "use_predicted_uncertainty": True, "uncertainty_scale": 1.0}},
    {"name": "uncert_18_churn40_scale1.0", "sampler": {"type": "uncertainty_aware", "num_steps": 18, "S_churn": 40, "S_min": 0, "S_max": float("inf"), "use_predicted_uncertainty": True, "uncertainty_scale": 1.0}},
    {"name": "uncert_18_churn40_scale0.5", "sampler": {"type": "uncertainty_aware", "num_steps": 18, "S_churn": 40, "S_min": 0, "S_max": float("inf"), "use_predicted_uncertainty": True, "uncertainty_scale": 0.5}},
    {"name": "uncert_18_churn80_scale1.0", "sampler": {"type": "uncertainty_aware", "num_steps": 18, "S_churn": 80, "S_min": 0, "S_max": float("inf"), "use_predicted_uncertainty": True, "uncertainty_scale": 1.0}},
    {"name": "uncert_18_churn80_scale0.5", "sampler": {"type": "uncertainty_aware", "num_steps": 18, "S_churn": 80, "S_min": 0, "S_max": float("inf"), "use_predicted_uncertainty": True, "uncertainty_scale": 0.5}},
]


# ---------------------------------------------------------------------------
# Model utilities
# ---------------------------------------------------------------------------

def _unwrap_compiled_model(model: torch.nn.Module | None) -> torch.nn.Module | None:
    if model is None:
        return None
    return getattr(model, "_orig_mod", model)


def _has_variance_head(model: torch.nn.Module | None) -> bool:
    unwrapped = _unwrap_compiled_model(model)
    return unwrapped is not None and hasattr(unwrapped, "variance_head")


def load_model(ckpt_path: str, device, perf_cfg=None, edm2_kwargs=None):
    """Load a model from .mdlus (Module archive) or .pt (EDM2 raw state dict).

    Sets use_fp16, profile_mode, and disables amp_mode.
    """
    if ckpt_path.endswith(".pt"):
        from helpers.edm2_preconditioning import EDM2PrecondSuperResolution
        if edm2_kwargs is None:
            raise ValueError(
                f"{ckpt_path} is a raw .pt state dict; provide generation.edm2 config."
            )
        net = EDM2PrecondSuperResolution(**edm2_kwargs)
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(state_dict, dict) and "optimizer_state_dict" in state_dict:
            raise ValueError(
                f"{ckpt_path} is a training checkpoint, not a model state dict. "
                "Use the extracted state-dict file instead."
            )
        net.load_state_dict(state_dict)
    else:
        use_apex_gn = getattr(perf_cfg, "use_apex_gn", False) if perf_cfg else False
        net = Module.from_checkpoint(ckpt_path, override_args={"use_apex_gn": use_apex_gn})
    if hasattr(net, "profile_mode"):
        net.profile_mode = getattr(perf_cfg, "profile_mode", False) if perf_cfg else False

    # use_fp16 is a read-only property on EDM2PrecondSuperResolution (set at init)
    try:
        net.use_fp16 = getattr(perf_cfg, "use_fp16", False) if perf_cfg else False
    except (AttributeError, TypeError):
        pass
    net.eval().requires_grad_(False).to(device).to(memory_format=torch.channels_last)
    if hasattr(net, "amp_mode"):
        net.amp_mode = False
    return net


def load_models(cfg, device, load_net_reg: bool, load_net_res: bool, edm2_kwargs=None):
    """Load regression and/or residual model checkpoints based on config."""
    perf_cfg = cfg.generation.perf

    net_res = None
    if load_net_res:
        res_ckpt = getattr(cfg.generation.io, "res_ckpt_filename", None)
        if res_ckpt is None:
            raise ValueError("res_ckpt_filename required for diffusion inference")
        net_res = load_model(to_absolute_path(res_ckpt), device, perf_cfg, edm2_kwargs)

    net_reg = None
    if load_net_reg:
        reg_ckpt = getattr(cfg.generation.io, "reg_ckpt_filename", None)
        if reg_ckpt is None:
            raise ValueError("reg_ckpt_filename required for regression inference")
        net_reg = load_model(to_absolute_path(reg_ckpt), device, perf_cfg)

    return net_reg, net_res


def maybe_compile_models(cfg, net_reg, net_res):
    """Apply torch.compile to both models if configured."""
    if getattr(cfg.generation.perf, "use_torch_compile", False):
        torch._dynamo.config.cache_size_limit = 264
        torch._dynamo.reset()
        if net_res is not None:
            net_res = torch.compile(net_res)
        if net_reg is not None:
            net_reg = torch.compile(net_reg)
    return net_reg, net_res


def build_sampler_fn(sampler_cfg, patching, net_guide=None, guidance_scale: float = 0.0, guidance_schedule_alpha: float = 0.0, trajectory_callback=None):
    """Create a sampler partial from config (accepts dict or OmegaConf)."""
    from helpers.stochastic_sampler import stochastic_sampler, uncertainty_aware_stochastic_sampler

    def _get(cfg, key, default=None):
        return cfg.get(key, default) if isinstance(cfg, dict) else getattr(cfg, key, default)

    sampler_type = _get(sampler_cfg, "type")
    num_steps = _get(sampler_cfg, "num_steps", 18)

    if sampler_type == "deterministic":
        return partial(
            deterministic_sampler,
            num_steps=num_steps,
            solver=_get(sampler_cfg, "solver", "heun"),
            patching=patching,
        )
    elif sampler_type == "stochastic":
        return partial(
            stochastic_sampler,
            patching=patching,
            num_steps=num_steps,
            S_churn=_get(sampler_cfg, "S_churn", 0),
            S_min=_get(sampler_cfg, "S_min", 0),
            S_max=_get(sampler_cfg, "S_max", float("inf")),
            net_guide=net_guide,
            guidance_scale=guidance_scale,
            guidance_schedule_alpha=_get(sampler_cfg, "guidance_schedule_alpha", guidance_schedule_alpha),
            trajectory_callback=trajectory_callback,
        )
    elif sampler_type == "uncertainty_aware":
        return partial(
            uncertainty_aware_stochastic_sampler,
            patching=patching,
            num_steps=num_steps,
            S_churn=_get(sampler_cfg, "S_churn", 0),
            S_min=_get(sampler_cfg, "S_min", 0),
            S_max=_get(sampler_cfg, "S_max", float("inf")),
            use_predicted_uncertainty=_get(sampler_cfg, "use_predicted_uncertainty", True),
            uncertainty_scale=_get(sampler_cfg, "uncertainty_scale", 1.0),
        )
    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}")


def setup_patching(cfg, img_shape):
    """Parse patching config. Returns (patching_obj_or_None, img_shape)."""
    from helpers.train_helpers import set_patch_shape

    if getattr(cfg.generation, "patching", False):
        patch_shape = (cfg.generation.patch_shape_y, cfg.generation.patch_shape_x)
    else:
        patch_shape = (None, None)

    use_patching, img_shape, patch_shape = set_patch_shape(img_shape, patch_shape)
    if use_patching:
        patching = GridPatching2D(
            img_shape=img_shape,
            patch_shape=patch_shape,
            boundary_pix=cfg.generation.boundary_pix,
            overlap_pix=cfg.generation.overlap_pix,
        )
    else:
        patching = None

    return patching, img_shape


# ---------------------------------------------------------------------------
# Dataset / sampler
# ---------------------------------------------------------------------------

def get_dataset_and_sampler(dataset_cfg, times, has_lead_time=False):
    """Get a dataset and sampler for generation."""
    (dataset, _) = init_dataset_from_config(dataset_cfg, batch_size=1)
    if has_lead_time:
        plot_times = times
    else:
        plot_times = [
            convert_datetime_to_cftime(
                datetime.datetime.strptime(time, "%Y-%m-%dT%H:%M:%S")
            )
            for time in times
        ]
    all_times = dataset.time()
    time_indices = []
    for t in plot_times:
        try:
            time_indices.append(all_times.index(t))
        except ValueError:
            continue
    return dataset, time_indices


# ---------------------------------------------------------------------------
# NetCDF output
# ---------------------------------------------------------------------------

def save_images(
    writer,
    dataset: DownscalingDataset,
    times,
    image_out,
    image_tar,
    image_lr,
    time_index,
    t_index,
    has_lead_time,
    predicted_uncertainty=None,
    save_inputs=True,
    output_channel_indices=None,
):
    """Save inference results to NetCDF via a NetCDFWriter."""
    image_lr2 = None
    if save_inputs:
        if image_lr is None:
            raise ValueError("image_lr is required when save_inputs=True")
        image_lr2 = dataset.denormalize_input(image_lr[0].unsqueeze(0).cpu().numpy())
    image_tar2 = dataset.denormalize_output(image_tar[0].unsqueeze(0).cpu().numpy())

    if image_tar2.ndim != 4:
        raise ValueError("image_tar2 must be 4-dimensional")

    output_channel_info = dataset.output_channels()
    if output_channel_indices is None:
        output_channel_indices = list(range(len(output_channel_info)))
    else:
        output_channel_indices = list(output_channel_indices)
    if len(set(output_channel_indices)) != len(output_channel_indices):
        raise ValueError("output_channel_indices must not contain duplicates")
    if any(
        channel_idx < 0 or channel_idx >= len(output_channel_info)
        for channel_idx in output_channel_indices
    ):
        raise ValueError("output_channel_indices contains an out-of-range channel")

    for idx in range(image_out.shape[0]):
        image_out2 = image_out[idx].unsqueeze(0)
        if image_out2.ndim != 4:
            raise ValueError("image_out2 must be 4-dimensional")
        image_out2 = dataset.denormalize_output(image_out2.cpu().numpy())

        time = times[t_index]
        writer.write_time(time_index, time)
        for channel_idx in output_channel_indices:
            info = output_channel_info[channel_idx]
            channel_name = info.name + info.level
            truth = image_tar2[0, channel_idx]

            writer.write_truth(channel_name, time_index, truth)
            writer.write_prediction(channel_name, time_index, idx, image_out2[0, channel_idx])

            if (
                predicted_uncertainty is not None
                and idx == 0
                and hasattr(writer, "write_uncertainty")
            ):
                writer.write_uncertainty(
                    channel_name, time_index, predicted_uncertainty[channel_idx]
                )

        if save_inputs:
            input_channel_info = dataset.input_channels()
            for channel_idx in range(len(input_channel_info)):
                info = input_channel_info[channel_idx]
                channel_name = info.name + info.level
                writer.write_input(channel_name, time_index, image_lr2[0, channel_idx])
                if channel_idx == image_lr2.shape[1] - 1:
                    break


# ---------------------------------------------------------------------------
# Evaluation timestamp resolution
# ---------------------------------------------------------------------------

TIME_FORMAT = "%Y-%m-%dT%H:%M:%S"


def resolve_times(generation_cfg):
    """Resolve the evaluation timestamps from a ``generation`` config node.

    Accepts one of two mutually exclusive sources:

    ``times``
        An explicit list of ``%Y-%m-%dT%H:%M:%S`` timestamps.
    ``times_range``
        ``[start, end, step_hours]``, expanded by ``get_time_from_range``.

    Either may be narrowed by an optional ``times_exclude``: a list of
    ``[start, end]`` windows, both endpoints inclusive, dropped after
    expansion. This lets a mostly-contiguous evaluation period be expressed as
    a range plus its data gaps instead of thousands of literal timestamps, and
    keeps the *reason* for each gap documentable next to it.

    Args:
        generation_cfg: the ``cfg.generation`` node.

    Returns:
        list[str]: timestamps in ``TIME_FORMAT``, ascending.
    """
    has_times_range = (
        hasattr(generation_cfg, "times_range")
        and generation_cfg.times_range is not None
    )
    has_times = hasattr(generation_cfg, "times") and generation_cfg.times is not None

    if has_times_range and has_times:
        raise ValueError(
            "Specify either generation.times_range or generation.times, not both."
        )
    if has_times_range:
        times = list(get_time_from_range(generation_cfg.times_range))
    elif has_times:
        times = [str(t) for t in generation_cfg.times]
    else:
        raise ValueError(
            "Either generation.times_range or generation.times must be set."
        )

    excluded = getattr(generation_cfg, "times_exclude", None)
    if not excluded:
        return times

    windows = []
    for i, window in enumerate(excluded):
        window = list(window)
        if len(window) != 2:
            raise ValueError(
                f"generation.times_exclude[{i}] must be [start, end], got {window!r}"
            )
        start, end = (
            datetime.datetime.strptime(str(bound), TIME_FORMAT) for bound in window
        )
        if start > end:
            raise ValueError(
                f"generation.times_exclude[{i}] has start after end: {window!r}"
            )
        windows.append((start, end))

    kept = []
    for stamp in times:
        moment = datetime.datetime.strptime(stamp, TIME_FORMAT)
        if not any(start <= moment <= end for start, end in windows):
            kept.append(stamp)
    if not kept:
        raise ValueError("generation.times_exclude removed every evaluation timestep.")
    return kept


# ---------------------------------------------------------------------------
# Per-timestep inference
# ---------------------------------------------------------------------------


def load_timestep_tensors(dataset, dataset_idx: int, device):
    """Fetch one dataset item and move it onto ``device`` as batched tensors.

    Args:
        dataset: the downscaling dataset.
        dataset_idx: index into ``dataset``.
        device: target torch device.

    Returns:
        tuple: ``(image_tar, image_lr, lead_time_label)`` where the first two are
        ``(1, C, H, W)`` float32 tensors (``image_lr`` in channels-last) and
        ``lead_time_label`` is a batched tensor or ``None``.
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
        label = lead_time_label[0]
        if isinstance(label, np.ndarray):
            label = torch.from_numpy(label)
        lead_time_label = label.unsqueeze(0).to(device).contiguous()
    else:
        lead_time_label = None

    return image_tar, image_lr, lead_time_label


def resolve_seed_batches(
    seed_batches,
    *,
    seed_mode: str = "fixed",
    timestamp=None,
    num_ensembles: int | None = None,
    seed_base: int = 0,
):
    """Return the seed batches to use for one timestep.

    With ``seed_mode="fixed"`` the caller's batches are reused for every
    timestep. With ``"timestamp"`` the seeds are derived from ``timestamp``, so
    each timestep draws a different ensemble.

    Every consumer of a timestep -- metrics, climate accumulation, saved
    predictions, and the example-event figures -- must resolve seeds through
    this function, or the plotted ensemble will not be the scored one.

    Args:
        seed_batches: the fixed batches, as produced by ``np.array_split``.
        seed_mode: ``"fixed"`` or ``"timestamp"``.
        timestamp: the timestep's time, required for ``"timestamp"``.
        num_ensembles: ensemble size, required for ``"timestamp"``.
        seed_base: offset applied to timestamp-derived seeds.

    Returns:
        list: seed batches for this timestep.
    """
    if seed_mode not in {"fixed", "timestamp"}:
        raise ValueError("seed_mode must be 'fixed' or 'timestamp'")
    if seed_mode == "fixed":
        return seed_batches
    if timestamp is None or num_ensembles is None:
        raise ValueError(
            "seed_mode='timestamp' requires both timestamp and num_ensembles"
        )
    seeds = timestamp_seeds(
        timestamp, n_members=num_ensembles, base_seed=seed_base
    )
    return np.array_split(seeds, len(seed_batches))


def generate_ensemble(
    *,
    net_res,
    sampler_fn,
    use_dropout_residual: bool,
    image_lr,
    img_shape,
    img_out_channels: int,
    device,
    mean_hr,
    lead_time_label,
    seed_batches,
    diffusion_kwargs: dict,
):
    """Sample residual ensemble members for one timestep.

    Runs either the dropout-residual or the diffusion sampler once per seed
    batch and concatenates the members.

    Returns:
        torch.Tensor: ``(N_ens, C, H, W)`` residuals, still normalized.
    """
    all_residuals = []
    for seed_batch in seed_batches:
        batch_size = len(seed_batch)
        with torch.no_grad():
            if use_dropout_residual:
                residual = dropout_residual_step(
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
                residual = diffusion_step(
                    net=net_res,
                    sampler_fn=sampler_fn,
                    img_shape=img_shape,
                    img_out_channels=img_out_channels,
                    rank_batches=[torch.tensor(seed_batch)],
                    img_lr=image_lr.expand(batch_size, -1, -1, -1).to(
                        memory_format=torch.channels_last
                    ),
                    rank=0,
                    device=device,
                    mean_hr=mean_hr,
                    lead_time_label=lead_time_label,
                    **diffusion_kwargs,
                )
        all_residuals.append(residual)

    return torch.cat(all_residuals, dim=0)
