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

"""
Generate script with time-step parallelism across GPUs.

Each GPU processes DIFFERENT time steps (vs original which splits ensembles).
All ensemble members for a time step are generated on the same GPU.

Includes efficient metric computation (RMSE, CRPS, Spread-Skill) for HP search.
Metrics match the calculations in wandb_eval/metrics_utils.py:
  - RMSE: sqrt(total_squared_error / total_elements) across all pixels and times
  - CRPS: Simplified formula: MAE - 0.5 * std (averaged over all elements)
  - Values are clipped to >= 0 before computation

Additional conditional metrics for extreme precipitation:
  - rmse_gt_{threshold}mm: RMSE only for pixels where target > threshold (default: 1mm)
  - crps_gt_{threshold}mm: CRPS only for pixels where target > threshold
  - rmse_95th_percentile: RMSE only for pixels in the 95th percentile of target
  - crps_95th_percentile: CRPS only for pixels in the 95th percentile of target

Example with 4 GPUs and 16 time steps:
  - GPU 0: times 0, 4, 8, 12
  - GPU 1: times 1, 5, 9, 13
  - GPU 2: times 2, 6, 10, 14
  - GPU 3: times 3, 7, 11, 15

Usage:
    torchrun --nproc_per_node=4 generate_parallel_times.py --config-name=gen_test

    # With metrics (for Optuna HP search):
    torchrun --nproc_per_node=4 generate_parallel_times.py --config-name=gen_test \
        generation.compute_metrics=true generation.metrics_output=metrics.json

    # With custom precipitation threshold:
    torchrun --nproc_per_node=4 generate_parallel_times.py --config-name=gen_test \
        generation.compute_metrics=true generation.precip_threshold=2.0
"""

import contextlib
import json
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from copy import deepcopy

import hydra

# HP Eval sampler configurations - used when generation.hp_eval=true
# Each config is evaluated and the best CRPS is selected
HP_EVAL_CONFIGS = [
    # Deterministic samplers (fast, reproducible)
    {
        "name": "det_heun_5",
        "sampler": {"type": "deterministic", "num_steps": 5, "solver": "heun"},
    },
    {
        "name": "det_euler_5",
        "sampler": {"type": "deterministic", "num_steps": 5, "solver": "euler"},
    },
    {
        "name": "det_euler_9",
        "sampler": {"type": "deterministic", "num_steps": 9, "solver": "euler"},
    },
    # Stochastic samplers (more steps, adds noise for diversity)
    {
        "name": "stoch_18_churn40",
        "sampler": {"type": "stochastic", "num_steps": 18, "S_churn": 40, "S_min": 0, "S_max": float("inf")},
    },
    {
        "name": "stoch_18_churn80",
        "sampler": {"type": "stochastic", "num_steps": 18, "S_churn": 80, "S_min": 0, "S_max": float("inf")},
    },
]


def create_sampler_fn(sampler_cfg, patching):
    """Create a sampler function from config dict or OmegaConf."""
    sampler_type = sampler_cfg.get("type") if isinstance(sampler_cfg, dict) else sampler_cfg.type
    
    if sampler_type == "deterministic":
        num_steps = sampler_cfg.get("num_steps", 5) if isinstance(sampler_cfg, dict) else sampler_cfg.num_steps
        solver = sampler_cfg.get("solver", "heun") if isinstance(sampler_cfg, dict) else sampler_cfg.solver
        return partial(
            deterministic_sampler,
            num_steps=num_steps,
            solver=solver,
            patching=patching,
        )
    elif sampler_type == "stochastic":
        num_steps = sampler_cfg.get("num_steps", 18) if isinstance(sampler_cfg, dict) else getattr(sampler_cfg, "num_steps", 18)
        S_churn = sampler_cfg.get("S_churn", 0) if isinstance(sampler_cfg, dict) else getattr(sampler_cfg, "S_churn", 0)
        S_min = sampler_cfg.get("S_min", 0) if isinstance(sampler_cfg, dict) else getattr(sampler_cfg, "S_min", 0)
        S_max = sampler_cfg.get("S_max", float("inf")) if isinstance(sampler_cfg, dict) else getattr(sampler_cfg, "S_max", float("inf"))
        return partial(
            stochastic_sampler,
            patching=patching,
            num_steps=num_steps,
            S_churn=S_churn,
            S_min=S_min,
            S_max=S_max,
        )
    else:
        raise ValueError(f"Unknown sampling method {sampler_type}")
from omegaconf import OmegaConf, DictConfig
from hydra.utils import to_absolute_path
import torch
import torch._dynamo
import numpy as np
import nvtx
import netCDF4 as nc


class MetricsAccumulator:
    """Efficient GPU-based metrics accumulator for ensemble forecasts.

    Computes RMSE, CRPS, and Spread-Skill matching the wandb_eval calculations.
    All computations done on GPU before CPU transfer for efficiency.
    
    Matches metrics_utils.py calculations:
    - RMSE: sqrt(total_squared_error / total_elements) across all pixels and times
    - CRPS: Simplified formula: MAE - 0.5 * std (averaged over all elements)
    - Values are clipped to >= 0 before computation
    
    Additional conditional metrics:
    - rmse_gt_1mm: RMSE only for pixels where target > 1mm
    - rmse_95th: RMSE only for pixels where target >= 95th percentile (per sample)
    - crps_gt_1mm: CRPS only for pixels where target > 1mm
    """

    def __init__(self, device, precip_threshold: float = 1.0):
        """
        Args:
            device: Torch device for computations
            precip_threshold: Threshold for "high precipitation" metrics (default: 1.0mm)
        """
        self.device = device
        self.precip_threshold = precip_threshold
        self.reset()

    def reset(self):
        # For RMSE: accumulate total squared error and element count
        self.squared_error_sum = 0.0
        self.total_elements = 0

        # For CRPS: accumulate (MAE - 0.5*std) weighted by element count
        self.crps_sum = 0.0
        self.crps_elements = 0

        # For spread-skill: accumulate spread and error for binning
        self.spread_sum = 0.0
        self.skill_sum = 0.0
        self.n_samples = 0

        # === Conditional metrics: high precipitation (target > threshold) ===
        self.squared_error_sum_gt_thresh = 0.0
        self.elements_gt_thresh = 0
        self.crps_sum_gt_thresh = 0.0
        self.crps_elements_gt_thresh = 0

        # === Conditional metrics: extreme precipitation (target >= 95th percentile) ===
        self.squared_error_sum_95th = 0.0
        self.elements_95th = 0
        self.crps_sum_95th = 0.0
        self.crps_elements_95th = 0

    @torch.no_grad()
    def update(self, prediction: torch.Tensor, target: torch.Tensor):
        """
        Update metrics with new predictions.

        Args:
            prediction: (num_ensembles, C, H, W) ensemble predictions
            target: (1, C, H, W) ground truth
        """
        # Ensure on same device
        prediction = prediction.to(self.device)
        target = target.to(self.device)

        # Clip values to >= 0 (matching wandb_eval)
        prediction = torch.clamp(prediction, min=0)
        target = torch.clamp(target, min=0)

        # Ensemble mean: (C, H, W)
        ens_mean = prediction.mean(dim=0)

        # Ensemble std (spread): (C, H, W)
        ens_std = prediction.std(dim=0)

        # Target without batch dim: (C, H, W)
        target_squeezed = target.squeeze(0)

        # === RMSE: Total squared error across all elements ===
        # This matches wandb_eval: sqrt(sum_squared_errors / total_elements)
        squared_error = (ens_mean - target_squeezed) ** 2
        self.squared_error_sum += squared_error.sum().item()
        self.total_elements += squared_error.numel()

        # === CRPS: Simplified formula matching wandb_eval ===
        # CRPS = MAE - 0.5 * std (element-wise, then averaged)
        mae = torch.abs(ens_mean - target_squeezed)
        spread_penalty = 0.5 * ens_std
        crps_values = mae - spread_penalty
        self.crps_sum += crps_values.sum().item()
        self.crps_elements += crps_values.numel()

        # === Spread-Skill: Per-sample metrics for ratio ===
        # Mean spread across all pixels for this sample
        spread = ens_std.mean()
        self.spread_sum += spread.item()

        # Skill (RMSE of this sample)
        mse = squared_error.mean()
        skill = torch.sqrt(mse)
        self.skill_sum += skill.item()

        self.n_samples += 1

        # === Conditional metrics: high precipitation (target > threshold) ===
        mask_gt_thresh = target_squeezed > self.precip_threshold
        if mask_gt_thresh.any():
            se_gt_thresh = squared_error[mask_gt_thresh]
            self.squared_error_sum_gt_thresh += se_gt_thresh.sum().item()
            self.elements_gt_thresh += se_gt_thresh.numel()

            crps_gt_thresh = crps_values[mask_gt_thresh]
            self.crps_sum_gt_thresh += crps_gt_thresh.sum().item()
            self.crps_elements_gt_thresh += crps_gt_thresh.numel()

        # === Conditional metrics: 95th percentile (per sample) ===
        # Compute 95th percentile threshold for this sample
        percentile_95 = torch.quantile(target_squeezed.flatten(), 0.95)
        mask_95th = target_squeezed >= percentile_95
        if mask_95th.any():
            se_95th = squared_error[mask_95th]
            self.squared_error_sum_95th += se_95th.sum().item()
            self.elements_95th += se_95th.numel()

            crps_95th = crps_values[mask_95th]
            self.crps_sum_95th += crps_95th.sum().item()
            self.crps_elements_95th += crps_95th.numel()

    def compute(self):
        """Compute final metrics."""
        if self.n_samples == 0:
            return {}

        # RMSE: sqrt(total_squared_error / total_elements)
        rmse = np.sqrt(self.squared_error_sum / self.total_elements) if self.total_elements > 0 else 0.0
        
        # CRPS: mean of (MAE - 0.5*std) across all elements
        crps = self.crps_sum / self.crps_elements if self.crps_elements > 0 else 0.0
        
        # Spread and skill averages
        spread = self.spread_sum / self.n_samples
        skill = self.skill_sum / self.n_samples

        # Spread-skill ratio (ideal = 1.0)
        spread_skill_ratio = spread / skill if skill > 0 else float('inf')

        # Conditional metrics: high precipitation
        rmse_gt_thresh = (np.sqrt(self.squared_error_sum_gt_thresh / self.elements_gt_thresh) 
                         if self.elements_gt_thresh > 0 else 0.0)
        crps_gt_thresh = (self.crps_sum_gt_thresh / self.crps_elements_gt_thresh 
                         if self.crps_elements_gt_thresh > 0 else 0.0)

        # Conditional metrics: 95th percentile
        rmse_95th = (np.sqrt(self.squared_error_sum_95th / self.elements_95th) 
                    if self.elements_95th > 0 else 0.0)
        crps_95th = (self.crps_sum_95th / self.crps_elements_95th 
                    if self.crps_elements_95th > 0 else 0.0)

        return {
            'rmse': rmse,
            'crps': crps,
            'spread': spread,
            'skill': skill,
            'spread_skill_ratio': spread_skill_ratio,
            'n_samples': self.n_samples,
            'total_elements': self.total_elements,
            # Conditional metrics
            f'rmse_gt_{self.precip_threshold}mm': rmse_gt_thresh,
            f'crps_gt_{self.precip_threshold}mm': crps_gt_thresh,
            f'elements_gt_{self.precip_threshold}mm': self.elements_gt_thresh,
            'rmse_95th_percentile': rmse_95th,
            'crps_95th_percentile': crps_95th,
            'elements_95th_percentile': self.elements_95th,
        }


from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.experimental.models.diffusion.preconditioning import (
    tEDMPrecondSuperRes,
)
from physicsnemo.utils.patching import GridPatching2D
from physicsnemo import Module
from physicsnemo.utils.diffusion import deterministic_sampler, stochastic_sampler
from physicsnemo.utils.corrdiff import (
    NetCDFWriter,
    get_time_from_range,
    regression_step,
    diffusion_step,
)

from helpers.generate_helpers import (
    get_dataset_and_sampler,
    save_images,
)
from helpers.train_helpers import set_patch_shape
from datasets.dataset import register_dataset


@hydra.main(version_base="1.2", config_path="conf", config_name="config_generate")
def main(cfg: DictConfig) -> None:
    """Generate with time-step parallelism across GPUs."""

    # Initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device

    # Initialize logger
    logger = PythonLogger("generate")
    logger0 = RankZeroLoggingWrapper(logger, dist)
    logger.file_logging("generate.log")

    num_ensembles = cfg.generation.num_ensembles
    seed_batch_size = cfg.generation.seed_batch_size

    # Metrics configuration
    compute_metrics = getattr(cfg.generation, 'compute_metrics', True)
    metrics_output = getattr(cfg.generation, 'metrics_output', 'metrics.json')
    skip_netcdf = getattr(cfg.generation, 'skip_netcdf', False)  # Skip NetCDF for faster HP search

    # Precipitation threshold for conditional metrics (default: 1.0mm)
    precip_threshold = getattr(cfg.generation, 'precip_threshold', 1.0)

    if compute_metrics:
        logger0.info(f"Metrics computation: ENABLED (output: {metrics_output})")
        logger0.info(f"  Precipitation threshold for conditional metrics: {precip_threshold}mm")
        metrics = MetricsAccumulator(device, precip_threshold=precip_threshold)
    else:
        metrics = None

    # Prepare ensemble seed batches
    seeds = list(np.arange(num_ensembles))
    num_seed_batches = (len(seeds) - 1) // seed_batch_size + 1
    seed_batches = np.array_split(seeds, num_seed_batches)

    # Synchronize
    if dist.world_size > 1:
        torch.distributed.barrier()

    # Parse the inference input times
    has_times_range = hasattr(cfg.generation, 'times_range') and cfg.generation.times_range is not None
    has_times = hasattr(cfg.generation, 'times') and cfg.generation.times is not None

    if has_times_range and has_times:
        raise ValueError("Either times_range or times must be provided, but not both")
    elif has_times_range:
        times = get_time_from_range(cfg.generation.times_range)
    elif has_times:
        times = cfg.generation.times
    else:
        raise ValueError("Either times_range or times must be provided in the configuration")

    # Create dataset object
    dataset_cfg = OmegaConf.to_container(cfg.dataset)
    register_dataset(cfg.dataset.type)
    logger0.info(f"Using dataset: {cfg.dataset.type}")

    if "has_lead_time" in cfg.generation:
        has_lead_time = cfg.generation["has_lead_time"]
    else:
        has_lead_time = False
    dataset, sampler = get_dataset_and_sampler(
        dataset_cfg=dataset_cfg, times=times, has_lead_time=has_lead_time
    )

    total_times = len(sampler)
    logger0.info(f"=== Time-Parallel Generation ===")
    logger0.info(f"  GPUs: {dist.world_size}")
    logger0.info(f"  Total time steps: {total_times}")
    logger0.info(f"  Ensembles per time step: {num_ensembles}")
    logger0.info(f"  Time steps per GPU: ~{(total_times + dist.world_size - 1) // dist.world_size}")

    img_shape = dataset.image_shape()
    img_out_channels = len(dataset.output_channels())

    # Parse the patch shape
    if cfg.generation.patching:
        patch_shape_x = cfg.generation.patch_shape_x
        patch_shape_y = cfg.generation.patch_shape_y
    else:
        patch_shape_x, patch_shape_y = None, None
    patch_shape = (patch_shape_y, patch_shape_x)
    use_patching, img_shape, patch_shape = set_patch_shape(img_shape, patch_shape)
    if use_patching:
        patching = GridPatching2D(
            img_shape=img_shape,
            patch_shape=patch_shape,
            boundary_pix=cfg.generation.boundary_pix,
            overlap_pix=cfg.generation.overlap_pix,
        )
        logger0.info("Patch-based training enabled")
    else:
        patching = None

    # Parse the inference mode
    if cfg.generation.inference_mode == "regression":
        load_net_reg, load_net_res = True, False
    elif cfg.generation.inference_mode == "diffusion":
        load_net_reg, load_net_res = False, True
    elif cfg.generation.inference_mode == "all":
        load_net_reg, load_net_res = True, True
    else:
        raise ValueError(f"Invalid inference mode {cfg.generation.inference_mode}")

    # Load diffusion network
    if load_net_res:
        res_ckpt_filename = getattr(cfg.generation.io, 'res_ckpt_filename', None)
        if res_ckpt_filename is None:
            raise ValueError("res_ckpt_filename not found in config")
        logger0.info(f'Loading residual network from "{res_ckpt_filename}"...')
        net_res = Module.from_checkpoint(
            to_absolute_path(res_ckpt_filename),
            override_args={"use_apex_gn": getattr(cfg.generation.perf, "use_apex_gn", False)},
        )
        net_res.profile_mode = getattr(cfg.generation.perf, "profile_mode", False)
        net_res.use_fp16 = getattr(cfg.generation.perf, "use_fp16", False)
        net_res = net_res.eval().to(device).to(memory_format=torch.channels_last)
        if hasattr(net_res, "amp_mode"):
            net_res.amp_mode = False
    else:
        net_res = None

    # Load regression network
    if load_net_reg:
        reg_ckpt_filename = getattr(cfg.generation.io, 'reg_ckpt_filename', None)
        if reg_ckpt_filename is None:
            raise ValueError("reg_ckpt_filename not found in config")
        logger0.info(f'Loading regression network from "{reg_ckpt_filename}"...')
        net_reg = Module.from_checkpoint(
            to_absolute_path(reg_ckpt_filename),
            override_args={"use_apex_gn": getattr(cfg.generation.perf, "use_apex_gn", False)},
        )
        net_reg.profile_mode = getattr(cfg.generation.perf, "profile_mode", False)
        net_reg.use_fp16 = getattr(cfg.generation.perf, "use_fp16", False)
        net_reg = net_reg.eval().to(device).to(memory_format=torch.channels_last)
        if hasattr(net_reg, "amp_mode"):
            net_reg.amp_mode = False
    else:
        net_reg = None

    # Torch compile if enabled
    if cfg.generation.perf.use_torch_compile:
        torch._dynamo.config.cache_size_limit = 264
        torch._dynamo.reset()
        if net_res:
            net_res = torch.compile(net_res)
        if net_reg:
            net_reg = torch.compile(net_reg)

    # Setup sampler function (use helper to allow hp_eval mode to swap samplers)
    sampler_fn = create_sampler_fn(cfg.sampler, patching)

    # Parse distribution settings
    distribution = getattr(cfg.generation, "distribution", None)
    student_t_nu = getattr(cfg.generation, "student_t_nu", None)
    P_mean = getattr(cfg.generation, "P_mean", None)
    P_std = getattr(cfg.generation, "P_std", None)

    def generate_for_time(image_lr, lead_time_label, current_sampler_fn=None):
        """Generate all ensemble members for a single time step.
        
        Args:
            image_lr: Low-resolution input image
            lead_time_label: Lead time label (if applicable)
            current_sampler_fn: Sampler function to use (defaults to outer sampler_fn)
        """
        if current_sampler_fn is None:
            current_sampler_fn = sampler_fn
            
        diffusion_step_kwargs = {}
        if distribution is not None:
            diffusion_step_kwargs["distribution"] = distribution
        if student_t_nu is not None:
            diffusion_step_kwargs["nu"] = student_t_nu
        if P_mean is not None:
            diffusion_step_kwargs["P_mean"] = P_mean
        if P_std is not None:
            diffusion_step_kwargs["P_std"] = P_std

        img_lr = image_lr.to(memory_format=torch.channels_last)
        all_outputs = []

        for seed_batch in seed_batches:
            batch_size = len(seed_batch)

            if net_reg:
                image_reg = regression_step(
                    net=net_reg,
                    img_lr=img_lr,
                    latents_shape=(batch_size, img_out_channels, img_shape[0], img_shape[1]),
                    lead_time_label=lead_time_label,
                )

            if net_res:
                mean_hr = image_reg[0:1] if (cfg.generation.hr_mean_conditioning and net_reg) else None
                rank_batches = [torch.tensor(seed_batch)]
                image_res = diffusion_step(
                    net=net_res,
                    sampler_fn=current_sampler_fn,
                    img_shape=img_shape,
                    img_out_channels=img_out_channels,
                    rank_batches=rank_batches,
                    img_lr=img_lr.expand(batch_size, -1, -1, -1).to(memory_format=torch.channels_last),
                    rank=0,
                    device=device,
                    mean_hr=mean_hr,
                    lead_time_label=lead_time_label,
                    **diffusion_step_kwargs,
                )

            if cfg.generation.inference_mode == "regression":
                batch_out = image_reg
            elif cfg.generation.inference_mode == "diffusion":
                batch_out = image_res
            else:
                batch_out = image_reg + image_res

            all_outputs.append(batch_out)

        return torch.cat(all_outputs, dim=0)

    # =========================================================================
    # HP Eval Mode: Run multiple sampler configs and pick best
    # =========================================================================
    hp_eval = getattr(cfg.generation, 'hp_eval', False)
    
    if hp_eval:
        logger0.info("=== HP Eval Mode: Testing multiple sampler configs ===")
        
        # Prepare dataset iteration info (shared across configs)
        all_times = dataset.time()
        times_list = [all_times[i] for i in sampler] if sampler else all_times
        max_times_per_gpu = (total_times + dist.world_size - 1) // dist.world_size
        
        all_config_results = {}
        
        for eval_cfg in HP_EVAL_CONFIGS:
            config_name = eval_cfg["name"]
            logger0.info(f"  Evaluating config: {config_name}")
            
            # Create sampler for this config
            config_sampler_fn = create_sampler_fn(eval_cfg["sampler"], patching)
            
            # Fresh metrics accumulator for this config
            config_metrics = MetricsAccumulator(device, precip_threshold=precip_threshold)
            
            # Generation loop for this config
            for iteration in range(max_times_per_gpu):
                time_idx = dist.rank + iteration * dist.world_size
                has_work = time_idx < total_times
                
                if has_work:
                    dataset_idx = sampler[time_idx]
                    
                    # Load data
                    image_tar, image_lr, *lead_time_label = dataset[dataset_idx]
                    
                    if isinstance(image_tar, np.ndarray):
                        image_tar = torch.from_numpy(image_tar)
                    if isinstance(image_lr, np.ndarray):
                        image_lr = torch.from_numpy(image_lr)
                    
                    image_tar = image_tar.unsqueeze(0).to(device=device, dtype=torch.float32)
                    image_lr = image_lr.unsqueeze(0).to(device=device, dtype=torch.float32).to(
                        memory_format=torch.channels_last
                    )
                    
                    if lead_time_label:
                        if isinstance(lead_time_label[0], np.ndarray):
                            lead_time_label = torch.from_numpy(lead_time_label[0])
                        else:
                            lead_time_label = lead_time_label[0]
                        lead_time_label = lead_time_label.unsqueeze(0).to(device).contiguous()
                    else:
                        lead_time_label = None
                    
                    # Generate with this config's sampler
                    image_out = generate_for_time(image_lr, lead_time_label, config_sampler_fn)
                    
                    # Update metrics
                    config_metrics.update(image_out, image_tar)
            
            # Reduce metrics across GPUs for this config
            local_metrics = config_metrics.compute()
            
            if dist.world_size > 1:
                # Gather metrics from all GPUs
                metrics_tensor = torch.tensor([
                    config_metrics.squared_error_sum,
                    float(config_metrics.total_elements),
                    config_metrics.crps_sum,
                    float(config_metrics.crps_elements),
                    config_metrics.spread_sum,
                    config_metrics.skill_sum,
                    float(config_metrics.n_samples),
                    config_metrics.squared_error_sum_gt_thresh,
                    float(config_metrics.elements_gt_thresh),
                    config_metrics.crps_sum_gt_thresh,
                    float(config_metrics.crps_elements_gt_thresh),
                    config_metrics.squared_error_sum_95th,
                    float(config_metrics.elements_95th),
                    config_metrics.crps_sum_95th,
                    float(config_metrics.crps_elements_95th),
                ], device=device)
                
                if dist.rank == 0:
                    gathered_metrics = [torch.zeros_like(metrics_tensor) for _ in range(dist.world_size)]
                else:
                    gathered_metrics = None
                
                torch.distributed.gather(metrics_tensor, gather_list=gathered_metrics, dst=0)
                
                if dist.rank == 0:
                    # Aggregate metrics
                    total_squared_error_sum = sum(m[0].item() for m in gathered_metrics)
                    total_elements = sum(int(m[1].item()) for m in gathered_metrics)
                    total_crps_sum = sum(m[2].item() for m in gathered_metrics)
                    total_crps_elements = sum(int(m[3].item()) for m in gathered_metrics)
                    total_spread_sum = sum(m[4].item() for m in gathered_metrics)
                    total_skill_sum = sum(m[5].item() for m in gathered_metrics)
                    total_n_samples = sum(int(m[6].item()) for m in gathered_metrics)
                    total_se_gt_thresh = sum(m[7].item() for m in gathered_metrics)
                    total_elements_gt_thresh = sum(int(m[8].item()) for m in gathered_metrics)
                    total_crps_gt_thresh = sum(m[9].item() for m in gathered_metrics)
                    total_crps_elements_gt_thresh = sum(int(m[10].item()) for m in gathered_metrics)
                    total_se_95th = sum(m[11].item() for m in gathered_metrics)
                    total_elements_95th = sum(int(m[12].item()) for m in gathered_metrics)
                    total_crps_95th = sum(m[13].item() for m in gathered_metrics)
                    total_crps_elements_95th = sum(int(m[14].item()) for m in gathered_metrics)
                    
                    rmse = np.sqrt(total_squared_error_sum / total_elements) if total_elements > 0 else 0.0
                    crps = total_crps_sum / total_crps_elements if total_crps_elements > 0 else 0.0
                    spread = total_spread_sum / total_n_samples if total_n_samples > 0 else 0.0
                    skill = total_skill_sum / total_n_samples if total_n_samples > 0 else 0.0
                    rmse_gt_thresh = np.sqrt(total_se_gt_thresh / total_elements_gt_thresh) if total_elements_gt_thresh > 0 else 0.0
                    crps_gt_thresh = total_crps_gt_thresh / total_crps_elements_gt_thresh if total_crps_elements_gt_thresh > 0 else 0.0
                    rmse_95th = np.sqrt(total_se_95th / total_elements_95th) if total_elements_95th > 0 else 0.0
                    crps_95th = total_crps_95th / total_crps_elements_95th if total_crps_elements_95th > 0 else 0.0
                    
                    config_final_metrics = {
                        'config_name': config_name,
                        'rmse': rmse,
                        'crps': crps,
                        'spread': spread,
                        'skill': skill,
                        'spread_skill_ratio': spread / skill if skill > 0 else float('inf'),
                        'n_samples': total_n_samples,
                        'total_elements': total_elements,
                        f'rmse_gt_{precip_threshold}mm': rmse_gt_thresh,
                        f'crps_gt_{precip_threshold}mm': crps_gt_thresh,
                        f'elements_gt_{precip_threshold}mm': total_elements_gt_thresh,
                        'rmse_95th_percentile': rmse_95th,
                        'crps_95th_percentile': crps_95th,
                        'elements_95th_percentile': total_elements_95th,
                    }
                    all_config_results[config_name] = config_final_metrics
                    logger0.info(f"    {config_name}: CRPS={crps:.6f}, RMSE={rmse:.6f}")
            else:
                # Single GPU
                if dist.rank == 0:
                    local_metrics['config_name'] = config_name
                    all_config_results[config_name] = local_metrics
                    logger0.info(f"    {config_name}: CRPS={local_metrics['crps']:.6f}, RMSE={local_metrics['rmse']:.6f}")
            
            # Synchronize before next config
            if dist.world_size > 1:
                torch.distributed.barrier()
        
        # Pick best config and write output (rank 0 only)
        if dist.rank == 0:
            best_config = None
            best_crps = float('inf')
            
            for config_name, result in all_config_results.items():
                if result['crps'] < best_crps:
                    best_crps = result['crps']
                    best_config = config_name
            
            logger0.info(f"=== HP Eval Results ===")
            logger0.info(f"  Best config: {best_config} with CRPS={best_crps:.6f}")
            
            # Write output with best config info
            final_output = {
                'success': True,
                'best_config': best_config,
                **all_config_results[best_config],
                'all_configs': all_config_results,
            }
            
            with open(metrics_output, 'w') as mf:
                json.dump(final_output, mf, indent=2)
            logger0.info(f"HP eval metrics saved to: {metrics_output}")
        
        logger0.info("HP Eval complete.")
        return  # Exit early - don't run normal generation

    # Setup output file (rank 0 only)
    output_path = getattr(cfg.generation.io, "output_filename", "corrdiff_output.nc")
    writer = None
    writer_threads = []
    f = None

    if skip_netcdf:
        logger0.info("NetCDF output: DISABLED (skip_netcdf=True)")
    else:
        logger0.info(f"Output: {output_path}")
        if dist.rank == 0:
            f = nc.Dataset(output_path, "w")
            f.cfg = str(cfg)
            import datetime
            f.generation_timestamp = datetime.datetime.now().isoformat()
            f.parallelization_mode = "time_parallel"
            f.num_gpus = dist.world_size

            writer = NetCDFWriter(
                f,
                lat=dataset.latitude(),
                lon=dataset.longitude(),
                input_channels=dataset.input_channels(),
                output_channels=dataset.output_channels(),
                has_lead_time=has_lead_time,
            )

            if cfg.generation.perf.io_syncronous:
                writer_executor = ThreadPoolExecutor(max_workers=cfg.generation.perf.num_writer_workers)
                writer_threads = []

    # Distribute time steps across GPUs
    all_times = dataset.time()
    times_list = [all_times[i] for i in sampler] if sampler else all_times

    # Each GPU gets every world_size-th time step
    rank_time_indices = list(range(dist.rank, total_times, dist.world_size))
    logger.info(f"[GPU {dist.rank}] Processing {len(rank_time_indices)} time steps: {rank_time_indices[:5]}...")

    # Calculate iterations needed (all GPUs must do same number for gather sync)
    max_times_per_gpu = (total_times + dist.world_size - 1) // dist.world_size

    # Profiling
    use_cuda_timing = torch.cuda.is_available()
    if use_cuda_timing:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

    # Main generation loop
    for iteration in range(max_times_per_gpu):
        # Calculate which time index this GPU handles in this iteration
        time_idx = dist.rank + iteration * dist.world_size
        has_work = time_idx < total_times

        if has_work:
            dataset_idx = sampler[time_idx]
            time_str = times_list[time_idx]
            logger.info(f"[GPU {dist.rank}] Iteration {iteration+1}/{max_times_per_gpu}: "
                       f"time_idx={time_idx}, time={time_str}")

            # Load data
            image_tar, image_lr, *lead_time_label = dataset[dataset_idx]

            if isinstance(image_tar, np.ndarray):
                image_tar = torch.from_numpy(image_tar)
            if isinstance(image_lr, np.ndarray):
                image_lr = torch.from_numpy(image_lr)

            image_tar = image_tar.unsqueeze(0).to(device=device, dtype=torch.float32)
            image_lr = image_lr.unsqueeze(0).to(device=device, dtype=torch.float32).to(
                memory_format=torch.channels_last
            )

            if lead_time_label:
                if isinstance(lead_time_label[0], np.ndarray):
                    lead_time_label = torch.from_numpy(lead_time_label[0])
                else:
                    lead_time_label = lead_time_label[0]
                lead_time_label = lead_time_label.unsqueeze(0).to(device).contiguous()
            else:
                lead_time_label = None

            # Generate
            image_out = generate_for_time(image_lr, lead_time_label)

            # Update metrics (on GPU, before CPU transfer - very efficient)
            if metrics is not None:
                metrics.update(image_out, image_tar)
        else:
            # Padding iteration - create dummy tensors for gather
            time_idx = -1
            in_channels = len(dataset.input_channels())
            image_out = torch.zeros(num_ensembles, img_out_channels, img_shape[0], img_shape[1],
                                   device=device, dtype=torch.float32)
            image_tar = torch.zeros(1, img_out_channels, img_shape[0], img_shape[1],
                                   device=device, dtype=torch.float32)
            image_lr = torch.zeros(1, in_channels, img_shape[0], img_shape[1],
                                  device=device, dtype=torch.float32)

        # Gather results from all GPUs
        if dist.world_size > 1:
            time_idx_tensor = torch.tensor([time_idx], device=device, dtype=torch.long)

            if dist.rank == 0:
                gathered_time_indices = [torch.zeros(1, dtype=torch.long, device=device)
                                        for _ in range(dist.world_size)]
                gathered_outputs = [torch.zeros_like(image_out) for _ in range(dist.world_size)]
                gathered_targets = [torch.zeros_like(image_tar) for _ in range(dist.world_size)]
                gathered_inputs = [torch.zeros_like(image_lr) for _ in range(dist.world_size)]
            else:
                gathered_time_indices = gathered_outputs = gathered_targets = gathered_inputs = None

            torch.distributed.barrier()
            torch.distributed.gather(time_idx_tensor, gather_list=gathered_time_indices, dst=0)
            torch.distributed.gather(image_out, gather_list=gathered_outputs, dst=0)
            torch.distributed.gather(image_tar, gather_list=gathered_targets, dst=0)
            torch.distributed.gather(image_lr, gather_list=gathered_inputs, dst=0)

            # Write on rank 0 (skip if skip_netcdf)
            if dist.rank == 0 and writer is not None:
                for gpu_idx in range(dist.world_size):
                    t_idx = gathered_time_indices[gpu_idx].item()
                    if t_idx < 0:
                        continue  # Skip padding

                    if cfg.generation.perf.io_syncronous:
                        writer_threads.append(
                            writer_executor.submit(
                                save_images, writer, dataset, list(times_list),
                                gathered_outputs[gpu_idx].cpu(),
                                gathered_targets[gpu_idx].cpu(),
                                gathered_inputs[gpu_idx].cpu(),
                                t_idx, t_idx, has_lead_time,
                            )
                        )
                    else:
                        save_images(
                            writer, dataset, list(times_list),
                            gathered_outputs[gpu_idx].cpu(),
                            gathered_targets[gpu_idx].cpu(),
                            gathered_inputs[gpu_idx].cpu(),
                            t_idx, t_idx, has_lead_time,
                        )
        else:
            # Single GPU (skip if skip_netcdf)
            if time_idx >= 0 and writer is not None:
                if cfg.generation.perf.io_syncronous:
                    writer_threads.append(
                        writer_executor.submit(
                            save_images, writer, dataset, list(times_list),
                            image_out.cpu(), image_tar.cpu(), image_lr.cpu(),
                            time_idx, time_idx, has_lead_time,
                        )
                    )
                else:
                    save_images(
                        writer, dataset, list(times_list),
                        image_out.cpu(), image_tar.cpu(), image_lr.cpu(),
                        time_idx, time_idx, has_lead_time,
                    )

    # Timing
    if use_cuda_timing:
        end.record()
        end.synchronize()
        elapsed = start.elapsed_time(end) / 1000.0
        if dist.rank == 0:
            logger.info(f"=== Performance ===")
            logger.info(f"  Total time: {elapsed:.2f}s")
            logger.info(f"  Time steps: {total_times}")
            logger.info(f"  Samples: {total_times * num_ensembles}")
            logger.info(f"  Throughput: {total_times * num_ensembles / elapsed:.2f} samples/s")

    # Compute and save metrics
    if metrics is not None:
        # Reduce metrics across GPUs
        local_metrics = metrics.compute()

        if dist.world_size > 1:
            # Gather metrics from all GPUs to rank 0
            # Send raw accumulator values for proper aggregation
            metrics_tensor = torch.tensor([
                metrics.squared_error_sum,           # 0: for RMSE
                float(metrics.total_elements),       # 1: element count for RMSE
                metrics.crps_sum,                    # 2: for CRPS
                float(metrics.crps_elements),        # 3: element count for CRPS
                metrics.spread_sum,                  # 4: for spread
                metrics.skill_sum,                   # 5: for skill
                float(metrics.n_samples),            # 6: sample count
                # Conditional metrics: > threshold
                metrics.squared_error_sum_gt_thresh, # 7: SE for high precip
                float(metrics.elements_gt_thresh),   # 8: element count for high precip
                metrics.crps_sum_gt_thresh,          # 9: CRPS sum for high precip
                float(metrics.crps_elements_gt_thresh),  # 10: CRPS element count for high precip
                # Conditional metrics: 95th percentile
                metrics.squared_error_sum_95th,      # 11: SE for 95th percentile
                float(metrics.elements_95th),        # 12: element count for 95th percentile
                metrics.crps_sum_95th,               # 13: CRPS sum for 95th percentile
                float(metrics.crps_elements_95th),   # 14: CRPS element count for 95th percentile
            ], device=device)

            if dist.rank == 0:
                gathered_metrics = [torch.zeros_like(metrics_tensor) for _ in range(dist.world_size)]
            else:
                gathered_metrics = None

            torch.distributed.gather(metrics_tensor, gather_list=gathered_metrics, dst=0)

            if dist.rank == 0:
                # Aggregate metrics properly
                total_squared_error_sum = sum(m[0].item() for m in gathered_metrics)
                total_elements = sum(int(m[1].item()) for m in gathered_metrics)
                total_crps_sum = sum(m[2].item() for m in gathered_metrics)
                total_crps_elements = sum(int(m[3].item()) for m in gathered_metrics)
                total_spread_sum = sum(m[4].item() for m in gathered_metrics)
                total_skill_sum = sum(m[5].item() for m in gathered_metrics)
                total_n_samples = sum(int(m[6].item()) for m in gathered_metrics)

                # Conditional: > threshold
                total_se_gt_thresh = sum(m[7].item() for m in gathered_metrics)
                total_elements_gt_thresh = sum(int(m[8].item()) for m in gathered_metrics)
                total_crps_gt_thresh = sum(m[9].item() for m in gathered_metrics)
                total_crps_elements_gt_thresh = sum(int(m[10].item()) for m in gathered_metrics)

                # Conditional: 95th percentile
                total_se_95th = sum(m[11].item() for m in gathered_metrics)
                total_elements_95th = sum(int(m[12].item()) for m in gathered_metrics)
                total_crps_95th = sum(m[13].item() for m in gathered_metrics)
                total_crps_elements_95th = sum(int(m[14].item()) for m in gathered_metrics)

                # RMSE: sqrt(total_squared_error / total_elements)
                rmse = np.sqrt(total_squared_error_sum / total_elements) if total_elements > 0 else 0.0
                # CRPS: mean of (MAE - 0.5*std)
                crps = total_crps_sum / total_crps_elements if total_crps_elements > 0 else 0.0
                # Spread and skill averages
                spread = total_spread_sum / total_n_samples if total_n_samples > 0 else 0.0
                skill = total_skill_sum / total_n_samples if total_n_samples > 0 else 0.0

                # Conditional metrics
                rmse_gt_thresh = np.sqrt(total_se_gt_thresh / total_elements_gt_thresh) if total_elements_gt_thresh > 0 else 0.0
                crps_gt_thresh = total_crps_gt_thresh / total_crps_elements_gt_thresh if total_crps_elements_gt_thresh > 0 else 0.0
                rmse_95th = np.sqrt(total_se_95th / total_elements_95th) if total_elements_95th > 0 else 0.0
                crps_95th = total_crps_95th / total_crps_elements_95th if total_crps_elements_95th > 0 else 0.0

                final_metrics = {
                    'rmse': rmse,
                    'crps': crps,
                    'spread': spread,
                    'skill': skill,
                    'n_samples': total_n_samples,
                    'total_elements': total_elements,
                    # Conditional metrics
                    f'rmse_gt_{precip_threshold}mm': rmse_gt_thresh,
                    f'crps_gt_{precip_threshold}mm': crps_gt_thresh,
                    f'elements_gt_{precip_threshold}mm': total_elements_gt_thresh,
                    'rmse_95th_percentile': rmse_95th,
                    'crps_95th_percentile': crps_95th,
                    'elements_95th_percentile': total_elements_95th,
                }
                final_metrics['spread_skill_ratio'] = spread / skill if skill > 0 else float('inf')
        else:
            final_metrics = local_metrics

        if dist.rank == 0:
            logger.info(f"=== Metrics (matching wandb_eval) ===")
            logger.info(f"  RMSE: {final_metrics['rmse']:.6f}")
            logger.info(f"  CRPS: {final_metrics['crps']:.6f}")
            logger.info(f"  Spread: {final_metrics['spread']:.6f}")
            logger.info(f"  Skill: {final_metrics['skill']:.6f}")
            logger.info(f"  Spread/Skill: {final_metrics['spread_skill_ratio']:.4f}")
            logger.info(f"  N samples: {final_metrics['n_samples']}")
            logger.info(f"  Total elements: {final_metrics.get('total_elements', 'N/A')}")
            logger.info(f"  --- Conditional Metrics ---")
            logger.info(f"  RMSE (target > {precip_threshold}mm): {final_metrics.get(f'rmse_gt_{precip_threshold}mm', 'N/A'):.6f}")
            logger.info(f"  CRPS (target > {precip_threshold}mm): {final_metrics.get(f'crps_gt_{precip_threshold}mm', 'N/A'):.6f}")
            logger.info(f"  Elements (target > {precip_threshold}mm): {final_metrics.get(f'elements_gt_{precip_threshold}mm', 'N/A')}")
            logger.info(f"  RMSE (95th percentile): {final_metrics.get('rmse_95th_percentile', 'N/A'):.6f}")
            logger.info(f"  CRPS (95th percentile): {final_metrics.get('crps_95th_percentile', 'N/A'):.6f}")
            logger.info(f"  Elements (95th percentile): {final_metrics.get('elements_95th_percentile', 'N/A')}")

            # Save to JSON for Optuna
            with open(metrics_output, 'w') as mf:
                json.dump(final_metrics, mf, indent=2)
            logger.info(f"Metrics saved to: {metrics_output}")

    # Wait for writers
    if dist.rank == 0 and writer_threads and not skip_netcdf:
        if cfg.generation.perf.io_syncronous:
            for thread in writer_threads:
                thread.result()
            writer_executor.shutdown()

    if dist.rank == 0 and f is not None:
        f.close()
    logger0.info("Generation complete.")


if __name__ == "__main__":
    main()
