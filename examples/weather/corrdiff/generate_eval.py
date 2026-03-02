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
Evaluation script for heteroscedastic diffusion models with post-diffusion rescaling.

Generates ensemble members using the standard stochastic sampler, then evaluates
multiple post-hoc rescaling strategies WITHOUT re-running diffusion:

  1. Baseline: raw diffusion output (no rescaling)
  2. T-only: uniform temperature scaling of residuals
  3. T*D_std: temperature + D_std spatial modulation of residuals

For each strategy and temperature, computes:
  - Ensemble CRPS (proper finite-ensemble estimate)
  - RMSE of ensemble mean
  - Spread (ensemble std)
  - Spread-skill reliability diagnostics from spread bins
  - Conditional metrics for extreme precipitation

Time-step parallelism across GPUs (each GPU handles different time steps).

Usage:
    # Single GPU:
    python generate_eval.py --config-name=eval_heteroscedastic

    # Multi-GPU:
    torchrun --nproc_per_node=4 generate_eval.py --config-name=eval_heteroscedastic
"""

import json
from functools import partial

import hydra
from omegaconf import OmegaConf, DictConfig
from hydra.utils import to_absolute_path
import torch
import torch._dynamo
import numpy as np

from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.utils.patching import GridPatching2D
from physicsnemo import Module
from physicsnemo.utils.corrdiff import (
    get_time_from_range,
    regression_step,
    diffusion_step,
)

# Use local stochastic_sampler which handles heteroscedastic tuple returns
from helpers.stochastic_sampler import stochastic_sampler

from helpers.generate_helpers import get_dataset_and_sampler
from helpers.train_helpers import set_patch_shape
from datasets.dataset import register_dataset


class MultiStrategyMetrics:
    """Accumulates metrics for multiple rescaling strategies simultaneously.

    For each (strategy, temperature) pair, tracks RMSE, CRPS, spread, and
    conditional metrics. All computation on GPU before CPU transfer.
    """

    def __init__(
        self,
        strategies,
        device,
        precip_threshold=1.0,
        spread_skill_bin_edges=None,
    ):
        """
        Args:
            strategies: list of (name, T, use_d_std) tuples
            device: torch device
            precip_threshold: threshold for conditional metrics (mm)
        """
        self.strategies = strategies
        self.device = device
        self.precip_threshold = precip_threshold
        if spread_skill_bin_edges is None:
            spread_skill_bin_edges = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, float("inf")]
        self.spread_skill_bin_edges = torch.tensor(
            spread_skill_bin_edges, device=device, dtype=torch.float32
        )
        if self.spread_skill_bin_edges.numel() < 2:
            raise ValueError("spread_skill_bin_edges must have at least two edges")
        self.n_spread_bins = self.spread_skill_bin_edges.numel() - 1
        self.accumulators = {name: self._empty_acc() for name, _, _ in strategies}

    @staticmethod
    def _ensemble_crps(pred_ens: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        """Element-wise finite-ensemble CRPS.

        pred_ens: (N_ens, C, H, W)
        obs: (C, H, W)
        returns: (C, H, W)
        """
        n_ens = pred_ens.shape[0]
        if n_ens < 1:
            raise ValueError("Ensemble must contain at least one member")

        term_obs = torch.abs(pred_ens - obs.unsqueeze(0)).mean(dim=0)

        if n_ens == 1:
            return term_obs

        pairwise = torch.abs(pred_ens.unsqueeze(0) - pred_ens.unsqueeze(1)).mean(dim=(0, 1))
        return term_obs - 0.5 * pairwise

    def _empty_acc(self):
        return {
            "se_sum": 0.0,
            "n_elements": 0,
            "crps_sum": 0.0,
            "crps_elements": 0,
            "spread_sum": 0.0,
            "skill_sum": 0.0,
            "n_samples": 0,
            # Conditional: > threshold
            "se_sum_gt": 0.0,
            "n_gt": 0,
            "crps_sum_gt": 0.0,
            "crps_n_gt": 0,
            # Conditional: 95th percentile
            "se_sum_95": 0.0,
            "n_95": 0,
            "crps_sum_95": 0.0,
            "crps_n_95": 0,
            # Conditional: 95th percentile over wet pixels only
            "se_sum_w95": 0.0,
            "n_w95": 0,
            "crps_sum_w95": 0.0,
            "crps_n_w95": 0,
            # Spread-skill reliability bins
            "spread_bin_sum": torch.zeros(self.n_spread_bins, device=self.device, dtype=torch.float64),
            "skill_bin_sum": torch.zeros(self.n_spread_bins, device=self.device, dtype=torch.float64),
            "bin_count": torch.zeros(self.n_spread_bins, device=self.device, dtype=torch.float64),
        }

    @torch.no_grad()
    def update(
        self,
        baseline_residuals,
        reg_mean,
        target,
        d_std_norm,
        denorm_fn,
    ):
        """Update all strategies for one time step.

        Args:
            baseline_residuals: (N_ens, C, H, W) raw diffusion residuals
            reg_mean: (1, C, H, W) regression mean (normalized)
            target: (1, C, H, W) ground truth (normalized)
            d_std_norm: (1, C, H, W) RMS-normalized D_std pattern, or None
            denorm_fn: callable that denormalizes (N, C, H, W) numpy arrays
        """
        baseline_residuals = baseline_residuals.to(self.device)
        reg_mean = reg_mean.to(self.device)
        target = target.to(self.device)

        for name, T, use_d_std in self.strategies:
            # Rescale residuals
            if use_d_std and d_std_norm is not None:
                d = d_std_norm.to(self.device)
                rescaled = baseline_residuals * T * d
            else:
                rescaled = baseline_residuals * T

            # Denormalize predictions and target
            pred_norm = reg_mean + rescaled  # (N_ens, C, H, W)
            pred_np = denorm_fn(pred_norm.cpu().numpy())
            tar_np = denorm_fn(target.cpu().numpy())

            # Clip to >= 0 (precipitation)
            pred_t = torch.from_numpy(np.clip(pred_np, 0, None)).to(self.device)
            tar_t = torch.from_numpy(np.clip(tar_np, 0, None)).to(self.device)

            ens_mean = pred_t.mean(dim=0)  # (C, H, W)
            ens_std = pred_t.std(dim=0)
            tar_sq = tar_t.squeeze(0)  # (C, H, W)

            # RMSE components
            se = (ens_mean - tar_sq) ** 2

            # Proper finite-ensemble CRPS
            crps_vals = self._ensemble_crps(pred_t, tar_sq)

            acc = self.accumulators[name]
            acc["se_sum"] += se.sum().item()
            acc["n_elements"] += se.numel()
            acc["crps_sum"] += crps_vals.sum().item()
            acc["crps_elements"] += crps_vals.numel()
            acc["spread_sum"] += ens_std.mean().item()
            acc["skill_sum"] += torch.sqrt(se.mean()).item()
            acc["n_samples"] += 1

            # Conditional: > threshold
            mask_gt = tar_sq > self.precip_threshold
            if mask_gt.any():
                acc["se_sum_gt"] += se[mask_gt].sum().item()
                acc["n_gt"] += mask_gt.sum().item()
                acc["crps_sum_gt"] += crps_vals[mask_gt].sum().item()
                acc["crps_n_gt"] += mask_gt.sum().item()

            # Conditional: 95th percentile
            p95 = torch.quantile(tar_sq.flatten(), 0.95)
            mask_95 = tar_sq >= p95
            if mask_95.any():
                acc["se_sum_95"] += se[mask_95].sum().item()
                acc["n_95"] += mask_95.sum().item()
                acc["crps_sum_95"] += crps_vals[mask_95].sum().item()
                acc["crps_n_95"] += mask_95.sum().item()

            # Conditional: W95th (95th percentile over wet pixels only)
            wet_mask = tar_sq > 0.0
            if wet_mask.any():
                p95_wet = torch.quantile(tar_sq[wet_mask], 0.95)
                mask_w95 = wet_mask & (tar_sq >= p95_wet)
                if mask_w95.any():
                    acc["se_sum_w95"] += se[mask_w95].sum().item()
                    acc["n_w95"] += mask_w95.sum().item()
                    acc["crps_sum_w95"] += crps_vals[mask_w95].sum().item()
                    acc["crps_n_w95"] += mask_w95.sum().item()

            # Spread-skill reliability bins (pixel-level)
            spread_flat = ens_std.flatten()
            skill_flat = torch.sqrt(se).flatten()
            bin_ids = torch.bucketize(
                spread_flat,
                boundaries=self.spread_skill_bin_edges[1:-1],
                right=False,
            )
            for b in range(self.n_spread_bins):
                in_bin = bin_ids == b
                if in_bin.any():
                    acc["spread_bin_sum"][b] += spread_flat[in_bin].sum().to(torch.float64)
                    acc["skill_bin_sum"][b] += skill_flat[in_bin].sum().to(torch.float64)
                    acc["bin_count"][b] += in_bin.sum().to(torch.float64)

    def compute(self):
        """Compute final metrics for all strategies."""
        results = {}
        for name, T, use_d_std in self.strategies:
            acc = self.accumulators[name]
            n = acc["n_samples"]
            if n == 0:
                continue
            ne = acc["n_elements"]
            bin_count = acc["bin_count"]
            valid_bins = bin_count > 0
            if valid_bins.any():
                mean_spread_bins = torch.zeros_like(bin_count)
                mean_skill_bins = torch.zeros_like(bin_count)
                mean_spread_bins[valid_bins] = acc["spread_bin_sum"][valid_bins] / bin_count[valid_bins]
                mean_skill_bins[valid_bins] = acc["skill_bin_sum"][valid_bins] / bin_count[valid_bins]
                x = mean_spread_bins[valid_bins]
                y = mean_skill_bins[valid_bins]
                w = bin_count[valid_bins]
                w_sum = w.sum()
                x_bar = (w * x).sum() / w_sum
                y_bar = (w * y).sum() / w_sum
                s_xx = (w * (x - x_bar) ** 2).sum()
                if s_xx > 0:
                    slope = ((w * (x - x_bar) * (y - y_bar)).sum() / s_xx).item()
                else:
                    slope = float("nan")
                intercept = (y_bar - slope * x_bar).item() if np.isfinite(slope) else float("nan")
            else:
                mean_spread_bins = torch.zeros(self.n_spread_bins, dtype=torch.float64)
                mean_skill_bins = torch.zeros(self.n_spread_bins, dtype=torch.float64)
                slope = float("nan")
                intercept = float("nan")

            results[name] = {
                "temperature": T,
                "use_d_std": use_d_std,
                "rmse": np.sqrt(acc["se_sum"] / ne) if ne > 0 else 0.0,
                "crps": acc["crps_sum"] / acc["crps_elements"] if acc["crps_elements"] > 0 else 0.0,
                "spread": acc["spread_sum"] / n,
                "skill": acc["skill_sum"] / n,
                "spread_skill_ratio": (acc["spread_sum"] / n) / (acc["skill_sum"] / n) if acc["skill_sum"] > 0 else float("inf"),
                "n_samples": n,
                f"rmse_gt_{self.precip_threshold}mm": np.sqrt(acc["se_sum_gt"] / acc["n_gt"]) if acc["n_gt"] > 0 else 0.0,
                f"crps_gt_{self.precip_threshold}mm": acc["crps_sum_gt"] / acc["crps_n_gt"] if acc["crps_n_gt"] > 0 else 0.0,
                "rmse_95th": np.sqrt(acc["se_sum_95"] / acc["n_95"]) if acc["n_95"] > 0 else 0.0,
                "crps_95th": acc["crps_sum_95"] / acc["crps_n_95"] if acc["crps_n_95"] > 0 else 0.0,
                "rmse_w95th": np.sqrt(acc["se_sum_w95"] / acc["n_w95"]) if acc["n_w95"] > 0 else 0.0,
                "crps_w95th": acc["crps_sum_w95"] / acc["crps_n_w95"] if acc["crps_n_w95"] > 0 else 0.0,
                "spread_skill_bin_edges": self.spread_skill_bin_edges.detach().cpu().tolist(),
                "spread_skill_bin_mean_spread": mean_spread_bins.detach().cpu().tolist(),
                "spread_skill_bin_mean_skill": mean_skill_bins.detach().cpu().tolist(),
                "spread_skill_reliability_slope": slope,
                "spread_skill_reliability_intercept": intercept,
            }
        return results

    def get_raw_accumulators(self):
        """Return raw accumulator values as a flat tensor for distributed gather."""
        vals = []
        for name, _, _ in self.strategies:
            acc = self.accumulators[name]
            vals.extend([
                acc["se_sum"], float(acc["n_elements"]),
                acc["crps_sum"], float(acc["crps_elements"]),
                acc["spread_sum"], acc["skill_sum"], float(acc["n_samples"]),
                acc["se_sum_gt"], float(acc["n_gt"]),
                acc["crps_sum_gt"], float(acc["crps_n_gt"]),
                acc["se_sum_95"], float(acc["n_95"]),
                acc["crps_sum_95"], float(acc["crps_n_95"]),
                acc["se_sum_w95"], float(acc["n_w95"]),
                acc["crps_sum_w95"], float(acc["crps_n_w95"]),
                *acc["spread_bin_sum"].tolist(),
                *acc["skill_bin_sum"].tolist(),
                *acc["bin_count"].tolist(),
            ])
        return torch.tensor(vals, device=self.device)

    def merge_gathered(self, gathered_tensors):
        """Merge gathered accumulator tensors from all GPUs."""
        n_fields = 19 + 3 * self.n_spread_bins  # fields per strategy
        for strat_idx, (name, _, _) in enumerate(self.strategies):
            acc = self.accumulators[name]
            for gt in gathered_tensors:
                offset = strat_idx * n_fields
                vals = gt[offset:offset + n_fields]
                acc["se_sum"] += vals[0].item()
                acc["n_elements"] += int(vals[1].item())
                acc["crps_sum"] += vals[2].item()
                acc["crps_elements"] += int(vals[3].item())
                acc["spread_sum"] += vals[4].item()
                acc["skill_sum"] += vals[5].item()
                acc["n_samples"] += int(vals[6].item())
                acc["se_sum_gt"] += vals[7].item()
                acc["n_gt"] += int(vals[8].item())
                acc["crps_sum_gt"] += vals[9].item()
                acc["crps_n_gt"] += int(vals[10].item())
                acc["se_sum_95"] += vals[11].item()
                acc["n_95"] += int(vals[12].item())
                acc["crps_sum_95"] += vals[13].item()
                acc["crps_n_95"] += int(vals[14].item())
                acc["se_sum_w95"] += vals[15].item()
                acc["n_w95"] += int(vals[16].item())
                acc["crps_sum_w95"] += vals[17].item()
                acc["crps_n_w95"] += int(vals[18].item())
                b0 = 19
                b1 = b0 + self.n_spread_bins
                b2 = b1 + self.n_spread_bins
                b3 = b2 + self.n_spread_bins
                acc["spread_bin_sum"] += vals[b0:b1].to(torch.float64)
                acc["skill_bin_sum"] += vals[b1:b2].to(torch.float64)
                acc["bin_count"] += vals[b2:b3].to(torch.float64)


@hydra.main(version_base="1.2", config_path="conf", config_name="config_generate")
def main(cfg: DictConfig) -> None:
    """Evaluate heteroscedastic model with post-diffusion rescaling."""

    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device

    logger = PythonLogger("generate_eval")
    logger0 = RankZeroLoggingWrapper(logger, dist)
    logger.file_logging("generate_eval.log")

    num_ensembles = cfg.generation.num_ensembles
    seed_batch_size = cfg.generation.seed_batch_size
    precip_threshold = getattr(cfg.generation, "precip_threshold", 1.0)

    # Temperature values to evaluate
    T_values = list(getattr(cfg.generation, "temperature_values", [0.5, 0.7, 1.0, 1.5, 2.0]))

    # Build strategy list: (name, T, use_d_std)
    strategies = [("baseline", 1.0, False)]
    for T in T_values:
        strategies.append((f"T_only_{T}", T, False))
        strategies.append((f"T_Dstd_{T}", T, True))

    logger0.info(f"=== Post-Diffusion Rescaling Evaluation ===")
    logger0.info(f"  Temperatures: {T_values}")
    logger0.info(f"  Strategies: {len(strategies)}")
    logger0.info(f"  Ensembles: {num_ensembles}")

    # Prepare seed batches
    seeds = list(np.arange(num_ensembles))
    num_seed_batches = (len(seeds) - 1) // seed_batch_size + 1
    seed_batches = np.array_split(seeds, num_seed_batches)

    if dist.world_size > 1:
        torch.distributed.barrier()

    # Parse times
    has_times_range = hasattr(cfg.generation, "times_range") and cfg.generation.times_range is not None
    has_times = hasattr(cfg.generation, "times") and cfg.generation.times is not None
    if has_times_range and has_times:
        raise ValueError("Provide times_range or times, not both")
    elif has_times_range:
        times = get_time_from_range(cfg.generation.times_range)
    elif has_times:
        times = cfg.generation.times
    else:
        raise ValueError("Either times_range or times must be provided")

    # Dataset
    dataset_cfg = OmegaConf.to_container(cfg.dataset)
    register_dataset(cfg.dataset.type)
    has_lead_time = cfg.generation.get("has_lead_time", False)
    dataset, sampler = get_dataset_and_sampler(
        dataset_cfg=dataset_cfg, times=times, has_lead_time=has_lead_time
    )
    total_times = len(sampler)
    img_shape = dataset.image_shape()
    img_out_channels = len(dataset.output_channels())

    logger0.info(f"  Time steps: {total_times}")
    logger0.info(f"  GPUs: {dist.world_size}")
    logger0.info(f"  Time steps per GPU: ~{(total_times + dist.world_size - 1) // dist.world_size}")

    # Patching
    if cfg.generation.patching:
        patch_shape = (cfg.generation.patch_shape_y, cfg.generation.patch_shape_x)
    else:
        patch_shape = (None, None)
    use_patching, img_shape, patch_shape = set_patch_shape(img_shape, patch_shape)
    if use_patching:
        patching = GridPatching2D(
            img_shape=img_shape, patch_shape=patch_shape,
            boundary_pix=cfg.generation.boundary_pix,
            overlap_pix=cfg.generation.overlap_pix,
        )
    else:
        patching = None

    # Load models
    if cfg.generation.inference_mode in ("diffusion", "all"):
        res_ckpt = to_absolute_path(cfg.generation.io.res_ckpt_filename)
        logger0.info(f'Loading diffusion model from "{res_ckpt}"...')
        net_res = Module.from_checkpoint(
            res_ckpt,
            override_args={"use_apex_gn": getattr(cfg.generation.perf, "use_apex_gn", False)},
        )
        net_res.profile_mode = getattr(cfg.generation.perf, "profile_mode", False)
        net_res.use_fp16 = getattr(cfg.generation.perf, "use_fp16", False)
        net_res = net_res.eval().to(device).to(memory_format=torch.channels_last)
        if hasattr(net_res, "amp_mode"):
            net_res.amp_mode = False
    else:
        net_res = None

    if cfg.generation.inference_mode in ("regression", "all"):
        reg_ckpt = to_absolute_path(cfg.generation.io.reg_ckpt_filename)
        logger0.info(f'Loading regression model from "{reg_ckpt}"...')
        net_reg = Module.from_checkpoint(
            reg_ckpt,
            override_args={"use_apex_gn": getattr(cfg.generation.perf, "use_apex_gn", False)},
        )
        net_reg.profile_mode = getattr(cfg.generation.perf, "profile_mode", False)
        net_reg.use_fp16 = getattr(cfg.generation.perf, "use_fp16", False)
        net_reg = net_reg.eval().to(device).to(memory_format=torch.channels_last)
        if hasattr(net_reg, "amp_mode"):
            net_reg.amp_mode = False
    else:
        net_reg = None

    # Torch compile
    if cfg.generation.perf.use_torch_compile:
        torch._dynamo.config.cache_size_limit = 264
        torch._dynamo.reset()
        if net_res:
            net_res = torch.compile(net_res)
        if net_reg:
            net_reg = torch.compile(net_reg)

    # Check if model supports variance head
    has_variance_head = hasattr(net_res, "variance_head") if net_res else False
    if has_variance_head:
        logger0.info("  Variance head detected — D_std spatial modulation enabled")
    else:
        logger0.info("  No variance head — D_std strategies will use uniform weights")

    # Sampler: always use standard stochastic sampler (clean diffusion)
    sampler_fn = partial(
        stochastic_sampler,
        patching=patching,
        num_steps=getattr(cfg.sampler, "num_steps", 18),
        S_churn=getattr(cfg.sampler, "S_churn", 0),
        S_min=getattr(cfg.sampler, "S_min", 0),
        S_max=getattr(cfg.sampler, "S_max", float("inf")),
    )

    # Distribution kwargs
    diffusion_kwargs = {}
    for key in ("distribution", "student_t_nu", "P_mean", "P_std"):
        val = getattr(cfg.generation, key, None)
        if val is not None:
            diffusion_kwargs[key] = val

    # Metrics accumulator
    metrics = MultiStrategyMetrics(strategies, device, precip_threshold)

    # D_std probe sigma
    d_std_sigma = getattr(cfg.generation, "d_std_sigma", 0.01)

    # Main loop — distribute time steps across GPUs
    all_times = dataset.time()
    times_list = [all_times[i] for i in sampler] if sampler else all_times
    max_iters = (total_times + dist.world_size - 1) // dist.world_size

    logger0.info(f"Starting evaluation loop...")

    for iteration in range(max_iters):
        time_idx = dist.rank + iteration * dist.world_size
        if time_idx >= total_times:
            continue

        dataset_idx = sampler[time_idx]
        logger.info(
            f"[GPU {dist.rank}] {iteration+1}/{max_iters}: "
            f"time_idx={time_idx} ({times_list[time_idx]})"
        )

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
            lt = lead_time_label[0]
            if isinstance(lt, np.ndarray):
                lt = torch.from_numpy(lt)
            lead_time_label = lt.unsqueeze(0).to(device).contiguous()
        else:
            lead_time_label = None

        # Step 1: Regression mean
        if net_reg:
            with torch.no_grad():
                image_reg = regression_step(
                    net=net_reg,
                    img_lr=image_lr,
                    latents_shape=(1, img_out_channels, img_shape[0], img_shape[1]),
                    lead_time_label=lead_time_label,
                )
            reg_mean = image_reg[0:1]
        else:
            reg_mean = torch.zeros(1, img_out_channels, img_shape[0], img_shape[1], device=device)

        # Step 2: Generate ensemble residuals (standard sampler, one run)
        if net_res:
            mean_hr = reg_mean if cfg.generation.hr_mean_conditioning else None
            all_residuals = []
            for seed_batch in seed_batches:
                batch_size = len(seed_batch)
                rank_batches = [torch.tensor(seed_batch)]
                with torch.no_grad():
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
            baseline_residuals = torch.cat(all_residuals, dim=0)  # (N_ens, C, H, W)
        else:
            baseline_residuals = torch.zeros(
                num_ensembles, img_out_channels, img_shape[0], img_shape[1], device=device
            )

        # Step 3: Get D_std from variance head (one forward pass)
        d_std_norm = None
        if has_variance_head and net_res is not None:
            with torch.no_grad():
                x_probe = (
                    torch.randn(1, img_out_channels, *img_shape, device=device) * d_std_sigma
                ).to(memory_format=torch.channels_last)
                x_lr_cond = image_lr
                if mean_hr is not None:
                    x_lr_cond = torch.cat([mean_hr, image_lr], dim=1).to(
                        memory_format=torch.channels_last
                    )
                out = net_res(
                    x_probe, x_lr_cond,
                    torch.tensor([d_std_sigma], device=device),
                    return_variance=True,
                )
                if isinstance(out, tuple):
                    _, d_std_raw = out
                    # Mean-normalize to align with calibrated_stochastic_sampler
                    d_std_mean = d_std_raw.mean()
                    if d_std_mean > 1e-8:
                        d_std_norm = d_std_raw / d_std_mean
                    else:
                        d_std_norm = torch.ones_like(d_std_raw)

        # Step 4: Update metrics for all strategies (no re-running diffusion)
        metrics.update(
            baseline_residuals=baseline_residuals,
            reg_mean=reg_mean,
            target=image_tar,
            d_std_norm=d_std_norm,
            denorm_fn=dataset.denormalize_output,
        )

    # Reduce metrics across GPUs
    if dist.world_size > 1:
        raw = metrics.get_raw_accumulators()
        if dist.rank == 0:
            gathered = [torch.zeros_like(raw) for _ in range(dist.world_size)]
        else:
            gathered = None
        torch.distributed.barrier()
        torch.distributed.gather(raw, gather_list=gathered, dst=0)

        if dist.rank == 0:
            # Reset and merge
            merged = MultiStrategyMetrics(strategies, device, precip_threshold)
            merged.merge_gathered(gathered)
            final = merged.compute()
        else:
            final = None
    else:
        final = metrics.compute()

    # Print and save results (rank 0 only)
    if dist.rank == 0 and final:
        metrics_output = getattr(cfg.generation, "metrics_output", "eval_metrics.json")

        logger.info("=" * 80)
        logger.info("POST-DIFFUSION RESCALING EVALUATION RESULTS")
        logger.info("=" * 80)
        header = f"{'Strategy':<25s} {'T':>5s} {'CRPS':>10s} {'RMSE':>10s} {'Spread':>10s} {'Skill':>10s} {'SS ratio':>10s} {'CRPS>1mm':>10s} {'CRPS 95th':>10s} {'CRPS W95th':>12s}"
        logger.info(header)
        logger.info("-" * 116)

        for name, T, use_d_std in strategies:
            if name not in final:
                continue
            m = final[name]
            logger.info(
                f"{name:<25s} {T:>5.1f} {m['crps']:>10.4f} {m['rmse']:>10.4f} "
                f"{m['spread']:>10.4f} {m['skill']:>10.4f} "
                f"{m['spread_skill_ratio']:>10.3f} "
                f"{m.get(f'crps_gt_{precip_threshold}mm', 0):>10.4f} "
                f"{m.get('crps_95th', 0):>10.4f} "
                f"{m.get('crps_w95th', 0):>12.4f}"
            )

        # Find best CRPS across all strategies
        best_name = min(final, key=lambda k: final[k]["crps"])
        best = final[best_name]
        logger.info("-" * 116)
        logger.info(f"Best CRPS: {best_name} (CRPS={best['crps']:.6f})")

        # Find best for extremes
        best_extreme = min(final, key=lambda k: final[k].get(f"crps_gt_{precip_threshold}mm", float("inf")))
        logger.info(f"Best CRPS (>{precip_threshold}mm): {best_extreme} "
                    f"(CRPS={final[best_extreme].get(f'crps_gt_{precip_threshold}mm', 0):.6f})")

        # Save JSON
        output = {
            "best_overall": best_name,
            "best_extreme": best_extreme,
            "precip_threshold": precip_threshold,
            "d_std_sigma": d_std_sigma,
            "d_std_normalization": "mean",
            "num_ensembles": num_ensembles,
            "n_time_steps": total_times,
            "strategies": final,
        }
        with open(metrics_output, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Results saved to: {metrics_output}")

    logger0.info("Evaluation complete.")


if __name__ == "__main__":
    main()
