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

"""Unified CorrDiff generation script.

Parallelism mode (generation.parallel_mode):
  "time"     – each GPU handles different time steps (default). Supports
               metrics computation and HP eval.
  "ensemble" – all GPUs collaborate per time step; ensemble seeds split across
               ranks. Supports CUDA profiling.

Optional features via config flags:
  generation.compute_metrics   – compute RMSE/CRPS/spread-skill, save to JSON
  generation.hp_eval           – sweep HP_EVAL_CONFIGS, pick best CRPS (time mode only)
  generation.skip_netcdf       – skip NetCDF output (useful for metrics-only runs)
  generation.guidance.scale    – autoguidance with a second (weaker) checkpoint
  generation.edm2              – load raw .pt EDM2 state dicts

Usage:
    # Time-parallel (default)
    torchrun --nproc_per_node=4 generate.py --config-name=config_generate \
        generation.io.reg_ckpt_filename=/path/to/reg.mdlus \
        generation.io.res_ckpt_filename=/path/to/diff.mdlus

    # Ensemble-parallel
    torchrun --nproc_per_node=4 generate.py --config-name=config_generate \
        generation.parallel_mode=ensemble \
        generation.io.reg_ckpt_filename=/path/to/reg.mdlus \
        generation.io.res_ckpt_filename=/path/to/diff.mdlus

    # With metrics + HP eval
    torchrun --nproc_per_node=4 generate.py --config-name=config_generate \
        generation.hp_eval=true generation.metrics_output=metrics.json

    # Autoguidance
    python generate.py --config-name=generate_autoguidance_norwayW \
        generation.guidance.scale=1.5 \
        generation.guidance.guide_ckpt_filename=/path/to/guide.mdlus
"""

import contextlib
import datetime
import inspect
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import hydra
import netCDF4 as nc
import numpy as np
import nvtx
import torch
import torch._dynamo
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf, DictConfig
from torch.distributed import gather

from physicsnemo.distributed import DistributedManager
from physicsnemo.experimental.models.diffusion.preconditioning import tEDMPrecondSuperRes
from physicsnemo.launch.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.utils.corrdiff import (
    NetCDFWriter,
    regression_step,
    diffusion_step,
)

from datasets.dataset import register_dataset
from helpers.generate_helpers import (
    HP_EVAL_CONFIGS,
    HP_EVAL_CONFIGS_HETEROSCEDASTIC,
    _has_variance_head,
    build_sampler_fn,
    get_dataset_and_sampler,
    load_model,
    load_models,
    load_timestep_tensors,
    maybe_compile_models,
    resolve_times,
    save_images,
    setup_patching,
)
from helpers.dropout_residual import dropout_residual_step
from helpers.metrics import MetricsAccumulator


# NOTE: conf/generate_norwayW.yaml was removed in 526b124 and no generation config
# remains under conf/, so this default does not resolve and generate.py cannot be
# launched as-is (jobs/helma/generate.slurm has nothing valid to pass either).
# Left as-is deliberately; restore a config here if ensemble-parallel generation,
# autoguidance, or the uncertainty maps are needed again.
@hydra.main(version_base="1.2", config_path="conf", config_name="generate_norwayW")
def main(cfg: DictConfig) -> None:
    # ------------------------------------------------------------------ setup
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device

    logger = PythonLogger("generate")
    logger0 = RankZeroLoggingWrapper(logger, dist)
    logger.file_logging("generate.log")

    if dist.world_size > 1:
        torch.distributed.barrier()

    # ------------------------------------------------------------------ times
    times = resolve_times(cfg.generation)

    # ----------------------------------------------------------------- dataset
    register_dataset(cfg.dataset.type)
    logger0.info(f"Using dataset: {cfg.dataset.type}")
    has_lead_time = cfg.generation.get("has_lead_time", False)
    dataset, sampler = get_dataset_and_sampler(
        dataset_cfg=OmegaConf.to_container(cfg.dataset),
        times=times,
        has_lead_time=has_lead_time,
    )
    if len(sampler) == 0:
        raise ValueError("No matching timesteps. Check times / dataset.years overlap.")
    total_times = len(sampler)
    img_shape = dataset.image_shape()
    img_out_channels = len(dataset.output_channels())
    in_channels = len(dataset.input_channels())
    logger0.info(f"Found {total_times} matching timesteps")

    # ---------------------------------------------------------------- patching
    patching, img_shape = setup_patching(cfg, img_shape)
    logger0.info(f"Patching: {'enabled' if patching else 'disabled'}")

    # -------------------------------------------------------- inference mode
    if cfg.generation.inference_mode == "regression":
        load_net_reg, load_net_res = True, False
    elif cfg.generation.inference_mode == "diffusion":
        load_net_reg, load_net_res = False, True
    elif cfg.generation.inference_mode == "all":
        load_net_reg, load_net_res = True, True
    else:
        raise ValueError(f"Invalid inference_mode: {cfg.generation.inference_mode}")

    # ---------------------------------------------------------- EDM2 kwargs
    edm2_kwargs = None
    if cfg.generation.get("edm2") is not None:
        img_in_channels = in_channels
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

    # ----------------------------------------------------------- load models
    logger0.info("Loading models...")
    net_reg, net_res = load_models(cfg, device, load_net_reg, load_net_res, edm2_kwargs=edm2_kwargs)

    # Optional guidance model (autoguidance)
    guidance_cfg = cfg.generation.get("guidance", None)
    guidance_scale = float(guidance_cfg.get("scale", 0.0)) if guidance_cfg else 0.0
    guide_ckpt = guidance_cfg.get("guide_ckpt_filename", None) if guidance_cfg else None
    if guidance_scale != 0.0 and guide_ckpt:
        logger0.info(f"Loading guidance model (scale={guidance_scale}): {guide_ckpt}")
        net_guide = load_model(to_absolute_path(guide_ckpt), device, cfg.generation.perf, edm2_kwargs)
    else:
        net_guide = None
        if guidance_scale != 0.0:
            logger0.warning("guidance.scale != 0 but no guide_ckpt_filename — running without guidance")

    # ---------------------------------------------------------- compile
    net_reg, net_res = maybe_compile_models(cfg, net_reg, net_res)

    # ----------------------------------------------------------- sampler
    residual_model_type = cfg.generation.get("residual_model_type", "diffusion")
    use_dropout_residual = residual_model_type in {"dropout_crps", "dropout_residual"}
    if use_dropout_residual and cfg.generation.inference_mode != "all":
        raise ValueError("dropout residual inference requires generation.inference_mode=all")
    if use_dropout_residual and cfg.generation.get("hp_eval", False):
        raise ValueError("generation.hp_eval is only supported for diffusion residual models")

    sampler_fn = build_sampler_fn(
        cfg.sampler, patching, net_guide=net_guide, guidance_scale=guidance_scale
    )

    # -------------------------------------------- distribution / student-t
    distribution = cfg.generation.get("distribution", None)
    student_t_nu = cfg.generation.get("student_t_nu", None)
    P_mean = cfg.generation.get("P_mean", None)
    P_std = cfg.generation.get("P_std", None)

    if distribution is not None and cfg.generation.inference_mode not in ["diffusion", "all"]:
        raise ValueError("distribution only valid for inference_mode 'diffusion'/'all'")
    if distribution not in ["normal", "student_t", None]:
        raise ValueError(f"Invalid distribution: {distribution}")
    if distribution == "student_t":
        if student_t_nu is None:
            raise ValueError("student_t_nu required for student_t distribution")
        if student_t_nu <= 2:
            raise ValueError(f"Expected nu > 2, got {student_t_nu}")
        if net_res and not isinstance(net_res, tEDMPrecondSuperRes):
            logger0.warning(f"Student-t with non-tEDM model ({type(net_res).__name__})")
    elif isinstance(net_res, tEDMPrecondSuperRes):
        logger0.warning(f"tEDMPrecondSuperRes should use student_t distribution, got {distribution}")

    diffusion_step_kwargs = {}
    if distribution is not None:
        diffusion_step_kwargs["distribution"] = distribution
    if student_t_nu is not None:
        diffusion_step_kwargs["nu"] = student_t_nu
    if P_mean is not None:
        diffusion_step_kwargs["P_mean"] = P_mean
    if P_std is not None:
        diffusion_step_kwargs["P_std"] = P_std

    # ----------------------------------------- heteroscedastic / uncertainty
    save_predicted_uncertainty = cfg.generation.get("save_predicted_uncertainty", False)
    uncertainty_output_name = cfg.generation.get("uncertainty_output_name", "predicted_std")
    uncertainty_sigma_probe = cfg.generation.get("uncertainty_sigma_probe", 0.01)
    heteroscedastic_mode_cfg = cfg.generation.get("use_heteroscedastic_model", "auto")
    detected_heteroscedastic = _has_variance_head(net_res)

    if isinstance(heteroscedastic_mode_cfg, str):
        hm = heteroscedastic_mode_cfg.lower()
        if hm not in {"auto", "true", "false"}:
            raise ValueError("use_heteroscedastic_model must be one of: auto, true, false")
        use_heteroscedastic_model = detected_heteroscedastic if hm == "auto" else hm == "true"
    else:
        use_heteroscedastic_model = bool(heteroscedastic_mode_cfg)

    if load_net_res:
        logger0.info(f"Heteroscedastic residual model: {detected_heteroscedastic}")
    if use_heteroscedastic_model and not detected_heteroscedastic:
        logger0.warning("use_heteroscedastic_model=true but residual model has no variance head")

    skip_netcdf = cfg.generation.get("skip_netcdf", False)
    can_save_predicted_uncertainty = (
        save_predicted_uncertainty
        and not skip_netcdf
        and load_net_res
        and use_heteroscedastic_model
        and detected_heteroscedastic
        and cfg.generation.inference_mode in ["diffusion", "all"]
    )
    if save_predicted_uncertainty and not can_save_predicted_uncertainty:
        logger0.warning("save_predicted_uncertainty=true but uncertainty disabled for this run")

    # --------------------------------------------------------- metrics config
    compute_metrics = cfg.generation.get("compute_metrics", False)
    metrics_output = cfg.generation.get("metrics_output", "metrics.json")
    precip_threshold = cfg.generation.get("precip_threshold", 1.0)

    # ----------------------------------------------------------------- times list
    all_times_ds = dataset.time()
    times_list = [all_times_ds[i] for i in sampler] if sampler else all_times_ds

    # ----------------------------------------- seed batch setup (time-parallel + HP eval)
    num_ensembles = cfg.generation.num_ensembles
    seed_batch_size = cfg.generation.seed_batch_size
    seed_batches = np.array_split(np.arange(num_ensembles), max(1, (num_ensembles - 1) // seed_batch_size + 1))

    def generate_for_time(image_lr, lead_time_label, fn=None):
        """Generate all ensemble members for one time step on this GPU."""
        fn = fn or sampler_fn
        img_lr = image_lr.to(memory_format=torch.channels_last)
        all_outputs = []
        for seed_batch in seed_batches:
            bs = len(seed_batch)
            if net_reg:
                image_reg = regression_step(
                    net=net_reg,
                    img_lr=img_lr,
                    latents_shape=(bs, img_out_channels, img_shape[0], img_shape[1]),
                    lead_time_label=lead_time_label,
                )
            if net_res:
                mean_hr = (
                    image_reg[0:1]
                    if (cfg.generation.hr_mean_conditioning and net_reg)
                    else None
                )
                if use_dropout_residual:
                    image_res = dropout_residual_step(
                        net=net_res,
                        img_lr=img_lr,
                        latents_shape=(bs, img_out_channels, img_shape[0], img_shape[1]),
                        mean_hr=mean_hr,
                        lead_time_label=lead_time_label,
                        seed=int(seed_batch[0]) if len(seed_batch) else None,
                    )
                else:
                    image_res = diffusion_step(
                        net=net_res,
                        sampler_fn=fn,
                        img_shape=img_shape,
                        img_out_channels=img_out_channels,
                        rank_batches=[torch.tensor(seed_batch)],
                        img_lr=img_lr.expand(bs, -1, -1, -1).to(memory_format=torch.channels_last),
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

    # ---------------------------------------------------------------- HP eval
    hp_eval = cfg.generation.get("hp_eval", False)
    if hp_eval:
        logger0.info("=== HP Eval Mode: Testing multiple sampler configs ===")
        is_heteroscedastic = _has_variance_head(net_res) if net_res else False
        eval_configs = HP_EVAL_CONFIGS_HETEROSCEDASTIC if is_heteroscedastic else HP_EVAL_CONFIGS
        if is_heteroscedastic:
            logger0.info("  Detected heteroscedastic model — using uncertainty-aware samplers")
        max_per_gpu = (total_times + dist.world_size - 1) // dist.world_size
        all_config_results = {}

        for eval_cfg in eval_configs:
            config_name = eval_cfg["name"]
            logger0.info(f"  Evaluating config: {config_name}")
            config_sampler_fn = build_sampler_fn(eval_cfg["sampler"], patching)
            config_metrics = MetricsAccumulator(precip_threshold=precip_threshold, device=device)

            for iteration in range(max_per_gpu):
                time_idx = dist.rank + iteration * dist.world_size
                if time_idx >= total_times:
                    continue
                dataset_idx = sampler[time_idx]
                image_tar, image_lr, lead_time_label = load_timestep_tensors(
                    dataset, dataset_idx, device
                )

                image_out = generate_for_time(image_lr, lead_time_label, config_sampler_fn)
                pred_t = torch.from_numpy(dataset.denormalize_output(image_out.cpu().numpy())).to(device)
                tar_t = torch.from_numpy(dataset.denormalize_output(image_tar.cpu().numpy())).to(device)
                config_metrics.update(pred_t, tar_t)

            config_metrics.reduce()
            if dist.rank == 0:
                result = config_metrics.to_dict()
                result["config_name"] = config_name
                all_config_results[config_name] = result
                logger0.info(f"    {config_name}: CRPS={result['crps']:.6f}, RMSE={result['rmse']:.6f}")

            if dist.world_size > 1:
                torch.distributed.barrier()

        if dist.rank == 0:
            best_config = min(all_config_results, key=lambda k: all_config_results[k]["crps"])
            best_crps = all_config_results[best_config]["crps"]
            logger0.info(f"=== Best config: {best_config} (CRPS={best_crps:.6f}) ===")
            output = {
                "success": True,
                "best_config": best_config,
                **all_config_results[best_config],
                "all_configs": all_config_results,
            }
            with open(metrics_output, "w") as mf:
                json.dump(output, mf, indent=2)
            logger0.info(f"HP eval metrics saved to: {metrics_output}")
        logger0.info("HP Eval complete.")
        return

    # -------------------------------------------------------- NetCDF setup
    output_path = cfg.generation.io.get("output_filename", "corrdiff_output.nc")
    f = None
    writer = None
    writer_executor = None
    writer_threads = []

    if not skip_netcdf and dist.rank == 0:
        f = nc.Dataset(output_path, "w")
        f.cfg = str(cfg)
        f.generation_timestamp = datetime.datetime.now().isoformat()
        f.inference_mode = cfg.generation.inference_mode
        if load_net_reg:
            f.regression_checkpoint = str(getattr(cfg.generation.io, "reg_ckpt_filename", "unknown"))
        if load_net_res:
            f.residual_checkpoint = str(getattr(cfg.generation.io, "res_ckpt_filename", "unknown"))

        writer_kwargs = {
            "lat": dataset.latitude(),
            "lon": dataset.longitude(),
            "input_channels": dataset.input_channels(),
            "output_channels": dataset.output_channels(),
            "has_lead_time": has_lead_time,
        }
        writer_sig = inspect.signature(NetCDFWriter.__init__).parameters
        if "save_uncertainty" in writer_sig:
            writer_kwargs["save_uncertainty"] = can_save_predicted_uncertainty
        if "uncertainty_group_name" in writer_sig:
            writer_kwargs["uncertainty_group_name"] = uncertainty_output_name
        writer = NetCDFWriter(f, **writer_kwargs)

        if cfg.generation.perf.io_syncronous:
            writer_executor = ThreadPoolExecutor(max_workers=cfg.generation.perf.num_writer_workers)

    # ------------------------------------------ uncertainty map prediction
    def predict_uncertainty_map(image_lr, mean_hr=None, lead_time_label=None):
        if not can_save_predicted_uncertainty or net_res is None:
            return None
        x_lr = image_lr.to(memory_format=torch.channels_last)
        if mean_hr is not None:
            x_lr = torch.cat((mean_hr.expand(x_lr.shape[0], -1, -1, -1), x_lr), dim=1)
        if patching:
            x_lr_net = patching.apply(input=x_lr, additional_input=image_lr)
            patch_embedding_selector = lambda emb: patching.apply(emb.expand(image_lr.shape[0], -1, -1, -1))
        else:
            x_lr_net = x_lr
            patch_embedding_selector = None
        optional_args = {}
        if lead_time_label is not None:
            optional_args["lead_time_label"] = lead_time_label
        x_probe = torch.randn(
            1, img_out_channels, img_shape[0], img_shape[1], device=device, dtype=torch.float32
        ).to(memory_format=torch.channels_last)
        x_probe_net = patching.apply(input=x_probe) if patching else x_probe
        sigma_probe = torch.tensor([uncertainty_sigma_probe], device=device)
        with torch.inference_mode():
            net_output = net_res(
                x_probe_net, x_lr_net, sigma_probe,
                embedding_selector=patch_embedding_selector,
                return_variance=True, **optional_args,
            )
        if not isinstance(net_output, tuple):
            logger0.warning("Residual model did not return variance tensor. Writing zeros.")
            return np.zeros((img_out_channels, img_shape[0], img_shape[1]), dtype=np.float32)
        _, d_std = net_output
        if patching:
            d_std = patching.fuse(input=d_std, batch_size=1)
        return d_std[0].detach().cpu().numpy()

    # ================================================================ main loop
    parallel_mode = cfg.generation.get("parallel_mode", "time")
    logger0.info(f"=== Generation: parallel_mode={parallel_mode}, {total_times} time steps ===")

    if parallel_mode == "time":
        # ---- Time-parallel: each GPU handles different time steps ----
        metrics = MetricsAccumulator(precip_threshold, device) if compute_metrics else None
        max_per_gpu = (total_times + dist.world_size - 1) // dist.world_size

        for iteration in range(max_per_gpu):
            time_idx = dist.rank + iteration * dist.world_size
            has_work = time_idx < total_times

            if has_work:
                dataset_idx = sampler[time_idx]
                image_tar, image_lr, lead_time_label = load_timestep_tensors(
                    dataset, dataset_idx, device
                )

                logger.info(f"[GPU {dist.rank}] time_idx={time_idx}, time={times_list[time_idx]}")
                image_out = generate_for_time(image_lr, lead_time_label)

                predicted_uncertainty = None
                if can_save_predicted_uncertainty:
                    mean_hr = None
                    if cfg.generation.hr_mean_conditioning and net_reg:
                        mean_hr = regression_step(
                            net=net_reg,
                            img_lr=image_lr,
                            latents_shape=(1, img_out_channels, img_shape[0], img_shape[1]),
                            lead_time_label=lead_time_label,
                        )[0:1]
                    predicted_uncertainty = predict_uncertainty_map(image_lr, mean_hr, lead_time_label)

                if metrics is not None:
                    pred_t = torch.from_numpy(dataset.denormalize_output(image_out.cpu().numpy())).to(device)
                    tar_t = torch.from_numpy(dataset.denormalize_output(image_tar.cpu().numpy())).to(device)
                    metrics.update(pred_t, tar_t)
            else:
                # Padding iteration — dummy tensors for gather sync
                time_idx = -1
                image_out = torch.zeros(num_ensembles, img_out_channels, *img_shape, device=device)
                image_tar = torch.zeros(1, img_out_channels, *img_shape, device=device)
                image_lr = torch.zeros(1, in_channels, *img_shape, device=device)
                predicted_uncertainty = np.zeros((img_out_channels, *img_shape), dtype=np.float32) if can_save_predicted_uncertainty else None

            # Gather + write
            if dist.world_size > 1:
                time_idx_t = torch.tensor([time_idx], device=device, dtype=torch.long)
                gathered_time_indices = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(dist.world_size)] if dist.rank == 0 else None
                gathered_outputs = [torch.zeros_like(image_out) for _ in range(dist.world_size)] if dist.rank == 0 else None
                gathered_targets = [torch.zeros_like(image_tar) for _ in range(dist.world_size)] if dist.rank == 0 else None
                gathered_inputs = [torch.zeros_like(image_lr) for _ in range(dist.world_size)] if dist.rank == 0 else None
                if can_save_predicted_uncertainty:
                    pu_tensor = torch.from_numpy(np.array(predicted_uncertainty)).to(device) if predicted_uncertainty is not None else torch.zeros(img_out_channels, *img_shape, device=device)
                    gathered_uncertainties = [torch.zeros(img_out_channels, *img_shape, device=device) for _ in range(dist.world_size)] if dist.rank == 0 else None
                else:
                    pu_tensor = None
                    gathered_uncertainties = None

                torch.distributed.barrier()
                torch.distributed.gather(time_idx_t, gather_list=gathered_time_indices, dst=0)
                torch.distributed.gather(image_out, gather_list=gathered_outputs, dst=0)
                torch.distributed.gather(image_tar, gather_list=gathered_targets, dst=0)
                torch.distributed.gather(image_lr, gather_list=gathered_inputs, dst=0)
                if can_save_predicted_uncertainty:
                    torch.distributed.gather(pu_tensor, gather_list=gathered_uncertainties, dst=0)

                if dist.rank == 0 and writer is not None:
                    for gpu_idx in range(dist.world_size):
                        t_idx = gathered_time_indices[gpu_idx].item()
                        if t_idx < 0:
                            continue
                        pu = gathered_uncertainties[gpu_idx].cpu().numpy() if can_save_predicted_uncertainty else None
                        args = (writer, dataset, list(times_list), gathered_outputs[gpu_idx].cpu(), gathered_targets[gpu_idx].cpu(), gathered_inputs[gpu_idx].cpu(), t_idx, t_idx, has_lead_time, pu)
                        if writer_executor:
                            writer_threads.append(writer_executor.submit(save_images, *args))
                        else:
                            save_images(*args)
            else:
                if time_idx >= 0 and writer is not None:
                    pu = predicted_uncertainty if isinstance(predicted_uncertainty, np.ndarray) else None
                    args = (writer, dataset, list(times_list), image_out.cpu(), image_tar.cpu(), image_lr.cpu(), time_idx, time_idx, has_lead_time, pu)
                    if writer_executor:
                        writer_threads.append(writer_executor.submit(save_images, *args))
                    else:
                        save_images(*args)

        if metrics is not None:
            metrics.reduce()
            if dist.rank == 0:
                final_metrics = metrics.to_dict()
                logger.info(f"RMSE: {final_metrics['rmse']:.4f}, CRPS: {final_metrics['crps']:.4f}, "
                            f"Spread/Skill: {final_metrics['spread_skill_ratio']:.4f}")
                with open(metrics_output, "w") as mf:
                    json.dump(final_metrics, mf, indent=2)
                logger.info(f"Metrics saved to {metrics_output}")

    else:
        # ---- Ensemble-parallel: all GPUs collaborate per time step ----
        seeds = list(np.arange(num_ensembles))
        num_batches = ((len(seeds) - 1) // (seed_batch_size * dist.world_size) + 1) * dist.world_size
        all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
        rank_batches = all_batches[dist.rank :: dist.world_size]

        warmup_steps = min(total_times - 1, 2)
        use_cuda_timing = torch.cuda.is_available()
        if use_cuda_timing:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
        else:
            class _DummyEvent:
                def record(self): pass
                def synchronize(self): pass
                def elapsed_time(self, _): return 0
            start = end = _DummyEvent()

        def generate_ensemble_fn():
            with nvtx.annotate("generate_fn", color="green"):
                img_lr = image_lr.to(memory_format=torch.channels_last)
                if net_reg:
                    with nvtx.annotate("regression_model", color="yellow"):
                        image_reg = regression_step(
                            net=net_reg,
                            img_lr=img_lr,
                            latents_shape=(sum(map(len, rank_batches)), img_out_channels, img_shape[0], img_shape[1]),
                            lead_time_label=lead_time_label,
                        )
                if net_res:
                    mean_hr = image_reg[0:1] if (cfg.generation.hr_mean_conditioning and net_reg) else None
                    with nvtx.annotate("residual_model", color="purple"):
                        if use_dropout_residual:
                            local_ensemble_size = sum(map(len, rank_batches))
                            image_res = dropout_residual_step(
                                net=net_res,
                                img_lr=img_lr,
                                latents_shape=(local_ensemble_size, img_out_channels, img_shape[0], img_shape[1]),
                                mean_hr=mean_hr,
                                lead_time_label=lead_time_label,
                                seed=int(rank_batches[0][0]) if rank_batches and len(rank_batches[0]) else None,
                            )
                        else:
                            image_res = diffusion_step(
                                net=net_res,
                                sampler_fn=sampler_fn,
                                img_shape=img_shape,
                                img_out_channels=img_out_channels,
                                rank_batches=rank_batches,
                                img_lr=img_lr.expand(seed_batch_size, -1, -1, -1).to(memory_format=torch.channels_last),
                                rank=dist.rank,
                                device=device,
                                mean_hr=mean_hr,
                                lead_time_label=lead_time_label,
                                **diffusion_step_kwargs,
                            )
                if cfg.generation.inference_mode == "regression":
                    out = image_reg
                elif cfg.generation.inference_mode == "diffusion":
                    out = image_res
                else:
                    out = image_reg + image_res

                if dist.world_size > 1:
                    gathered = [torch.zeros_like(out) for _ in range(dist.world_size)] if dist.rank == 0 else None
                    torch.distributed.barrier()
                    gather(out, gather_list=gathered if dist.rank == 0 else None, dst=0)
                    return torch.cat(gathered) if dist.rank == 0 else None
                return out

        torch_cuda_profiler = torch.cuda.profiler.profile() if torch.cuda.is_available() else contextlib.nullcontext()
        torch_nvtx_profiler = torch.autograd.profiler.emit_nvtx() if torch.cuda.is_available() else contextlib.nullcontext()

        data_loader = torch.utils.data.DataLoader(dataset=dataset, sampler=sampler, batch_size=1, pin_memory=True)
        batch_size = 1
        time_index = -1

        with torch_cuda_profiler:
            with torch_nvtx_profiler:
                for index, (image_tar, image_lr, *lead_time_label) in enumerate(data_loader):
                    time_index += 1
                    logger0.info(f"Starting index: {time_index}")
                    if time_index == warmup_steps:
                        start.record()

                    lead_time_label = lead_time_label[0].to(dist.device).contiguous() if lead_time_label else None
                    image_lr = image_lr.to(device=device).to(torch.float32).to(memory_format=torch.channels_last)
                    image_tar = image_tar.to(device=device).to(torch.float32)
                    image_out = generate_ensemble_fn()

                    if dist.rank == 0:
                        predicted_uncertainty = None
                        if can_save_predicted_uncertainty:
                            mean_hr = None
                            if cfg.generation.hr_mean_conditioning and net_reg:
                                mean_hr = regression_step(
                                    net=net_reg,
                                    img_lr=image_lr,
                                    latents_shape=(1, img_out_channels, img_shape[0], img_shape[1]),
                                    lead_time_label=lead_time_label,
                                )[0:1]
                            predicted_uncertainty = predict_uncertainty_map(image_lr, mean_hr, lead_time_label)

                        batch_size = image_out.shape[0]
                        args = (writer, dataset, list(times_list), image_out.cpu(), image_tar.cpu(), image_lr.cpu(), time_index, index, has_lead_time, predicted_uncertainty)
                        if writer_executor:
                            writer_threads.append(writer_executor.submit(save_images, *args))
                        else:
                            save_images(*args)

        end.record()
        end.synchronize()
        elapsed = start.elapsed_time(end) / 1000.0 if use_cuda_timing else 0
        timed_steps = time_index + 1 - warmup_steps
        if dist.rank == 0 and use_cuda_timing and timed_steps > 0:
            logger.info(f"Total time: {elapsed:.2f}s for {timed_steps} steps, {batch_size} members")
            logger.info(f"Avg per member: {elapsed / timed_steps / batch_size:.4f}s")

    # ------------------------------------------------------------ cleanup
    if dist.rank == 0:
        if writer_threads and writer_executor:
            for thread in writer_threads:
                thread.result()
            writer_executor.shutdown()
        if f is not None:
            f.close()
    logger0.info("Generation complete.")


if __name__ == "__main__":
    main()
