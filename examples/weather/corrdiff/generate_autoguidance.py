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

"""Minimal generation script — base for autoguidance experiments.

Stripped-down version of generate.py — no patching, no heteroscedastic
variance head, no student-t, no async I/O, no torch.compile, single-process
or DDP. Works with any EDMPrecond-compatible model (EDM1 or EDM2).
Autoguidance support (net_guide + guidance_scale) to be added here.
"""

from functools import partial

import hydra
import netCDF4 as nc
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from physicsnemo import Module
from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.utils.corrdiff import (
    NetCDFWriter,
    get_time_from_range,
    regression_step,
    diffusion_step,
)

from datasets.dataset import init_dataset_from_config, register_dataset
from helpers.generate_helpers import get_dataset_and_sampler, save_images
from helpers.stochastic_sampler import stochastic_sampler
from helpers.train_helpers import set_patch_shape


@hydra.main(version_base="1.2", config_path="conf", config_name="generate_autoguidance_norwayW")
def main(cfg: DictConfig) -> None:
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device

    logger = PythonLogger("generate_autoguidance")
    logger0 = RankZeroLoggingWrapper(logger, dist)

    # ------------------------------------------------------------------ seeds
    seeds = list(np.arange(cfg.generation.num_ensembles))
    num_batches = (
        (len(seeds) - 1) // (cfg.generation.seed_batch_size * dist.world_size) + 1
    ) * dist.world_size
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.rank :: dist.world_size]

    if dist.world_size > 1:
        torch.distributed.barrier()

    # --------------------------------------------------------------- time axis
    if hasattr(cfg.generation, "times_range") and cfg.generation.times_range is not None:
        times = get_time_from_range(cfg.generation.times_range)
    elif hasattr(cfg.generation, "times") and cfg.generation.times is not None:
        times = cfg.generation.times
    else:
        raise ValueError("Provide either generation.times or generation.times_range")

    # -------------------------------------------------------------- dataset
    register_dataset(cfg.dataset.type)
    dataset_cfg = OmegaConf.to_container(cfg.dataset)
    dataset, sampler = get_dataset_and_sampler(dataset_cfg=dataset_cfg, times=times)

    if len(sampler) == 0:
        raise ValueError(
            "No matching timesteps found in the dataset. "
            "Check that generation.times_range / times overlaps with dataset.years."
        )
    logger0.info(f"Found {len(sampler)} matching timesteps")

    img_shape = dataset.image_shape()
    img_out_channels = len(dataset.output_channels())

    # Patch shape (no patching for EDM2 by default; kept for completeness)
    use_patching, img_shape, _ = set_patch_shape(img_shape, (None, None))
    if use_patching:
        raise RuntimeError("Patching is not supported in generate_edm2.py")

    # ----------------------------------------------------------- load models
    # Regression model — needed to supply mean_hr when hr_mean_conditioning=True
    reg_ckpt = to_absolute_path(cfg.generation.io.reg_ckpt_filename)
    logger0.info(f"Loading regression model from {reg_ckpt}")
    net_reg = Module.from_checkpoint(reg_ckpt)
    net_reg.eval().requires_grad_(False).to(device)
    if hasattr(net_reg, "amp_mode"):
        net_reg.amp_mode = False

    # Main diffusion model (strong / later checkpoint)
    res_ckpt = to_absolute_path(cfg.generation.io.res_ckpt_filename)
    logger0.info(f"Loading diffusion model from {res_ckpt}")
    net_res = Module.from_checkpoint(res_ckpt)
    net_res.eval().requires_grad_(False).to(device)
    if hasattr(net_res, "amp_mode"):
        net_res.amp_mode = False
    if hasattr(net_res, "use_fp16"):
        net_res.use_fp16 = cfg.generation.perf.get("use_fp16", False)

    # Guidance model (weak / earlier checkpoint) — optional
    guidance_scale = float(cfg.generation.guidance.get("scale", 0.0))
    guide_ckpt = cfg.generation.guidance.get("guide_ckpt_filename", None)
    if guide_ckpt and guidance_scale != 0.0:
        guide_ckpt = to_absolute_path(guide_ckpt)
        logger0.info(f"Loading guidance model from {guide_ckpt} (scale={guidance_scale})")
        net_guide = Module.from_checkpoint(guide_ckpt)
        net_guide.eval().requires_grad_(False).to(device)
        if hasattr(net_guide, "amp_mode"):
            net_guide.amp_mode = False
        if hasattr(net_guide, "use_fp16"):
            net_guide.use_fp16 = cfg.generation.perf.get("use_fp16", False)
    else:
        net_guide = None
        if guidance_scale != 0.0:
            logger0.warning("guidance_scale != 0 but no guide_ckpt_filename set — running without guidance")
        else:
            logger0.info("Autoguidance disabled (scale=0)")

    # ----------------------------------------------------------- sampler
    sampler_fn = partial(
        stochastic_sampler,
        patching=None,
        num_steps=cfg.sampler.num_steps,
        S_churn=cfg.sampler.get("S_churn", 0),
        S_min=cfg.sampler.get("S_min", 0),
        S_max=cfg.sampler.get("S_max", float("inf")),
        net_guide=net_guide,
        guidance_scale=guidance_scale,
    )

    # --------------------------------------------------------- output file
    output_path = cfg.generation.io.get("output_filename", "edm2_output.nc")
    logger0.info(f"Saving output to {output_path}")
    if dist.rank == 0:
        f = nc.Dataset(output_path, "w")
        f.cfg = str(cfg)
        writer = NetCDFWriter(
            f,
            lat=dataset.latitude(),
            lon=dataset.longitude(),
            input_channels=dataset.input_channels(),
            output_channels=dataset.output_channels(),
            has_lead_time=False,
        )

    # -------------------------------------------------------- generation loop
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset, sampler=sampler, batch_size=1, pin_memory=True
    )
    all_times = dataset.time()
    times_list = [all_times[i] for i in sampler] if sampler else all_times

    for time_index, (image_tar, image_lr, *_) in enumerate(data_loader):
        logger0.info(f"Processing timestep {time_index}")

        image_lr = image_lr.to(device=device, dtype=torch.float32)
        image_tar = image_tar.to(device=device, dtype=torch.float32)

        # Regression step to get mean_hr for conditioning
        with torch.inference_mode():
            image_reg = regression_step(
                net=net_reg,
                img_lr=image_lr,
                latents_shape=(
                    sum(map(len, rank_batches)),
                    img_out_channels,
                    img_shape[0],
                    img_shape[1],
                ),
            )
        mean_hr = image_reg[0:1]  # (1, C_out, H, W)

        # Diffusion step
        with torch.inference_mode():
            image_res = diffusion_step(
                net=net_res,
                sampler_fn=sampler_fn,
                img_shape=img_shape,
                img_out_channels=img_out_channels,
                rank_batches=rank_batches,
                img_lr=image_lr.expand(cfg.generation.seed_batch_size, -1, -1, -1),
                rank=dist.rank,
                device=device,
                mean_hr=mean_hr,
            )

        # Gather across ranks if DDP
        if dist.world_size > 1:
            from torch.distributed import gather
            if dist.rank == 0:
                gathered = [torch.zeros_like(image_res) for _ in range(dist.world_size)]
            else:
                gathered = None
            torch.distributed.barrier()
            gather(image_res, gather_list=gathered if dist.rank == 0 else None, dst=0)
            if dist.rank == 0:
                image_out = torch.cat(gathered)
            else:
                image_out = None
        else:
            image_out = image_res

        if dist.rank == 0:
            save_images(
                writer=writer,
                dataset=dataset,
                times=list(times_list),
                image_out=image_out.cpu(),
                image_tar=image_tar.cpu(),
                image_lr=image_lr.cpu(),
                time_index=time_index,
                t_index=time_index,
                has_lead_time=False,
            )

    if dist.rank == 0:
        f.close()
    logger0.info("Generation complete.")


if __name__ == "__main__":
    main()
