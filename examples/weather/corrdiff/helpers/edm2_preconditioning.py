# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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

"""EDM2 preconditioning wrapper for super-resolution (CorrDiff style).

Wraps ``EDM2UNet`` with the EDM/EDM2 preconditioning formulas so that its
call signature matches the existing ``EDMPrecondSuperResolution`` used
throughout CorrDiff's training and generation code.
"""

from typing import List, Literal, Tuple, Union

import numpy as np
import torch

from helpers.edm2_networks import EDM2UNet, mp_cat


class EDM2PrecondSuperResolution(torch.nn.Module):
    """EDM2 preconditioning for super-resolution with spatial positional grid.

    Implements the same preconditioning as ``EDMPrecondSuperResolution``
    (Karras et al. 2022, Equations 7-9) but with the EDM2 MP-UNet backbone
    instead of ``SongUNetPosEmbd``.

    Spatial positional embeddings are generated here in the preconditioner
    (following the ``SongUNetPosEmbd`` pattern) and concatenated to the
    network input using ``mp_cat`` — the magnitude-preserving concatenation
    from EDM2 — so the combined tensor entering the network has controlled
    magnitude. The first encoder block then applies pixel-norm, which gives
    the final guarantee of unit magnitude at every spatial location.

    The forward signature is intentionally identical to
    ``EDMPrecondSuperResolution.forward`` so that existing loss functions
    (``ResidualLoss``, ``CalibratedResidualLoss``, etc.) and samplers work
    without modification.

    Parameters
    ----------
    img_resolution : int or list/tuple of int
        Spatial resolution of the image. A single int is used directly as the
        U-Net resolution ladder base. A list/tuple ``[H, W]`` uses
        ``min(H, W)`` for the U-Net ladder and ``(H, W)`` for the positional
        grid — this is the recommended form.
    img_in_channels : int
        Number of low-resolution conditioning channels.
    img_out_channels : int
        Number of high-resolution output channels.
    use_fp16 : bool, optional
        Run the model in FP16 on CUDA, by default False.
    sigma_data : float, optional
        Expected standard deviation of the training data, by default 0.5.
    sigma_min : float, optional
        Minimum noise level, by default 0.0.
    sigma_max : float, optional
        Maximum noise level, by default inf.
    N_grid_channels : int, optional
        Number of spatial positional embedding channels appended to the input,
        by default 4. Set to 0 to disable. For ``gridtype="sinusoidal"`` must
        be 4 or a multiple of 4.
    gridtype : {"sinusoidal", "linear"}, optional
        Type of spatial positional grid, by default "sinusoidal".
    grid_mp_balance : float, optional
        ``t`` parameter for ``mp_cat`` when appending the positional grid to
        the data channels. ``t=0`` keeps only data, ``t=1`` keeps only grid,
        ``t=0.5`` (default) gives equal per-channel weight to both groups.
    **unet_kwargs
        Additional keyword arguments forwarded to ``EDM2UNet`` (e.g.
        ``model_channels``, ``channel_mult``, ``num_blocks``,
        ``attn_resolutions``, ``dropout``, ``res_balance``, …).
    """

    def __init__(
        self,
        img_resolution: Union[int, List[int], Tuple[int, int]],
        img_in_channels: int,
        img_out_channels: int,
        use_fp16: bool = False,
        sigma_data: float = 0.5,
        sigma_min: float = 0.0,
        sigma_max: float = float("inf"),
        N_grid_channels: int = 4,
        gridtype: Literal["sinusoidal", "linear"] = "sinusoidal",
        grid_mp_balance: float = 0.5,
        **unet_kwargs,
    ):
        super().__init__()

        # Unpack resolution: [H, W] for grid generation, min(H,W) for U-Net.
        if isinstance(img_resolution, (list, tuple)):
            self._img_shape_y = int(img_resolution[0])
            self._img_shape_x = int(img_resolution[1])
            unet_resolution = min(img_resolution)
        else:
            self._img_shape_y = int(img_resolution)
            self._img_shape_x = int(img_resolution)
            unet_resolution = int(img_resolution)

        self.img_resolution = unet_resolution
        self.img_in_channels = img_in_channels
        self.img_out_channels = img_out_channels
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self._use_fp16 = use_fp16
        self.N_grid_channels = N_grid_channels
        self.grid_mp_balance = grid_mp_balance

        # Build positional grid (same logic as SongUNetPosEmbd._get_positional_embedding).
        if N_grid_channels > 0:
            grid = self._make_sinusoidal_grid(
                gridtype, N_grid_channels, self._img_shape_y, self._img_shape_x
            )
            self.register_buffer("pos_embd", grid.float(), persistent=False)
        else:
            self.pos_embd = None

        # The underlying network receives:
        #   [c_in * noisy_hr  ||  lr_cond  mp_cat  grid]
        # so its img_in_channels = img_out_channels + img_in_channels + N_grid_channels.
        self.model = EDM2UNet(
            img_resolution=unet_resolution,
            img_in_channels=img_out_channels + img_in_channels + N_grid_channels,
            img_out_channels=img_out_channels,
            **unet_kwargs,
        )

        if use_fp16:
            self.to(torch.float16)

    # ------------------------------------------------------------------
    # Positional grid construction (mirrors SongUNetPosEmbd logic)

    @staticmethod
    def _make_sinusoidal_grid(
        gridtype: str,
        N_grid_channels: int,
        img_shape_y: int,
        img_shape_x: int,
    ) -> torch.Tensor:
        """Return a fixed spatial positional grid of shape (N_grid_channels, H, W)."""
        if gridtype == "linear":
            if N_grid_channels != 2:
                raise ValueError("N_grid_channels must be 2 for gridtype='linear'")
            y = np.linspace(-1, 1, img_shape_y)
            x = np.linspace(-1, 1, img_shape_x)
            grid_x, grid_y = np.meshgrid(x, y)
            grid = torch.from_numpy(np.stack((grid_y, grid_x), axis=0))
            grid.requires_grad_(False)
            return grid

        # sinusoidal
        if N_grid_channels == 4:
            x1 = np.sin(np.linspace(0, 2 * np.pi, img_shape_x))
            x2 = np.cos(np.linspace(0, 2 * np.pi, img_shape_x))
            y1 = np.sin(np.linspace(0, 2 * np.pi, img_shape_y))
            y2 = np.cos(np.linspace(0, 2 * np.pi, img_shape_y))
            grid_x1, grid_y1 = np.meshgrid(x1, y1)
            grid_x2, grid_y2 = np.meshgrid(x2, y2)
            grid = torch.from_numpy(np.stack((grid_x1, grid_y1, grid_x2, grid_y2), axis=0))
        elif N_grid_channels % 4 == 0:
            num_freq = N_grid_channels // 4
            freq_bands = 2.0 ** np.linspace(0.0, num_freq, num=num_freq)
            gx, gy = np.meshgrid(
                np.linspace(0, 2 * np.pi, img_shape_x),
                np.linspace(0, 2 * np.pi, img_shape_y),
            )
            channels = []
            for freq in freq_bands:
                for fn in [np.sin, np.cos]:
                    channels.append(fn(gx * freq))
                    channels.append(fn(gy * freq))
            grid = torch.from_numpy(np.stack(channels, axis=0))
        else:
            raise ValueError(
                f"For gridtype='sinusoidal', N_grid_channels must be 4 or a multiple of 4, "
                f"got {N_grid_channels}."
            )
        grid.requires_grad_(False)
        return grid

    # ------------------------------------------------------------------

    @property
    def use_fp16(self) -> bool:
        return self._use_fp16

    @staticmethod
    def round_sigma(sigma):
        return torch.as_tensor(sigma)

    def forward(
        self,
        x: torch.Tensor,
        img_lr: torch.Tensor,
        sigma: torch.Tensor,
        force_fp32: bool = False,
        **model_kwargs,  # absorb SongUNet-specific kwargs (embedding_selector, etc.)
    ) -> torch.Tensor:
        """Denoise ``x`` conditioned on ``img_lr`` at noise level ``sigma``.

        Parameters
        ----------
        x : torch.Tensor
            Noisy high-resolution image, shape (B, C_hr, H, W).
        img_lr : torch.Tensor
            Low-resolution conditioning image, shape (B, C_lr, H, W).
        sigma : torch.Tensor
            Per-sample noise level, shape (B,) or broadcastable.
        force_fp32 : bool, optional
            Override ``use_fp16`` and run in FP32, by default False.
        **model_kwargs
            Ignored. Present for API compatibility with
            ``EDMPrecondSuperResolution`` (which accepts ``embedding_selector``,
            ``global_index``, ``augment_labels``, etc.).

        Returns
        -------
        torch.Tensor
            Denoised high-resolution image, shape (B, C_hr, H, W).
        """
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        dtype = (
            torch.float16
            if (self._use_fp16 and not force_fp32 and x.device.type == "cuda")
            else torch.float32
        )

        # EDM2 preconditioning — identical formulas to EDM (Eq. 7-9).
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.flatten().log() / 4

        # Data channels: [c_in * noisy_hr || lr_cond]  (B, C_hr + C_lr, H, W)
        data_in = torch.cat([c_in * x, img_lr.to(x.dtype)], dim=1).to(dtype)

        # Positional grid: magnitude-preserving concatenation via mp_cat.
        # mp_cat normalises by channel count so the combined tensor entering
        # the network has controlled magnitude. The first encoder Block then
        # applies pixel-norm (normalize(x, dim=1)) as the final guarantee.
        if self.pos_embd is not None:
            grid = self.pos_embd.to(dtype).unsqueeze(0).expand(x.shape[0], -1, -1, -1)
            x_in = mp_cat(data_in, grid, t=self.grid_mp_balance)
        else:
            x_in = data_in

        F_x = self.model(x_in, c_noise)
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x
