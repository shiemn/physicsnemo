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

"""DiT-based preconditioner wrappers for CorrDiff.

Drop-in replacements for ``EDM2PrecondSuperResolution`` (diffusion) and the
``UNet`` wrapper (regression), using ``DiTSuperRes`` as the backbone.

The noise-level preconditioning formulas (EDM Equations 7-9) are identical to
those in ``EDM2PrecondSuperResolution``.  Standard ``torch.cat`` is used
instead of ``mp_cat`` because DiT uses LayerNorm / adaLN which are invariant
to input scale — the magnitude-preserving concatenation needed by the
pixel-norm in EDM2's UNet is not required here.
"""

from typing import List, Literal, Optional, Tuple, Union

import torch

from physicsnemo import Module
from physicsnemo.models.meta import ModelMetaData

from helpers.dit_networks import DiTSuperRes, DiTSuperResV2
from helpers.edm2_preconditioning import EDM2PrecondSuperResolution


class DiTPrecondSuperResolution(Module):
    """EDM2 preconditioning with a DiT backbone for super-resolution diffusion.

    Identical forward signature to ``EDM2PrecondSuperResolution`` so that all
    existing loss functions (``ResidualLoss``, ``CalibratedResidualLoss``, …)
    and samplers work without modification.

    Parameters
    ----------
    img_resolution : int or [H, W]
        High-resolution spatial size.
    img_in_channels : int
        Number of low-resolution conditioning channels.
    img_out_channels : int
        Number of high-resolution output channels.
    use_fp16 : bool
        Run the forward pass in FP16 on CUDA (default False).
    sigma_data : float
        Expected standard deviation of training data (default 0.5).
    sigma_min : float
        Minimum noise level (default 0.0).
    sigma_max : float or None
        Maximum noise level (default None = inf).
    N_grid_channels : int
        Positional-grid channels appended to the input (default 4).
    gridtype : str
        ``"sinusoidal"`` (default) or ``"linear"``.
    patch_size : int or (int, int)
        DiT patch size (default 8).
    hidden_size : int
        Transformer embedding dimension (default 768).
    depth : int
        Number of DiT blocks (default 12).
    num_heads : int
        Number of attention heads (default 12).
    mlp_ratio : float
        MLP hidden-dim ratio (default 4.0).
    attention_backend : str
        ``"transformer_engine"`` (default), ``"timm"``, or ``"natten2d"``.
    layernorm_backend : str
        ``"torch"`` (default) or ``"apex"``.
    **ignored_kwargs
        Absorbs UNet-specific keys injected by ``train.py`` (e.g.
        ``checkpoint_level``, ``use_apex_gn``, ``profile_mode``,
        ``amp_mode``, ``grid_mp_balance``).
    """

    # Allow evaluate.py / train.py to pass UNet-specific keys via
    # override_args without raising ValueError.
    _overridable_args = {"use_apex_gn", "checkpoint_level", "profile_mode", "amp_mode"}

    def __init__(
        self,
        img_resolution: Union[int, List[int], Tuple[int, int]],
        img_in_channels: int,
        img_out_channels: int,
        use_fp16: bool = False,
        sigma_data: float = 0.5,
        sigma_min: float = 0.0,
        sigma_max: Optional[float] = None,
        N_grid_channels: int = 4,
        gridtype: Literal["sinusoidal", "linear"] = "sinusoidal",
        patch_size: Union[int, Tuple[int, int]] = 8,
        hidden_size: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        attention_backend: str = "transformer_engine",
        layernorm_backend: str = "torch",
        **_ignored_kwargs,
    ):
        super().__init__(meta=ModelMetaData(name="DiTPrecondSuperResolution"))

        if isinstance(img_resolution, (list, tuple)):
            self._img_shape_y = int(img_resolution[0])
            self._img_shape_x = int(img_resolution[1])
        else:
            self._img_shape_y = self._img_shape_x = int(img_resolution)

        self.img_in_channels = img_in_channels
        self.img_out_channels = img_out_channels
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max if sigma_max is not None else float("inf")
        self._use_fp16 = use_fp16
        self.N_grid_channels = N_grid_channels

        if N_grid_channels > 0:
            grid = EDM2PrecondSuperResolution._make_sinusoidal_grid(
                gridtype, N_grid_channels, self._img_shape_y, self._img_shape_x
            )
            self.register_buffer("pos_embd", grid.float(), persistent=False)
        else:
            self.pos_embd = None

        # Total input channels: noisy HR + LR conditioning + positional grid.
        total_in = img_out_channels + img_in_channels + N_grid_channels
        self.model = DiTSuperRes(
            img_resolution=[self._img_shape_y, self._img_shape_x],
            img_in_channels=total_in,
            img_out_channels=img_out_channels,
            patch_size=patch_size,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            attention_backend=attention_backend,
            layernorm_backend=layernorm_backend,
        )

        if use_fp16:
            self.to(torch.float16)

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
        **_model_kwargs,
    ) -> torch.Tensor:
        """Denoise ``x`` conditioned on ``img_lr`` at noise level ``sigma``.

        Parameters
        ----------
        x : torch.Tensor
            Noisy high-resolution image, shape ``(B, C_hr, H, W)``.
        img_lr : torch.Tensor
            Low-resolution conditioning image, shape ``(B, C_lr, H, W)``.
        sigma : torch.Tensor
            Per-sample noise level, shape ``(B,)`` or broadcastable.
        force_fp32 : bool
            Override ``use_fp16`` and run in FP32.
        **model_kwargs
            Ignored (API compatibility with ``EDMPrecondSuperResolution``).

        Returns
        -------
        torch.Tensor
            Denoised high-resolution image, shape ``(B, C_hr, H, W)``.
        """
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        dtype = (
            torch.float16
            if (self._use_fp16 and not force_fp32 and x.device.type == "cuda")
            else torch.float32
        )

        # EDM preconditioning (Equations 7-9, Karras et al. 2022).
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.flatten().log() / 4

        # Compose input: [c_in * noisy_hr | lr_cond | grid].
        # Standard cat is correct here — DiT's LayerNorm is scale-invariant
        # so magnitude-preserving concat (mp_cat) is not needed.
        parts = [c_in * x, img_lr.to(dtype)]
        if self.pos_embd is not None:
            grid = self.pos_embd.to(dtype).unsqueeze(0).expand(x.shape[0], -1, -1, -1)
            parts.append(grid)
        x_in = torch.cat(parts, dim=1)

        F_x = self.model(x_in, c_noise)
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x


class DiTRegressionUNet(Module):
    """DiT-based deterministic regression model.

    Drop-in replacement for the ``UNet`` wrapper used for the regression
    (mean-prediction) stage of CorrDiff.

    The model receives only the low-resolution conditioning image (plus a
    positional grid) and predicts the high-resolution mean directly — there is
    no noise level to condition on.  A constant ``t=0`` tensor is passed to
    the DiT's timestep embedder, which effectively turns adaLN into a fixed
    bias injection (a plain ViT at inference).

    Parameters
    ----------
    img_resolution : int or [H, W]
        High-resolution spatial size.
    img_in_channels : int
        Number of low-resolution conditioning channels.
    img_out_channels : int
        Number of high-resolution output channels.
    use_fp16 : bool
        Run in FP16 on CUDA (default False).
    N_grid_channels : int
        Positional-grid channels (default 4).
    gridtype : str
        ``"sinusoidal"`` (default) or ``"linear"``.
    patch_size, hidden_size, depth, num_heads, mlp_ratio,
    attention_backend, layernorm_backend :
        Same as ``DiTPrecondSuperResolution``.
    **ignored_kwargs
        Absorbs UNet-specific keys from ``train.py``.
    """

    _overridable_args = {
        "use_apex_gn",
        "checkpoint_level",
        "profile_mode",
        "amp_mode",
        "embedding_type",
    }

    def __init__(
        self,
        img_resolution: Union[int, List[int], Tuple[int, int]],
        img_in_channels: int,
        img_out_channels: int,
        use_fp16: bool = False,
        N_grid_channels: int = 4,
        gridtype: Literal["sinusoidal", "linear"] = "sinusoidal",
        patch_size: Union[int, Tuple[int, int]] = 8,
        hidden_size: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        attention_backend: str = "transformer_engine",
        layernorm_backend: str = "torch",
        **_ignored_kwargs,
    ):
        super().__init__(meta=ModelMetaData(name="DiTRegressionUNet"))

        if isinstance(img_resolution, (list, tuple)):
            self._img_shape_y = int(img_resolution[0])
            self._img_shape_x = int(img_resolution[1])
        else:
            self._img_shape_y = self._img_shape_x = int(img_resolution)

        self.img_in_channels = img_in_channels
        self.img_out_channels = img_out_channels
        self.N_grid_channels = N_grid_channels
        self._use_fp16 = use_fp16

        if N_grid_channels > 0:
            grid = EDM2PrecondSuperResolution._make_sinusoidal_grid(
                gridtype, N_grid_channels, self._img_shape_y, self._img_shape_x
            )
            self.register_buffer("pos_embd", grid.float(), persistent=False)
        else:
            self.pos_embd = None

        # Input channels: LR conditioning + positional grid.
        total_in = img_in_channels + N_grid_channels
        self.model = DiTSuperRes(
            img_resolution=[self._img_shape_y, self._img_shape_x],
            img_in_channels=total_in,
            img_out_channels=img_out_channels,
            patch_size=patch_size,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            attention_backend=attention_backend,
            layernorm_backend=layernorm_backend,
        )

        if use_fp16:
            self.to(torch.float16)

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
        force_fp32: bool = False,
        **_model_kwargs,
    ) -> torch.Tensor:
        """Predict the high-resolution mean from low-resolution input.

        Parameters
        ----------
        x : torch.Tensor
            Ignored placeholder (zeros from ``RegressionLoss``), shape
            ``(B, C_hr, H, W)``.
        img_lr : torch.Tensor
            Low-resolution conditioning image, shape ``(B, C_lr, H, W)``.
        force_fp32 : bool
            Override ``use_fp16`` and run in FP32.
        **model_kwargs
            Ignored (API compatibility).

        Returns
        -------
        torch.Tensor
            Predicted high-resolution image, shape ``(B, C_hr, H, W)``.
        """
        dtype = (
            torch.float16
            if (self._use_fp16 and not force_fp32 and x.device.type == "cuda")
            else torch.float32
        )

        parts = [img_lr.to(dtype)]
        if self.pos_embd is not None:
            grid = self.pos_embd.to(dtype).unsqueeze(0).expand(x.shape[0], -1, -1, -1)
            parts.append(grid)
        lr_cond = torch.cat(parts, dim=1)

        # Constant t=0: adaLN receives a fixed conditioning vector → plain ViT.
        t_zero = torch.zeros(x.shape[0], device=x.device, dtype=torch.float32)
        out = self.model(lr_cond, t_zero)
        return out.to(torch.float32)


# ---------------------------------------------------------------------------
# V2 preconditioners — use DiTSuperResV2 (RoPE, no sinusoidal grid)
# ---------------------------------------------------------------------------


class DiTPrecondSuperResolutionV2(Module):
    """EDM2 preconditioning with DiTSuperResV2 (RoPE 2D + QK-RMSNorm).

    Same external interface as ``DiTPrecondSuperResolution`` but uses
    ``DiTSuperResV2``:

    * No sinusoidal positional grid (N_grid_channels is always 0).
    * RoPE 2D handles positional encoding inside every attention block.
    * Resolution-agnostic: no bicubic pos-embed interpolation at inference.

    Parameters mirror ``DiTPrecondSuperResolution`` (N_grid_channels / gridtype
    dropped; ``use_qk_norm`` added).
    """

    _overridable_args = {"use_apex_gn", "checkpoint_level", "profile_mode", "amp_mode"}

    def __init__(
        self,
        img_resolution: Union[int, List[int], Tuple[int, int]],
        img_in_channels: int,
        img_out_channels: int,
        use_fp16: bool = False,
        sigma_data: float = 0.5,
        sigma_min: float = 0.0,
        sigma_max: Optional[float] = None,
        patch_size: Union[int, Tuple[int, int]] = 8,
        hidden_size: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        use_qk_norm: bool = True,
        **_ignored_kwargs,
    ):
        super().__init__(meta=ModelMetaData(name="DiTPrecondSuperResolutionV2"))

        if isinstance(img_resolution, (list, tuple)):
            self._img_shape_y = int(img_resolution[0])
            self._img_shape_x = int(img_resolution[1])
        else:
            self._img_shape_y = self._img_shape_x = int(img_resolution)

        self.img_in_channels = img_in_channels
        self.img_out_channels = img_out_channels
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max if sigma_max is not None else float("inf")
        self._use_fp16 = use_fp16

        # V2: no sinusoidal grid — total input = noisy HR + LR conditioning
        total_in = img_out_channels + img_in_channels
        self.model = DiTSuperResV2(
            img_resolution=[self._img_shape_y, self._img_shape_x],
            img_in_channels=total_in,
            img_out_channels=img_out_channels,
            patch_size=patch_size,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            use_qk_norm=use_qk_norm,
        )

        if use_fp16:
            self.to(torch.float16)

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
        **_model_kwargs,
    ) -> torch.Tensor:
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        dtype = (
            torch.float16
            if (self._use_fp16 and not force_fp32 and x.device.type == "cuda")
            else torch.float32
        )

        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.flatten().log() / 4

        x_in = torch.cat([c_in * x, img_lr.to(dtype)], dim=1)
        F_x = self.model(x_in, c_noise)
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x


class DiTRegressionUNetV2(Module):
    """DiT V2 deterministic regression model (RoPE 2D + QK-RMSNorm).

    Drop-in replacement for ``DiTRegressionUNet`` using ``DiTSuperResV2``.
    No sinusoidal positional grid — RoPE handles spatial encoding.
    """

    _overridable_args = {
        "use_apex_gn",
        "checkpoint_level",
        "profile_mode",
        "amp_mode",
        "embedding_type",
    }

    def __init__(
        self,
        img_resolution: Union[int, List[int], Tuple[int, int]],
        img_in_channels: int,
        img_out_channels: int,
        use_fp16: bool = False,
        patch_size: Union[int, Tuple[int, int]] = 8,
        hidden_size: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        use_qk_norm: bool = True,
        **_ignored_kwargs,
    ):
        super().__init__(meta=ModelMetaData(name="DiTRegressionUNetV2"))

        if isinstance(img_resolution, (list, tuple)):
            self._img_shape_y = int(img_resolution[0])
            self._img_shape_x = int(img_resolution[1])
        else:
            self._img_shape_y = self._img_shape_x = int(img_resolution)

        self.img_in_channels = img_in_channels
        self.img_out_channels = img_out_channels
        self._use_fp16 = use_fp16

        # V2: input is only the LR conditioning (no sinusoidal grid appended)
        self.model = DiTSuperResV2(
            img_resolution=[self._img_shape_y, self._img_shape_x],
            img_in_channels=img_in_channels,
            img_out_channels=img_out_channels,
            patch_size=patch_size,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            use_qk_norm=use_qk_norm,
        )

        if use_fp16:
            self.to(torch.float16)

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
        force_fp32: bool = False,
        **_model_kwargs,
    ) -> torch.Tensor:
        dtype = (
            torch.float16
            if (self._use_fp16 and not force_fp32 and x.device.type == "cuda")
            else torch.float32
        )
        t_zero = torch.zeros(x.shape[0], device=x.device, dtype=torch.float32)
        out = self.model(img_lr.to(dtype), t_zero)
        return out.to(torch.float32)
