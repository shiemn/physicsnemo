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

"""DiT backbone for CorrDiff super-resolution.

``DiTSuperRes`` wraps the physicsnemo ``DiT`` (via composition, not subclassing)
and adds:
  - A variable-resolution positional-embedding interpolation so the same
    checkpoint can be used at different spatial sizes.
  - A forward signature compatible with ``EDM2PrecondSuperResolution``:
    ``(x_in, noise_labels) -> output``.
"""

import math
from typing import List, Literal, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from physicsnemo.experimental.models.dit.dit import DiT


class DiTSuperRes(nn.Module):
    """DiT backbone for super-resolution diffusion / regression.

    Parameters
    ----------
    img_resolution : int or [H, W]
        Nominal training resolution.  Used to size the positional embedding;
        different resolutions are handled at forward time via bicubic
        interpolation.
    img_in_channels : int
        Total number of input channels fed into the network.  For the
        diffusion model this is ``C_hr + C_lr + N_grid``; for the regression
        model it is ``C_lr + N_grid``.
    img_out_channels : int
        Number of output channels (= C_hr for both models).
    hidden_size : int
        Transformer embedding dimension.
    depth : int
        Number of DiT blocks.
    num_heads : int
        Number of attention heads per block.
    patch_size : int or (int, int)
        Spatial patch size.  Both H and W must be divisible by the respective
        patch dimension.
    mlp_ratio : float
        MLP hidden-dim multiplier inside each DiT block.
    attention_backend : str
        One of ``"transformer_engine"`` (H100 optimised, default),
        ``"timm"``, or ``"natten2d"``.
    layernorm_backend : str
        ``"torch"`` (default) or ``"apex"``.
    dit_initialization : bool
        Apply the DiT-specific weight initialisation (default True).
    """

    def __init__(
        self,
        img_resolution: Union[int, List[int], Tuple[int, int]],
        img_in_channels: int,
        img_out_channels: int,
        hidden_size: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        patch_size: Union[int, Tuple[int, int]] = 8,
        mlp_ratio: float = 4.0,
        attention_backend: Literal[
            "transformer_engine", "timm", "natten2d"
        ] = "transformer_engine",
        layernorm_backend: Literal["apex", "torch"] = "torch",
        dit_initialization: bool = True,
    ):
        super().__init__()

        if isinstance(img_resolution, (list, tuple)):
            nominal_h, nominal_w = int(img_resolution[0]), int(img_resolution[1])
        else:
            nominal_h = nominal_w = int(img_resolution)

        patch_size = (
            patch_size
            if isinstance(patch_size, (list, tuple))
            else (patch_size, patch_size)
        )
        self.patch_size = tuple(patch_size)
        self.out_channels = img_out_channels
        self.hidden_size = hidden_size
        self.h_patches_nom = nominal_h // self.patch_size[0]
        self.w_patches_nom = nominal_w // self.patch_size[1]

        self.backbone = DiT(
            input_size=(nominal_h, nominal_w),
            in_channels=img_in_channels,
            out_channels=img_out_channels,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            attention_backend=attention_backend,
            layernorm_backend=layernorm_backend,
            condition_dim=None,
            dit_initialization=dit_initialization,
        )

    def forward(self, x: torch.Tensor, noise_labels: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape ``(B, C_in, H, W)``.  H and W must each be
            divisible by the respective patch dimension.
        noise_labels : torch.Tensor
            Scalar noise level per sample, shape ``(B,)``.  This is the
            ``c_noise`` value from the EDM2 preconditioner (``log(sigma)/4``).

        Returns
        -------
        torch.Tensor
            Output tensor, shape ``(B, C_out, H, W)``.
        """
        B, _, H, W = x.shape
        p_h, p_w = self.patch_size
        h_p = H // p_h
        w_p = W // p_w

        # 1. Patchify — PatchEmbed2D.proj is a stride-p Conv2d; for any
        #    resolution divisible by patch_size the ZeroPad2d is a no-op.
        x_emb = self.backbone.x_embedder(x)          # (B, D, h_p, w_p)
        x_tok = x_emb.flatten(2).transpose(1, 2)     # (B, h_p*w_p, D)

        # 2. Positional embedding — interpolate when resolution differs from
        #    the nominal training resolution.
        if (h_p, w_p) != (self.h_patches_nom, self.w_patches_nom):
            pos = self.backbone.pos_embed.reshape(
                1, self.h_patches_nom, self.w_patches_nom, self.hidden_size
            ).permute(0, 3, 1, 2)                    # (1, D, h_nom, w_nom)
            pos = F.interpolate(
                pos.float(), size=(h_p, w_p), mode="bicubic", align_corners=False
            ).to(x_tok.dtype)
            pos = pos.permute(0, 2, 3, 1).reshape(1, h_p * w_p, self.hidden_size)
        else:
            pos = self.backbone.pos_embed

        x_tok = x_tok + pos                          # (B, T, D)

        # 3. Noise / timestep conditioning via the learnable PositionalEmbedding.
        #    noise_labels = c_noise = log(sigma)/4 — valid float input for ger().
        c = self.backbone.t_embedder(noise_labels)   # (B, D)

        # 4. Transformer blocks.
        for block in self.backbone.blocks:
            x_tok = block(x_tok, c)                  # (B, T, D)

        # 5. Output projection.
        x_tok = self.backbone.proj_layer(x_tok, c)   # (B, T, p_h*p_w*C_out)

        # 6. Unpatchify using actual h_p, w_p (not the nominal stored values).
        x_tok = x_tok.reshape(B, h_p, w_p, p_h, p_w, self.out_channels)
        x_tok = torch.einsum("nhwpqc->nchpwq", x_tok)
        x_tok = x_tok.reshape(B, self.out_channels, h_p * p_h, w_p * p_w)

        return x_tok


# ---------------------------------------------------------------------------
# V2: RoPE 2D + QK-RMSNorm — no learned positional bias
# ---------------------------------------------------------------------------


def _apply_rope_2d(
    x: torch.Tensor,
    cos_h: torch.Tensor,
    sin_h: torch.Tensor,
    cos_w: torch.Tensor,
    sin_w: torch.Tensor,
) -> torch.Tensor:
    """Apply 2D rotary positional embeddings to x.

    x            : (B, heads, T, head_dim),  head_dim divisible by 4
    cos_h/sin_h  : (1, 1, T, head_dim//4) — height-axis frequencies
    cos_w/sin_w  : (1, 1, T, head_dim//4) — width-axis frequencies
    """
    qh = x.shape[-1] // 4
    x1, x2, x3, x4 = x.split(qh, dim=-1)
    h_real = x1 * cos_h - x2 * sin_h
    h_imag = x1 * sin_h + x2 * cos_h
    w_real = x3 * cos_w - x4 * sin_w
    w_imag = x3 * sin_w + x4 * cos_w
    return torch.cat([h_real, h_imag, w_real, w_imag], dim=-1)


class RotaryEmbedding2D:
    """Stateless 2-D RoPE helper.  No learnable parameters."""

    def get_freqs(
        self,
        h: int,
        w: int,
        head_dim: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(cos_h, sin_h, cos_w, sin_w)``, each ``(1, 1, h*w, head_dim//4)``."""
        assert head_dim % 4 == 0, f"head_dim must be divisible by 4, got {head_dim}"
        qh = head_dim // 4

        theta = 1.0 / (
            10_000 ** (torch.arange(qh, device=device, dtype=torch.float32) / qh)
        )

        rows = torch.arange(h, device=device, dtype=torch.float32)
        cols = torch.arange(w, device=device, dtype=torch.float32)

        # Token at grid position (i, j) → row_freq from row i, col_freq from col j
        row_freqs = torch.outer(rows, theta).unsqueeze(1).expand(h, w, qh).reshape(h * w, qh)
        col_freqs = torch.outer(cols, theta).unsqueeze(0).expand(h, w, qh).reshape(h * w, qh)

        return (
            row_freqs.cos()[None, None],
            row_freqs.sin()[None, None],
            col_freqs.cos()[None, None],
            col_freqs.sin()[None, None],
        )


class _SinusoidalEmbedding(nn.Module):
    """Sinusoidal embedding for scalar timestep / noise level."""

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        half = dim // 2
        freqs = torch.exp(
            -math.log(10_000) * torch.arange(half, dtype=torch.float32) / half
        )
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = t.float()[:, None] * self.freqs[None, :]  # (B, half)
        return torch.cat([emb.cos(), emb.sin()], dim=-1)  # (B, dim)


class DiTBlockV2(nn.Module):
    """DiT block with QK-RMSNorm + 2D RoPE + FlashAttention.

    adaLN-Zero conditioning (DiT paper convention).  RoPE frequencies are
    passed in from the parent model — computed once per forward, shared across
    all blocks.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_qk_norm: bool = True,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)

        # QK-RMSNorm (FLUX / SD3 practice for training stability)
        if use_qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim)
            self.k_norm = nn.RMSNorm(self.head_dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_size),
        )

        # adaLN-Zero: 6 modulation scalars (shift/scale/gate × attn and MLP)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        cos_h: torch.Tensor,
        sin_h: torch.Tensor,
        cos_w: torch.Tensor,
        sin_w: torch.Tensor,
    ) -> torch.Tensor:
        mods = self.adaLN_modulation(c).chunk(6, dim=-1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = [
            m.unsqueeze(1) for m in mods
        ]

        # Self-attention with adaLN conditioning
        B, T, D = x.shape
        x_norm = self.norm1(x) * (1 + scale_msa) + shift_msa

        qkv = self.qkv(x_norm)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q = _apply_rope_2d(q, cos_h, sin_h, cos_w, sin_w)
        k = _apply_rope_2d(k, cos_h, sin_h, cos_w, sin_w)

        # FlashAttention on H100 via torch.compile / SDPA dispatch
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(B, T, D)
        x = x + gate_msa * self.proj(attn)

        # MLP with adaLN conditioning
        x_norm2 = self.norm2(x) * (1 + scale_mlp) + shift_mlp
        x = x + gate_mlp * self.mlp(x_norm2)

        return x


class DiTSuperResV2(nn.Module):
    """DiT V2 backbone: RoPE 2D + QK-RMSNorm, no learned positional bias.

    Drop-in replacement for ``DiTSuperRes`` with the same external interface::

        forward(x: (B, C_in, H, W), noise_labels: (B,)) -> (B, C_out, H, W)

    Key differences from V1:

    * No ``pos_embed`` buffer — RoPE inside every block handles positioning.
    * Resolution-agnostic: any H×W divisible by patch_size works without
      bicubic interpolation.
    * QK-RMSNorm before RoPE for training stability (FLUX/SD3 practice).
    * ``F.scaled_dot_product_attention`` triggers FlashAttention on H100.
    * Self-contained block stack — no physicsnemo DiT backbone dependency.

    Parameters
    ----------
    img_resolution : int or [H, W]
        Nominal resolution.  Not used internally; kept for API parity with V1.
    img_in_channels : int
    img_out_channels : int
    hidden_size : int
    depth : int
    num_heads : int
    patch_size : int or (int, int)
    mlp_ratio : float
    use_qk_norm : bool
        If False, skip QK-RMSNorm (default True).
    """

    def __init__(
        self,
        img_resolution: Union[int, List[int], Tuple[int, int]],
        img_in_channels: int,
        img_out_channels: int,
        hidden_size: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        patch_size: Union[int, Tuple[int, int]] = 8,
        mlp_ratio: float = 4.0,
        use_qk_norm: bool = True,
    ):
        super().__init__()

        patch_size = (
            patch_size if isinstance(patch_size, (list, tuple)) else (patch_size, patch_size)
        )
        self.patch_size = tuple(patch_size)
        self.out_channels = img_out_channels
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # img_resolution is unused internally (RoPE is resolution-agnostic); stored for serialisation parity
        del img_resolution

        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        assert self.head_dim % 4 == 0, (
            f"head_dim ({self.head_dim}) must be divisible by 4 for 2D RoPE"
        )

        # Patchify: stride-p Conv2d → (B, D, h_p, w_p)
        self.x_embedder = nn.Conv2d(
            img_in_channels,
            hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        # Noise-level conditioning: sinusoidal → 2-layer MLP
        self.t_sin = _SinusoidalEmbedding(hidden_size)
        self.t_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.blocks = nn.ModuleList([
            DiTBlockV2(hidden_size, num_heads, mlp_ratio, use_qk_norm) for _ in range(depth)
        ])

        # Final adaLN norm + output projection (DiT paper zero-init convention)
        self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.final_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )
        self.final_proj = nn.Linear(
            hidden_size,
            self.patch_size[0] * self.patch_size[1] * img_out_channels,
        )
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)

        self.rope = RotaryEmbedding2D()

    def forward(self, x: torch.Tensor, noise_labels: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        p_h, p_w = self.patch_size
        h_p, w_p = H // p_h, W // p_w

        # 1. Patchify
        x_tok = self.x_embedder(x).flatten(2).transpose(1, 2)  # (B, T, D)

        # 2. Noise conditioning
        c = self.t_proj(self.t_sin(noise_labels))               # (B, D)

        # 3. RoPE frequencies — computed once, passed to every block
        cos_h, sin_h, cos_w, sin_w = self.rope.get_freqs(
            h_p, w_p, self.head_dim, x.device
        )

        # 4. Transformer blocks
        for block in self.blocks:
            x_tok = block(x_tok, c, cos_h, sin_h, cos_w, sin_w)

        # 5. Final adaLN + projection
        shift, scale = self.final_modulation(c).chunk(2, dim=-1)
        x_tok = (
            self.final_norm(x_tok) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        )
        x_tok = self.final_proj(x_tok)                          # (B, T, p_h*p_w*C_out)

        # 6. Unpatchify
        x_tok = x_tok.reshape(B, h_p, w_p, p_h, p_w, self.out_channels)
        x_tok = torch.einsum("nhwpqc->nchpwq", x_tok)
        x_tok = x_tok.reshape(B, self.out_channels, H, W)

        return x_tok
