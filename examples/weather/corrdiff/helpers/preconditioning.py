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
Preconditioning schemes used in the paper"Elucidating the Design Space of
Diffusion-Based Generative Models".
"""

import importlib
import math
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Tuple, Union

import numpy as np
import torch

from physicsnemo.models.diffusion.song_unet import UNetBlock, checkpoint, silu
from physicsnemo.models.diffusion.utils import _wrapped_property
from physicsnemo.models.meta import ModelMetaData
from physicsnemo.models.module import Module

network_module = importlib.import_module("physicsnemo.models.diffusion")


class TemporalCorrectionRegression(Module):
    """Regression UNet with a compact learned temporal correction input.

    Temporal inputs are expected as frame-major channel stacks:
    ``[frame0_dynamic, frame0_invariant, frame1_dynamic, ...]``.  The model
    keeps the center dynamic channels and center invariants, learns a correction
    from dynamic residuals relative to the center frame, and feeds
    ``[center_dynamic, correction_dynamic, center_invariants]`` to the wrapped
    regression UNet.
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
        img_resolution: Union[int, Tuple[int, int], List[int]],
        img_in_channels: int,
        img_out_channels: int,
        use_fp16: bool = False,
        N_grid_channels: int = 4,
        num_frames: int = 3,
        center_index: int = 1,
        dynamic_channels: int = 8,
        invariant_channels: int = 1,
        hidden_multiplier: int = 1,
        kernel_size: int = 3,
        zero_init_output: bool = True,
        model_type: Literal[
            "SongUNetPosEmbd", "SongUNetPosLtEmbd", "SongUNet", "DhariwalUNet"
        ] = "SongUNetPosEmbd",
        **model_kwargs: Any,
    ):
        super().__init__(meta=ModelMetaData(name="TemporalCorrectionRegression"))

        if num_frames < 2:
            raise ValueError("num_frames must be at least 2")
        if not 0 <= center_index < num_frames:
            raise ValueError(
                f"center_index must be in [0, {num_frames}), got {center_index}"
            )
        if dynamic_channels <= 0:
            raise ValueError("dynamic_channels must be positive")
        if invariant_channels < 0:
            raise ValueError("invariant_channels must be non-negative")
        frame_channels = dynamic_channels + invariant_channels
        expected_channels = num_frames * frame_channels
        if img_in_channels != expected_channels:
            raise ValueError(
                "TemporalCorrectionRegression expected "
                f"{expected_channels} input channels "
                f"({num_frames} frames x {frame_channels} channels), "
                f"got {img_in_channels}."
            )
        if hidden_multiplier <= 0:
            raise ValueError("hidden_multiplier must be positive")
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd")

        self.img_resolution = img_resolution
        self.img_in_channels = img_in_channels
        self.img_out_channels = img_out_channels
        self.N_grid_channels = N_grid_channels
        self.num_frames = num_frames
        self.center_index = center_index
        self.dynamic_channels = dynamic_channels
        self.invariant_channels = invariant_channels
        self.frame_channels = frame_channels
        self.mixed_img_in_channels = dynamic_channels * 2 + invariant_channels
        self._use_fp16 = use_fp16

        residual_channels = (num_frames - 1) * dynamic_channels
        hidden_channels = hidden_multiplier * dynamic_channels
        padding = kernel_size // 2
        self.temporal_mixer = torch.nn.Sequential(
            torch.nn.Conv2d(
                residual_channels,
                hidden_channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=dynamic_channels,
            ),
            torch.nn.SiLU(),
            torch.nn.Conv2d(hidden_channels, dynamic_channels, kernel_size=1),
        )
        if zero_init_output:
            torch.nn.init.zeros_(self.temporal_mixer[-1].weight)
            if self.temporal_mixer[-1].bias is not None:
                torch.nn.init.zeros_(self.temporal_mixer[-1].bias)

        model_class = getattr(network_module, "UNet")
        self.model = model_class(
            img_resolution=img_resolution,
            img_in_channels=self.mixed_img_in_channels + N_grid_channels,
            img_out_channels=img_out_channels,
            use_fp16=use_fp16,
            model_type=model_type,
            **model_kwargs,
        )

        if use_fp16:
            self.to(torch.float16)

    @property
    def use_fp16(self) -> bool:
        return self._use_fp16

    @use_fp16.setter
    def use_fp16(self, value: bool):
        self._use_fp16 = bool(value)
        self.model.use_fp16 = bool(value)

    @staticmethod
    def round_sigma(sigma):
        return torch.as_tensor(sigma)

    def mix_conditioning(self, img_lr: torch.Tensor) -> torch.Tensor:
        """Return ``[center_dynamic, correction_dynamic, center_invariants]``."""
        if img_lr.ndim != 4:
            raise ValueError(
                f"Expected img_lr with shape (B, C, H, W), got {tuple(img_lr.shape)}"
            )
        if img_lr.shape[1] != self.img_in_channels:
            raise ValueError(
                f"Expected {self.img_in_channels} temporal input channels, "
                f"got {img_lr.shape[1]}"
            )

        b, _, h, w = img_lr.shape
        frames = img_lr.reshape(
            b, self.num_frames, self.frame_channels, h, w
        )
        dynamic = frames[:, :, : self.dynamic_channels]
        center_dynamic = dynamic[:, self.center_index]

        residuals = [
            dynamic[:, frame_idx] - center_dynamic
            for frame_idx in range(self.num_frames)
            if frame_idx != self.center_index
        ]
        residuals = torch.cat(residuals, dim=1)
        correction = self.temporal_mixer(residuals)

        parts = [center_dynamic, correction]
        if self.invariant_channels:
            center_invariants = frames[
                :,
                self.center_index,
                self.dynamic_channels : self.frame_channels,
            ]
            parts.append(center_invariants)
        return torch.cat(parts, dim=1)

    def forward(
        self,
        x: torch.Tensor,
        img_lr: torch.Tensor,
        force_fp32: bool = False,
        **model_kwargs,
    ) -> torch.Tensor:
        mixed_img_lr = self.mix_conditioning(img_lr)
        return self.model(
            x=x,
            img_lr=mixed_img_lr,
            force_fp32=force_fp32,
            **model_kwargs,
        )


class MidResolutionTemporalAdapter(torch.nn.Module):
    """Small zero-initialized temporal residual adapter for one UNet resolution."""

    def __init__(
        self,
        feature_channels: int,
        dynamic_channels: int = 8,
        hidden_channels: int = 64,
        zero_init_output: bool = True,
    ):
        super().__init__()
        if feature_channels <= 0:
            raise ValueError("feature_channels must be positive")
        if dynamic_channels <= 0:
            raise ValueError("dynamic_channels must be positive")
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive")

        temporal_channels = dynamic_channels * 3
        self.feature_proj = torch.nn.Conv2d(feature_channels, hidden_channels, 1)
        self.temporal_proj = torch.nn.Sequential(
            torch.nn.Conv2d(temporal_channels, hidden_channels, 3, padding=1),
            torch.nn.SiLU(),
        )
        self.local_mixer = torch.nn.Sequential(
            torch.nn.Conv2d(
                hidden_channels,
                hidden_channels,
                3,
                padding=1,
                groups=hidden_channels,
            ),
            torch.nn.SiLU(),
            torch.nn.Conv2d(hidden_channels, hidden_channels, 1),
            torch.nn.SiLU(),
        )
        self.output_proj = torch.nn.Conv2d(hidden_channels, feature_channels, 1)
        if zero_init_output:
            torch.nn.init.zeros_(self.output_proj.weight)
            if self.output_proj.bias is not None:
                torch.nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: torch.Tensor, temporal_features: torch.Tensor) -> torch.Tensor:
        output_dtype = x.dtype
        adapter_dtype = self.feature_proj.weight.dtype
        x = x.to(dtype=adapter_dtype)
        temporal_features = temporal_features.to(device=x.device, dtype=adapter_dtype)
        hidden = self.feature_proj(x) + self.temporal_proj(temporal_features)
        hidden = self.local_mixer(hidden)
        return self.output_proj(hidden).to(dtype=output_dtype)


class MidResTemporalAdapterRegression(Module):
    """Raw-stacked temporal regression UNet with mid-resolution residual adapters.

    The raw temporal conditioning stack is passed unchanged into the underlying
    UNet.  Lightweight adapters are inserted in the encoder after selected
    resolution blocks and are initialized as no-ops.
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
        img_resolution: Union[int, Tuple[int, int], List[int]],
        img_in_channels: int,
        img_out_channels: int,
        use_fp16: bool = False,
        N_grid_channels: int = 4,
        num_frames: int = 3,
        center_index: int = 1,
        dynamic_channels: int = 8,
        invariant_channels: int = 1,
        adapter_hidden_channels: int = 64,
        adapter_scale: float = 1.0,
        adapter_hook_names: Union[Tuple[str, ...], List[str]] = (
            "256x256_block3",
            "128x128_block3",
        ),
        zero_init_output: bool = True,
        model_type: Literal[
            "SongUNetPosEmbd", "SongUNetPosLtEmbd", "SongUNet", "DhariwalUNet"
        ] = "SongUNetPosEmbd",
        **model_kwargs: Any,
    ):
        super().__init__(meta=ModelMetaData(name="MidResTemporalAdapterRegression"))

        if num_frames != 3:
            raise ValueError(
                "MidResTemporalAdapterRegression currently expects exactly "
                "3 frames: past, center, future."
            )
        if center_index != 1:
            raise ValueError(
                "MidResTemporalAdapterRegression currently expects center_index=1 "
                "for [past, center, future] inputs."
            )
        if dynamic_channels <= 0:
            raise ValueError("dynamic_channels must be positive")
        if invariant_channels < 0:
            raise ValueError("invariant_channels must be non-negative")
        if adapter_hidden_channels <= 0:
            raise ValueError("adapter_hidden_channels must be positive")
        if not adapter_hook_names:
            raise ValueError("adapter_hook_names must contain at least one hook")

        frame_channels = dynamic_channels + invariant_channels
        expected_channels = num_frames * frame_channels
        if img_in_channels != expected_channels:
            raise ValueError(
                "MidResTemporalAdapterRegression expected "
                f"{expected_channels} input channels "
                f"({num_frames} frames x {frame_channels} channels), "
                f"got {img_in_channels}."
            )

        self.img_resolution = img_resolution
        self.img_in_channels = img_in_channels
        self.img_out_channels = img_out_channels
        self.N_grid_channels = N_grid_channels
        self.num_frames = num_frames
        self.center_index = center_index
        self.dynamic_channels = dynamic_channels
        self.invariant_channels = invariant_channels
        self.frame_channels = frame_channels
        self.adapter_hidden_channels = adapter_hidden_channels
        self.adapter_scale = adapter_scale
        self.adapter_hook_names = tuple(adapter_hook_names)
        self._use_fp16 = use_fp16

        model_class = getattr(network_module, "UNet")
        self.model = model_class(
            img_resolution=img_resolution,
            img_in_channels=img_in_channels + N_grid_channels,
            img_out_channels=img_out_channels,
            use_fp16=use_fp16,
            model_type=model_type,
            **model_kwargs,
        )

        enc = self.model.model.enc
        adapters = {}
        missing_hooks = [hook for hook in self.adapter_hook_names if hook not in enc]
        if missing_hooks:
            raise ValueError(
                "MidResTemporalAdapterRegression adapter hooks not found in encoder: "
                + ", ".join(missing_hooks)
            )
        for hook_name in self.adapter_hook_names:
            feature_channels = getattr(enc[hook_name], "out_channels", None)
            if feature_channels is None:
                raise ValueError(
                    f"Could not infer feature channels for adapter hook {hook_name}."
                )
            adapters[hook_name] = MidResolutionTemporalAdapter(
                feature_channels=feature_channels,
                dynamic_channels=dynamic_channels,
                hidden_channels=adapter_hidden_channels,
                zero_init_output=zero_init_output,
            )
        self.adapters = torch.nn.ModuleDict(adapters)

        if use_fp16:
            self.to(torch.float16)

    @property
    def use_fp16(self) -> bool:
        return self._use_fp16

    @use_fp16.setter
    def use_fp16(self, value: bool):
        self._use_fp16 = bool(value)
        self.model.use_fp16 = bool(value)
        self.adapters.to(torch.float16 if value else torch.float32)

    @staticmethod
    def round_sigma(sigma):
        return torch.as_tensor(sigma)

    def _split_dynamic(self, img_lr: torch.Tensor) -> torch.Tensor:
        if img_lr.ndim != 4:
            raise ValueError(
                f"Expected img_lr with shape (B, C, H, W), got {tuple(img_lr.shape)}"
            )
        if img_lr.shape[1] != self.img_in_channels:
            raise ValueError(
                f"Expected {self.img_in_channels} temporal input channels, "
                f"got {img_lr.shape[1]}"
            )
        b, _, h, w = img_lr.shape
        frames = img_lr.reshape(b, self.num_frames, self.frame_channels, h, w)
        return frames[:, :, : self.dynamic_channels]

    def temporal_features_for_resolution(
        self, img_lr: torch.Tensor, size: Tuple[int, int], dtype: torch.dtype
    ) -> torch.Tensor:
        dynamic = self._split_dynamic(img_lr)
        center = dynamic[:, self.center_index]
        past = dynamic[:, 0]
        future = dynamic[:, 2]
        temporal = torch.cat(
            [past - center, future - center, future - past],
            dim=1,
        )
        if temporal.shape[-2:] != size:
            temporal = torch.nn.functional.interpolate(
                temporal,
                size=size,
                mode="bilinear",
                align_corners=False,
            )
        return temporal.to(dtype=dtype)

    def _apply_adapter(
        self, hook_name: str, x: torch.Tensor, img_lr: torch.Tensor
    ) -> torch.Tensor:
        temporal = self.temporal_features_for_resolution(
            img_lr, size=x.shape[-2:], dtype=x.dtype
        )
        return x + self.adapter_scale * self.adapters[hook_name](x, temporal)

    def _forward_songunet_with_adapters(
        self,
        x: torch.Tensor,
        noise_labels: torch.Tensor,
        class_labels: torch.Tensor | None,
        img_lr: torch.Tensor,
        global_index=None,
        embedding_selector=None,
        augment_labels=None,
        lead_time_label=None,
    ) -> torch.Tensor:
        inner = self.model.model
        if embedding_selector is not None and global_index is not None:
            raise ValueError("Cannot provide both embedding_selector and global_index.")

        if hasattr(inner, "pos_embd") and (
            (inner.pos_embd is not None) or (inner.lt_embd is not None)
        ):
            if embedding_selector is not None:
                selected_pos_embd = inner.positional_embedding_selector(
                    x,
                    embedding_selector,
                    lead_time_label=lead_time_label,
                )
            else:
                selected_pos_embd = inner.positional_embedding_indexing(
                    x,
                    global_index=global_index,
                    lead_time_label=lead_time_label,
                )
            x = torch.cat((x, selected_pos_embd.to(x.dtype)), dim=1)

        if (
            inner.use_apex_gn
            and (not x.is_contiguous(memory_format=torch.channels_last))
            and x.dim() == 4
        ):
            x = x.to(memory_format=torch.channels_last)

        if inner.embedding_type != "zero":
            emb = inner.map_noise(noise_labels)
            emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)
            if inner.map_label is not None:
                tmp = class_labels
                if inner.training and inner.label_dropout:
                    tmp = tmp * (
                        torch.rand([x.shape[0], 1], device=x.device)
                        >= inner.label_dropout
                    ).to(tmp.dtype)
                emb = emb + inner.map_label(tmp * np.sqrt(inner.map_label.in_features))
            if inner.map_augment is not None and augment_labels is not None:
                emb = emb + inner.map_augment(augment_labels)
            emb = silu(inner.map_layer0(emb))
            emb = silu(inner.map_layer1(emb))
        else:
            emb = torch.zeros(
                (noise_labels.shape[0], inner.emb_channels),
                device=x.device,
                dtype=x.dtype,
            )

        skips = []
        aux = x
        for name, block in inner.enc.items():
            if "aux_down" in name:
                aux = block(aux)
            elif "aux_skip" in name:
                x = skips[-1] = x + block(aux)
            elif "aux_residual" in name:
                x = skips[-1] = aux = (x + block(aux)) / np.sqrt(2)
            elif "_conv" in name:
                x = block(x)
                if inner.additive_pos_embed:
                    x = x + inner.spatial_emb.to(dtype=x.dtype)
                skips.append(x)
            else:
                if isinstance(block, UNetBlock):
                    if (
                        math.floor(math.sqrt(x.shape[-2] * x.shape[-1]))
                        > inner.checkpoint_threshold
                    ):
                        x = checkpoint(block, x, emb, use_reentrant=False)
                    else:
                        x = block(x, emb)
                else:
                    x = block(x)
                if name in self.adapters:
                    x = self._apply_adapter(name, x, img_lr)
                skips.append(x)

        aux = None
        tmp = None
        for name, block in inner.dec.items():
            if "aux_up" in name:
                aux = block(aux)
            elif "aux_norm" in name:
                tmp = block(x)
            elif "aux_conv" in name:
                tmp = block(silu(tmp))
                aux = tmp if aux is None else tmp + aux
            else:
                if x.shape[1] != block.in_channels:
                    x = torch.cat([x, skips.pop()], dim=1)
                if (
                    math.floor(math.sqrt(x.shape[-2] * x.shape[-1]))
                    > inner.checkpoint_threshold
                    and "_block" in name
                ) or (
                    math.floor(math.sqrt(x.shape[-2] * x.shape[-1]))
                    > (inner.checkpoint_threshold / 2)
                    and "_up" in name
                ):
                    x = checkpoint(block, x, emb, use_reentrant=False)
                else:
                    x = block(x, emb)

        if getattr(inner, "lead_time_mode", False) and inner.prob_channels:
            scalar = inner.scalar
            if aux.dtype != scalar.dtype:
                scalar = scalar.to(aux.dtype)
            if inner.training:
                aux[:, inner.prob_channels] = aux[:, inner.prob_channels] * scalar
            else:
                aux[:, inner.prob_channels] = (
                    (aux[:, inner.prob_channels] * scalar)
                    .softmax(dim=1)
                    .to(aux.dtype)
                )
        return aux

    def forward(
        self,
        x: torch.Tensor,
        img_lr: torch.Tensor,
        force_fp32: bool = False,
        **model_kwargs,
    ) -> torch.Tensor:
        if img_lr is None:
            raise ValueError("MidResTemporalAdapterRegression requires img_lr.")
        self._split_dynamic(img_lr)
        x = torch.cat((x, img_lr), dim=1)
        dtype = (
            torch.float16
            if (self.use_fp16 and not force_fp32 and x.device.type == "cuda")
            else torch.float32
        )

        F_x = self._forward_songunet_with_adapters(
            x.to(dtype),
            torch.zeros(x.shape[0], dtype=dtype, device=x.device),
            class_labels=None,
            img_lr=img_lr,
            **model_kwargs,
        )
        if (F_x.dtype != dtype) and not torch.is_autocast_enabled():
            raise ValueError(f"Expected the dtype to be {dtype}, but got {F_x.dtype}.")
        return F_x.to(torch.float32)


class LocalTemporalAttentionRegression(Module):
    """Regression UNet with local temporal attention over dynamic inputs.

    The wrapper preserves the existing regression API while replacing raw
    temporal channel stacking with ``[center_dynamic, attended_context,
    center_invariants]`` conditioning.  Attention is computed on a coarse grid
    and attends from the center frame to local windows in non-center frames.
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
        img_resolution: Union[int, Tuple[int, int], List[int]],
        img_in_channels: int,
        img_out_channels: int,
        use_fp16: bool = False,
        N_grid_channels: int = 4,
        num_frames: int = 3,
        center_index: int = 1,
        dynamic_channels: int = 8,
        invariant_channels: int = 1,
        embed_channels: int = 32,
        num_heads: int = 4,
        attention_stride: int = 4,
        window_radius: int = 2,
        output_mode: Literal["center_attended_elevation"] = "center_attended_elevation",
        model_type: Literal[
            "SongUNetPosEmbd", "SongUNetPosLtEmbd", "SongUNet", "DhariwalUNet"
        ] = "SongUNetPosEmbd",
        **model_kwargs: Any,
    ):
        super().__init__(meta=ModelMetaData(name="LocalTemporalAttentionRegression"))

        if output_mode != "center_attended_elevation":
            raise ValueError(
                "LocalTemporalAttentionRegression only supports "
                "output_mode='center_attended_elevation'"
            )
        if num_frames < 2:
            raise ValueError("num_frames must be at least 2")
        if not 0 <= center_index < num_frames:
            raise ValueError(
                f"center_index must be in [0, {num_frames}), got {center_index}"
            )
        if dynamic_channels <= 0:
            raise ValueError("dynamic_channels must be positive")
        if invariant_channels < 0:
            raise ValueError("invariant_channels must be non-negative")
        frame_channels = dynamic_channels + invariant_channels
        expected_channels = num_frames * frame_channels
        if img_in_channels != expected_channels:
            raise ValueError(
                "LocalTemporalAttentionRegression expected "
                f"{expected_channels} input channels "
                f"({num_frames} frames x {frame_channels} channels), "
                f"got {img_in_channels}."
            )
        if embed_channels <= 0:
            raise ValueError("embed_channels must be positive")
        if num_heads <= 0 or embed_channels % num_heads != 0:
            raise ValueError("embed_channels must be divisible by num_heads")
        if attention_stride <= 0:
            raise ValueError("attention_stride must be positive")
        if window_radius < 0:
            raise ValueError("window_radius must be non-negative")

        self.img_resolution = img_resolution
        self.img_in_channels = img_in_channels
        self.img_out_channels = img_out_channels
        self.N_grid_channels = N_grid_channels
        self.num_frames = num_frames
        self.center_index = center_index
        self.dynamic_channels = dynamic_channels
        self.invariant_channels = invariant_channels
        self.frame_channels = frame_channels
        self.embed_channels = embed_channels
        self.num_heads = num_heads
        self.head_dim = embed_channels // num_heads
        self.attention_stride = attention_stride
        self.window_radius = window_radius
        self.window_size = 2 * window_radius + 1
        self.source_indices = [
            frame_idx for frame_idx in range(num_frames) if frame_idx != center_index
        ]
        self.mixed_img_in_channels = dynamic_channels * 2 + invariant_channels
        self._use_fp16 = use_fp16

        self.input_stem = torch.nn.Sequential(
            torch.nn.Conv2d(dynamic_channels, embed_channels, kernel_size=3, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(embed_channels, embed_channels, kernel_size=3, padding=1),
        )
        self.query_proj = torch.nn.Conv2d(embed_channels, embed_channels, kernel_size=1)
        self.key_proj = torch.nn.Conv2d(embed_channels, embed_channels, kernel_size=1)
        self.value_proj = torch.nn.Conv2d(embed_channels, embed_channels, kernel_size=1)
        self.context_proj = torch.nn.Sequential(
            torch.nn.Conv2d(embed_channels, embed_channels, kernel_size=3, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(embed_channels, dynamic_channels, kernel_size=1),
        )

        model_class = getattr(network_module, "UNet")
        self.model = model_class(
            img_resolution=img_resolution,
            img_in_channels=self.mixed_img_in_channels + N_grid_channels,
            img_out_channels=img_out_channels,
            use_fp16=use_fp16,
            model_type=model_type,
            **model_kwargs,
        )

        if use_fp16:
            self.to(torch.float16)

    @property
    def use_fp16(self) -> bool:
        return self._use_fp16

    @use_fp16.setter
    def use_fp16(self, value: bool):
        self._use_fp16 = bool(value)
        self.model.use_fp16 = bool(value)

    @staticmethod
    def round_sigma(sigma):
        return torch.as_tensor(sigma)

    def _split_frames(self, img_lr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if img_lr.ndim != 4:
            raise ValueError(
                f"Expected img_lr with shape (B, C, H, W), got {tuple(img_lr.shape)}"
            )
        if img_lr.shape[1] != self.img_in_channels:
            raise ValueError(
                f"Expected {self.img_in_channels} temporal input channels, "
                f"got {img_lr.shape[1]}"
            )
        b, _, h, w = img_lr.shape
        frames = img_lr.reshape(b, self.num_frames, self.frame_channels, h, w)
        dynamic = frames[:, :, : self.dynamic_channels]
        invariants = frames[:, :, self.dynamic_channels : self.frame_channels]
        return dynamic, invariants

    def _coarse_features(self, dynamic: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = dynamic.shape
        x = dynamic.reshape(b * t, c, h, w)
        features = self.input_stem(x)
        if self.attention_stride > 1:
            features = torch.nn.functional.avg_pool2d(
                features,
                kernel_size=self.attention_stride,
                stride=self.attention_stride,
                ceil_mode=False,
            )
        _, e, hc, wc = features.shape
        return features.reshape(b, t, e, hc, wc)

    def compute_attention(
        self,
        img_lr: torch.Tensor,
        return_weights: bool = True,
    ):
        """Compute full-resolution attended dynamic context.

        Returns ``(context, weights)`` when ``return_weights`` is true.  The
        weight tensor has shape ``(B, heads, Hc, Wc, source_frame, win_y, win_x)``.
        """
        dynamic, _ = self._split_frames(img_lr)
        b, _, _, h, w = dynamic.shape
        features = self._coarse_features(dynamic)
        _, _, _, hc, wc = features.shape
        n_locations = hc * wc
        n_sources = len(self.source_indices)
        window_area = self.window_size * self.window_size

        center_features = features[:, self.center_index]
        source_features = features[:, self.source_indices]

        query = self.query_proj(center_features)
        query = query.reshape(b, self.num_heads, self.head_dim, n_locations)

        source_flat = source_features.reshape(
            b * n_sources, self.embed_channels, hc, wc
        )
        keys = self.key_proj(source_flat)
        values = self.value_proj(source_flat)

        keys = torch.nn.functional.unfold(
            keys,
            kernel_size=self.window_size,
            padding=self.window_radius,
        )
        values = torch.nn.functional.unfold(
            values,
            kernel_size=self.window_size,
            padding=self.window_radius,
        )
        keys = keys.reshape(
            b,
            n_sources,
            self.num_heads,
            self.head_dim,
            window_area,
            n_locations,
        )
        values = values.reshape(
            b,
            n_sources,
            self.num_heads,
            self.head_dim,
            window_area,
            n_locations,
        )

        scale = self.head_dim ** -0.5
        scores = (query[:, None, :, :, None, :] * keys).sum(dim=3) * scale
        scores = scores.permute(0, 2, 4, 1, 3).reshape(
            b, self.num_heads, n_locations, n_sources * window_area
        )
        weights_flat = torch.softmax(scores, dim=-1)
        weights = weights_flat.reshape(
            b, self.num_heads, n_locations, n_sources, window_area
        ).permute(0, 1, 3, 4, 2)

        context = (weights[:, :, :, None] * values.permute(0, 2, 1, 3, 4, 5)).sum(
            dim=(2, 4)
        )
        context = context.reshape(b, self.embed_channels, hc, wc)
        context = self.context_proj(context)
        if (hc, wc) != (h, w):
            context = torch.nn.functional.interpolate(
                context,
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )

        if not return_weights:
            return context

        weights_out = weights_flat.reshape(
            b,
            self.num_heads,
            hc,
            wc,
            n_sources,
            self.window_size,
            self.window_size,
        )
        return context, weights_out

    def mix_conditioning(self, img_lr: torch.Tensor) -> torch.Tensor:
        dynamic, invariants = self._split_frames(img_lr)
        center_dynamic = dynamic[:, self.center_index]
        attended_context = self.compute_attention(img_lr, return_weights=False)

        parts = [center_dynamic, attended_context]
        if self.invariant_channels:
            parts.append(invariants[:, self.center_index])
        return torch.cat(parts, dim=1)

    def forward(
        self,
        x: torch.Tensor,
        img_lr: torch.Tensor,
        force_fp32: bool = False,
        **model_kwargs,
    ) -> torch.Tensor:
        mixed_img_lr = self.mix_conditioning(img_lr)
        return self.model(
            x=x,
            img_lr=mixed_img_lr,
            force_fp32=force_fp32,
            **model_kwargs,
        )


class FeatureTemporalAttentionRegression(Module):
    """Regression UNet with feature-level temporal cross-attention.

    Temporal dynamic inputs are encoded with a shared per-frame stem. The center
    frame then attends separately to past and future feature maps, keeping the
    aligned latent products distinct for the downstream UNet.
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
        img_resolution: Union[int, Tuple[int, int], List[int]],
        img_in_channels: int,
        img_out_channels: int,
        use_fp16: bool = False,
        N_grid_channels: int = 4,
        num_frames: int = 3,
        center_index: int = 1,
        dynamic_channels: int = 8,
        invariant_channels: int = 1,
        embed_channels: int = 32,
        num_heads: int = 4,
        attention_stride: int = 4,
        window_radius: int = 2,
        output_mode: Literal[
            "center_aligned_latents_elevation"
        ] = "center_aligned_latents_elevation",
        model_type: Literal[
            "SongUNetPosEmbd", "SongUNetPosLtEmbd", "SongUNet", "DhariwalUNet"
        ] = "SongUNetPosEmbd",
        **model_kwargs: Any,
    ):
        super().__init__(meta=ModelMetaData(name="FeatureTemporalAttentionRegression"))

        if output_mode != "center_aligned_latents_elevation":
            raise ValueError(
                "FeatureTemporalAttentionRegression only supports "
                "output_mode='center_aligned_latents_elevation'"
            )
        if num_frames != 3:
            raise ValueError(
                "FeatureTemporalAttentionRegression currently expects exactly "
                "3 frames: past, center, future."
            )
        if center_index != 1:
            raise ValueError(
                "FeatureTemporalAttentionRegression currently expects "
                "center_index=1 for [past, center, future] inputs."
            )
        if dynamic_channels <= 0:
            raise ValueError("dynamic_channels must be positive")
        if invariant_channels < 0:
            raise ValueError("invariant_channels must be non-negative")
        if embed_channels <= 0:
            raise ValueError("embed_channels must be positive")
        if num_heads <= 0 or embed_channels % num_heads != 0:
            raise ValueError("embed_channels must be divisible by num_heads")
        if attention_stride <= 0:
            raise ValueError("attention_stride must be positive")
        if window_radius < 0:
            raise ValueError("window_radius must be non-negative")

        frame_channels = dynamic_channels + invariant_channels
        expected_channels = num_frames * frame_channels
        if img_in_channels != expected_channels:
            raise ValueError(
                "FeatureTemporalAttentionRegression expected "
                f"{expected_channels} input channels "
                f"({num_frames} frames x {frame_channels} channels), "
                f"got {img_in_channels}."
            )

        self.img_resolution = img_resolution
        self.img_in_channels = img_in_channels
        self.img_out_channels = img_out_channels
        self.N_grid_channels = N_grid_channels
        self.num_frames = num_frames
        self.center_index = center_index
        self.dynamic_channels = dynamic_channels
        self.invariant_channels = invariant_channels
        self.frame_channels = frame_channels
        self.embed_channels = embed_channels
        self.num_heads = num_heads
        self.head_dim = embed_channels // num_heads
        self.attention_stride = attention_stride
        self.window_radius = window_radius
        self.window_size = 2 * window_radius + 1
        self.source_indices = [0, 2]
        self.source_names = ["past", "future"]
        self.mixed_img_in_channels = embed_channels * 3 + invariant_channels
        self._use_fp16 = use_fp16

        self.input_stem = torch.nn.Sequential(
            torch.nn.Conv2d(dynamic_channels, embed_channels, kernel_size=3, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(embed_channels, embed_channels, kernel_size=3, padding=1),
            torch.nn.SiLU(),
        )
        self.temporal_embeddings = torch.nn.Parameter(
            torch.zeros(num_frames, embed_channels)
        )
        torch.nn.init.normal_(self.temporal_embeddings, mean=0.0, std=0.02)
        self.query_proj = torch.nn.Conv2d(embed_channels, embed_channels, kernel_size=1)
        self.key_proj = torch.nn.Conv2d(embed_channels, embed_channels, kernel_size=1)
        self.value_proj = torch.nn.Conv2d(embed_channels, embed_channels, kernel_size=1)
        self.context_proj = torch.nn.Sequential(
            torch.nn.Conv2d(embed_channels, embed_channels, kernel_size=3, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(embed_channels, embed_channels, kernel_size=1),
        )

        model_class = getattr(network_module, "UNet")
        self.model = model_class(
            img_resolution=img_resolution,
            img_in_channels=self.mixed_img_in_channels + N_grid_channels,
            img_out_channels=img_out_channels,
            use_fp16=use_fp16,
            model_type=model_type,
            **model_kwargs,
        )

        if use_fp16:
            self.to(torch.float16)

    @property
    def use_fp16(self) -> bool:
        return self._use_fp16

    @use_fp16.setter
    def use_fp16(self, value: bool):
        self._use_fp16 = bool(value)
        self.model.use_fp16 = bool(value)

    @staticmethod
    def round_sigma(sigma):
        return torch.as_tensor(sigma)

    def _split_frames(self, img_lr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if img_lr.ndim != 4:
            raise ValueError(
                f"Expected img_lr with shape (B, C, H, W), got {tuple(img_lr.shape)}"
            )
        if img_lr.shape[1] != self.img_in_channels:
            raise ValueError(
                f"Expected {self.img_in_channels} temporal input channels, "
                f"got {img_lr.shape[1]}"
            )
        b, _, h, w = img_lr.shape
        frames = img_lr.reshape(b, self.num_frames, self.frame_channels, h, w)
        dynamic = frames[:, :, : self.dynamic_channels]
        invariants = frames[:, :, self.dynamic_channels : self.frame_channels]
        return dynamic, invariants

    def _encode_features(
        self, dynamic: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, t, c, h, w = dynamic.shape
        x = dynamic.reshape(b * t, c, h, w)
        full_features = self.input_stem(x).reshape(
            b, t, self.embed_channels, h, w
        )
        embedded = full_features + self.temporal_embeddings[
            None, :, :, None, None
        ].to(dtype=full_features.dtype, device=full_features.device)
        if self.attention_stride > 1:
            coarse_features = torch.nn.functional.avg_pool2d(
                embedded.reshape(b * t, self.embed_channels, h, w),
                kernel_size=self.attention_stride,
                stride=self.attention_stride,
                ceil_mode=False,
            )
            _, _, hc, wc = coarse_features.shape
            coarse_features = coarse_features.reshape(
                b, t, self.embed_channels, hc, wc
            )
        else:
            coarse_features = embedded
        return full_features, coarse_features

    def _attend_to_source(
        self,
        query: torch.Tensor,
        source_features: torch.Tensor,
        output_size: Tuple[int, int],
        return_weights: bool,
    ):
        b, _, hc, wc = source_features.shape
        n_locations = hc * wc
        window_area = self.window_size * self.window_size

        keys = self.key_proj(source_features)
        values = self.value_proj(source_features)
        keys = torch.nn.functional.unfold(
            keys,
            kernel_size=self.window_size,
            padding=self.window_radius,
        )
        values = torch.nn.functional.unfold(
            values,
            kernel_size=self.window_size,
            padding=self.window_radius,
        )
        keys = keys.reshape(
            b,
            self.num_heads,
            self.head_dim,
            window_area,
            n_locations,
        )
        values = values.reshape(
            b,
            self.num_heads,
            self.head_dim,
            window_area,
            n_locations,
        )

        scale = self.head_dim ** -0.5
        scores = (query[:, :, :, None, :] * keys).sum(dim=2) * scale
        scores = scores.permute(0, 1, 3, 2)
        weights_flat = torch.softmax(scores, dim=-1)
        weights = weights_flat.permute(0, 1, 3, 2)

        context = (weights[:, :, None] * values).sum(dim=3)
        context = context.reshape(b, self.embed_channels, hc, wc)
        context = self.context_proj(context)
        if (hc, wc) != output_size:
            context = torch.nn.functional.interpolate(
                context,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )

        if not return_weights:
            return context

        weights_out = weights_flat.reshape(
            b,
            self.num_heads,
            hc,
            wc,
            self.window_size,
            self.window_size,
        )
        return context, weights_out

    def _compute_attention_from_features(
        self,
        coarse_features: torch.Tensor,
        output_size: Tuple[int, int],
        return_weights: bool = True,
    ):
        b, _, _, hc, wc = coarse_features.shape
        n_locations = hc * wc

        center_features = coarse_features[:, self.center_index]
        query = self.query_proj(center_features)
        query = query.reshape(b, self.num_heads, self.head_dim, n_locations)

        contexts = {}
        weights_by_source = {}
        for source_name, source_idx in zip(self.source_names, self.source_indices):
            if return_weights:
                context, weights = self._attend_to_source(
                    query,
                    coarse_features[:, source_idx],
                    output_size=output_size,
                    return_weights=True,
                )
                weights_by_source[source_name] = weights
            else:
                context = self._attend_to_source(
                    query,
                    coarse_features[:, source_idx],
                    output_size=output_size,
                    return_weights=False,
                )
            contexts[source_name] = context

        if not return_weights:
            return contexts
        return contexts, weights_by_source

    def compute_attention(
        self,
        img_lr: torch.Tensor,
        return_weights: bool = True,
    ):
        """Compute separate full-resolution aligned past/future latent contexts."""
        dynamic, _ = self._split_frames(img_lr)
        _, coarse_features = self._encode_features(dynamic)
        return self._compute_attention_from_features(
            coarse_features,
            output_size=dynamic.shape[-2:],
            return_weights=return_weights,
        )

    def mix_conditioning(self, img_lr: torch.Tensor) -> torch.Tensor:
        dynamic, invariants = self._split_frames(img_lr)
        full_features, coarse_features = self._encode_features(dynamic)
        center_features = full_features[:, self.center_index]
        aligned = self._compute_attention_from_features(
            coarse_features,
            output_size=dynamic.shape[-2:],
            return_weights=False,
        )

        parts = [center_features, aligned["past"], aligned["future"]]
        if self.invariant_channels:
            parts.append(invariants[:, self.center_index])
        return torch.cat(parts, dim=1)

    def forward(
        self,
        x: torch.Tensor,
        img_lr: torch.Tensor,
        force_fp32: bool = False,
        **model_kwargs,
    ) -> torch.Tensor:
        mixed_img_lr = self.mix_conditioning(img_lr)
        return self.model(
            x=x,
            img_lr=mixed_img_lr,
            force_fp32=force_fp32,
            **model_kwargs,
        )


class _PyramidTemporalAttentionLevel(torch.nn.Module):
    """One coarse local temporal attention level used by the pyramid wrapper."""

    def __init__(
        self,
        dynamic_channels: int,
        embed_channels: int,
        num_heads: int,
        attention_stride: int,
        window_radius: int,
        num_frames: int,
        use_temporal_embeddings: bool = False,
    ):
        super().__init__()
        if embed_channels <= 0:
            raise ValueError("embed_channels must be positive")
        if num_heads <= 0 or embed_channels % num_heads != 0:
            raise ValueError("embed_channels must be divisible by num_heads")
        if attention_stride <= 0:
            raise ValueError("attention_stride must be positive")
        if window_radius < 0:
            raise ValueError("window_radius must be non-negative")

        self.dynamic_channels = dynamic_channels
        self.embed_channels = embed_channels
        self.num_heads = num_heads
        self.head_dim = embed_channels // num_heads
        self.attention_stride = attention_stride
        self.window_radius = window_radius
        self.window_size = 2 * window_radius + 1
        self.num_frames = num_frames
        self.use_temporal_embeddings = use_temporal_embeddings

        self.input_stem = torch.nn.Sequential(
            torch.nn.Conv2d(dynamic_channels, embed_channels, kernel_size=3, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(embed_channels, embed_channels, kernel_size=3, padding=1),
        )
        if use_temporal_embeddings:
            self.temporal_embeddings = torch.nn.Parameter(
                torch.zeros(num_frames, embed_channels)
            )
            torch.nn.init.normal_(self.temporal_embeddings, mean=0.0, std=0.02)
        else:
            self.register_parameter("temporal_embeddings", None)
        self.query_proj = torch.nn.Conv2d(embed_channels, embed_channels, kernel_size=1)
        self.key_proj = torch.nn.Conv2d(embed_channels, embed_channels, kernel_size=1)
        self.value_proj = torch.nn.Conv2d(embed_channels, embed_channels, kernel_size=1)
        self.context_proj = torch.nn.Sequential(
            torch.nn.Conv2d(embed_channels, embed_channels, kernel_size=3, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(embed_channels, dynamic_channels, kernel_size=1),
        )

    def _coarse_features(self, dynamic: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = dynamic.shape
        x = dynamic.reshape(b * t, c, h, w)
        features = self.input_stem(x)
        if self.attention_stride > 1:
            features = torch.nn.functional.avg_pool2d(
                features,
                kernel_size=self.attention_stride,
                stride=self.attention_stride,
                ceil_mode=False,
            )
        _, e, hc, wc = features.shape
        return features.reshape(b, t, e, hc, wc)

    def forward(
        self,
        dynamic: torch.Tensor,
        center_index: int,
        source_indices: List[int],
        return_weights: bool = True,
    ):
        b, _, _, h, w = dynamic.shape
        features = self._coarse_features(dynamic)
        if self.temporal_embeddings is not None:
            features = features + self.temporal_embeddings[None, :, :, None, None].to(
                dtype=features.dtype,
                device=features.device,
            )
        _, _, _, hc, wc = features.shape
        n_locations = hc * wc
        n_sources = len(source_indices)
        window_area = self.window_size * self.window_size

        center_features = features[:, center_index]
        source_features = features[:, source_indices]

        query = self.query_proj(center_features)
        query = query.reshape(b, self.num_heads, self.head_dim, n_locations)

        source_flat = source_features.reshape(
            b * n_sources, self.embed_channels, hc, wc
        )
        keys = self.key_proj(source_flat)
        values = self.value_proj(source_flat)

        keys = torch.nn.functional.unfold(
            keys,
            kernel_size=self.window_size,
            padding=self.window_radius,
        )
        values = torch.nn.functional.unfold(
            values,
            kernel_size=self.window_size,
            padding=self.window_radius,
        )
        keys = keys.reshape(
            b,
            n_sources,
            self.num_heads,
            self.head_dim,
            window_area,
            n_locations,
        )
        values = values.reshape(
            b,
            n_sources,
            self.num_heads,
            self.head_dim,
            window_area,
            n_locations,
        )

        scale = self.head_dim ** -0.5
        scores = (query[:, None, :, :, None, :] * keys).sum(dim=3) * scale
        scores = scores.permute(0, 2, 4, 1, 3).reshape(
            b, self.num_heads, n_locations, n_sources * window_area
        )
        weights_flat = torch.softmax(scores, dim=-1)
        weights = weights_flat.reshape(
            b, self.num_heads, n_locations, n_sources, window_area
        ).permute(0, 1, 3, 4, 2)

        context = (weights[:, :, :, None] * values.permute(0, 2, 1, 3, 4, 5)).sum(
            dim=(2, 4)
        )
        context = context.reshape(b, self.embed_channels, hc, wc)
        context = self.context_proj(context)
        if (hc, wc) != (h, w):
            context = torch.nn.functional.interpolate(
                context,
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )

        if not return_weights:
            return context

        weights_out = weights_flat.reshape(
            b,
            self.num_heads,
            hc,
            wc,
            n_sources,
            self.window_size,
            self.window_size,
        )
        return context, weights_out


class PyramidLocalTemporalAttentionRegression(Module):
    """Regression UNet with multi-scale local temporal attention context."""

    _overridable_args = {
        "use_apex_gn",
        "checkpoint_level",
        "profile_mode",
        "amp_mode",
        "embedding_type",
    }

    def __init__(
        self,
        img_resolution: Union[int, Tuple[int, int], List[int]],
        img_in_channels: int,
        img_out_channels: int,
        use_fp16: bool = False,
        N_grid_channels: int = 4,
        num_frames: int = 3,
        center_index: int = 1,
        dynamic_channels: int = 8,
        invariant_channels: int = 1,
        levels: Union[List[Dict[str, Any]], Tuple[Dict[str, Any], ...], None] = None,
        fusion_channels: int = 32,
        use_temporal_embeddings: bool = False,
        output_mode: Literal[
            "center_pyramid_elevation"
        ] = "center_pyramid_elevation",
        model_type: Literal[
            "SongUNetPosEmbd", "SongUNetPosLtEmbd", "SongUNet", "DhariwalUNet"
        ] = "SongUNetPosEmbd",
        **model_kwargs: Any,
    ):
        super().__init__(
            meta=ModelMetaData(name="PyramidLocalTemporalAttentionRegression")
        )

        if output_mode != "center_pyramid_elevation":
            raise ValueError(
                "PyramidLocalTemporalAttentionRegression only supports "
                "output_mode='center_pyramid_elevation'"
            )
        if num_frames < 2:
            raise ValueError("num_frames must be at least 2")
        if not 0 <= center_index < num_frames:
            raise ValueError(
                f"center_index must be in [0, {num_frames}), got {center_index}"
            )
        if dynamic_channels <= 0:
            raise ValueError("dynamic_channels must be positive")
        if invariant_channels < 0:
            raise ValueError("invariant_channels must be non-negative")
        if fusion_channels <= 0:
            raise ValueError("fusion_channels must be positive")

        frame_channels = dynamic_channels + invariant_channels
        expected_channels = num_frames * frame_channels
        if img_in_channels != expected_channels:
            raise ValueError(
                "PyramidLocalTemporalAttentionRegression expected "
                f"{expected_channels} input channels "
                f"({num_frames} frames x {frame_channels} channels), "
                f"got {img_in_channels}."
            )

        if levels is None:
            levels = [
                {
                    "name": "local",
                    "attention_stride": 4,
                    "window_radius": 2,
                    "embed_channels": 16,
                    "num_heads": 2,
                },
                {
                    "name": "mesoscale",
                    "attention_stride": 8,
                    "window_radius": 4,
                    "embed_channels": 32,
                    "num_heads": 4,
                },
                {
                    "name": "broad",
                    "attention_stride": 16,
                    "window_radius": 4,
                    "embed_channels": 32,
                    "num_heads": 4,
                },
            ]
        if not levels:
            raise ValueError("levels must contain at least one attention level")

        self.img_resolution = img_resolution
        self.img_in_channels = img_in_channels
        self.img_out_channels = img_out_channels
        self.N_grid_channels = N_grid_channels
        self.num_frames = num_frames
        self.center_index = center_index
        self.dynamic_channels = dynamic_channels
        self.invariant_channels = invariant_channels
        self.frame_channels = frame_channels
        self.fusion_channels = fusion_channels
        self.use_temporal_embeddings = use_temporal_embeddings
        self.source_indices = [
            frame_idx for frame_idx in range(num_frames) if frame_idx != center_index
        ]
        self.mixed_img_in_channels = dynamic_channels * 2 + invariant_channels
        self._use_fp16 = use_fp16

        self.level_names = []
        self.attention_levels = torch.nn.ModuleList()
        seen_names = set()
        for level_idx, level_cfg in enumerate(levels):
            level_cfg = dict(level_cfg)
            name = str(level_cfg.pop("name", f"level_{level_idx}"))
            if name in seen_names:
                raise ValueError(f"Duplicate temporal attention level name: {name}")
            seen_names.add(name)
            allowed_keys = {
                "embed_channels",
                "num_heads",
                "attention_stride",
                "window_radius",
                "use_temporal_embeddings",
            }
            unknown_keys = sorted(set(level_cfg) - allowed_keys)
            if unknown_keys:
                raise ValueError(
                    f"Unknown temporal attention level keys for {name}: {unknown_keys}"
                )
            self.level_names.append(name)
            self.attention_levels.append(
                _PyramidTemporalAttentionLevel(
                    dynamic_channels=dynamic_channels,
                    embed_channels=int(level_cfg.get("embed_channels", 32)),
                    num_heads=int(level_cfg.get("num_heads", 4)),
                    attention_stride=int(level_cfg.get("attention_stride", 4)),
                    window_radius=int(level_cfg.get("window_radius", 2)),
                    num_frames=num_frames,
                    use_temporal_embeddings=bool(
                        level_cfg.get(
                            "use_temporal_embeddings",
                            use_temporal_embeddings,
                        )
                    ),
                )
            )

        self.context_fusion = torch.nn.Sequential(
            torch.nn.Conv2d(
                dynamic_channels * len(self.attention_levels),
                fusion_channels,
                kernel_size=3,
                padding=1,
            ),
            torch.nn.SiLU(),
            torch.nn.Conv2d(fusion_channels, dynamic_channels, kernel_size=1),
        )

        model_class = getattr(network_module, "UNet")
        self.model = model_class(
            img_resolution=img_resolution,
            img_in_channels=self.mixed_img_in_channels + N_grid_channels,
            img_out_channels=img_out_channels,
            use_fp16=use_fp16,
            model_type=model_type,
            **model_kwargs,
        )

        if use_fp16:
            self.to(torch.float16)

    @property
    def use_fp16(self) -> bool:
        return self._use_fp16

    @use_fp16.setter
    def use_fp16(self, value: bool):
        self._use_fp16 = bool(value)
        self.model.use_fp16 = bool(value)

    @staticmethod
    def round_sigma(sigma):
        return torch.as_tensor(sigma)

    def _split_frames(self, img_lr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if img_lr.ndim != 4:
            raise ValueError(
                f"Expected img_lr with shape (B, C, H, W), got {tuple(img_lr.shape)}"
            )
        if img_lr.shape[1] != self.img_in_channels:
            raise ValueError(
                f"Expected {self.img_in_channels} temporal input channels, "
                f"got {img_lr.shape[1]}"
            )
        b, _, h, w = img_lr.shape
        frames = img_lr.reshape(b, self.num_frames, self.frame_channels, h, w)
        dynamic = frames[:, :, : self.dynamic_channels]
        invariants = frames[:, :, self.dynamic_channels : self.frame_channels]
        return dynamic, invariants

    def compute_attention(
        self,
        img_lr: torch.Tensor,
        return_weights: bool = True,
    ):
        dynamic, _ = self._split_frames(img_lr)
        contexts = []
        weights_by_level = {}

        for name, level in zip(self.level_names, self.attention_levels):
            if return_weights:
                context, weights = level(
                    dynamic,
                    center_index=self.center_index,
                    source_indices=self.source_indices,
                    return_weights=True,
                )
                weights_by_level[name] = weights
            else:
                context = level(
                    dynamic,
                    center_index=self.center_index,
                    source_indices=self.source_indices,
                    return_weights=False,
                )
            contexts.append(context)

        fused_context = self.context_fusion(torch.cat(contexts, dim=1))
        if not return_weights:
            return fused_context
        return fused_context, weights_by_level

    def mix_conditioning(self, img_lr: torch.Tensor) -> torch.Tensor:
        dynamic, invariants = self._split_frames(img_lr)
        center_dynamic = dynamic[:, self.center_index]
        attended_context = self.compute_attention(img_lr, return_weights=False)

        parts = [center_dynamic, attended_context]
        if self.invariant_channels:
            parts.append(invariants[:, self.center_index])
        return torch.cat(parts, dim=1)

    def forward(
        self,
        x: torch.Tensor,
        img_lr: torch.Tensor,
        force_fp32: bool = False,
        **model_kwargs,
    ) -> torch.Tensor:
        mixed_img_lr = self.mix_conditioning(img_lr)
        return self.model(
            x=x,
            img_lr=mixed_img_lr,
            force_fp32=force_fp32,
            **model_kwargs,
        )


@dataclass
class VPPrecondMetaData(ModelMetaData):
    """VPPrecond meta data"""

    name: str = "VPPrecond"
    # Optimization
    jit: bool = False
    cuda_graphs: bool = False
    amp_cpu: bool = False
    amp_gpu: bool = True
    torch_fx: bool = False
    # Data type
    bf16: bool = False
    # Inference
    onnx: bool = False
    # Physics informed
    func_torch: bool = False
    auto_grad: bool = False


class VPPrecond(Module):
    """
    Preconditioning corresponding to the variance preserving (VP) formulation.

    Parameters
    ----------
    img_resolution : int
        Image resolution.
    img_channels : int
        Number of color channels.
    label_dim : int
        Number of class labels, 0 = unconditional, by default 0.
    use_fp16 : bool
        Execute the underlying model at FP16 precision?, by default False.
    beta_d : float
        Extent of the noise level schedule, by default 19.9.
    beta_min : float
        Initial slope of the noise level schedule, by default 0.1.
    M : int
        Original number of timesteps in the DDPM formulation, by default 1000.
    epsilon_t : float
        Minimum t-value used during training, by default 1e-5.
    model_type :str
        Class name of the underlying model, by default "SongUNet".
    **model_kwargs : dict
        Keyword arguments for the underlying model.

    Note
    ----
    Reference: Song, Y., Sohl-Dickstein, J., Kingma, D.P., Kumar, A., Ermon, S. and
    Poole, B., 2020. Score-based generative modeling through stochastic differential
    equations. arXiv preprint arXiv:2011.13456.
    """

    def __init__(
        self,
        img_resolution: int,
        img_channels: int,
        label_dim: int = 0,
        use_fp16: bool = False,
        beta_d: float = 19.9,
        beta_min: float = 0.1,
        M: int = 1000,
        epsilon_t: float = 1e-5,
        model_type: str = "SongUNet",
        **model_kwargs: dict,
    ):
        super().__init__(meta=VPPrecondMetaData)
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.M = M
        self.epsilon_t = epsilon_t
        self.sigma_min = float(self.sigma(epsilon_t))
        self.sigma_max = float(self.sigma(1))
        model_class = getattr(network_module, model_type)
        self.model = model_class(
            img_resolution=img_resolution,
            in_channels=img_channels,
            out_channels=img_channels,
            label_dim=label_dim,
            **model_kwargs,
        )  # TODO needs better handling

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = (
            None
            if self.label_dim == 0
            else torch.zeros([1, self.label_dim], device=x.device)
            if class_labels is None
            else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        )
        dtype = (
            torch.float16
            if (self.use_fp16 and not force_fp32 and x.device.type == "cuda")
            else torch.float32
        )

        c_skip = 1
        c_out = -sigma
        c_in = 1 / (sigma**2 + 1).sqrt()
        c_noise = (self.M - 1) * self.sigma_inv(sigma)

        F_x = self.model(
            (c_in * x).to(dtype),
            c_noise.flatten(),
            class_labels=class_labels,
            **model_kwargs,
        )
        if (F_x.dtype != dtype) and not torch.is_autocast_enabled():
            raise ValueError(
                f"Expected the dtype to be {dtype}, but got {F_x.dtype} instead."
            )

        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def sigma(self, t: Union[float, torch.Tensor]):
        """
        Compute the sigma(t) value for a given t based on the VP formulation.

        The function calculates the noise level schedule for the diffusion process based
        on the given parameters `beta_d` and `beta_min`.

        Parameters
        ----------
        t : Union[float, torch.Tensor]
            The timestep or set of timesteps for which to compute sigma(t).

        Returns
        -------
        torch.Tensor
            The computed sigma(t) value(s).
        """
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t**2) + self.beta_min * t).exp() - 1).sqrt()

    def sigma_inv(self, sigma: Union[float, torch.Tensor]):
        """
        Compute the inverse of the sigma function for a given sigma.

        This function effectively calculates t from a given sigma(t) based on the
        parameters `beta_d` and `beta_min`.

        Parameters
        ----------
        sigma : Union[float, torch.Tensor]
            The sigma(t) value or set of sigma(t) values for which to compute the
            inverse.

        Returns
        -------
        torch.Tensor
            The computed t value(s) corresponding to the provided sigma(t).
        """
        sigma = torch.as_tensor(sigma)
        return (
            (self.beta_min**2 + 2 * self.beta_d * (1 + sigma**2).log()).sqrt()
            - self.beta_min
        ) / self.beta_d

    def round_sigma(self, sigma: Union[float, List, torch.Tensor]):
        """
        Convert a given sigma value(s) to a tensor representation.

        Parameters
        ----------
        sigma : Union[float list, torch.Tensor]
            The sigma value(s) to convert.

        Returns
        -------
        torch.Tensor
            The tensor representation of the provided sigma value(s).
        """
        return torch.as_tensor(sigma)


@dataclass
class VEPrecondMetaData(ModelMetaData):
    """VEPrecond meta data"""

    name: str = "VEPrecond"
    # Optimization
    jit: bool = False
    cuda_graphs: bool = False
    amp_cpu: bool = False
    amp_gpu: bool = True
    torch_fx: bool = False
    # Data type
    bf16: bool = False
    # Inference
    onnx: bool = False
    # Physics informed
    func_torch: bool = False
    auto_grad: bool = False


class VEPrecond(Module):
    """
    Preconditioning corresponding to the variance exploding (VE) formulation.

    Parameters
    ----------
    img_resolution : int
        Image resolution.
    img_channels : int
        Number of color channels.
    label_dim : int
        Number of class labels, 0 = unconditional, by default 0.
    use_fp16 : bool
        Execute the underlying model at FP16 precision?, by default False.
    sigma_min : float
        Minimum supported noise level, by default 0.02.
    sigma_max : float
        Maximum supported noise level, by default 100.0.
    model_type :str
        Class name of the underlying model, by default "SongUNet".
    **model_kwargs : dict
        Keyword arguments for the underlying model.

    Note
    ----
    Reference: Song, Y., Sohl-Dickstein, J., Kingma, D.P., Kumar, A., Ermon, S. and
    Poole, B., 2020. Score-based generative modeling through stochastic differential
    equations. arXiv preprint arXiv:2011.13456.
    """

    def __init__(
        self,
        img_resolution: int,
        img_channels: int,
        label_dim: int = 0,
        use_fp16: bool = False,
        sigma_min: float = 0.02,
        sigma_max: float = 100.0,
        model_type: str = "SongUNet",
        **model_kwargs: dict,
    ):
        super().__init__(meta=VEPrecondMetaData)
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        model_class = getattr(network_module, model_type)
        self.model = model_class(
            img_resolution=img_resolution,
            in_channels=img_channels,
            out_channels=img_channels,
            label_dim=label_dim,
            **model_kwargs,
        )  # TODO needs better handling

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = (
            None
            if self.label_dim == 0
            else torch.zeros([1, self.label_dim], device=x.device)
            if class_labels is None
            else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        )
        dtype = (
            torch.float16
            if (self.use_fp16 and not force_fp32 and x.device.type == "cuda")
            else torch.float32
        )

        c_skip = 1
        c_out = sigma
        c_in = 1
        c_noise = (0.5 * sigma).log()

        F_x = self.model(
            (c_in * x).to(dtype),
            c_noise.flatten(),
            class_labels=class_labels,
            **model_kwargs,
        )
        if (F_x.dtype != dtype) and not torch.is_autocast_enabled():
            raise ValueError(
                f"Expected the dtype to be {dtype}, but got {F_x.dtype} instead."
            )

        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma: Union[float, List, torch.Tensor]):
        """
        Convert a given sigma value(s) to a tensor representation.

        Parameters
        ----------
        sigma : Union[float list, torch.Tensor]
            The sigma value(s) to convert.

        Returns
        -------
        torch.Tensor
            The tensor representation of the provided sigma value(s).
        """
        return torch.as_tensor(sigma)


@dataclass
class iDDPMPrecondMetaData(ModelMetaData):
    """iDDPMPrecond meta data"""

    name: str = "iDDPMPrecond"
    # Optimization
    jit: bool = False
    cuda_graphs: bool = False
    amp_cpu: bool = False
    amp_gpu: bool = True
    torch_fx: bool = False
    # Data type
    bf16: bool = False
    # Inference
    onnx: bool = False
    # Physics informed
    func_torch: bool = False
    auto_grad: bool = False


class iDDPMPrecond(Module):
    """
    Preconditioning corresponding to the improved DDPM (iDDPM) formulation.

    Parameters
    ----------
    img_resolution : int
        Image resolution.
    img_channels : int
        Number of color channels.
    label_dim : int
        Number of class labels, 0 = unconditional, by default 0.
    use_fp16 : bool
        Execute the underlying model at FP16 precision?, by default False.
    C_1 : float
        Timestep adjustment at low noise levels., by default 0.001.
    C_2 : float
        Timestep adjustment at high noise levels., by default 0.008.
    M: int
        Original number of timesteps in the DDPM formulation, by default 1000.
    model_type :str
        Class name of the underlying model, by default "DhariwalUNet".
    **model_kwargs : dict
        Keyword arguments for the underlying model.

    Note
    ----
    Reference: Nichol, A.Q. and Dhariwal, P., 2021, July. Improved denoising diffusion
    probabilistic models. In International Conference on Machine Learning
    (pp. 8162-8171). PMLR.
    """

    def __init__(
        self,
        img_resolution,
        img_channels,
        label_dim=0,
        use_fp16=False,
        C_1=0.001,
        C_2=0.008,
        M=1000,
        model_type="DhariwalUNet",
        **model_kwargs,
    ):
        super().__init__(meta=iDDPMPrecondMetaData)
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.C_1 = C_1
        self.C_2 = C_2
        self.M = M
        model_class = getattr(network_module, model_type)
        self.model = model_class(
            img_resolution=img_resolution,
            in_channels=img_channels,
            out_channels=img_channels * 2,
            label_dim=label_dim,
            **model_kwargs,
        )  # TODO needs better handling

        u = torch.zeros(M + 1)
        for j in range(M, 0, -1):  # M, ..., 1
            u[j - 1] = (
                (u[j] ** 2 + 1)
                / (self.alpha_bar(j - 1) / self.alpha_bar(j)).clip(min=C_1)
                - 1
            ).sqrt()
        self.register_buffer("u", u)
        self.sigma_min = float(u[M - 1])
        self.sigma_max = float(u[0])

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = (
            None
            if self.label_dim == 0
            else torch.zeros([1, self.label_dim], device=x.device)
            if class_labels is None
            else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        )
        dtype = (
            torch.float16
            if (self.use_fp16 and not force_fp32 and x.device.type == "cuda")
            else torch.float32
        )

        c_skip = 1
        c_out = -sigma
        c_in = 1 / (sigma**2 + 1).sqrt()
        c_noise = (
            self.M - 1 - self.round_sigma(sigma, return_index=True).to(torch.float32)
        )

        F_x = self.model(
            (c_in * x).to(dtype),
            c_noise.flatten(),
            class_labels=class_labels,
            **model_kwargs,
        )
        if (F_x.dtype != dtype) and not torch.is_autocast_enabled():
            raise ValueError(
                f"Expected the dtype to be {dtype}, but got {F_x.dtype} instead."
            )

        D_x = c_skip * x + c_out * F_x[:, : self.img_channels].to(torch.float32)
        return D_x

    def alpha_bar(self, j):
        """
        Compute the alpha_bar(j) value for a given j based on the iDDPM formulation.

        Parameters
        ----------
        j : Union[int, torch.Tensor]
            The timestep or set of timesteps for which to compute alpha_bar(j).

        Returns
        -------
        torch.Tensor
            The computed alpha_bar(j) value(s).
        """
        j = torch.as_tensor(j)
        return (0.5 * np.pi * j / self.M / (self.C_2 + 1)).sin() ** 2

    def round_sigma(self, sigma, return_index=False):
        """
        Round the provided sigma value(s) to the nearest value(s) in a
        pre-defined set `u`.

        Parameters
        ----------
        sigma : Union[float, list, torch.Tensor]
            The sigma value(s) to round.
        return_index : bool, optional
            Whether to return the index/indices of the rounded value(s) in `u` instead
            of the rounded value(s) themselves, by default False.

        Returns
        -------
        torch.Tensor
            The rounded sigma value(s) or their index/indices in `u`, depending on the
            value of `return_index`.
        """
        sigma = torch.as_tensor(sigma)
        index = torch.cdist(
            sigma.to(self.u.device).to(torch.float32).reshape(1, -1, 1),
            self.u.reshape(1, -1, 1),
        ).argmin(2)
        result = index if return_index else self.u[index.flatten()].to(sigma.dtype)
        return result.reshape(sigma.shape).to(sigma.device)


@dataclass
class EDMPrecondMetaData(ModelMetaData):
    """EDMPrecond meta data"""

    name: str = "EDMPrecond"
    # Optimization
    jit: bool = False
    cuda_graphs: bool = False
    amp_cpu: bool = False
    amp_gpu: bool = True
    torch_fx: bool = False
    # Data type
    bf16: bool = False
    # Inference
    onnx: bool = False
    # Physics informed
    func_torch: bool = False
    auto_grad: bool = False


class EDMPrecond(Module):
    """
    Improved preconditioning proposed in the paper "Elucidating the Design Space of
    Diffusion-Based Generative Models" (EDM)

    Parameters
    ----------
    img_resolution : int
        Image resolution.
    img_channels : int
        Number of color channels (for both input and output). If your model
        requires a different number of input or output chanels,
        override this by passing either of the optional
        img_in_channels or img_out_channels args
    label_dim : int
        Number of class labels, 0 = unconditional, by default 0.
    use_fp16 : bool
        Execute the underlying model at FP16 precision?, by default False.
    sigma_min : float
        Minimum supported noise level, by default 0.0.
    sigma_max : float
        Maximum supported noise level, by default inf.
    sigma_data : float
        Expected standard deviation of the training data, by default 0.5.
    model_type :str
        Class name of the underlying model, by default "DhariwalUNet".
    img_in_channels: int
        Optional setting for when number of input channels =/= number of output
        channels. If set, will override img_channels for the input
        This is useful in the case of additional (conditional) channels
    img_out_channels: int
        Optional setting for when number of input channels =/= number of output
        channels. If set, will override img_channels for the output
    **model_kwargs : dict
        Keyword arguments for the underlying model.

    Note
    ----
    Reference: Karras, T., Aittala, M., Aila, T. and Laine, S., 2022. Elucidating the
    design space of diffusion-based generative models. Advances in Neural Information
    Processing Systems, 35, pp.26565-26577.
    """

    def __init__(
        self,
        img_resolution,
        img_channels,
        label_dim=0,
        use_fp16=False,
        sigma_min=0.0,
        sigma_max=float("inf"),
        sigma_data=0.5,
        model_type="DhariwalUNet",
        img_in_channels=None,
        img_out_channels=None,
        **model_kwargs,
    ):
        super().__init__(meta=EDMPrecondMetaData)
        self.img_resolution = img_resolution
        if img_in_channels is not None:
            img_in_channels = img_in_channels
        else:
            img_in_channels = img_channels
        if img_out_channels is not None:
            img_out_channels = img_out_channels
        else:
            img_out_channels = img_channels

        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data

        model_class = getattr(network_module, model_type)
        self.model = model_class(
            img_resolution=img_resolution,
            in_channels=img_in_channels,
            out_channels=img_out_channels,
            label_dim=label_dim,
            **model_kwargs,
        )  # TODO needs better handling

    def forward(
        self,
        x,
        sigma,
        condition=None,
        class_labels=None,
        force_fp32=False,
        **model_kwargs,
    ):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = (
            None
            if self.label_dim == 0
            else torch.zeros([1, self.label_dim], device=x.device)
            if class_labels is None
            else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        )
        dtype = (
            torch.float16
            if (self.use_fp16 and not force_fp32 and x.device.type == "cuda")
            else torch.float32
        )

        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.log() / 4

        arg = c_in * x

        if condition is not None:
            arg = torch.cat([arg, condition], dim=1)

        F_x = self.model(
            arg.to(dtype),
            c_noise.flatten(),
            class_labels=class_labels,
            **model_kwargs,
        )

        if (F_x.dtype != dtype) and not torch.is_autocast_enabled():
            raise ValueError(
                f"Expected the dtype to be {dtype}, but got {F_x.dtype} instead."
            )
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    @staticmethod
    def round_sigma(sigma: Union[float, List, torch.Tensor]):
        """
        Convert a given sigma value(s) to a tensor representation.

        Parameters
        ----------
        sigma : Union[float list, torch.Tensor]
            The sigma value(s) to convert.

        Returns
        -------
        torch.Tensor
            The tensor representation of the provided sigma value(s).
        """
        return torch.as_tensor(sigma)


@dataclass
class EDMPrecondSuperResolutionMetaData(ModelMetaData):
    """EDMPrecondSuperResolution meta data"""

    name: str = "EDMPrecondSuperResolution"
    # Optimization
    jit: bool = False
    cuda_graphs: bool = False
    amp_cpu: bool = False
    amp_gpu: bool = True
    torch_fx: bool = False
    # Data type
    bf16: bool = False
    # Inference
    onnx: bool = False
    # Physics informed
    func_torch: bool = False
    auto_grad: bool = False


class EDMPrecondSuperResolution(Module):
    """
    Improved preconditioning proposed in the paper "Elucidating the Design Space of
    Diffusion-Based Generative Models" (EDM).

    This is a variant of `EDMPrecond` that is specifically designed for super-resolution
    tasks. It wraps a neural network that predicts the denoised high-resolution image
    given a noisy high-resolution image, and additional conditioning that includes a
    low-resolution image, and a noise level.

    Parameters
    ----------
    img_resolution : Union[int, Tuple[int, int]]
        Spatial resolution :math:`(H, W)` of the image. If a single int is provided,
        the image is assumed to be square.
    img_in_channels : int
        Number of input channels in the low-resolution input image.
    img_out_channels : int
        Number of output channels in the high-resolution output image.
    use_fp16 : bool, optional
        Whether to use half-precision floating point (FP16) for model execution,
        by default False.
    model_type : str, optional
        Class name of the underlying model. Must be one of the following:
        'SongUNet', 'SongUNetPosEmbd', 'SongUNetPosLtEmbd', 'DhariwalUNet'.
        Defaults to 'SongUNetPosEmbd'.
    sigma_data : float, optional
        Expected standard deviation of the training data, by default 0.5.
    sigma_min : float, optional
        Minimum supported noise level, by default 0.0.
    sigma_max : float, optional
        Maximum supported noise level, by default inf.
    **model_kwargs : dict
        Keyword arguments passed to the underlying model `__init__` method.

    See Also
    --------
    For information on model types and their usage:
    :class:`~physicsnemo.models.diffusion.SongUNet`: Basic U-Net for diffusion models
    :class:`~physicsnemo.models.diffusion.SongUNetPosEmbd`: U-Net with positional embeddings
    :class:`~physicsnemo.models.diffusion.SongUNetPosLtEmbd`: U-Net with positional and lead-time embeddings

    Please refer to the documentation of these classes for details on how to call
    and use these models directly.

    Note
    ----
    References:
    - Karras, T., Aittala, M., Aila, T. and Laine, S., 2022. Elucidating the
    design space of diffusion-based generative models. Advances in Neural Information
    Processing Systems, 35, pp.26565-26577.
    - Mardani, M., Brenowitz, N., Cohen, Y., Pathak, J., Chen, C.Y.,
    Liu, C.C.,Vahdat, A., Kashinath, K., Kautz, J. and Pritchard, M., 2023.
    Generative Residual Diffusion Modeling for Km-scale Atmospheric Downscaling.
    arXiv preprint arXiv:2309.15214.
    """

    # Classes that can be wrapped by this UNet class.
    _wrapped_classes = {
        "SongUNetPosEmbd",
        "SongUNetPosLtEmbd",
        "SongUNet",
        "DhariwalUNet",
    }

    # Arguments of the __init__ method that can be overridden with the
    # ``Module.from_checkpoint`` method. Here, since we use splatted arguments
    # for the wrapped model instance, we allow overriding of any overridable
    # argument of the wrapped classes.
    _overridable_args = set.union(
        *(
            getattr(getattr(network_module, cls_name), "_overridable_args", set())
            for cls_name in _wrapped_classes
        )
    )

    def __init__(
        self,
        img_resolution: Union[int, Tuple[int, int]],
        img_in_channels: int,
        img_out_channels: int,
        use_fp16: bool = False,
        model_type: Literal[
            "SongUNetPosEmbd", "SongUNetPosLtEmbd", "SongUNet", "DhariwalUNet"
        ] = "SongUNetPosEmbd",
        sigma_data: float = 0.5,
        sigma_min=0.0,
        sigma_max=float("inf"),
        **model_kwargs: Any,
    ):
        super().__init__(meta=EDMPrecondSuperResolutionMetaData)

        # Validation
        if model_type not in self._wrapped_classes:
            raise ValueError(
                f"Model type '{model_type}' is not supported. "
                f"Must be one of: {', '.join(self._wrapped_classes)}"
            )

        self.img_resolution = img_resolution
        self.img_in_channels = img_in_channels
        self.img_out_channels = img_out_channels
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        model_class = getattr(network_module, model_type)
        self.model = model_class(
            img_resolution=img_resolution,
            in_channels=img_in_channels + img_out_channels,
            out_channels=img_out_channels,
            **model_kwargs,
        )  # TODO needs better handling
        self.scaling_fn = self._scaling_fn
        self.use_fp16 = use_fp16

    @property
    def use_fp16(self):
        """
        bool: Whether the model uses float16 precision.

        Returns
        -------
        bool
            True if the model is in float16 mode, False otherwise.
        """
        return self._use_fp16

    @use_fp16.setter
    def use_fp16(self, value: bool):
        """
        Set whether the model should use float16 precision.

        Parameters
        ----------
        value : bool
            If True, moves the model to torch.float16. If False, moves to torch.float32.

        Raises
        ------
        ValueError
            If `value` is not a boolean.
        """
        # NOTE: allow 0/1 values for older checkpoints
        if not (isinstance(value, bool) or value in [0, 1]):
            raise ValueError(
                f"`use_fp16` must be a boolean, but got {type(value).__name__}."
            )
        self._use_fp16 = value
        if value:
            self.to(torch.float16)
        else:
            self.to(torch.float32)

    @staticmethod
    def _scaling_fn(
        x: torch.Tensor, img_lr: torch.Tensor, c_in: torch.Tensor
    ) -> torch.Tensor:
        """
        Scale input tensors by first scaling the high-resolution tensor and then
        concatenating with the low-resolution tensor.

        Parameters
        ----------
        x : torch.Tensor
            Noisy high-resolution image of shape (B, C_hr, H, W).
        img_lr : torch.Tensor
            Low-resolution image of shape (B, C_lr, H, W).
        c_in : torch.Tensor
            Scaling factor of shape (B, 1, 1, 1).

        Returns
        -------
        torch.Tensor
            Scaled and concatenated tensor of shape (B, C_in+C_out, H, W).
        """
        return torch.cat([c_in * x, img_lr.to(x.dtype)], dim=1)

    # Properties delegated to the wrapped model
    amp_mode = _wrapped_property(
        "amp_mode",
        "model",
        "Set to ``True`` when using automatic mixed precision.",
    )
    profile_mode = _wrapped_property(
        "profile_mode",
        "model",
        "Set to ``True`` to enable profiling of the wrapped model.",
    )

    def forward(
        self,
        x: torch.Tensor,
        img_lr: torch.Tensor,
        sigma: torch.Tensor,
        force_fp32: bool = False,
        **model_kwargs: Any,
    ) -> torch.Tensor:
        """
        Forward pass of the EDMPrecondSuperResolution model wrapper.

        This method applies the EDM preconditioning to compute the denoised image
        from a noisy high-resolution image and low-resolution conditioning image.

        Parameters
        ----------
        x : torch.Tensor
            Noisy high-resolution image of shape (B, C_hr, H, W). The number of
            channels `C_hr` should be equal to `img_out_channels`.
        img_lr : torch.Tensor
            Low-resolution conditioning image of shape (B, C_lr, H, W). The number
            of channels `C_lr` should be equal to `img_in_channels`.
        sigma : torch.Tensor
            Noise level of shape (B) or (B, 1) or (B, 1, 1, 1).
        force_fp32 : bool, optional
            Whether to force FP32 precision regardless of the `use_fp16` attribute,
            by default False.
        **model_kwargs : dict
            Additional keyword arguments to pass to the underlying model
            `self.model` forward method.

        Returns
        -------
        torch.Tensor
            Denoised high-resolution image of shape (B, C_hr, H, W).

        Raises
        ------
        ValueError
            If the model output dtype doesn't match the expected dtype.
        """
        # Concatenate input channels
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        dtype = (
            torch.float16
            if (self.use_fp16 and not force_fp32 and x.device.type == "cuda")
            else torch.float32
        )

        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.log() / 4

        if img_lr is None:
            arg = c_in * x
        else:
            arg = self.scaling_fn(x, img_lr, c_in)
        arg = arg.to(dtype)

        F_x = self.model(
            arg,
            c_noise.flatten(),
            class_labels=None,
            **model_kwargs,
        )

        if (F_x.dtype != dtype) and not torch.is_autocast_enabled():
            raise ValueError(
                f"Expected the dtype to be {dtype}, but got {F_x.dtype} instead."
            )

        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    @staticmethod
    def round_sigma(sigma: Union[float, List, torch.Tensor]) -> torch.Tensor:
        """
        Convert a given sigma value(s) to a tensor representation.

        Parameters
        ----------
        sigma : Union[float, List, torch.Tensor]
            Sigma value(s) to convert.

        Returns
        -------
        torch.Tensor
            Tensor representation of sigma values.

        See Also
        --------
        EDMPrecond.round_sigma
        """
        return EDMPrecond.round_sigma(sigma)


@dataclass
class HeteroscedasticEDMPrecondSRMetaData(ModelMetaData):
    """HeteroscedasticEDMPrecondSR meta data"""

    name: str = "HeteroscedasticEDMPrecondSR"
    # Optimization
    jit: bool = False
    cuda_graphs: bool = False
    amp_cpu: bool = False
    amp_gpu: bool = True
    torch_fx: bool = False
    # Data type
    bf16: bool = False
    # Inference
    onnx: bool = False
    # Physics informed
    func_torch: bool = False
    auto_grad: bool = False


class HeteroscedasticEDMPrecondSR(EDMPrecondSuperResolution):
    """
    Heteroscedastic variant of EDMPrecondSuperResolution that predicts both
    the denoised mean and a spatially-varying uncertainty (standard deviation).

    This model extends EDMPrecondSuperResolution by adding a variance prediction
    head that outputs the predicted uncertainty for each pixel. The uncertainty
    can be trained with a Gaussian CRPS loss for calibration and used during
    inference to modulate the stochastic sampler's noise injection.

    The forward pass returns a tuple (D_mean, D_std) where:
    - D_mean: denoised prediction (same as parent class)
    - D_std: predicted standard deviation in data space (not scaled by c_out)

    Parameters
    ----------
    img_resolution : Union[int, Tuple[int, int]]
        Spatial resolution (H, W) of the image.
    img_in_channels : int
        Number of input channels in the low-resolution input image.
    img_out_channels : int
        Number of output channels in the high-resolution output image.
    use_fp16 : bool, optional
        Whether to use FP16 precision, by default False.
    model_type : str, optional
        Class name of the underlying model, by default 'SongUNetPosEmbd'.
    sigma_data : float, optional
        Expected standard deviation of training data, by default 0.5.
    sigma_min : float, optional
        Minimum supported noise level, by default 0.0.
    sigma_max : float, optional
        Maximum supported noise level, by default inf.
    variance_channels : int, optional
        Number of channels for variance prediction. If None, uses img_out_channels.
        By default None.
    min_std : float, optional
        Minimum standard deviation to prevent collapse, by default 1e-4.
    **model_kwargs : dict
        Keyword arguments passed to the underlying model.

    Note
    ----
    The variance head hooks into the UNet decoder's level-0 features (typically
    128 channels) and applies a small CNN to predict spatially-varying uncertainty.
    This is decoupled from D_mean (not a monotonic function of the 1-ch output)
    while remaining lightweight (~37K additional parameters).

    References:
    - Karras, T., Aittala, M., Aila, T. and Laine, S., 2022. Elucidating the
      design space of diffusion-based generative models. NeurIPS 2022.
    - Gneiting, T. and Raftery, A.E., 2007. Strictly proper scoring rules,
      prediction, and estimation. JASA, 102(477), pp.359-378.
    """

    def __init__(
        self,
        img_resolution: Union[int, Tuple[int, int]],
        img_in_channels: int,
        img_out_channels: int,
        use_fp16: bool = False,
        model_type: Literal[
            "SongUNetPosEmbd", "SongUNetPosLtEmbd", "SongUNet", "DhariwalUNet"
        ] = "SongUNetPosEmbd",
        sigma_data: float = 0.5,
        sigma_min: float = 0.0,
        sigma_max: float = float("inf"),
        variance_channels: int = None,
        min_std: float = 1e-4,
        **model_kwargs: Any,
    ):
        # Initialize the parent class (this creates the main UNet)
        super().__init__(
            img_resolution=img_resolution,
            img_in_channels=img_in_channels,
            img_out_channels=img_out_channels,
            use_fp16=use_fp16,
            model_type=model_type,
            sigma_data=sigma_data,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            **model_kwargs,
        )

        # Store heteroscedastic-specific parameters
        self.variance_channels = variance_channels or img_out_channels
        self.min_std = min_std

        # Hook into UNet decoder to capture rich features (e.g. 128 channels)
        # instead of the 1-channel final output
        aux_conv_layer = None
        for name, module in self.model.dec.items():
            if name.endswith("_aux_conv"):
                aux_conv_layer = module  # last match = level 0 (full resolution)
        if aux_conv_layer is None:
            raise RuntimeError(
                "Could not find aux_conv in UNet decoder. "
                "The variance head requires access to decoder features."
            )

        unet_feature_channels = aux_conv_layer.in_channels  # typically 128

        self._captured_features = None

        def _capture_hook(module, input, output):
            self._captured_features = input[0]

        self._feature_hook_handle = aux_conv_layer.register_forward_hook(
            _capture_hook
        )

        self._unet_feature_channels = unet_feature_channels
        self._set_variance_head_mlp()

        # Update metadata
        self.meta = HeteroscedasticEDMPrecondSRMetaData

    def _set_variance_head_mlp(self) -> None:
        """Build the default multi-layer variance head."""
        hidden_channels = max(self._unet_feature_channels // 4, 16)
        self.variance_head = torch.nn.Sequential(
            torch.nn.Conv2d(
                self._unet_feature_channels, hidden_channels, kernel_size=3, padding=1
            ),
            torch.nn.SiLU(),
            torch.nn.Conv2d(hidden_channels, self.variance_channels, kernel_size=1),
        )
        # Initialize last layer for stable startup: softplus(-2) ≈ 0.13
        torch.nn.init.zeros_(self.variance_head[-1].weight)
        torch.nn.init.constant_(self.variance_head[-1].bias, -2.0)
        self._variance_head_input = "features"

    def _set_variance_head_legacy(self, in_channels: int | None = None) -> None:
        """Build a legacy single-layer variance head for old checkpoints."""
        if in_channels is None:
            in_channels = self._unet_feature_channels
        self.variance_head = torch.nn.Conv2d(
            in_channels, self.variance_channels, kernel_size=1
        )
        self._variance_head_input = (
            "features" if in_channels == self._unet_feature_channels else "d_mean"
        )

    def load_state_dict(self, state_dict, strict: bool = True):
        """Load both modern and legacy heteroscedastic checkpoints.

        Legacy checkpoints used a single convolutional `variance_head` with keys:
        `variance_head.weight` and `variance_head.bias`.
        """
        has_legacy_head = (
            "variance_head.weight" in state_dict
            and "variance_head.bias" in state_dict
        )
        has_mlp_head = any(key.startswith("variance_head.0.") for key in state_dict)

        if has_legacy_head and not has_mlp_head:
            legacy_in_channels = state_dict["variance_head.weight"].shape[1]
            needs_rebuild = isinstance(self.variance_head, torch.nn.Sequential) or (
                isinstance(self.variance_head, torch.nn.Conv2d)
                and self.variance_head.in_channels != legacy_in_channels
            )
            if needs_rebuild:
                self._set_variance_head_legacy(in_channels=legacy_in_channels)

        if has_mlp_head and isinstance(self.variance_head, torch.nn.Conv2d):
            self._set_variance_head_mlp()

        return super().load_state_dict(state_dict, strict=strict)

    def forward(
        self,
        x: torch.Tensor,
        img_lr: torch.Tensor,
        sigma: torch.Tensor,
        force_fp32: bool = False,
        return_variance: bool = True,
        **model_kwargs: Any,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass returning both denoised mean and predicted uncertainty.

        Parameters
        ----------
        x : torch.Tensor
            Noisy high-resolution image of shape (B, C_hr, H, W).
        img_lr : torch.Tensor
            Low-resolution conditioning image of shape (B, C_lr, H, W).
        sigma : torch.Tensor
            Noise level of shape (B) or (B, 1) or (B, 1, 1, 1).
        force_fp32 : bool, optional
            Whether to force FP32 precision, by default False.
        return_variance : bool, optional
            Whether to return variance prediction. If False, behaves like parent.
            By default True.
        **model_kwargs : dict
            Additional keyword arguments for the underlying model.

        Returns
        -------
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            If return_variance=True:
                Tuple of (D_mean, D_std) where:
                - D_mean: denoised prediction of shape (B, C_hr, H, W)
                - D_std: predicted std of shape (B, variance_channels, H, W)
            If return_variance=False:
                D_mean only (same as parent class)
        """
        if not return_variance:
            # Fall back to parent behavior (hook still fires, so clean up)
            result = super().forward(x, img_lr, sigma, force_fp32, **model_kwargs)
            self._captured_features = None
            return result

        # Preconditioning setup (same as parent)
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        dtype = (
            torch.float16
            if (self.use_fp16 and not force_fp32 and x.device.type == "cuda")
            else torch.float32
        )

        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.log() / 4

        # Prepare input
        if img_lr is None:
            arg = c_in * x
        else:
            arg = self.scaling_fn(x, img_lr, c_in)
        arg = arg.to(dtype)

        # Forward through UNet — the hook captures 128-ch decoder features
        F_x = self.model(
            arg,
            c_noise.flatten(),
            class_labels=None,
            **model_kwargs,
        )

        if (F_x.dtype != dtype) and not torch.is_autocast_enabled():
            raise ValueError(
                f"Expected the dtype to be {dtype}, but got {F_x.dtype} instead."
            )

        # Denoised mean (same as parent)
        D_mean = c_skip * x + c_out * F_x.to(torch.float32)

        # Variance prediction from rich UNet decoder features
        features = self._captured_features
        self._captured_features = None  # release reference
        if features is None:
            raise RuntimeError(
                "Variance head features were not captured from the UNet decoder."
            )

        variance_input_mode = getattr(self, "_variance_head_input", "features")
        if variance_input_mode == "features":
            variance_input = features.to(torch.float32)
        elif variance_input_mode == "d_mean":
            variance_input = D_mean.to(torch.float32)
        else:
            raise RuntimeError(f"Unknown variance head input mode: {variance_input_mode}")

        log_var = self.variance_head(variance_input)
        D_std = torch.nn.functional.softplus(log_var, beta=1.0) + self.min_std

        return D_mean, D_std


# NOTE: This is a deprecated version of the EDMPrecondSuperResolution model.
# This was used to maintain backwards compatibility and allow loading old models.
@dataclass
class EDMPrecondSRMetaData(ModelMetaData):
    """EDMPrecondSR meta data"""

    name: str = "EDMPrecondSR"
    # Optimization
    jit: bool = False
    cuda_graphs: bool = False
    amp_cpu: bool = False
    amp_gpu: bool = True
    torch_fx: bool = False
    # Data type
    bf16: bool = False
    # Inference
    onnx: bool = False
    # Physics informed
    func_torch: bool = False
    auto_grad: bool = False


class EDMPrecondSR(EDMPrecondSuperResolution):
    """
    NOTE: This is a deprecated version of the EDMPrecondSuperResolution model.
    This was used to maintain backwards compatibility and allow loading old models.
    Please use the EDMPrecondSuperResolution model instead.

    Improved preconditioning proposed in the paper "Elucidating the Design Space of
    Diffusion-Based Generative Models" (EDM) for super-resolution tasks

    Parameters
    ----------
    img_resolution : int
        Image resolution.
    img_channels : int
        Number of color channels (deprecated, not used).
    img_in_channels : int
        Number of input color channels.
    img_out_channels : int
        Number of output color channels.
    use_fp16 : bool
        Execute the underlying model at FP16 precision?, by default False.
    sigma_min : float
        Minimum supported noise level, by default 0.0.
    sigma_max : float
        Maximum supported noise level, by default inf.
    sigma_data : float
        Expected standard deviation of the training data, by default 0.5.
    model_type :str
        Class name of the underlying model, by default "SongUNetPosEmbd".
    scale_cond_input : bool
        Whether to scale the conditional input (deprecated), by default True.
    **model_kwargs : dict
        Keyword arguments for the underlying model.

    Note
    ----
    References:
    - Karras, T., Aittala, M., Aila, T. and Laine, S., 2022. Elucidating the
    design space of diffusion-based generative models. Advances in Neural Information
    Processing Systems, 35, pp.26565-26577.
    - Mardani, M., Brenowitz, N., Cohen, Y., Pathak, J., Chen, C.Y.,
    Liu, C.C.,Vahdat, A., Kashinath, K., Kautz, J. and Pritchard, M., 2023.
    Generative Residual Diffusion Modeling for Km-scale Atmospheric Downscaling.
    arXiv preprint arXiv:2309.15214.
    """

    def __init__(
        self,
        img_resolution,
        img_channels,  # deprecated
        img_in_channels,
        img_out_channels,
        use_fp16=False,
        sigma_min=0.0,
        sigma_max=float("inf"),
        sigma_data=0.5,
        model_type="SongUNetPosEmbd",
        scale_cond_input=True,  # deprecated
        **model_kwargs,
    ):
        warnings.warn(
            "EDMPrecondSR is deprecated and will be removed in a future version. "
            "Please use EDMPrecondSuperResolution instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        super().__init__(
            img_resolution=img_resolution,
            img_in_channels=img_in_channels,
            img_out_channels=img_out_channels,
            use_fp16=use_fp16,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            sigma_data=sigma_data,
            model_type=model_type,
            **model_kwargs,
        )

        if scale_cond_input:
            warnings.warn(
                "The `scale_cond_input=True` option does not properly scale the conditional input "
                "and is deprecated. It is highly recommended to set `scale_cond_input=False`. "
                "However, for loading a checkpoint previously trained with `scale_cond_input=True`, "
                "this flag must be set to `True` to ensure compatibility. "
                "For more details, see https://github.com/NVIDIA/modulus/issues/229.",
                DeprecationWarning,
            )
            self.scaling_fn = self._legacy_scaling_fn

        # Store deprecated parameters for backward compatibility
        self.img_channels = img_channels
        self.scale_cond_input = scale_cond_input

    @staticmethod
    def _legacy_scaling_fn(
        x: torch.Tensor, img_lr: torch.Tensor, c_in: torch.Tensor
    ) -> torch.Tensor:
        """
        This function does not properly scale the conditional input
        (see https://github.com/NVIDIA/modulus/issues/229)
        and will be deprecated.

        Concatenate and scale the high-resolution and low-resolution tensors.

        Parameters
        ----------
        x : torch.Tensor
            Noisy high-resolution image of shape (B, C_hr, H, W).
        img_lr : torch.Tensor
            Low-resolution image of shape (B, C_lr, H, W).
        c_in : torch.Tensor
            Scaling factor of shape (B, 1, 1, 1).

        Returns
        -------
        torch.Tensor
            Scaled and concatenated tensor of shape (B, C_in+C_out, H, W).
        """
        return c_in * torch.cat([x, img_lr.to(x.dtype)], dim=1)

    def forward(
        self,
        x,
        img_lr,
        sigma,
        force_fp32=False,
        **model_kwargs,
    ):
        """
        Forward pass of the EDMPrecondSR model wrapper.

        Parameters
        ----------
        x : torch.Tensor
            Noisy high-resolution image of shape (B, C_hr, H, W).
        img_lr : torch.Tensor
            Low-resolution conditioning image of shape (B, C_lr, H, W).
        sigma : torch.Tensor
            Noise level of shape (B) or (B, 1) or (B, 1, 1, 1).
        force_fp32 : bool, optional
            Whether to force FP32 precision regardless of the `use_fp16` attribute,
            by default False.
        **model_kwargs : dict
            Additional keyword arguments to pass to the underlying model.

        Returns
        -------
        torch.Tensor
            Denoised high-resolution image of shape (B, C_hr, H, W).
        """
        return super().forward(
            x=x, img_lr=img_lr, sigma=sigma, force_fp32=force_fp32, **model_kwargs
        )


class VEPrecond_dfsr(torch.nn.Module):
    """
    Preconditioning for dfsr model, modified from class VEPrecond, where the input
    argument 'sigma' in forward propagation function is used to receive the timestep
    of the backward diffusion process.

    Parameters
    ----------
    img_resolution : int
        Image resolution.
    img_channels : int
        Number of color channels.
    label_dim : int
        Number of class labels, 0 = unconditional, by default 0.
    use_fp16 : bool
        Execute the underlying model at FP16 precision?, by default False.
    sigma_min : float
        Minimum supported noise level, by default 0.02.
    sigma_max : float
        Maximum supported noise level, by default 100.0.
    model_type :str
        Class name of the underlying model, by default "SongUNet".
    **model_kwargs : dict
        Keyword arguments for the underlying model.

    Note
    ----
    Reference: Ho J, Jain A, Abbeel P. Denoising diffusion probabilistic models.
    Advances in neural information processing systems. 2020;33:6840-51.
    """

    def __init__(
        self,
        img_resolution: int,
        img_channels: int,
        label_dim: int = 0,
        use_fp16: bool = False,
        sigma_min: float = 0.02,
        sigma_max: float = 100.0,
        dataset_mean: float = 5.85e-05,
        dataset_scale: float = 4.79,
        model_type: str = "SongUNet",
        **model_kwargs: dict,
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        model_class = getattr(network_module, model_type)
        self.model = model_class(
            img_resolution=img_resolution,
            in_channels=self.img_channels,
            out_channels=img_channels,
            label_dim=label_dim,
            **model_kwargs,
        )  # TODO needs better handling

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        # print("sigma: ", sigma)
        class_labels = (
            None
            if self.label_dim == 0
            else torch.zeros([1, self.label_dim], device=x.device)
            if class_labels is None
            else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        )
        dtype = (
            torch.float16
            if (self.use_fp16 and not force_fp32 and x.device.type == "cuda")
            else torch.float32
        )

        c_in = 1
        c_noise = sigma  # Change the definitation of c_noise to avoid -inf values for zero sigma

        F_x = self.model(
            (c_in * x).to(dtype),
            c_noise.flatten(),
            class_labels=class_labels,
            **model_kwargs,
        )

        if F_x.dtype != dtype:
            raise ValueError(
                f"Expected the dtype to be {dtype}, but got {F_x.dtype} instead."
            )

        return F_x


class VEPrecond_dfsr_cond(torch.nn.Module):
    """
    Preconditioning for dfsr model with physics-informed conditioning input, modified
    from class VEPrecond, where the input argument 'sigma' in forward propagation function
    is used to receive the timestep of the backward diffusion process. The gradient of PDE
    residual with respect to the vorticity in the governing Navier-Stokes equation is computed
    as the physics-informed conditioning variable and is combined with the backward diffusion
    timestep before being sent to the underlying model for noise prediction.

    Parameters
    ----------
    img_resolution : int
        Image resolution.
    img_channels : int
        Number of color channels.
    label_dim : int
        Number of class labels, 0 = unconditional, by default 0.
    use_fp16 : bool
        Execute the underlying model at FP16 precision?, by default False.
    sigma_min : float
        Minimum supported noise level, by default 0.02.
    sigma_max : float
        Maximum supported noise level, by default 100.0.
    model_type :str
        Class name of the underlying model, by default "SongUNet".
    **model_kwargs : dict
        Keyword arguments for the underlying model.

    Note
    ----
    Reference:
    [1] Song, Y., Sohl-Dickstein, J., Kingma, D.P., Kumar, A., Ermon, S. and
    Poole, B., 2020. Score-based generative modeling through stochastic differential
    equations. arXiv preprint arXiv:2011.13456.
    [2] Shu D, Li Z, Farimani AB. A physics-informed diffusion model for high-fidelity
    flow field reconstruction. Journal of Computational Physics. 2023 Apr 1;478:111972.
    """

    def __init__(
        self,
        img_resolution: int,
        img_channels: int,
        label_dim: int = 0,
        use_fp16: bool = False,
        sigma_min: float = 0.02,
        sigma_max: float = 100.0,
        dataset_mean: float = 5.85e-05,
        dataset_scale: float = 4.79,
        model_type: str = "SongUNet",
        **model_kwargs: dict,
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        model_class = getattr(network_module, model_type)
        self.model = model_class(
            img_resolution=img_resolution,
            in_channels=model_kwargs["model_channels"] * 2,
            out_channels=img_channels,
            label_dim=label_dim,
            **model_kwargs,
        )  # TODO needs better handling

        # modules to embed residual loss
        self.conv_in = torch.nn.Conv2d(
            img_channels,
            model_kwargs["model_channels"],
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="circular",
        )
        self.emb_conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                img_channels,
                model_kwargs["model_channels"],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            torch.nn.GELU(),
            torch.nn.Conv2d(
                model_kwargs["model_channels"],
                model_kwargs["model_channels"],
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="circular",
            ),
        )
        self.dataset_mean = dataset_mean
        self.dataset_scale = dataset_scale

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = (
            None
            if self.label_dim == 0
            else torch.zeros([1, self.label_dim], device=x.device)
            if class_labels is None
            else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        )
        dtype = (
            torch.float16
            if (self.use_fp16 and not force_fp32 and x.device.type == "cuda")
            else torch.float32
        )

        c_in = 1
        c_noise = sigma

        # Compute physics-informed conditioning information using vorticity residual
        dx = (
            self.voriticity_residual((x * self.dataset_scale + self.dataset_mean))
            / self.dataset_scale
        )
        x = self.conv_in(x)
        cond_emb = self.emb_conv(dx)
        x = torch.cat((x, cond_emb), dim=1)

        F_x = self.model(
            (c_in * x).to(dtype),
            c_noise.flatten(),
            class_labels=class_labels,
            **model_kwargs,
        )

        if F_x.dtype != dtype:
            raise ValueError(
                f"Expected the dtype to be {dtype}, but got {F_x.dtype} instead."
            )
        return F_x

    def voriticity_residual(self, w, re=1000.0, dt=1 / 32):
        """
        Compute the gradient of PDE residual with respect to a given vorticity w using the
        spectrum method.

        Parameters
        ----------
        w: torch.Tensor
            The fluid flow data sample (vorticity).
        re: float
            The value of Reynolds number used in the governing Navier-Stokes equation.
        dt: float
            Time step used to compute the time-derivative of vorticity included in the governing
            Navier-Stokes equation.

        Returns
        -------
        torch.Tensor
            The computed vorticity gradient.
        """

        # w [b t h w]
        w = w.clone()
        w.requires_grad_(True)
        nx = w.size(2)
        device = w.device

        w_h = torch.fft.fft2(w[:, 1:-1], dim=[2, 3])
        # Wavenumbers in y-direction
        k_max = nx // 2
        N = nx
        k_x = (
            torch.cat(
                (
                    torch.arange(start=0, end=k_max, step=1, device=device),
                    torch.arange(start=-k_max, end=0, step=1, device=device),
                ),
                0,
            )
            .reshape(N, 1)
            .repeat(1, N)
            .reshape(1, 1, N, N)
        )
        k_y = (
            torch.cat(
                (
                    torch.arange(start=0, end=k_max, step=1, device=device),
                    torch.arange(start=-k_max, end=0, step=1, device=device),
                ),
                0,
            )
            .reshape(1, N)
            .repeat(N, 1)
            .reshape(1, 1, N, N)
        )
        # Negative Laplacian in Fourier space
        lap = k_x**2 + k_y**2
        lap[..., 0, 0] = 1.0
        psi_h = w_h / lap

        u_h = 1j * k_y * psi_h
        v_h = -1j * k_x * psi_h
        wx_h = 1j * k_x * w_h
        wy_h = 1j * k_y * w_h
        wlap_h = -lap * w_h

        u = torch.fft.irfft2(u_h[..., :, : k_max + 1], dim=[2, 3])
        v = torch.fft.irfft2(v_h[..., :, : k_max + 1], dim=[2, 3])
        wx = torch.fft.irfft2(wx_h[..., :, : k_max + 1], dim=[2, 3])
        wy = torch.fft.irfft2(wy_h[..., :, : k_max + 1], dim=[2, 3])
        wlap = torch.fft.irfft2(wlap_h[..., :, : k_max + 1], dim=[2, 3])
        advection = u * wx + v * wy

        wt = (w[:, 2:, :, :] - w[:, :-2, :, :]) / (2 * dt)

        # establish forcing term
        x = torch.linspace(0, 2 * np.pi, nx + 1, device=device)
        x = x[0:-1]
        X, Y = torch.meshgrid(x, x)
        f = -4 * torch.cos(4 * Y)

        residual = wt + (advection - (1.0 / re) * wlap + 0.1 * w[:, 1:-1]) - f
        residual_loss = (residual**2).mean()
        dw = torch.autograd.grad(residual_loss, w)[0]

        return dw
