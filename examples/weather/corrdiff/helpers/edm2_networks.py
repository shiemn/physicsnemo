# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from NVlabs/edm2 (training/networks_edm2.py)
# Original license: CC BY-NC-SA 4.0
# https://github.com/NVlabs/edm2
#
# Modifications for CorrDiff super-resolution use:
#   - Removed @persistence.persistent_class decorators (not needed outside EDM2 repo)
#   - Inlined torch_utils.misc.const_like() as a local helper
#   - Split img_channels into img_in_channels + img_out_channels for SR
#   - Removed class-label conditioning (not used in SR setting)

"""EDM2 magnitude-preserving U-Net architecture.

Reference:
    Karras et al., "Analyzing and Improving the Training Dynamics of
    Diffusion Models", NeurIPS 2024. https://arxiv.org/abs/2312.02696
"""

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Inline replacement for torch_utils.misc.const_like


def _const_like(ref, value):
    f = np.float32(value)
    return torch.tensor(f, dtype=ref.dtype, device=ref.device)


# ---------------------------------------------------------------------------
# Normalize given tensor to unit magnitude with respect to the given
# dimensions. Default = all dimensions except the first.


def normalize(x, dim=None, eps=1e-4):
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)


# ---------------------------------------------------------------------------
# Upsample or downsample the given tensor with the given filter,
# or keep it as is.


def resample(x, f=[1, 1], mode="keep"):
    if mode == "keep":
        return x
    f = np.float32(f)
    assert f.ndim == 1 and len(f) % 2 == 0
    pad = (len(f) - 1) // 2
    f = f / f.sum()
    f = np.outer(f, f)[np.newaxis, np.newaxis, :, :]
    f = _const_like(x, f)
    c = x.shape[1]
    if mode == "down":
        return torch.nn.functional.conv2d(
            x, f.tile([c, 1, 1, 1]), groups=c, stride=2, padding=(pad,)
        )
    assert mode == "up"
    return torch.nn.functional.conv_transpose2d(
        x, (f * 4).tile([c, 1, 1, 1]), groups=c, stride=2, padding=(pad,)
    )


# ---------------------------------------------------------------------------
# Magnitude-preserving SiLU (Equation 81).


def mp_silu(x):
    return torch.nn.functional.silu(x) / 0.596


# ---------------------------------------------------------------------------
# Magnitude-preserving sum (Equation 88).


def mp_sum(a, b, t=0.5):
    return a.lerp(b, t) / np.sqrt((1 - t) ** 2 + t ** 2)


# ---------------------------------------------------------------------------
# Magnitude-preserving concatenation (Equation 103).


def mp_cat(a, b, dim=1, t=0.5):
    Na = a.shape[dim]
    Nb = b.shape[dim]
    C = np.sqrt((Na + Nb) / ((1 - t) ** 2 + t ** 2))
    wa = C / np.sqrt(Na) * (1 - t)
    wb = C / np.sqrt(Nb) * t
    return torch.cat([wa * a, wb * b], dim=dim)


# ---------------------------------------------------------------------------
# Magnitude-preserving Fourier features (Equation 75).


class MPFourier(torch.nn.Module):
    def __init__(self, num_channels, bandwidth=1):
        super().__init__()
        self.register_buffer(
            "freqs", 2 * np.pi * torch.randn(num_channels) * bandwidth
        )
        self.register_buffer("phases", 2 * np.pi * torch.rand(num_channels))

    def forward(self, x):
        y = x.to(torch.float32)
        y = y.ger(self.freqs.to(torch.float32))
        y = y + self.phases.to(torch.float32)
        y = y.cos() * np.sqrt(2)
        return y.to(x.dtype)


# ---------------------------------------------------------------------------
# Magnitude-preserving convolution or fully-connected layer (Equation 47)
# with forced weight normalization (Equation 66).


class MPConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel):
        super().__init__()
        self.out_channels = out_channels
        self.weight = torch.nn.Parameter(
            torch.randn(out_channels, in_channels, *kernel)
        )

    def forward(self, x, gain=1):
        w = self.weight.to(torch.float32)
        if self.training:
            with torch.no_grad():
                self.weight.copy_(normalize(w))  # forced weight normalization
        w = normalize(w)  # traditional weight normalization
        w = w * (gain / np.sqrt(w[0].numel()))  # magnitude-preserving scaling
        w = w.to(x.dtype)
        if w.ndim == 2:
            return x @ w.t()
        assert w.ndim == 4
        return torch.nn.functional.conv2d(x, w, padding=(w.shape[-1] // 2,))


# ---------------------------------------------------------------------------
# U-Net encoder/decoder block with optional self-attention (Figure 21).


class Block(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        emb_channels,
        flavor="enc",
        resample_mode="keep",
        resample_filter=[1, 1],
        attention=False,
        channels_per_head=64,
        dropout=0,
        res_balance=0.3,
        attn_balance=0.3,
        clip_act=256,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.flavor = flavor
        self.resample_filter = resample_filter
        self.resample_mode = resample_mode
        self.num_heads = out_channels // channels_per_head if attention else 0
        self.dropout = dropout
        self.res_balance = res_balance
        self.attn_balance = attn_balance
        self.clip_act = clip_act
        self.emb_gain = torch.nn.Parameter(torch.zeros([]))
        self.conv_res0 = MPConv(
            out_channels if flavor == "enc" else in_channels,
            out_channels,
            kernel=[3, 3],
        )
        self.emb_linear = MPConv(emb_channels, out_channels, kernel=[])
        self.conv_res1 = MPConv(out_channels, out_channels, kernel=[3, 3])
        self.conv_skip = (
            MPConv(in_channels, out_channels, kernel=[1, 1])
            if in_channels != out_channels
            else None
        )
        self.attn_qkv = (
            MPConv(out_channels, out_channels * 3, kernel=[1, 1])
            if self.num_heads != 0
            else None
        )
        self.attn_proj = (
            MPConv(out_channels, out_channels, kernel=[1, 1])
            if self.num_heads != 0
            else None
        )

    def forward(self, x, emb):
        # Main branch.
        x = resample(x, f=self.resample_filter, mode=self.resample_mode)
        if self.flavor == "enc":
            if self.conv_skip is not None:
                x = self.conv_skip(x)
            x = normalize(x, dim=1)  # pixel norm

        # Residual branch.
        y = self.conv_res0(mp_silu(x))
        c = self.emb_linear(emb, gain=self.emb_gain) + 1
        y = mp_silu(y * c.unsqueeze(2).unsqueeze(3).to(y.dtype))
        if self.training and self.dropout != 0:
            y = torch.nn.functional.dropout(y, p=self.dropout)
        y = self.conv_res1(y)

        # Connect the branches.
        if self.flavor == "dec" and self.conv_skip is not None:
            x = self.conv_skip(x)
        x = mp_sum(x, y, t=self.res_balance)

        # Self-attention.
        if self.num_heads != 0:
            y = self.attn_qkv(x)
            y = y.reshape(
                y.shape[0], self.num_heads, -1, 3, y.shape[2] * y.shape[3]
            )
            q, k, v = normalize(y, dim=2).unbind(3)  # pixel norm & split
            w = torch.einsum(
                "nhcq,nhck->nhqk", q, k / np.sqrt(q.shape[2])
            ).softmax(dim=3)
            y = torch.einsum("nhqk,nhck->nhcq", w, v)
            y = self.attn_proj(y.reshape(*x.shape))
            x = mp_sum(x, y, t=self.attn_balance)

        # Clip activations.
        if self.clip_act is not None:
            x = x.clip_(-self.clip_act, self.clip_act)
        return x


# ---------------------------------------------------------------------------
# EDM2 U-Net adapted for super-resolution (Figure 21).
#
# Key difference from the original: img_channels is split into
# img_in_channels (LR conditioning + noisy HR) and img_out_channels (HR
# prediction). The +1 constant-ones channel trick from EDM2 is preserved.


class EDM2UNet(torch.nn.Module):
    """EDM2 magnitude-preserving U-Net for super-resolution.

    Parameters
    ----------
    img_resolution : int
        Spatial resolution used to name blocks and determine which levels
        have self-attention. For non-square images use ``min(H, W)``.
        Must be consistent with ``attn_resolutions``.
    img_in_channels : int
        Number of channels fed into the network: LR conditioning channels
        concatenated with noisy HR channels (handled by the preconditioner).
    img_out_channels : int
        Number of HR output channels to predict.
    model_channels : int, optional
        Base channel multiplier, by default 192.
    channel_mult : list of int, optional
        Per-resolution channel multipliers, by default [1, 2, 3, 4].
    channel_mult_noise : int or None, optional
        Multiplier for noise embedding width. None = use channel_mult[0].
    channel_mult_emb : int or None, optional
        Multiplier for final embedding width. None = use max(channel_mult).
    num_blocks : int, optional
        Residual blocks per resolution level, by default 3.
    attn_resolutions : list of int, optional
        Resolutions (in pixels) at which to apply self-attention,
        by default [16, 8].
    concat_balance : float, optional
        mp_cat balance for skip connections, by default 0.5.
    **block_kwargs
        Additional keyword arguments forwarded to every ``Block``.
    """

    def __init__(
        self,
        img_resolution,
        img_in_channels,
        img_out_channels,
        model_channels=192,
        channel_mult=[1, 2, 3, 4],
        channel_mult_noise=None,
        channel_mult_emb=None,
        num_blocks=3,
        attn_resolutions=[16, 8],
        concat_balance=0.5,
        **block_kwargs,
    ):
        super().__init__()
        cblock = [model_channels * x for x in channel_mult]
        cnoise = (
            model_channels * channel_mult_noise
            if channel_mult_noise is not None
            else cblock[0]
        )
        cemb = (
            model_channels * channel_mult_emb
            if channel_mult_emb is not None
            else max(cblock)
        )
        self.concat_balance = concat_balance
        self.out_gain = torch.nn.Parameter(torch.zeros([]))

        # Noise embedding only (no class labels for SR).
        self.emb_fourier = MPFourier(cnoise)
        self.emb_noise = MPConv(cnoise, cemb, kernel=[])

        # Encoder — initial cout includes the +1 constant-ones channel.
        self.enc = torch.nn.ModuleDict()
        cout = img_in_channels + 1  # +1 for the constant ones channel
        for level, channels in enumerate(cblock):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = channels
                self.enc[f"{res}x{res}_conv"] = MPConv(cin, cout, kernel=[3, 3])
            else:
                self.enc[f"{res}x{res}_down"] = Block(
                    cout,
                    cout,
                    cemb,
                    flavor="enc",
                    resample_mode="down",
                    **block_kwargs,
                )
            for idx in range(num_blocks):
                cin = cout
                cout = channels
                self.enc[f"{res}x{res}_block{idx}"] = Block(
                    cin,
                    cout,
                    cemb,
                    flavor="enc",
                    attention=(res in attn_resolutions),
                    **block_kwargs,
                )

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        skips = [block.out_channels for block in self.enc.values()]
        for level, channels in reversed(list(enumerate(cblock))):
            res = img_resolution >> level
            if level == len(cblock) - 1:
                self.dec[f"{res}x{res}_in0"] = Block(
                    cout, cout, cemb, flavor="dec", attention=True, **block_kwargs
                )
                self.dec[f"{res}x{res}_in1"] = Block(
                    cout, cout, cemb, flavor="dec", **block_kwargs
                )
            else:
                self.dec[f"{res}x{res}_up"] = Block(
                    cout, cout, cemb, flavor="dec", resample_mode="up", **block_kwargs
                )
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = channels
                self.dec[f"{res}x{res}_block{idx}"] = Block(
                    cin,
                    cout,
                    cemb,
                    flavor="dec",
                    attention=(res in attn_resolutions),
                    **block_kwargs,
                )
        self.out_conv = MPConv(cout, img_out_channels, kernel=[3, 3])

    def forward(self, x, noise_labels):
        """
        Parameters
        ----------
        x : torch.Tensor
            Concatenated [c_in * noisy_hr, lr_conditioning] of shape
            (B, img_in_channels, H, W). Scaling is applied by the preconditioner.
        noise_labels : torch.Tensor
            Per-sample noise level encoding, shape (B,). Typically c_noise = log(sigma)/4.

        Returns
        -------
        torch.Tensor
            Raw network output F_x of shape (B, img_out_channels, H, W).
            The preconditioner applies c_skip * x + c_out * F_x to get D_x.
        """
        # Noise embedding.
        emb = mp_silu(self.emb_noise(self.emb_fourier(noise_labels)))

        # Append constant ones channel (EDM2 trick, kept for SR).
        x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)

        # Encoder.
        skips = []
        for name, block in self.enc.items():
            x = block(x) if "conv" in name else block(x, emb)
            skips.append(x)

        # Decoder.
        for name, block in self.dec.items():
            if "block" in name:
                x = mp_cat(x, skips.pop(), t=self.concat_balance)
            x = block(x, emb)

        x = self.out_conv(x, gain=self.out_gain)
        return x
