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
Interpretability helpers for the CorrDiff diffusion model.

All functions operate on a frozen CorrDiff residual network (net_res) and
the paired regression network (net_reg). No training of CorrDiff occurs here.
The only 'trainable' component is the SAE in run_sae_training(), which is a
post-hoc probe trained on cached activations.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from torch import Tensor


# ---------------------------------------------------------------------------
# Sigma schedule (EDM Karras et al. 2022, Algorithm 2)
# ---------------------------------------------------------------------------

def make_sigma_schedule(
    net: nn.Module,
    num_steps: int,
    sigma_min: Optional[float] = None,
    sigma_max: Optional[float] = None,
    rho: float = 7.0,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Return the EDM sigma schedule as a 1-D tensor of length num_steps+1 (last=0)."""
    device = device or next(net.parameters()).device
    s_min = max(sigma_min or 0.002, net.sigma_min)
    s_max = min(sigma_max or 800, net.sigma_max)
    step_indices = torch.arange(num_steps, device=device)
    t_steps = (
        s_max ** (1 / rho)
        + step_indices / (num_steps - 1) * (s_min ** (1 / rho) - s_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])
    return t_steps


# ---------------------------------------------------------------------------
# Differentiable Euler sampler
# ---------------------------------------------------------------------------

def euler_sample(
    net: nn.Module,
    latents: Tensor,
    img_lr: Tensor,
    mean_hr: Optional[Tensor] = None,
    num_steps: int = 6,
    sigma_min: Optional[float] = None,
    sigma_max: Optional[float] = None,
    rho: float = 7.0,
    use_checkpoint: bool = True,
    class_labels: Optional[Tensor] = None,
    lead_time_label: Optional[Tensor] = None,
    trajectory_hook: Optional[Callable] = None,
) -> Tensor:
    """
    Backprop-safe Euler sampler (single forward call per step, no Heun correction).

    Gradients flow through the full sampling chain. Use gradient checkpointing
    (use_checkpoint=True) to keep memory at O(1 step) instead of O(num_steps).

    Parameters
    ----------
    net : nn.Module
        Diffusion model (EDMPrecondSuperResolution).
    latents : Tensor
        Initial noise sample, shape (B, C_out, H, W).
    img_lr : Tensor
        Conditioning input, shape (B, C_lr, H, W).
    mean_hr : Tensor, optional
        Regression mean, shape (B or 1, C_out, H, W). Concatenated with img_lr.
    num_steps : int
        Number of Euler steps (default 6, sufficient for attribution gradients).
    use_checkpoint : bool
        If True, use torch.utils.checkpoint to reduce peak memory.
    trajectory_hook : callable, optional
        Called at each step with (step, sigma, x_next, denoised).

    Returns
    -------
    Tensor
        Denoised sample, same shape as latents.
    """
    device = latents.device
    t_steps = make_sigma_schedule(net, num_steps, sigma_min, sigma_max, rho, device)

    # Build conditioning tensor [mean_hr, img_lr] (matching stochastic_sampler)
    x_lr = img_lr
    if mean_hr is not None:
        x_lr = torch.cat(
            (mean_hr.expand(img_lr.shape[0], -1, -1, -1), img_lr), dim=1
        )

    optional = {}
    if lead_time_label is not None:
        optional["lead_time_label"] = lead_time_label

    x_next = latents * t_steps[0]

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):

        def _euler_step(x_in, t_in, t_out):
            out = net(x_in, x_lr, t_in, class_labels, **optional)
            if isinstance(out, tuple):
                out = out[0]
            d = (x_in - out) / t_in
            return x_in + (t_out - t_in) * d

        if use_checkpoint and x_next.requires_grad:
            x_next = cp.checkpoint(
                _euler_step, x_next, t_cur, t_next, use_reentrant=False
            )
        else:
            x_next = _euler_step(x_next, t_cur, t_next)

        if trajectory_hook is not None:
            trajectory_hook(step=i, sigma=float(t_cur), x_next=x_next.detach())

    return x_next


# ---------------------------------------------------------------------------
# Ensemble generation (differentiable or not)
# ---------------------------------------------------------------------------

def run_ensemble(
    net: nn.Module,
    img_lr: Tensor,
    mean_hr: Optional[Tensor] = None,
    n_members: int = 10,
    num_steps: int = 6,
    seeds: Optional[List[int]] = None,
    lead_time_label: Optional[Tensor] = None,
    differentiable: bool = False,
    use_checkpoint: bool = True,
    device: Optional[torch.device] = None,
) -> Tuple[List[Tensor], List[Tensor]]:
    """
    Generate an ensemble of samples.

    Returns
    -------
    preds : list of Tensor, each (B, C_out, H, W)
        One tensor per ensemble member.
    latent_list : list of Tensor
        Fixed latents for each member (re-use for gradient evaluations).
    """
    device = device or img_lr.device
    if seeds is None:
        seeds = list(range(n_members))

    C_out = 1  # precipitation
    H, W = img_lr.shape[-2], img_lr.shape[-1]

    preds, latent_list = [], []
    ctx = torch.enable_grad if differentiable else torch.no_grad
    with ctx():
        for seed in seeds:
            g = torch.Generator(device=device)
            g.manual_seed(seed)
            latents = torch.randn(
                (img_lr.shape[0], C_out, H, W), generator=g, device=device
            )
            latent_list.append(latents.detach())
            if differentiable:
                y = euler_sample(
                    net, latents, img_lr, mean_hr,
                    num_steps=num_steps, use_checkpoint=use_checkpoint,
                    lead_time_label=lead_time_label,
                )
            else:
                with torch.no_grad():
                    y = euler_sample(
                        net, latents, img_lr, mean_hr,
                        num_steps=num_steps, use_checkpoint=False,
                        lead_time_label=lead_time_label,
                    )
            preds.append(y)
    return preds, latent_list


def ensemble_mean_and_std(preds: List[Tensor]) -> Tuple[Tensor, Tensor]:
    """Return per-pixel mean and std across ensemble members."""
    stack = torch.stack(preds, dim=0)  # (N, B, C, H, W)
    return stack.mean(0), stack.std(0)


# ---------------------------------------------------------------------------
# σ-level attribution (forward-only, one UNet call per sigma)
# ---------------------------------------------------------------------------

def compute_sigma_saliency(
    net: nn.Module,
    img_lr: Tensor,
    mean_hr: Optional[Tensor] = None,
    sigma: float = 1.0,
    x_noise: Optional[Tensor] = None,
    lead_time_label: Optional[Tensor] = None,
    target_fn: Optional[Callable] = None,
) -> Tensor:
    """
    Gradient-based saliency of a single denoiser call D(x_noise, sigma) w.r.t. img_lr.

    This is NOT a full sampling loop — it probes the denoiser at one noise level.
    Cheap fast approximation for D1.

    Parameters
    ----------
    sigma : float
        Noise level at which to evaluate the denoiser.
    x_noise : Tensor, optional
        Noisy input. If None, uses zero (denoiser call at pure conditioning mode).
    target_fn : callable, optional
        Maps denoised output → scalar. Default: spatial mean.

    Returns
    -------
    Tensor
        Gradient of target w.r.t. img_lr, same shape as img_lr.
    """
    device = img_lr.device
    img_lr_in = img_lr.detach().requires_grad_(True)

    x_lr = img_lr_in
    if mean_hr is not None:
        x_lr = torch.cat(
            (mean_hr.expand(img_lr_in.shape[0], -1, -1, -1), img_lr_in), dim=1
        )

    if x_noise is None:
        x_noise = torch.zeros(
            img_lr_in.shape[0], 1, img_lr_in.shape[2], img_lr_in.shape[3],
            device=device
        )

    t = torch.tensor([sigma], device=device)
    optional = {}
    if lead_time_label is not None:
        optional["lead_time_label"] = lead_time_label

    out = net(x_noise, x_lr, t, None, **optional)
    if isinstance(out, tuple):
        out = out[0]

    if target_fn is None:
        loss = out.mean()
    else:
        loss = target_fn(out)

    loss.backward()
    return img_lr_in.grad.detach()


def compute_ig_at_sigma(
    net: nn.Module,
    img_lr: Tensor,
    baseline: Tensor,
    mean_hr: Optional[Tensor] = None,
    sigma: float = 1.0,
    n_steps: int = 20,
    lead_time_label: Optional[Tensor] = None,
    target_fn: Optional[Callable] = None,
) -> Tensor:
    """
    Integrated gradients of D(·, sigma) w.r.t. img_lr, from baseline to img_lr.

    Returns per-channel IG attribution, shape (B, C_lr, H, W).
    """
    device = img_lr.device
    alphas = torch.linspace(0, 1, n_steps, device=device)
    ig = torch.zeros_like(img_lr)

    for alpha in alphas:
        interp = baseline + alpha * (img_lr - baseline)
        grad = compute_sigma_saliency(
            net, interp, mean_hr=mean_hr, sigma=sigma,
            lead_time_label=lead_time_label, target_fn=target_fn,
        )
        ig += grad

    ig = ig / n_steps * (img_lr - baseline)
    return ig


# ---------------------------------------------------------------------------
# Sparse Autoencoder (post-hoc probe on frozen activations)
# ---------------------------------------------------------------------------

class TopKSAE(nn.Module):
    """
    TopK sparse autoencoder for mechanistic feature discovery.

    Trained on cached activations of a frozen model.
    """

    def __init__(self, input_dim: int, dict_size: int, k: int = 32):
        super().__init__()
        self.k = k
        self.W_enc = nn.Parameter(torch.empty(input_dim, dict_size))
        self.W_dec = nn.Parameter(torch.empty(dict_size, input_dim))
        self.b_enc = nn.Parameter(torch.zeros(dict_size))
        self.b_dec = nn.Parameter(torch.zeros(input_dim))
        nn.init.kaiming_uniform_(self.W_enc)
        self.W_dec.data = self.W_enc.data.T.clone()

    def encode(self, x: Tensor) -> Tensor:
        """Return sparse feature activations (topk mask applied)."""
        pre = x @ self.W_enc + self.b_enc
        topk_vals, topk_idx = pre.topk(self.k, dim=-1)
        acts = torch.zeros_like(pre)
        acts.scatter_(-1, topk_idx, topk_vals.relu())
        return acts

    def decode(self, acts: Tensor) -> Tensor:
        return acts @ self.W_dec + self.b_dec

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Returns (reconstruction, activations)."""
        acts = self.encode(x)
        recon = self.decode(acts)
        return recon, acts


def run_sae_training(
    activations: Tensor,
    dict_size: int,
    k: int = 32,
    n_epochs: int = 50,
    lr: float = 2e-3,
    device: Optional[torch.device] = None,
) -> TopKSAE:
    """
    Train a TopK SAE on a matrix of cached activations.

    Parameters
    ----------
    activations : Tensor
        Shape (N, D) — N activation vectors of dimension D.
        Collected from a frozen model; CorrDiff is NOT retrained.
    dict_size : int
        Number of dictionary features (typically 4× D).
    k : int
        TopK sparsity target (L0 ≈ k per token).
    n_epochs : int
        Training epochs over the activation cache.

    Returns
    -------
    TopKSAE
        Trained SAE. Evaluate reconstruction quality with
        (recon - acts).pow(2).mean() — target < 0.1.
    """
    device = device or activations.device
    activations = activations.to(device).float()
    D = activations.shape[-1]

    sae = TopKSAE(D, dict_size, k).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)

    for epoch in range(n_epochs):
        perm = torch.randperm(len(activations), device=device)
        total_loss = 0.0
        for i in range(0, len(activations), 256):
            batch = activations[perm[i : i + 256]]
            opt.zero_grad()
            recon, acts = sae(batch)
            loss = (recon - batch).pow(2).mean()
            loss.backward()
            with torch.no_grad():
                sae.W_dec.data = nn.functional.normalize(sae.W_dec.data, dim=1)
            opt.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"  SAE epoch {epoch+1}/{n_epochs}  loss={total_loss:.4f}")

    return sae


# ---------------------------------------------------------------------------
# Activation capture context manager (generic, from regression notebook pattern)
# ---------------------------------------------------------------------------

class ActivationCapture:
    """
    Captures forward-hook activations from named sub-modules.

    Usage::

        with ActivationCapture(net, ["enc.32x32_block0", "dec.16x16_block0"]) as cap:
            net(...)
        acts = cap.activations  # dict of layer_name -> Tensor
    """

    def __init__(self, net: nn.Module, layer_keys: List[str]):
        self.net = net
        self.layer_keys = layer_keys
        self.activations: Dict[str, Tensor] = {}
        self._handles = []

    def _resolve(self, key: str) -> nn.Module:
        parts = key.split(".")
        m = self.net
        for p in parts:
            m = m[p] if isinstance(m, nn.ModuleDict) else getattr(m, p)
        return m

    def __enter__(self):
        self.activations = {}
        for key in self.layer_keys:
            try:
                layer = self._resolve(key)
            except (KeyError, AttributeError) as e:
                raise AttributeError(f"Layer '{key}' not found in model: {e}")

            def make_hook(k):
                def hook(module, inp, out):
                    if isinstance(out, tuple):
                        out = out[0]
                    self.activations[k] = out
                return hook

            self._handles.append(layer.register_forward_hook(make_hook(key)))
        return self

    def __exit__(self, *args):
        for h in self._handles:
            h.remove()
        self._handles = []


# ---------------------------------------------------------------------------
# Activation patching across sampler steps
# ---------------------------------------------------------------------------

def activation_patch_euler_sample(
    net: nn.Module,
    latents: Tensor,
    img_lr: Tensor,
    patch_acts: Tensor,
    patch_layer_key: str,
    patch_step: int,
    mean_hr: Optional[Tensor] = None,
    num_steps: int = 6,
    sigma_min: Optional[float] = None,
    sigma_max: Optional[float] = None,
    rho: float = 7.0,
    lead_time_label: Optional[Tensor] = None,
) -> Tensor:
    """
    Run the Euler sampler, replacing activations at `patch_layer_key` with
    `patch_acts` during step `patch_step` only.

    Used for D4: measures how much information the denoiser has locked in at
    each stage of the trajectory.

    Parameters
    ----------
    patch_acts : Tensor
        Replacement activations to inject (must match the layer output shape).
    patch_layer_key : str
        Dotted module path, e.g. "model.dec.16x16_block0".
    patch_step : int
        Which denoising step (0-indexed) to inject the patch.

    Returns
    -------
    Tensor
        Final sample with patched activations.
    """
    device = latents.device
    t_steps = make_sigma_schedule(net, num_steps, sigma_min, sigma_max, rho, device)

    x_lr = img_lr
    if mean_hr is not None:
        x_lr = torch.cat(
            (mean_hr.expand(img_lr.shape[0], -1, -1, -1), img_lr), dim=1
        )

    optional = {}
    if lead_time_label is not None:
        optional["lead_time_label"] = lead_time_label

    # Resolve patch layer
    parts = patch_layer_key.split(".")
    layer = net
    for p in parts:
        layer = layer[p] if isinstance(layer, nn.ModuleDict) else getattr(layer, p)

    x_next = latents * t_steps[0]

    with torch.no_grad():
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):

            if i == patch_step:
                handle = layer.register_forward_hook(
                    lambda m, inp, out: patch_acts
                )

            out = net(x_next, x_lr, t_cur, None, **optional)
            if isinstance(out, tuple):
                out = out[0]

            if i == patch_step:
                handle.remove()

            d = (x_next - out) / t_cur
            x_next = x_next + (t_next - t_cur) * d

    return x_next
