from contextlib import contextmanager
from typing import Optional

import torch
from torch import Tensor


def proper_ensemble_crps(pred_ens: Tensor, target: Tensor) -> Tensor:
    """Finite-ensemble CRPS for tensors shaped (B, M, C, H, W)."""
    if pred_ens.ndim != 5:
        raise ValueError(f"pred_ens must be 5D (B, M, C, H, W), got {pred_ens.shape}")
    if target.ndim != 4:
        raise ValueError(f"target must be 4D (B, C, H, W), got {target.shape}")
    if pred_ens.shape[0] != target.shape[0] or pred_ens.shape[2:] != target.shape[1:]:
        raise ValueError(
            f"Shape mismatch between pred_ens {pred_ens.shape} and target {target.shape}"
        )

    term_obs = torch.abs(pred_ens - target.unsqueeze(1)).mean(dim=1)
    if pred_ens.shape[1] == 1:
        return term_obs

    pairwise = torch.abs(pred_ens.unsqueeze(1) - pred_ens.unsqueeze(2)).mean(dim=(1, 2))
    return term_obs - 0.5 * pairwise


@contextmanager
def enable_dropout_only(module: torch.nn.Module):
    """Temporarily activate dropout layers while preserving all other modes.

    PhysicsNeMo's diffusion UNet blocks use functional dropout controlled by
    the block's ``training`` flag rather than ``nn.Dropout`` modules, so we also
    enable modules with a positive numeric ``dropout`` attribute.
    """
    states = []
    for child in module.modules():
        has_dropout_module = isinstance(child, torch.nn.modules.dropout._DropoutNd)
        dropout_prob = getattr(child, "dropout", 0.0)
        has_functional_dropout = isinstance(dropout_prob, (float, int)) and dropout_prob > 0.0
        if has_dropout_module or has_functional_dropout:
            states.append((child, child.training))
            child.train(True)
    try:
        yield
    finally:
        for child, was_training in states:
            child.train(was_training)


class DropoutResidualCRPSLoss:
    """CRPS loss for a one-step MC-dropout residual model."""

    def __init__(
        self,
        regression_net: torch.nn.Module,
        ensemble_size: int,
        hr_mean_conditioning: bool = True,
        residual_mae_weight: float = 0.0,
    ):
        if ensemble_size < 1:
            raise ValueError(f"ensemble_size must be >= 1, got {ensemble_size}")
        self.regression_net = regression_net
        self.ensemble_size = int(ensemble_size)
        self.hr_mean_conditioning = hr_mean_conditioning
        self.residual_mae_weight = float(residual_mae_weight)
        self.latest_components = {}

    def _regression_mean(
        self,
        img_clean: Tensor,
        img_lr: Tensor,
        lead_time_label: Optional[Tensor],
    ) -> Tensor:
        zeros = torch.zeros_like(img_clean)
        kwargs = {"lead_time_label": lead_time_label} if lead_time_label is not None else {}
        with torch.no_grad():
            return self.regression_net(x=zeros, img_lr=img_lr, **kwargs)

    def __call__(
        self,
        net: torch.nn.Module,
        img_clean: Tensor,
        img_lr: Tensor,
        lead_time_label: Optional[Tensor] = None,
        augment_pipe=None,
        **_,
    ) -> Tensor:
        if augment_pipe is not None:
            raise NotImplementedError("DropoutResidualCRPSLoss does not support augment_pipe")
        if img_clean.shape[0] != img_lr.shape[0] or img_clean.shape[2:] != img_lr.shape[2:]:
            raise ValueError(
                f"Shape mismatch between img_clean {img_clean.shape} and img_lr {img_lr.shape}"
            )

        batch_size = img_clean.shape[0]
        ensemble_size = self.ensemble_size
        y_mean = self._regression_mean(img_clean, img_lr, lead_time_label)
        cond = torch.cat((y_mean, img_lr), dim=1) if self.hr_mean_conditioning else img_lr

        cond_rep = cond.repeat_interleave(ensemble_size, dim=0)
        residual_input = torch.zeros(
            batch_size * ensemble_size,
            img_clean.shape[1],
            img_clean.shape[2],
            img_clean.shape[3],
            device=img_clean.device,
            dtype=img_clean.dtype,
        )
        residual_input = residual_input.to(memory_format=torch.channels_last)

        kwargs = {}
        if lead_time_label is not None:
            kwargs["lead_time_label"] = lead_time_label.repeat_interleave(
                ensemble_size, dim=0
            )

        with enable_dropout_only(net):
            residual = net(x=residual_input, img_lr=cond_rep, **kwargs)

        residual = residual.reshape(
            batch_size, ensemble_size, img_clean.shape[1], img_clean.shape[2], img_clean.shape[3]
        )
        pred_ens = y_mean.unsqueeze(1) + residual
        crps = proper_ensemble_crps(pred_ens, img_clean)
        loss = crps.mean(dim=(1, 2, 3))

        if self.residual_mae_weight:
            target_residual = img_clean - y_mean
            residual_mae = torch.abs(residual - target_residual.unsqueeze(1)).mean(
                dim=(1, 2, 3, 4)
            )
            loss = loss + self.residual_mae_weight * residual_mae
        else:
            residual_mae = torch.zeros_like(loss)

        with torch.no_grad():
            if ensemble_size > 1:
                spread_term = 0.5 * torch.abs(
                    pred_ens.unsqueeze(1) - pred_ens.unsqueeze(2)
                ).mean()
            else:
                spread_term = torch.zeros((), device=img_clean.device, dtype=img_clean.dtype)
            self.latest_components = {
                "loss_crps_mean": loss.mean().detach(),
                "dropout_skill_mean": torch.abs(pred_ens - img_clean.unsqueeze(1)).mean().detach(),
                "dropout_spread_term_mean": spread_term.detach(),
                "dropout_residual_mae_mean": residual_mae.mean().detach(),
                "dropout_ensemble_size": torch.tensor(
                    float(ensemble_size), device=img_clean.device
                ),
            }
        return loss


def dropout_residual_step(
    net: torch.nn.Module,
    img_lr: Tensor,
    latents_shape,
    mean_hr: Optional[Tensor] = None,
    lead_time_label: Optional[Tensor] = None,
    seed: Optional[int] = None,
) -> Tensor:
    """Generate one-step residual ensemble members with MC dropout enabled."""
    if img_lr.shape[0] != latents_shape[0]:
        if img_lr.shape[0] != 1:
            raise ValueError(
                f"img_lr batch must be 1 or match ensemble size, got {img_lr.shape[0]}"
            )
        img_lr = img_lr.expand(latents_shape[0], -1, -1, -1)

    cond = img_lr
    if mean_hr is not None:
        if mean_hr.shape[0] == 1:
            mean_hr = mean_hr.expand(latents_shape[0], -1, -1, -1)
        elif mean_hr.shape[0] != latents_shape[0]:
            raise ValueError(
                f"mean_hr batch must be 1 or match ensemble size, got {mean_hr.shape[0]}"
            )
        cond = torch.cat((mean_hr, img_lr), dim=1)

    x = torch.zeros(latents_shape, dtype=cond.dtype, device=cond.device)
    x = x.to(memory_format=torch.channels_last)
    cond = cond.to(memory_format=torch.channels_last)

    kwargs = {}
    if lead_time_label is not None:
        if lead_time_label.shape[0] == 1:
            lead_time_label = lead_time_label.expand(latents_shape[0], -1)
        kwargs["lead_time_label"] = lead_time_label

    rng_devices = [cond.device] if cond.device.type == "cuda" else []
    with torch.random.fork_rng(devices=rng_devices, enabled=seed is not None):
        if seed is not None:
            torch.manual_seed(int(seed))
        with torch.inference_mode():
            with enable_dropout_only(net):
                return net(x=x, img_lr=cond, **kwargs)
