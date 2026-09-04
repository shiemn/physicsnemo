"""EDM training on full normalized targets, without a regression model."""

import torch


class DirectEDMLoss:
    """Full-domain Gaussian EDM loss with the same noise defaults as ResidualLoss."""

    def __init__(self, P_mean=0.0, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(
        self, net, img_clean, img_lr, augment_pipe=None,
        lead_time_label=None, patching=None, use_patch_grad_acc=False,
    ):
        if patching is not None or use_patch_grad_acc:
            raise ValueError("DirectEDMLoss supports full-domain training only")
        if img_clean.shape[0] != img_lr.shape[0] or img_clean.shape[2:] != img_lr.shape[2:]:
            raise ValueError("Target and conditioning must share batch and spatial dimensions")
        combined = torch.cat((img_clean, img_lr), dim=1)
        combined, augment_labels = (
            augment_pipe(combined) if augment_pipe is not None else (combined, None)
        )
        target = combined[:, :img_clean.shape[1]]
        conditioning = combined[:, img_clean.shape[1]:]
        sigma = (
            torch.randn((target.shape[0], 1, 1, 1), device=target.device)
            * self.P_std + self.P_mean
        ).exp()
        weight = (sigma.square() + self.sigma_data**2) / (sigma * self.sigma_data).square()
        noisy_target = target + torch.randn_like(target) * sigma
        kwargs = {"augment_labels": augment_labels}
        if lead_time_label is not None:
            kwargs["lead_time_label"] = lead_time_label
        denoised = net(noisy_target, conditioning, sigma, **kwargs)
        return weight * (denoised - target).square()
