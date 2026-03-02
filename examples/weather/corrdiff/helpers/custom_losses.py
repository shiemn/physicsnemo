from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn

from physicsnemo.utils.patching import RandomPatching2D

class IntensityResidualLoss:
    """

    Attributes
    ----------
    regression_net : torch.nn.Module
        The regression network used for computing residuals.
    P_mean : float
        Mean value for noise level computation.
    P_std : float
        Standard deviation for noise level computation.
    sigma_data : float
        Standard deviation for data weighting.
    hr_mean_conditioning : bool
        Flag indicating whether to use high-resolution mean for conditioning.

    Note
    ----
    Reference: Mardani, M., Brenowitz, N., Cohen, Y., Pathak, J., Chen, C.Y.,
    Liu, C.C., Vahdat, A., Kashinath, K., Kautz, J. and Pritchard, M., 2023.
    Generative Residual Diffusion Modeling for Km-scale Atmospheric
    Downscaling. arXiv preprint arXiv:2309.15214.
    """

    def __init__(
        self,
        regression_net: torch.nn.Module,
        P_mean: float = 0.0,
        P_std: float = 1.2,
        sigma_data: float = 0.5,
        hr_mean_conditioning: bool = False,
        average_intensity_weight: float = 0.1,
        maximum_intensity_weight: float = 0.5,
    ):
        """
        Arguments
        ----------
        regression_net : torch.nn.Module
            Pre-trained regression network used to compute residuals.
            Expected signature: `net(zero_input, y_lr,
            lead_time_label=lead_time_label, augment_labels=augment_labels)` or
            `net(zero_input, y_lr, augment_labels=augment_labels)`, where:
                zero_input (torch.Tensor): Zero tensor of shape (B, C_hr, H, W)
                y_lr (torch.Tensor): Low-resolution input of shape (B, C_lr, H, W)
                lead_time_label (torch.Tensor, optional): Optional lead time labels
                augment_labels (torch.Tensor, optional): Optional augmentation labels
            Returns:
                torch.Tensor: Predictions of shape (B, C_hr, H, W)

        P_mean : float, optional
            Mean value for noise level computation, by default 0.0.

        P_std : float, optional
            Standard deviation for noise level computation, by default 1.2.

        sigma_data : float, optional
            Standard deviation for data weighting, by default 0.5.

        hr_mean_conditioning : bool, optional
            Whether to use high-resolution mean for conditioning predicted, by default False.
            When True, the mean prediction from `regression_net` is channel-wise
            concatenated with `img_lr` for conditioning.

        average_intensity_weight : float, optional
            Weight for the average intensity loss (as a fraction of the weight of the "normal" loss), by default 0.1.

        maximum_intensity_weight : float, optional
            Weight for the maximum intensity loss (as a fraction of the weight of the "normal" loss), by default 0.5.
        """
        self.regression_net = regression_net
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.hr_mean_conditioning = hr_mean_conditioning
        self.y_mean = None
        self.average_intensity_weight = average_intensity_weight
        self.maximum_intensity_weight = maximum_intensity_weight

    def __call__(
        self,
        net: torch.nn.Module,
        img_clean: Tensor,
        img_lr: Tensor,
        patching: Optional[RandomPatching2D] = None,
        lead_time_label: Optional[Tensor] = None,
        augment_pipe: Optional[
            Callable[[Tensor], Tuple[Tensor, Optional[Tensor]]]
        ] = None,
        use_patch_grad_acc: bool = False,
    ) -> Tensor:
        """
        Calculate and return the loss for denoising score matching.

        This method computes a mixture loss that combines deterministic
        regression with denoising score matching. It first computes residuals
        using the regression network, then applies the diffusion process to
        these residuals.

        In addition to the standard denoising score matching loss, this method
        also supports optional patching for multi-diffusion. In this case, the spatial
        dimensions of the input are decomposed into `P` smaller patches of shape
        (H_patch, W_patch), that are grouped along the batch dimension, and the
        model is applied to each patch individually. In the following, if `patching`
        is not provided, then the input is not patched and `P=1` and `(H_patch,
        W_patch) = (H, W)`. When patching is used, the original non-patched conditioning is
        interpolated onto a spatial grid of shape `(H_patch, W_patch)` and channel-wise
        concatenated to the patched conditioning. This ensures that each patch
        maintains global information from the entire domain.

        The diffusion model `net` is expected to be conditioned on an input with
        `C_cond` channels, which should be:
            - `C_cond = C_lr` if `hr_mean_conditioning` is `False` and
              `patching` is None.
            - `C_cond = C_hr + C_lr` if `hr_mean_conditioning` is `True` and
              `patching` is None.
            - `C_cond = C_hr + 2*C_lr` if `hr_mean_conditioning` is `True` and
              `patching` is not None.
            - `C_cond = 2*C_lr` if `hr_mean_conditioning` is `False` and
              `patching` is not None.
        Additionally, `C_cond` should also include any embedding channels,
        such as positional embeddings or time embeddings.

        Note: this loss function does not apply any reduction.

        Parameters
        ----------
        net : torch.nn.Module
            The neural network model for the diffusion process.
            Expected signature: `net(latent, y_lr, sigma,
            embedding_selector=embedding_selector, lead_time_label=lead_time_label,
            augment_labels=augment_labels)`, where:
                latent (torch.Tensor): Noisy input of shape (B[*P], C_hr, H_patch, W_patch)
                y_lr (torch.Tensor): Conditioning of shape (B[*P], C_cond, H_patch, W_patch)
                sigma (torch.Tensor): Noise level of shape (B[*P], 1, 1, 1)
                embedding_selector (callable, optional): Function to select
                    positional embeddings. Only used if `patching` is provided.
                lead_time_label (torch.Tensor, optional): Lead time labels.
                augment_labels (torch.Tensor, optional): Augmentation labels
            Returns:
                torch.Tensor: Predictions of shape (B[*P], C_hr, H_patch, W_patch)

        img_clean : torch.Tensor
            High-resolution input images of shape (B, C_hr, H, W).
            Used as ground truth and for data augmentation if 'augment_pipe' is provided.

        img_lr : torch.Tensor
            Low-resolution input images of shape (B, C_lr, H, W).
            Used as input to the regression network and conditioning for the
            diffusion process.

        patching : Optional[RandomPatching2D], optional
            Patching strategy for processing large images, by default None. See
            :class:`physicsnemo.utils.patching.RandomPatching2D` for details.
            When provided, the patching strategy is used for both image patches
            and positional embeddings selection in the diffusion model `net`.
            Transforms tensors from shape (B, C, H, W) to (B*P, C, H_patch,
            W_patch).

        lead_time_label : Optional[torch.Tensor], optional
            Labels for lead-time aware predictions, by default None.
            Shape can vary based on model requirements, typically (B,) or scalar.

        augment_pipe : Optional[Callable[[torch.Tensor], Tuple[torch.Tensor, Optional[torch.Tensor]]]]
            Data augmentation function.
            Expected signature:
                img_tot (torch.Tensor): Concatenated high and low resolution images
                    of shape (B, C_hr+C_lr, H, W)
            Returns:
                Tuple[torch.Tensor, Optional[torch.Tensor]]:
                    - Augmented images of shape (B, C_hr+C_lr, H, W)
                    - Optional augmentation labels
        use_patch_grad_acc: bool, optional
            A boolean flag indicating whether to enable multi-iterations of patching accumulations
            for amortizing regression cost. Default False.

        Returns
        -------
        torch.Tensor
            If patching is not used:
                A tensor of shape (B, C_hr, H, W) representing the per-sample loss.
            If patching is used:
                A tensor of shape (B*P, C_hr, H_patch, W_patch) representing
                the per-patch loss.

        Raises
        ------
        ValueError
            If patching is provided but is not an instance of RandomPatching2D.
            If shapes of img_clean and img_lr are incompatible.
        """

        # Safety check: enforce patching object
        if patching and not isinstance(patching, RandomPatching2D):
            raise ValueError("patching must be a 'RandomPatching2D' object.")
        # Safety check: enforce shapes
        if (
            img_clean.shape[0] != img_lr.shape[0]
            or img_clean.shape[2:] != img_lr.shape[2:]
        ):
            raise ValueError(
                f"Shape mismatch between img_clean {img_clean.shape} and "
                f"img_lr {img_lr.shape}. "
                f"Batch size, height and width must match."
            )

        # augment for conditional generation
        img_tot = torch.cat((img_clean, img_lr), dim=1)
        y_tot, augment_labels = (
            augment_pipe(img_tot) if augment_pipe is not None else (img_tot, None)
        )
        y = y_tot[:, : img_clean.shape[1], :, :]
        y_lr = y_tot[:, img_clean.shape[1] :, :, :]
        y_lr_res = y_lr
        batch_size = y.shape[0]

        # if using multi-iterations of patching, switch to optimized version
        if use_patch_grad_acc:
            # form residual
            if self.y_mean is None:
                if lead_time_label is not None:
                    y_mean = self.regression_net(
                        torch.zeros_like(y, device=img_clean.device),
                        y_lr_res,
                        lead_time_label=lead_time_label,
                        augment_labels=augment_labels,
                    )
                else:
                    y_mean = self.regression_net(
                        torch.zeros_like(y, device=img_clean.device),
                        y_lr_res,
                        augment_labels=augment_labels,
                    )
                self.y_mean = y_mean

        # if on full domain, or if using patching without multi-iterations
        else:
            # form residual
            if lead_time_label is not None:
                y_mean = self.regression_net(
                    torch.zeros_like(y, device=img_clean.device),
                    y_lr_res,
                    lead_time_label=lead_time_label,
                    augment_labels=augment_labels,
                )
            else:
                y_mean = self.regression_net(
                    torch.zeros_like(y, device=img_clean.device),
                    y_lr_res,
                    augment_labels=augment_labels,
                )

            self.y_mean = y_mean

        y = y - self.y_mean

        if self.hr_mean_conditioning:
            y_lr = torch.cat((self.y_mean, y_lr), dim=1)

        # patchified training
        # conditioning: cat(y_mean, y_lr, input_interp, pos_embd), 4+12+100+4
        # removed patch_embedding_selector due to compilation issue with dynamo.
        if patching:
            # Patched residual
            # (batch_size * patch_num, c_out, patch_shape_y, patch_shape_x)
            y_patched = patching.apply(input=y)
            # Patched conditioning on y_lr and interp(img_lr)
            # (batch_size * patch_num, 2*c_in, patch_shape_y, patch_shape_x)
            y_lr_patched = patching.apply(input=y_lr, additional_input=img_lr)

            y = y_patched
            y_lr = y_lr_patched

        # Noise
        rnd_normal = torch.randn([y.shape[0], 1, 1, 1], device=img_clean.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2

        

        # Input + noise
        latent = y + torch.randn_like(y) * sigma

        if lead_time_label is not None:
            D_yn = net(
                latent,
                y_lr,
                sigma,
                embedding_selector=None,
                global_index=(
                    patching.global_index(batch_size, img_clean.device)
                    if patching is not None
                    else None
                ),
                lead_time_label=lead_time_label,
                augment_labels=augment_labels,
            )
        else:
            D_yn = net(
                latent,
                y_lr,
                sigma,
                embedding_selector=None,
                global_index=(
                    patching.global_index(batch_size, img_clean.device)
                    if patching is not None
                    else None
                ),
                augment_labels=augment_labels,
            )

        maximum_intensity = torch.max(y)
        maximum_intensity_fake = torch.max(D_yn)

        average_intensity = torch.mean(y)
        average_intensity_fake = torch.mean(D_yn)

        average_intensity_error = nn.functional.mse_loss(average_intensity, average_intensity_fake)
        maximum_intensity_error = nn.functional.mse_loss(maximum_intensity, maximum_intensity_fake)


        loss = weight * ((D_yn - y) ** 2) + self.average_intensity_weight * average_intensity_error + self.maximum_intensity_weight * maximum_intensity_error

        return loss


class GaussianCRPSLoss:
    """
    Closed-form CRPS (Continuous Ranked Probability Score) for Gaussian predictive
    distributions N(mu, sigma^2).

    For a Gaussian predictive distribution, CRPS has an analytical form:
        CRPS(mu, sigma, y) = sigma * [z * (2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi)]

    where:
        z = (y - mu) / sigma
        phi(z) = standard normal PDF
        Phi(z) = standard normal CDF

    This is a strictly proper scoring rule, meaning it is minimized when the
    predictive distribution matches the true data-generating distribution.

    Note
    ----
    Reference: Gneiting, T. and Raftery, A.E., 2007. Strictly proper scoring rules,
    prediction, and estimation. Journal of the American Statistical Association,
    102(477), pp.359-378.
    """

    def __init__(self, reduction: str = "none"):
        """
        Arguments
        ----------
        reduction : str, optional
            Reduction to apply: 'none', 'mean', or 'sum'. Default is 'none'.
        """
        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(f"reduction must be 'none', 'mean', or 'sum', got {reduction}")
        self.reduction = reduction

    def __call__(self, mu: Tensor, sigma: Tensor, y: Tensor) -> Tensor:
        """
        Compute the Gaussian CRPS.

        Parameters
        ----------
        mu : torch.Tensor
            Predicted mean, shape (B, C, H, W) or any broadcastable shape.
        sigma : torch.Tensor
            Predicted standard deviation (must be positive), same shape as mu.
        y : torch.Tensor
            Ground truth observations, same shape as mu.

        Returns
        -------
        torch.Tensor
            CRPS values. Shape depends on reduction:
            - 'none': same shape as input
            - 'mean' or 'sum': scalar
        """
        # Ensure numerical stability
        sigma = sigma.clamp(min=1e-6)

        # Standardized residual
        z = (y - mu) / sigma

        # Standard normal PDF: phi(z) = exp(-z^2/2) / sqrt(2*pi)
        sqrt_2 = 1.4142135623730951
        sqrt_pi = 1.7724538509055159
        sqrt_2_pi = 2.5066282746310002  # sqrt(2 * pi)

        phi = torch.exp(-0.5 * z ** 2) / sqrt_2_pi

        # Standard normal CDF: Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
        Phi = 0.5 * (1.0 + torch.erf(z / sqrt_2))

        # CRPS formula for Gaussian: sigma * [z * (2*Phi - 1) + 2*phi - 1/sqrt(pi)]
        crps = sigma * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / sqrt_pi)

        # Apply reduction
        if self.reduction == "mean":
            return crps.mean()
        elif self.reduction == "sum":
            return crps.sum()
        return crps


class LogisticCRPSLoss:
    """
    Closed-form CRPS for logistic predictive distributions Logistic(mu, s).

    The logistic distribution has heavier tails than the Gaussian, making it
    better suited for variables like precipitation where the residual
    distribution is heavy-tailed and potentially asymmetric in practice.

    For a logistic distribution with location mu and scale s:
        CRPS(mu, s, y) = (y - mu) + 2s * softplus(-(y - mu)/s) - s

    where softplus(x) = log(1 + exp(x)).

    The logistic has variance = s^2 * pi^2 / 3, so std = s * pi / sqrt(3).

    References
    ----------
    Jordan, A., Krueger, F. and Lerch, S., 2019. Evaluating probabilistic
    forecasts with scoringRules. Journal of Statistical Software, 90(12).
    """

    def __init__(self, reduction: str = "none"):
        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(
                f"reduction must be 'none', 'mean', or 'sum', got {reduction}"
            )
        self.reduction = reduction

    def __call__(self, mu: Tensor, scale: Tensor, y: Tensor) -> Tensor:
        """
        Compute the logistic CRPS.

        Parameters
        ----------
        mu : torch.Tensor
            Predicted location, shape (B, C, H, W).
        scale : torch.Tensor
            Predicted scale (must be positive), same shape as mu.
        y : torch.Tensor
            Ground truth observations, same shape as mu.

        Returns
        -------
        torch.Tensor
            CRPS values with specified reduction.
        """
        scale = scale.clamp(min=1e-6)
        z = (y - mu) / scale

        # CRPS = (y - mu) + 2s * softplus(-z) - s
        # F.softplus is numerically stable for all z
        crps = (y - mu) + 2.0 * scale * torch.nn.functional.softplus(-z) - scale

        if self.reduction == "mean":
            return crps.mean()
        elif self.reduction == "sum":
            return crps.sum()
        return crps


class CalibratedResidualLoss:
    """
    Residual loss with Gaussian CRPS for uncertainty calibration.

    This loss combines the standard denoising score matching loss with a
    Gaussian CRPS term that trains the network to predict calibrated uncertainty.
    The network must output both a mean prediction and a standard deviation.

    Total loss:
        L = L_denoise + crps_weight * L_crps

    where:
        L_denoise = weight * (D_mean - y)^2  (standard EDM loss)
        L_crps = GaussianCRPS(D_mean, D_std, y)

    The predicted D_std can then be used during inference to modulate the
    stochastic sampler's noise injection, achieving calibrated ensemble spread.

    Attributes
    ----------
    regression_net : torch.nn.Module
        The regression network used for computing residuals.
    P_mean : float
        Mean value for noise level computation.
    P_std : float
        Standard deviation for noise level computation.
    sigma_data : float
        Standard deviation for data weighting.
    hr_mean_conditioning : bool
        Flag indicating whether to use high-resolution mean for conditioning.
    crps_weight : float
        Weight for the CRPS loss term relative to the denoising loss.

    Note
    ----
    Reference: Mardani, M., Brenowitz, N., Cohen, Y., Pathak, J., Chen, C.Y.,
    Liu, C.C., Vahdat, A., Kashinath, K., Kautz, J. and Pritchard, M., 2023.
    Generative Residual Diffusion Modeling for Km-scale Atmospheric
    Downscaling. arXiv preprint arXiv:2309.15214.
    """

    def __init__(
        self,
        regression_net: torch.nn.Module,
        P_mean: float = 0.0,
        P_std: float = 1.2,
        sigma_data: float = 0.5,
        hr_mean_conditioning: bool = False,
        crps_weight: float = 0.1,
        min_std: float = 1e-4,
    ):
        """
        Arguments
        ----------
        regression_net : torch.nn.Module
            Pre-trained regression network used to compute residuals.

        P_mean : float, optional
            Mean value for noise level computation, by default 0.0.

        P_std : float, optional
            Standard deviation for noise level computation, by default 1.2.

        sigma_data : float, optional
            Standard deviation for data weighting, by default 0.5.

        hr_mean_conditioning : bool, optional
            Whether to use high-resolution mean for conditioning, by default False.

        crps_weight : float, optional
            Weight for the CRPS loss term. Higher values encourage better
            uncertainty calibration at the cost of potentially worse mean
            predictions. Default is 0.1.

        min_std : float, optional
            Minimum standard deviation to prevent collapse to zero. Default is 1e-4.
        """
        self.regression_net = regression_net
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.hr_mean_conditioning = hr_mean_conditioning
        self.crps_weight = crps_weight
        self.min_std = min_std
        self.y_mean = None
        self.crps_loss = GaussianCRPSLoss(reduction="none")
        self.latest_components = {}

    def __call__(
        self,
        net: torch.nn.Module,
        img_clean: Tensor,
        img_lr: Tensor,
        patching: Optional[RandomPatching2D] = None,
        lead_time_label: Optional[Tensor] = None,
        augment_pipe: Optional[
            Callable[[Tensor], Tuple[Tensor, Optional[Tensor]]]
        ] = None,
        use_patch_grad_acc: bool = False,
    ) -> Tensor:
        """
        Calculate the combined denoising + CRPS loss.

        The network is expected to return a tuple (D_mean, D_std) where:
        - D_mean: denoised prediction (B, C, H, W)
        - D_std: predicted uncertainty/standard deviation (B, C, H, W)

        Parameters
        ----------
        net : torch.nn.Module
            The neural network model for the diffusion process.
            Must return (D_mean, D_std) tuple when called.

        img_clean : torch.Tensor
            High-resolution ground truth images of shape (B, C_hr, H, W).

        img_lr : torch.Tensor
            Low-resolution conditioning images of shape (B, C_lr, H, W).

        patching : Optional[RandomPatching2D], optional
            Patching strategy for processing large images, by default None.

        lead_time_label : Optional[torch.Tensor], optional
            Labels for lead-time aware predictions, by default None.

        augment_pipe : Optional[Callable], optional
            Data augmentation function, by default None.

        use_patch_grad_acc : bool, optional
            Whether to enable patch gradient accumulation, by default False.

        Returns
        -------
        torch.Tensor
            Combined loss tensor of shape (B, C, H, W) or (B*P, C, H_patch, W_patch).
        """
        # Safety checks
        if patching and not isinstance(patching, RandomPatching2D):
            raise ValueError("patching must be a 'RandomPatching2D' object.")
        if (
            img_clean.shape[0] != img_lr.shape[0]
            or img_clean.shape[2:] != img_lr.shape[2:]
        ):
            raise ValueError(
                f"Shape mismatch between img_clean {img_clean.shape} and "
                f"img_lr {img_lr.shape}. "
                f"Batch size, height and width must match."
            )

        # Augment for conditional generation
        img_tot = torch.cat((img_clean, img_lr), dim=1)
        y_tot, augment_labels = (
            augment_pipe(img_tot) if augment_pipe is not None else (img_tot, None)
        )
        y = y_tot[:, : img_clean.shape[1], :, :]
        y_lr = y_tot[:, img_clean.shape[1] :, :, :]
        y_lr_res = y_lr
        batch_size = y.shape[0]

        # Compute regression mean (residual computation)
        if use_patch_grad_acc:
            if self.y_mean is None:
                if lead_time_label is not None:
                    y_mean = self.regression_net(
                        torch.zeros_like(y, device=img_clean.device),
                        y_lr_res,
                        lead_time_label=lead_time_label,
                        augment_labels=augment_labels,
                    )
                else:
                    y_mean = self.regression_net(
                        torch.zeros_like(y, device=img_clean.device),
                        y_lr_res,
                        augment_labels=augment_labels,
                    )
                self.y_mean = y_mean
        else:
            if lead_time_label is not None:
                y_mean = self.regression_net(
                    torch.zeros_like(y, device=img_clean.device),
                    y_lr_res,
                    lead_time_label=lead_time_label,
                    augment_labels=augment_labels,
                )
            else:
                y_mean = self.regression_net(
                    torch.zeros_like(y, device=img_clean.device),
                    y_lr_res,
                    augment_labels=augment_labels,
                )
            self.y_mean = y_mean

        # Compute residual
        y = y - self.y_mean

        # HR mean conditioning
        if self.hr_mean_conditioning:
            y_lr = torch.cat((self.y_mean, y_lr), dim=1)

        # Patchified training
        if patching:
            y_patched = patching.apply(input=y)
            y_lr_patched = patching.apply(input=y_lr, additional_input=img_lr)
            y = y_patched
            y_lr = y_lr_patched

        # Sample noise level
        rnd_normal = torch.randn([y.shape[0], 1, 1, 1], device=img_clean.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

        # Add noise to input
        latent = y + torch.randn_like(y) * sigma

        # Forward pass - expect (D_mean, D_std) tuple
        if lead_time_label is not None:
            net_output = net(
                latent,
                y_lr,
                sigma,
                embedding_selector=None,
                global_index=(
                    patching.global_index(batch_size, img_clean.device)
                    if patching is not None
                    else None
                ),
                lead_time_label=lead_time_label,
                augment_labels=augment_labels,
            )
        else:
            net_output = net(
                latent,
                y_lr,
                sigma,
                embedding_selector=None,
                global_index=(
                    patching.global_index(batch_size, img_clean.device)
                    if patching is not None
                    else None
                ),
                augment_labels=augment_labels,
            )

        # Handle both heteroscedastic (tuple) and standard (tensor) outputs
        if isinstance(net_output, tuple):
            D_mean, D_std = net_output
            # Enforce minimum std to prevent collapse
            D_std = D_std.clamp(min=self.min_std)
        else:
            # Fallback for non-heteroscedastic networks
            D_mean = net_output
            D_std = None

        # Standard denoising loss
        loss_denoise = weight * ((D_mean - y) ** 2)

        # Add CRPS loss if we have uncertainty predictions
        # Scale CRPS by the same EDM weight so balance is constant across sigma.
        if D_std is not None:
            loss_crps = self.crps_loss(D_mean, D_std, y)
            loss = weight * (
                (D_mean - y) ** 2
                + self.crps_weight * loss_crps
            )
            self.latest_components = {
                "loss_denoise_mean": loss_denoise.detach().mean(),
                "loss_crps_mean": loss_crps.detach().mean(),
                "loss_crps_weighted_mean": (
                    (weight * self.crps_weight * loss_crps).detach().mean()
                ),
                "loss_total_mean": loss.detach().mean(),
                "pred_scale_mean": D_std.detach().mean(),
            }
        else:
            loss = loss_denoise
            self.latest_components = {
                "loss_denoise_mean": loss_denoise.detach().mean(),
                "loss_total_mean": loss.detach().mean(),
            }

        return loss


class ThresholdWeightedGaussianCRPSLoss:
    """
    Threshold-weighted CRPS (twCRPS) for Gaussian predictive distributions.

    Emphasizes scoring accuracy above a threshold t, making the loss sensitive
    to extreme events where standard CRPS has vanishing gradients w.r.t. sigma.

    For a Gaussian N(mu, sigma^2) and threshold t, the closed-form twCRPS is:

        twCRPS(mu, sigma, y; t) = CRPS(mu, sigma, max(y, t))
                                 - CRPS(mu, sigma, t)
                                 + |max(y, t) - t| * (1 - 2*Phi((t - mu)/sigma))

    This is a proper scoring rule (Gneiting & Ranjan, 2011).

    Parameters
    ----------
    threshold : float
        The threshold above which to emphasize scoring. Events below this
        threshold are effectively down-weighted.
    reduction : str
        Reduction to apply: 'none', 'mean', or 'sum'. Default is 'none'.

    References
    ----------
    Gneiting, T. and Ranjan, R., 2011. Comparing density forecasts using
    threshold- and quantile-weighted scoring rules. Journal of Business &
    Economic Statistics, 29(3), pp.411-422.
    """

    def __init__(self, threshold: float = 0.0, reduction: str = "none"):
        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(
                f"reduction must be 'none', 'mean', or 'sum', got {reduction}"
            )
        self.threshold = threshold
        self.reduction = reduction

    @staticmethod
    def _crps_gaussian(mu: Tensor, sigma: Tensor, y: Tensor) -> Tensor:
        """Raw per-element Gaussian CRPS (no reduction)."""
        sigma = sigma.clamp(min=1e-6)
        z = (y - mu) / sigma
        sqrt_2 = 1.4142135623730951
        sqrt_pi = 1.7724538509055159
        sqrt_2_pi = 2.5066282746310002
        phi = torch.exp(-0.5 * z**2) / sqrt_2_pi
        Phi = 0.5 * (1.0 + torch.erf(z / sqrt_2))
        return sigma * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / sqrt_pi)

    def __call__(self, mu: Tensor, sigma: Tensor, y: Tensor) -> Tensor:
        """
        Compute threshold-weighted Gaussian CRPS.

        Parameters
        ----------
        mu : Tensor
            Predicted mean, shape (B, C, H, W).
        sigma : Tensor
            Predicted std (positive), same shape as mu.
        y : Tensor
            Ground truth, same shape as mu.

        Returns
        -------
        Tensor
            twCRPS values with specified reduction.
        """
        t = self.threshold
        sigma = sigma.clamp(min=1e-6)

        y_clamp = torch.clamp(y, min=t)

        # twCRPS = CRPS(mu, sigma, max(y, t)) - CRPS(mu, sigma, t)
        #        + |max(y, t) - t| * (1 - 2*Phi((t - mu)/sigma))
        crps_y = self._crps_gaussian(mu, sigma, y_clamp)
        crps_t = self._crps_gaussian(mu, sigma, torch.full_like(y, t))

        sqrt_2 = 1.4142135623730951
        Phi_t = 0.5 * (1.0 + torch.erf((t - mu) / (sigma * sqrt_2)))
        correction = (y_clamp - t) * (1.0 - 2.0 * Phi_t)

        tw_crps = crps_y - crps_t + correction

        if self.reduction == "mean":
            return tw_crps.mean()
        elif self.reduction == "sum":
            return tw_crps.sum()
        return tw_crps


class ThresholdWeightedLogisticCRPSLoss:
    """
    Threshold-weighted CRPS (twCRPS) for logistic predictive distributions.

    Analogous to ThresholdWeightedGaussianCRPSLoss but using the logistic
    distribution, which has heavier tails better suited for precipitation.

    Uses the general twCRPS decomposition (valid for any distribution F):
        twCRPS(mu, s, y; t) = CRPS(mu, s, max(y,t)) - CRPS(mu, s, t)
                             + |max(y,t) - t| * (1 - 2*F((t - mu)/s))

    where F is the logistic CDF (sigmoid function).

    References
    ----------
    Gneiting, T. and Ranjan, R., 2011. Comparing density forecasts using
    threshold- and quantile-weighted scoring rules. Journal of Business &
    Economic Statistics, 29(3), pp.411-422.
    """

    def __init__(self, threshold: float = 0.0, reduction: str = "none"):
        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(
                f"reduction must be 'none', 'mean', or 'sum', got {reduction}"
            )
        self.threshold = threshold
        self.reduction = reduction

    @staticmethod
    def _crps_logistic(mu: Tensor, scale: Tensor, y: Tensor) -> Tensor:
        """Raw per-element logistic CRPS (no reduction)."""
        scale = scale.clamp(min=1e-6)
        z = (y - mu) / scale
        return (y - mu) + 2.0 * scale * torch.nn.functional.softplus(-z) - scale

    def __call__(self, mu: Tensor, scale: Tensor, y: Tensor) -> Tensor:
        """
        Compute threshold-weighted logistic CRPS.

        Parameters
        ----------
        mu : Tensor
            Predicted location, shape (B, C, H, W).
        scale : Tensor
            Predicted scale (positive), same shape as mu.
        y : Tensor
            Ground truth, same shape as mu.

        Returns
        -------
        Tensor
            twCRPS values with specified reduction.
        """
        t = self.threshold
        scale = scale.clamp(min=1e-6)
        y_clamp = torch.clamp(y, min=t)

        crps_y = self._crps_logistic(mu, scale, y_clamp)
        crps_t = self._crps_logistic(mu, scale, torch.full_like(y, t))

        # Logistic CDF at threshold: F(t) = sigmoid((t - mu) / scale)
        F_t = torch.sigmoid((t - mu) / scale)
        correction = (y_clamp - t) * (1.0 - 2.0 * F_t)

        tw_crps = crps_y - crps_t + correction

        if self.reduction == "mean":
            return tw_crps.mean()
        elif self.reduction == "sum":
            return tw_crps.sum()
        return tw_crps


class CalibratedResidualLossV2:
    """
    Residual loss with threshold-weighted CRPS for extreme-event-aware
    uncertainty calibration.

    Like CalibratedResidualLoss, but replaces Gaussian CRPS with a combination
    of standard CRPS and threshold-weighted CRPS (twCRPS). The twCRPS term
    ensures the variance head receives gradient signal for extreme events,
    where standard CRPS gradients w.r.t. sigma vanish.

    Total loss:
        L = L_denoise + crps_weight * L_crps + tw_crps_weight * L_twcrps

    The standard CRPS calibrates the bulk of the distribution, while twCRPS
    calibrates the upper tail above a specified threshold.

    Parameters
    ----------
    regression_net : torch.nn.Module
        Pre-trained regression network for residual computation.
    P_mean : float
        Mean for noise level sampling. Default 0.0.
    P_std : float
        Std for noise level sampling. Default 1.2.
    sigma_data : float
        Data std for EDM weighting. Default 0.5.
    hr_mean_conditioning : bool
        Whether to condition on HR mean. Default False.
    crps_weight : float
        Weight for standard Gaussian CRPS. Default 0.1.
    tw_crps_weight : float
        Weight for threshold-weighted CRPS. Default 0.1.
    tw_threshold : float
        Threshold for twCRPS (in normalized residual space). Default 0.0.
        Values above this get extra scoring emphasis.
    min_std : float
        Minimum std to prevent collapse. Default 1e-4.
    crps_distribution : str
        Distribution family for CRPS computation. 'gaussian' uses the
        standard normal CRPS, 'logistic' uses the logistic CRPS which has
        heavier tails better suited for precipitation. Default 'gaussian'.

    References
    ----------
    Gneiting, T. and Ranjan, R., 2011. Comparing density forecasts using
    threshold- and quantile-weighted scoring rules.
    """

    def __init__(
        self,
        regression_net: torch.nn.Module,
        P_mean: float = 0.0,
        P_std: float = 1.2,
        sigma_data: float = 0.5,
        hr_mean_conditioning: bool = False,
        crps_weight: float = 0.1,
        tw_crps_weight: float = 0.1,
        tw_threshold: float = 0.0,
        min_std: float = 1e-4,
        crps_distribution: str = "gaussian",
    ):
        self.regression_net = regression_net
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.hr_mean_conditioning = hr_mean_conditioning
        self.crps_weight = crps_weight
        self.tw_crps_weight = tw_crps_weight
        self.min_std = min_std
        self.y_mean = None
        self.latest_components = {}

        if crps_distribution == "gaussian":
            self.crps_loss = GaussianCRPSLoss(reduction="none")
            self.tw_crps_loss = ThresholdWeightedGaussianCRPSLoss(
                threshold=tw_threshold, reduction="none"
            )
        elif crps_distribution == "logistic":
            self.crps_loss = LogisticCRPSLoss(reduction="none")
            self.tw_crps_loss = ThresholdWeightedLogisticCRPSLoss(
                threshold=tw_threshold, reduction="none"
            )
        else:
            raise ValueError(
                f"crps_distribution must be 'gaussian' or 'logistic', "
                f"got '{crps_distribution}'"
            )

    def __call__(
        self,
        net: torch.nn.Module,
        img_clean: Tensor,
        img_lr: Tensor,
        patching: Optional[RandomPatching2D] = None,
        lead_time_label: Optional[Tensor] = None,
        augment_pipe: Optional[
            Callable[[Tensor], Tuple[Tensor, Optional[Tensor]]]
        ] = None,
        use_patch_grad_acc: bool = False,
    ) -> Tensor:
        """
        Calculate the combined denoising + CRPS + twCRPS loss.

        Same interface as CalibratedResidualLoss.
        """
        # Safety checks
        if patching and not isinstance(patching, RandomPatching2D):
            raise ValueError("patching must be a 'RandomPatching2D' object.")
        if (
            img_clean.shape[0] != img_lr.shape[0]
            or img_clean.shape[2:] != img_lr.shape[2:]
        ):
            raise ValueError(
                f"Shape mismatch between img_clean {img_clean.shape} and "
                f"img_lr {img_lr.shape}."
            )

        # Augment for conditional generation
        img_tot = torch.cat((img_clean, img_lr), dim=1)
        y_tot, augment_labels = (
            augment_pipe(img_tot) if augment_pipe is not None else (img_tot, None)
        )
        y = y_tot[:, : img_clean.shape[1], :, :]
        y_lr = y_tot[:, img_clean.shape[1] :, :, :]
        y_lr_res = y_lr
        batch_size = y.shape[0]

        # Compute regression mean
        if use_patch_grad_acc:
            if self.y_mean is None:
                if lead_time_label is not None:
                    y_mean = self.regression_net(
                        torch.zeros_like(y, device=img_clean.device),
                        y_lr_res,
                        lead_time_label=lead_time_label,
                        augment_labels=augment_labels,
                    )
                else:
                    y_mean = self.regression_net(
                        torch.zeros_like(y, device=img_clean.device),
                        y_lr_res,
                        augment_labels=augment_labels,
                    )
                self.y_mean = y_mean
        else:
            if lead_time_label is not None:
                y_mean = self.regression_net(
                    torch.zeros_like(y, device=img_clean.device),
                    y_lr_res,
                    lead_time_label=lead_time_label,
                    augment_labels=augment_labels,
                )
            else:
                y_mean = self.regression_net(
                    torch.zeros_like(y, device=img_clean.device),
                    y_lr_res,
                    augment_labels=augment_labels,
                )
            self.y_mean = y_mean

        # Compute residual
        y = y - self.y_mean

        # HR mean conditioning
        if self.hr_mean_conditioning:
            y_lr = torch.cat((self.y_mean, y_lr), dim=1)

        # Patchified training
        if patching:
            y_patched = patching.apply(input=y)
            y_lr_patched = patching.apply(input=y_lr, additional_input=img_lr)
            y = y_patched
            y_lr = y_lr_patched

        # Sample noise level
        rnd_normal = torch.randn([y.shape[0], 1, 1, 1], device=img_clean.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2

        # Add noise
        latent = y + torch.randn_like(y) * sigma

        # Forward pass
        if lead_time_label is not None:
            net_output = net(
                latent, y_lr, sigma,
                embedding_selector=None,
                global_index=(
                    patching.global_index(batch_size, img_clean.device)
                    if patching is not None else None
                ),
                lead_time_label=lead_time_label,
                augment_labels=augment_labels,
            )
        else:
            net_output = net(
                latent, y_lr, sigma,
                embedding_selector=None,
                global_index=(
                    patching.global_index(batch_size, img_clean.device)
                    if patching is not None else None
                ),
                augment_labels=augment_labels,
            )

        # Handle both heteroscedastic and standard outputs
        if isinstance(net_output, tuple):
            D_mean, D_std = net_output
            D_std = D_std.clamp(min=self.min_std)
        else:
            D_mean = net_output
            D_std = None

        # Standard denoising loss
        loss_denoise = weight * ((D_mean - y) ** 2)

        # Add CRPS + twCRPS if we have uncertainty
        # Scale CRPS by the same EDM weight so that the relative balance between
        # denoising and CRPS stays constant across all sigma levels.
        if D_std is not None:
            loss_crps = self.crps_loss(D_mean, D_std, y)
            loss_tw_crps = self.tw_crps_loss(D_mean, D_std, y)
            loss = weight * (
                (D_mean - y) ** 2
                + self.crps_weight * loss_crps
                + self.tw_crps_weight * loss_tw_crps
            )
            self.latest_components = {
                "loss_denoise_mean": loss_denoise.detach().mean(),
                "loss_crps_mean": loss_crps.detach().mean(),
                "loss_tw_crps_mean": loss_tw_crps.detach().mean(),
                "loss_crps_weighted_mean": (
                    (weight * self.crps_weight * loss_crps).detach().mean()
                ),
                "loss_tw_crps_weighted_mean": (
                    (weight * self.tw_crps_weight * loss_tw_crps).detach().mean()
                ),
                "loss_total_mean": loss.detach().mean(),
                "pred_scale_mean": D_std.detach().mean(),
            }
        else:
            loss = loss_denoise
            self.latest_components = {
                "loss_denoise_mean": loss_denoise.detach().mean(),
                "loss_total_mean": loss.detach().mean(),
            }

        return loss
