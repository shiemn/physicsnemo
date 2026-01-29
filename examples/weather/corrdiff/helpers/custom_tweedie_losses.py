import torch
import torch.nn.functional as F
from typing import Optional, Callable, Tuple
from torch import Tensor

from torch.nn.functional import softplus

class RegressionLoss:
    """
    Regression loss function for the deterministic predictions.
    THIS IS A SLIGHT REFACTOR OF THE NVIDIA ORIGINAL VERSION
    Note: this loss does not apply any reduction.

    Attributes
    ----------
    sigma_data: float
        Standard deviation for data. Deprecated and ignored.

    Note
    ----
    Reference: Mardani, M., Brenowitz, N., Cohen, Y., Pathak, J., Chen, C.Y.,
    Liu, C.C.,Vahdat, A., Kashinath, K., Kautz, J. and Pritchard, M., 2023.
    Generative Residual Diffusion Modeling for Km-scale Atmospheric Downscaling.
    arXiv preprint arXiv:2309.15214.
    """

    def __init__(self):
        """
        Arguments
        ----------
        """
        return

    def __call__(self,*args,**kwargs):
        """
        Calculate and return the regression loss for
        deterministic predictions.

        Parameters
        ----------
        net : torch.nn.Module
            The neural network model that will make predictions.
            Expected signature: `net(x, img_lr,
            augment_labels=augment_labels, force_fp32=False)`, where:
                x (torch.Tensor): Tensor of shape (B, C_hr, H, W). Is zero-filled.
                img_lr (torch.Tensor): Low-resolution input of shape (B, C_lr, H, W)
                augment_labels (torch.Tensor, optional): Optional augmentation
                labels, returned by `augment_pipe`.
                force_fp32 (bool, optional): Whether to force the model to use
                fp32, by default False.
            Returns:
                torch.Tensor: Predictions of shape (B, C_hr, H, W)

        img_clean : torch.Tensor
            High-resolution input images of shape (B, C_hr, H, W).
            Used as ground truth and for data augmentation if 'augment_pipe' is provided.

        img_lr : torch.Tensor
            Low-resolution input images of shape (B, C_lr, H, W).
            Used as input to the neural network.

        augment_pipe : callable, optional
            An optional data augmentation function.
            Expected signature:
                img_tot (torch.Tensor): Concatenated high and low resolution
                    images of shape (B, C_hr+C_lr, H, W)
            Returns:
                Tuple[torch.Tensor, Optional[torch.Tensor]]:
                    - Augmented images of shape (B, C_hr+C_lr, H, W)
                    - Optional augmentation labels

        lead_time_label : Optional[torch.Tensor], optional
            Lead time labels for temporal predictions, by default None.
            Shape can vary based on model requirements, typically (B,) or scalar.
        -------
        torch.Tensor
            A tensor representing the per-sample element-wise squared
            difference between the network's predictions and the high
            resolution images `img_clean` (possibly data-augmented by
            `augment_pipe`).
            Shape: (B, C_hr, H, W), same as `img_clean`.
        """

        targ, pred = self._prep_samples(*args,**kwargs)
        loss = self._compute_loss(targ,pred)

        # keep_samples=kwargs.get("keep_samples", False)
        # if keep_samples:
        #     return loss, targ, pred
        # else:
        #     return loss, None, None
        return loss
        
    def _prep_samples(
        self,
        net: torch.nn.Module,
        img_clean: torch.Tensor,
        img_lr: torch.Tensor,
        augment_pipe: Optional[
            Callable[[torch.Tensor], Tuple[torch.Tensor, Optional[torch.Tensor]]]
        ] = None,
        lead_time_label: Optional[torch.Tensor] = None,
        keep_samples: Optional[bool] = False,
    ) -> torch.Tensor:

        img_tot = torch.cat((img_clean, img_lr), dim=1)
        y_tot, augment_labels = (
            augment_pipe(img_tot) if augment_pipe is not None else (img_tot, None)
        )
        y = y_tot[:, : img_clean.shape[1], :, :]
        y_lr = y_tot[:, img_clean.shape[1] :, :, :]

        zero_input = torch.zeros_like(y, device=img_clean.device)

        if lead_time_label is not None:
            D_yn = net(
                zero_input,
                y_lr,
                force_fp32=False,
                lead_time_label=lead_time_label,
                augment_labels=augment_labels,
            )
        else:
            D_yn = net(
                zero_input,
                y_lr,
                force_fp32=False,
                augment_labels=augment_labels,
            )

        return y, D_yn
        
    def _compute_loss(self,targ,pred):
        
        loss = self.weight * ((pred - targ) ** 2)

        return loss
        
        
class RegressionLossTweedie(RegressionLoss):
    """
    Tweedie deviance loss function for the deterministic predictions.
    Takes a keyword argument p, which parameterises the loss. Defaults to mse.
    Note: this loss does not apply any reduction.
    ----
    Reference: Hunt, Kieran M., 2025.
    Stop using root-mean-square error as a precipitation target!
    arXiv preprint https://arxiv.org/abs/2509.08369.
    """
    def __init__(self,p=0):
        self.p= p
        super().__init__()

    def introduce(self):
        text=f"Tweedie Loss with p = {self.p}"
        if self.p==0:
            text=text+" (equivalent to mse)"
        return text

    def _compute_loss(self, targ, pred):
        loss = self.tweedie_deviance(pred,targ)

        return loss
    
    def tweedie_deviance(self,x,y,p=None):
        """"x is model mean prediction, y is verifying target.
        Deviance is not symmetric, so do not transpose!
        """
        if p is None:
            p=self.p
        if p==0:
            return self.mse(x,y)
        
        elif p>1 and p<2:
            return self._tweedie_deviance(x,y,p)
        
        else: 
            raise(ValueError('Tweedie Deviance not currently supported for non-zero p outside 1<p<2'))
        
    def _tweedie_deviance(self,x,y,p):
        #This enforces positive predictions. We've set
        #a sharp beta value to keep it close to the
        #true problem.
        """For numerical stability the softplus
           implementation reverts
           to the linear function when 
           input*β>threshold, threshold=20"""
        x=softplus(x,beta=10)
        y=softplus(y,beta=10)

        dev = 2 * (
            torch.pow(y,2 - p) / ((1 - p) * (2 - p))
            - (y * torch.pow(x,1 - p)) / (1 - p)
            + torch.pow(x,2 - p) / (2 - p)
        )
        return dev
    
    def mse(self,x,y):
        return (y-x)**2


class FractionsTweedieLoss(RegressionLossTweedie):
    """Takes keywords p, as for Tweedie, and d, which 
    determines a number of gridcells of 'leeway' for 
    loss computation. Compute time scales
    quadratically with d."""
    
    def __init__(self,p=0,d=1):
        self.d= d
        super().__init__(p=p)

    def introduce(self):
        text=f"Fractions Tweedie Loss with p = {self.p}"
        if self.p==0:
            text=text+" (equivalent to mse)"

        text=text+f" and d = {self.d}"
        if self.d==0:
            text=text+" (equivalent to Tweedie Loss)"

        return text

    
    def _compute_loss(self,targ,pred):
        if self.d==0:
            pass
        else:
            D_yn,y=self._unfold_and_sort(pred,targ)

        loss = self.tweedie_deviance(D_yn,y)
        #we don't reshape data because its about to be summed anyway.
        return loss
   
    def _unfold_and_sort(self,X,Y,d=None):
        
        if X.ndim!=4:
            raise(ValueError(f'Shape of data must be rank 4, instead {X.shape}'))
        
        d= d or self.d
        ksize = 2 * d + 1
        pad = d

        # Extract sliding patches (unfold)
        X_patches = F.unfold(X, kernel_size=ksize, padding=pad)  # (B, C*k*k, H*W)
        Y_patches = F.unfold(Y, kernel_size=ksize, padding=pad)
        
        # Sort each patch along the patch dimension
        X_sorted, _ = torch.sort(X_patches, dim=1)
        Y_sorted, _ = torch.sort(Y_patches, dim=1)

        return X_sorted/ksize, Y_sorted/ksize
    
class FlexiLoss(FractionsTweedieLoss):
    """For multivariate targets, pass a list of 
    ds and ps. of length = number of target variables."""
    
    def introduce(self):
        intro=f'FlexiLoss with d= {self.d} and p= {self.p}'
        return intro

    def _compute_loss(self,targ,pred,collapse=False):
        """If collapse is True, the loss computation should be
        slightly quicker, but it assumes that the loss is going 
        to be immediately summed anyway, which could be dangerous."""
        D_yn=pred
        y=targ
        B,C,X,Y=y.shape
        losses=[]
        for i in range(C):
            d=self.d[i]
            p=self.p[i]
            pred=D_yn[:,[i],:,:]
            targ=y[:,[i],:,:]
            if d!=0:
                pred,targ=self._unfold_and_sort(pred,targ,d=d)
            loss = self.tweedie_deviance(pred,targ,p=p)

            if collapse:
                j=0
                losses.append(loss.sum())
            else:
                j=1
                if d!=0:
                    loss=loss.sum(axis=1,keepdims=True).reshape([B,1,X,Y])
                losses.append(loss)
        loss=torch.concat(losses,axis=j)
        return loss


# class ResidualLoss:
#     """
#     Mixture loss function for denoising score matching.

#     This class implements a loss function that combines deterministic
#     regression with denoising score matching. It uses a pre-trained regression
#     network to compute residuals before applying the diffusion process.

#     Parameters
#     ----------
#     regression_net : torch.nn.Module
#         The regression network used for computing residuals.
#     P_mean : float
#         Mean value for noise level computation.
#     P_std : float
#         Standard deviation for noise level computation.
#     sigma_data : float
#         Standard deviation for data weighting.
#     hr_mean_conditioning : bool
#         Flag indicating whether to use high-resolution mean for conditioning.

#     Note
#     ----
#     Reference: Mardani, M., Brenowitz, N., Cohen, Y., Pathak, J., Chen, C.Y.,
#     Liu, C.C., Vahdat, A., Kashinath, K., Kautz, J. and Pritchard, M., 2023.
#     Generative Residual Diffusion Modeling for Km-scale Atmospheric
#     Downscaling. arXiv preprint arXiv:2309.15214.
#     """

#     def __init__(
#         self,
#         regression_net: torch.nn.Module,
#         P_mean: float = 0.0,
#         P_std: float = 1.2,
#         sigma_data: float = 0.5,
#         hr_mean_conditioning: bool = False
#     ):
#         """
#         Arguments
#         ----------
#         regression_net : torch.nn.Module
#             Pre-trained regression network used to compute residuals.
#             Expected signature: `net(zero_input, y_lr,
#             lead_time_label=lead_time_label, augment_labels=augment_labels)` or
#             `net(zero_input, y_lr, augment_labels=augment_labels)`, where:
#                 zero_input (torch.Tensor): Zero tensor of shape (B, C_hr, H, W)
#                 y_lr (torch.Tensor): Low-resolution input of shape (B, C_lr, H, W)
#                 lead_time_label (torch.Tensor, optional): Optional lead time labels
#                 augment_labels (torch.Tensor, optional): Optional augmentation labels
#             Returns:
#                 torch.Tensor: Predictions of shape (B, C_hr, H, W)

#         P_mean : float, optional
#             Mean value for noise level computation, by default 0.0.

#         P_std : float, optional
#             Standard deviation for noise level computation, by default 1.2.

#         sigma_data : float, optional
#             Standard deviation for data weighting, by default 0.5.

#         hr_mean_conditioning : bool, optional
#             Whether to use high-resolution mean for conditioning predicted, by default False.
#             When True, the mean prediction from `regression_net` is channel-wise
#             concatenated with `img_lr` for conditioning.
#         """
#         self.regression_net = regression_net
#         self.P_mean = P_mean
#         self.P_std = P_std
#         self.sigma_data = sigma_data
#         self.hr_mean_conditioning = hr_mean_conditioning
#         self.y_mean = None

#     def get_noise_params(self, y: Tensor) -> Tensor:
#         """
#         Compute the noise parameters to apply denoising score matching.

#         Parameters
#         ----------
#         y : torch.Tensor
#             Latent state of shape :math:`(B, *)`. Only used to determine the shape of
#             the noise and create tensors on the same device.

#         Returns
#         -------
#         Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
#             - Noise ``n`` of shape :math:`(B, *)` to be added to the latent state.
#             - Noise level ``sigma`` of shape :math:`(B, 1, 1, 1)`.
#             - Weight ``weight`` of shape :math:`(B, 1, 1, 1)` to multiply the loss.
#         """
#         # Sample noise level
#         rnd_normal = torch.randn([y.shape[0], 1, 1, 1], device=y.device)
#         sigma = (rnd_normal * self.P_std + self.P_mean).exp()
#         # Loss weight
#         weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
#         # Sample noise
#         n = torch.randn_like(y) * sigma
#         return n, sigma, weight

#     def __call__(
#         self,
#         net: torch.nn.Module,
#         img_clean: Tensor,
#         img_lr: Tensor,
#         patching: Optional[RandomPatching2D] = None,
#         lead_time_label: Optional[Tensor] = None,
#         augment_pipe: Optional[
#             Callable[[Tensor], Tuple[Tensor, Optional[Tensor]]]
#         ] = None,
#         use_patch_grad_acc: bool = False,
#         keep_samples=False
#     ) -> Tensor:
#         """
#         Calculate and return the loss for denoising score matching.

#         This method computes a mixture loss that combines deterministic
#         regression with denoising score matching. It first computes residuals
#         using the regression network, then applies the diffusion process to
#         these residuals.

#         The diffusion model `net` is expected to be conditioned on an input with
#         `C_cond` channels, which should be:
#             - `C_cond = C_lr` if `hr_mean_conditioning` is `False` and
#               `patching` is None.
#             - `C_cond = C_hr + C_lr` if `hr_mean_conditioning` is `True` and
#               `patching` is None.
#             - `C_cond = C_hr + 2*C_lr` if `hr_mean_conditioning` is `True` and
#               `patching` is not None.
#             - `C_cond = 2*C_lr` if `hr_mean_conditioning` is `False` and
#               `patching` is not None.
#         Additionally, `C_cond` should also include any embedding channels,
#         such as positional embeddings or time embeddings.

#         Note: this loss function does not apply any reduction.

#         Parameters
#         ----------
#         net : torch.nn.Module
#             The neural network model for the diffusion process.
#             Expected signature: `net(latent, y_lr, sigma,
#             embedding_selector=embedding_selector, lead_time_label=lead_time_label,
#             augment_labels=augment_labels)`, where:
#                 latent (torch.Tensor): Noisy input of shape (B[*P], C_hr, H_patch, W_patch)
#                 y_lr (torch.Tensor): Conditioning of shape (B[*P], C_cond, H_patch, W_patch)
#                 sigma (torch.Tensor): Noise level of shape (B[*P], 1, 1, 1)
#                 embedding_selector (callable, optional): Function to select
#                     positional embeddings. Only used if `patching` is provided.
#                 lead_time_label (torch.Tensor, optional): Lead time labels.
#                 augment_labels (torch.Tensor, optional): Augmentation labels
#             Returns:
#                 torch.Tensor: Predictions of shape (B[*P], C_hr, H_patch, W_patch)

#         img_clean : torch.Tensor
#             High-resolution input images of shape (B, C_hr, H, W).
#             Used as ground truth and for data augmentation if 'augment_pipe' is provided.

#         img_lr : torch.Tensor
#             Low-resolution input images of shape (B, C_lr, H, W).
#             Used as input to the regression network and conditioning for the
#             diffusion process.
#         Returns
#         -------
#         torch.Tensor
#                 A tensor of shape (B, C_hr, H, W) representing the per-sample loss.

#         Raises
#         ------
#         ValueError
#             If shapes of img_clean and img_lr are incompatible.
#         """

#         # Safety check: enforce shapes
#         if (
#             img_clean.shape[0] != img_lr.shape[0]
#             or img_clean.shape[2:] != img_lr.shape[2:]
#         ):
#             raise ValueError(
#                 f"Shape mismatch between img_clean {img_clean.shape} and "
#                 f"img_lr {img_lr.shape}. "
#                 f"Batch size, height and width must match."
#             )
#         y,y_lr=img_clean,img_lr
#         y_lr_reg = y_lr
#         batch_size = y.shape[0]

#         y_reg = self.regression_net(
#             torch.zeros_like(y, device=img_clean.device),
#             y_lr_reg
#         )


#         y = y - y_reg

#         if self.hr_mean_conditioning:
#             y_lr = torch.cat((y_reg, y_lr), dim=1)

#         # Add noise to the latent state
#         n, sigma, weight = self.get_noise_params(y)

#         D_yn = net(
#             y + n,
#             y_lr,
#             sigma,
#         )
        
#         loss = weight * ((D_yn - y) ** 2)
#         if keep_samples:
#             return loss, y, D_yn 
#         else:
#             return loss, None, None
