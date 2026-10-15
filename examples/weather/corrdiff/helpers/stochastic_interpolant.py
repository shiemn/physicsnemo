"""Paired stochastic interpolants for CWB channels and selected radar bases."""

from __future__ import annotations

import hashlib
import math
import weakref
from collections.abc import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from physicsnemo import Module
from physicsnemo.models.diffusion.song_unet import SongUNetPosEmbd
from physicsnemo.models.meta import ModelMetaData


def _schedule(config: Mapping, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return schedule value and time derivative, with shape matching ``t``."""
    kind = config["type"]
    if kind == "linear":
        start, end = float(config["start"]), float(config["end"])
        return start + (end - start) * t, torch.full_like(t, end - start)
    if kind == "polynomial":
        power = float(config["power"])
        if power < 1:
            raise ValueError("Polynomial schedule power must be at least one")
        return t.pow(power), power * t.pow(power - 1)
    raise ValueError(f"Unsupported SI schedule: {kind!r}")


def _validate_schedules(alpha, beta, sigma):
    endpoints = torch.tensor([0.0, 1.0])
    for name, config, expected in (
        ("alpha", alpha, (1.0, 0.0)),
        ("beta", beta, (0.0, 1.0)),
        ("sigma", sigma, (None, 0.0)),
    ):
        values, derivatives = _schedule(config, endpoints)
        if not torch.isfinite(values).all() or not torch.isfinite(derivatives).all():
            raise ValueError(f"{name} schedule must be finite on [0, 1]")
        for value, target in zip(values, expected):
            if target is not None and not torch.isclose(value, torch.tensor(target)):
                raise ValueError(f"{name} schedule has incompatible SI endpoint")
    sigma_values, _ = _schedule(sigma, torch.linspace(0, 1, 11))
    if (sigma_values < 0).any():
        raise ValueError("sigma schedule must be nonnegative on [0, 1]")


def _channel_name(channel) -> str:
    return str(channel.name)


def build_channel_spec(dataset, output_names: Sequence[str], base_names: Sequence[str | None],
                       base_type: str = "matching_input") -> dict:
    """Resolve selected channel positions and active CWB/ERA5 affine statistics."""
    if not hasattr(dataset, "in_channels") or not hasattr(dataset, "out_channels"):
        raise ValueError("First SI integration requires the non-temporal CWB dataset")
    if getattr(dataset, "n_history", 0) != 0:
        raise ValueError("First SI integration does not support historical input frames")
    output_names = list(output_names)
    base_names = list(base_names)
    if not output_names or len(output_names) != len(base_names):
        raise ValueError("SI output and base channel lists must be nonempty and equally long")
    if len(set(output_names)) != len(output_names) or len(set(base_names)) != len(base_names):
        raise ValueError("SI channel lists must not contain duplicates")
    if base_type == "matching_input":
        if output_names != base_names:
            raise ValueError("matching_input requires the same physical variable for each output/base pair")
    elif base_type in ("regression_mean", "gaussian_lowres"):
        if output_names[0] != "maximum_radar_reflectivity" or base_names[0] is not None:
            raise ValueError("Radar SI requires radar first, with a null ERA5 base name")
        if any(output != base for output, base in zip(output_names[1:], base_names[1:])):
            raise ValueError("Non-radar SI channels require matching ERA5 base names")
    else:
        raise ValueError(f"Unsupported SI base_type: {base_type!r}")

    output_meta = dataset.output_channels()
    input_meta = dataset.input_channels()
    if [_channel_name(ch) for ch in output_meta] != output_names:
        raise ValueError("SI output names must equal the selected CWB channels in dataset order")
    base_indices = []
    for name in base_names:
        if name is None:
            # Radar has no matching ERA5 channel; -1 marks its external base.
            base_indices.append(-1)
            continue
        matches = [
            index for index, channel in enumerate(input_meta)
            if _channel_name(channel) == name and str(channel.level) == ""
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one surface ERA5 input for {name!r}; got {len(matches)}")
        base_indices.append(matches[0])

    info = dataset.info()
    input_center, input_scale = (np.asarray(v) for v in info["input_normalization"])
    output_center, output_scale = (np.asarray(v) for v in info["target_normalization"])
    selected_inputs = np.asarray(dataset.in_channels)[[max(index, 0) for index in base_indices]]
    selected_outputs = np.asarray(dataset.out_channels)
    selected_center = input_center[selected_inputs].astype(np.float32)
    selected_scale = input_scale[selected_inputs].astype(np.float32)
    for pos, index in enumerate(base_indices):
        if index == -1:
            selected_center[pos], selected_scale[pos] = 0.0, 1.0
    result = {
        "img_in_channels": len(input_meta),
        "img_out_channels": len(output_meta),
        "base_input_indices": base_indices,
        "input_centers": selected_center.tolist(),
        "input_scales": selected_scale.tolist(),
        "output_centers": output_center[selected_outputs].astype(np.float32).tolist(),
        "output_scales": output_scale[selected_outputs].astype(np.float32).tolist(),
        "output_channel_names": output_names,
        "base_input_channel_names": base_names,
    }
    if any(not np.isfinite(v) for key in ("input_centers", "input_scales", "output_centers", "output_scales") for v in result[key]):
        raise ValueError("SI normalization statistics must be finite")
    if any(v <= 0 for key in ("input_scales", "output_scales") for v in result[key]):
        raise ValueError("SI normalization scales must be positive")
    return result


def matching_conditioning(img_lr, base_indices, input_centers, input_scales,
                         output_centers, output_scales):
    """Move matched ERA5 channels to target units and omit them from ``y_rest``."""
    if img_lr.ndim != 4:
        raise ValueError("SI conditioning must have batch, channel and two spatial dimensions")
    base = img_lr[:, base_indices].float()
    x0 = (base * input_scales + input_centers - output_centers) / output_scales
    other_indices = [i for i in range(img_lr.shape[1]) if i not in base_indices]
    return x0, img_lr[:, other_indices].float()


class StochasticInterpolant(Module):
    """Direct drift network; ``x0`` is target-normalized and excluded from ``y_rest``."""

    _overridable_args = {"use_apex_gn"}

    def __init__(
        self,
        img_resolution,
        img_in_channels: int,
        img_out_channels: int,
        base_input_indices,
        input_centers,
        input_scales,
        output_centers,
        output_scales,
        output_channel_names,
        base_input_channel_names,
        alpha_schedule,
        beta_schedule,
        sigma_schedule,
        base_type: str = "matching_input",
        use_fp16: bool = False,
        gaussian_lowres_shape=(36, 36),
        **model_kwargs,
    ):
        super().__init__(meta=ModelMetaData(name="StochasticInterpolant"))
        if base_type not in ("matching_input", "regression_mean", "gaussian_lowres"):
            raise ValueError(f"Unsupported SI base_type: {base_type!r}")
        _validate_schedules(alpha_schedule, beta_schedule, sigma_schedule)
        if model_kwargs.get("embedding_type") == "zero":
            raise ValueError("SI requires a nonzero time embedding")
        self.alpha_schedule = dict(alpha_schedule)
        self.beta_schedule = dict(beta_schedule)
        self.sigma_schedule = dict(sigma_schedule)
        self.base_type = base_type
        self.img_in_channels = int(img_in_channels)
        self.img_out_channels = int(img_out_channels)
        self.base_input_indices = tuple(int(i) for i in base_input_indices)
        self.output_channel_names = tuple(output_channel_names)
        self.base_input_channel_names = tuple(base_input_channel_names)
        self.use_fp16 = bool(use_fp16)
        self._validated_regression = None
        self.gaussian_lowres_shape = tuple(int(value) for value in gaussian_lowres_shape)
        if len(self.gaussian_lowres_shape) != 2 or min(self.gaussian_lowres_shape) < 1:
            raise ValueError("Gaussian low-resolution shape must contain two positive dimensions")
        if len(self.base_input_indices) != self.img_out_channels:
            raise ValueError("One base specification is required per SI target channel")
        if len(self.output_channel_names) != self.img_out_channels or len(self.base_input_channel_names) != self.img_out_channels:
            raise ValueError("SI channel names must match the target channel count")
        matching_indices = tuple(index for index in self.base_input_indices if index >= 0)
        if len(set(matching_indices)) != len(matching_indices):
            raise ValueError("SI base input indices must be unique")
        if any(i >= self.img_in_channels for i in matching_indices):
            raise ValueError("SI base input index outside the selected conditioning stack")
        if base_type == "matching_input":
            if len(matching_indices) != self.img_out_channels or self.output_channel_names != self.base_input_channel_names:
                raise ValueError("matching_input requires one matching ERA5 channel per target")
        elif (self.output_channel_names[0] != "maximum_radar_reflectivity"
              or self.base_input_indices[0] != -1 or self.base_input_channel_names[0] is not None
              or any(index < 0 or output != base for index, output, base in zip(
                  self.base_input_indices[1:], self.output_channel_names[1:], self.base_input_channel_names[1:]))):
            raise ValueError("Radar base requires radar first and matching non-radar ERA5 channels")

        for name, values in (
            ("input_centers", input_centers),
            ("input_scales", input_scales),
            ("output_centers", output_centers),
            ("output_scales", output_scales),
        ):
            tensor = torch.as_tensor(values, dtype=torch.float32).reshape(1, -1, 1, 1)
            if tensor.shape[1] != self.img_out_channels:
                raise ValueError(f"{name} must have one value per output channel")
            self.register_buffer(name, tensor)

        grid_channels = int(model_kwargs.get("N_grid_channels", 4))
        self.model = SongUNetPosEmbd(
            img_resolution=img_resolution,
            in_channels=2 * self.img_out_channels + self.img_in_channels - len(matching_indices) + grid_channels,
            out_channels=self.img_out_channels,
            **model_kwargs,
        )

    @property
    def amp_mode(self):
        return self.model.amp_mode

    @amp_mode.setter
    def amp_mode(self, value):
        self.model.amp_mode = value

    @property
    def profile_mode(self):
        return self.model.profile_mode

    @profile_mode.setter
    def profile_mode(self, value):
        self.model.profile_mode = value

    def validate_dataset(self, spec: Mapping):
        if (self.img_in_channels != spec["img_in_channels"]
                or self.img_out_channels != spec["img_out_channels"]
                or self.base_input_indices != tuple(spec["base_input_indices"])
                or self.output_channel_names != tuple(spec["output_channel_names"])
                or self.base_input_channel_names != tuple(spec["base_input_channel_names"])):
            raise ValueError("SI checkpoint and evaluation dataset use different channel selections")
        for name in ("input_centers", "input_scales", "output_centers", "output_scales"):
            expected = torch.as_tensor(spec[name], device=getattr(self, name).device)
            if not torch.allclose(getattr(self, name).flatten(), expected, rtol=1e-5, atol=1e-5):
                raise ValueError(f"SI checkpoint and evaluation dataset differ in {name}")

    def validate_radar_regression(self, regression_net, img_shape):
        """Check the fixed DS1 radar-only regression before using its target-normalized output."""
        if regression_net is None:
            raise ValueError("regression_mean requires the frozen radar regression network")
        if regression_net.training:
            raise ValueError("Radar regression must be eval-only with frozen parameters")
        if self._validated_regression is not None:
            model_ref, checked_shape = self._validated_regression
            if model_ref() is regression_net and checked_shape == tuple(img_shape):
                return
        if any(param.requires_grad for param in regression_net.parameters()):
            raise ValueError("Radar regression must be eval-only with frozen parameters")
        if (getattr(regression_net, "img_out_channels", None) != 1
                or getattr(regression_net, "img_in_channels", None) != self.img_in_channels + 4):
            raise ValueError("Radar regression requires one output and the complete ERA5 stack plus four grid channels")
        if (getattr(regression_net, "img_shape_y", None), getattr(regression_net, "img_shape_x", None)) != tuple(img_shape):
            raise ValueError("Radar regression checkpoint and SI target grid have different shapes")
        self._validated_regression = (weakref.ref(regression_net), tuple(img_shape))

    def conditioning(self, img_lr: torch.Tensor, regression_net=None,
                     gaussian_generator=None) -> tuple[torch.Tensor, torch.Tensor]:
        if img_lr.ndim != 4 or img_lr.shape[1] != self.img_in_channels:
            raise ValueError("SI conditioning has incorrect channel count")
        if self.base_type == "matching_input":
            return matching_conditioning(
                img_lr, self.base_input_indices,
                self.input_centers, self.input_scales,
                self.output_centers, self.output_scales,
            )
        if self.base_type == "regression_mean":
            self.validate_radar_regression(regression_net, img_lr.shape[2:])
            with torch.no_grad():
                radar = regression_net(
                    x=torch.zeros((img_lr.shape[0], 1, *img_lr.shape[2:]), device=img_lr.device),
                    img_lr=img_lr,
                ).float()
        else:
            radar = F.interpolate(
                torch.randn((img_lr.shape[0], 1, *self.gaussian_lowres_shape),
                            device=img_lr.device, generator=gaussian_generator),
                size=img_lr.shape[2:], mode="bilinear", align_corners=False,
            )
        if radar.shape != (img_lr.shape[0], 1, *img_lr.shape[2:]):
            raise ValueError("Radar base must have one target-grid channel")
        if not torch.is_grad_enabled() and not torch.isfinite(radar).all():
            raise ValueError("Radar base must be finite")
        matched = self.base_input_indices[1:]
        if matched:
            other = img_lr[:, matched].float()
            other = (other * self.input_scales[:, 1:] + self.input_centers[:, 1:]
                     - self.output_centers[:, 1:]) / self.output_scales[:, 1:]
            x0 = torch.cat((radar, other), dim=1)
        else:
            x0 = radar
        y_rest = img_lr[:, [index for index in range(img_lr.shape[1]) if index not in matched]].float()
        return x0, y_rest

    def coefficients(self, t: torch.Tensor):
        return tuple(_schedule(config, t) for config in (
            self.alpha_schedule, self.beta_schedule, self.sigma_schedule
        ))

    def forward(self, xt, x0, y_rest, t):
        if xt.shape != x0.shape or xt.shape[1] != self.img_out_channels:
            raise ValueError("SI state and base must have the same target shape")
        expected_rest = self.img_in_channels - sum(index >= 0 for index in self.base_input_indices)
        if y_rest.shape[0] != xt.shape[0] or y_rest.shape[1] != expected_rest or y_rest.shape[2:] != xt.shape[2:]:
            raise ValueError("SI extra conditioning has incorrect shape")
        if t.ndim != 1 or t.shape[0] != xt.shape[0]:
            raise ValueError("SI artificial time must have one value per batch item")
        image = torch.cat((xt, x0, y_rest), dim=1)
        dtype = torch.float16 if self.use_fp16 and image.device.type == "cuda" else torch.float32
        result = self.model(image.to(dtype), t.to(dtype), class_labels=None)
        return result.float()


class StochasticInterpolantLoss:
    """Unreduced CDSI drift MSE for the existing CorrDiff train loop."""

    def __init__(self, regression_net=None):
        self.regression_net = regression_net

    def __call__(self, net, img_clean, img_lr, augment_pipe=None,
                 lead_time_label=None, patching=None, use_patch_grad_acc=False):
        if patching is not None or use_patch_grad_acc or augment_pipe is not None:
            raise ValueError("First SI path supports only full-domain training")
        if lead_time_label is not None:
            raise ValueError("First SI path does not support temporal lead-time conditioning")
        module = net.module if hasattr(net, "module") else net
        if getattr(module, "base_type", "matching_input") == "regression_mean":
            x0, y_rest = module.conditioning(img_lr, regression_net=self.regression_net)
        else:
            x0, y_rest = module.conditioning(img_lr)
        x1 = img_clean.float()
        if x0.shape != x1.shape:
            raise ValueError("SI base and target must have identical shape")
        t = torch.rand(x1.shape[0], device=x1.device).clamp_(min=1e-5, max=1 - 1e-5)
        eps = torch.randn_like(x1)
        (alpha, alpha_dot), (beta, beta_dot), (sigma, sigma_dot) = module.coefficients(t)
        def wide(value):
            return value.reshape(-1, 1, 1, 1)
        xt = wide(alpha) * x0 + wide(beta) * x1 + wide(t.sqrt() * sigma) * eps
        drift_target = wide(alpha_dot) * x0 + wide(beta_dot) * x1 + wide(t.sqrt() * sigma_dot) * eps
        drift = net(xt, x0, y_rest, t)
        return (drift - drift_target).square()


@torch.no_grad()
def sample_ensemble(net: StochasticInterpolant, img_lr: torch.Tensor,
                    seed_batches: Sequence[Sequence[int]], num_steps: int,
                    regression_net=None, gaussian_base_seed: int | None = None,
                    timestamp=None) -> torch.Tensor:
    """Euler-Maruyama SI sampling with one independent generator per member."""
    if num_steps < 1:
        raise ValueError("SI num_steps must be positive")
    module = getattr(net, "_orig_mod", net)
    base_type = getattr(module, "base_type", "matching_input")
    if base_type == "regression_mean":
        x0_single, y_rest_single = module.conditioning(img_lr, regression_net=regression_net)
    elif base_type == "gaussian_lowres":
        if gaussian_base_seed is None or timestamp is None:
            raise ValueError("Gaussian SI evaluation requires a base seed and timestamp")
        # A stable hash keeps x0 independent of time-list order and ensemble seeds.
        payload = f"{int(gaussian_base_seed)}:{timestamp}".encode("utf-8")
        gaussian_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63)
        generator = torch.Generator(device=img_lr.device).manual_seed(gaussian_seed)
        x0_single, y_rest_single = module.conditioning(img_lr, gaussian_generator=generator)
    else:
        x0_single, y_rest_single = module.conditioning(img_lr)
    if x0_single.shape[0] != 1:
        raise ValueError("SI ensemble sampling expects one weather time at a time")
    samples = []
    dt = 1.0 / num_steps
    for batch in seed_batches:
        seeds = [int(seed) for seed in batch]
        if not seeds:
            continue
        generators = [torch.Generator(device=x0_single.device).manual_seed(seed) for seed in seeds]
        x0 = x0_single.expand(len(seeds), -1, -1, -1)
        y_rest = y_rest_single.expand(len(seeds), -1, -1, -1)
        state = x0.clone()
        for step in range(num_steps):
            t = torch.full((len(seeds),), step * dt, device=state.device)
            drift = net(state, x0, y_rest, t)
            sigma, _ = module.coefficients(t)[2]
            noise = torch.stack([
                torch.randn(x0_single.shape[1:], device=state.device, dtype=state.dtype, generator=gen)
                for gen in generators
            ])
            state = state + drift * dt + sigma.reshape(-1, 1, 1, 1) * math.sqrt(dt) * noise
        samples.append(state)
    if not samples:
        raise ValueError("SI sampling requires at least one seed")
    return torch.cat(samples, dim=0)
