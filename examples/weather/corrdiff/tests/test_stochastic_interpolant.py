"""Small, data-free checks for the CWB stochastic-interpolant path."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from physicsnemo import Module
from helpers.stochastic_interpolant import (
    StochasticInterpolant,
    StochasticInterpolantLoss,
    _schedule,
    build_channel_spec,
    matching_conditioning,
    sample_ensemble,
)


class FakeDataset:
    in_channels = [0, 17, 18, 19]
    out_channels = [1, 2, 3]

    def input_channels(self):
        return [
            SimpleNamespace(name="tcwv", level=""),
            SimpleNamespace(name="temperature_2m", level=""),
            SimpleNamespace(name="eastward_wind_10m", level=""),
            SimpleNamespace(name="northward_wind_10m", level=""),
        ]

    def output_channels(self):
        return [
            SimpleNamespace(name="temperature_2m", level=""),
            SimpleNamespace(name="eastward_wind_10m", level=""),
            SimpleNamespace(name="northward_wind_10m", level=""),
        ]

    def info(self):
        input_center = np.zeros(20, dtype=np.float32)
        input_scale = np.ones(20, dtype=np.float32)
        input_center[17:20] = [280, 1, -2]
        input_scale[17:20] = [5, 4, 6]
        return {
            "input_normalization": (input_center, input_scale),
            "target_normalization": (
                np.asarray([0, 282, 0, 0], dtype=np.float32),
                np.asarray([1, 8, 20, 20], dtype=np.float32),
            ),
        }


class TinyDrift(torch.nn.Module):
    """Implements only the public SI interface, avoiding a large U-Net in tests."""

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.weight = torch.nn.Parameter(torch.tensor(0.1))

    def conditioning(self, img_lr):
        return img_lr[:, : self.channels], img_lr[:, self.channels :]

    def coefficients(self, t):
        return (
            _schedule({"type": "linear", "start": 1, "end": 0}, t),
            _schedule({"type": "polynomial", "power": 2}, t),
            _schedule({"type": "linear", "start": 1, "end": 0}, t),
        )

    def forward(self, xt, x0, y_rest, t):
        assert x0.shape == xt.shape
        assert y_rest.shape[1] == 1
        assert t.shape == (xt.shape[0],)
        return self.weight * xt


def test_schedules_match_cdsi_endpoints_and_derivatives():
    t = torch.tensor([0.0, 0.5, 1.0])
    alpha, alpha_dot = _schedule({"type": "linear", "start": 1, "end": 0}, t)
    beta, beta_dot = _schedule({"type": "polynomial", "power": 2}, t)
    sigma, sigma_dot = _schedule({"type": "linear", "start": 1, "end": 0}, t)
    torch.testing.assert_close(alpha, torch.tensor([1.0, 0.5, 0.0]))
    torch.testing.assert_close(beta, torch.tensor([0.0, 0.25, 1.0]))
    torch.testing.assert_close(sigma, alpha)
    torch.testing.assert_close(alpha_dot, -torch.ones_like(t))
    torch.testing.assert_close(beta_dot, 2 * t)
    torch.testing.assert_close(sigma_dot, -torch.ones_like(t))


def test_channel_spec_uses_selected_archive_statistics_and_order():
    spec = build_channel_spec(
        FakeDataset(),
        ["temperature_2m", "eastward_wind_10m", "northward_wind_10m"],
        ["temperature_2m", "eastward_wind_10m", "northward_wind_10m"],
    )
    assert spec["base_input_indices"] == [1, 2, 3]
    assert spec["input_centers"] == [280, 1, -2]
    assert spec["output_centers"] == [282, 0, 0]
    with pytest.raises(ValueError, match="dataset order"):
        build_channel_spec(FakeDataset(), ["eastward_wind_10m"], ["eastward_wind_10m"])


def test_matching_conditioning_converts_units_and_removes_duplicate_channels():
    img_lr = torch.tensor([[[[7.0]], [[1.0]], [[-0.5]], [[2.0]]]])
    x0, y_rest = matching_conditioning(
        img_lr,
        (1, 2, 3),
        torch.tensor([280.0, 1.0, -2.0]).reshape(1, 3, 1, 1),
        torch.tensor([5.0, 4.0, 6.0]).reshape(1, 3, 1, 1),
        torch.tensor([282.0, 0.0, 0.0]).reshape(1, 3, 1, 1),
        torch.tensor([8.0, 20.0, 20.0]).reshape(1, 3, 1, 1),
    )
    torch.testing.assert_close(x0.flatten(), torch.tensor([0.375, -0.05, 0.5]))
    torch.testing.assert_close(y_rest.flatten(), torch.tensor([7.0]))


def test_loss_is_unreduced_and_backpropagates_for_one_and_three_channels():
    for channels in (1, 3):
        model = TinyDrift(channels)
        target = torch.randn(2, channels, 4, 4)
        conditioning = torch.randn(2, channels + 1, 4, 4)
        loss = StochasticInterpolantLoss()(model, target, conditioning)
        assert loss.shape == target.shape
        assert torch.isfinite(loss).all()
        loss.sum().backward()
        assert model.weight.grad is not None
        assert torch.isfinite(model.weight.grad)


def test_loss_uses_cdsi_drift_target(monkeypatch):
    monkeypatch.setattr(torch, "rand", lambda size, device: torch.full((size,), 0.25, device=device))
    monkeypatch.setattr(torch, "randn_like", lambda value: torch.ones_like(value))
    model = TinyDrift(1)
    x0 = torch.full((1, 1, 2, 2), 2.0)
    target = torch.full((1, 1, 2, 2), 4.0)
    conditioning = torch.cat((x0, torch.zeros_like(x0)), dim=1)
    loss = StochasticInterpolantLoss()(model, target, conditioning)
    xt = 0.75 * x0 + 0.25**2 * target + 0.5 * 0.75
    drift_target = -x0 + 2 * 0.25 * target - 0.5
    torch.testing.assert_close(loss, (0.1 * xt - drift_target).square())


def test_euler_maruyama_seeds_are_independent_of_batch_partition():
    model = TinyDrift(1).eval()
    conditioning = torch.zeros(1, 2, 4, 4)
    split = sample_ensemble(model, conditioning, [[11], [12]], num_steps=4)
    together = sample_ensemble(model, conditioning, [[11, 12]], num_steps=4)
    torch.testing.assert_close(split, together)
    assert not torch.allclose(split[0], split[1])


def test_wrapper_forward_and_checkpoint_round_trip(tmp_path):
    model = StochasticInterpolant(
        img_resolution=[8, 8],
        img_in_channels=2,
        img_out_channels=1,
        base_input_indices=[0],
        input_centers=[280.0],
        input_scales=[5.0],
        output_centers=[282.0],
        output_scales=[8.0],
        output_channel_names=["temperature_2m"],
        base_input_channel_names=["temperature_2m"],
        alpha_schedule={"type": "linear", "start": 1.0, "end": 0.0},
        beta_schedule={"type": "polynomial", "power": 2},
        sigma_schedule={"type": "linear", "start": 1.0, "end": 0.0},
        model_channels=32,
        channel_mult=[1, 1],
        num_blocks=1,
        attn_resolutions=[],
        N_grid_channels=4,
        gridtype="sinusoidal",
        embedding_type="fourier",
    ).eval()
    img_lr = torch.randn(1, 2, 8, 8)
    x0, y_rest = model.conditioning(img_lr)
    with torch.no_grad():
        expected = model(x0, x0, y_rest, torch.tensor([0.5]))
    assert expected.shape == (1, 1, 8, 8)
    path = str(tmp_path / "si.mdlus")
    model.save(path)
    loaded = Module.from_checkpoint(path, override_args={"use_apex_gn": False}).eval()
    assert isinstance(loaded, StochasticInterpolant)
    with torch.no_grad():
        actual = loaded(x0, x0, y_rest, torch.tensor([0.5]))
    torch.testing.assert_close(actual, expected)
