import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers.dropout_residual import DropoutResidualCRPSLoss, dropout_residual_step


class FrozenRegression(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.25))

    def forward(self, x, img_lr, **_):
        return x + self.bias


class ResidualNet(torch.nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.dropout = torch.nn.Dropout2d(dropout)
        self.proj = torch.nn.Conv2d(4, 1, kernel_size=1)
        self.last_batch = None

    def forward(self, x, img_lr, **_):
        self.last_batch = x.shape[0]
        return self.proj(self.dropout(torch.cat((x, img_lr), dim=1)))


class FunctionalDropoutResidualNet(torch.nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.proj = torch.nn.Conv2d(4, 1, kernel_size=1)

    def forward(self, x, img_lr, **_):
        h = torch.cat((x, img_lr), dim=1)
        h = torch.nn.functional.dropout(h, p=self.dropout, training=self.training)
        return self.proj(h)


def test_dropout_residual_crps_uses_batch_size_as_ensemble_size():
    batch_size_per_gpu = 4
    regression = FrozenRegression()
    residual = ResidualNet(dropout=0.0)
    loss_fn = DropoutResidualCRPSLoss(
        regression_net=regression,
        ensemble_size=batch_size_per_gpu,
        hr_mean_conditioning=True,
    )

    img_clean = torch.randn(batch_size_per_gpu, 1, 8, 8)
    img_lr = torch.randn(batch_size_per_gpu, 2, 8, 8)
    loss = loss_fn(net=residual, img_clean=img_clean, img_lr=img_lr)

    assert loss.shape == (batch_size_per_gpu,)
    assert residual.last_batch == batch_size_per_gpu * batch_size_per_gpu
    assert loss_fn.latest_components["dropout_ensemble_size"].item() == batch_size_per_gpu


def test_dropout_residual_crps_can_use_ensemble_larger_than_batch():
    regression = FrozenRegression()
    residual = ResidualNet(dropout=0.0)
    loss_fn = DropoutResidualCRPSLoss(
        regression_net=regression,
        ensemble_size=2,
        hr_mean_conditioning=True,
    )

    img_clean = torch.randn(1, 1, 8, 8)
    img_lr = torch.randn(1, 2, 8, 8)
    loss = loss_fn(net=residual, img_clean=img_clean, img_lr=img_lr)

    assert loss.shape == (1,)
    assert residual.last_batch == 2
    assert loss_fn.latest_components["dropout_ensemble_size"].item() == 2


def test_dropout_residual_crps_backprops_only_to_residual_model():
    regression = FrozenRegression()
    residual = ResidualNet(dropout=0.0)
    loss_fn = DropoutResidualCRPSLoss(
        regression_net=regression,
        ensemble_size=2,
        hr_mean_conditioning=True,
    )

    img_clean = torch.randn(2, 1, 8, 8)
    img_lr = torch.randn(2, 2, 8, 8)
    loss_fn(net=residual, img_clean=img_clean, img_lr=img_lr).sum().backward()

    assert regression.bias.grad is None
    assert residual.proj.weight.grad is not None
    assert torch.count_nonzero(residual.proj.weight.grad) > 0


def test_dropout_residual_step_generates_stochastic_members_with_seed_control():
    torch.manual_seed(0)
    residual = ResidualNet(dropout=0.75)
    residual.eval()
    img_lr = torch.randn(1, 2, 8, 8)
    mean_hr = torch.randn(1, 1, 8, 8)

    out_a = dropout_residual_step(
        residual,
        img_lr=img_lr,
        latents_shape=(4, 1, 8, 8),
        mean_hr=mean_hr,
        seed=123,
    )
    out_b = dropout_residual_step(
        residual,
        img_lr=img_lr,
        latents_shape=(4, 1, 8, 8),
        mean_hr=mean_hr,
        seed=123,
    )
    out_c = dropout_residual_step(
        residual,
        img_lr=img_lr,
        latents_shape=(4, 1, 8, 8),
        mean_hr=mean_hr,
        seed=124,
    )

    torch.testing.assert_close(out_a, out_b)
    assert not torch.allclose(out_a, out_c)
    assert not torch.allclose(out_a[0], out_a[1])
    assert residual.training is False


def test_dropout_residual_step_enables_functional_dropout_blocks():
    torch.manual_seed(0)
    residual = FunctionalDropoutResidualNet(dropout=0.75)
    residual.eval()
    img_lr = torch.randn(1, 2, 8, 8)
    mean_hr = torch.randn(1, 1, 8, 8)

    out = dropout_residual_step(
        residual,
        img_lr=img_lr,
        latents_shape=(4, 1, 8, 8),
        mean_hr=mean_hr,
        seed=123,
    )

    assert not torch.allclose(out[0], out[1])
    assert residual.training is False
