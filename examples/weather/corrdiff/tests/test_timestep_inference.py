"""Tests for the shared per-timestep inference helpers.

These back the fix for two divergences in evaluate.py: the example-event
figures used to draw a different ensemble than the one that was scored
(fixed seeds vs generation.seed_mode=timestamp) and returned unclamped
fields. Seed resolution and clamping now happen in one place, and these
tests pin that.
"""

import numpy as np
import pytest
import torch

from helpers.generate_helpers import (
    generate_ensemble,
    load_timestep_tensors,
    resolve_seed_batches,
)


class _FakeDataset:
    """Two channels on a 4x4 grid; denormalization pushes values negative."""

    def __init__(self, with_lead_time=False, as_numpy=True):
        self.with_lead_time = with_lead_time
        self.as_numpy = as_numpy

    def __getitem__(self, idx):
        target = np.full((2, 4, 4), float(idx), dtype=np.float32)
        inputs = np.full((3, 4, 4), float(idx) + 0.5, dtype=np.float32)
        if not self.as_numpy:
            target, inputs = torch.from_numpy(target), torch.from_numpy(inputs)
        if self.with_lead_time:
            return target, inputs, np.array([idx], dtype=np.int64)
        return target, inputs

    def denormalize_output(self, x):
        return np.asarray(x) * 10.0 - 5.0


@pytest.mark.parametrize("as_numpy", [True, False])
def test_load_timestep_tensors_shapes_and_batching(as_numpy):
    target, inputs, lead = load_timestep_tensors(
        _FakeDataset(as_numpy=as_numpy), 3, torch.device("cpu")
    )

    assert target.shape == (1, 2, 4, 4)
    assert inputs.shape == (1, 3, 4, 4)
    assert target.dtype == torch.float32 and inputs.dtype == torch.float32
    assert lead is None
    assert float(target[0, 0, 0, 0]) == 3.0


def test_load_timestep_tensors_batches_lead_time_when_present():
    _, _, lead = load_timestep_tensors(
        _FakeDataset(with_lead_time=True), 7, torch.device("cpu")
    )
    assert lead is not None
    assert lead.shape == (1, 1)


def test_resolve_seed_batches_fixed_mode_passes_through():
    batches = [np.array([1, 2]), np.array([3, 4])]
    assert resolve_seed_batches(batches, seed_mode="fixed") is batches


def test_resolve_seed_batches_timestamp_mode_varies_by_timestamp():
    import datetime

    batches = [np.array([0, 0]), np.array([0, 0])]
    kwargs = dict(seed_mode="timestamp", num_ensembles=4, seed_base=0)

    first = resolve_seed_batches(
        batches, timestamp=datetime.datetime(2005, 3, 1, 6), **kwargs
    )
    second = resolve_seed_batches(
        batches, timestamp=datetime.datetime(2005, 3, 2, 6), **kwargs
    )
    again = resolve_seed_batches(
        batches, timestamp=datetime.datetime(2005, 3, 1, 6), **kwargs
    )

    # Same shape as the fixed batches, so downstream batching is unchanged.
    assert len(first) == len(batches)
    assert sum(len(b) for b in first) == 4
    # Different timestamps draw different ensembles; the same one repeats.
    assert not np.array_equal(np.concatenate(first), np.concatenate(second))
    assert np.array_equal(np.concatenate(first), np.concatenate(again))


def test_resolve_seed_batches_rejects_unknown_mode():
    with pytest.raises(ValueError, match="seed_mode"):
        resolve_seed_batches([np.array([1])], seed_mode="random")


def test_resolve_seed_batches_timestamp_mode_requires_context():
    with pytest.raises(ValueError, match="requires both"):
        resolve_seed_batches([np.array([1])], seed_mode="timestamp")


def test_generate_ensemble_concatenates_members_in_batch_order():
    seen = []

    def fake_dropout_residual_step(*, latents_shape, seed, **_):
        seen.append(seed)
        return torch.full((latents_shape[0], 2, 4, 4), float(seed))

    import helpers.generate_helpers as gh

    original = gh.dropout_residual_step
    gh.dropout_residual_step = fake_dropout_residual_step
    try:
        members = generate_ensemble(
            net_res=object(),
            sampler_fn=None,
            use_dropout_residual=True,
            image_lr=torch.zeros(1, 3, 4, 4),
            img_shape=(4, 4),
            img_out_channels=2,
            device=torch.device("cpu"),
            mean_hr=None,
            lead_time_label=None,
            seed_batches=[np.array([11, 12]), np.array([21])],
            diffusion_kwargs={},
        )
    finally:
        gh.dropout_residual_step = original

    assert seen == [11, 21]
    assert members.shape == (3, 2, 4, 4)
    assert [float(m[0, 0, 0]) for m in members] == [11.0, 11.0, 21.0]


# ---------------------------------------------------------------------------
# Regression tests for the event-plot / metric-path divergence
# ---------------------------------------------------------------------------


def _run_timestep(monkeypatch, **overrides):
    """Drive evaluate._run_single_timestep with stub nets on CPU."""
    evaluate = pytest.importorskip("evaluate")
    import helpers.generate_helpers as gh

    monkeypatch.setattr(
        evaluate,
        "regression_step",
        lambda *, net, img_lr, latents_shape, lead_time_label: torch.zeros(latents_shape),
    )
    monkeypatch.setattr(
        gh,
        "dropout_residual_step",
        # Encode the seed into the values so the drawn ensemble is identifiable.
        # Modulo keeps it well inside float32 resolution.
        lambda *, latents_shape, seed, **_: torch.full(
            latents_shape, float(seed % 1000)
        ),
    )

    kwargs = dict(
        dataset=_FakeDataset(),
        dataset_idx=0,
        net_reg=object(),
        net_res=object(),
        sampler_fn=None,
        use_dropout_residual=True,
        img_shape=(4, 4),
        img_out_channels=2,
        device=torch.device("cpu"),
        hr_mean_conditioning=False,
        diffusion_kwargs={},
        seed_batches=[np.array([0, 0]), np.array([0, 0])],
    )
    kwargs.update(overrides)
    return evaluate._run_single_timestep(**kwargs)


def test_event_plot_path_honours_timestamp_seed_mode(monkeypatch):
    """seed_mode=timestamp must vary the drawn ensemble by timestep.

    Before the fix this path always used the caller's fixed seeds, so the
    plotted event was a different draw than the one the metrics scored.
    """
    import datetime

    ts_kwargs = dict(seed_mode="timestamp", num_ensembles=4, seed_base=0)
    _, first, _ = _run_timestep(
        monkeypatch, timestamp=datetime.datetime(2005, 3, 1, 6), **ts_kwargs
    )
    _, second, _ = _run_timestep(
        monkeypatch, timestamp=datetime.datetime(2005, 3, 2, 6), **ts_kwargs
    )
    _, fixed, _ = _run_timestep(monkeypatch)

    assert not np.array_equal(first, second)
    assert not np.array_equal(first, fixed)


def test_event_plot_path_clamps_nonnegative_channels(monkeypatch):
    """Returned fields must be clamped like the metric path.

    _FakeDataset.denormalize_output maps 0 -> -5, so an unclamped path yields
    negative precipitation in the plotted figure.
    """
    reg, ens, target = _run_timestep(monkeypatch, nonnegative_channels=[0])

    assert reg[0].min() >= 0.0
    assert ens[:, 0].min() >= 0.0
    assert target[0].min() >= 0.0
    # Channel 1 is not declared non-negative, so it is left alone.
    assert target[1].min() < 0.0


def test_event_plot_path_unclamped_without_channel_list(monkeypatch):
    """Omitting nonnegative_channels preserves the previous behaviour."""
    _, _, target = _run_timestep(monkeypatch)
    assert target[0].min() < 0.0
