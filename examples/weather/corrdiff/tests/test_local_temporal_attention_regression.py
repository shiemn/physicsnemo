import os

import pytest
import torch

from physicsnemo import Module

from helpers.preconditioning import LocalTemporalAttentionRegression


def _make_model(**kwargs):
    defaults = {
        "img_resolution": [16, 16],
        "img_in_channels": 27,
        "img_out_channels": 1,
        "model_channels": 8,
        "channel_mult": [1],
        "embed_channels": 16,
        "num_heads": 4,
        "attention_stride": 4,
        "window_radius": 1,
    }
    defaults.update(kwargs)
    return LocalTemporalAttentionRegression(**defaults)


def test_local_temporal_attention_mixes_to_center_context_and_invariant_channels():
    model = _make_model()
    img_lr = torch.randn(2, 27, 16, 16)

    mixed = model.mix_conditioning(img_lr)

    assert mixed.shape == (2, 17, 16, 16)
    torch.testing.assert_close(mixed[:, :8], img_lr[:, 9:17])
    torch.testing.assert_close(mixed[:, 16:17], img_lr[:, 17:18])


def test_local_temporal_attention_weights_sum_over_source_window_domain():
    model = _make_model()
    img_lr = torch.randn(2, 27, 16, 16)

    context, weights = model.compute_attention(img_lr, return_weights=True)

    assert context.shape == (2, 8, 16, 16)
    assert weights.shape == (2, 4, 4, 4, 2, 3, 3)
    torch.testing.assert_close(
        weights.sum(dim=(-1, -2, -3)),
        torch.ones(2, 4, 4, 4),
        rtol=1e-6,
        atol=1e-6,
    )


def test_local_temporal_attention_forward_output_shape():
    model = _make_model()
    x = torch.zeros(2, 1, 16, 16)
    img_lr = torch.randn(2, 27, 16, 16)

    out = model(x=x, img_lr=img_lr)

    assert out.shape == (2, 1, 16, 16)


def test_local_temporal_attention_rejects_invalid_shapes_and_heads():
    with pytest.raises(ValueError, match="expected .*27 input channels"):
        _make_model(img_in_channels=26)
    with pytest.raises(ValueError, match="divisible"):
        _make_model(embed_channels=10, num_heads=4)

    model = _make_model()
    with pytest.raises(ValueError, match="Expected 27 temporal input channels"):
        model.mix_conditioning(torch.randn(1, 26, 16, 16))


def test_local_temporal_attention_checkpoint_round_trip(tmp_path):
    model = _make_model()
    ckpt_path = os.path.join(tmp_path, "local_temporal_attention.mdlus")

    model.save(ckpt_path)
    loaded = Module.from_checkpoint(ckpt_path)

    img_lr = torch.randn(1, 27, 16, 16)
    x = torch.zeros(1, 1, 16, 16)
    assert loaded.mix_conditioning(img_lr).shape == (1, 17, 16, 16)
    assert loaded(x=x, img_lr=img_lr).shape == (1, 1, 16, 16)
