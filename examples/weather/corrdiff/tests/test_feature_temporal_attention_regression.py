import os

import pytest
import torch

from physicsnemo import Module

from helpers.preconditioning import FeatureTemporalAttentionRegression


def _make_model(**kwargs):
    defaults = {
        "img_resolution": [32, 32],
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
    return FeatureTemporalAttentionRegression(**defaults)


def test_feature_attention_mixes_center_past_future_latents_and_invariant():
    model = _make_model()
    img_lr = torch.randn(2, 27, 32, 32)

    mixed = model.mix_conditioning(img_lr)

    assert mixed.shape == (2, 49, 32, 32)
    frames = img_lr.reshape(2, 3, 9, 32, 32)
    torch.testing.assert_close(mixed[:, -1:], frames[:, 1, 8:9])


def test_feature_attention_weights_sum_separately_for_past_and_future():
    model = _make_model()
    img_lr = torch.randn(2, 27, 32, 32)

    contexts, weights = model.compute_attention(img_lr, return_weights=True)

    assert set(contexts) == {"past", "future"}
    assert set(weights) == {"past", "future"}
    assert contexts["past"].shape == (2, 16, 32, 32)
    assert contexts["future"].shape == (2, 16, 32, 32)
    assert weights["past"].shape == (2, 4, 8, 8, 3, 3)
    assert weights["future"].shape == (2, 4, 8, 8, 3, 3)
    for source_weights in weights.values():
        torch.testing.assert_close(
            source_weights.sum(dim=(-1, -2)),
            torch.ones(2, 4, 8, 8),
            rtol=1e-6,
            atol=1e-6,
        )


def test_feature_attention_forward_output_shape():
    model = _make_model()
    x = torch.zeros(2, 1, 32, 32)
    img_lr = torch.randn(2, 27, 32, 32)

    out = model(x=x, img_lr=img_lr)

    assert out.shape == (2, 1, 32, 32)


def test_feature_attention_rejects_invalid_shapes_and_configs():
    with pytest.raises(ValueError, match="expected .*27 input channels"):
        _make_model(img_in_channels=26)
    with pytest.raises(ValueError, match="exactly 3 frames"):
        _make_model(num_frames=4, img_in_channels=36)
    with pytest.raises(ValueError, match="center_index=1"):
        _make_model(center_index=0)
    with pytest.raises(ValueError, match="divisible"):
        _make_model(embed_channels=10, num_heads=4)
    with pytest.raises(ValueError, match="positive"):
        _make_model(attention_stride=0)

    model = _make_model()
    with pytest.raises(ValueError, match="Expected 27 temporal input channels"):
        model.mix_conditioning(torch.randn(1, 26, 32, 32))


def test_feature_attention_checkpoint_round_trip(tmp_path):
    model = _make_model()
    ckpt_path = os.path.join(tmp_path, "feature_temporal_attention.mdlus")

    model.save(ckpt_path)
    loaded = Module.from_checkpoint(ckpt_path)

    img_lr = torch.randn(1, 27, 32, 32)
    x = torch.zeros(1, 1, 32, 32)
    assert loaded.mix_conditioning(img_lr).shape == (1, 49, 32, 32)
    assert loaded(x=x, img_lr=img_lr).shape == (1, 1, 32, 32)


def test_feature_attention_temporal_embeddings_break_swap_symmetry():
    torch.manual_seed(0)
    model = _make_model()
    img_lr = torch.randn(1, 27, 32, 32)
    frames = img_lr.reshape(1, 3, 9, 32, 32)
    swapped = frames.clone()
    swapped[:, 0] = frames[:, 2]
    swapped[:, 2] = frames[:, 0]
    swapped = swapped.reshape_as(img_lr)

    mixed = model.mix_conditioning(img_lr)
    swapped_mixed = model.mix_conditioning(swapped)

    assert not torch.allclose(mixed, swapped_mixed)
