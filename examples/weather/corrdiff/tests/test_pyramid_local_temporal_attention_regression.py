import os

import pytest
import torch

from physicsnemo import Module

from helpers.preconditioning import PyramidLocalTemporalAttentionRegression


def _make_model(**kwargs):
    defaults = {
        "img_resolution": [32, 32],
        "img_in_channels": 27,
        "img_out_channels": 1,
        "model_channels": 8,
        "channel_mult": [1],
        "levels": [
            {
                "name": "local",
                "embed_channels": 16,
                "num_heads": 4,
                "attention_stride": 4,
                "window_radius": 1,
            },
            {
                "name": "broad",
                "embed_channels": 16,
                "num_heads": 4,
                "attention_stride": 8,
                "window_radius": 2,
            },
        ],
        "fusion_channels": 16,
    }
    defaults.update(kwargs)
    return PyramidLocalTemporalAttentionRegression(**defaults)


def test_pyramid_attention_mixes_to_center_context_and_invariant_channels():
    model = _make_model()
    img_lr = torch.randn(2, 27, 32, 32)

    mixed = model.mix_conditioning(img_lr)

    assert mixed.shape == (2, 17, 32, 32)
    torch.testing.assert_close(mixed[:, :8], img_lr[:, 9:17])
    torch.testing.assert_close(mixed[:, 16:17], img_lr[:, 17:18])


def test_pyramid_attention_weights_sum_per_level():
    model = _make_model()
    img_lr = torch.randn(2, 27, 32, 32)

    context, weights_by_level = model.compute_attention(img_lr, return_weights=True)

    assert context.shape == (2, 8, 32, 32)
    assert set(weights_by_level) == {"local", "broad"}
    assert weights_by_level["local"].shape == (2, 4, 8, 8, 2, 3, 3)
    assert weights_by_level["broad"].shape == (2, 4, 4, 4, 2, 5, 5)
    for weights in weights_by_level.values():
        torch.testing.assert_close(
            weights.sum(dim=(-1, -2, -3)),
            torch.ones_like(weights[..., 0, 0, 0]),
            rtol=1e-6,
            atol=1e-6,
        )


def test_pyramid_attention_forward_output_shape():
    model = _make_model()
    x = torch.zeros(2, 1, 32, 32)
    img_lr = torch.randn(2, 27, 32, 32)

    out = model(x=x, img_lr=img_lr)

    assert out.shape == (2, 1, 32, 32)


def test_pyramid_attention_rejects_invalid_shapes_and_configs():
    with pytest.raises(ValueError, match="expected .*27 input channels"):
        _make_model(img_in_channels=26)
    with pytest.raises(ValueError, match="divisible"):
        _make_model(
            levels=[
                {
                    "name": "bad_heads",
                    "embed_channels": 10,
                    "num_heads": 4,
                    "attention_stride": 4,
                    "window_radius": 1,
                }
            ]
        )
    with pytest.raises(ValueError, match="positive"):
        _make_model(
            levels=[
                {
                    "name": "bad_stride",
                    "embed_channels": 16,
                    "num_heads": 4,
                    "attention_stride": 0,
                    "window_radius": 1,
                }
            ]
        )
    with pytest.raises(ValueError, match="Duplicate"):
        _make_model(
            levels=[
                {"name": "same", "attention_stride": 4},
                {"name": "same", "attention_stride": 8},
            ]
        )

    model = _make_model()
    with pytest.raises(ValueError, match="Expected 27 temporal input channels"):
        model.mix_conditioning(torch.randn(1, 26, 32, 32))


def test_pyramid_attention_checkpoint_round_trip(tmp_path):
    model = _make_model()
    ckpt_path = os.path.join(tmp_path, "pyramid_local_temporal_attention.mdlus")

    model.save(ckpt_path)
    loaded = Module.from_checkpoint(ckpt_path)

    img_lr = torch.randn(1, 27, 32, 32)
    x = torch.zeros(1, 1, 32, 32)
    assert loaded.mix_conditioning(img_lr).shape == (1, 17, 32, 32)
    assert loaded(x=x, img_lr=img_lr).shape == (1, 1, 32, 32)


def test_pyramid_attention_temporal_embeddings_break_source_frame_symmetry():
    torch.manual_seed(0)
    model = _make_model(use_temporal_embeddings=True)
    assert model.use_temporal_embeddings is True
    assert all(
        level.temporal_embeddings is not None
        for level in model.attention_levels
    )

    img_lr = torch.randn(1, 27, 32, 32)
    frames = img_lr.reshape(1, 3, 9, 32, 32)
    swapped = frames.clone()
    swapped[:, 0] = frames[:, 2]
    swapped[:, 2] = frames[:, 0]
    swapped = swapped.reshape_as(img_lr)

    context = model.compute_attention(img_lr, return_weights=False)
    swapped_context = model.compute_attention(swapped, return_weights=False)

    assert not torch.allclose(context, swapped_context)
