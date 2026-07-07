import os

import pytest
import torch

from physicsnemo import Module

from helpers.preconditioning import MidResTemporalAdapterRegression


def _make_model(**kwargs):
    defaults = {
        "img_resolution": [32, 32],
        "img_in_channels": 27,
        "img_out_channels": 1,
        "model_channels": 8,
        "channel_mult": [1, 2, 2],
        "num_blocks": 1,
        "dropout": 0.0,
        "adapter_hidden_channels": 8,
        "adapter_hook_names": ["16x16_block0", "8x8_block0"],
    }
    defaults.update(kwargs)
    return MidResTemporalAdapterRegression(**defaults)


def test_midres_adapter_accepts_sym3h_temporal_stack():
    model = _make_model()

    assert model.img_in_channels == 27
    assert model.num_frames == 3
    assert model.center_index == 1
    assert set(model.adapters.keys()) == {"16x16_block0", "8x8_block0"}


def test_midres_adapter_forward_output_shape():
    model = _make_model()
    x = torch.zeros(2, 1, 32, 32)
    img_lr = torch.randn(2, 27, 32, 32)

    out = model(x=x, img_lr=img_lr)

    assert out.shape == (2, 1, 32, 32)


def test_midres_adapter_zero_init_is_noop_path():
    torch.manual_seed(0)
    model = _make_model()
    x = torch.zeros(1, 1, 32, 32)
    img_lr = torch.randn(1, 27, 32, 32)

    out_default = model(x=x, img_lr=img_lr)
    model.adapter_scale = 0.0
    out_disabled = model(x=x, img_lr=img_lr)

    torch.testing.assert_close(out_default, out_disabled, rtol=1e-6, atol=1e-6)
    for adapter in model.adapters.values():
        assert torch.count_nonzero(adapter.output_proj.weight) == 0
        assert torch.count_nonzero(adapter.output_proj.bias) == 0


def test_midres_adapter_temporal_features_use_past_center_future_differences():
    model = _make_model()
    img_lr = torch.zeros(1, 27, 32, 32)
    frames = img_lr.reshape(1, 3, 9, 32, 32)
    frames[:, 0, :8] = 1.0
    frames[:, 1, :8] = 3.0
    frames[:, 2, :8] = 7.0

    temporal = model.temporal_features_for_resolution(
        img_lr, size=(16, 16), dtype=torch.float32
    )

    assert temporal.shape == (1, 24, 16, 16)
    torch.testing.assert_close(temporal[:, :8], torch.full((1, 8, 16, 16), -2.0))
    torch.testing.assert_close(temporal[:, 8:16], torch.full((1, 8, 16, 16), 4.0))
    torch.testing.assert_close(temporal[:, 16:24], torch.full((1, 8, 16, 16), 6.0))


def test_midres_adapter_rejects_invalid_shapes_and_configs():
    with pytest.raises(ValueError, match="expected .*27 input channels"):
        _make_model(img_in_channels=26)
    with pytest.raises(ValueError, match="exactly 3 frames"):
        _make_model(num_frames=4, img_in_channels=36)
    with pytest.raises(ValueError, match="center_index=1"):
        _make_model(center_index=0)
    with pytest.raises(ValueError, match="adapter hooks not found"):
        _make_model(adapter_hook_names=["256x256_block3"])

    model = _make_model()
    with pytest.raises(ValueError, match="Expected 27 temporal input channels"):
        model(x=torch.zeros(1, 1, 32, 32), img_lr=torch.randn(1, 26, 32, 32))


def test_midres_adapter_hooks_are_called():
    model = _make_model()
    calls = {name: 0 for name in model.adapters}

    handles = []
    for name, adapter in model.adapters.items():
        handles.append(
            adapter.register_forward_hook(
                lambda _module, _inputs, _output, hook_name=name: calls.__setitem__(
                    hook_name, calls[hook_name] + 1
                )
            )
        )

    try:
        model(x=torch.zeros(1, 1, 32, 32), img_lr=torch.randn(1, 27, 32, 32))
    finally:
        for handle in handles:
            handle.remove()

    assert calls == {"16x16_block0": 1, "8x8_block0": 1}


def test_midres_adapter_checkpoint_round_trip(tmp_path):
    model = _make_model()
    ckpt_path = os.path.join(tmp_path, "midres_temporal_adapter.mdlus")

    model.save(ckpt_path)
    loaded = Module.from_checkpoint(ckpt_path)

    img_lr = torch.randn(1, 27, 32, 32)
    x = torch.zeros(1, 1, 32, 32)
    assert loaded(x=x, img_lr=img_lr).shape == (1, 1, 32, 32)
