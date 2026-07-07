import pytest
import torch

from helpers.preconditioning import TemporalCorrectionRegression


def _make_model(**kwargs):
    defaults = {
        "img_resolution": [8, 8],
        "img_in_channels": 27,
        "img_out_channels": 1,
        "model_channels": 8,
        "channel_mult": [1],
    }
    defaults.update(kwargs)
    return TemporalCorrectionRegression(**defaults)


def test_temporal_correction_mixes_to_center_correction_and_invariant_channels():
    model = _make_model()
    img_lr = torch.randn(2, 27, 8, 8)

    mixed = model.mix_conditioning(img_lr)

    assert mixed.shape == (2, 17, 8, 8)
    torch.testing.assert_close(mixed[:, :8], img_lr[:, 9:17])
    torch.testing.assert_close(mixed[:, 16:17], img_lr[:, 17:18])


def test_temporal_correction_zero_init_outputs_zero_correction():
    model = _make_model()
    img_lr = torch.randn(2, 27, 8, 8)

    mixed = model.mix_conditioning(img_lr)

    torch.testing.assert_close(mixed[:, 8:16], torch.zeros_like(mixed[:, 8:16]))


def test_temporal_correction_rejects_wrong_channel_count():
    with pytest.raises(ValueError, match="expected .*27 input channels"):
        _make_model(img_in_channels=26)

    model = _make_model()
    with pytest.raises(ValueError, match="Expected 27 temporal input channels"):
        model.mix_conditioning(torch.randn(1, 26, 8, 8))


def test_temporal_correction_rejects_missing_center_frame():
    with pytest.raises(ValueError, match="center_index"):
        _make_model(center_index=3)
