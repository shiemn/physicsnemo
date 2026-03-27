# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Tests for EDM2 model loading via helpers.generate_helpers.load_model.

Verifies that:
  - .mdlus checkpoints can be saved and reloaded via Module.from_checkpoint
  - .pt state dicts can be saved and reloaded into EDM2PrecondSuperResolution
  - Training checkpoints (with optimizer_state_dict) are rejected with a clear error
  - load_model handles read-only properties (use_fp16) without crashing
  - Round-trip save/load produces identical forward outputs

Run:
    pytest tests/test_edm2_loading.py -v
"""

import os
import tempfile
from types import SimpleNamespace

import pytest
import torch


# ── Fixtures ────────────────────────────────────────────────────────────────

# Minimal EDM2 kwargs for a tiny model (fast to instantiate, no GPU needed)
TINY_EDM2_KWARGS = {
    "img_resolution": [16, 16],
    "img_in_channels": 3,
    "img_out_channels": 1,
    "use_fp16": False,
    "sigma_data": 0.5,
    "model_channels": 16,
    "channel_mult": [1, 2],
    "num_blocks": 1,
    "attn_resolutions": [],
    "N_grid_channels": 4,
    "gridtype": "sinusoidal",
    "grid_mp_balance": 0.5,
    "dropout": 0.0,
    "res_balance": 0.3,
    "attn_balance": 0.3,
    "clip_act": 256,
    "channels_per_head": 16,
}


@pytest.fixture
def tiny_model():
    """Create a small EDM2 model for testing."""
    from helpers.edm2_preconditioning import EDM2PrecondSuperResolution

    return EDM2PrecondSuperResolution(**TINY_EDM2_KWARGS)


@pytest.fixture
def saved_pt(tiny_model, tmp_path):
    """Save a tiny model's state dict as a .pt file and return the path."""
    path = str(tmp_path / "EDM2PrecondSuperResolution.0.100000.pt")
    torch.save(tiny_model.state_dict(), path)
    return path


@pytest.fixture
def saved_training_ckpt(tmp_path):
    """Save a fake training checkpoint (optimizer state) as a .pt file."""
    path = str(tmp_path / "checkpoint.0.100000.pt")
    torch.save(
        {"optimizer_state_dict": {"param_groups": []}, "epoch": 100000},
        path,
    )
    return path


# ── Tests ───────────────────────────────────────────────────────────────────


class TestEDM2MdlusCheckpoint:
    """Test .mdlus save/load round-trip via physicsnemo.Module."""

    def test_mdlus_roundtrip(self, tiny_model, tmp_path):
        """Save as .mdlus and reload via Module.from_checkpoint — outputs must match."""
        from physicsnemo import Module

        path = str(tmp_path / "test.mdlus")
        tiny_model.eval()
        tiny_model.save(path)

        loaded = Module.from_checkpoint(path)
        loaded.eval()

        x = torch.randn(1, 1, 16, 16)
        img_lr = torch.randn(1, 3, 16, 16)
        sigma = torch.tensor([0.5])

        with torch.no_grad():
            out_orig = tiny_model(x, img_lr, sigma)
            out_loaded = loaded(x, img_lr, sigma)

        torch.testing.assert_close(out_orig, out_loaded)

    def test_mdlus_restores_class(self, tiny_model, tmp_path):
        """Module.from_checkpoint returns an EDM2PrecondSuperResolution instance."""
        from physicsnemo import Module
        from helpers.edm2_preconditioning import EDM2PrecondSuperResolution

        path = str(tmp_path / "test.mdlus")
        tiny_model.save(path)
        loaded = Module.from_checkpoint(path)
        assert isinstance(loaded, EDM2PrecondSuperResolution)

    def test_load_model_helper_mdlus(self, tiny_model, tmp_path):
        """load_model loads a .mdlus checkpoint without needing edm2_kwargs."""
        from helpers.generate_helpers import load_model

        path = str(tmp_path / "test.mdlus")
        tiny_model.save(path)

        perf_cfg = SimpleNamespace(use_fp16=False, use_apex_gn=False, profile_mode=False)
        # No edm2_kwargs needed — architecture is embedded in the checkpoint
        net = load_model(path, device="cpu", perf_cfg=perf_cfg, edm2_kwargs=None)
        assert net is not None
        assert net.training is False


class TestEDM2StateDict:
    """Test raw .pt state dict save/load round-trip."""

    def test_load_state_dict_roundtrip(self, tiny_model, saved_pt):
        """A fresh model loaded from saved state dict produces identical output."""
        from helpers.edm2_preconditioning import EDM2PrecondSuperResolution

        loaded = EDM2PrecondSuperResolution(**TINY_EDM2_KWARGS)
        loaded.load_state_dict(torch.load(saved_pt, map_location="cpu", weights_only=False))
        loaded.eval()
        tiny_model.eval()

        x = torch.randn(1, 1, 16, 16)
        img_lr = torch.randn(1, 3, 16, 16)
        sigma = torch.tensor([0.5])

        with torch.no_grad():
            out_orig = tiny_model(x, img_lr, sigma)
            out_loaded = loaded(x, img_lr, sigma)

        torch.testing.assert_close(out_orig, out_loaded)

    def test_use_fp16_is_readonly_property(self, tiny_model):
        """use_fp16 is a read-only property — setting it should raise."""
        with pytest.raises((AttributeError, TypeError)):
            tiny_model.use_fp16 = True


class TestLoadModelHelper:
    """Test the load_model helper from helpers.generate_helpers."""

    def test_load_pt_checkpoint(self, saved_pt):
        """load_model successfully loads a .pt state dict with edm2_kwargs."""
        from helpers.generate_helpers import load_model

        perf_cfg = SimpleNamespace(
            use_fp16=False,
            use_apex_gn=False,
            profile_mode=False,
        )
        net = load_model(saved_pt, device="cpu", perf_cfg=perf_cfg, edm2_kwargs=TINY_EDM2_KWARGS)
        assert net.training is False  # should be in eval mode

    def test_load_pt_without_edm2_kwargs_raises(self, saved_pt):
        """load_model raises ValueError when loading .pt without edm2_kwargs."""
        from helpers.generate_helpers import load_model

        perf_cfg = SimpleNamespace(use_fp16=False, use_apex_gn=False, profile_mode=False)
        with pytest.raises(ValueError, match="raw .pt state dict"):
            load_model(saved_pt, device="cpu", perf_cfg=perf_cfg, edm2_kwargs=None)

    def test_training_checkpoint_rejected(self, saved_training_ckpt):
        """load_model rejects training checkpoints (checkpoint.0.*.pt)."""
        from helpers.generate_helpers import load_model

        perf_cfg = SimpleNamespace(use_fp16=False, use_apex_gn=False, profile_mode=False)
        with pytest.raises(ValueError, match="training checkpoint"):
            load_model(
                saved_training_ckpt, device="cpu", perf_cfg=perf_cfg, edm2_kwargs=TINY_EDM2_KWARGS
            )

    def test_readonly_use_fp16_no_crash(self, saved_pt):
        """load_model does not crash on models with read-only use_fp16."""
        from helpers.generate_helpers import load_model

        perf_cfg = SimpleNamespace(
            use_fp16=True,  # would crash if we try to set it
            use_apex_gn=False,
            profile_mode=False,
        )
        # Should not raise
        net = load_model(saved_pt, device="cpu", perf_cfg=perf_cfg, edm2_kwargs=TINY_EDM2_KWARGS)
        assert net is not None
