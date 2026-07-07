# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Climatological evaluation accumulator for the paper-protocol eval flow.

Implements the JAMES-paper Section-3.1 climatological assessment:

* Distributional statistics over the whole evaluation period, computed as the
  paper's *expected value of the statistic*: each ensemble member's statistic is
  computed over its full collection of day-gridpoints, then averaged across
  members.  Reported statistics:

    - Dry%      percentage of dry day-gridpoints   (value < dry_threshold mm)
    - Mean      mean of wet day-gridpoints          (value > wet_threshold mm)
    - SD        std of wet day-gridpoints
    - Median    median of wet day-gridpoints
    - P99       99th percentile of wet day-gridpoints

  Relative bias vs. the reference (RCM truth) is reported as
  ``100 * (model - ref) / ref`` alongside the raw model and reference values.

* Each statistic is computed twice: on the **raw** field and on a **3x3
  moving-window average** of the field (``scipy.ndimage.uniform_filter``).  The
  smoothed version captures the compounded spatial aspect of each statistic, as
  proposed in the paper.

* Per-gridpoint **bias maps** for Mean and SD (exact, from per-pixel running
  sums).  Per-pixel Median/P99 maps are optional and gated behind
  ``compute_quantile_maps`` (per-pixel histograms; memory grows with map_bins).

This module is purely additive — it does not modify or depend on the existing
``MetricsAccumulator``.  Inputs are **denormalized (physical, mm)** single-channel
fields, matching the units stored in the prediction NetCDF files.

Ensemble size is not fixed: every statistic is valid for any member count N>=1
(Median/P99 come from histograms; degenerate single-member ensembles are fine).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter


def moving_average_2d(field: np.ndarray, size: int = 3) -> np.ndarray:
    """Apply a 2-D moving-average (box) filter over the last two axes.

    Args:
        field: array with spatial dims as the last two axes, shape (..., H, W).
        size:  moving-window size (paper uses 3).

    Returns:
        Smoothed array of the same shape.  Edge handling uses mode="nearest".
    """
    field = np.asarray(field, dtype=np.float64)
    if field.ndim == 2:
        return uniform_filter(field, size=size, mode="nearest")
    # Apply per leading-index 2-D slice without smoothing across channels/members.
    out = np.empty_like(field)
    flat = field.reshape(-1, field.shape[-2], field.shape[-1])
    out_flat = out.reshape(-1, field.shape[-2], field.shape[-1])
    for i in range(flat.shape[0]):
        out_flat[i] = uniform_filter(flat[i], size=size, mode="nearest")
    return out


def _hist_quantile(counts: np.ndarray, lo: float, hi: float, q: float) -> float:
    """Estimate a quantile from a 1-D histogram via linear CDF interpolation.

    Args:
        counts: histogram counts over ``len(counts)`` equal-width bins in [lo, hi].
        lo, hi: histogram range.
        q:      quantile level in [0, 1].

    Returns:
        Interpolated quantile value, or NaN if the histogram is empty.
    """
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return float("nan")
    nbins = counts.shape[0]
    edges = np.linspace(lo, hi, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    cdf = np.cumsum(counts) / total
    return float(np.interp(q, cdf, centers))


def _relbias(model: float, ref: float) -> float:
    if not np.isfinite(model) or not np.isfinite(ref) or ref == 0.0:
        return float("nan")
    return 100.0 * (model - ref) / ref


class _StatState:
    """Accumulators for a single window type (raw or 3x3-smoothed)."""

    def __init__(
        self,
        img_shape: tuple[int, int],
        dry_threshold: float,
        wet_threshold: float,
        hist_lo: float,
        hist_hi: float,
        hist_bins: int,
        compute_quantile_maps: bool,
        map_bins: int,
    ):
        self.H, self.W = img_shape
        self.dry_threshold = dry_threshold
        self.wet_threshold = wet_threshold
        self.hist_lo = hist_lo
        self.hist_hi = hist_hi
        self.hist_bins = hist_bins
        self.compute_quantile_maps = compute_quantile_maps
        self.map_bins = map_bins

        self.n_members: int | None = None  # set lazily on first model update

        # Per-member domain-wide accumulators (allocated lazily once N is known)
        self.dry_count = None       # (M,)
        self.total_count = None     # (M,)
        self.wet_count = None       # (M,)
        self.wet_sum = None         # (M,)
        self.wet_sumsq = None       # (M,)
        self.wet_hist = None        # (M, hist_bins)

        # Reference (truth) domain-wide accumulators
        self.ref_dry_count = 0.0
        self.ref_total_count = 0.0
        self.ref_wet_count = 0.0
        self.ref_wet_sum = 0.0
        self.ref_wet_sumsq = 0.0
        self.ref_wet_hist = np.zeros(hist_bins, dtype=np.float64)

        # Per-gridpoint map accumulators (pooled over members for the model)
        self.model_sum = np.zeros((self.H, self.W), dtype=np.float64)
        self.model_sumsq = np.zeros((self.H, self.W), dtype=np.float64)
        self.model_dry = np.zeros((self.H, self.W), dtype=np.float64)  # dry-count
        self.model_n = 0.0  # number of (member, time) fields pooled
        self.ref_sum = np.zeros((self.H, self.W), dtype=np.float64)
        self.ref_sumsq = np.zeros((self.H, self.W), dtype=np.float64)
        self.ref_dry = np.zeros((self.H, self.W), dtype=np.float64)    # dry-count
        self.ref_n = 0.0    # number of time fields

        # Optional per-pixel histograms for Median/P99 maps
        self.model_pix_hist = (
            np.zeros((self.H * self.W, map_bins), dtype=np.float32)
            if compute_quantile_maps
            else None
        )
        self.ref_pix_hist = (
            np.zeros((self.H * self.W, map_bins), dtype=np.float32)
            if compute_quantile_maps
            else None
        )

    # ------------------------------------------------------------------
    def _wet_hist_counts(self, values: np.ndarray) -> np.ndarray:
        wet = values[values > self.wet_threshold]
        if wet.size == 0:
            return np.zeros(self.hist_bins, dtype=np.float64)
        idx = np.floor(
            (wet - self.hist_lo) / max(self.hist_hi - self.hist_lo, 1e-12) * self.hist_bins
        ).astype(np.int64)
        np.clip(idx, 0, self.hist_bins - 1, out=idx)
        return np.bincount(idx, minlength=self.hist_bins).astype(np.float64)

    def _pix_bin_idx(self, field_flat: np.ndarray) -> np.ndarray:
        idx = np.floor(
            (field_flat - self.hist_lo) / max(self.hist_hi - self.hist_lo, 1e-12) * self.map_bins
        ).astype(np.int64)
        return np.clip(idx, 0, self.map_bins - 1)

    def update_model(self, pred: np.ndarray) -> None:
        """Accumulate one timestep of model fields. pred: (M, H, W) physical units."""
        pred = np.asarray(pred, dtype=np.float64)
        M = pred.shape[0]
        if self.n_members is None:
            self.n_members = M
            self.dry_count = np.zeros(M, dtype=np.float64)
            self.total_count = np.zeros(M, dtype=np.float64)
            self.wet_count = np.zeros(M, dtype=np.float64)
            self.wet_sum = np.zeros(M, dtype=np.float64)
            self.wet_sumsq = np.zeros(M, dtype=np.float64)
            self.wet_hist = np.zeros((M, self.hist_bins), dtype=np.float64)
        elif M != self.n_members:
            raise ValueError(
                f"Inconsistent ensemble size: got {M}, expected {self.n_members}"
            )

        npix = self.H * self.W
        for m in range(M):
            f = pred[m]
            wet_mask = f > self.wet_threshold
            wet_vals = f[wet_mask]
            self.dry_count[m] += np.count_nonzero(f < self.dry_threshold)
            self.total_count[m] += npix
            self.wet_count[m] += wet_vals.size
            self.wet_sum[m] += wet_vals.sum()
            self.wet_sumsq[m] += np.square(wet_vals).sum()
            self.wet_hist[m] += self._wet_hist_counts(f)

        # Per-gridpoint pooled map accumulators
        self.model_sum += pred.sum(axis=0)
        self.model_sumsq += np.square(pred).sum(axis=0)
        self.model_dry += (pred < self.dry_threshold).sum(axis=0)
        self.model_n += M
        if self.model_pix_hist is not None:
            flat = pred.reshape(M, npix)
            for m in range(M):
                idx = self._pix_bin_idx(flat[m])
                self.model_pix_hist[np.arange(npix), idx] += 1.0

    def update_ref(self, target: np.ndarray) -> None:
        """Accumulate one timestep of the reference field. target: (H, W)."""
        t = np.asarray(target, dtype=np.float64)
        wet_mask = t > self.wet_threshold
        wet_vals = t[wet_mask]
        self.ref_dry_count += np.count_nonzero(t < self.dry_threshold)
        self.ref_total_count += t.size
        self.ref_wet_count += wet_vals.size
        self.ref_wet_sum += wet_vals.sum()
        self.ref_wet_sumsq += np.square(wet_vals).sum()
        self.ref_wet_hist += self._wet_hist_counts(t)

        self.ref_sum += t
        self.ref_sumsq += np.square(t)
        self.ref_dry += (t < self.dry_threshold)
        self.ref_n += 1
        if self.ref_pix_hist is not None:
            idx = self._pix_bin_idx(t.reshape(-1))
            self.ref_pix_hist[np.arange(self.H * self.W), idx] += 1.0

    # ------------------------------------------------------------------
    def _member_stats(self) -> dict:
        """Per-member statistics arrays (length M)."""
        dry_pct = 100.0 * self.dry_count / np.maximum(self.total_count, 1.0)
        wet_mean = np.divide(
            self.wet_sum, self.wet_count,
            out=np.full_like(self.wet_sum, np.nan), where=self.wet_count > 0,
        )
        var = np.divide(
            self.wet_sumsq, self.wet_count,
            out=np.full_like(self.wet_sumsq, np.nan), where=self.wet_count > 0,
        ) - np.square(wet_mean)
        wet_sd = np.sqrt(np.clip(var, 0.0, None))
        median = np.array([
            _hist_quantile(self.wet_hist[m], self.hist_lo, self.hist_hi, 0.5)
            for m in range(self.n_members)
        ])
        p99 = np.array([
            _hist_quantile(self.wet_hist[m], self.hist_lo, self.hist_hi, 0.99)
            for m in range(self.n_members)
        ])
        return {"dry_pct": dry_pct, "mean": wet_mean, "sd": wet_sd,
                "median": median, "p99": p99}

    def _ref_stats(self) -> dict:
        dry_pct = 100.0 * self.ref_dry_count / max(self.ref_total_count, 1.0)
        mean = self.ref_wet_sum / self.ref_wet_count if self.ref_wet_count > 0 else float("nan")
        var = (self.ref_wet_sumsq / self.ref_wet_count - mean ** 2) if self.ref_wet_count > 0 else float("nan")
        sd = float(np.sqrt(max(var, 0.0))) if np.isfinite(var) else float("nan")
        median = _hist_quantile(self.ref_wet_hist, self.hist_lo, self.hist_hi, 0.5)
        p99 = _hist_quantile(self.ref_wet_hist, self.hist_lo, self.hist_hi, 0.99)
        return {"dry_pct": dry_pct, "mean": mean, "sd": sd, "median": median, "p99": p99}

    def table_rows(self) -> list[dict]:
        """Return one row per statistic with model (expected), reference and rel-bias."""
        if self.n_members is None:
            return []
        member = self._member_stats()
        ref = self._ref_stats()
        rows = []
        for stat in ["dry_pct", "mean", "sd", "median", "p99"]:
            model_val = float(np.nanmean(member[stat]))
            ref_val = float(ref[stat])
            rows.append({
                "statistic": stat,
                "model": model_val,
                "reference": ref_val,
                "rel_bias_pct": _relbias(model_val, ref_val),
            })
        return rows

    def bias_maps(self) -> dict:
        """Per-gridpoint Mean/SD maps and their bias maps (model - reference)."""
        out = {}
        if self.model_n <= 0 or self.ref_n <= 0:
            return out
        model_mean = self.model_sum / self.model_n
        model_var = self.model_sumsq / self.model_n - np.square(model_mean)
        model_sd = np.sqrt(np.clip(model_var, 0.0, None))
        ref_mean = self.ref_sum / self.ref_n
        ref_var = self.ref_sumsq / self.ref_n - np.square(ref_mean)
        ref_sd = np.sqrt(np.clip(ref_var, 0.0, None))
        out["mean_model"] = model_mean
        out["mean_ref"] = ref_mean
        out["mean_bias"] = model_mean - ref_mean
        out["sd_model"] = model_sd
        out["sd_ref"] = ref_sd
        out["sd_bias"] = model_sd - ref_sd

        # Dry% maps (per-gridpoint fraction below the dry threshold)
        out["dry_pct_model"] = 100.0 * self.model_dry / self.model_n
        out["dry_pct_ref"] = 100.0 * self.ref_dry / self.ref_n
        out["dry_pct_bias"] = out["dry_pct_model"] - out["dry_pct_ref"]

        if self.model_pix_hist is not None and self.ref_pix_hist is not None:
            for q, name in [(0.5, "median"), (0.99, "p99")]:
                m_map = self._pix_quantile_map(self.model_pix_hist, q)
                r_map = self._pix_quantile_map(self.ref_pix_hist, q)
                out[f"{name}_model"] = m_map
                out[f"{name}_ref"] = r_map
                out[f"{name}_bias"] = m_map - r_map
        return out

    def _pix_quantile_map(self, pix_hist: np.ndarray, q: float) -> np.ndarray:
        edges = np.linspace(self.hist_lo, self.hist_hi, self.map_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        counts = pix_hist.astype(np.float64)
        total = counts.sum(axis=1, keepdims=True)
        safe = np.where(total > 0, total, 1.0)
        cdf = np.cumsum(counts, axis=1) / safe
        out = np.full(counts.shape[0], np.nan, dtype=np.float64)
        for i in range(counts.shape[0]):
            if total[i, 0] > 0:
                out[i] = np.interp(q, cdf[i], centers)
        return out.reshape(self.H, self.W)


class ClimatologyAccumulator:
    """Accumulates the paper's climatological statistics (raw and 3x3-smoothed).

    Usage::

        acc = ClimatologyAccumulator(img_shape=(H, W))
        for pred_ens, target in stream:   # pred_ens (N,H,W), target (H,W), mm
            acc.update(pred_ens, target)
        rows = acc.to_table()             # list of dicts for W&B Table / JSON
        maps = acc.bias_maps()            # {"raw": {...}, "s3x3": {...}}
    """

    def __init__(
        self,
        img_shape: tuple[int, int],
        dry_threshold: float = 1.0,
        wet_threshold: float = 1.0,
        hist_lo: float = 0.0,
        hist_hi: float = 300.0,
        hist_bins: int = 3000,
        smooth_size: int = 3,
        compute_smoothed: bool = True,
        compute_quantile_maps: bool = False,
        map_bins: int = 100,
    ):
        self.smooth_size = smooth_size
        self.compute_smoothed = compute_smoothed
        kwargs = dict(
            img_shape=img_shape, dry_threshold=dry_threshold,
            wet_threshold=wet_threshold, hist_lo=hist_lo, hist_hi=hist_hi,
            hist_bins=hist_bins, compute_quantile_maps=compute_quantile_maps,
            map_bins=map_bins,
        )
        self.raw = _StatState(**kwargs)
        self.s3x3 = _StatState(**kwargs) if compute_smoothed else None

        # Per-gridpoint RMSE of the ensemble mean (paper Fig 2; gridpoint only).
        self.img_shape = img_shape
        self._se_sum = np.zeros(img_shape, dtype=np.float64)
        self._se_n = 0.0

    def update(self, pred_ens: np.ndarray, target: np.ndarray) -> None:
        """Accumulate one timestep.

        Args:
            pred_ens: (N, H, W) ensemble predictions, physical units (mm).
            target:   (H, W) reference, physical units (mm).
        """
        pred_ens = np.asarray(pred_ens, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        if pred_ens.ndim != 3 or target.ndim != 2:
            raise ValueError(
                f"Expected pred_ens (N,H,W) and target (H,W), got "
                f"{pred_ens.shape} and {target.shape}"
            )
        self.raw.update_model(pred_ens)
        self.raw.update_ref(target)
        if self.s3x3 is not None:
            self.s3x3.update_model(moving_average_2d(pred_ens, self.smooth_size))
            self.s3x3.update_ref(moving_average_2d(target, self.smooth_size))

        # RMSE map: squared error of the ensemble mean vs. the reference.
        ens_mean = pred_ens.mean(axis=0)
        self._se_sum += np.square(ens_mean - target)
        self._se_n += 1

    def reduce(self) -> None:
        """Distributed reduce hook for API parity. The offline driver runs
        single-process, so this is a no-op. (Left as an extension point.)"""
        return

    def to_table(self) -> list[dict]:
        """Flat list of rows: window x statistic x (model, reference, rel-bias)."""
        rows = []
        for row in self.raw.table_rows():
            rows.append({"window": "raw", **row})
        if self.s3x3 is not None:
            for row in self.s3x3.table_rows():
                rows.append({"window": "3x3", **row})
        return rows

    def bias_maps(self) -> dict:
        out = {"raw": self.raw.bias_maps()}
        if self.s3x3 is not None:
            out["s3x3"] = self.s3x3.bias_maps()
        return out

    def rmse_map(self) -> np.ndarray | None:
        """Per-gridpoint RMSE of the ensemble mean over the eval period."""
        if self._se_n <= 0:
            return None
        return np.sqrt(self._se_sum / self._se_n)


class SpatialDistributionAccumulator:
    """Collects per-timestep domain-level spatial statistics for Q-Q plots (Fig 5).

    For each timestep it records two scalars for both the reference and the
    ensemble-mean field: the percentage of wet gridpoints (> wet_threshold) and
    the spatial mean rainfall over those wet gridpoints.  Quantiles of these
    per-day series form the Q-Q curves (the "simulated" side uses the ensemble
    mean as a faithful-enough stand-in for the pooled simulations).
    """

    def __init__(self, wet_threshold: float = 1.0):
        self.wet_threshold = wet_threshold
        self.ref_wetpct: list[float] = []
        self.model_wetpct: list[float] = []
        self.ref_wetmean: list[float] = []
        self.model_wetmean: list[float] = []

    def update(self, pred_ens: np.ndarray, target: np.ndarray) -> None:
        ens_mean = np.asarray(pred_ens, dtype=np.float64).mean(axis=0)
        tar = np.asarray(target, dtype=np.float64)
        for field, pct, mean in (
            (tar, self.ref_wetpct, self.ref_wetmean),
            (ens_mean, self.model_wetpct, self.model_wetmean),
        ):
            wet = field > self.wet_threshold
            pct.append(100.0 * wet.mean())
            mean.append(float(field[wet].mean()) if wet.any() else 0.0)

    def qq_panels(self, n_q: int = 100) -> list[dict]:
        """Build the three Q-Q panels (wet %, spatial-mean, tail) for qq_figure."""
        levels = np.linspace(0.01, 0.99, n_q)
        tail = np.linspace(0.90, 0.999, n_q)

        def q(arr, lv):
            return np.quantile(np.asarray(arr), lv) if len(arr) else np.zeros_like(lv)

        return [
            {
                "title": "Distribution of Wet Gridpoints",
                "xlabel": "Reference (%)", "ylabel": "Simulated (%)",
                "curves": [{"label": "CorrDiff",
                            "ref": q(self.ref_wetpct, levels),
                            "sim": q(self.model_wetpct, levels)}],
            },
            {
                "title": "Distribution of Spatial Mean Rainfall",
                "xlabel": "Reference (mm)", "ylabel": "Simulated (mm)",
                "curves": [{"label": "CorrDiff",
                            "ref": q(self.ref_wetmean, levels),
                            "sim": q(self.model_wetmean, levels)}],
            },
            {
                "title": "Spatial Mean Rainfall (P90–P99.9)",
                "xlabel": "Reference (mm)", "ylabel": "Simulated (mm)",
                "curves": [{"label": "CorrDiff",
                            "ref": q(self.ref_wetmean, tail),
                            "sim": q(self.model_wetmean, tail)}],
            },
        ]
