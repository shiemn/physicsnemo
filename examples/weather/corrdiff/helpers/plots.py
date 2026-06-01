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

"""Diagnostic plot utilities and distributed accumulators for CorrDiff evaluation.

All plot functions return a ``matplotlib.figure.Figure`` suitable for
``wandb.Image(fig)`` — no disk I/O required.

Accumulators (HistogramAccumulator, RAPSDAccumulator) support distributed
``reduce()`` via ``torch.distributed.all_reduce`` and can be used in the same
pattern as ``helpers.metrics.MetricsAccumulator``.
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch


def _normalize_channel_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _is_temperature_channel(name: str) -> bool:
    normalized = _normalize_channel_name(name)
    return normalized in {
        "t2m",
        "temperature2m",
        "airtemperature2m",
        "tas",
        "temperature",
    }

# ---------------------------------------------------------------------------
# Cartopy import (optional — fall back to plain pcolormesh if unavailable)
# ---------------------------------------------------------------------------
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    _CARTOPY_AVAILABLE = True
except ImportError:
    _CARTOPY_AVAILABLE = False


# ===========================================================================
# Distributed accumulators
# ===========================================================================


class HistogramAccumulator:
    """Accumulates histograms across timesteps and GPU ranks.

    Uses 150 linearly-spaced bins with a data-driven range. Tracks ground-truth, the
    average-of-member-histograms (the model's marginal distribution), and up
    to 3 individual ensemble members.

    Supports distributed reduction via ``torch.distributed.all_reduce``.

    Usage (multi-GPU)::

        hist = HistogramAccumulator(device=device)
        for pred_ens, target in local_timesteps:   # physical units, >= 0
            hist.update(pred_ens, target)
        hist.reduce()
        if dist.rank == 0:
            fig = plot_log_histogram(hist)
    """

    N_BINS = 150
    N_MEMBERS_TRACKED = 3

    def __init__(self, device=None):
        self.device = device if device is not None else torch.device("cpu")
        self.global_min = torch.tensor(float("inf"), dtype=torch.float64)
        self.global_max = torch.tensor(float("-inf"), dtype=torch.float64)
        self._n_fine = 1000
        self._fine_min = -1.0
        self._fine_max = 1.0
        self.target_counts = torch.zeros(self._n_fine, dtype=torch.float64)
        # Sum of per-member histograms (divided by n_members_accumulated at plot time)
        self.ens_avg_counts = torch.zeros(self._n_fine, dtype=torch.float64)
        self.n_members_accumulated = torch.tensor(0, dtype=torch.float64)
        self.member_counts = [
            torch.zeros(self._n_fine, dtype=torch.float64)
            for _ in range(self.N_MEMBERS_TRACKED)
        ]

    def _bin_idx(self, vals: torch.Tensor) -> torch.Tensor:
        """Map values to fine-grid bin indices."""
        span = max(self._fine_max - self._fine_min, 1e-9)
        idx = ((vals - self._fine_min) / span * self._n_fine).long()
        return idx.clamp(0, self._n_fine - 1)

    def _rebin_counts(self, counts: torch.Tensor, old_min: float, old_max: float) -> torch.Tensor:
        old_edges = np.linspace(old_min, old_max, self._n_fine + 1)
        old_centers = 0.5 * (old_edges[:-1] + old_edges[1:])
        new_counts = torch.zeros_like(counts)
        idx = np.floor((old_centers - self._fine_min) / max(self._fine_max - self._fine_min, 1e-9) * self._n_fine)
        idx = np.clip(idx.astype(np.int64), 0, self._n_fine - 1)
        src = counts.numpy()
        dst = new_counts.numpy()
        np.add.at(dst, idx, src)
        return new_counts

    def _ensure_range(self, min_val: float, max_val: float) -> None:
        if not np.isfinite(min_val) or not np.isfinite(max_val):
            return

        if min_val == max_val:
            pad = max(abs(min_val) * 0.05, 1.0)
            min_val -= pad
            max_val += pad

        if not np.isfinite(float(self.global_min)) or not np.isfinite(float(self.global_max)):
            span = max_val - min_val
            margin = max(span * 0.05, 1.0)
            self._fine_min = min_val - margin
            self._fine_max = max_val + margin
            return

        if min_val >= self._fine_min and max_val <= self._fine_max:
            return

        old_min = self._fine_min
        old_max = self._fine_max
        span = max(max(max_val, self._fine_max) - min(min_val, self._fine_min), 1.0)
        margin = max(span * 0.05, 1.0)
        self._fine_min = min(min_val, self._fine_min) - margin
        self._fine_max = max(max_val, self._fine_max) + margin
        self.target_counts = self._rebin_counts(self.target_counts, old_min, old_max)
        self.ens_avg_counts = self._rebin_counts(self.ens_avg_counts, old_min, old_max)
        self.member_counts = [self._rebin_counts(mc, old_min, old_max) for mc in self.member_counts]

    @torch.no_grad()
    def update(self, pred_ens: torch.Tensor, target: torch.Tensor) -> None:
        """Accumulate histogram counts for one timestep.

        Args:
            pred_ens: (N_ens, C, H, W) ensemble predictions in physical units (mm).
            target:   (C, H, W) ground truth in physical units (mm).
        """
        pred_ens = pred_ens.float().cpu()
        target = target.float().cpu()

        cur_min = min(float(target.min()), float(pred_ens.min()))
        cur_max = max(float(target.max()), float(pred_ens.max()))
        self._ensure_range(cur_min, cur_max)
        self.global_min = torch.tensor(min(float(self.global_min), cur_min), dtype=torch.float64)
        self.global_max = torch.tensor(max(float(self.global_max), cur_max), dtype=torch.float64)

        # Histogram target
        t_vals = target.flatten()
        t_idx = self._bin_idx(t_vals)
        self.target_counts.scatter_add_(
            0, t_idx, torch.ones(len(t_idx), dtype=torch.float64)
        )

        # Histogram all ensemble members and accumulate their per-timestep average.
        n_ens = pred_ens.shape[0]
        ens_avg_step_counts = torch.zeros(self._n_fine, dtype=torch.float64)
        for i in range(n_ens):
            v = pred_ens[i].flatten()
            vi = self._bin_idx(v)
            ens_avg_step_counts.scatter_add_(
                0, vi, torch.ones(len(vi), dtype=torch.float64)
            )
        self.ens_avg_counts += ens_avg_step_counts / max(n_ens, 1)
        self.n_members_accumulated += n_ens

        # Histogram individual members (first N_MEMBERS_TRACKED only, for display)
        for i in range(min(self.N_MEMBERS_TRACKED, n_ens)):
            v = pred_ens[i].flatten()
            vi = self._bin_idx(v)
            self.member_counts[i].scatter_add_(
                0, vi, torch.ones(len(vi), dtype=torch.float64)
            )

    def reduce(self) -> None:
        """All-reduce histogram counts across all distributed ranks (in-place)."""
        if (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
        ):
            return
        dev = self.device
        # Reduce global max
        gmin = self.global_min.to(dev)
        torch.distributed.all_reduce(gmin, op=torch.distributed.ReduceOp.MIN)
        self.global_min = gmin.cpu()

        gm = self.global_max.to(dev)
        torch.distributed.all_reduce(gm, op=torch.distributed.ReduceOp.MAX)
        self.global_max = gm.cpu()

        global_min = float(self.global_min)
        global_max = float(self.global_max)
        if np.isfinite(global_min) and np.isfinite(global_max):
            if global_min == global_max:
                pad = max(abs(global_min) * 0.05, 1.0)
                global_min -= pad
                global_max += pad
            span = global_max - global_min
            margin = max(span * 0.05, 1.0)
            target_fine_min = global_min - margin
            target_fine_max = global_max + margin
            if target_fine_min != self._fine_min or target_fine_max != self._fine_max:
                old_min = self._fine_min
                old_max = self._fine_max
                self._fine_min = target_fine_min
                self._fine_max = target_fine_max
                self.target_counts = self._rebin_counts(self.target_counts, old_min, old_max)
                self.ens_avg_counts = self._rebin_counts(self.ens_avg_counts, old_min, old_max)
                self.member_counts = [self._rebin_counts(mc, old_min, old_max) for mc in self.member_counts]

        # Reduce counts
        nm = self.n_members_accumulated.to(dev)
        torch.distributed.all_reduce(nm, op=torch.distributed.ReduceOp.SUM)
        self.n_members_accumulated = nm.cpu()

        for counts in [self.target_counts, self.ens_avg_counts] + self.member_counts:
            c = counts.to(dev)
            torch.distributed.all_reduce(c, op=torch.distributed.ReduceOp.SUM)
            counts.copy_(c.cpu())

    def get_quantiles(self, n_quantiles: int = 100) -> dict:
        """Compute quantiles from the accumulated fine-grid CDFs.

        Args:
            n_quantiles: Number of linearly-spaced quantile levels in (0, 1).

        Returns:
            dict with keys:
                levels:   (n_quantiles,) quantile probability levels
                target:   (n_quantiles,) observed quantile values
                ens_avg:  (n_quantiles,) ensemble-average quantile values
                members:  list of (n_quantiles,) per-member quantile arrays
        """
        fine_edges = np.linspace(self._fine_min, self._fine_max, self._n_fine + 1)
        fine_centers = 0.5 * (fine_edges[:-1] + fine_edges[1:])
        levels = np.linspace(0.01, 0.99, n_quantiles)

        def counts_to_quantiles(counts: torch.Tensor) -> np.ndarray:
            fc = counts.numpy().astype(np.float64)
            total = fc.sum()
            if total == 0:
                return np.zeros(n_quantiles)
            cdf = np.cumsum(fc) / total
            return np.interp(levels, cdf, fine_centers)

        return {
            "levels": levels,
            "target": counts_to_quantiles(self.target_counts),
            "ens_avg": counts_to_quantiles(self.ens_avg_counts),
            "members": [counts_to_quantiles(mc) for mc in self.member_counts],
        }

    def get_rebinned(self) -> dict:
        """Re-bin fine-grid counts into N_BINS linear bins covering the observed range.

        Returns dict with bin_centers, target, ens_mean, members (list of arrays).
        """
        min_val = float(self.global_min)
        max_val = float(self.global_max)
        if not np.isfinite(min_val) or not np.isfinite(max_val):
            min_val, max_val = -1.0, 1.0
        if min_val == max_val:
            pad = max(abs(min_val) * 0.05, 1.0)
            min_val -= pad
            max_val += pad
        span = max_val - min_val
        margin = max(span * 0.05, 1.0)
        edges = np.linspace(min_val - margin, max_val + margin, self.N_BINS + 1)

        fine_edges = np.linspace(self._fine_min, self._fine_max, self._n_fine + 1)
        fine_centers = 0.5 * (fine_edges[:-1] + fine_edges[1:])

        def rebin(fine_counts):
            fc = fine_counts.numpy()
            out = np.zeros(self.N_BINS)
            idx = np.digitize(fine_centers, edges) - 1
            idx = np.clip(idx, 0, self.N_BINS - 1)
            for b in range(self.N_BINS):
                out[b] = fc[idx == b].sum()
            return out

        return {
            "bin_edges": edges,
            "bin_centers": 0.5 * (edges[:-1] + edges[1:]),
            "target": rebin(self.target_counts),
            "ens_avg_hist": rebin(self.ens_avg_counts),
            "members": [rebin(mc) for mc in self.member_counts],
        }


class RAPSDAccumulator:
    """Accumulates mean radially averaged power spectral density (RAPSD) across timesteps.

    Uses ``numpy.fft.fft2`` + ``fftshift`` to compute the 2-D power spectrum,
    then radially averages into **50 logarithmically-spaced** frequency bins.
    Channels are averaged together.

    Supports distributed reduction via ``torch.distributed.all_reduce``.

    Usage (multi-GPU)::

        rapsd = RAPSDAccumulator(img_shape=(H, W), dx_km=2.0, device=device)
        for pred_ens, target in local_timesteps:
            rapsd.update(pred_ens, target)
        rapsd.reduce()
        if dist.rank == 0:
            pred_psd = (rapsd.pred_psd_sum / rapsd.n_samples).numpy()
            tar_psd  = (rapsd.target_psd_sum / rapsd.n_samples).numpy()
            fig = plot_rapsd(rapsd.bin_centers, pred_psd, tar_psd, dx_km=2.0)
    """

    N_BINS = 50

    def __init__(self, img_shape: tuple[int, int], dx_km: float = 2.0, device=None):
        """
        Args:
            img_shape: (H, W) spatial dimensions of the field.
            dx_km: Grid spacing in km.  Used to convert frequency to physical units.
            device: Torch device for distributed reduce.
        """
        H, W = img_shape
        self.H, self.W = H, W
        self.dx_km = dx_km
        self.device = device if device is not None else torch.device("cpu")

        # Shifted frequency grid (matches fftshift output)
        ky = np.fft.fftshift(np.fft.fftfreq(H, d=dx_km))  # cycles / km
        kx = np.fft.fftshift(np.fft.fftfreq(W, d=dx_km))
        KX, KY = np.meshgrid(kx, ky)
        k_rad = np.sqrt(KX**2 + KY**2)

        # Logarithmic binning (50 bins)
        k_min = k_rad[k_rad > 0].min()
        k_max = k_rad.max()
        self.freq_bins = np.logspace(np.log10(k_min), np.log10(k_max), self.N_BINS + 1)
        self.bin_centers = 0.5 * (self.freq_bins[:-1] + self.freq_bins[1:])

        # Precompute bin assignments
        k_rad_flat = k_rad.flatten()
        self._k_rad_flat = k_rad_flat

        self.pred_psd_sum = torch.zeros(self.N_BINS, dtype=torch.float64)
        self.target_psd_sum = torch.zeros(self.N_BINS, dtype=torch.float64)
        self.n_samples = 0

    def _field_rapsd(self, field_chw: np.ndarray) -> np.ndarray:
        """Compute channel-averaged RAPSD for a single field.

        Args:
            field_chw: (C, H, W) numpy array in physical units.

        Returns:
            (N_BINS,) radially averaged power spectral density.
        """
        C = field_chw.shape[0]
        psd = np.zeros(self.N_BINS, dtype=np.float64)
        for c in range(C):
            fft2 = np.fft.fftshift(np.fft.fft2(field_chw[c]))
            power = np.abs(fft2) ** 2
            power_flat = power.flatten()
            for b in range(self.N_BINS):
                mask = (self._k_rad_flat >= self.freq_bins[b]) & (
                    self._k_rad_flat < self.freq_bins[b + 1]
                )
                if np.any(mask):
                    psd[b] += power_flat[mask].mean()
        return psd / C

    @torch.no_grad()
    def update(self, pred_ens: torch.Tensor, target: torch.Tensor) -> None:
        """Accumulate RAPSD for one timestep.

        Args:
            pred_ens: (N_ens, C, H, W) ensemble predictions in physical units.
            target:   (C, H, W) ground truth in physical units.
        """
        ens_mean_np = pred_ens.float().mean(0).cpu().numpy()
        target_np = target.float().cpu().numpy()
        self.pred_psd_sum += torch.from_numpy(self._field_rapsd(ens_mean_np))
        self.target_psd_sum += torch.from_numpy(self._field_rapsd(target_np))
        self.n_samples += 1

    def reduce(self) -> None:
        """All-reduce PSD sums and sample count across all distributed ranks (in-place)."""
        if (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
        ):
            return
        ps = self.pred_psd_sum.to(self.device)
        ts = self.target_psd_sum.to(self.device)
        ns = torch.tensor([self.n_samples], dtype=torch.float64, device=self.device)
        torch.distributed.all_reduce(ps, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(ts, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(ns, op=torch.distributed.ReduceOp.SUM)
        self.pred_psd_sum = ps.cpu()
        self.target_psd_sum = ts.cpu()
        self.n_samples = int(ns.item())


# ===========================================================================
# Plot functions
# ===========================================================================


def plot_example_event(
    pred_ens_np: np.ndarray,
    target_np: np.ndarray,
    reg_mean_np: np.ndarray,
    time_str: str,
    channel_names: list[str],
    lat: np.ndarray | None = None,
    lon: np.ndarray | None = None,
    plot_channels: list[int] | None = None,
) -> plt.Figure:
    """Georeferenced spatial maps for a single forecast event.

    Produces a 5-row layout per channel (using a 6-column GridSpec):
      Row 0: [Target (cols 0:3)       | Regression mean (cols 3:6)]
      Row 1: [Ensemble mean (cols 0:3) | Ensemble spread (cols 3:6)]
      Rows 2-4: 3×3 grid of individual ensemble members (cols 0:2, 2:4, 4:6)

    Uses cartopy (PlateCarree projection, coastlines, borders) when available;
    falls back to plain pcolormesh otherwise.

    Args:
        pred_ens_np: (N_ens, C, H, W) ensemble predictions in physical units.
        target_np:   (C, H, W) ground truth in physical units.
        reg_mean_np: (C, H, W) regression-only prediction in physical units.
        time_str:    ISO-8601 timestamp string used as the figure title.
        channel_names: List of channel name strings (length == C).
        lat: (H, W) latitude array.  If None, row/col indices are used.
        lon: (H, W) longitude array.  If None, row/col indices are used.
        plot_channels: Optional list of channel indices to include in the plot.
            Defaults to all channels.  Use this to avoid extremely tall figures
            when there are many output variables.

    Returns:
        matplotlib Figure.
    """
    import matplotlib.gridspec as gridspec

    if isinstance(reg_mean_np, np.ndarray) and reg_mean_np.ndim == 4:
        reg_mean_np = reg_mean_np[0]  # (1, C, H, W) -> (C, H, W)

    all_C = target_np.shape[0]
    channels_to_plot = list(plot_channels) if plot_channels is not None else list(range(all_C))
    C = len(channels_to_plot)
    n_members_to_show = min(pred_ens_np.shape[0], 9)
    use_cartopy = _CARTOPY_AVAILABLE and lat is not None and lon is not None

    proj = ccrs.PlateCarree() if use_cartopy else None
    transform = ccrs.PlateCarree() if use_cartopy else None

    cmap_spread = "YlOrRd"

    ens_mean_np = pred_ens_np.mean(0)   # (C, H, W)
    ens_spread_np = pred_ens_np.std(0)  # (C, H, W)

    if lat is not None and lon is not None:
        x, y = lon, lat
    else:
        H, W = target_np.shape[-2], target_np.shape[-1]
        x, y = np.arange(W), np.arange(H)

    # Layout: 5 rows × 6 cols per channel
    #   rows 0-1: 4 summary panels (each spans 3 cols)
    #   rows 2-4: 3×3 member grid (each spans 2 cols)
    rows_per_channel = 5
    n_gs_cols = 6
    fig = plt.figure(figsize=(14, 14 * C))
    gs = gridspec.GridSpec(
        rows_per_channel * C, n_gs_cols,
        figure=fig,
        hspace=0.35,
        wspace=0.3,
    )
    fig.suptitle(f"Event: {time_str}", fontsize=14)
    fig.subplots_adjust(top=0.96)

    def make_ax(gs_slice):
        if use_cartopy:
            return fig.add_subplot(gs_slice, projection=proj)
        return fig.add_subplot(gs_slice)

    for plot_idx, c in enumerate(channels_to_plot):
        channel_label = channel_names[c] if c < len(channel_names) else f"ch{c}"
        ro = rows_per_channel * plot_idx  # row offset in GridSpec

        # Auto-detect colorscale: use symmetric diverging cmap for signed variables
        all_data = np.concatenate([
            target_np[c].flatten(),
            reg_mean_np[c].flatten(),
            ens_mean_np[c].flatten(),
        ])
        data_min = float(np.nanpercentile(all_data, 1))
        data_max = float(np.nanpercentile(all_data, 99))
        abs_data_max = float(np.nanpercentile(np.abs(all_data), 99))
        if data_min < -0.05 * max(abs(data_min), data_max, 1e-9):
            # Signed variable: symmetric colormap centered at 0
            cmap_intensity = "RdBu_r"
            vmax_intensity = max(abs(data_min), abs_data_max)
            vmin_intensity = -vmax_intensity
            cbar_label = channel_label
        elif _is_temperature_channel(channel_label):
            cmap_intensity = "coolwarm"
            vmin_intensity = data_min
            vmax_intensity = data_max
            cbar_label = channel_label
        else:
            cmap_intensity = "Blues"
            vmin_intensity = max(0.0, data_min)
            vmax_intensity = data_max
            cbar_label = channel_label

        vmax_spread = float(np.nanpercentile(ens_spread_np[c], 99))

        def label(title, _ch_label=channel_label):
            return f"{_ch_label}: {title}" if C > 1 else title

        def draw_field_labeled(ax, field, cmap, vmin, vmax, title, unit=cbar_label):
            if use_cartopy:
                ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor="black")
                ax.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle=":", edgecolor="black")
                pcm = ax.pcolormesh(
                    x, y, field, cmap=cmap, vmin=vmin, vmax=vmax,
                    transform=transform, shading="auto",
                )
            else:
                pcm = ax.pcolormesh(x, y, field, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
            plt.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04, shrink=0.8, label=unit)
            ax.set_title(title, fontsize=10)

        # Summary row 0: Target | Regression mean
        # Summary row 1: Ensemble mean | Ensemble spread
        summary = [
            (gs[ro + 0, 0:3], target_np[c],    cmap_intensity, vmin_intensity, vmax_intensity, "Target",          cbar_label),
            (gs[ro + 0, 3:6], reg_mean_np[c],   cmap_intensity, vmin_intensity, vmax_intensity, "Regression mean", cbar_label),
            (gs[ro + 1, 0:3], ens_mean_np[c],   cmap_intensity, vmin_intensity, vmax_intensity, "Ensemble mean",   cbar_label),
            (gs[ro + 1, 3:6], ens_spread_np[c], cmap_spread,    0.0,            vmax_spread,    "Ensemble spread", cbar_label),
        ]
        for gs_slice, field, cmap, vmin, vmax, title, unit in summary:
            draw_field_labeled(make_ax(gs_slice), field, cmap, vmin, vmax, label(title), unit)

        # Members 3×3 grid (rows 2-4, cols 0:2 / 2:4 / 4:6)
        for m in range(n_members_to_show):
            mr = m // 3  # 0, 1, 2
            mc = m % 3   # 0, 1, 2
            col_start = mc * 2
            ax = make_ax(gs[ro + 2 + mr, col_start : col_start + 2])
            draw_field_labeled(ax, pred_ens_np[m, c], cmap_intensity, vmin_intensity, vmax_intensity, label(f"Member {m}"), cbar_label)

    return fig


def plot_spread_skill(
    bin_mean_spread: list[float] | np.ndarray,
    bin_mean_skill: list[float] | np.ndarray,
    slope: float | None = None,
    intercept: float | None = None,
    bin_mode: str = "fixed",
) -> plt.Figure:
    """Spread-skill reliability plot (Mardani et al. 2025, Fig. 2).

    X-axis is RMSE, Y-axis is ensemble standard deviation.
    A perfectly calibrated ensemble lies on y = x.

    Args:
        bin_mean_spread: Mean ensemble std per spread bin.
        bin_mean_skill:  Mean RMSE per spread bin.
        slope:     Optional slope of the weighted linear fit (from MetricsAccumulator).
        intercept: Optional intercept of the weighted linear fit.

    Returns:
        matplotlib Figure.
    """
    spread = np.asarray(bin_mean_spread)
    skill = np.asarray(bin_mean_skill)

    # Remove zero-count bins (they appear as 0.0 in the lists)
    valid = (spread > 0) | (skill > 0)
    spread = spread[valid]
    skill = skill[valid]

    fig, ax = plt.subplots(figsize=(6, 6))

    # X = RMSE, Y = spread (matching CorrDiff paper convention)
    point_color = "mediumseagreen" if bin_mode == "quantile" else "steelblue"
    ax.scatter(skill, spread, color=point_color, s=60, zorder=3, label="Bins")

    # Perfect calibration reference
    combined_max = max(spread.max(), skill.max()) * 1.05
    ref = np.array([0.0, combined_max])
    ax.plot(ref, ref, "k--", linewidth=1.2, label="y = x (perfect)")

    # Linear fit from MetricsAccumulator (fit spread as function of RMSE)
    if slope is not None and np.isfinite(slope) and intercept is not None:
        fit_x = np.array([skill.min(), skill.max()])
        fit_y = slope * fit_x + intercept
        ax.plot(fit_x, fit_y, "r-", linewidth=1.2, label=f"fit: slope={slope:.2f}")

    ax.set_xlabel("RMSE (mm)", fontsize=11)
    ax.set_ylabel("Spread (mm)", fontsize=11)
    ax.set_title("Spread-Skill Reliability", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.4)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    return fig


def plot_rank_histogram(rank_counts: list[float] | np.ndarray) -> plt.Figure:
    """Rank histogram for ensemble calibration (Mardani et al. 2025, Fig. 2).

    A flat histogram indicates a well-calibrated ensemble. U-shaped means
    under-dispersive; dome-shaped means over-dispersive.

    Args:
        rank_counts: Array of length (N_ens + 1) with rank frequency counts.

    Returns:
        matplotlib Figure.
    """
    counts = np.asarray(rank_counts, dtype=float)
    n_ranks = len(counts)
    total = counts.sum()
    if total == 0:
        total = 1.0

    freq = counts / total
    expected = 1.0 / n_ranks

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(1, n_ranks + 1), freq, color="steelblue", edgecolor="white", linewidth=0.5)
    ax.axhline(expected, color="black", linestyle="--", linewidth=1.2, label="Uniform")

    ax.set_xlabel("Rank", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("Rank Histogram", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    return fig


def plot_log_histogram(hist_acc: "HistogramAccumulator") -> plt.Figure:
    """Log-scale histogram comparing predicted and observed distributions.

    Uses 150 linear bins with fill_between style, showing ground truth,
    the ensemble-average marginal distribution, and an example individual member.

    Args:
        hist_acc: A HistogramAccumulator with accumulated counts.

    Returns:
        matplotlib Figure.
    """
    data = hist_acc.get_rebinned()
    bin_centers = data["bin_centers"]
    truth = data["target"]
    ens_avg = data["ens_avg_hist"]
    members = data["members"]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Ground truth
    ax.fill_between(bin_centers, truth, alpha=0.3, color="black", label="Ground Truth")
    ax.plot(bin_centers, truth, color="black", linewidth=2)

    if ens_avg.sum() > 0:
        ax.fill_between(bin_centers, ens_avg, alpha=0.25, color="tab:red", label="Ensemble Avg")
        ax.plot(bin_centers, ens_avg, color="tab:red", linewidth=2)

    # Single ensemble member for context
    if members and members[0].sum() > 0:
        ax.plot(bin_centers, members[0], color="tab:orange", linewidth=1.5, alpha=0.9, label="Member 0")

    ax.set_yscale("log")
    ax.set_ylim(bottom=1.0)
    ax.set_xlabel("Precipitation (mm/3hr)", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("Precipitation Distribution", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_qq(hist_acc: "HistogramAccumulator", n_quantiles: int = 100) -> plt.Figure:
    """Q-Q plot comparing predicted vs observed precipitation quantiles.

    X-axis is observed quantiles (mm), Y-axis is predicted quantiles (mm).
    A 1:1 dashed line represents a perfect distribution match.

    Args:
        hist_acc:    A HistogramAccumulator with accumulated counts.
        n_quantiles: Number of quantile levels to evaluate.

    Returns:
        matplotlib Figure.
    """
    data = hist_acc.get_quantiles(n_quantiles=n_quantiles)
    obs = data["target"]
    pred = data["ens_avg"]

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(obs, pred, s=15, color="steelblue", zorder=3, label="Quantiles")

    ref_max = max(obs.max(), pred.max()) * 1.05
    ax.plot([0, ref_max], [0, ref_max], "k--", linewidth=1.2, label="1:1 (perfect)")

    ax.set_xlabel("Observed quantiles (mm)", fontsize=11)
    ax.set_ylabel("Predicted quantiles (mm)", fontsize=11)
    ax.set_title("Q-Q Plot", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    return fig


def _value_axis_label(diagnostic_info: dict | None) -> str:
    if not diagnostic_info:
        return "Value"
    label = diagnostic_info.get("label") or "Value"
    unit = diagnostic_info.get("unit")
    return f"{label} ({unit})" if unit else str(label)


def _metric_axis_label(metric_name: str, diagnostic_info: dict | None) -> str:
    unit = diagnostic_info.get("unit") if diagnostic_info else None
    return f"{metric_name} ({unit})" if unit else metric_name


def _counts_to_pdf(counts: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=float)
    widths = np.diff(np.asarray(bin_edges, dtype=float))
    total = counts.sum()
    if total <= 0:
        return np.zeros_like(counts, dtype=float)
    return counts / max(total, 1.0) / np.maximum(widths, 1e-12)


def _pdf_to_log_pdf(pdf: np.ndarray) -> np.ndarray:
    pdf = np.asarray(pdf, dtype=float)
    out = np.full_like(pdf, np.nan, dtype=float)
    valid = pdf > 0
    out[valid] = np.log(pdf[valid])
    return out


def plot_diagnostic_panel(
    metrics_dict: dict,
    acc_label: str,
    hist_acc: "HistogramAccumulator",
    rapsd_acc: "RAPSDAccumulator",
    rapsd_dx_km: float = 2.0,
    diagnostic_info: dict | None = None,
) -> plt.Figure | None:
    """Combined 2×3 diagnostic panel: spread-skill, rank histogram, Q-Q, log histogram, RAPSD.

    Args:
        metrics_dict: Flat metrics dict from MetricsAccumulator.to_dict() (with prefix).
        acc_label:    Prefix used in metrics_dict keys (e.g. "diffusion" or "regression").
        hist_acc:     HistogramAccumulator with accumulated counts.
        rapsd_acc:    RAPSDAccumulator with accumulated PSD.
        rapsd_dx_km:  Grid spacing in km (for RAPSD title).

    Returns:
        A matplotlib Figure, or None if no data is available.
    """
    value_label = _value_axis_label(diagnostic_info)
    panel_label = diagnostic_info.get("label") if diagnostic_info else None

    fig, axes = plt.subplots(2, 3, figsize=(21, 10))
    title = f"Diagnostics — {acc_label}"
    if panel_label:
        title += f" — {panel_label}"
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # ── Top-left: Spread-Skill ────────────────────────────────────────────────
    ax = axes[0, 0]
    spread_key = f"{acc_label}/spread_skill_bin_mean_spread"
    skill_key = f"{acc_label}/spread_skill_bin_mean_skill"
    if spread_key in metrics_dict and skill_key in metrics_dict:
        spread = np.asarray(metrics_dict[spread_key])
        skill = np.asarray(metrics_dict[skill_key])
        valid = (spread > 0) | (skill > 0)
        spread, skill = spread[valid], skill[valid]
        ax.scatter(skill, spread, color="steelblue", s=60, zorder=3, label="Bins")
        combined_max = max(spread.max(), skill.max()) * 1.05
        ref = np.array([0.0, combined_max])
        ax.plot(ref, ref, "k--", linewidth=1.2, label="y = x (perfect)")
        slope = metrics_dict.get(f"{acc_label}/spread_skill_reliability_slope")
        intercept = metrics_dict.get(f"{acc_label}/spread_skill_reliability_intercept")
        if slope is not None and np.isfinite(slope) and intercept is not None:
            fit_x = np.array([skill.min(), skill.max()])
            ax.plot(fit_x, slope * fit_x + intercept, "r-", linewidth=1.2,
                    label=f"fit: slope={slope:.2f}")
        ax.set_xlabel(_metric_axis_label("RMSE", diagnostic_info), fontsize=10)
        ax.set_ylabel(_metric_axis_label("Spread", diagnostic_info), fontsize=10)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.4)
    ax.set_title("Spread-Skill", fontsize=11)

    # ── Top-right: Rank Histogram ─────────────────────────────────────────────
    ax = axes[0, 1]
    rank_key = f"{acc_label}/rank_histogram"
    if rank_key in metrics_dict:
        counts = np.asarray(metrics_dict[rank_key], dtype=float)
        total = counts.sum() or 1.0
        freq = counts / total
        n_ranks = len(counts)
        ax.bar(range(1, n_ranks + 1), freq, color="steelblue", edgecolor="white", linewidth=0.5)
        ax.axhline(1.0 / n_ranks, color="black", linestyle="--", linewidth=1.2, label="Uniform")
        ax.set_xlabel("Rank", fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
    ax.set_title("Rank Histogram", fontsize=11)

    # ── Top-right: Q-Q Plot ───────────────────────────────────────────────────
    ax = axes[0, 2]
    if hist_acc.target_counts.sum() > 0:
        qq = hist_acc.get_quantiles()
        obs = qq["target"]
        pred = qq["ens_avg"]
        ref_min = min(obs.min(), pred.min())
        ref_max = max(obs.max(), pred.max())
        if ref_min == ref_max:
            pad = max(abs(ref_min) * 0.05, 1.0)
            ref_min -= pad
            ref_max += pad
        else:
            pad = max((ref_max - ref_min) * 0.05, 1.0)
            ref_min -= pad
            ref_max += pad
        ax.scatter(obs, pred, s=15, color="steelblue", zorder=3, label="Quantiles")
        ax.plot([ref_min, ref_max], [ref_min, ref_max], "k--", linewidth=1.2, label="1:1 (perfect)")
        ax.set_xlabel(f"Observed {value_label}", fontsize=10)
        ax.set_ylabel(f"Predicted {value_label}", fontsize=10)
        ax.set_xlim(ref_min, ref_max)
        ax.set_ylim(ref_min, ref_max)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    ax.set_title("Q-Q Plot", fontsize=11)

    # ── Bottom-left: Log Histogram ────────────────────────────────────────────
    ax = axes[1, 0]
    if hist_acc.target_counts.sum() > 0:
        data = hist_acc.get_rebinned()
        bin_centers = data["bin_centers"]
        truth = data["target"]
        ens_avg = data["ens_avg_hist"]
        members = data["members"]
        ax.fill_between(bin_centers, truth, alpha=0.3, color="black", label="Ground Truth")
        ax.plot(bin_centers, truth, color="black", linewidth=2)
        if ens_avg.sum() > 0:
            ax.fill_between(bin_centers, ens_avg, alpha=0.25, color="tab:red", label="Ensemble Avg")
            ax.plot(bin_centers, ens_avg, color="tab:red", linewidth=2)
        if members and members[0].sum() > 0:
            ax.plot(bin_centers, members[0], color="tab:orange", linewidth=1.5, alpha=0.9, label="Member 0")
        ax.set_yscale("log")
        ax.set_ylim(bottom=1.0)
        ax.set_xlabel(value_label, fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    ax.set_title("Distribution", fontsize=11)

    # ── Bottom-middle: RAPSD ─────────────────────────────────────────────────
    ax = axes[1, 1]
    if rapsd_acc.n_samples > 0:
        pred_psd = (rapsd_acc.pred_psd_sum / rapsd_acc.n_samples).numpy()
        tar_psd = (rapsd_acc.target_psd_sum / rapsd_acc.n_samples).numpy()
        freq = np.asarray(rapsd_acc.bin_centers, dtype=float)
        valid = (freq > 0) & (pred_psd > 0) & (tar_psd > 0)
        ax.loglog(freq[valid], tar_psd[valid], "k-", linewidth=2, label="Ground Truth")
        ax.loglog(freq[valid], pred_psd[valid], "r--", linewidth=2, label="Prediction")
        ax.set_xlabel("Spatial Frequency (km⁻¹)", fontsize=10)
        ax.set_ylabel("PSD", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, which="both", ls="--", alpha=0.6)
    ax.set_title("RAPSD", fontsize=11)

    # ── Bottom-right: Log PDF ────────────────────────────────────────────────
    ax = axes[1, 2]
    if hist_acc.target_counts.sum() > 0:
        data = hist_acc.get_rebinned()
        bin_centers = data["bin_centers"]
        bin_edges = data["bin_edges"]
        truth_log_pdf = _pdf_to_log_pdf(_counts_to_pdf(data["target"], bin_edges))
        ens_avg_log_pdf = _pdf_to_log_pdf(_counts_to_pdf(data["ens_avg_hist"], bin_edges))
        member0_log_pdf = None
        if data["members"] and np.asarray(data["members"][0]).sum() > 0:
            member0_log_pdf = _pdf_to_log_pdf(_counts_to_pdf(data["members"][0], bin_edges))

        valid_truth = np.isfinite(truth_log_pdf)
        if valid_truth.any():
            ax.plot(bin_centers[valid_truth], truth_log_pdf[valid_truth], color="black", linewidth=2, label="Ground Truth")

        valid_ens = np.isfinite(ens_avg_log_pdf)
        if valid_ens.any():
            ax.plot(bin_centers[valid_ens], ens_avg_log_pdf[valid_ens], color="tab:red", linewidth=2, label="Ensemble Avg")

        if member0_log_pdf is not None:
            valid_member = np.isfinite(member0_log_pdf)
            if valid_member.any():
                ax.plot(bin_centers[valid_member], member0_log_pdf[valid_member], color="tab:orange", linewidth=1.5, alpha=0.9, label="Member 0")

        ax.set_xlabel(value_label, fontsize=10)
        ax.set_ylabel("log(PDF)", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    ax.set_title("log(PDF)", fontsize=11)

    fig.tight_layout()
    return fig


def plot_rapsd(
    freq_centers: np.ndarray,
    pred_psd: np.ndarray,
    target_psd: np.ndarray,
    dx_km: float = 2.0,
) -> plt.Figure:
    """Radially Averaged Power Spectral Density (RAPSD) plot.

    X-axis shows spatial frequency in km⁻¹ (log-log axes).

    Args:
        freq_centers: (n_bins,) radial frequency bin centres in cycles/km.
        pred_psd:     (n_bins,) mean predicted RAPSD.
        target_psd:   (n_bins,) mean observed RAPSD.
        dx_km: Grid spacing in km (used for title only).

    Returns:
        matplotlib Figure.
    """
    freq_centers = np.asarray(freq_centers, dtype=float)
    pred_psd = np.asarray(pred_psd, dtype=float)
    target_psd = np.asarray(target_psd, dtype=float)

    # Filter out zero-power bins
    valid = (freq_centers > 0) & (pred_psd > 0) & (target_psd > 0)
    freq = freq_centers[valid]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.loglog(freq, target_psd[valid], "k-", linewidth=2, label="Ground Truth")
    ax.loglog(freq, pred_psd[valid], "r--", linewidth=2, label="Prediction")

    ax.set_xlabel("Spatial Frequency (km⁻¹)", fontsize=11)
    ax.set_ylabel("Power Spectral Density", fontsize=11)
    ax.set_title("Radially Averaged Power Spectral Density (RAPSD)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", ls="--", alpha=0.6)

    fig.tight_layout()
    return fig
