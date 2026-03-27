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

"""Centralized metric implementations for CorrDiff ensemble evaluation.

All metrics operate on **denormalized (physical unit)** predictions.
Callers must denormalize model outputs before calling update().
Predictions and targets are automatically clipped to >= 0 (precipitation).

Key correctness guarantees vs. the legacy MetricsAccumulator:

* CRPS: uses the proper finite-ensemble formula
    CRPS = E|X - y| - 0.5 * E|X - X'|
  instead of the broken approximation MAE - 0.5*std, which can go negative.

* Spread-Skill ratio: accumulated as separate sums (spread_sum, skill_sum)
  and divided *after* all_reduce so the ratio is computed from the global
  totals, not as an average of per-rank ratios.

* Conditional metrics (>threshold, 95th, wet-95th): run on physical-unit
  values so the thresholds have meaningful units (mm).

* Distributed reduce: uses all_reduce so every rank ends up with the same
  final accumulators — no gather/scatter boilerplate needed.
"""

import math

import numpy as np
import torch


def proper_twcrps(
    pred_ens: torch.Tensor, target: torch.Tensor, threshold: float
) -> torch.Tensor:
    """Threshold-weighted CRPS via the chaining function v(x) = max(x, t).

    twCRPS(F, y; t) = CRPS(max(F, t), max(y, t))

    Focuses skill on the right tail of the distribution (values above threshold t).
    Equivalent to scoring the censored-at-t distribution. Always >= 0.

    Args:
        pred_ens:  (N_ens, *spatial) ensemble predictions in physical units (mm).
        target:    (*spatial) ground truth in physical units (mm).
        threshold: Precipitation threshold t (mm).

    Returns:
        twCRPS tensor of shape (*spatial). Always >= 0 for valid inputs.
    """
    return proper_crps(
        torch.clamp(pred_ens, min=threshold),
        torch.clamp(target, min=threshold),
    )


def proper_crps(pred_ens: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Element-wise proper finite-ensemble CRPS.

    CRPS(F, y) = E_X|X - y| - 0.5 * E_{X,X'}|X - X'|

    Args:
        pred_ens: (N_ens, *spatial) ensemble predictions.
        target:   (*spatial) ground truth, same shape as pred_ens[0].

    Returns:
        CRPS tensor of shape (*spatial).  Always >= 0 for valid inputs.
    """
    n_ens = pred_ens.shape[0]
    if n_ens < 1:
        raise ValueError("Ensemble must contain at least one member.")

    term_obs = torch.abs(pred_ens - target.unsqueeze(0)).mean(dim=0)

    if n_ens == 1:
        return term_obs  # degenerate: no ensemble spread

    # Pairwise mean absolute difference across all member pairs
    pairwise = torch.abs(
        pred_ens.unsqueeze(0) - pred_ens.unsqueeze(1)
    ).mean(dim=(0, 1))
    return term_obs - 0.5 * pairwise


class MetricsAccumulator:
    """Accumulates ensemble forecast metrics over multiple timesteps.

    Metrics computed:
        - RMSE of ensemble mean
        - Proper finite-ensemble CRPS
        - Threshold-weighted CRPS (twCRPS) at configurable mm thresholds (default 5, 10 mm)
        - Spread  (mean ensemble std across pixels)
        - Skill   (mean per-sample RMSE, averaged over timesteps)
        - Spread-Skill ratio  (spread / skill; ideal = 1.0)
        - RMSE / CRPS for pixels where target > precip_threshold  (mm)
        - RMSE / CRPS for 95th-percentile pixels  (per sample)
        - RMSE / CRPS for wet-95th pixels  (95th pct over wet pixels only)
        - Spread-Skill reliability bins  (for calibration diagnostics)

    Usage (single GPU):
        acc = MetricsAccumulator(precip_threshold=1.0, device=device)
        for pred_ens, target in ...:          # physical units
            acc.update(pred_ens, target)
        metrics = acc.to_dict()

    Usage (multi-GPU):
        acc = MetricsAccumulator(precip_threshold=1.0, device=device)
        for pred_ens, target in local_timesteps:
            acc.update(pred_ens, target)
        acc.reduce()                          # all_reduce across ranks
        if dist.rank == 0:
            metrics = acc.to_dict()
    """

    _DEFAULT_BIN_EDGES = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, float("inf")]

    def __init__(
        self,
        precip_threshold: float = 1.0,
        device=None,
        spread_skill_bin_edges=None,
        twcrps_thresholds=None,
        skip_spread_skill: bool = False,
        bin_mode: str = "fixed",
        n_quantile_bins: int = 20,
    ):
        """
        Args:
            precip_threshold: Threshold (mm) for conditional "high precip" metrics.
            device: Torch device.  Defaults to CPU.
            spread_skill_bin_edges: Bin edge list for spread-skill reliability.
                Defaults to [0, 0.1, 0.25, 0.5, 1, 2, 4, 8, inf].
                Ignored when bin_mode="quantile".
            twcrps_thresholds: List of precipitation thresholds (mm) for twCRPS.
                Defaults to [5.0, 10.0].
            skip_spread_skill: If True, skip spread/skill/spread-skill ratio
                computation and omit those keys from to_dict(). Use for
                regression-only evaluation where spread is always 0.
            bin_mode: "fixed" (default) bins pixels by fixed spread edges;
                "quantile" uses equal-count bins computed from the data,
                giving each point on the spread-skill plot equal statistical
                weight.
            n_quantile_bins: Number of equal-count bins for bin_mode="quantile".
                Ignored when bin_mode="fixed".
        """
        self.precip_threshold = precip_threshold
        self.device = device if device is not None else torch.device("cpu")
        self.twcrps_thresholds = list(twcrps_thresholds) if twcrps_thresholds is not None else [5.0, 10.0]
        self.skip_spread_skill = skip_spread_skill
        self.bin_mode = bin_mode
        self.n_quantile_bins = n_quantile_bins

        if spread_skill_bin_edges is None:
            spread_skill_bin_edges = self._DEFAULT_BIN_EDGES
        self.bin_edges = torch.tensor(
            spread_skill_bin_edges, device=self.device, dtype=torch.float32
        )
        self.n_bins = self.bin_edges.numel() - 1
        self._reset()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset(self):
        # Scalar accumulators (float, accumulated as Python floats on CPU)
        self.se_sum = 0.0
        self.n_elements = 0
        self.crps_sum = 0.0
        self.crps_elements = 0
        self.variance_sum = 0.0  # accumulates mean(ensemble_variance) per sample; sqrt taken in to_dict()
        self.skill_sum = 0.0
        self.n_samples = 0

        # Conditional: target > threshold
        self.se_sum_gt = 0.0
        self.n_gt = 0
        self.crps_sum_gt = 0.0
        self.crps_n_gt = 0

        # Conditional: 95th percentile (per sample)
        self.se_sum_95 = 0.0
        self.n_95 = 0
        self.crps_sum_95 = 0.0
        self.crps_n_95 = 0

        # Conditional: wet-95th (95th pct over wet pixels)
        self.se_sum_w95 = 0.0
        self.n_w95 = 0
        self.crps_sum_w95 = 0.0
        self.crps_n_w95 = 0

        # Threshold-weighted CRPS (per threshold)
        self.twcrps_sums = [0.0] * len(self.twcrps_thresholds)
        self.twcrps_n = [0] * len(self.twcrps_thresholds)

        # Spread-Skill reliability bins (float64 tensors)
        if self.bin_mode == "quantile":
            # Fine grid (500 bins, 0–30 mm) accumulates (count, skill_sum) per
            # narrow spread bucket; quantile edges are computed lazily in to_dict().
            self._fine_n = 500
            self._fine_max = 30.0  # mm; values above are clamped to the last bin
            self.fine_count = torch.zeros(self._fine_n, device=self.device, dtype=torch.float64)
            self.fine_skill_sum = torch.zeros(self._fine_n, device=self.device, dtype=torch.float64)
        else:
            self.spread_bin_sum = torch.zeros(self.n_bins, device=self.device, dtype=torch.float64)
            self.skill_bin_sum = torch.zeros(self.n_bins, device=self.device, dtype=torch.float64)
            self.bin_count = torch.zeros(self.n_bins, device=self.device, dtype=torch.float64)

        # Rank histogram (allocated lazily on first update, size = n_ens + 1)
        self.rank_counts = None

    def _scalars_as_tensor(self) -> torch.Tensor:
        base = [
            self.se_sum, float(self.n_elements),
            self.crps_sum, float(self.crps_elements),
            self.variance_sum, self.skill_sum, float(self.n_samples),
            self.se_sum_gt, float(self.n_gt),
            self.crps_sum_gt, float(self.crps_n_gt),
            self.se_sum_95, float(self.n_95),
            self.crps_sum_95, float(self.crps_n_95),
            self.se_sum_w95, float(self.n_w95),
            self.crps_sum_w95, float(self.crps_n_w95),
        ]
        # Append twCRPS: [sum_t0, n_t0, sum_t1, n_t1, ...]
        for s, n in zip(self.twcrps_sums, self.twcrps_n):
            base.extend([s, float(n)])
        return torch.tensor(base, dtype=torch.float64, device=self.device)

    def _load_scalars_from_tensor(self, t: torch.Tensor):
        self.se_sum = t[0].item()
        self.n_elements = int(t[1].item())
        self.crps_sum = t[2].item()
        self.crps_elements = int(t[3].item())
        self.variance_sum = t[4].item()
        self.skill_sum = t[5].item()
        self.n_samples = int(t[6].item())
        self.se_sum_gt = t[7].item()
        self.n_gt = int(t[8].item())
        self.crps_sum_gt = t[9].item()
        self.crps_n_gt = int(t[10].item())
        self.se_sum_95 = t[11].item()
        self.n_95 = int(t[12].item())
        self.crps_sum_95 = t[13].item()
        self.crps_n_95 = int(t[14].item())
        self.se_sum_w95 = t[15].item()
        self.n_w95 = int(t[16].item())
        self.crps_sum_w95 = t[17].item()
        self.crps_n_w95 = int(t[18].item())
        # twCRPS starts at index 19, two values per threshold
        for i in range(len(self.twcrps_thresholds)):
            offset = 19 + 2 * i
            self.twcrps_sums[i] = t[offset].item()
            self.twcrps_n[i] = int(t[offset + 1].item())

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update(self, pred_ens: torch.Tensor, target: torch.Tensor):
        """Accumulate metrics for one timestep.

        Args:
            pred_ens: (N_ens, C, H, W) ensemble predictions in physical units.
            target:   (C, H, W) or (1, C, H, W) ground truth in physical units.
        """
        pred_ens = pred_ens.to(self.device)
        target = target.to(self.device)

        if target.ndim == 4:
            target = target.squeeze(0)  # -> (C, H, W)

        # Clip to >= 0 (precipitation cannot be negative)
        pred_ens = torch.clamp(pred_ens, min=0.0)
        target = torch.clamp(target, min=0.0)

        ens_mean = pred_ens.mean(dim=0)  # (C, H, W)

        # RMSE
        se = (ens_mean - target) ** 2
        self.se_sum += se.sum().item()
        self.n_elements += se.numel()

        # Proper finite-ensemble CRPS
        crps_vals = proper_crps(pred_ens, target)
        self.crps_sum += crps_vals.sum().item()
        self.crps_elements += crps_vals.numel()

        # Threshold-weighted CRPS
        for i, thresh in enumerate(self.twcrps_thresholds):
            tw = proper_twcrps(pred_ens, target, thresh)
            self.twcrps_sums[i] += tw.sum().item()
            self.twcrps_n[i] += tw.numel()

        self.n_samples += 1

        # Spread & Skill (per-sample, averaged in to_dict)
        # Fortin et al. (2014): aggregate spread = sqrt(mean(variance)), not mean(std).
        # Accumulate mean variance with finite-ensemble correction (R+1)/R, take sqrt in to_dict().
        n_ens = pred_ens.shape[0]
        if not self.skip_spread_skill:
            ens_var = pred_ens.var(dim=0, unbiased=False) * (1.0 + 1.0 / n_ens)
            self.variance_sum += ens_var.mean().item()
            self.skill_sum += torch.sqrt(se.mean()).item()

        # Conditional: target > threshold
        mask_gt = target > self.precip_threshold
        if mask_gt.any():
            self.se_sum_gt += se[mask_gt].sum().item()
            self.n_gt += mask_gt.sum().item()
            self.crps_sum_gt += crps_vals[mask_gt].sum().item()
            self.crps_n_gt += mask_gt.sum().item()

        # Conditional: 95th percentile (per sample)
        p95 = torch.quantile(target.flatten(), 0.95)
        mask_95 = target >= p95
        if mask_95.any():
            self.se_sum_95 += se[mask_95].sum().item()
            self.n_95 += mask_95.sum().item()
            self.crps_sum_95 += crps_vals[mask_95].sum().item()
            self.crps_n_95 += mask_95.sum().item()

        # Conditional: wet-95th (95th pct over wet pixels only)
        wet = target > 0.0
        if wet.any():
            p95_wet = torch.quantile(target[wet], 0.95)
            mask_w95 = wet & (target >= p95_wet)
            if mask_w95.any():
                self.se_sum_w95 += se[mask_w95].sum().item()
                self.n_w95 += mask_w95.sum().item()
                self.crps_sum_w95 += crps_vals[mask_w95].sum().item()
                self.crps_n_w95 += mask_w95.sum().item()

        # Spread-Skill reliability bins
        if not self.skip_spread_skill:
            spread_flat = torch.sqrt(ens_var).flatten()
            skill_flat = torch.sqrt(se).flatten()
            if self.bin_mode == "quantile":
                idx = (spread_flat / self._fine_max * self._fine_n).long().clamp(0, self._fine_n - 1)
                ones = torch.ones(len(idx), dtype=torch.float64, device=self.device)
                self.fine_count.scatter_add_(0, idx, ones)
                self.fine_skill_sum.scatter_add_(0, idx, skill_flat.to(torch.float64))
            else:
                bin_ids = torch.bucketize(
                    spread_flat,
                    boundaries=self.bin_edges[1:-1],
                    right=False,
                )
                for b in range(self.n_bins):
                    in_bin = bin_ids == b
                    if in_bin.any():
                        self.spread_bin_sum[b] += spread_flat[in_bin].sum().to(torch.float64)
                        self.skill_bin_sum[b] += skill_flat[in_bin].sum().to(torch.float64)
                        self.bin_count[b] += in_bin.sum().to(torch.float64)

        # Rank histogram: for each pixel, count how many members are below target
        # Rank ranges from 0 (target below all members) to n_ens (target above all)
        target_flat = target.flatten()  # (C*H*W,)
        pred_flat = pred_ens.flatten(1)  # (n_ens, C*H*W)
        ranks = (pred_flat < target_flat.unsqueeze(0)).sum(dim=0)  # (C*H*W,)
        if self.rank_counts is None:
            self.rank_counts = torch.zeros(n_ens + 1, device=self.device, dtype=torch.float64)
        counts = torch.bincount(ranks.long(), minlength=n_ens + 1).to(torch.float64)
        self.rank_counts += counts[:n_ens + 1]

    def reduce(self) -> None:
        """All-reduce raw accumulators across all distributed ranks (in-place).

        Must be called before ``to_dict()`` when using more than one GPU.
        After this call every rank holds the same global totals.
        """
        if (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
        ):
            return

        scalars = self._scalars_as_tensor()
        torch.distributed.all_reduce(scalars, op=torch.distributed.ReduceOp.SUM)
        self._load_scalars_from_tensor(scalars)

        if not self.skip_spread_skill:
            if self.bin_mode == "quantile":
                torch.distributed.all_reduce(self.fine_count, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(self.fine_skill_sum, op=torch.distributed.ReduceOp.SUM)
            else:
                torch.distributed.all_reduce(self.spread_bin_sum, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(self.skill_bin_sum, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(self.bin_count, op=torch.distributed.ReduceOp.SUM)

        if self.rank_counts is not None:
            torch.distributed.all_reduce(self.rank_counts, op=torch.distributed.ReduceOp.SUM)

    def to_dict(self, prefix: str = "") -> dict:
        """Compute final metrics from accumulated sums.

        Ratios (spread-skill ratio) are computed from the global totals
        *after* reduce(), never from per-rank averages.

        Args:
            prefix: Optional string to prepend to every metric key
                    (e.g. ``"regression/"`` or ``"diffusion/"``).

        Returns:
            Flat dict suitable for ``wandb.log()`` or JSON serialisation.
        """
        n = self.n_samples
        if n == 0:
            return {}

        out = {
            "rmse": float(np.sqrt(self.se_sum / self.n_elements))
                    if self.n_elements > 0 else 0.0,
            "crps": float(self.crps_sum / self.crps_elements)
                    if self.crps_elements > 0 else 0.0,
            "n_samples": n,
            f"rmse_gt_{self.precip_threshold}mm":
                float(np.sqrt(self.se_sum_gt / self.n_gt)) if self.n_gt > 0 else 0.0,
            f"crps_gt_{self.precip_threshold}mm":
                float(self.crps_sum_gt / self.crps_n_gt) if self.crps_n_gt > 0 else 0.0,
            "rmse_95th":
                float(np.sqrt(self.se_sum_95 / self.n_95)) if self.n_95 > 0 else 0.0,
            "crps_95th":
                float(self.crps_sum_95 / self.crps_n_95) if self.crps_n_95 > 0 else 0.0,
            "rmse_w95th":
                float(np.sqrt(self.se_sum_w95 / self.n_w95)) if self.n_w95 > 0 else 0.0,
            "crps_w95th":
                float(self.crps_sum_w95 / self.crps_n_w95) if self.crps_n_w95 > 0 else 0.0,
        }

        # Threshold-weighted CRPS
        for thresh, s, tw_n in zip(self.twcrps_thresholds, self.twcrps_sums, self.twcrps_n):
            key = f"twcrps_{thresh}mm"
            out[key] = float(s / tw_n) if tw_n > 0 else 0.0

        # Spread, Skill, Spread-Skill ratio & reliability diagnostics
        if not self.skip_spread_skill:
            # Fortin et al. (2014): spread = sqrt(mean(variance)), not mean(std)
            spread = float(np.sqrt(self.variance_sum / n)) if n > 0 else 0.0
            skill = self.skill_sum / n
            global_rmse = float(np.sqrt(self.se_sum / self.n_elements)) if self.n_elements > 0 else 0.0
            out["spread"] = float(spread)
            out["skill"] = float(skill)
            out["spread_skill_ratio"] = float(spread / skill) if skill > 0 else float("inf")
            out["spread_skill_ratio_global"] = float(spread / global_rmse) if global_rmse > 0 else float("inf")

            if self.bin_mode == "quantile":
                total = self.fine_count.sum()
                if total > 0:
                    fine_centers = (
                        (torch.arange(self._fine_n, dtype=torch.float64, device=self.device) + 0.5)
                        / self._fine_n * self._fine_max
                    )
                    cdf = torch.cumsum(self.fine_count, 0) / total
                    # Assign each fine bin to a quantile bin by where its CDF value falls
                    q_boundaries = torch.linspace(0, 1, self.n_quantile_bins + 1, device=self.device)[1:-1]
                    q_bin_ids = torch.bucketize(cdf, q_boundaries, right=False).clamp(0, self.n_quantile_bins - 1)
                    mean_spread_q = torch.zeros(self.n_quantile_bins, dtype=torch.float64, device=self.device)
                    mean_skill_q = torch.zeros(self.n_quantile_bins, dtype=torch.float64, device=self.device)
                    bin_count_q = torch.zeros(self.n_quantile_bins, dtype=torch.float64, device=self.device)
                    for b in range(self.n_quantile_bins):
                        m = (q_bin_ids == b) & (self.fine_count > 0)
                        if m.any():
                            cnt = self.fine_count[m].sum()
                            mean_spread_q[b] = (fine_centers[m] * self.fine_count[m]).sum() / cnt
                            mean_skill_q[b] = self.fine_skill_sum[m].sum() / cnt
                            bin_count_q[b] = cnt
                    valid = bin_count_q > 0
                    if valid.any():
                        x, y, w = mean_skill_q[valid], mean_spread_q[valid], bin_count_q[valid]
                        out["spread_skill_bin_mean_spread"] = mean_spread_q[valid].cpu().tolist()
                        out["spread_skill_bin_mean_skill"] = mean_skill_q[valid].cpu().tolist()
                        out["spread_skill_bin_mode"] = "quantile"
            else:
                valid = self.bin_count > 0
                if valid.any():
                    mean_spread = torch.zeros_like(self.bin_count)
                    mean_skill = torch.zeros_like(self.bin_count)
                    mean_spread[valid] = self.spread_bin_sum[valid] / self.bin_count[valid]
                    mean_skill[valid] = self.skill_bin_sum[valid] / self.bin_count[valid]
                    x, y, w = mean_skill[valid], mean_spread[valid], self.bin_count[valid]
                    out["spread_skill_bin_edges"] = self.bin_edges.cpu().tolist()
                    out["spread_skill_bin_mean_spread"] = mean_spread.cpu().tolist()
                    out["spread_skill_bin_mean_skill"] = mean_skill.cpu().tolist()
                    out["spread_skill_bin_mode"] = "fixed"

            # Weighted linear fit shared by both modes
            if "spread_skill_bin_mean_spread" in out:
                w_sum = w.sum()
                x_bar = (w * x).sum() / w_sum
                y_bar = (w * y).sum() / w_sum
                s_xx = (w * (x - x_bar) ** 2).sum()
                if s_xx > 0:
                    slope = ((w * (x - x_bar) * (y - y_bar)).sum() / s_xx).item()
                else:
                    slope = float("nan")
                intercept = (
                    (y_bar - slope * x_bar).item() if np.isfinite(slope) else float("nan")
                )
                out["spread_skill_reliability_slope"] = slope
                out["spread_skill_reliability_intercept"] = intercept

        # Rank histogram
        if self.rank_counts is not None:
            out["rank_histogram"] = self.rank_counts.cpu().tolist()

        if prefix:
            out = {f"{prefix}{k}": v for k, v in out.items()}

        return out
