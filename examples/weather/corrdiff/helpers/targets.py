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

"""Targeted-day spatial diagnostics for the paper-protocol eval flow.

Implements the JAMES-paper Section-3.2 per-target metrics:

* ``crps_map``            spatial (per-gridpoint) CRPS field, reusing the proper
                          finite-ensemble formula from ``helpers.metrics``.
* ``out_of_envelope_map`` per-gridpoint signed distance of the reference value
                          from the ensemble [min, max] envelope (0 if inside),
                          available raw or after a 3x3 moving average.
* ``sal_scores``          pySTEPS-compatible Structure / Amplitude / Location
                          score (Wernli et al. 2008, 2009). If pySTEPS is
                          installed, call it directly; otherwise use a vendored
                          implementation with pySTEPS threshold semantics.

All inputs are **denormalized (physical, mm)** fields.  Works for any ensemble
size N>=1.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage

from helpers.metrics import proper_crps
from helpers.climatology import moving_average_2d

try:
    from pysteps.verification.salscores import sal as _pysteps_sal
except ImportError:
    _pysteps_sal = None


def crps_map(
    pred_ens: np.ndarray, target: np.ndarray, smooth_size: int | None = None
) -> np.ndarray:
    """Per-gridpoint CRPS field (no spatial reduction).

    Args:
        pred_ens:    (N, H, W) ensemble predictions, physical units.
        target:      (H, W) reference, physical units.
        smooth_size: if set (e.g. 3), apply a moving-average to members and
                     reference before scoring (paper Fig 8 uses a prior 3x3
                     window; the unsmoothed map is the paper's Fig S2).

    Returns:
        (H, W) CRPS field.  Collapses to MAE for a degenerate (N=1) ensemble.
    """
    pred = np.asarray(pred_ens)
    tar = np.asarray(target)
    if smooth_size:
        pred = moving_average_2d(pred, smooth_size)
        tar = moving_average_2d(tar, smooth_size)
    pred_t = torch.as_tensor(pred, dtype=torch.float64)
    tar_t = torch.as_tensor(tar, dtype=torch.float64)
    return proper_crps(pred_t, tar_t).cpu().numpy()


def out_of_envelope_map(
    pred_ens: np.ndarray, target: np.ndarray, smooth_size: int | None = None
) -> np.ndarray:
    """Signed out-of-envelope error map (paper Section 2.3).

    For each gridpoint with reference value ``y`` and member set ``S``::

        y - max(S)  if y > max(S)      (reference above the envelope)
        y - min(S)  if y < min(S)      (reference below the envelope)
        0           otherwise          (reference inside the envelope)

    Args:
        pred_ens:    (N, H, W) ensemble predictions, physical units.
        target:      (H, W) reference, physical units.
        smooth_size: if set (e.g. 3), apply a moving-average to members and
                     reference before computing the envelope, per the paper's
                     "prior 3x3 moving averages" variant.

    Returns:
        (H, W) signed error map (mm); 0 where the reference is within the envelope.
    """
    pred = np.asarray(pred_ens, dtype=np.float64)
    tar = np.asarray(target, dtype=np.float64)
    if smooth_size:
        pred = moving_average_2d(pred, smooth_size)
        tar = moving_average_2d(tar, smooth_size)
    env_min = pred.min(axis=0)
    env_max = pred.max(axis=0)
    err = np.zeros_like(tar)
    above = tar > env_max
    below = tar < env_min
    err[above] = tar[above] - env_max[above]
    err[below] = tar[below] - env_min[below]
    return err


def _pysteps_threshold(field: np.ndarray, thr_factor: float, thr_quantile: float) -> float:
    """pySTEPS SAL threshold: ``thr_factor * quantile(field > min(field))``."""
    arr = np.asarray(field, dtype=np.float64)
    if arr.size == 0 or not np.isfinite(arr).any():
        return float("inf")
    zero_value = np.nanmin(arr)
    wet = arr[arr > zero_value]
    if wet.size == 0:
        return float("inf")
    return float(thr_factor * np.nanquantile(wet, thr_quantile))


def _sal_detect_objects(field: np.ndarray, thr_factor: float, thr_quantile: float) -> list[dict]:
    """Detect SAL objects with pySTEPS threshold semantics.

    This fallback intentionally keeps object detection simple: connected
    components above the pySTEPS threshold. When pySTEPS itself is installed,
    ``sal_scores`` calls pySTEPS directly, including its thunderstorm detector.
    """
    arr = np.nan_to_num(np.asarray(field, dtype=np.float64), nan=0.0)
    threshold = _pysteps_threshold(arr, thr_factor, thr_quantile)
    mask = arr >= threshold if np.isfinite(threshold) else np.zeros_like(arr, bool)
    labels, n_obj = ndimage.label(mask)
    objects: list[dict] = []
    if n_obj <= 0:
        return objects
    obj_ids = np.arange(1, n_obj + 1)
    obj_sums = np.atleast_1d(ndimage.sum(arr, labels, obj_ids)).astype(float)
    obj_max = np.atleast_1d(ndimage.maximum(arr, labels, obj_ids)).astype(float)
    obj_coms = np.atleast_2d(np.array(ndimage.center_of_mass(arr, labels, obj_ids)))
    for label, intensity_sum, max_intensity, centroid in zip(
        obj_ids, obj_sums, obj_max, obj_coms
    ):
        objects.append(
            {
                "label": int(label),
                "intensity_sum": float(intensity_sum),
                "max_intensity": float(max_intensity),
                "weighted_centroid": np.asarray(centroid, dtype=float),
            }
        )
    return objects


def _sal_scaled_volume(objects: list[dict]) -> float:
    """pySTEPS scaled volume descriptor for detected SAL objects."""
    if not objects:
        return 0.0
    weighted_volumes = []
    intensity_sums = []
    for obj in objects:
        intensity_sum = float(obj["intensity_sum"])
        max_intensity = float(obj["max_intensity"])
        if intensity_sum == 0.0 or max_intensity <= 0.0:
            intensity_vol = 0.0
        else:
            intensity_vol = intensity_sum * (intensity_sum / max_intensity)
        weighted_volumes.append(intensity_vol)
        intensity_sums.append(intensity_sum)
    total = np.nansum(intensity_sums)
    return float(np.nansum(weighted_volumes) / total) if total > 0 else 0.0


def _sal_l1(prediction: np.ndarray, observation: np.ndarray) -> float:
    """pySTEPS first location component."""
    maximum_distance = np.sqrt(observation.shape[0] ** 2 + observation.shape[1] ** 2)
    if maximum_distance <= 0:
        return float("nan")
    obs_com = np.array(ndimage.center_of_mass(np.nan_to_num(observation)))
    pred_com = np.array(ndimage.center_of_mass(np.nan_to_num(prediction)))
    dist = np.hypot(pred_com[1] - obs_com[1], pred_com[0] - obs_com[0])
    return float(dist / maximum_distance)


def _sal_weighted_distance(
    field: np.ndarray, thr_factor: float, thr_quantile: float
) -> float:
    """pySTEPS weighted object distance from total-field center of mass."""
    arr = np.nan_to_num(np.asarray(field, dtype=np.float64), nan=0.0)
    objects = _sal_detect_objects(arr, thr_factor, thr_quantile)
    if not objects:
        return float("nan")
    centroid_total = np.array(ndimage.center_of_mass(arr))
    weighted_dist_sum = 0.0
    precip_sum = 0.0
    for obj in objects:
        centroid = np.asarray(obj["weighted_centroid"], dtype=float)
        dist = np.sqrt((centroid[1] - centroid_total[1]) ** 2 + (centroid[0] - centroid_total[0]) ** 2)
        intensity_sum = float(obj["intensity_sum"])
        weighted_dist_sum += intensity_sum * dist
        precip_sum += intensity_sum
    return float(weighted_dist_sum / precip_sum) if precip_sum > 0 else float("nan")


def _sal_l2(
    prediction: np.ndarray,
    observation: np.ndarray,
    thr_factor: float,
    thr_quantile: float,
) -> float:
    maximum_distance = np.sqrt(observation.shape[0] ** 2 + observation.shape[1] ** 2)
    if maximum_distance <= 0:
        return float("nan")
    obs_r = _sal_weighted_distance(observation, thr_factor, thr_quantile)
    pred_r = _sal_weighted_distance(prediction, thr_factor, thr_quantile)
    return float(2.0 * (abs(obs_r - pred_r) / maximum_distance))


def _vendored_pysteps_sal(
    prediction: np.ndarray,
    observation: np.ndarray,
    thr_factor: float,
    thr_quantile: float,
) -> dict:
    """pySTEPS SAL formulas with connected-component object detection fallback."""
    pred = np.asarray(prediction, dtype=np.float64)
    obs = np.asarray(observation, dtype=np.float64)
    pred_objects = _sal_detect_objects(pred, thr_factor, thr_quantile)
    obs_objects = _sal_detect_objects(obs, thr_factor, thr_quantile)
    pred_volume = _sal_scaled_volume(pred_objects)
    obs_volume = _sal_scaled_volume(obs_objects)
    with np.errstate(divide="ignore", invalid="ignore"):
        S = np.divide(pred_volume - obs_volume, 0.5 * (pred_volume + obs_volume))
    mean_obs = np.nanmean(obs)
    mean_pred = np.nanmean(pred)
    with np.errstate(divide="ignore", invalid="ignore"):
        A = np.divide(mean_pred - mean_obs, 0.5 * (mean_pred + mean_obs))
    L1 = _sal_l1(pred, obs)
    L2 = _sal_l2(pred, obs, thr_factor, thr_quantile)
    return {"S": float(S), "A": float(A), "L": float(L1 + L2), "L1": L1, "L2": L2}


def sal_scores(
    pred_field: np.ndarray,
    ref_field: np.ndarray,
    f_factor: float = 1.0 / 15.0,
    thr_quantile: float = 0.95,
) -> dict:
    """pySTEPS-compatible Structure-Amplitude-Location scores.

    pySTEPS computes each field's object threshold as
    ``thr_factor * quantile(field > min(field), thr_quantile)`` and uses that
    object set for the structure and second location components. If pySTEPS is
    installed, this function calls it directly; otherwise it uses the same SAL
    equations with scipy connected-component object detection.

    Args:
        pred_field: (H, W) predicted/simulated field, physical units.
        ref_field:  (H, W) reference field, physical units.
        f_factor:     SAL object-threshold factor, default 1/15.
        thr_quantile: Wet quantile used by pySTEPS to derive thresholds.

    Returns:
        dict with keys "S", "A", "L" (and "L1", "L2" for the fallback path).
    """
    pred_field = np.asarray(pred_field, dtype=np.float64)
    ref_field = np.asarray(ref_field, dtype=np.float64)
    if _pysteps_sal is not None:
        S, A, L = _pysteps_sal(
            pred_field,
            ref_field,
            thr_factor=f_factor,
            thr_quantile=thr_quantile,
        )
        return {"S": float(S), "A": float(A), "L": float(L), "L1": float("nan"), "L2": float("nan")}
    return _vendored_pysteps_sal(pred_field, ref_field, f_factor, thr_quantile)


def sal_distribution(
    pred_ens: np.ndarray,
    ref_field: np.ndarray,
    f_factor: float = 1.0 / 15.0,
    thr_quantile: float = 0.95,
) -> dict:
    """Per-member pySTEPS-style SAL scores for one target.

    Args:
        pred_ens:  (N, H, W) ensemble predictions, physical units.
        ref_field: (H, W) reference, physical units.
        f_factor:     pySTEPS object-threshold factor.
        thr_quantile: pySTEPS wet quantile for object thresholding.

    Returns:
        dict with arrays "S", "A", "L" of length N (NaNs dropped per member only
        where a component is undefined).
    """
    pred_ens = np.asarray(pred_ens, dtype=np.float64)
    S, A, L = [], [], []
    for m in range(pred_ens.shape[0]):
        sal = sal_scores(pred_ens[m], ref_field, f_factor=f_factor, thr_quantile=thr_quantile)
        S.append(sal["S"])
        A.append(sal["A"])
        L.append(sal["L"])
    ref_threshold = _pysteps_threshold(ref_field, f_factor, thr_quantile)
    return {
        "S": np.array(S),
        "A": np.array(A),
        "L": np.array(L),
        "threshold": float(ref_threshold),
        "thr_quantile": float(thr_quantile),
        "implementation": "pysteps" if _pysteps_sal is not None else "vendored_pysteps",
    }
