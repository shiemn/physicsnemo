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

"""Field styles for the paper-protocol plots — single source of truth.

A ``FieldStyle`` bundles the colormap, the normalization mode, the unit label,
and (for discrete modes) the level edges for one plotted quantity.  The
``STYLES`` registry maps a quantity name (``precip``, ``rmse``, ``crps``,
``relbias``, ``ooe``, ``spread``) to its style, so colours/units/levels live in
exactly one place and every figure draws consistently.

``resolve_norm(style, field)`` turns a style + a data field into a matplotlib
``Normalize``/``BoundaryNorm`` and a (possibly discrete) colormap, ready for
``pcolormesh``/``colorbar``.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, Normalize


# Paper colorbar level edges (Figs 3/4 relative bias %, Fig 7 out-of-envelope mm).
_RELBIAS_LEVELS = [-100, -60, -40, -20, -15, -10, -5, 5, 10, 15, 20, 40, 60, 100]
_OOE_LEVELS = [-50, -25, -15, -10, -5, -2.5, 2.5, 5, 10, 15]


@dataclass(frozen=True)
class FieldStyle:
    """Visual style for one plotted quantity.

    Args:
        cmap:   matplotlib colormap name.
        mode:   "sequential"  -> Normalize(0, p99(field))
                "diverging"   -> symmetric Normalize(-p99(|field|), +p99(|field|))
                "relbias"     -> discrete BoundaryNorm over fixed % levels
                "ooe"         -> discrete BoundaryNorm over fixed mm levels
        unit:   colorbar label.
        levels: explicit level edges (discrete modes); falls back to mode default.
    """

    cmap: str
    mode: str
    unit: str = ""
    levels: tuple[float, ...] | None = None


STYLES: dict[str, FieldStyle] = {
    # Absolute reference / model fields
    "precip": FieldStyle(cmap="Blues", mode="sequential", unit="mm"),
    "spread": FieldStyle(cmap="YlOrRd", mode="sequential", unit="mm"),
    # Metric fields
    "rmse": FieldStyle(cmap="magma_r", mode="sequential", unit="RMSE (mm)"),
    "crps": FieldStyle(cmap="Oranges", mode="sequential", unit="CRPS / MAE (mm)"),
    "relbias": FieldStyle(cmap="BrBG", mode="relbias", unit="Relative bias (%)",
                          levels=tuple(_RELBIAS_LEVELS)),
    "ooe": FieldStyle(cmap="PuOr", mode="ooe", unit="Exceed. / error (mm)",
                      levels=tuple(_OOE_LEVELS)),
}


def get_style(name: str) -> FieldStyle:
    """Return a registered style, raising a clear error for unknown names."""
    try:
        return STYLES[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown field style {name!r}; known: {sorted(STYLES)}"
        ) from exc


def _finite(field: np.ndarray) -> np.ndarray:
    arr = np.asarray(field, dtype=float).ravel()
    return arr[np.isfinite(arr)]


def resolve_norm(style: FieldStyle, fields):
    """Resolve a style + one or more fields into (cmap, norm, is_discrete).

    Args:
        style:  the FieldStyle.
        fields: a single 2-D array or a list of arrays (pooled to set the range
                for the continuous modes so a group of panels shares one scale).

    Returns:
        (cmap, norm, is_discrete) where ``cmap`` is a matplotlib Colormap,
        ``norm`` a Normalize/BoundaryNorm, and ``is_discrete`` True for the
        level-based modes (so the caller can add an extended colorbar).
    """
    if isinstance(fields, (list, tuple)):
        pooled = np.concatenate([_finite(f) for f in fields if f is not None]) \
            if any(f is not None for f in fields) else np.array([0.0])
    else:
        pooled = _finite(fields)
    if pooled.size == 0:
        pooled = np.array([0.0, 1.0])

    if style.mode == "sequential":
        vmax = float(np.nanpercentile(pooled, 99))
        vmax = vmax if vmax > 0 else 1.0
        return plt.get_cmap(style.cmap), Normalize(vmin=0.0, vmax=vmax), False

    if style.mode == "diverging":
        vmax = float(np.nanpercentile(np.abs(pooled), 99))
        vmax = vmax if vmax > 0 else 1.0
        return plt.get_cmap(style.cmap), Normalize(vmin=-vmax, vmax=vmax), False

    # Discrete level-based modes (relbias, ooe)
    levels = list(style.levels) if style.levels is not None else (
        _RELBIAS_LEVELS if style.mode == "relbias" else _OOE_LEVELS
    )
    nbands = len(levels) - 1
    cmap = plt.get_cmap(style.cmap, nbands).copy()
    cmap.set_under(cmap(0))
    cmap.set_over(cmap(nbands - 1))
    norm = BoundaryNorm(levels, ncolors=nbands)
    return cmap, norm, True
