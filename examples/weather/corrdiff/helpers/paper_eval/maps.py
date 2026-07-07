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

"""Map-grid primitive — the single function behind paper Figs 2, 3, 4, 7, 8.

``plot_map_grid`` renders a rows × columns grid of 2-D fields with per-column
styles and one shared colorbar per distinct style group, optional cartopy
coastlines, and an optional per-panel ``P5 / Mean / P95`` annotation.  Every
spatial figure in the paper-protocol flow is a configuration of this one call.
"""

from __future__ import annotations

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable

from .styles import FieldStyle, resolve_norm

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    _CARTOPY_AVAILABLE = True
except ImportError:
    _CARTOPY_AVAILABLE = False


def _use_cartopy(lat, lon) -> bool:
    return _CARTOPY_AVAILABLE and lat is not None and lon is not None


def _coords(field, lat, lon):
    if lat is not None and lon is not None:
        return lon, lat
    H, W = field.shape[-2], field.shape[-1]
    return np.arange(W), np.arange(H)


def geo_axes(fig, cell, lat, lon):
    """Create an axis for a gridspec ``cell``, georeferenced if cartopy is usable."""
    if _use_cartopy(lat, lon):
        return fig.add_subplot(cell, projection=ccrs.PlateCarree())
    return fig.add_subplot(cell)


def draw_field(ax, field, x, y, cmap, norm, lat, lon):
    """Draw one field into ``ax`` with the given cmap/norm; add coastlines."""
    if _use_cartopy(lat, lon):
        ax.add_feature(cfeature.COASTLINE, linewidth=0.7, edgecolor="black")
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":", edgecolor="black")
        return ax.pcolormesh(
            x, y, field, cmap=cmap, norm=norm,
            transform=ccrs.PlateCarree(), shading="auto",
        )
    ax.set_aspect("equal", adjustable="box")
    return ax.pcolormesh(x, y, field, cmap=cmap, norm=norm, shading="auto")


def _annotation(field: np.ndarray) -> str:
    vals = np.asarray(field, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return ""
    p5, p95 = np.percentile(vals, [5, 95])
    return f"P5: {p5:.2f}  Mean: {vals.mean():.2f}  P95: {p95:.2f}"


def _array_panel_aspect(fields: dict) -> float:
    """Return representative ``height / width`` for ungeoreferenced fields."""
    for field in fields.values():
        arr = np.asarray(field)
        if arr.ndim >= 2 and arr.shape[-1] > 0:
            return float(arr.shape[-2]) / float(arr.shape[-1])
    return 1.0


def plot_map_grid(
    fields: dict,
    rows: list[str],
    columns: list[str],
    style: FieldStyle | None = None,
    column_styles: dict | None = None,
    lat: np.ndarray | None = None,
    lon: np.ndarray | None = None,
    annotate: bool = True,
    title: str | None = None,
    panel_size: float = 3.2,
    panel_height: float | None = None,
) -> plt.Figure:
    """Render a rows × columns grid of 2-D map panels.

    Args:
        fields:        dict ``(row_label, col_label) -> 2-D array`` (NaN-safe);
                       missing keys leave a blank panel.
        rows, columns: ordered label lists defining the grid.
        style:         a single FieldStyle applied to every column.
        column_styles: optional ``{col_label: FieldStyle}`` overriding ``style``
                       per column (used for "Reference (absolute) + metric cols").
        lat, lon:      (H, W) arrays for georeferencing (optional).
        annotate:      draw a per-panel ``P5 / Mean / P95`` line.
        title:         figure suptitle.
        panel_size:    inches per panel width.
        panel_height:  optional inches per row; defaults to ``panel_size``.

    Returns:
        matplotlib Figure with one shared colorbar per distinct style group.
    """
    if style is None and column_styles is None:
        raise ValueError("Provide either style or column_styles.")

    col_style = {c: (column_styles or {}).get(c, style) for c in columns}
    if any(s is None for s in col_style.values()):
        raise ValueError("Every column needs a style (via style or column_styles).")

    nrows, ncols = len(rows), len(columns)

    # Group columns by the style object so each group gets one shared colorbar.
    groups: dict[int, list[str]] = {}
    for c in columns:
        groups.setdefault(id(col_style[c]), []).append(c)
    n_groups = len(groups)

    has_row_labels = any(str(row) for row in rows)
    if panel_height is not None:
        row_height = panel_height
    elif lat is None or lon is None:
        row_height = panel_size * _array_panel_aspect(fields)
    else:
        row_height = panel_size
    fig = plt.figure(
        figsize=(
            panel_size * ncols + 1.2 + (1.1 if has_row_labels else 0.0),
            row_height * nrows + 1.0,
        )
    )
    # Reserve a thin right strip for the per-group colorbars.
    outer = gridspec.GridSpec(
        1, 2, width_ratios=[ncols, 0.35 * n_groups], wspace=0.18, figure=fig
    )
    grid = gridspec.GridSpecFromSubplotSpec(
        nrows, ncols, subplot_spec=outer[0], hspace=0.28, wspace=0.12
    )
    cbar_gs = gridspec.GridSpecFromSubplotSpec(
        1, n_groups, subplot_spec=outer[1], wspace=1.4
    )

    # Resolve one (cmap, norm) per style group from all its panels' data.
    group_norm = {}
    for gid, cols in groups.items():
        st = col_style[cols[0]]
        pooled = [fields.get((r, c)) for r in rows for c in cols]
        group_norm[gid] = resolve_norm(st, pooled)

    for ri, row in enumerate(rows):
        for ci, col in enumerate(columns):
            ax = geo_axes(fig, grid[ri, ci], lat, lon)
            field = fields.get((row, col))
            if field is not None:
                cmap, norm, _ = group_norm[id(col_style[col])]
                x, y = _coords(field, lat, lon)
                draw_field(ax, field, x, y, cmap, norm, lat, lon)
                if annotate:
                    ax.set_xlabel(_annotation(field), fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
            if ri == 0:
                ax.set_title(col, fontsize=10)
            if ci == 0 and row:
                ax.text(-0.08, 0.5, row, transform=ax.transAxes,
                        va="center", ha="right", fontsize=8, fontweight="bold",
                        linespacing=1.25, clip_on=False)

    # One colorbar per style group.
    for gi, (gid, cols) in enumerate(groups.items()):
        st = col_style[cols[0]]
        cmap, norm, is_discrete = group_norm[gid]
        cax = fig.add_subplot(cbar_gs[0, gi])
        sm = ScalarMappable(norm=norm, cmap=cmap)
        fig.colorbar(
            sm, cax=cax, label=st.unit,
            extend="both" if is_discrete else "neither",
        )

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")
        fig.subplots_adjust(top=0.92)
    if has_row_labels:
        fig.subplots_adjust(left=0.18)
    return fig
