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

"""Semantic figure wrappers — thin assemblers the driver calls.

Each wrapper builds the ``fields`` dict (or chart inputs), picks styles from the
registry, and delegates to ``maps.plot_map_grid`` / ``charts.*``.  This keeps
``evaluate_paper.py`` readable and confines all paper-figure layout choices here.
"""

from __future__ import annotations

import numpy as np

from .charts import plot_qq_triptych, plot_rapsd, plot_sal_epoch_scatter, plot_sal_grid
from .maps import plot_map_grid
from .styles import get_style


def relbias_map(model: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Per-gridpoint relative bias (%) with a divide-by-zero guard."""
    model = np.asarray(model, dtype=float)
    ref = np.asarray(ref, dtype=float)
    out = np.full(ref.shape, np.nan, dtype=float)
    m = np.isfinite(model) & np.isfinite(ref) & (ref != 0)
    out[m] = 100.0 * (model[m] - ref[m]) / ref[m]
    return out


# --- Climatological figures -------------------------------------------------

def rmse_map_figure(rmse_map, lat=None, lon=None, model="CorrDiff", title="RMSE"):
    """Fig 2: single per-gridpoint RMSE map (one model column)."""
    return plot_map_grid(
        fields={("", model): rmse_map},
        rows=[""], columns=[model], style=get_style("rmse"),
        lat=lat, lon=lon, title=title,
    )


def bias_map_figure(stat_label, window_maps: dict, lat=None, lon=None,
                    model="CorrDiff", title: str | None = None):
    """Figs 3/4: a statistic's reference field + model relative-bias, gridpoint & 3x3.

    Args:
        stat_label:  e.g. "Mean", "SD", "Median", "P99", "Dry%".
        window_maps: ``{row_label: {"model": 2-D, "ref": 2-D}}``, e.g.
                     ``{"Gridpoint": {...}, "3x3": {...}}``.
    """
    rows = list(window_maps.keys())
    columns = ["Reference", model]
    fields = {}
    for row, m in window_maps.items():
        fields[(row, "Reference")] = m["ref"]
        fields[(row, model)] = relbias_map(m["model"], m["ref"])
    column_styles = {"Reference": get_style("precip"), model: get_style("relbias")}
    return plot_map_grid(fields, rows, columns, column_styles=column_styles,
                         lat=lat, lon=lon,
                         title=title or f"{stat_label} — relative bias")


# --- Targeted figures (target rows × [Reference, model] columns) ------------

def _target_map_grid(targets: list[dict], metric_key: str, metric_style: str,
                     title: str, lat=None, lon=None, model="CorrDiff"):
    rows = [t.get("display_label", t["label"]) for t in targets]
    columns = ["Reference", model]
    fields = {}
    for t in targets:
        row_label = t.get("display_label", t["label"])
        fields[(row_label, "Reference")] = t["reference"]
        fields[(row_label, model)] = t[metric_key]
    column_styles = {"Reference": get_style("precip"), model: get_style(metric_style)}
    return plot_map_grid(fields, rows, columns, column_styles=column_styles,
                         lat=lat, lon=lon, title=title,
                         panel_size=3.5, panel_height=1.55)


def crps_map_figure(targets: list[dict], lat=None, lon=None, model="CorrDiff",
                    title: str = "CRPS (3x3)"):
    """Fig 8: per-target CRPS maps (3x3-smoothed) beside the reference field."""
    return _target_map_grid(targets, "crps", "crps", title, lat, lon, model)


def out_of_envelope_figure(targets: list[dict], lat=None, lon=None,
                           model="CorrDiff",
                           title: str = "Out-of-envelope (3x3)"):
    """Fig 7: per-target out-of-envelope maps (3x3) beside the reference field."""
    return _target_map_grid(targets, "ooe", "ooe", title, lat, lon, model)


def sal_figure(targets: list[dict], model="CorrDiff", title: str | None = None):
    """Fig 9: per-target SAL diagrams (one model column)."""
    return plot_sal_grid(targets, col_label=model, title=title)


def sal_epoch_figure(targets: list[dict], model="CorrDiff", title: str | None = None):
    """Epoch-grouped SAL scatter; x=S, y=A, color=L, marker=kind."""
    return plot_sal_epoch_scatter(targets, col_label=model, title=title)


# --- Distribution charts ----------------------------------------------------

def qq_figure(panels: list[dict], title: str | None = None):
    """Fig 5: Q-Q triptych (wet %, spatial-mean rainfall, tail)."""
    return plot_qq_triptych(panels, title=title)


def rapsd_figure(curves: list[dict], dx_km: float = 2.0, title: str | None = None):
    """Fig 6: RAPSD vs wavelength."""
    return plot_rapsd(curves, dx_km=dx_km, title=title)
