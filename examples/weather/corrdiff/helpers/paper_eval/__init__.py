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

"""Reusable plotting library for the paper-protocol evaluation flow.

Layers:
    styles   — FieldStyle dataclass + STYLES registry (colors/units/levels)
    maps     — plot_map_grid primitive (Figs 2, 3, 4, 7, 8) + draw primitives
    charts   — Q-Q (Fig 5), RAPSD (Fig 6), SAL (Fig 9)
    figures  — thin semantic wrappers the driver calls
"""

from .styles import FieldStyle, STYLES, get_style, resolve_norm
from .maps import plot_map_grid, draw_field, geo_axes
from .charts import (
    plot_qq_panel,
    plot_qq_triptych,
    plot_rapsd,
    plot_sal_diagram,
    plot_sal_epoch_scatter,
    plot_sal_grid,
    draw_sal,
)
from .figures import (
    relbias_map,
    rmse_map_figure,
    bias_map_figure,
    crps_map_figure,
    out_of_envelope_figure,
    sal_epoch_figure,
    sal_figure,
    qq_figure,
    rapsd_figure,
)
from .io import (
    PredictionFile,
    TargetSpec,
    EvalOutputs,
    NetCDFStream,
    normalize_timestamp,
    format_timestamp,
    resolve_channel,
    prediction_file_entries,
    target_entries,
    stream_target_index,
    eval_outputs,
)
from .results import ArtifactManifest, ensure_output_dirs, write_json, write_csv, save_figure
from .run import ProductSelection, product_selection, preflight_lines

__all__ = [
    "FieldStyle", "STYLES", "get_style", "resolve_norm",
    "plot_map_grid", "draw_field", "geo_axes",
    "plot_qq_panel", "plot_qq_triptych", "plot_rapsd", "plot_sal_diagram",
    "plot_sal_epoch_scatter", "plot_sal_grid", "draw_sal",
    "relbias_map", "rmse_map_figure", "bias_map_figure", "crps_map_figure",
    "out_of_envelope_figure", "sal_epoch_figure", "sal_figure", "qq_figure",
    "rapsd_figure",
    "PredictionFile", "TargetSpec", "EvalOutputs", "NetCDFStream",
    "normalize_timestamp", "format_timestamp", "resolve_channel",
    "prediction_file_entries", "target_entries", "stream_target_index",
    "eval_outputs", "ArtifactManifest", "ensure_output_dirs", "write_json",
    "write_csv", "save_figure",
    "ProductSelection", "product_selection", "preflight_lines",
]
