# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small orchestration helpers for the paper-protocol evaluation driver."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .io import as_plain_container


@dataclass(frozen=True)
class ProductSelection:
    climatology: bool = True
    targets: bool = True
    compare_models: bool = False


def product_selection(cfg) -> ProductSelection:
    products = as_plain_container(cfg.eval.get("products", None)) or {}
    return ProductSelection(
        climatology=bool(products.get("climatology", True)),
        targets=bool(products.get("targets", True)),
        compare_models=bool(products.get("compare_models", False)),
    )


def preflight_lines(
    run_tag: str,
    output_root: Path,
    products: ProductSelection,
    prediction_files: list[dict],
    selected_targets: list[dict],
) -> list[str]:
    lines = [
        "=" * 70,
        "PAPER EVAL PREFLIGHT",
        f"  run_tag: {run_tag}",
        f"  output_root: {output_root}",
        (
            "  products: "
            f"climatology={products.climatology} "
            f"targets={products.targets} "
            f"compare_models={products.compare_models}"
        ),
    ]
    if prediction_files:
        lines.append(f"  prediction inputs: {len(prediction_files)}")
        for entry in prediction_files:
            lines.append(
                f"    - epoch={entry['epoch']} label={entry['label']} "
                f"model={entry.get('model', '')} path={entry['path']}"
            )
    if selected_targets:
        lines.append(f"  selected targets: {len(selected_targets)}")
        for target in selected_targets:
            lines.append(
                f"    - {target.get('epoch')} {target.get('kind')} "
                f"label={target.get('label')} idx={target.get('time_idx')} "
                f"timestamp={target.get('timestamp')}"
            )
    return lines
