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

"""Non-map charts for the paper-protocol flow: Q-Q (Fig 5), RAPSD (Fig 6), SAL (Fig 9)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D


def plot_qq_panel(ax, curves: list[dict], xlabel: str, ylabel: str, title: str) -> None:
    """Draw one Q-Q panel with a 1:1 reference line.

    Args:
        ax:     target axis.
        curves: list of ``{"label": str, "ref": array, "sim": array}`` (quantiles).
        xlabel, ylabel, title: axis labels.
    """
    lo, hi = np.inf, -np.inf
    for c in curves:
        ref = np.asarray(c["ref"], dtype=float)
        sim = np.asarray(c["sim"], dtype=float)
        ax.plot(ref, sim, lw=1.6, label=c["label"])
        if ref.size:
            lo = min(lo, np.nanmin(ref), np.nanmin(sim))
            hi = max(hi, np.nanmax(ref), np.nanmax(sim))
    if not np.isfinite(lo):
        lo, hi = 0.0, 1.0
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, label="1:1")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


def plot_qq_triptych(panels: list[dict], title: str | None = None) -> plt.Figure:
    """Render the paper's three-panel Q-Q figure (Fig 5).

    Args:
        panels: list of up to 3 dicts, each
            ``{"curves": [...], "xlabel": str, "ylabel": str, "title": str}``.

    Returns:
        matplotlib Figure.
    """
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5.5))
    if n == 1:
        axes = [axes]
    for ax, p in zip(axes, panels):
        plot_qq_panel(ax, p["curves"], p.get("xlabel", "Reference"),
                      p.get("ylabel", "Simulated"), p.get("title", ""))
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")
        fig.subplots_adjust(top=0.84)
    fig.tight_layout()
    return fig


def plot_rapsd(curves: list[dict], dx_km: float = 2.0,
               title: str | None = None) -> plt.Figure:
    """Radially averaged power spectral density vs **wavelength (km)** (Fig 6).

    Args:
        curves: list of ``{"label": str, "freq": array (cycles/km), "psd": array,
                "style": optional matplotlib fmt}``.
        dx_km:  grid spacing (title only).

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    for c in curves:
        freq = np.asarray(c["freq"], dtype=float)
        psd = np.asarray(c["psd"], dtype=float)
        valid = (freq > 0) & (psd > 0)
        wavelength = 1.0 / freq[valid]  # km
        order = np.argsort(wavelength)
        ax.loglog(wavelength[order], psd[valid][order], c.get("style", "-"),
                  lw=2, label=c["label"])
    ax.invert_xaxis()  # large scales (long wavelength) on the left, like the paper
    ax.set_xlabel("Wavelength (km)", fontsize=11)
    ax.set_ylabel("RAPSD", fontsize=11)
    ax.set_title(
        title or f"Radially Averaged Power Spectral Density (dx={dx_km:g} km)",
        fontsize=12,
    )
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def draw_sal(ax, S, A, L, reference: dict | None = None):
    """Draw one SAL scatter into ``ax``; returns the scatter mappable (or None)."""
    S = np.asarray(S, dtype=float)
    A = np.asarray(A, dtype=float)
    L = np.asarray(L, dtype=float)
    valid = np.isfinite(S) & np.isfinite(A)

    sc = None
    if valid.any():
        sc = ax.scatter(
            S[valid], A[valid],
            c=np.clip(np.where(np.isfinite(L[valid]), L[valid], 0.0), 0, 2),
            cmap="Reds", vmin=0.0, vmax=2.0, s=72, alpha=0.85,
            edgecolor="black", linewidth=0.35, zorder=2,
        )
    if reference is not None and np.isfinite(reference.get("S", np.nan)):
        ax.scatter([reference["S"]], [reference["A"]], marker="*", s=220,
                   facecolor="white", edgecolor="black", linewidth=1.2, zorder=4)
    ax.axvline(0.0, color="black", lw=1.0)
    ax.axhline(0.0, color="black", lw=1.0)
    ax.set_xlim(-2.05, 2.05)
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, alpha=0.3)
    return sc


def _sal_l_norm(rows: list[dict]) -> Normalize:
    vals = []
    for r in rows:
        arr = np.asarray(r["L"], dtype=float)
        vals.append(arr[np.isfinite(arr)])
    finite = np.concatenate([v for v in vals if v.size]) if any(v.size for v in vals) else np.array([])
    if finite.size == 0:
        return Normalize(vmin=0.0, vmax=1.0)
    vmax = float(np.nanpercentile(finite, 95))
    vmax = max(vmax, float(np.nanmax(finite)), 0.05)
    return Normalize(vmin=0.0, vmax=vmax)


def _sal_marker(kind: str) -> str:
    return "^" if str(kind).lower() == "extreme" else "o"


def _sal_case_label(row: dict) -> str:
    if row.get("case_id"):
        return str(row["case_id"])
    label = row.get("display_label", row.get("label", "target"))
    parts = [p.strip() for p in str(label).splitlines() if p.strip()]
    if len(parts) >= 2:
        return f"{parts[0]} {parts[1].split()[0]}"
    return parts[0] if parts else "target"


def plot_sal_diagram(
    S: np.ndarray,
    A: np.ndarray,
    L: np.ndarray,
    title: str = "SAL diagram",
    reference: dict | None = None,
) -> plt.Figure:
    """Single-panel SAL scatter (x=Structure, y=Amplitude, color=Location)."""
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = draw_sal(ax, S, A, L, reference=reference)
    if sc is not None:
        fig.colorbar(sc, ax=ax, label="Location (L)")
    ax.set_xlabel("Structure (S)", fontsize=11)
    ax.set_ylabel("Amplitude (A)", fontsize=11)
    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    return fig


def plot_sal_epoch_scatter(
    rows: list[dict],
    col_label: str = "CorrDiff",
    title: str | None = None,
) -> plt.Figure:
    """SAL scatter grouped by epoch, with Location (L) encoded as color."""
    epochs = ["current", "mid", "end"]
    present = {str(r.get("epoch", "")).lower() for r in rows}
    epoch_order = [e for e in epochs if e in present] + sorted(present - set(epochs))
    if not epoch_order:
        epoch_order = ["targets"]

    norm = _sal_l_norm(rows)
    cmap = plt.get_cmap("Reds")
    fig, axes = plt.subplots(
        1, len(epoch_order), figsize=(5.4 * len(epoch_order), 5.2),
        squeeze=False, sharex=True, sharey=True,
    )
    for ax, epoch in zip(axes[0], epoch_order):
        epoch_rows = [
            r for r in rows
            if str(r.get("epoch", "targets")).lower() == epoch
            or (epoch == "targets" and "epoch" not in r)
        ]
        for r in epoch_rows:
            S = np.asarray(r["S"], dtype=float)
            A = np.asarray(r["A"], dtype=float)
            L = np.asarray(r["L"], dtype=float)
            valid = np.isfinite(S) & np.isfinite(A)
            if not valid.any():
                continue
            colors = np.where(np.isfinite(L[valid]), L[valid], 0.0)
            ax.scatter(
                S[valid], A[valid],
                c=colors, cmap=cmap, norm=norm, marker=_sal_marker(r.get("kind", "")),
                s=90, alpha=0.82, edgecolor="black", linewidth=0.4,
                label=str(r.get("kind", "target")),
            )
            med_s = float(np.nanmedian(S[valid]))
            med_a = float(np.nanmedian(A[valid]))
            ax.annotate(
                _sal_case_label(r), xy=(med_s, med_a), xytext=(4, 4),
                textcoords="offset points", fontsize=7, alpha=0.85,
            )
        ax.axvline(0.0, color="black", lw=1.0)
        ax.axhline(0.0, color="black", lw=1.0)
        ax.set_xlim(-2.05, 2.05)
        ax.set_ylim(-1.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.set_title(epoch.upper(), fontsize=11, fontweight="bold")
        ax.set_xlabel("Structure (S)", fontsize=10)
    axes[0, 0].set_ylabel("Amplitude (A)", fontsize=10)

    handles = [
        Line2D([0], [0], marker="^", color="w", label="extreme",
               markerfacecolor="0.75", markeredgecolor="black", markersize=8),
        Line2D([0], [0], marker="o", color="w", label="normal",
               markerfacecolor="0.75", markeredgecolor="black", markersize=8),
    ]
    axes[0, -1].legend(handles=handles, loc="lower right", title="Case kind", fontsize=8)

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.02, 0.90, 0.88 if title else 0.96))
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cax = fig.add_axes([0.925, 0.22, 0.018, 0.58])
    vmax = norm.vmax if np.isfinite(norm.vmax) else 1.0
    fig.colorbar(sm, cax=cax, label=f"Location (L), scaled 0-{vmax:.2f}")
    fig.text(0.5, 0.02, f"Model: {col_label}", ha="center", fontsize=10)
    return fig


def plot_sal_grid(
    rows: list[dict],
    col_label: str = "CorrDiff",
    title: str | None = None,
) -> plt.Figure:
    """SAL diagrams by target (paper Fig 9 layout, one model column).

    Args:
        rows:      list of ``{"label": str, "S":, "A":, "L":, "reference": dict|None}``.
        col_label: column header (model name).

    Returns:
        matplotlib Figure with one SAL panel per target and a shared Location colorbar.
    """
    n = len(rows)
    ncols = min(3, max(1, n))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.0 * ncols, 4.15 * nrows), squeeze=False
    )
    last_sc = None
    for i, r in enumerate(rows):
        ax = axes[i // ncols, i % ncols]
        sc = draw_sal(ax, r["S"], r["A"], r["L"], reference=r.get("reference"))
        last_sc = sc or last_sc
        S = np.asarray(r["S"], dtype=float)
        A = np.asarray(r["A"], dtype=float)
        L = np.asarray(r["L"], dtype=float)
        valid = np.isfinite(S) & np.isfinite(A)
        label = r.get("display_label", r["label"])
        ax.set_title(label, fontsize=10, fontweight="bold", loc="left")
        ax.set_xlabel("Structure (S)", fontsize=9)
        ax.set_ylabel("Amplitude (A)", fontsize=9)
        if valid.any():
            stats = (
                f"median S={np.nanmedian(S[valid]):.2f}, "
                f"A={np.nanmedian(A[valid]):.2f}, "
                f"L={np.nanmedian(L[np.isfinite(L)]):.2f}"
            )
        else:
            stats = "no finite SAL scores"
        ax.text(
            0.02, 0.98, stats, transform=ax.transAxes,
            ha="left", va="top", fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "0.85", "alpha": 0.9, "pad": 2.5},
        )
    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 0.88, 0.92 if title else 0.97))
    if last_sc is not None:
        cax = fig.add_axes([0.91, 0.22, 0.018, 0.56])
        fig.colorbar(last_sc, cax=cax, label="Location (L)")
    fig.text(0.5, 0.01, f"Model: {col_label}", ha="center", fontsize=10)
    return fig
