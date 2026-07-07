import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

cmp = pytest.importorskip("compare_paper_models")


def _result_payload():
    table = []
    for window in ["raw", "3x3"]:
        for stat in ["dry_pct", "mean", "sd", "median", "p99"]:
            table.append({
                "window": window,
                "statistic": stat,
                "model": 1.0,
                "reference": 2.0,
                "rel_bias_pct": -10.0,
            })
    return {
        "run_tag": "test",
        "climatology": {
            group: {
                "channel": "precipitation",
                "n_times": 2,
                "n_ensemble": 1,
                "files": [],
                "table": list(table),
            }
            for group in ["all_periods", "current", "mid", "end"]
        },
        "targets": {
            "channel": "precipitation",
            "n_targets": 2,
            "n_ensemble": 1,
            "per_target": [
                {
                    "epoch": "current",
                    "kind": "extreme",
                    "timestamp": "2000-01-01 00:00:00",
                    "max_precip_mm": 10.0,
                    "crps_mean": 0.2,
                    "S": 0.1,
                    "A": -0.1,
                    "L": {"mean": 0.02, "median": 0.02},
                },
                {
                    "epoch": "current",
                    "kind": "normal",
                    "timestamp": "2000-01-02 00:00:00",
                    "max_precip_mm": 2.0,
                    "crps_mean": 0.1,
                    "S": 0.3,
                    "A": 0.1,
                    "L": 0.04,
                },
            ],
        },
    }


def _write_result(root: Path):
    root.mkdir(parents=True)
    (root / "paper_eval_results.json").write_text(
        json.dumps(_result_payload()), encoding="utf-8"
    )


def _write_png(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.imshow(np.ones((3, 3)))
    ax.axis("off")
    fig.savefig(path, dpi=20)
    plt.close(fig)


def test_find_semantic_png_supports_nested_and_flat(tmp_path):
    nested_root = tmp_path / "nested"
    flat_root = tmp_path / "flat"
    _write_result(nested_root)
    _write_result(flat_root)
    _write_png(nested_root / "images" / "climatology/all_periods/bias_mean_0_hash.png")
    _write_png(flat_root / "images" / "climatology_all_periods_bias_sd.png")

    nested = cmp.ModelRun("a", "A", "#000000", nested_root, nested_root / "images")
    flat = cmp.ModelRun("b", "B", "#000000", flat_root, flat_root / "images")

    assert cmp.find_semantic_png(nested, "climatology/all_periods/bias_mean")
    assert cmp.find_semantic_png(flat, "climatology/all_periods/bias_sd")


def test_run_comparison_writes_tables_and_figures(tmp_path):
    roots = []
    for model in ["t0", "past12h"]:
        root = tmp_path / model
        _write_result(root)
        _write_png(root / "images" / "climatology/all_periods/bias_mean_0_hash.png")
        roots.append(root)

    models = [
        cmp.ModelRun("t0", "t0", "#666666", roots[0], roots[0] / "images"),
        cmp.ModelRun("past12h", "past12h", "#2878b5", roots[1], roots[1] / "images"),
    ]
    out = tmp_path / "out"
    result = cmp.run_comparison(models, out)

    assert result["summary"].exists()
    assert (out / "tables/climatology_bias_comparison.csv").exists()
    assert (out / "tables/target_crps_comparison.csv").exists()
    assert (out / "figures/main/targets_crps_summary_lines_models.png").exists()
    assert not (out / "figures/main/targets_crps_summary_models.png").exists()
    assert (
        out / "figures/appendix/contact_sheets/climatology_all_periods_bias_mean_models.png"
    ).exists()


def test_config_loader_and_structured_outputs(tmp_path):
    roots = []
    for model, color in [("t0", "#666666"), ("past12h", "#2878b5")]:
        root = tmp_path / model
        payload = _result_payload()
        payload["climatology"]["all_periods"]["rapsd"] = {
            "freq": [0.1, 0.2, 0.4],
            "reference_psd": [4.0, 2.0, 1.0],
            "model_psd": [3.0, 1.5, 0.8],
            "dx_km": 2.0,
        }
        payload["climatology"]["all_periods"]["qq"] = [{
            "title": "Wet",
            "xlabel": "Reference",
            "ylabel": "Model",
            "curves": [{"label": "CorrDiff", "ref": [0, 1], "sim": [0, 0.9]}],
        }]
        root.mkdir(parents=True)
        (root / "paper_eval_results.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        data_dir = root / "data"
        data_dir.mkdir()
        np.savez_compressed(
            data_dir / "targets_fields.npz",
            reference=np.ones((2, 4, 4)),
            crps=np.ones((2, 4, 4)) * 0.2,
            out_of_envelope=np.ones((2, 4, 4)) * 0.1,
            case_id=np.array(["A", "B"]),
            epoch=np.array(["current", "current"]),
            kind=np.array(["extreme", "normal"]),
            label=np.array(["a", "b"]),
            display_label=np.array(["A", "B"]),
        )
        roots.append((model, color, root))

    cfg_path = tmp_path / "compare.yaml"
    cfg_path.write_text(
        f"""
compare:
  output_dir: {tmp_path / "out"}
  models:
    - id: t0
      label: t0
      color: "#666666"
      result_dir: {roots[0][2]}
    - id: past12h
      label: past12h
      color: "#2878b5"
      result_dir: {roots[1][2]}
""",
        encoding="utf-8",
    )

    models, output_dir = cmp.load_compare_config(str(cfg_path))
    result = cmp.run_comparison(models, output_dir)

    assert result["summary"].exists()
    assert (
        output_dir
        / "figures/main/climatology_all_periods_rapsd_structured_models.png"
    ).exists()
    assert (
        output_dir
        / "figures/main/climatology_all_periods_qq_structured_models.png"
    ).exists()
    assert (
        output_dir
        / "figures/main/targets_combined_crps_maps_structured_models.png"
    ).exists()
