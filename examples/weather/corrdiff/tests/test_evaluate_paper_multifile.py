import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OmegaConf = pytest.importorskip("omegaconf").OmegaConf
nc = pytest.importorskip("netCDF4")
ep = pytest.importorskip("evaluate_paper")


class DummyLogger:
    def info(self, *_args, **_kwargs):
        pass


class DummyTable:
    def __init__(self, columns):
        self.columns = columns
        self.rows = []

    def add_data(self, *values):
        self.rows.append(values)


def _write_prediction_nc(path, start_hour, values):
    with nc.Dataset(path, "w") as f:
        f.createDimension("time", len(values))
        f.createDimension("ensemble", 1)
        f.createDimension("y", 4)
        f.createDimension("x", 4)

        lat = f.createVariable("lat", "f4", ("y", "x"))
        lon = f.createVariable("lon", "f4", ("y", "x"))
        lat[:] = np.zeros((4, 4), dtype=np.float32)
        lon[:] = np.zeros((4, 4), dtype=np.float32)

        time = f.createVariable("time", "f8", ("time",))
        time.units = "hours since 2000-01-01 00:00:00"
        time.calendar = "standard"
        time[:] = np.arange(start_hour, start_hour + len(values))

        truth = f.createGroup("truth")
        pred = f.createGroup("prediction")
        tvar = truth.createVariable("precipitation", "f4", ("time", "y", "x"))
        pvar = pred.createVariable(
            "precipitation", "f4", ("ensemble", "time", "y", "x")
        )
        for i, value in enumerate(values):
            field = np.full((4, 4), value, dtype=np.float32)
            tvar[i] = field
            pvar[0, i] = field + 0.5


def _patch_wandb_and_figures(monkeypatch):
    monkeypatch.setattr(ep.wandb, "Table", DummyTable)
    monkeypatch.setattr(ep.wandb, "Image", lambda fig: fig)
    for name in [
        "rmse_map_figure",
        "bias_map_figure",
        "qq_figure",
        "rapsd_figure",
        "crps_map_figure",
        "out_of_envelope_figure",
        "sal_epoch_figure",
        "sal_figure",
    ]:
        monkeypatch.setattr(ep, name, lambda *args, **kwargs: plt.figure())


def _cfg(tmp_path, files, targets=None):
    return OmegaConf.create({
        "eval": {
            "prediction_files": files,
            "targets": targets,
            "precip_channel": "auto",
            "dry_threshold": 1.0,
            "wet_threshold": 1.0,
            "hist_max_mm": 20.0,
            "hist_bins": 20,
            "compute_smoothed": True,
            "compute_quantile_maps": False,
            "map_bins": 10,
            "smooth_size": 3,
            "sal_f_factor": 1.0 / 15.0,
            "rapsd_dx_km": 2.0,
            "output_json": str(tmp_path / "paper_eval_results.json"),
        }
    })


def test_stream_target_index_resolves_time_idx_and_timestamp(tmp_path):
    path = tmp_path / "predictions.nc"
    _write_prediction_nc(path, start_hour=0, values=[1.0, 2.0, 3.0])

    stream = ep._NetCDFStream(str(path), "auto")
    try:
        assert ep._stream_target_index(stream, {"label": "a", "time_idx": 1}) == 1
        assert ep._stream_target_index(
            stream,
            {
                "label": "a",
                "time_idx": 2,
                "timestamp": "2000-01-01 02:00:00",
            },
        ) == 2
        assert ep._stream_target_index(
            stream, {"label": "a", "timestamp": "2000-01-01 01:00:00"}
        ) == 1
        with pytest.raises(ValueError, match="timestamp mismatch"):
            ep._stream_target_index(
                stream,
                {
                    "label": "a",
                    "time_idx": 1,
                    "timestamp": "2000-01-01 02:00:00",
                },
            )
    finally:
        stream.close()


def test_prediction_file_entries_and_targets_parse(tmp_path):
    path = tmp_path / "predictions.nc"
    _write_prediction_nc(path, start_hour=0, values=[1.0])
    cfg = _cfg(
        tmp_path,
        [{"epoch": "current", "label": "current_2005", "path": str(path)}],
        [{
            "epoch": "current",
            "kind": "normal",
            "label": "current_2005",
            "timestamp": "2000-01-01 00:00",
            "time_idx": 0,
        }],
    )

    assert ep._prediction_file_entries(cfg) == [{
        "epoch": "current",
        "label": "current_2005",
        "path": str(path),
    }]
    assert ep._target_entries(cfg)[0]["timestamp"] == "2000-01-01 00:00:00"


def test_inputs_and_target_set_parse(tmp_path):
    path = tmp_path / "predictions.nc"
    target_set = tmp_path / "targets.yaml"
    _write_prediction_nc(path, start_hour=0, values=[1.0])
    target_set.write_text(
        """
targets:
  - epoch: current
    kind: normal
    label: current_2005
    timestamp: "2000-01-01 00:00"
    time_idx: 0
""",
        encoding="utf-8",
    )
    cfg = OmegaConf.create({
        "eval": {
            "inputs": [{
                "model": "sym3h",
                "epoch": "current",
                "label": "current_2005",
                "path": str(path),
            }],
            "target_set": str(target_set),
        }
    })

    entries = ep._prediction_file_entries(cfg)
    targets = ep._target_entries(cfg)

    assert entries[0]["model"] == "sym3h"
    assert entries[0]["label"] == "current_2005"
    assert targets[0]["timestamp"] == "2000-01-01 00:00:00"


def test_multifile_climatology_writes_pooled_and_epoch_summaries(tmp_path, monkeypatch):
    _patch_wandb_and_figures(monkeypatch)
    current = tmp_path / "current.nc"
    future = tmp_path / "future.nc"
    _write_prediction_nc(current, start_hour=0, values=[1.0, 2.0])
    _write_prediction_nc(future, start_hour=2, values=[3.0, 4.0])
    cfg = _cfg(
        tmp_path,
        [
            {"epoch": "current", "label": "current_2005", "path": str(current)},
            {"epoch": "end", "label": "end_2100", "path": str(future)},
        ],
    )

    json_out = {}
    payload = ep._run_climatology_multifile(
        cfg, DummyLogger(), ep._prediction_file_entries(cfg), json_out
    )

    assert "climatology/all_periods/bias_table" in payload
    assert "climatology/current/bias_table" in payload
    assert "climatology/end/bias_table" in payload
    assert json_out["climatology"]["all_periods"]["n_times"] == 4
    assert json_out["climatology"]["current"]["n_times"] == 2
    assert json_out["climatology"]["end"]["n_times"] == 2


def test_output_layout_writes_manifest_tables_figures_and_data(tmp_path, monkeypatch):
    _patch_wandb_and_figures(monkeypatch)
    path = tmp_path / "predictions.nc"
    _write_prediction_nc(path, start_hour=0, values=[1.0, 5.0])
    cfg = _cfg(
        tmp_path,
        [{"epoch": "current", "label": "current_2005", "path": str(path)}],
        [{
            "epoch": "current",
            "kind": "extreme",
            "label": "current_2005",
            "timestamp": "2000-01-01 01:00:00",
            "time_idx": 1,
        }],
    )
    cfg.eval.outputs = {
        "root": str(tmp_path / "paper_out"),
        "json": "paper_eval_results.json",
        "tables": "tables",
        "figures": "figures",
        "data": "data",
        "manifest": "manifest.json",
    }
    ep._RUN_OUTPUTS = ep.paper_io.eval_outputs(cfg, "test")
    ep.paper_results.ensure_output_dirs(ep._RUN_OUTPUTS)
    ep._RUN_MANIFEST = ep.paper_results.ArtifactManifest(ep._RUN_OUTPUTS.root)
    json_out = {}

    ep._run_climatology_multifile(
        cfg, DummyLogger(), ep._prediction_file_entries(cfg), json_out
    )
    ep._run_targets_from_entries(
        cfg, DummyLogger(), ep._prediction_file_entries(cfg), ep._target_entries(cfg), json_out
    )
    ep._RUN_MANIFEST.write(ep._RUN_OUTPUTS.manifest_path)

    assert (tmp_path / "paper_out/tables/climatology_all_periods_bias_table.csv").exists()
    assert (tmp_path / "paper_out/figures/climatology/all_periods/rmse_map.png").exists()
    assert (tmp_path / "paper_out/data/targets_fields.npz").exists()
    assert (tmp_path / "paper_out/manifest.json").exists()

    ep._RUN_OUTPUTS = None
    ep._RUN_MANIFEST = None


def test_targets_from_entries_preserves_kind_and_epoch(tmp_path, monkeypatch):
    _patch_wandb_and_figures(monkeypatch)
    path = tmp_path / "predictions.nc"
    _write_prediction_nc(path, start_hour=0, values=[1.0, 5.0])
    cfg = _cfg(
        tmp_path,
        [{"epoch": "current", "label": "current_2005", "path": str(path)}],
        [{
            "epoch": "current",
            "kind": "extreme",
            "label": "current_2005",
            "timestamp": "2000-01-01 01:00:00",
            "time_idx": 1,
        }],
    )

    json_out = {}
    payload = ep._run_targets_from_entries(
        cfg, DummyLogger(), ep._prediction_file_entries(cfg), ep._target_entries(cfg), json_out
    )

    assert "targets/current/crps_maps" in payload
    assert "targets/current/out_of_envelope" in payload
    assert "targets/combined/crps_maps" in payload
    assert "targets/sal_scatter_by_epoch" in payload
    assert "targets/sal_case_grid" in payload
    assert json_out["targets"]["n_targets"] == 1
    target = json_out["targets"]["per_target"][0]
    assert target["epoch"] == "current"
    assert target["kind"] == "extreme"
    assert target["timestamp"] == "2000-01-01 01:00:00"


def test_sal_epoch_scatter_encodes_location_and_kind():
    from helpers.paper_eval.charts import plot_sal_epoch_scatter

    fig = plot_sal_epoch_scatter([
        {
            "label": "current extreme",
            "display_label": "CUR extreme\n2000-01-01 00Z\nmax 5 mm",
            "epoch": "current",
            "kind": "extreme",
            "S": np.array([0.2, 0.3]),
            "A": np.array([0.1, 0.2]),
            "L": np.array([0.01, 0.08]),
        },
        {
            "label": "current normal",
            "display_label": "CUR normal\n2000-01-01 01Z\nmax 1 mm",
            "epoch": "current",
            "kind": "normal",
            "S": np.array([0.0, 0.1]),
            "A": np.array([-0.1, 0.0]),
            "L": np.array([0.02, 0.04]),
        },
    ])

    try:
        axes = fig.axes
        assert len(axes) >= 2  # plot axis + colorbar axis
        assert axes[-1].get_ylabel().startswith("Location (L), scaled")
        markers = {coll.get_paths()[0].vertices.shape[0] for coll in axes[0].collections}
        assert len(markers) >= 2
    finally:
        plt.close(fig)


def test_pysteps_style_sal_perfect_forecast_is_zero():
    from helpers.targets import sal_scores

    field = np.zeros((12, 12), dtype=float)
    field[3:6, 4:8] = 10.0

    sal = sal_scores(field, field)

    assert sal["S"] == pytest.approx(0.0)
    assert sal["A"] == pytest.approx(0.0)
    assert sal["L"] == pytest.approx(0.0)


def test_pysteps_style_sal_amplitude_matches_domain_mean_formula():
    from helpers.targets import sal_distribution

    ref = np.ones((8, 8), dtype=float)
    pred = np.stack([ref * 2.0])

    sal = sal_distribution(pred, ref)

    assert sal["A"][0] == pytest.approx((2.0 - 1.0) / (0.5 * (2.0 + 1.0)))
    assert sal["implementation"] in {"pysteps", "vendored_pysteps"}
    assert sal["thr_quantile"] == pytest.approx(0.95)
