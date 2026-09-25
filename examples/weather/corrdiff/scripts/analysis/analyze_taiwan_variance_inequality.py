#!/usr/bin/env python3
"""Map the pixelwise Taiwan variance inequality for a regression evaluation.

Run this on Helma, not on the login node and not on the Mac: the input NetCDF
already lives on Helma's HNVME filesystem.  From
``~/corrdiffProjektSimon/code/examples/weather/corrdiff`` on Helma, submit:

    sbatch \\
      --job-name=var-radar \\
      --partition=preempt_cpu \\
      --nodes=1 \\
      --ntasks=1 \\
      --cpus-per-task=5 \\
      --time=00:15:00 \\
      --output=/hnvme/workspace/b214cb11-helma-ecodata/downscaling/daniel/outputs/ablation_1_and_2_models/slurm/variance-radar-%j.out \\
      --wrap="srun --ntasks=1 --cpus-per-task=5 apptainer exec \\
        --bind $HOME/corrdiffProjektSimon/code/examples/weather/corrdiff:/workspace/corrdiff \\
        --bind /hnvme/workspace/b214cb11-helma-ecodata/downscaling/daniel/outputs/ablation_1_and_2_models:/outputs \\
        /hnvme/workspace/b214cb11-helma-ecodata/downscaling/apptainer/corrdiff_ngc.sif \\
        python /workspace/corrdiff/scripts/analysis/analyze_taiwan_variance_inequality.py \\
          --predictions /outputs/eval/taiwan_ablation_radar_regression_2m/predictions.nc \\
          --output-dir /outputs/analysis/variance_inequality/radar_regression_2m"

``--predictions`` is the completed regression evaluation file.  The script
expects its ``truth`` group to contain x and its one-member ``prediction``
group to contain mu_hat(y), both for ``--channel`` (radar by default).
``--output-dir`` is created on HNVME and receives ``variance_ratio.nc``,
``variance_ratio.png``, and ``summary.json``.  Copy only the PNG to the Mac
afterwards if that is all that is needed locally.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np

if __package__:
    from .common import read_times
else:
    from common import read_times


DEFAULT_CHANNEL = "maximum_radar_reflectivity"


def git_commit() -> str | None:
    """Return the CorrDiff checkout revision, when Git is available."""

    corrdiff_dir = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "-C", str(corrdiff_dir), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def load_regression_fields(
    path: Path, channel: str, expected_times: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, str | None]:
    """Load target x and one-member regression prediction mu_hat(y)."""

    if not path.is_file():
        raise FileNotFoundError(path)

    with Dataset(path) as source:
        if "truth" not in source.groups or "prediction" not in source.groups:
            raise ValueError("predictions file must contain truth and prediction groups")
        if channel not in source.groups["truth"].variables:
            raise KeyError(f"{channel!r} is absent from the truth group")
        if channel not in source.groups["prediction"].variables:
            raise KeyError(f"{channel!r} is absent from the prediction group")
        if "lat" not in source.variables or "lon" not in source.variables:
            raise ValueError("predictions file must contain root lat and lon variables")

        target_variable = source.groups["truth"].variables[channel]
        prediction_variable = source.groups["prediction"].variables[channel]
        if target_variable.ndim != 3 or prediction_variable.ndim != 4:
            raise ValueError(
                "expected truth(channel)=(time,y,x) and "
                "prediction(channel)=(ensemble,time,y,x)"
            )
        if prediction_variable.shape[0] != 1:
            raise ValueError(
                "this control is defined for the one-member regression output; "
                f"found {prediction_variable.shape[0]} ensemble members"
            )
        if target_variable.shape != prediction_variable.shape[1:]:
            raise ValueError("truth and prediction spatial/time dimensions differ")

        times = read_times(source)
        if len(times) != target_variable.shape[0]:
            raise ValueError("decoded time coordinate disagrees with truth time dimension")
        if len(times) != expected_times:
            raise ValueError(
                f"expected {expected_times} evaluation times, found {len(times)}"
            )

        target = np.asarray(target_variable[:], dtype=np.float64)
        regression = np.asarray(prediction_variable[0], dtype=np.float64)
        lat = np.asarray(source.variables["lat"][:], dtype=np.float64)
        lon = np.asarray(source.variables["lon"][:], dtype=np.float64)
        if lat.shape != target.shape[1:] or lon.shape != target.shape[1:]:
            raise ValueError("lat/lon shapes do not match the field grid")
        return target, regression, lat, lon, len(times), getattr(
            target_variable, "units", None
        )


def variance_maps(
    target: np.ndarray, regression: np.ndarray, min_target_variance: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate Var_t(x), Var_t(x-mu_hat(y)), and their pixelwise ratio."""

    residual = target - regression
    target_variance = np.var(target, axis=0)
    residual_variance = np.var(residual, axis=0)
    valid = (
        np.isfinite(target_variance)
        & np.isfinite(residual_variance)
        & (target_variance > min_target_variance)
    )
    ratio = np.full(target_variance.shape, np.nan, dtype=np.float64)
    ratio[valid] = residual_variance[valid] / target_variance[valid]
    return target_variance, residual_variance, ratio, valid


def write_netcdf(
    path: Path,
    lat: np.ndarray,
    lon: np.ndarray,
    target_variance: np.ndarray,
    residual_variance: np.ndarray,
    ratio: np.ndarray,
    valid: np.ndarray,
    units: str | None,
    source_path: Path,
    n_times: int,
    min_target_variance: float,
) -> None:
    """Write the numerical map outputs with grid coordinates and provenance."""

    with Dataset(path, "w", format="NETCDF4") as output:
        y_size, x_size = ratio.shape
        output.createDimension("y", y_size)
        output.createDimension("x", x_size)
        for name, values in (("lat", lat), ("lon", lon)):
            variable = output.createVariable(name, "f8", ("y", "x"), zlib=True)
            variable[:, :] = values
        for name, values, long_name in (
            ("target_variance", target_variance, "Variance over time of x"),
            (
                "residual_variance",
                residual_variance,
                "Variance over time of x - mu_hat(y)",
            ),
            (
                "variance_ratio",
                ratio,
                "Residual variance divided by target variance",
            ),
        ):
            variable = output.createVariable(
                name, "f8", ("y", "x"), zlib=True, fill_value=np.nan
            )
            variable[:, :] = values
            variable.long_name = long_name
            if units and name != "variance_ratio":
                variable.units = f"({units})2"
        mask = output.createVariable("valid_ratio", "i1", ("y", "x"), zlib=True)
        mask[:, :] = valid.astype(np.int8)
        mask.long_name = "One where the variance ratio is defined"
        output.source_predictions = str(source_path)
        output.n_times = n_times
        output.ddof = 0
        output.min_target_variance = min_target_variance
        commit = git_commit()
        if commit:
            output.git_commit = commit


def plot_ratio(path: Path, lon: np.ndarray, lat: np.ndarray, ratio: np.ndarray) -> None:
    """Plot a geographic ratio map with one as the neutral colour."""

    finite = ratio[np.isfinite(ratio)]
    if finite.size == 0:
        raise ValueError("no grid point has a defined variance ratio")
    vmax = max(1.01, float(np.quantile(finite, 0.99)))
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("0.75")
    figure, axis = plt.subplots(figsize=(9, 7), constrained_layout=True)
    image = axis.pcolormesh(
        lon,
        lat,
        ratio,
        shading="auto",
        cmap=cmap,
        norm=colors.TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=vmax),
    )
    colorbar = figure.colorbar(image, ax=axis, extend="max")
    colorbar.set_label(r"$\mathrm{Var}(x - \hat{\mu}(y)) / \mathrm{Var}(x)$")
    axis.set_title("Taiwan radar regression: pixelwise variance ratio")
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    figure.savefig(path, dpi=180, facecolor="white")
    plt.close(figure)


def summary(
    ratio: np.ndarray,
    valid: np.ndarray,
    source_path: Path,
    n_times: int,
    min_target_variance: float,
) -> dict:
    """Return compact, JSON-serialisable diagnostics for the map."""

    values = ratio[valid]
    if not len(values):
        raise ValueError("no grid point has a defined variance ratio")
    return {
        "source_predictions": str(source_path),
        "n_times": n_times,
        "ddof": 0,
        "min_target_variance": min_target_variance,
        "valid_pixels": int(valid.sum()),
        "total_pixels": int(valid.size),
        "fraction_ratio_le_1": float(np.mean(values <= 1.0)),
        "ratio_min": float(np.min(values)),
        "ratio_median": float(np.median(values)),
        "ratio_mean": float(np.mean(values)),
        "ratio_max": float(np.max(values)),
        "git_commit": git_commit(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--channel", default=DEFAULT_CHANNEL)
    parser.add_argument("--expected-times", type=int, default=256)
    parser.add_argument("--min-target-variance", type=float, default=0.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.expected_times < 1:
        raise ValueError("expected-times must be positive")
    if args.min_target_variance < 0:
        raise ValueError("min-target-variance must be non-negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    target, regression, lat, lon, n_times, units = load_regression_fields(
        args.predictions, args.channel, args.expected_times
    )
    target_variance, residual_variance, ratio, valid = variance_maps(
        target, regression, args.min_target_variance
    )
    write_netcdf(
        args.output_dir / "variance_ratio.nc",
        lat,
        lon,
        target_variance,
        residual_variance,
        ratio,
        valid,
        units,
        args.predictions,
        n_times,
        args.min_target_variance,
    )
    plot_ratio(args.output_dir / "variance_ratio.png", lon, lat, ratio)
    result = summary(
        ratio,
        valid,
        args.predictions,
        n_times,
        args.min_target_variance,
    )
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    print(f"Wrote analysis to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
