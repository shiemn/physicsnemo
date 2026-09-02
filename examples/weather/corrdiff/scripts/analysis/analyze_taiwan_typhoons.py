#!/usr/bin/env python3
"""Matched storm-relative diagnostics for the Taiwan 2021 temporal models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from netCDF4 import Dataset, date2num
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter

if __package__:
    from .common import ModelSource, common_times, parse_model, time_indices
else:
    from common import ModelSource, common_times, parse_model, time_indices


CHANNEL = "maximum_radar_reflectivity"
THRESHOLDS_DBZ = (20.0, 30.0, 40.0)
FSS_SCALES_PX = (1, 3, 9, 27)
GRID_SPACING_KM = 3.0
MODEL_COLORS = {"Baseline": "#4c566a", "Symmetric 1 h": "#0072b2", "Symmetric 3 h": "#d55e00"}
STORM_WINDOWS = {
    "in-fa": ("2106", "In-fa", "2021-07-17T00:00:00Z", "2021-07-28T00:00:00Z"),
    "lupit": ("2109", "Lupit", "2021-08-02T00:00:00Z", "2021-08-10T00:00:00Z"),
    "chanthu": ("2114", "Chanthu", "2021-09-06T00:00:00Z", "2021-09-19T00:00:00Z"),
}


def storm_times(common: pd.DatetimeIndex, storm: str) -> pd.DatetimeIndex:
    _, _, start, stop = STORM_WINDOWS[storm]
    start_time, stop_time = pd.Timestamp(start), pd.Timestamp(stop)
    return common[(common >= start_time) & (common < stop_time)]


def fss(prediction: np.ndarray, truth: np.ndarray, threshold: float, scale_px: int) -> float:
    """Fractions skill score for two 2-D fields."""
    pred_fraction = uniform_filter(
        (prediction >= threshold).astype(np.float32), size=max(1, int(scale_px)), mode="constant"
    )
    truth_fraction = uniform_filter(
        (truth >= threshold).astype(np.float32), size=max(1, int(scale_px)), mode="constant"
    )
    denominator = float(np.mean(pred_fraction**2) + np.mean(truth_fraction**2))
    if denominator <= 1e-12:
        return np.nan
    return float(1.0 - np.mean((pred_fraction - truth_fraction) ** 2) / denominator)


def haversine_grid_km(lat: np.ndarray, lon: np.ndarray, center_lat: float, center_lon: float) -> np.ndarray:
    lat1 = np.deg2rad(center_lat)
    lat2 = np.deg2rad(lat)
    dlat = lat2 - lat1
    dlon = np.deg2rad(lon - center_lon)
    value = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 6371.0 * 2.0 * np.arcsin(np.sqrt(np.clip(value, 0.0, 1.0)))


def storm_relative_xy_km(
    lat: np.ndarray, lon: np.ndarray, center_lat: float, center_lon: float
) -> tuple[np.ndarray, np.ndarray]:
    y = (lat - center_lat) * 111.2
    x = (lon - center_lon) * 111.2 * np.cos(np.deg2rad(center_lat))
    return x, y


def weighted_morphology(
    field: np.ndarray,
    x_km: np.ndarray,
    y_km: np.ndarray,
    radius_km: np.ndarray,
    threshold: float = 20.0,
    max_radius_km: float = 600.0,
) -> dict[str, float]:
    weights = np.maximum(field - threshold, 0.0)
    weights = np.where(radius_km <= max_radius_km, weights, 0.0)
    total = float(np.sum(weights))
    if total <= 1e-6:
        return {"centroid_x_km": np.nan, "centroid_y_km": np.nan, "orientation_deg": np.nan, "anisotropy": np.nan}
    cx = float(np.sum(weights * x_km) / total)
    cy = float(np.sum(weights * y_km) / total)
    dx, dy = x_km - cx, y_km - cy
    cxx = float(np.sum(weights * dx * dx) / total)
    cyy = float(np.sum(weights * dy * dy) / total)
    cxy = float(np.sum(weights * dx * dy) / total)
    eigenvalues, eigenvectors = np.linalg.eigh(np.array([[cxx, cxy], [cxy, cyy]]))
    major = eigenvectors[:, np.argmax(eigenvalues)]
    orientation = float(np.degrees(np.arctan2(major[1], major[0])) % 180.0)
    anisotropy = float((eigenvalues[-1] - eigenvalues[0]) / max(eigenvalues.sum(), 1e-12))
    return {"centroid_x_km": cx, "centroid_y_km": cy, "orientation_deg": orientation, "anisotropy": anisotropy}


def orientation_difference_deg(first: float, second: float) -> float:
    if not np.isfinite(first) or not np.isfinite(second):
        return np.nan
    difference = abs(first - second) % 180.0
    return float(min(difference, 180.0 - difference))


def load_tracks(path: Path) -> pd.DataFrame:
    tracks = pd.read_csv(path)
    tracks["time"] = pd.to_datetime(tracks["time"], utc=True)
    return tracks


def interpolate_track(track: pd.DataFrame, times: pd.DatetimeIndex) -> pd.DataFrame:
    source_ns = track["time"].astype("int64").to_numpy()
    target_ns = times.astype("int64").to_numpy()
    return pd.DataFrame(
        {
            "time": times,
            "center_latitude": np.interp(target_ns, source_ns, track["latitude"], left=np.nan, right=np.nan),
            "center_longitude": np.interp(target_ns, source_ns, track["longitude"], left=np.nan, right=np.nan),
            "central_pressure_hpa": np.interp(
                target_ns, source_ns, track["central_pressure_hpa"], left=np.nan, right=np.nan
            ),
            "max_sustained_wind_kt": np.interp(
                target_ns, source_ns, track["max_sustained_wind_kt"], left=np.nan, right=np.nan
            ),
        }
    )


def radial_profile(field: np.ndarray, radius_km: np.ndarray, edges: np.ndarray) -> np.ndarray:
    indices = np.digitize(radius_km.ravel(), edges) - 1
    values = field.ravel()
    output = np.full(len(edges) - 1, np.nan)
    for index in range(len(output)):
        selected = indices == index
        if np.any(selected):
            output[index] = float(np.nanmean(values[selected]))
    return output


def load_selected_fields(
    model: ModelSource, native_times: pd.DatetimeIndex, selected: pd.DatetimeIndex
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    indices = time_indices(native_times, selected)
    with Dataset(model.path) as ds:
        lat = np.asarray(ds.variables["lat"][:], dtype=np.float32)
        lon = np.asarray(ds.variables["lon"][:], dtype=np.float32)
        truth = np.asarray(ds.groups["truth"].variables[CHANNEL][indices], dtype=np.float32)
        prediction = np.asarray(ds.groups["prediction"].variables[CHANNEL][:, indices], dtype=np.float32)
    return lat, lon, truth, prediction


def analyze_storm(
    storm: str,
    models: list[ModelSource],
    native_times: dict[str, pd.DatetimeIndex],
    selected: pd.DatetimeIndex,
    tracks: pd.DataFrame,
    output_dir: Path,
) -> None:
    storm_id, storm_name, _, _ = STORM_WINDOWS[storm]
    storm_dir = output_dir / storm
    storm_dir.mkdir(parents=True, exist_ok=True)
    track = tracks[tracks["storm_id"].astype(str).str.zfill(4) == storm_id].sort_values("time")
    interpolated_track = interpolate_track(track, selected)
    interpolated_track.to_csv(storm_dir / "interpolated_jma_track.csv", index=False)

    lat, lon, truth, _ = load_selected_fields(models[0], native_times[models[0].label], selected)
    pixel_area = GRID_SPACING_KM**2
    radial_edges = np.arange(0.0, 651.0, 50.0)
    frame_rows: list[dict] = []
    fss_rows: list[dict] = []
    morphology_rows: list[dict] = []
    radial_rows: list[dict] = []
    continuity_rows: list[dict] = []
    predictions_for_plot: dict[str, np.ndarray] = {}

    truth_areas = {threshold: np.sum(truth >= threshold, axis=(1, 2)) * pixel_area for threshold in THRESHOLDS_DBZ}
    for model in models:
        model_lat, model_lon, model_truth, prediction = load_selected_fields(
            model, native_times[model.label], selected
        )
        if not np.allclose(model_lat, lat) or not np.allclose(model_lon, lon):
            raise ValueError(f"grid differs for {model.label}")
        if not np.allclose(model_truth, truth, equal_nan=True):
            raise ValueError(f"truth differs for {model.label} on {storm_name}")
        ensemble_mean = np.mean(prediction, axis=0)
        predictions_for_plot[model.label] = ensemble_mean
        errors = ensemble_mean - truth
        for time_index, timestamp in enumerate(selected):
            frame = {
                "storm": storm_name,
                "time": timestamp,
                "model": model.label,
                "rmse_dbz": float(np.sqrt(np.mean(errors[time_index] ** 2))),
                "mae_dbz": float(np.mean(np.abs(errors[time_index]))),
                "bias_dbz": float(np.mean(errors[time_index])),
            }
            for threshold in THRESHOLDS_DBZ:
                member_areas = np.sum(prediction[:, time_index] >= threshold, axis=(1, 2)) * pixel_area
                frame[f"truth_area_ge_{int(threshold)}dbz_km2"] = float(truth_areas[threshold][time_index])
                frame[f"prediction_area_ge_{int(threshold)}dbz_km2"] = float(np.mean(member_areas))
                frame[f"area_error_ge_{int(threshold)}dbz_km2"] = float(np.mean(member_areas) - truth_areas[threshold][time_index])
                for scale_px in FSS_SCALES_PX:
                    scores = [
                        fss(member, truth[time_index], threshold, scale_px)
                        for member in prediction[:, time_index]
                    ]
                    fss_rows.append(
                        {
                            "storm": storm_name,
                            "time": timestamp,
                            "model": model.label,
                            "threshold_dbz": threshold,
                            "scale_px": scale_px,
                            "scale_km": scale_px * GRID_SPACING_KM,
                            "fss": float(np.nanmean(scores)) if np.any(np.isfinite(scores)) else np.nan,
                        }
                    )
            frame_rows.append(frame)

            center_lat = interpolated_track.iloc[time_index]["center_latitude"]
            center_lon = interpolated_track.iloc[time_index]["center_longitude"]
            radius = haversine_grid_km(lat, lon, center_lat, center_lon)
            x_km, y_km = storm_relative_xy_km(lat, lon, center_lat, center_lon)
            truth_shape = weighted_morphology(truth[time_index], x_km, y_km, radius)
            predicted_shape = weighted_morphology(ensemble_mean[time_index], x_km, y_km, radius)
            morphology_rows.append(
                {
                    "storm": storm_name,
                    "time": timestamp,
                    "model": model.label,
                    **{f"truth_{key}": value for key, value in truth_shape.items()},
                    **{f"prediction_{key}": value for key, value in predicted_shape.items()},
                    "centroid_error_km": float(
                        np.hypot(
                            predicted_shape["centroid_x_km"] - truth_shape["centroid_x_km"],
                            predicted_shape["centroid_y_km"] - truth_shape["centroid_y_km"],
                        )
                    ),
                    "orientation_error_deg": orientation_difference_deg(
                        predicted_shape["orientation_deg"], truth_shape["orientation_deg"]
                    ),
                }
            )
            truth_profile = radial_profile(truth[time_index], radius, radial_edges)
            predicted_profile = radial_profile(ensemble_mean[time_index], radius, radial_edges)
            for radial_index in range(len(radial_edges) - 1):
                radial_rows.append(
                    {
                        "storm": storm_name,
                        "time": timestamp,
                        "model": model.label,
                        "radius_inner_km": radial_edges[radial_index],
                        "radius_outer_km": radial_edges[radial_index + 1],
                        "truth_mean_dbz": truth_profile[radial_index],
                        "prediction_mean_dbz": predicted_profile[radial_index],
                    }
                )

        consecutive = np.diff(selected.astype("int64")) == pd.Timedelta(hours=1).value
        truth_change = np.diff(truth, axis=0)
        prediction_change = np.diff(prediction, axis=1)
        for change_index in np.flatnonzero(consecutive):
            member_change_error = prediction_change[:, change_index] - truth_change[change_index]
            continuity_rows.append(
                {
                    "storm": storm_name,
                    "time": selected[change_index + 1],
                    "model": model.label,
                    "change_rmse_dbz": float(np.mean(np.sqrt(np.mean(member_change_error**2, axis=(1, 2))))),
                    "change_mae_dbz": float(np.mean(np.mean(np.abs(member_change_error), axis=(1, 2)))),
                    "truth_step_rmse_dbz": float(np.sqrt(np.mean(truth_change[change_index] ** 2))),
                    "prediction_step_rmse_dbz": float(
                        np.mean(np.sqrt(np.mean(prediction_change[:, change_index] ** 2, axis=(1, 2))))
                    ),
                }
            )
        print(f"{storm_name}: analyzed {model.label} ({len(selected)} hours)", flush=True)

    frames = pd.DataFrame(frame_rows)
    fss_frame = pd.DataFrame(fss_rows)
    morphology = pd.DataFrame(morphology_rows)
    radial = pd.DataFrame(radial_rows)
    continuity = pd.DataFrame(continuity_rows)
    frames.to_csv(storm_dir / "frame_metrics.csv", index=False)
    fss_frame.to_csv(storm_dir / "fss.csv", index=False)
    morphology.to_csv(storm_dir / "storm_relative_morphology.csv", index=False)
    radial.to_csv(storm_dir / "radial_profiles.csv", index=False)
    continuity.to_csv(storm_dir / "continuity.csv", index=False)
    summary = summarize_storm(frames, fss_frame, morphology, continuity)
    summary.to_csv(storm_dir / "summary.csv", index=False)
    plot_diagnostics(storm_name, frames, fss_frame, continuity, radial, summary, storm_dir)
    plot_filmstrip(
        storm_name, selected, lat, lon, truth, predictions_for_plot, interpolated_track, truth_areas[40.0], storm_dir
    )


def summarize_storm(
    frames: pd.DataFrame, fss_frame: pd.DataFrame, morphology: pd.DataFrame, continuity: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for model in frames["model"].drop_duplicates():
        model_frames = frames[frames["model"] == model]
        model_fss = fss_frame[fss_frame["model"] == model]
        model_shape = morphology[morphology["model"] == model]
        model_continuity = continuity[continuity["model"] == model]
        row = {
            "model": model,
            "hours": len(model_frames),
            "mean_rmse_dbz": model_frames["rmse_dbz"].mean(),
            "mean_mae_dbz": model_frames["mae_dbz"].mean(),
            "mean_bias_dbz": model_frames["bias_dbz"].mean(),
            "mean_change_rmse_dbz": model_continuity["change_rmse_dbz"].mean(),
            "mean_centroid_error_km": model_shape["centroid_error_km"].mean(),
            "mean_orientation_error_deg": model_shape["orientation_error_deg"].mean(),
        }
        for threshold in THRESHOLDS_DBZ:
            key = f"area_error_ge_{int(threshold)}dbz_km2"
            row[f"mean_abs_{key}"] = model_frames[key].abs().mean()
            for scale_px in FSS_SCALES_PX:
                selected = model_fss[
                    (model_fss["threshold_dbz"] == threshold) & (model_fss["scale_px"] == scale_px)
                ]
                row[f"mean_fss_{int(threshold)}dbz_{scale_px * int(GRID_SPACING_KM)}km"] = selected["fss"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def plot_diagnostics(
    storm_name: str,
    frames: pd.DataFrame,
    fss_frame: pd.DataFrame,
    continuity: pd.DataFrame,
    radial: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    def plot_segments(axis, frame, column, *, color, label):
        ordered = frame.sort_values("time").copy()
        ordered["time"] = pd.to_datetime(ordered["time"], utc=True)
        groups = ordered["time"].diff().gt(pd.Timedelta(hours=1.5)).cumsum()
        first = True
        for _, segment in ordered.groupby(groups):
            axis.plot(
                segment["time"],
                segment[column],
                color=color,
                linewidth=1.2,
                label=label if first else None,
            )
            first = False

    labels = list(frames["model"].drop_duplicates())
    colors = {label: MODEL_COLORS.get(label, plt.cm.tab10(index)) for index, label in enumerate(labels)}
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    for label in labels:
        color = colors[label]
        frame = frames[frames["model"] == label]
        plot_segments(axes[0, 0], frame, "rmse_dbz", color=color, label=label)
        plot_segments(axes[0, 1], frame, "prediction_area_ge_30dbz_km2", color=color, label=label)
        selected_fss = fss_frame[
            (fss_frame["model"] == label)
            & (fss_frame["threshold_dbz"] == 30.0)
            & (fss_frame["scale_km"] == 27.0)
        ]
        plot_segments(axes[1, 0], selected_fss, "fss", color=color, label=label)
        selected_continuity = continuity[continuity["model"] == label]
        plot_segments(axes[1, 1], selected_continuity, "change_rmse_dbz", color=color, label=label)
    truth_frame = frames[frames["model"] == labels[0]]
    plot_segments(axes[0, 1], truth_frame, "truth_area_ge_30dbz_km2", color="black", label="Radar")
    axes[0, 0].set_ylabel("RMSE (dBZ)")
    axes[0, 1].set_ylabel("Area ≥30 dBZ (km²)")
    axes[1, 0].set_ylabel("FSS, 30 dBZ at 27 km")
    axes[1, 1].set_ylabel("Hour-to-hour change RMSE (dBZ)")
    for axis in axes.flat:
        axis.grid(True, color="0.88", linewidth=0.6)
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axes[0, 0].legend(ncol=len(labels), fontsize=9)
    axes[0, 1].legend(ncol=2, fontsize=9)
    fig.suptitle(f"Typhoon {storm_name}: matched Taiwan temporal-model diagnostics")
    fig.tight_layout()
    fig.savefig(output_dir / "diagnostics.png", dpi=180)
    fig.savefig(output_dir / "diagnostics.pdf")
    plt.close(fig)

    mean_radial = radial.groupby(["model", "radius_inner_km", "radius_outer_km"], as_index=False)[
        ["truth_mean_dbz", "prediction_mean_dbz"]
    ].mean()
    fig, axis = plt.subplots(figsize=(8, 5))
    truth_profile = mean_radial[mean_radial["model"] == labels[0]]
    centers = (truth_profile["radius_inner_km"] + truth_profile["radius_outer_km"]) / 2.0
    axis.plot(centers, truth_profile["truth_mean_dbz"], color="black", linewidth=2, label="Radar")
    for label in labels:
        profile = mean_radial[mean_radial["model"] == label]
        centers = (profile["radius_inner_km"] + profile["radius_outer_km"]) / 2.0
        axis.plot(centers, profile["prediction_mean_dbz"], color=colors[label], linewidth=1.8, label=label)
    axis.set(xlabel="Radius from JMA centre (km)", ylabel="Mean reflectivity (dBZ)")
    axis.grid(True, color="0.88")
    axis.legend()
    axis.set_title(f"Typhoon {storm_name}: time-mean radial reflectivity")
    fig.tight_layout()
    fig.savefig(output_dir / "radial_profiles.png", dpi=180)
    plt.close(fig)

    with (output_dir / "summary.json").open("w") as stream:
        json.dump(summary.to_dict(orient="records"), stream, indent=2)


def plot_filmstrip(
    storm_name: str,
    times: pd.DatetimeIndex,
    lat: np.ndarray,
    lon: np.ndarray,
    truth: np.ndarray,
    predictions: dict[str, np.ndarray],
    track: pd.DataFrame,
    truth_area_40: np.ndarray,
    output_dir: Path,
) -> None:
    peak = int(np.argmax(truth_area_40))
    offsets = (-12, -6, 0, 6, 12, 18)
    indices = sorted(set(int(np.clip(peak + offset, 0, len(times) - 1)) for offset in offsets))
    rows = [("Radar", truth), *predictions.items()]
    fig, axes = plt.subplots(len(rows), len(indices), figsize=(3.0 * len(indices), 2.65 * len(rows)), squeeze=False)
    mesh = None
    for row_index, (label, fields) in enumerate(rows):
        for column_index, time_index in enumerate(indices):
            axis = axes[row_index, column_index]
            mesh = axis.pcolormesh(lon, lat, fields[time_index], shading="auto", cmap="turbo", vmin=0, vmax=55)
            axis.scatter(
                track.iloc[time_index]["center_longitude"],
                track.iloc[time_index]["center_latitude"],
                marker="x",
                s=32,
                linewidth=1.4,
                color="white",
            )
            if row_index == 0:
                axis.set_title(times[time_index].strftime("%b %d\n%H UTC"), fontsize=9)
            if column_index == 0:
                axis.set_ylabel(label)
            axis.set_xticks([])
            axis.set_yticks([])
    assert mesh is not None
    color_axis = fig.add_axes((0.945, 0.15, 0.012, 0.68))
    fig.colorbar(mesh, cax=color_axis, label="Maximum radar reflectivity (dBZ)")
    fig.suptitle(f"Typhoon {storm_name}: evolution around peak ≥40 dBZ area", y=0.995)
    fig.subplots_adjust(left=0.055, right=0.925, bottom=0.03, top=0.94, wspace=0.03, hspace=0.08)
    fig.savefig(output_dir / "filmstrip.png", dpi=180)
    plt.close(fig)


def extract_cases(
    models: list[ModelSource],
    native_times: dict[str, pd.DatetimeIndex],
    common: pd.DatetimeIndex,
    storms: list[str],
    output_path: Path,
) -> None:
    selected_parts = [storm_times(common, storm) for storm in storms]
    selected = selected_parts[0].append(selected_parts[1:])
    storm_index = np.concatenate([np.full(len(part), index, dtype=np.int16) for index, part in enumerate(selected_parts)])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_suffix(output_path.suffix + ".partial")
    if partial.exists():
        raise FileExistsError(partial)
    with Dataset(models[0].path) as reference:
        y_size, x_size = reference.groups["truth"].variables[CHANNEL].shape[-2:]
        ensembles = reference.groups["prediction"].variables[CHANNEL].shape[0]
        time_units = reference.variables["time"].units
        calendar = getattr(reference.variables["time"], "calendar", "standard")
        lat = reference.variables["lat"][:]
        lon = reference.variables["lon"][:]
    try:
        with Dataset(partial, "w", format="NETCDF4") as output:
            output.createDimension("model", len(models))
            output.createDimension("storm", len(storms))
            output.createDimension("ensemble", ensembles)
            output.createDimension("time", len(selected))
            output.createDimension("y", y_size)
            output.createDimension("x", x_size)
            output.createVariable("model", str, ("model",))[:] = np.asarray([item.label for item in models], dtype=object)
            output.createVariable("storm", str, ("storm",))[:] = np.asarray([STORM_WINDOWS[item][1] for item in storms], dtype=object)
            output.createVariable("storm_index", "i2", ("time",))[:] = storm_index
            time_var = output.createVariable("time", "i8", ("time",))
            time_var.units, time_var.calendar = time_units, calendar
            time_var[:] = date2num(selected.to_pydatetime(), units=time_units, calendar=calendar)
            output.createVariable("lat", "f4", ("y", "x"))[:] = lat
            output.createVariable("lon", "f4", ("y", "x"))[:] = lon
            truth_out = output.createVariable(CHANNEL + "_truth", "f4", ("time", "y", "x"))
            prediction_out = output.createVariable(
                CHANNEL + "_prediction", "f4", ("model", "ensemble", "time", "y", "x"),
                chunksizes=(1, 1, 1, y_size, x_size),
            )
            output.grid_spacing_km = GRID_SPACING_KM
            output.source = "Matched 2021 Taiwan CorrDiff typhoon cases"
            reference_indices = time_indices(native_times[models[0].label], selected)
            with Dataset(models[0].path) as source:
                source_truth = source.groups["truth"].variables[CHANNEL]
                for start in range(0, len(selected), 16):
                    stop = min(start + 16, len(selected))
                    truth_out[start:stop] = source_truth[reference_indices[start:stop]]
            for model_index, model in enumerate(models):
                indices = time_indices(native_times[model.label], selected)
                with Dataset(model.path) as source:
                    variable = source.groups["prediction"].variables[CHANNEL]
                    for start in range(0, len(selected), 8):
                        stop = min(start + 8, len(selected))
                        prediction_out[model_index, :, start:stop] = variable[:, indices[start:stop]]
                        print(f"extract {model.label}: {stop}/{len(selected)}", flush=True)
        partial.replace(output_path)
    except Exception:
        if partial.exists():
            partial.unlink()
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", type=parse_model, required=True)
    parser.add_argument("--storm", action="append", choices=tuple(STORM_WINDOWS), default=[])
    parser.add_argument(
        "--track-file",
        type=Path,
        default=Path(__file__).with_name("data")
        / "jma_best_track_2021_selected.txt",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/taiwan_typhoon_analysis"))
    parser.add_argument("--extract-output", type=Path)
    parser.add_argument("--extract-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    models = args.model
    storms = args.storm or ["chanthu", "lupit", "in-fa"]
    for model in models:
        if not model.path.is_file():
            raise FileNotFoundError(model.path)
    common, native = common_times(models)
    selected_counts = {storm: len(storm_times(common, storm)) for storm in storms}
    print(f"common timestamps: {len(common)}; selected: {selected_counts}", flush=True)
    if args.extract_output:
        extract_cases(models, native, common, storms, args.extract_output)
    if not args.extract_only:
        tracks = load_tracks(args.track_file)
        for storm in storms:
            selected = storm_times(common, storm)
            if not len(selected):
                raise ValueError(f"no common timestamps in {storm}")
            analyze_storm(storm, models, native, selected, tracks, args.output_dir)
        metadata = {
            "models": [{"label": model.label, "path": str(model.path)} for model in models],
            "storms": storms,
            "common_timestamp_count": len(common),
            "selected_counts": selected_counts,
            "thresholds_dbz": THRESHOLDS_DBZ,
            "fss_scales_km": [item * GRID_SPACING_KM for item in FSS_SCALES_PX],
            "track_source": "JMA RSMC Tokyo best-track archive, bst_all.zip",
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with (args.output_dir / "metadata.json").open("w") as stream:
            json.dump(metadata, stream, indent=2)


if __name__ == "__main__":
    main()
