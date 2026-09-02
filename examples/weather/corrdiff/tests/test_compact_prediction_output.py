from datetime import datetime
from types import SimpleNamespace

import netCDF4 as nc
import numpy as np
import torch

from helpers.generate_helpers import save_images
from physicsnemo.utils.corrdiff.utils import NetCDFWriter


class _Dataset:
    def denormalize_input(self, _value):
        raise AssertionError("compact output must not denormalize inputs")

    def denormalize_output(self, value):
        return value.numpy() if torch.is_tensor(value) else np.asarray(value)

    def output_channels(self):
        return [SimpleNamespace(name="precipitation", level="")]

    def input_channels(self):
        raise AssertionError("compact output must not inspect input channels")


class _FourChannelDataset(_Dataset):
    _names = [
        "maximum_radar_reflectivity",
        "temperature_2m",
        "eastward_wind_10m",
        "northward_wind_10m",
    ]

    def output_channels(self):
        return [SimpleNamespace(name=name, level="") for name in self._names]


class _Writer:
    def __init__(self):
        self.times = []
        self.truth = []
        self.predictions = []
        self.inputs = []

    def write_time(self, index, value):
        self.times.append((index, value))

    def write_truth(self, name, index, value):
        self.truth.append((name, index, np.asarray(value)))

    def write_prediction(self, name, time_index, ensemble_index, value):
        self.predictions.append(
            (name, time_index, ensemble_index, np.asarray(value))
        )

    def write_input(self, *args):
        self.inputs.append(args)


def test_save_images_can_omit_inputs_for_two_member_output():
    writer = _Writer()
    image_out = torch.arange(8, dtype=torch.float32).reshape(2, 1, 2, 2)
    image_tar = torch.ones((1, 1, 2, 2), dtype=torch.float32)

    save_images(
        writer=writer,
        dataset=_Dataset(),
        times=["2005-01-02T00:00:00"],
        image_out=image_out,
        image_tar=image_tar,
        image_lr=None,
        time_index=0,
        t_index=0,
        has_lead_time=False,
        save_inputs=False,
    )

    assert writer.inputs == []
    assert len(writer.predictions) == 2
    assert {entry[2] for entry in writer.predictions} == {0, 1}
    assert all(entry[0] == "precipitation" for entry in writer.predictions)
    assert len(writer.truth) == 2


def test_compact_netcdf_has_two_members_and_no_input_variables(tmp_path):
    output_path = tmp_path / "predictions.nc"
    channel = SimpleNamespace(name="precipitation", level="")
    dataset = _Dataset()

    with nc.Dataset(output_path, "w") as nc_file:
        writer = NetCDFWriter(
            nc_file,
            lat=np.zeros((2, 2), dtype=np.float32),
            lon=np.ones((2, 2), dtype=np.float32),
            input_channels=[],
            output_channels=[channel],
        )
        save_images(
            writer=writer,
            dataset=dataset,
            times=[datetime(2005, 1, 2)],
            image_out=torch.arange(8, dtype=torch.float32).reshape(2, 1, 2, 2),
            image_tar=torch.ones((1, 1, 2, 2), dtype=torch.float32),
            image_lr=None,
            time_index=0,
            t_index=0,
            has_lead_time=False,
            save_inputs=False,
        )

    with nc.Dataset(output_path) as nc_file:
        assert nc_file.groups["prediction"]["precipitation"].shape == (2, 1, 2, 2)
        assert nc_file.groups["truth"]["precipitation"].shape == (1, 2, 2)
        assert list(nc_file.groups["input"].variables) == []


def test_compact_netcdf_can_save_one_of_four_model_outputs(tmp_path):
    output_path = tmp_path / "predictions.nc"
    dataset = _FourChannelDataset()
    saved_channel = dataset.output_channels()[0]

    with nc.Dataset(output_path, "w") as nc_file:
        writer = NetCDFWriter(
            nc_file,
            lat=np.zeros((2, 2), dtype=np.float32),
            lon=np.ones((2, 2), dtype=np.float32),
            input_channels=[],
            output_channels=[saved_channel],
        )
        save_images(
            writer=writer,
            dataset=dataset,
            times=[datetime(2021, 2, 1)],
            image_out=torch.arange(32, dtype=torch.float32).reshape(2, 4, 2, 2),
            image_tar=torch.ones((1, 4, 2, 2), dtype=torch.float32),
            image_lr=None,
            time_index=0,
            t_index=0,
            has_lead_time=False,
            save_inputs=False,
            output_channel_indices=[0],
        )

    with nc.Dataset(output_path) as nc_file:
        assert list(nc_file.groups["prediction"].variables) == [
            "maximum_radar_reflectivity"
        ]
        assert list(nc_file.groups["truth"].variables) == [
            "maximum_radar_reflectivity"
        ]
        assert nc_file.groups["prediction"]["maximum_radar_reflectivity"].shape == (
            2,
            1,
            2,
            2,
        )
        assert list(nc_file.groups["input"].variables) == []
