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

import datetime
import math
from typing import List, Tuple, Union

import json
import numpy as np
from numba import jit, prange
import xarray as xr
import os

from physicsnemo.utils.diffusion import convert_datetime_to_cftime

from datasets.base import ChannelMetadata, DownscalingDataset

import torch
import h5py

norway_bounds_small = {
    "selected_area": [250, 30],
    "selected_area_size": 256,
    "selected_area_small": [250 + 32, 30 + 32],
    "selected_area_size_small": 256 - 64
}
norway_bounds_large = {
    "selected_area": [265, 60],
    "selected_area_size": 512,
    "selected_area_small": [265, 60],
    "selected_area_size_small": 512
}


class NorwayDatasetH5(DownscalingDataset):
    def __init__(
        self,
        data_path: str,
        stats_path: str,
        input_variables: Union[List[str], None] = ["hus500","hus850","ua500","ua850","va500","va850","ta500","ta850"],
        output_variables: Union[List[str], None] = ['precipitation'],
        invariant_variables: Union[List[str], None] = [], #("elevation"),
        invariant_variables_path: Union[str, None] = None,
        years: Union[List[int], None] = None,
        bounds: str = "small",
    ):
        self.data_path = data_path

        self.input_variables = input_variables
        self.output_variables = output_variables
        self.invariant_variables = invariant_variables

        self.coarsening_factor = 1
        self.bounds = bounds
        if bounds == "small":
            print("Using the small crop over eastern Norway.")
            bounds_dict = norway_bounds_small
        elif bounds == "large":
            print("Using the large crop over southern Norway.")
            bounds_dict = norway_bounds_large
        else:
            raise ValueError(f"Unknown bounds: {bounds}")


        self.selected_area = bounds_dict['selected_area']
        self.selected_area_size = bounds_dict['selected_area_size']
        self.selected_area_small = bounds_dict['selected_area_small']
        self.selected_area_size_small = bounds_dict['selected_area_size_small']
        
        self.normalize = True

        self.lat = None
        self.lon = None

        stage = 'all'
        self.years = years
        

        self.train_years = list(range(1986,2002))
        self.val_years = [2002, 2003]
        self.test_years = [2004, 2005]

        if self.invariant_variables is not None and len(self.invariant_variables) > 0:
            if not os.path.exists(invariant_variables_path):
                raise ValueError(f"Static orography (and lat/lon) data not found at {invariant_variables_path}. Please provide the correct path.")
            print(f"Found static orography data. Loading orography, lat, and lon from {invariant_variables_path}.")
            self.static_variables, self.lat, self.lon = self._prepare_static_data(invariant_variables_path, bounds_dict)


        else:
            self.static_variables = None

       
        if os.path.exists(os.path.join(data_path, "predictor_mean_std.pt")):
            print(f"Found normalization coefficients. Loading them from {os.path.join(data_path, 'predictor_mean_std.pt')}.")
            self.predictor_mean, self.predictor_std = torch.load(os.path.join(data_path, "predictor_mean_std.pt"), weights_only=False)
            self.predictor_mean = self.predictor_mean.astype(np.float32)
            self.predictor_std = self.predictor_std.astype(np.float32)

        if os.path.exists(os.path.join(data_path, "target_mean_std.pt")):
            print(f"Found normalization coefficients. Loading them from {os.path.join(data_path, 'target_mean_std.pt')}.")
            self.target_mean, self.target_std = torch.load(os.path.join(data_path, "target_mean_std.pt"), weights_only=False)
            self.target_mean = self.target_mean.astype(np.float32)
            self.target_std = self.target_std.astype(np.float32)
            self.target_mean *= 3600
            self.target_std *= 3600

        else:
            raise ValueError(f"No normalization coefficients found at {data_path}. Please run the calculate_mean_std.py script to calculate them.")

        if self.years is None:
            if stage == 'train':
                print("Return dataset in train stage.")
                self.years = list(range(1986,2002))
            elif stage == 'val' or stage == 'valid' or stage == 'validation':
                print("Return dataset in validation stage.")
                self.years = [2002, 2003]
            elif stage == 'test':
                print("Return dataset in test stage.")
                self.years = self.test_years
            else:
                print("Return dataset in full stage. All years")
                self.years = list(range(1986,2006))
        else:
            self.years = list(self.years)
            print(f"Return dataset in custom stage. Years {self.years}")



        self.files_target = [os.path.join(self.data_path,f) for f in os.listdir(data_path) if f.startswith("targets") and f.endswith(".h5")]
        self.files_predictant = [os.path.join(self.data_path,f) for f in os.listdir(data_path) if f.startswith("predictors_") and f.endswith(".h5")]

        # Filter files by year
        self.files_target = sorted([file for file in self.files_target if int(file[-7:-3]) in self.years])
        self.files_predictant = sorted([file for file in self.files_predictant if int(file[-7:-3]) in self.years])

        assert len(self.files_target) == len(self.files_predictant), "Number of Target/Predictor Files do not match"
        print(f"Number of Files {len(self.files_predictant)}")

        self.file_lengths = []

        for file in self.files_predictant:
            with h5py.File(file, 'r') as f:
                self.file_lengths.append(f['predictors'].shape[0])

        for idx, file in enumerate(self.files_target):
            with h5py.File(file, 'r') as f:
                assert self.file_lengths[idx] == f['targets'].shape[0], f"File {file} has different number of samples"
        
        assert len(self.files_target) > 0, f"No files found in the dataset directory {self.data_path}"

        self.cum_file_length = np.cumsum(self.file_lengths)
                

    def __getitem__(self, idx):
        """Return a tuple of:
        - target_field: High-resolution HCLIM3 output data
        - input_field: Low-resolution coarsened HCLIM3 input data
        - lead_time_label: Lead time (None for now)
        """

        # print(f"Getting item at index {idx}.")

        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_idx: int = np.searchsorted(self.cum_file_length, idx, side='right')
        if file_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self.cum_file_length[file_idx - 1] 

        with h5py.File(self.files_predictant[file_idx], 'r') as f:
            predictor = f['predictors'][local_idx].astype(np.float32)

        with h5py.File(self.files_target[file_idx], 'r') as f:
            target = f['targets'][local_idx].astype(np.float32)
        
        #Convert Precipitation from kg/m^2/s to mm per 3 hours
        target = target * 3600
        
        if self.normalize:
            predictor = (predictor - self.predictor_mean[:, None, None]) / self.predictor_std[:, None, None]
            target = (target - self.target_mean) / self.target_std

        predictor = self.upsample(predictor)

        if self.static_variables is not None:
            predictor = np.concatenate((predictor, self.static_variables), axis=0)

        if self.bounds == 'small':
            predictor = predictor[:, 32:-32, 32:-32]

        # print(f"Predictor shape: {predictor.shape}, Target shape: {target.shape}")
        # print(f"Predictor dtype: {predictor.dtype}, Target dtype: {target.dtype}")


        return target[None, :], predictor
        
    def upsample(self, x: np.ndarray) -> np.ndarray:
        """
        Upsample the input data to the target resolution.
        """
        
        return x.repeat(4, axis=1).repeat(4, axis=2)

    def __len__(self):
        return sum(self.file_lengths)

    def latitude(self):
        return self.lat
    


    def longitude(self):
        return self.lon

    def time(self) -> List:
        """Get time values from the dataset."""

        print("Returning time values for the dataset...")

        print(f"Years: {self.years}, File lengths: {self.file_lengths}")
        
        # Build a list of all timestamps based on years and 3-hour intervals
        datetimes = []
        for i, year in enumerate(self.years):
            # Start from January 1st of each year
            current_date = datetime.datetime(year, 1, 1, 0, 0, 0)
            
            # Generate all timestamps for the year with 3-hour intervals
            for i in range(self.file_lengths[i]):
                datetimes.append(current_date)
                current_date += datetime.timedelta(hours=3)

        print(f"Time values generated for years {self.years}: {len(datetimes)} timestamps.")
        print(f"First timestamp: {datetimes[0]}, Last timestamp: {datetimes[-1]}")

        return [convert_datetime_to_cftime(t) for t in datetimes]


    def get_prob_channel_index(self):
        """
        Get prob_channel_index list one more dimension
        """
        return self.prob_channel_index + [len(self.output_variables) - 1]

    def input_channels(self):
        """Metadata for the input channels. A list of ChannelMetadata, one for each channel"""
        inputs = [ChannelMetadata(name=v) for v in self.input_variables]
        invariants = [
            ChannelMetadata(name=v, auxiliary=True) for v in self.invariant_variables
        ]
        return inputs + invariants

    def output_channels(self):
        return [ChannelMetadata(name=n) for n in self.output_variables]

    def info(self):
        return {
            "input_normalization": (
                self.predictor_mean.squeeze(),
                self.predictor_std.squeeze(),
            ),
            "target_normalization": (
                self.target_mean.squeeze(),
                self.target_std.squeeze(),
            ),
        }

    def image_shape(self) -> Tuple[int, int]:
        """Get the (height, width) of the data (same for input and output)."""
        if self.bounds == "small":
            return 192, 192
        elif self.bounds == "large":
            return 512, 512
        else:
            raise ValueError('Unknown value for bounds')
    
    def _prepare_static_data(self, orography_path, bounds: dict):
        selected_area = bounds['selected_area']
        selected_area_size = bounds["selected_area_size"]
        selected_area_small = bounds['selected_area_small']
        selected_area_size_small = bounds['selected_area_size_small']

        static = xr.open_dataset(os.path.join(orography_path, 'orog_NEU-3_ECMWF-ERAINT_evaluation_r1i1p1_HCLIMcom-HCLIM38-AROME_x2yn2v1_fx.nc'))
        orography = static['orog'][selected_area[0]:selected_area[0]+selected_area_size, selected_area[1]:selected_area[1]+selected_area_size]

        lat = static['lat'][selected_area_small[0]:selected_area_small[0]+selected_area_size_small, selected_area_small[1]:selected_area_small[1]+selected_area_size_small]
        lon = static['lon'][selected_area_small[0]:selected_area_small[0]+selected_area_size_small, selected_area_small[1]:selected_area_small[1]+selected_area_size_small]


        orography = (orography - orography.min()) / (orography.max() - orography.min())
        orography = np.array(orography.values).astype(np.float32)[None, :, :]
        return orography, lat, lon
    

    def denormalize_output(self, x: np.ndarray) -> np.ndarray:
        """Convert output from normalized data to physical units."""
        return (x * self.target_std) + self.target_mean
