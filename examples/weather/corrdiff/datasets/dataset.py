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

from datetime import datetime
from typing import Iterable, Tuple, Union
import copy
import importlib.util
from pathlib import Path

import numpy as np
import torch

from physicsnemo.utils.diffusion import InfiniteSampler
from physicsnemo.distributed import DistributedManager

from datasets import base, cwb, hrrrmini, gefs_hrrr, norway, cordex_bench
from datasets.base import ChannelMetadata


# this maps all known dataset types to the corresponding init function
known_datasets = {
    "cwb": cwb.get_zarr_dataset,
    "hrrr_mini": hrrrmini.HRRRMiniDataset,
    "gefs_hrrr": gefs_hrrr.HrrrForecastGEFSDataset,
    "norway": norway.NorwayDatasetH5,
    "cordex_bench": cordex_bench.CordexBenchDataset,
}


class TemporalInputDataset(base.DownscalingDataset):
    """Wrap a downscaling dataset with temporal input conditioning.

    The center sample provides the target and any auxiliary labels. Inputs from
    the configured offsets are concatenated along the channel dimension.
    """

    def __init__(
        self,
        dataset: base.DownscalingDataset,
        offsets: list[int],
        boundary: str = "drop",
        strict_time_step_hours: Union[int, float, None] = None,
    ):
        self._dataset = dataset
        self.offsets = self._validate_offsets(offsets)
        self.boundary = boundary
        self.strict_time_step_hours = strict_time_step_hours
        self._base_input_channels = dataset.input_channels()

        if boundary != "drop":
            raise ValueError(
                f"Unsupported temporal_inputs.boundary={boundary!r}. "
                "Only 'drop' is currently supported."
            )

        self._center_indices = self._build_center_indices()

    @staticmethod
    def _validate_offsets(offsets: list[int]) -> list[int]:
        if not offsets:
            raise ValueError("temporal_inputs.offsets must be a non-empty list")
        if any(not isinstance(offset, int) for offset in offsets):
            raise ValueError("temporal_inputs.offsets must contain only integers")
        if len(set(offsets)) != len(offsets):
            raise ValueError("temporal_inputs.offsets must not contain duplicates")
        return list(offsets)

    @staticmethod
    def _time_to_datetime(value) -> datetime:
        if isinstance(value, datetime):
            return value
        if isinstance(value, np.datetime64):
            seconds = (
                value.astype("datetime64[s]") - np.datetime64("1970-01-01T00:00:00")
            ) / np.timedelta64(1, "s")
            return datetime.utcfromtimestamp(float(seconds))
        if all(hasattr(value, attr) for attr in ("year", "month", "day")):
            return datetime(
                value.year,
                value.month,
                value.day,
                getattr(value, "hour", 0),
                getattr(value, "minute", 0),
                getattr(value, "second", 0),
            )
        raise TypeError(
            f"strict_time_step_hours requires datetime-like dataset times, got {type(value).__name__}"
        )

    def _has_expected_time_offsets(self, center_idx: int, all_times: list) -> bool:
        if self.strict_time_step_hours is None:
            return True
        center_time = self._time_to_datetime(all_times[center_idx])
        for offset in self.offsets:
            offset_time = self._time_to_datetime(all_times[center_idx + offset])
            expected_seconds = float(offset) * float(self.strict_time_step_hours) * 3600.0
            actual_seconds = (offset_time - center_time).total_seconds()
            if abs(actual_seconds - expected_seconds) > 1.0:
                return False
        return True

    def _build_center_indices(self) -> list[int]:
        n_samples = len(self._dataset)
        min_offset = min(self.offsets)
        max_offset = max(self.offsets)
        start = max(0, -min_offset)
        stop = min(n_samples, n_samples - max_offset)
        candidates = list(range(start, stop))

        if self.strict_time_step_hours is None:
            return candidates

        all_times = self._dataset.time()
        return [
            center_idx
            for center_idx in candidates
            if self._has_expected_time_offsets(center_idx, all_times)
        ]

    @staticmethod
    def _concat_inputs(inputs: list):
        first = inputs[0]
        if torch.is_tensor(first):
            return torch.cat(inputs, dim=0)
        return np.concatenate(inputs, axis=0)

    @staticmethod
    def _offset_suffix(offset: int) -> str:
        if offset < 0:
            return f"tm{abs(offset)}"
        if offset > 0:
            return f"tp{offset}"
        return "t0"

    def __getitem__(self, idx):
        center_idx = self._center_indices[idx]
        center_sample = self._dataset[center_idx]
        target = center_sample[0]
        extras = center_sample[2:]

        temporal_inputs = [
            self._dataset[center_idx + offset][1] for offset in self.offsets
        ]
        return (target, self._concat_inputs(temporal_inputs), *extras)

    def __len__(self):
        return len(self._center_indices)

    def longitude(self) -> np.ndarray:
        return self._dataset.longitude()

    def latitude(self) -> np.ndarray:
        return self._dataset.latitude()

    def input_channels(self) -> list[ChannelMetadata]:
        channels = []
        for offset in self.offsets:
            suffix = self._offset_suffix(offset)
            for channel in self._base_input_channels:
                name = channel.name if channel.level else f"{channel.name}_{suffix}"
                level = f"{channel.level}_{suffix}" if channel.level else channel.level
                channels.append(
                    ChannelMetadata(
                        name=name,
                        level=level,
                        auxiliary=channel.auxiliary,
                    )
                )
        return channels

    def output_channels(self) -> list[ChannelMetadata]:
        return self._dataset.output_channels()

    def time(self) -> list:
        all_times = self._dataset.time()
        return [all_times[i] for i in self._center_indices]

    def image_shape(self) -> Tuple[int, int]:
        return self._dataset.image_shape()

    def _apply_input_transform(self, x: np.ndarray, transform_name: str) -> np.ndarray:
        base_channels = len(self._base_input_channels)
        if x.shape[-3] != base_channels * len(self.offsets):
            raise ValueError(
                f"Expected {base_channels * len(self.offsets)} temporal input channels, "
                f"got {x.shape[-3]}."
            )
        blocks = np.split(x, len(self.offsets), axis=-3)
        transform = getattr(self._dataset, transform_name)
        return np.concatenate([transform(block) for block in blocks], axis=-3)

    def normalize_input(self, x: np.ndarray) -> np.ndarray:
        return self._apply_input_transform(x, "normalize_input")

    def denormalize_input(self, x: np.ndarray) -> np.ndarray:
        return self._apply_input_transform(x, "denormalize_input")

    def normalize_output(self, x: np.ndarray) -> np.ndarray:
        return self._dataset.normalize_output(x)

    def denormalize_output(self, x: np.ndarray) -> np.ndarray:
        return self._dataset.denormalize_output(x)

    def info(self) -> dict:
        return self._dataset.info()


def maybe_wrap_temporal_inputs(
    dataset: base.DownscalingDataset,
    temporal_cfg: Union[dict, None],
) -> base.DownscalingDataset:
    """Wrap ``dataset`` with temporal inputs if requested by config."""

    if temporal_cfg is None:
        return dataset
    if not isinstance(temporal_cfg, dict):
        raise ValueError("dataset.temporal_inputs must be a mapping or null")

    temporal_cfg = copy.deepcopy(temporal_cfg)
    offsets = temporal_cfg.pop("offsets", None)
    boundary = temporal_cfg.pop("boundary", "drop")
    strict_time_step_hours = temporal_cfg.pop("strict_time_step_hours", None)
    if temporal_cfg:
        raise ValueError(
            f"Unknown temporal_inputs option(s): {', '.join(sorted(temporal_cfg))}"
        )
    return TemporalInputDataset(
        dataset,
        offsets=offsets,
        boundary=boundary,
        strict_time_step_hours=strict_time_step_hours,
    )


def register_dataset(dataset_spec: str) -> None:
    """
    Register a new dataset class from a file path specification.

    Parameters
    ----------
    dataset_spec : str
        String specification in the format "path_to_file.py::dataset_class"

    Raises
    ------
    ValueError
        If the dataset_spec format is invalid or if the file doesn't exist
    ImportError
        If the dataset class cannot be imported
    """
    if dataset_spec in known_datasets:
        return  # Dataset already registered
    try:
        file_path, class_name = dataset_spec.split("::")
    except ValueError:
        raise ValueError(
            "Invalid dataset specification. Expected format: "
            "'path_to_file.py::dataset_class'"
        )
    if class_name in known_datasets:
        return  # Dataset already registered

    # Convert to Path and validate
    file_path = Path(file_path)
    if not file_path.exists():
        raise ValueError(f"Dataset file not found: {file_path}")
    if not file_path.suffix == ".py":
        raise ValueError(f"Dataset file must be a Python file: {file_path}")

    # Import the module and get the class
    spec = importlib.util.spec_from_file_location(file_path.stem, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    try:
        dataset_class = getattr(module, class_name)
    except AttributeError:
        raise ImportError(f"Could not find dataset class '{class_name}' in {file_path}")

    # Register the dataset
    known_datasets[dataset_spec] = dataset_class
    return


def init_train_valid_datasets_from_config(
    dataset_cfg: dict,
    dataloader_cfg: Union[dict, None] = None,
    batch_size: int = 1,
    seed: int = 0,
    validation_dataset_cfg: Union[dict, None] = None,
    validation: bool = True,
    sampler_start_idx: int = 0,
) -> Tuple[
    base.DownscalingDataset,
    Iterable,
    Union[base.DownscalingDataset, None],
    Union[Iterable, None],
]:
    """
    A wrapper function for managing the train-test split for the CWB dataset.

    Parameters:
    - dataset_cfg (dict): Configuration for the dataset.
    - dataloader_cfg (dict, optional): Configuration for the dataloader. Defaults to None.
    - batch_size (int): The number of samples in each batch of data. Defaults to 1.
    - seed (int): The random seed for dataset shuffling. Defaults to 0.
    - validation (bool): A flag to determine whether to create a validation dataset. Defaults to True.
    - sampler_start_idx (int): The initial index of the sampler to use for resuming training. Defaults to 0.

    Returns:
    - Tuple[base.DownscalingDataset, Iterable, Optional[base.DownscalingDataset], Optional[Iterable]]: A tuple containing the training dataset and iterator, and optionally the validation dataset and iterator if `validation` is True.
    """

    config = copy.deepcopy(dataset_cfg)
    (dataset, dataset_iter) = init_dataset_from_config(
        config,
        dataloader_cfg,
        batch_size=batch_size,
        seed=seed,
        sampler_start_idx=sampler_start_idx,
    )
    if validation:
        valid_dataset_cfg = copy.deepcopy(config)
        if validation_dataset_cfg:
            valid_dataset_cfg.update(validation_dataset_cfg)
        (valid_dataset, valid_dataset_iter) = init_dataset_from_config(
            valid_dataset_cfg, dataloader_cfg, batch_size=batch_size, seed=seed
        )
    else:
        valid_dataset = valid_dataset_iter = None

    return dataset, dataset_iter, valid_dataset, valid_dataset_iter


def init_dataset_from_config(
    dataset_cfg: dict,
    dataloader_cfg: Union[dict, None] = None,
    batch_size: int = 1,
    seed: int = 0,
    sampler_start_idx: int = 0,
) -> Tuple[base.DownscalingDataset, Iterable]:
    dataset_cfg = copy.deepcopy(dataset_cfg)
    dataset_type = dataset_cfg.pop("type", "cwb")
    if "validation" in dataset_cfg:
        # handled by init_train_valid_datasets_from_config
        del dataset_cfg["validation"]
    temporal_cfg = dataset_cfg.pop("temporal_inputs", None)
    dataset_init_func = known_datasets[dataset_type]

    dataset_obj = dataset_init_func(**dataset_cfg)
    dataset_obj = maybe_wrap_temporal_inputs(dataset_obj, temporal_cfg)
    if dataloader_cfg is None:
        dataloader_cfg = {}

    dist = DistributedManager()
    dataset_sampler = InfiniteSampler(
        dataset=dataset_obj,
        rank=dist.rank,
        num_replicas=dist.world_size,
        seed=seed,
        start_idx=sampler_start_idx,
    )

    dataset_iterator = iter(
        torch.utils.data.DataLoader(
            dataset=dataset_obj,
            sampler=dataset_sampler,
            batch_size=batch_size,
            worker_init_fn=None,
            **dataloader_cfg,
        )
    )

    return (dataset_obj, dataset_iterator)
