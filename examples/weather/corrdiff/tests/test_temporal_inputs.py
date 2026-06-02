import datetime
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.base import ChannelMetadata, DownscalingDataset
from datasets.dataset import TemporalInputDataset, maybe_wrap_temporal_inputs


class DummyTemporalDataset(DownscalingDataset):
    def __init__(self):
        self._times = [
            datetime.datetime(2020, 1, 1, hour)
            for hour in range(5)
        ]

    def __getitem__(self, idx):
        target = np.full((1, 2, 2), idx, dtype=np.float32)
        input_field = np.stack(
            [
                np.full((2, 2), idx, dtype=np.float32),
                np.full((2, 2), idx + 10, dtype=np.float32),
            ],
            axis=0,
        )
        return target, input_field, idx * 100

    def __len__(self):
        return len(self._times)

    def longitude(self):
        return np.zeros((2, 2), dtype=np.float32)

    def latitude(self):
        return np.zeros((2, 2), dtype=np.float32)

    def input_channels(self):
        return [
            ChannelMetadata(name="u", level="500"),
            ChannelMetadata(name="orog", auxiliary=True),
        ]

    def output_channels(self):
        return [ChannelMetadata(name="precip")]

    def time(self):
        return self._times

    def image_shape(self):
        return 2, 2

    def normalize_input(self, x):
        return x + 1

    def denormalize_input(self, x):
        return x - 1

    def normalize_output(self, x):
        return x + 2

    def denormalize_output(self, x):
        return x - 2


def test_maybe_wrap_temporal_inputs_disabled_returns_base_dataset():
    dataset = DummyTemporalDataset()

    assert maybe_wrap_temporal_inputs(dataset, None) is dataset


def test_temporal_inputs_accept_physical_hour_offsets():
    dataset = maybe_wrap_temporal_inputs(
        DummyTemporalDataset(),
        {
            "offset_hours": [-1, 0, 1],
            "strict_time_step_hours": 1,
        },
    )

    target, input_field, label = dataset[0]

    assert len(dataset) == 3
    np.testing.assert_array_equal(target, np.full((1, 2, 2), 1, dtype=np.float32))
    assert label == 100
    assert input_field.shape == (6, 2, 2)
    np.testing.assert_array_equal(
        input_field[:, 0, 0],
        np.array([0, 10, 1, 11, 2, 12], dtype=np.float32),
    )


def test_temporal_inputs_reject_non_integral_physical_offsets():
    dataset = DummyTemporalDataset()

    try:
        maybe_wrap_temporal_inputs(
            dataset,
            {
                "offset_hours": [-1.5, 0, 1.5],
                "strict_time_step_hours": 1,
            },
        )
    except ValueError as exc:
        assert "integer multiples" in str(exc)
    else:
        raise AssertionError("Expected non-integral physical offsets to fail")


def test_temporal_inputs_stack_offsets_and_preserve_center_sample_extras():
    dataset = TemporalInputDataset(
        DummyTemporalDataset(),
        offsets=[-1, 0, 1],
        strict_time_step_hours=1,
    )

    target, input_field, label = dataset[0]

    assert len(dataset) == 3
    np.testing.assert_array_equal(target, np.full((1, 2, 2), 1, dtype=np.float32))
    assert label == 100
    assert input_field.shape == (6, 2, 2)
    np.testing.assert_array_equal(input_field[:, 0, 0], np.array([0, 10, 1, 11, 2, 12], dtype=np.float32))


def test_temporal_inputs_time_returns_center_times():
    dataset = TemporalInputDataset(DummyTemporalDataset(), offsets=[-1, 0, 1])

    assert dataset.time() == [
        datetime.datetime(2020, 1, 1, 1),
        datetime.datetime(2020, 1, 1, 2),
        datetime.datetime(2020, 1, 1, 3),
    ]


def test_temporal_input_channels_are_suffixed_and_unique():
    dataset = TemporalInputDataset(DummyTemporalDataset(), offsets=[-1, 0, 1])

    channels = dataset.input_channels()
    names = [channel.name + channel.level for channel in channels]

    assert names == ["u500_tm1", "orog_tm1", "u500_t0", "orog_t0", "u500_tp1", "orog_tp1"]
    assert len(names) == len(set(names))
    assert channels[1].auxiliary is True


def test_temporal_denormalize_input_delegates_per_temporal_block():
    dataset = TemporalInputDataset(DummyTemporalDataset(), offsets=[-1, 0, 1])
    x = np.ones((1, 6, 2, 2), dtype=np.float32)

    denormalized = dataset.denormalize_input(x)

    np.testing.assert_array_equal(denormalized, np.zeros_like(x))


def test_temporal_inputs_strict_time_step_drops_gaps():
    base = DummyTemporalDataset()
    base._times[2] = datetime.datetime(2020, 1, 1, 5)

    dataset = TemporalInputDataset(
        base,
        offsets=[-1, 0, 1],
        strict_time_step_hours=1,
    )

    assert len(dataset) == 0
