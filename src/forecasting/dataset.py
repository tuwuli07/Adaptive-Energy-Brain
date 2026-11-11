"""PyTorch datasets over the leakage-safe Part 2 forecasting sequences."""

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class ForecastDatasetError(ValueError):
    """Invalid sequence arrays, timestamps, masks, or graph metadata."""


def _utc_string(value: np.datetime64) -> str:
    return np.datetime_as_string(value, unit="s") + "Z"


class ForecastingDataset(Dataset):
    """Samples one partition prepared by :func:`src.data.sequences.prepare_sequences`.

    Each item contains node history ``X`` [lookback, nodes, features], optional
    global history, physical-unit targets ``y`` [horizon, nodes, targets],
    masks, UTC timestamps, and partition/sample metadata. The Part 2 sequence
    builder remains responsible for creating windows and applying the saved,
    training-fitted input scalers.
    """

    def __init__(
        self,
        sequence_data: Mapping[str, Any],
        *,
        expected_partition: str | None = None,
    ) -> None:
        if not isinstance(sequence_data, Mapping):
            raise ForecastDatasetError("sequence_data must be a prepared sequence mapping.")
        try:
            arrays = sequence_data["arrays"]
            metadata = sequence_data["metadata"]
            graph = sequence_data["graph"]
        except KeyError as exc:
            raise ForecastDatasetError(f"Prepared sequence data is missing {exc.args[0]!r}.") from exc
        if not all(isinstance(item, Mapping) for item in (arrays, metadata, graph)):
            raise ForecastDatasetError("arrays, metadata, and graph must be mappings.")

        partition = metadata.get("partition")
        if partition not in {"train", "validation", "test"}:
            raise ForecastDatasetError("metadata.partition must be train, validation, or test.")
        if expected_partition is not None and partition != expected_partition:
            raise ForecastDatasetError(
                f"Expected {expected_partition!r} data, received {partition!r}."
            )
        self.partition = partition
        self.metadata = dict(metadata)

        sources = metadata.get("source_metadata")
        if not isinstance(sources, Mapping) or not sources:
            raise ForecastDatasetError("Source preprocessing provenance is required.")
        for source_id, source_metadata in sources.items():
            split = source_metadata.get("split", {}) if isinstance(source_metadata, Mapping) else {}
            if (
                split.get("name") != partition
                or split.get("fitted_on") != "train"
                or split.get("boundaries") != metadata.get("boundaries")
            ):
                raise ForecastDatasetError(
                    f"{source_id}: expected {partition} data transformed with train-fitted preprocessing."
                )

        node_order = metadata.get("node_order")
        if (
            not isinstance(node_order, list)
            or not node_order
            or any(not isinstance(node, str) or not node for node in node_order)
            or len(set(node_order)) != len(node_order)
        ):
            raise ForecastDatasetError("metadata.node_order must contain unique node IDs in order.")
        self.node_order = tuple(node_order)

        required = (
            "X",
            "global_X",
            "y",
            "input_timestamps",
            "target_timestamps",
            "node_input_mask",
            "target_mask",
        )
        missing = [key for key in required if key not in arrays]
        if missing:
            raise ForecastDatasetError(f"Sequence arrays are missing {missing}.")
        self._arrays = {key: np.asarray(arrays[key]) for key in required}

        x = self._arrays["X"]
        global_x = self._arrays["global_X"]
        y = self._arrays["y"]
        if x.ndim != 4 or y.ndim != 4 or global_x.ndim != 3:
            raise ForecastDatasetError(
                "Expected X [sample, lookback, node, feature], "
                "global_X [sample, lookback, feature], and "
                "y [sample, horizon, node, target]."
            )
        if x.dtype != np.float32 or global_x.dtype != np.float32 or y.dtype != np.float32:
            raise ForecastDatasetError("X, global_X, and y must use float32 values.")
        sample_count, lookback, node_count, feature_count = x.shape
        if sample_count < 1:
            raise ForecastDatasetError("A partition must contain at least one sequence.")
        if node_count != len(self.node_order) or y.shape[2] != node_count:
            raise ForecastDatasetError("Input and target node dimensions must match node_order.")
        if feature_count != 1 or y.shape[3] != 1:
            raise ForecastDatasetError(
                "The audited forecasting schema requires one power input feature and one power target."
            )
        if global_x.shape[:2] != (sample_count, lookback):
            raise ForecastDatasetError("global_X sample/lookback dimensions do not match X.")
        if y.shape[0] != sample_count or y.shape[1] < 1:
            raise ForecastDatasetError("y must contain one nonempty horizon for each input.")
        config = metadata.get("config", {})
        if config.get("lookback_hours") != lookback or config.get("horizon_hours") != y.shape[1]:
            raise ForecastDatasetError("Sequence dimensions do not match their recorded configuration.")
        stride = config.get("stride_hours")
        if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
            raise ForecastDatasetError("config.stride_hours must be a positive integer.")
        if metadata.get("target_unit") != "kW":
            raise ForecastDatasetError("Forecast targets must remain in physical kW.")
        for name, values in (("X", x), ("global_X", global_x), ("y", y)):
            if not np.isfinite(values).all():
                raise ForecastDatasetError(f"{name} contains missing or non-finite observations.")
        if (y < 0).any():
            raise ForecastDatasetError("Demand and available-generation targets must be nonnegative.")

        input_times = self._arrays["input_timestamps"]
        target_times = self._arrays["target_timestamps"]
        if (
            input_times.dtype != np.dtype("datetime64[ns]")
            or target_times.dtype != np.dtype("datetime64[ns]")
            or input_times.shape != (sample_count, lookback)
            or target_times.shape != (sample_count, y.shape[1])
        ):
            raise ForecastDatasetError(
                "UTC input/target timestamps must be datetime64[ns] arrays aligned to sequences."
            )
        if np.isnat(input_times).any() or np.isnat(target_times).any():
            raise ForecastDatasetError("Sequence timestamps cannot be NaT.")
        one_hour = np.timedelta64(1, "h")
        if (
            (np.diff(input_times, axis=1) != one_hour).any()
            or (np.diff(target_times, axis=1) != one_hour).any()
            or (input_times[:, -1] >= target_times[:, 0]).any()
        ):
            raise ForecastDatasetError("Windows must be hourly and targets must follow their inputs.")
        if sample_count > 1 and (
            (np.diff(input_times[:, 0]) <= np.timedelta64(0, "ns")).any()
            or (np.diff(target_times[:, 0]) <= np.timedelta64(0, "ns")).any()
        ):
            raise ForecastDatasetError("Sequence samples must be in chronological order.")
        if sample_count > 1 and (
            (np.diff(input_times[:, 0]) != stride * one_hour).any()
            or (np.diff(target_times[:, 0]) != stride * one_hour).any()
        ):
            raise ForecastDatasetError("Sequence samples do not match configured stride_hours.")

        boundaries = metadata.get("boundaries")
        if not isinstance(boundaries, Mapping):
            raise ForecastDatasetError("Partition time boundaries are required.")
        try:
            start = np.datetime64(boundaries["start_utc"].replace("Z", "").replace("+00:00", ""), "ns")
            end = np.datetime64(boundaries["end_exclusive_utc"].replace("Z", "").replace("+00:00", ""), "ns")
        except (KeyError, AttributeError, TypeError, ValueError) as exc:
            raise ForecastDatasetError("Invalid UTC partition boundaries.") from exc
        if (
            start >= end
            or input_times.min() < start
            or target_times.max() >= end
        ):
            raise ForecastDatasetError(
                "Input windows and complete target horizons must stay inside their partition."
            )

        for name in ("node_input_mask", "target_mask"):
            mask = self._arrays[name]
            if mask.dtype != np.bool_ or mask.shape != (node_count, 1):
                raise ForecastDatasetError(f"{name} must be a boolean [nodes, 1] mask.")
        if (
            np.any(x[:, :, ~self._arrays["node_input_mask"][:, 0], :] != 0)
            or np.any(y[:, :, ~self._arrays["target_mask"][:, 0], :] != 0)
        ):
            raise ForecastDatasetError(
                "Masked structural nodes must use zero placeholders and be excluded by their masks."
            )

        self._x = x
        self._global_x = global_x
        self._y = y
        self._input_times = input_times
        self._target_times = target_times
        self._node_input_mask = self._arrays["node_input_mask"]
        self._target_mask = self._arrays["target_mask"]
        self.graph = graph

    def __len__(self) -> int:
        return self._x.shape[0]

    def __getitem__(self, index: int) -> dict[str, Any]:
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise TypeError("ForecastingDataset indices must be integers.")
        index = int(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("ForecastingDataset index out of range.")
        input_timestamps = tuple(_utc_string(value) for value in self._input_times[index])
        target_timestamps = tuple(_utc_string(value) for value in self._target_times[index])
        return {
            "X": torch.from_numpy(self._x[index].copy()),
            "global_X": torch.from_numpy(self._global_x[index].copy()),
            "y": torch.from_numpy(self._y[index].copy()),
            "node_input_mask": torch.from_numpy(self._node_input_mask.copy()),
            "target_mask": torch.from_numpy(self._target_mask.copy()),
            "input_timestamps": input_timestamps,
            "target_timestamps": target_timestamps,
            "metadata": {
                "partition": self.partition,
                "sample_index": index,
                "input_cutoff_time_utc": input_timestamps[-1],
            },
        }
