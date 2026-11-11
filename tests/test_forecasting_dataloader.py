"""Synthetic fixture tests for causal forecasting datasets and graph loaders."""

from copy import deepcopy

import numpy as np
import pytest
import torch

from src.data.graph_builder import build_graph
from src.forecasting.dataloader import (
    ForecastDataLoaderError,
    build_forecasting_dataloaders,
    inverse_scale_with_training_scaler,
    move_batch_to_device,
    validate_forecasting_graph,
)
from src.forecasting.dataset import ForecastDatasetError, ForecastingDataset


NODE_ORDER = ["B01", "B02", "B03", "B04", "B05", "PV01", "W01", "BUS01"]


def _partition(partition: str, start: str, sample_count: int = 5):
    lookback, horizon, nodes = 4, 3, len(NODE_ORDER)
    sample_offsets = np.arange(sample_count)[:, None]
    input_offsets = np.arange(lookback)[None, :]
    target_offsets = np.arange(horizon)[None, :]
    timestamp_start = np.datetime64(start, "ns")
    input_timestamps = timestamp_start + (sample_offsets + input_offsets) * np.timedelta64(1, "h")
    target_timestamps = (
        timestamp_start
        + (sample_offsets + lookback + target_offsets) * np.timedelta64(1, "h")
    )
    partition_end = timestamp_start + np.timedelta64(
        sample_count + lookback + horizon - 1, "h"
    )
    graph = build_graph()
    node_mask = np.ones((nodes, 1), dtype=bool)
    node_mask[-1] = False
    inputs = np.arange(sample_count * lookback * nodes, dtype=np.float32).reshape(
        sample_count, lookback, nodes, 1
    ) / 10
    inputs[:, :, -1, :] = 0
    targets = (
        np.arange(sample_count * horizon * nodes, dtype=np.float32).reshape(
            sample_count, horizon, nodes, 1
        )
        + 100
    )
    targets[:, :, -1, :] = 0
    global_inputs = np.arange(sample_count * lookback * 3, dtype=np.float32).reshape(
        sample_count, lookback, 3
    ) / 5
    source_metadata = {
        "synthetic_fixture": {
            "split": {
                "name": partition,
                "fitted_on": "train",
                "boundaries": {
                    "start_utc": f"{start}Z",
                    "end_exclusive_utc": np.datetime_as_string(partition_end, unit="s") + "Z",
                },
            }
        }
    }
    return {
        "arrays": {
            "X": inputs,
            "global_X": global_inputs,
            "y": targets,
            "input_timestamps": input_timestamps.astype("datetime64[ns]"),
            "target_timestamps": target_timestamps.astype("datetime64[ns]"),
            "node_input_mask": node_mask,
            "target_mask": node_mask.copy(),
        },
        "graph": graph,
        "metadata": {
            "partition": partition,
            "boundaries": {
                "start_utc": f"{start}Z",
                "end_exclusive_utc": np.datetime_as_string(partition_end, unit="s") + "Z",
            },
            "config": {
                "lookback_hours": lookback,
                "horizon_hours": horizon,
                "stride_hours": 1,
            },
            "node_order": list(NODE_ORDER),
            "target_unit": "kW",
            "source_metadata": source_metadata,
        },
    }


@pytest.fixture
def partition_results():
    return {
        "train": _partition("train", "2026-01-01T00:00:00"),
        "validation": _partition("validation", "2026-02-01T00:00:00"),
        "test": _partition("test", "2026-03-01T00:00:00"),
    }


def test_dataset_length_and_expected_tensor_dimensions(partition_results):
    dataset = ForecastingDataset(partition_results["train"])
    sample = dataset[0]
    assert len(dataset) == 5
    assert sample["X"].shape == (4, 8, 1)
    assert sample["global_X"].shape == (4, 3)
    assert sample["y"].shape == (3, 8, 1)
    assert sample["X"].dtype == sample["y"].dtype == torch.float32
    assert sample["target_mask"].shape == (8, 1)
    assert sample["target_mask"][-1, 0].item() is False


def test_sample_windows_and_targets_have_correct_historical_alignment(partition_results):
    sample = ForecastingDataset(partition_results["train"])[2]
    np.testing.assert_array_equal(
        sample["X"].numpy(), partition_results["train"]["arrays"]["X"][2]
    )
    np.testing.assert_array_equal(
        sample["y"].numpy(), partition_results["train"]["arrays"]["y"][2]
    )
    assert sample["input_timestamps"][0] == "2026-01-01T02:00:00Z"
    assert sample["input_timestamps"][-1] == "2026-01-01T05:00:00Z"
    assert sample["target_timestamps"][0] == "2026-01-01T06:00:00Z"
    assert sample["target_timestamps"][-1] == "2026-01-01T08:00:00Z"
    assert sample["metadata"]["input_cutoff_time_utc"] == sample["input_timestamps"][-1]


def test_validation_and_test_never_borrow_context_across_split_boundaries(partition_results):
    loaders = build_forecasting_dataloaders(partition_results, batch_size=3)
    for partition in ("train", "validation", "test"):
        first = next(iter(loaders[partition]))
        bounds = partition_results[partition]["metadata"]["boundaries"]
        start = np.datetime64(bounds["start_utc"].replace("Z", ""), "ns")
        end = np.datetime64(bounds["end_exclusive_utc"].replace("Z", ""), "ns")
        for inputs, targets in zip(
            first["input_timestamps"], first["target_timestamps"]
        ):
            input_times = np.asarray(
                [stamp.removesuffix("Z") for stamp in inputs], dtype="datetime64[ns]"
            )
            target_times = np.asarray(
                [stamp.removesuffix("Z") for stamp in targets], dtype="datetime64[ns]"
            )
            assert input_times.min() >= start
            assert input_times.max() < target_times.min()
            assert target_times.max() < end


def test_future_target_changes_do_not_change_historical_input_features(partition_results):
    original = deepcopy(partition_results["train"])
    changed = deepcopy(partition_results["train"])
    changed["arrays"]["y"][:, :, :-1, :] = 50
    before = ForecastingDataset(original)[0]
    after = ForecastingDataset(changed)[0]
    torch.testing.assert_close(before["X"], after["X"])
    assert not torch.equal(before["y"], after["y"])


def test_graph_matches_node_order_and_has_no_fabricated_edge_weights(partition_results):
    loaders = build_forecasting_dataloaders(partition_results, batch_size=2)
    graph = loaders.graph
    assert graph.node_order == tuple(NODE_ORDER)
    assert graph.edge_index.shape == (2, 14)
    assert graph.edge_index.dtype == torch.int64
    assert graph.node_attributes.shape == (8, 6)
    assert graph.edge_weight is None
    assert graph.metadata["construction"] == "assumed_shared_bus"
    assert all(
        edge["basis"] == "modeling_assumption"
        for edge in graph.metadata["edges"]
    )
    assert graph.disconnected_nodes == ()


def test_disconnected_nodes_are_supported_without_invented_edges(partition_results):
    graph = deepcopy(partition_results["train"]["graph"])
    graph["edge_index"] = np.array([[0, 1], [1, 0]], dtype=np.int64)
    disconnected = validate_forecasting_graph(
        graph, node_count=8, expected_node_order=NODE_ORDER
    )
    assert disconnected.disconnected_nodes == tuple(NODE_ORDER[2:])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("edge_index", np.array([[0], [8]], dtype=np.int64), "outside"),
        ("edge_index", np.array([[0], [1]], dtype=np.int32), "int64"),
        ("node_attributes", np.zeros((7, 6), dtype=np.float32), "node_attributes"),
    ],
)
def test_invalid_graph_indices_and_dimensions_are_rejected(
    partition_results, field, value, message
):
    graph = deepcopy(partition_results["train"]["graph"])
    graph[field] = value
    with pytest.raises(ForecastDataLoaderError, match=message):
        validate_forecasting_graph(graph, node_count=8, expected_node_order=NODE_ORDER)


def test_graph_rejects_mismatched_partition_node_order(partition_results):
    invalid = deepcopy(partition_results)
    invalid["validation"]["metadata"]["node_order"][0] = "OTHER"
    with pytest.raises(ForecastDataLoaderError, match="node order"):
        build_forecasting_dataloaders(invalid)


def test_optional_edge_weights_are_validated_not_synthesized(partition_results):
    graph = deepcopy(partition_results["train"]["graph"])
    graph["edge_weight"] = np.ones(14, dtype=np.float32)
    validated = validate_forecasting_graph(graph, node_count=8, expected_node_order=NODE_ORDER)
    np.testing.assert_array_equal(validated.edge_weight.numpy(), np.ones(14))
    graph["edge_weight"] = np.ones(13, dtype=np.float32)
    with pytest.raises(ForecastDataLoaderError, match="one value per edge"):
        validate_forecasting_graph(graph, node_count=8, expected_node_order=NODE_ORDER)


def test_variable_batches_keep_last_incomplete_batch_and_cpu_device(partition_results):
    loaders = build_forecasting_dataloaders(
        partition_results,
        batch_size=2,
        shuffle_train=False,
        device="cpu",
    )
    batches = list(loaders.train)
    assert [len(batch["X"]) for batch in batches] == [2, 2, 1]
    assert all(batch["X"].device.type == "cpu" for batch in batches)
    moved = move_batch_to_device(batches[0], "cpu")
    assert moved["X"].device.type == "cpu"
    assert moved["target_timestamps"] == batches[0]["target_timestamps"]


def test_training_shuffle_keeps_each_window_chronological_and_is_seeded(partition_results):
    first = build_forecasting_dataloaders(partition_results, batch_size=2, seed=19)
    second = build_forecasting_dataloaders(partition_results, batch_size=2, seed=19)
    first_order = [
        metadata["sample_index"]
        for batch in first.train
        for metadata in batch["metadata"]
    ]
    second_order = [
        metadata["sample_index"]
        for batch in second.train
        for metadata in batch["metadata"]
    ]
    assert first_order == second_order
    assert first_order != list(range(5))
    for batch in first.train:
        for inputs, targets in zip(batch["input_timestamps"], batch["target_timestamps"]):
            assert inputs[-1] < targets[0]
            assert all(a < b for a, b in zip(inputs, inputs[1:]))
            assert all(a < b for a, b in zip(targets, targets[1:]))


def test_validation_and_test_are_deterministic_and_preserve_timestamps(partition_results):
    loaders = build_forecasting_dataloaders(partition_results, batch_size=2)
    for partition in ("validation", "test"):
        first = list(loaders[partition])
        second = list(loaders[partition])
        assert [
            metadata["sample_index"]
            for batch in first
            for metadata in batch["metadata"]
        ] == list(range(5))
        for left, right in zip(first, second):
            torch.testing.assert_close(left["X"], right["X"])
            assert left["target_timestamps"] == right["target_timestamps"]


def test_train_fitted_preprocessing_metadata_is_required(partition_results):
    invalid = deepcopy(partition_results)
    invalid["validation"]["metadata"]["source_metadata"]["synthetic_fixture"]["split"][
        "fitted_on"
    ] = "validation"
    with pytest.raises(ForecastDatasetError, match="train-fitted"):
        ForecastingDataset(invalid["validation"])


@pytest.mark.parametrize(
    ("array_name", "mutator", "message"),
    [
        ("X", lambda values: values.astype(np.float64), "float32"),
        ("X", lambda values: values.copy() * np.nan, "non-finite"),
        ("y", lambda values: values[:, :, :-1, :], "node dimensions"),
    ],
)
def test_invalid_observations_and_dimensions_are_rejected(
    partition_results, array_name, mutator, message
):
    invalid = deepcopy(partition_results["train"])
    invalid["arrays"][array_name] = mutator(invalid["arrays"][array_name])
    with pytest.raises(ForecastDatasetError, match=message):
        ForecastingDataset(invalid)


def test_dataset_rejects_chronology_and_boundary_leaks(partition_results):
    invalid = deepcopy(partition_results["train"])
    invalid["arrays"]["target_timestamps"][0, 0] = invalid["arrays"]["input_timestamps"][0, -1]
    with pytest.raises(ForecastDatasetError, match="targets must follow"):
        ForecastingDataset(invalid)
    invalid = deepcopy(partition_results["train"])
    invalid["metadata"]["boundaries"]["end_exclusive_utc"] = "2026-01-01T00:00:00Z"
    invalid["metadata"]["source_metadata"]["synthetic_fixture"]["split"][
        "boundaries"
    ] = invalid["metadata"]["boundaries"]
    with pytest.raises(ForecastDatasetError, match="inside their partition"):
        ForecastingDataset(invalid)


def test_existing_train_scaler_inverse_round_trip_keeps_targets_in_kw(partition_results):
    scaler = {
        "method": "standard",
        "fitted_on": "train",
        "entities": {"B01": {"building_load_kw": {"mean": 20.0, "scale": 4.0}}},
    }
    standardized = np.array([-1.0, 0.0, 1.5])
    physical = inverse_scale_with_training_scaler(
        standardized, scaler, entity_id="B01", feature="building_load_kw"
    )
    np.testing.assert_array_equal(physical, [16.0, 20.0, 26.0])
    dataset = ForecastingDataset(partition_results["train"])
    np.testing.assert_array_equal(
        dataset[0]["y"].numpy(), partition_results["train"]["arrays"]["y"][0]
    )


def test_builders_require_all_splits_and_valid_options(partition_results):
    with pytest.raises(ForecastDataLoaderError, match="exactly"):
        build_forecasting_dataloaders({"train": partition_results["train"]})
    with pytest.raises(ForecastDataLoaderError, match="batch_size"):
        build_forecasting_dataloaders(partition_results, batch_size=0)
