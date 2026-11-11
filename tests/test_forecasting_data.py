"""Offline graph ordering and strictly chronological forecast tensor checks."""

from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from src.data.graph_builder import GraphDataError, build_graph, load_forecasting_config
from src.data.sequences import (
    SequenceDataError,
    load_partition_sources,
    prepare_sequences,
    save_forecasting_bundle,
)
from src.data.splitting import save_splits, split_datasets, verify_split_bundle


def make_sources(hours=60, partition="train"):
    stamps = pd.date_range("2026-01-01", periods=hours, freq="h", tz="UTC")
    strings = stamps.strftime("%Y-%m-%dT%H:%M:%SZ")
    bounds = {"start_utc": strings[0], "end_exclusive_utc": (stamps[-1] + pd.Timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"), "hours": hours}
    sources = []

    def add(source_id, kind, frame, units, scaled):
        sources.append({"source_id": source_id, "record_type": kind, "data": frame, "units": units,
                        "metadata": {"source_id": source_id, "origin": "simulated", "synthetic": True,
                                     "source_metadata": {"note": "Unit-test observations only"},
                                     "split": {"name": partition, "fitted_on": "train", "boundaries": bounds,
                                               "scaled_units": {column: "1" for column in scaled}}}})

    buildings = []
    for index in range(5):
        values = np.arange(hours, dtype=float) + 100 * index + 1
        buildings.append(pd.DataFrame({"timestamp_utc": strings, "interval_hours": 1,
                                       "building_id": f"B0{index + 1}", "building_load_kw": values,
                                       "building_load_kw_scaled": values / 10}))
    add("fixture_demand", "demand", pd.concat(buildings, ignore_index=True),
        {"interval_hours": "h", "building_load_kw": "kW"}, ["building_load_kw_scaled"])
    for index, (asset, technology) in enumerate((("PV01", "solar"), ("W01", "wind"))):
        values = np.arange(hours, dtype=float) + 501 + index * 100
        add("fixture_" + technology, "generation", pd.DataFrame({
            "timestamp_utc": strings, "interval_hours": 1, "asset_id": asset, "generation_type": technology,
            "available_generation_kw": values, "available_generation_kw_scaled": values / 10}),
            {"interval_hours": "h", "available_generation_kw": "kW"}, ["available_generation_kw_scaled"])
    weather = pd.DataFrame({"timestamp_utc": strings, "interval_hours": 1, "site_id": "SYNTHETIC_SITE"})
    units = {"interval_hours": "h"}
    scaled = []
    for column, unit in (("irradiance_w_m2", "W/m^2"), ("air_temperature_c", "degC"), ("wind_speed_m_s", "m/s")):
        weather[column] = np.arange(hours, dtype=float)
        weather[column + "_scaled"] = np.arange(hours, dtype=float) / 10
        units[column] = unit
        scaled.append(column + "_scaled")
    add("fixture_weather", "weather", weather, units, scaled)
    return sources


def test_default_graph_order_attributes_and_assumed_edges():
    graph = build_graph()
    assert graph["metadata"]["node_order"] == ["B01", "B02", "B03", "B04", "B05", "PV01", "W01", "BUS01"]
    assert graph["node_attributes"].shape == (8, 6)
    assert graph["node_attributes"].dtype == np.float32
    assert graph["edge_index"].shape == (2, 14)
    assert graph["edge_index"].dtype == np.int64
    assert np.all((graph["edge_index"] >= 0) & (graph["edge_index"] < 8))
    assert all(edge["basis"] == "modeling_assumption" for edge in graph["metadata"]["edges"])
    assert graph["node_attributes"][-1, -1] == 0 # Unknown bus rating is explicitly masked.
    pairs = {tuple(pair) for pair in graph["edge_index"].T}
    assert all((target, source) in pairs for source, target in pairs)


def test_explicit_physical_edges_require_documentation():
    config = load_forecasting_config()
    config["graph"].update(construction="explicit", edges=[
        {"source": "B01", "target": "BUS01", "basis": "documented_physical", "reference": "Test-only documentation identifier"},
    ])
    assert build_graph(config)["edge_index"].shape == (2, 1)
    del config["graph"]["edges"][0]["reference"]
    with pytest.raises(GraphDataError, match="reference"):
        build_graph(config)


@pytest.mark.parametrize("problem", ["unknown_node", "duplicate_node", "duplicate_edge", "self_loop"])
def test_invalid_graphs_rejected(problem):
    config = load_forecasting_config()
    if problem == "duplicate_node":
        config["graph"]["nodes"][1]["id"] = "B01"
    else:
        edge = {"source": "B01", "target": "BUS01", "basis": "modeling_assumption", "reference": "Test fixture"}
        config["graph"].update(construction="explicit", edges=[edge])
        if problem == "unknown_node":
            edge["target"] = "UNKNOWN"
        elif problem == "self_loop":
            edge["target"] = "B01"
        else:
            config["graph"]["edges"].append(deepcopy(edge))
    with pytest.raises(GraphDataError):
        build_graph(config)


def test_default_sequence_dimensions_masks_and_units():
    result = prepare_sequences(make_sources())
    arrays = result["arrays"]
    assert arrays["X"].shape == (13, 24, 8, 1)
    assert arrays["y"].shape == (13, 24, 8, 1)
    assert arrays["global_X"].shape == (13, 24, 3)
    assert arrays["X"].dtype == arrays["y"].dtype == np.float32
    assert arrays["target_mask"].shape == arrays["node_input_mask"].shape == (8, 1)
    assert arrays["target_mask"].sum() == 7
    assert not arrays["target_mask"][-1, 0]
    assert (arrays["X"][:, :, -1, :] == 0).all()
    assert result["metadata"]["target_unit"] == "kW"


def test_targets_strictly_follow_history_and_values_match_node_order():
    result = prepare_sequences(make_sources())
    arrays = result["arrays"]
    assert (arrays["input_timestamps"][:, -1] < arrays["target_timestamps"][:, 0]).all()
    np.testing.assert_allclose(arrays["X"][0, :, 0, 0], np.arange(1, 25) / 10)
    np.testing.assert_allclose(arrays["y"][0, :, 0, 0], np.arange(25, 49))
    assert arrays["y"][0, 0, 1, 0] == 125
    assert arrays["y"][0, 0, 5, 0] == 525


def test_future_targets_do_not_enter_earlier_input_window():
    sources = make_sources()
    before = prepare_sequences(sources)
    future = sources[0]["data"]["timestamp_utc"] >= "2026-01-02T00:00:00Z"
    sources[0]["data"].loc[future, ["building_load_kw", "building_load_kw_scaled"]] = 99999
    after = prepare_sequences(sources)
    np.testing.assert_array_equal(before["arrays"]["X"][0], after["arrays"]["X"][0])
    assert not np.array_equal(before["arrays"]["y"][0], after["arrays"]["y"][0])


@pytest.mark.parametrize("partition", ["validation", "test"])
def test_heldout_windows_stay_in_their_split(partition):
    result = prepare_sequences(make_sources(partition=partition))
    bounds = result["metadata"]["boundaries"]
    arrays = result["arrays"]
    assert arrays["input_timestamps"].min() >= np.datetime64(bounds["start_utc"].replace("Z", ""))
    assert arrays["target_timestamps"].max() < np.datetime64(bounds["end_exclusive_utc"].replace("Z", ""))


def test_mixed_splits_and_out_of_bounds_rows_rejected():
    sources = make_sources()
    sources[1]["metadata"]["split"]["name"] = "test"
    with pytest.raises(SequenceDataError, match="Do not mix"):
        prepare_sequences(sources)
    sources = make_sources()
    sources[0]["metadata"]["split"]["boundaries"] = {"start_utc": "2026-01-02T00:00:00Z", "end_exclusive_utc": "2026-01-04T00:00:00Z"}
    with pytest.raises(SequenceDataError):
        prepare_sequences(sources)


def test_node_order_is_independent_of_input_source_row_order():
    sources = make_sources()
    before = prepare_sequences(sources)
    for source in sources:
        source["data"] = source["data"].iloc[::-1]
    after = prepare_sequences(list(reversed(sources)))
    np.testing.assert_array_equal(before["arrays"]["X"], after["arrays"]["X"])
    assert before["metadata"]["node_order"] == after["metadata"]["node_order"]


def test_configurable_windows_stride_and_reordered_nodes():
    config = load_forecasting_config()
    config.update(lookback_hours=12, horizon_hours=6, stride_hours=3)
    config["graph"]["nodes"] = list(reversed(config["graph"]["nodes"]))
    result = prepare_sequences(make_sources(), config)
    assert result["arrays"]["X"].shape == (15, 12, 8, 1)
    assert result["arrays"]["y"].shape == (15, 6, 8, 1)
    assert result["metadata"]["node_order"][0] == "BUS01"
    assert not result["arrays"]["target_mask"][0, 0]


@pytest.mark.parametrize("problem", ["missing_node", "missing_value", "duplicate", "gap", "wrong_unit", "unfitted", "wrong_generation"])
def test_invalid_observations_rejected(problem):
    sources = make_sources()
    if problem == "missing_node":
        sources[0]["data"] = sources[0]["data"].query("building_id != 'B01'")
    elif problem == "missing_value":
        sources[0]["data"].loc[0, "building_load_kw_scaled"] = np.nan
    elif problem == "duplicate":
        sources[0]["data"] = pd.concat([sources[0]["data"], sources[0]["data"].iloc[[0]]])
    elif problem == "gap":
        sources[0]["data"] = sources[0]["data"].drop(index=5)
    elif problem == "wrong_unit":
        sources[0]["units"]["building_load_kw"] = "kWh"
    elif problem == "unfitted":
        sources[0]["metadata"]["split"]["fitted_on"] = "test"
    else:
        sources[1]["data"]["generation_type"] = "wind"
    with pytest.raises(SequenceDataError):
        prepare_sequences(sources)


def test_future_residual_inputs_and_short_partitions_rejected():
    config = load_forecasting_config()
    config["graph"]["nodes"][0]["input_column"] = "forecast_error_kw"
    with pytest.raises(SequenceDataError, match="future information"):
        prepare_sequences(make_sources(), config)
    with pytest.raises(SequenceDataError, match="shorter"):
        prepare_sequences(make_sources(hours=40))


def test_input_sources_and_configuration_remain_unchanged():
    sources, config = make_sources(), load_forecasting_config()
    before_sources, before_config = deepcopy(sources), deepcopy(config)
    prepare_sequences(sources, config)
    for before, after in zip(before_sources, sources):
        pd.testing.assert_frame_equal(before["data"], after["data"])
        assert before["metadata"] == after["metadata"]
    assert config == before_config


def test_saved_numpy_bundle_loads_without_pickle_and_preserves_split_hashes(tmp_path):
    raw_sources = make_sources(hours=360)
    for source in raw_sources:
        source["data"] = source["data"].drop(columns=[column for column in source["data"] if column.endswith("_scaled")])
        del source["metadata"]["split"]
    splits = save_splits(split_datasets(raw_sources), "fixture", processed_root=tmp_path)
    boundaries = json.loads((splits / "preprocessing.json").read_text())["boundaries"]
    assert [boundaries[name]["hours"] for name in ("train", "validation", "test")] == [252, 54, 54]
    assert len(load_partition_sources(splits, "train")) == 4
    destination = save_forecasting_bundle(splits, "fixture", processed_root=tmp_path)
    assert verify_split_bundle(splits)
    with np.load(destination / "graph.npz", allow_pickle=False) as graph:
        assert graph["edge_index"].shape == (2, 14)
    with np.load(destination / "test.npz", allow_pickle=False) as arrays:
        assert arrays["X"].shape == (7, 24, 8, 1)
    metadata = json.loads((destination / "test.metadata.json").read_text())
    assert all(item["synthetic"] for item in metadata["source_metadata"].values())
    with pytest.raises(SequenceDataError, match="Refusing existing"):
        save_forecasting_bundle(splits, "fixture", processed_root=tmp_path)
