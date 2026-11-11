"""Chronological alignment, train-only fitting and protected split artifacts."""

from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from src.data.splitting import (
    SplittingError,
    apply_scaler,
    build_windows,
    load_split_config,
    save_splits,
    split_datasets,
    verify_split_bundle,
)


@pytest.fixture
def source():
    times = pd.date_range("2026-01-01", periods=20, freq="h", tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    frames = [pd.DataFrame({"timestamp_utc": times, "interval_hours": 1,
                            "building_id": building, "building_load_kw": np.arange(1, 21) * factor})
              for building, factor in (("B01", 1), ("B02", 10))]
    return {"source_id": "test_demand", "record_type": "demand", "data": pd.concat(frames, ignore_index=True),
            "units": {"interval_hours": "h", "building_load_kw": "kW"},
            "metadata": {"source_id": "test_demand", "origin": "simulated", "synthetic": True,
                         "source_metadata": {"note": "Synthetic unit-test calculation fixture"}}}


def test_default_ratios_boundaries_no_overlap_and_building_alignment(source):
    result = split_datasets([source])
    previous_end = None
    times_by_split = {}
    for partition, count in (("train", 14), ("validation", 3), ("test", 3)):
        payload = result["partitions"][partition][0]
        frame = payload["data"]
        assert len(frame) == 2 * count
        assert frame.groupby("building_id")["timestamp_utc"].apply(list).iloc[0] == frame.groupby("building_id")["timestamp_utc"].apply(list).iloc[1]
        times = set(frame["timestamp_utc"])
        assert previous_end is None or previous_end < min(times)
        previous_end = max(times)
        times_by_split[partition] = times
        assert result["preprocessing"]["boundaries"][partition]["hours"] == count
    assert not times_by_split["train"] & times_by_split["validation"]
    assert not times_by_split["validation"] & times_by_split["test"]
    assert not times_by_split["train"] & times_by_split["test"]


def test_scaler_uses_training_only_and_is_per_building(source):
    result = split_datasets([source])
    scaler = result["preprocessing"]["scalers"]["test_demand"]
    for entity, multiplier in (("B01", 1), ("B02", 10)):
        fitted = scaler["entities"][entity]["building_load_kw"]
        assert fitted["count"] == 14
        assert fitted["mean"] == pytest.approx(np.arange(1, 15).mean() * multiplier)
        assert fitted["scale"] == pytest.approx(np.arange(1, 15).std(ddof=0) * multiplier)
    train = result["partitions"]["train"][0]["data"]
    assert train.groupby("building_id")["building_load_kw_scaled"].mean().abs().max() < 1e-12


def test_future_extremes_do_not_change_fitted_parameters(source):
    original = split_datasets([source])
    source["data"].loc[source["data"]["timestamp_utc"] >= "2026-01-01T14:00:00Z", "building_load_kw"] = 1_000_000
    altered = split_datasets([source])
    assert original["preprocessing"]["scalers"] == altered["preprocessing"]["scalers"]
    assert altered["partitions"]["test"][0]["data"]["building_load_kw_scaled"].max() > 1000


def test_saved_parameters_apply_without_refitting(source):
    result = split_datasets([source])
    parameters = json.loads(json.dumps(result["preprocessing"]["scalers"]["test_demand"]))
    frame = result["partitions"]["validation"][0]["data"]
    unscaled = frame.drop(columns=["building_load_kw_scaled"])
    pd.testing.assert_frame_equal(apply_scaler(unscaled, parameters), frame)


def test_configurable_proportions(source):
    config = load_split_config()
    config["fractions"] = {"train": 0.5, "validation": 0.25, "test": 0.25}
    result = split_datasets([source], config)
    assert [result["preprocessing"]["boundaries"][name]["hours"] for name in ("train", "validation", "test")] == [10, 5, 5]


def test_unsorted_input_is_sorted_without_mutation(source):
    source["data"] = source["data"].iloc[::-1].reset_index(drop=True)
    before = deepcopy(source)
    result = split_datasets([source])
    pd.testing.assert_frame_equal(source["data"], before["data"])
    assert source["metadata"] == before["metadata"]
    payload = result["partitions"]["train"][0]
    assert payload["metadata"]["source_metadata"] == before["metadata"]["source_metadata"]
    assert payload["metadata"]["synthetic"] is True
    assert "building_load_kw" in payload["data"]
    assert payload["units"]["building_load_kw"] == "kW"
    assert payload["metadata"]["split"]["scaled_units"]["building_load_kw_scaled"] == "1"


def test_multiple_sources_use_same_boundaries(source):
    other = deepcopy(source)
    other["source_id"] = "other_demand"
    other["metadata"]["source_id"] = "other_demand"
    other["data"]["building_id"] = other["data"]["building_id"].replace({"B01": "B03", "B02": "B04"})
    result = split_datasets([source, other])
    for partition in result["partitions"].values():
        assert set(partition[0]["data"]["timestamp_utc"]) == set(partition[1]["data"]["timestamp_utc"])


def test_constant_training_feature_uses_unit_scale(source):
    source["data"]["building_load_kw"] = 5
    result = split_datasets([source])
    for entity in result["preprocessing"]["scalers"]["test_demand"]["entities"].values():
        assert entity["building_load_kw"]["scale"] == 1
        assert entity["building_load_kw"]["constant"]
    assert (result["partitions"]["test"][0]["data"]["building_load_kw_scaled"] == 0).all()


@pytest.mark.parametrize("fractions", [
    {"train": 0.8, "validation": 0.15, "test": 0.15},
    {"train": 0, "validation": 0.5, "test": 0.5},
    {"train": float("nan"), "validation": 0.15, "test": 0.15},
])
def test_invalid_fractions_rejected(source, fractions):
    config = load_split_config()
    config["fractions"] = fractions
    with pytest.raises(SplittingError, match="fractions"):
        split_datasets([source], config)


def test_missing_entity_interval_and_duplicate_records_rejected(source):
    missing = deepcopy(source)
    missing["data"] = missing["data"].drop(index=5)
    with pytest.raises(SplittingError, match="missing entity intervals"):
        split_datasets([missing])
    source["data"] = pd.concat([source["data"], source["data"].iloc[[0]]], ignore_index=True)
    with pytest.raises(SplittingError, match="duplicate"):
        split_datasets([source])


def test_misaligned_entity_coverage_rejected(source):
    source["data"] = source["data"].drop(index=20)
    with pytest.raises(SplittingError, match="identical hourly timeline"):
        split_datasets([source])


def test_insufficient_samples_rejected(source):
    source["data"] = source["data"].groupby("building_id").head(2)
    with pytest.raises(SplittingError, match="nonempty partitions"):
        split_datasets([source])


def test_known_presplit_interpolation_is_rejected(source):
    source["report"] = {"operations": [{"operation": "fill_interpolate", "uses_future_endpoint": True}]}
    with pytest.raises(SplittingError, match="Pre-split"):
        split_datasets([source])


def test_unknown_inference_entity_and_missing_values_rejected(source):
    result = split_datasets([source])
    scaler = result["preprocessing"]["scalers"]["test_demand"]
    frame = source["data"].copy()
    frame["building_id"] = "new_entity"
    with pytest.raises(SplittingError, match="no training-fitted scaler"):
        apply_scaler(frame, scaler)
    frame = source["data"].copy()
    frame.loc[0, "building_load_kw"] = float("nan")
    with pytest.raises(SplittingError, match="no imputation"):
        apply_scaler(frame, scaler)


def test_windows_are_historical_and_stay_within_each_partition(source):
    result = split_datasets([source])
    for name, payloads in result["partitions"].items():
        windows = build_windows(payloads[0], features=["building_load_kw_scaled"],
                                target="building_load_kw", lookback=1, horizon=2)
        assert windows
        bounds = result["preprocessing"]["boundaries"][name]
        for window in windows:
            assert max(window["input_timestamps"]) < min(window["target_timestamps"])
            assert min(window["input_timestamps"]) >= bounds["start_utc"]
            assert max(window["target_timestamps"]) < bounds["end_exclusive_utc"]


def test_unsplit_windows_and_boundary_contamination_rejected(source):
    with pytest.raises(SplittingError, match="labeled partition"):
        build_windows(source, features=["building_load_kw"], target="building_load_kw", lookback=1, horizon=1)
    result = split_datasets([source])
    partition = result["partitions"]["train"][0]
    partition["data"].loc[0, "timestamp_utc"] = "2026-01-02T00:00:00Z"
    with pytest.raises(SplittingError, match="out-of-boundary"):
        build_windows(partition, features=["building_load_kw"], target="building_load_kw", lookback=1, horizon=1)


def test_saved_bundle_no_overwrite_and_test_tampering_detected(source, tmp_path):
    result = split_datasets([source])
    path = save_splits(result, "unit_test", processed_root=tmp_path)
    assert verify_split_bundle(path)
    parameters = json.loads((path / "preprocessing.json").read_text())
    assert parameters["fitted_on"] == "train"
    test_file = path / "test/test_demand.csv"
    original = test_file.read_bytes()
    with pytest.raises(SplittingError, match="Refusing existing"):
        save_splits(result, "unit_test", processed_root=tmp_path)
    assert test_file.read_bytes() == original
    test_file.write_bytes(original + b"\n")
    assert verify_split_bundle(path, partitions=("train", "validation"))
    with pytest.raises(SplittingError, match="missing/changed"):
        verify_split_bundle(path)
