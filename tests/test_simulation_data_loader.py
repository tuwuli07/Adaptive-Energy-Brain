"""Read-only physical data extraction from synthetic unit-test fixtures."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from src.data.schema import SchemaValidationError, validate_dataset
from src.data.splitting import save_splits, split_datasets
from src.simulation.data_loader import (
    SimulationDataError,
    load_simulation_data,
    load_simulation_split,
    prepare_simulation_data,
)


def make_sources(hours=4):
    timestamps = pd.date_range("2026-01-01", periods=hours, freq="h", tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    scenario = {"buildings": [{"id": "B01"}, {"id": "B02"}],
                "solar": {"asset_id": "PV01"}, "wind": {"asset_id": "W01"}}
    result = []

    def add(identifier, kind, frame, units):
        result.append({"source_id": identifier, "record_type": kind,
                       "measurement_kind": "interval_mean", "data": frame,
                       "units": units, "metadata": {"source_id": identifier,
                       "origin": "simulated", "synthetic": True, "currency_code": "USD",
                       "source_metadata": {"scenario": deepcopy(scenario), "note": "Unit-test fixture only"}}})

    loads = [pd.DataFrame({"timestamp_utc": timestamps, "interval_hours": 1,
                           "building_id": building, "building_load_kw": np.arange(hours) + base})
             for building, base in (("B01", 10), ("B02", 20))]
    add("fixture_demand", "demand", pd.concat(loads, ignore_index=True),
        {"interval_hours": "h", "building_load_kw": "kW"})
    for asset, kind, power in (("PV01", "solar", 3), ("W01", "wind", 4)):
        add("fixture_" + kind, "generation", pd.DataFrame({"timestamp_utc": timestamps,
            "interval_hours": 1, "asset_id": asset, "generation_type": kind,
            "available_generation_kw": power}), {"interval_hours": "h", "available_generation_kw": "kW"})
    add("fixture_prices", "price", pd.DataFrame({"timestamp_utc": timestamps,
        "interval_hours": 1, "grid_id": "GRID01", "import_price": 0.1, "export_price": 0.03}),
        {"interval_hours": "h", "import_price": "USD/kWh", "export_price": "USD/kWh"})
    return result


def test_valid_inputs_and_building_aggregation():
    data = prepare_simulation_data(make_sources())
    assert len(data) == 4
    observation = data[0]
    assert observation.timestamp_utc == "2026-01-01T00:00:00Z"
    assert dict(observation.building_loads_kw) == {"B01": 10, "B02": 20}
    assert observation.campus_load_kw == 30
    assert observation.pv_available_kw == 3
    assert observation.wind_available_kw == 4
    assert observation.import_price == 0.1
    assert observation.export_price == 0.03
    assert data.metadata["currency_code"] == "USD"


def test_unordered_records_return_chronological_observations():
    sources = make_sources()
    for source in sources:
        source["data"] = source["data"].iloc[::-1]
    data = prepare_simulation_data(list(reversed(sources)))
    assert [row.timestamp_utc for row in data] == sorted(row.timestamp_utc for row in data)
    assert [row.campus_load_kw for row in data] == [30, 32, 34, 36]


def test_interval_selection_is_continuous_and_end_exclusive():
    data = prepare_simulation_data(make_sources(), start="2026-01-01T01:00:00Z", end="2026-01-01T03:00:00Z")
    assert len(data) == 2
    assert data[0].campus_load_kw == 32
    assert data[-1].timestamp_utc == "2026-01-01T02:00:00Z"


@pytest.mark.parametrize("kind,column", [
    ("demand", "building_load_kw"), ("generation", "available_generation_kw"),
    ("price", "import_price"), ("demand", "timestamp_utc"),
])
def test_missing_required_columns_reported(kind, column):
    sources = make_sources()
    source = next(source for source in sources if source["record_type"] == kind)
    source["data"] = source["data"].drop(columns=[column])
    with pytest.raises(SimulationDataError, match=column):
        prepare_simulation_data(sources)


def test_absent_source_and_missing_mandatory_values_fail():
    with pytest.raises(SimulationDataError, match="sources are required"):
        prepare_simulation_data(make_sources()[:-1])
    sources = make_sources()
    sources[0]["data"]["building_load_kw"] = np.nan
    with pytest.raises(SimulationDataError, match="building_load_kw"):
        prepare_simulation_data(sources)


def test_absent_export_price_is_unavailable_not_zero():
    sources = make_sources()
    prices = sources[-1]
    prices["data"] = prices["data"].drop(columns=["export_price"])
    prices["units"].pop("export_price")
    data = prepare_simulation_data(sources)
    assert all(row.export_price is None for row in data)
    assert data.metadata["unavailable_export_price_hours"] == 4
    # Default canonical schema remains strict outside the explicit loader opt-in.
    with pytest.raises(SchemaValidationError, match="export_price"):
        validate_dataset(prices["data"].to_dict("records"), "price",
                         units=prices["units"], metadata=prices["metadata"])


def test_partial_export_price_missingness_preserved():
    sources = make_sources()
    sources[-1]["data"].loc[1, "export_price"] = np.nan
    data = prepare_simulation_data(sources)
    assert data[0].export_price == 0.03
    assert data[1].export_price is None


def test_compatible_power_energy_and_price_units_converted_with_audit():
    sources = make_sources()
    demand = sources[0]
    demand["data"]["building_load_kw"] *= 1000
    demand["data"]["demand_energy_kwh"] = demand["data"]["building_load_kw"]
    demand["units"].update(building_load_kw="W", demand_energy_kwh="Wh")
    prices = sources[-1]
    prices["data"][["import_price", "export_price"]] *= 1000
    prices["units"].update(import_price="USD/MWh", export_price="USD/MWh")
    before = deepcopy(sources)
    data = prepare_simulation_data(sources)
    assert data[0].campus_load_kw == 30
    assert data[0].import_price == pytest.approx(0.1)
    assert data[0].export_price == pytest.approx(0.03)
    assert len(data.metadata["sources"]["fixture_demand"]["unit_conversions"]) == 2
    for original, after in zip(before, sources):
        pd.testing.assert_frame_equal(original["data"], after["data"])


@pytest.mark.parametrize("unit", ["kWh", "unknown", "USD/kWh"])
def test_incompatible_power_units_rejected(unit):
    sources = make_sources()
    sources[0]["units"]["building_load_kw"] = unit
    with pytest.raises(SimulationDataError, match="Unsupported unit conversion"):
        prepare_simulation_data(sources)


def test_currency_is_not_implicitly_converted():
    sources = make_sources()
    sources[-1]["units"]["import_price"] = "EUR/kWh"
    with pytest.raises(SimulationDataError, match="currency conversions are never inferred"):
        prepare_simulation_data(sources)


def test_duplicate_and_aggregate_demand_rows_not_counted_twice():
    sources = make_sources()
    sources[0]["data"] = pd.concat([sources[0]["data"], sources[0]["data"].iloc[[0]]], ignore_index=True)
    with pytest.raises(SimulationDataError, match="duplicate record key"):
        prepare_simulation_data(sources)
    sources = make_sources()
    aggregate = sources[0]["data"].iloc[:4].copy()
    aggregate["building_id"] = "CAMPUS_TOTAL"
    sources[0]["data"] = pd.concat([sources[0]["data"], aggregate])
    with pytest.raises(SimulationDataError, match="Unregistered building/aggregate"):
        prepare_simulation_data(sources)


def test_duplicate_across_sources_rejected():
    sources = make_sources()
    extra = deepcopy(sources[0])
    extra["source_id"] = extra["metadata"]["source_id"] = "other_load_meter"
    with pytest.raises(SimulationDataError, match="Duplicate timestamp/building_id"):
        prepare_simulation_data(sources + [extra])


def test_discontinuous_and_incomplete_entity_coverage_reported():
    sources = make_sources()
    sources[0]["data"] = sources[0]["data"].drop(index=1)
    with pytest.raises(SimulationDataError, match="Missing/discontinuous demand interval"):
        prepare_simulation_data(sources)
    # A continuous subinterval remains usable; no imputation of the earlier gap.
    assert len(prepare_simulation_data(sources, start="2026-01-01T02:00:00Z")) == 2


@pytest.mark.parametrize("start,end", [
    ("not-a-date", None), ("2026-01-01T00:30:00Z", None),
    ("2026-01-01T00:00:00", None), ("2026-01-01T06:00:00+06:00", None),
    ("2026-01-01T02:00:00Z", "2026-01-01T01:00:00Z"),
    ("2025-12-31T23:00:00Z", None),
])
def test_invalid_selection_reported(start, end):
    with pytest.raises(SimulationDataError):
        prepare_simulation_data(make_sources(), start=start, end=end)


def test_registry_required_if_not_documented_and_can_be_supplied_explicitly():
    sources = make_sources()
    for source in sources:
        source["metadata"]["source_metadata"] = {"note": "Unspecified campus registry"}
    with pytest.raises(SimulationDataError, match="explicit buildings entity registry"):
        prepare_simulation_data(sources)
    assert len(prepare_simulation_data(sources, building_ids=["B01", "B02"],
                                     solar_asset_ids=["PV01"], wind_asset_ids=["W01"])) == 4


def test_multiple_registered_solar_assets_are_summed_once():
    sources = make_sources()
    second = deepcopy(sources[1])
    second["source_id"] = second["metadata"]["source_id"] = "second_pv"
    second["data"]["asset_id"] = "PV02"
    data = prepare_simulation_data(sources + [second], solar_asset_ids=["PV01", "PV02"])
    assert data[0].pv_available_kw == 6
    assert dict(data[0].pv_asset_powers_kw) == {"PV01": 3, "PV02": 3}


def test_origin_labels_and_metadata_are_preserved_defensively():
    sources = make_sources()
    sources[0]["metadata"].update(origin="measured", synthetic=False)
    sources[1]["metadata"].update(origin="derived", derived_kind="model", synthetic=False)
    data = prepare_simulation_data(sources)
    assert data.metadata["source_origin_summary"] == {"fixture_demand": "measured", "fixture_solar": "modeled", "fixture_wind": "synthetic", "fixture_prices": "synthetic"}
    metadata = data.metadata
    metadata["sources"]["fixture_wind"]["metadata"]["synthetic"] = False
    assert data.metadata["sources"]["fixture_wind"]["metadata"]["synthetic"] is True
    with pytest.raises(TypeError):
        data[0].building_loads_kw["B01"] = 0
    with pytest.raises(FrozenInstanceError):
        data[0].campus_load_kw = 0


def test_manifest_load_preserves_file_bytes_and_checks_recorded_hash(tmp_path):
    sources = make_sources()
    manifest = {"datasets": []}
    snapshots = {}
    for source in sources:
        path = tmp_path / (source["source_id"] + ".csv")
        source["data"].to_csv(path, index=False)
        snapshots[path.name] = path.read_bytes()
        entry = {key: value for key, value in source.items() if key != "data"}
        entry.update(path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        manifest["datasets"].append(entry)
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest))
    assert len(load_simulation_data(manifest_path)) == 4
    for name, before in snapshots.items():
        assert (tmp_path / name).read_bytes() == before
    (tmp_path / next(iter(snapshots))).write_bytes(b"changed fixture")
    with pytest.raises(SimulationDataError, match="checksum mismatch"):
        load_simulation_data(manifest_path)


def test_split_loader_uses_only_physical_values_and_respects_boundaries(tmp_path):
    sources = make_sources(hours=20)
    path = save_splits(split_datasets(sources), "loader_test", processed_root=tmp_path)
    data = load_simulation_split(path, "train")
    assert len(data) == 14
    assert data[0].campus_load_kw == 30
    assert data.metadata["sources"]["fixture_demand"]["excluded_learning_columns"] == ["building_load_kw_scaled"]
    with pytest.raises(SimulationDataError, match="crosses its temporal split"):
        load_simulation_split(path, "train", end="2026-01-01T15:00:00Z")


def test_missing_manifest_is_useful_error(tmp_path):
    with pytest.raises(SimulationDataError, match="Cannot load processed datasets"):
        load_simulation_data(tmp_path / "missing.yaml")
