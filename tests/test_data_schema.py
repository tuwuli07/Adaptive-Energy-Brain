"""Offline canonical-schema tests; sample records are explicitly synthetic."""

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from src.data.schema import (
    SchemaValidationError,
    load_schema,
    validate_dataset,
    validate_record,
)


UTC = "2026-01-01T00:00:00Z"
METADATA = {
    "source_id": "synthetic_test",
    "origin": "simulated",
    "source_metadata": {"license": "test fixture", "original_columns": ["load"]},
}
SAMPLES = {
    "demand": (
        {"timestamp_utc": UTC, "interval_hours": 1, "building_id": "B01",
         "building_load_kw": 10.0},
        {"interval_hours": "h", "building_load_kw": "kW"},
    ),
    "generation": (
        {"timestamp_utc": UTC, "interval_hours": 1, "asset_id": "PV01",
         "generation_type": "solar", "available_generation_kw": 5.0},
        {"interval_hours": "h", "available_generation_kw": "kW"},
    ),
    "weather": (
        {"timestamp_utc": UTC, "interval_hours": 1, "site_id": "SITE01",
         "air_temperature_c": 25.0},
        {"interval_hours": "h", "air_temperature_c": "degC"},
    ),
    "forecast": (
        {"timestamp_utc": "2026-01-01T01:00:00Z", "interval_hours": 1,
         "forecast_issue_time_utc": UTC, "forecast_target_id": "B01",
         "forecast_target_kind": "building_demand", "predicted_power_kw": 9.0},
        {"interval_hours": "h", "predicted_power_kw": "kW"},
    ),
    "price": (
        {"timestamp_utc": UTC, "interval_hours": 1, "grid_id": "GRID01",
         "import_price": 10.0, "export_price": 3.0},
        {"interval_hours": "h", "import_price": "BDT/kWh", "export_price": "BDT/kWh"},
    ),
    "simulation_state": (
        {"timestamp_utc": UTC, "asset_id": "BAT01",
         "battery_energy_kwh": 50.0, "battery_soc": 0.5},
        {"battery_energy_kwh": "kWh", "battery_soc": "1"},
    ),
}


def sample(kind="demand"):
    row, units = deepcopy(SAMPLES[kind])
    metadata = deepcopy(METADATA)
    if kind in {"forecast", "simulation_state"}:
        metadata.update(origin="derived", derived_kind="model" if kind == "forecast" else "simulator")
    if kind == "price":
        metadata["currency_code"] = "BDT"
    return row, units, metadata


@pytest.mark.parametrize("kind", SAMPLES)
def test_valid_record_types(kind):
    row, units, metadata = sample(kind)
    result = validate_record(row, kind, units=units, metadata=metadata)
    assert result["records"] == [row]
    assert result["units"] == units
    assert result["metadata"] == metadata


def test_yaml_covers_required_record_types_and_describes_all_columns():
    schema = load_schema()
    assert set(schema["record_types"]) == set(SAMPLES)
    for definition in schema["columns"].values():
        assert definition["description"]
        if definition["type"] == "number":
            assert definition["unit"]


@pytest.mark.parametrize("kind", SAMPLES)
def test_missing_required_column(kind):
    row, units, metadata = sample(kind)
    del row["timestamp_utc"]
    with pytest.raises(SchemaValidationError, match="missing required columns.*timestamp_utc"):
        validate_record(row, kind, units=units, metadata=metadata)


@pytest.mark.parametrize("timestamp", [
    "2026-01-01", "2026-01-01T00:00:00", "2026-01-01T06:00:00+06:00",
    "2026-02-30T00:00:00Z", "not-a-date", "2026-01-01T00:30:00Z",
    "2026-01-01T00:00:01Z", "2026-01-01T00:00:00.5Z", None,
])
def test_invalid_or_missing_timestamps(timestamp):
    row, units, metadata = sample()
    row["timestamp_utc"] = timestamp
    with pytest.raises(SchemaValidationError, match="timestamp_utc"):
        validate_record(row, "demand", units=units, metadata=metadata)


@pytest.mark.parametrize("unit", ["kWh", "W", "MW", "kw", "unsupported"])
def test_power_units_are_not_silently_converted(unit):
    row, units, metadata = sample()
    units["building_load_kw"] = unit
    with pytest.raises(SchemaValidationError, match="building_load_kw: incompatible unit"):
        validate_record(row, "demand", units=units, metadata=metadata)


def test_energy_does_not_accept_power_unit():
    row, units, metadata = sample("simulation_state")
    units["battery_energy_kwh"] = "kW"
    with pytest.raises(SchemaValidationError, match="battery_energy_kwh: incompatible unit"):
        validate_record(row, "simulation_state", units=units, metadata=metadata)


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -1, True, "10"])
def test_invalid_required_numbers(value):
    row, units, metadata = sample()
    row["building_load_kw"] = value
    with pytest.raises(SchemaValidationError, match="building_load_kw"):
        validate_record(row, "demand", units=units, metadata=metadata)


def test_optional_values_may_be_null_but_not_nan():
    row, units, metadata = sample()
    units["occupancy_fraction"] = "1"
    row["occupancy_fraction"] = None
    assert validate_record(row, "demand", units=units, metadata=metadata)
    row["occupancy_fraction"] = float("nan")
    with pytest.raises(SchemaValidationError, match="finite number"):
        validate_record(row, "demand", units=units, metadata=metadata)


def test_energy_power_relationship_is_checked_without_conversion():
    row, units, metadata = sample()
    row["demand_energy_kwh"] = 10.0
    units["demand_energy_kwh"] = "kWh"
    assert validate_record(row, "demand", units=units, metadata=metadata)
    row["demand_energy_kwh"] = 10000
    with pytest.raises(SchemaValidationError, match="kW and kWh are distinct"):
        validate_record(row, "demand", units=units, metadata=metadata)


def test_multiple_buildings_and_duplicate_utc_keys():
    row, units, metadata = sample()
    second = {**row, "building_id": "B02"}
    result = validate_dataset([row, second], "demand", units=units, metadata=metadata)
    assert len(result["records"]) == 2
    duplicate = {**row, "timestamp_utc": "2026-01-01T00:00:00+00:00"}
    with pytest.raises(SchemaValidationError, match="duplicate record key"):
        validate_dataset([row, duplicate], "demand", units=units, metadata=metadata)


def test_preserves_source_metadata_without_mutating_inputs():
    row, units, metadata = sample()
    before = deepcopy((row, units, metadata))
    result = validate_record(row, "demand", units=units, metadata=metadata)
    result["records"][0]["building_load_kw"] = 999
    result["metadata"]["source_metadata"]["original_columns"].append("changed")
    assert (row, units, metadata) == before


def test_unknown_columns_rejected_with_metadata_guidance():
    row, units, metadata = sample()
    row["source_sensor_name"] = "original header"
    with pytest.raises(SchemaValidationError, match="metadata.source_metadata"):
        validate_record(row, "demand", units=units, metadata=metadata)


@pytest.mark.parametrize("kind,column,value,unit", [
    ("simulation_state", "battery_soc", 50, "1"),
    ("weather", "air_temperature_c", -300, "degC"),
    ("weather", "relative_humidity_fraction", 80, "1"),
    ("weather", "wind_speed_m_s", -1, "m/s"),
    ("generation", "generation_capacity_kw", 0, "kW"),
    ("demand", "interval_hours", 0.5, "h"),
])
def test_range_and_resolution_validation(kind, column, value, unit):
    row, units, metadata = sample(kind)
    row[column] = value
    units[column] = unit
    with pytest.raises(SchemaValidationError, match=column):
        validate_record(row, kind, units=units, metadata=metadata)


def test_weather_needs_a_non_null_observation():
    row, units, metadata = sample("weather")
    row["air_temperature_c"] = None
    with pytest.raises(SchemaValidationError, match="weather observation"):
        validate_record(row, "weather", units=units, metadata=metadata)


def test_generation_rating_and_technology():
    row, units, metadata = sample("generation")
    row["generation_type"] = "wind"
    assert validate_record(row, "generation", units=units, metadata=metadata)
    row["generation_type"] = "coal"
    with pytest.raises(SchemaValidationError, match="generation_type"):
        validate_record(row, "generation", units=units, metadata=metadata)
    row["generation_type"] = "solar"
    row["generation_capacity_kw"] = 1
    units["generation_capacity_kw"] = "kW"
    with pytest.raises(SchemaValidationError, match="exceeds generation_capacity"):
        validate_record(row, "generation", units=units, metadata=metadata)


def test_forecast_causality_and_bounds():
    row, units, metadata = sample("forecast")
    row["forecast_issue_time_utc"] = "2026-01-01T02:00:00Z"
    with pytest.raises(SchemaValidationError, match="cannot follow"):
        validate_record(row, "forecast", units=units, metadata=metadata)
    row["forecast_issue_time_utc"] = UTC
    row["lower_power_kw"] = 20
    units["lower_power_kw"] = "kW"
    with pytest.raises(SchemaValidationError, match="lower_power_kw exceeds"):
        validate_record(row, "forecast", units=units, metadata=metadata)


def test_battery_state_consistency_and_flow_exclusivity():
    row, units, metadata = sample("simulation_state")
    row["battery_capacity_kwh"] = 100
    units["battery_capacity_kwh"] = "kWh"
    assert validate_record(row, "simulation_state", units=units, metadata=metadata)
    row["battery_capacity_kwh"] = 200
    with pytest.raises(SchemaValidationError, match="must equal battery_soc"):
        validate_record(row, "simulation_state", units=units, metadata=metadata)
    row["battery_capacity_kwh"] = 100
    row.update(battery_charge_kw=1, battery_discharge_kw=1, interval_hours=1)
    units.update(battery_charge_kw="kW", battery_discharge_kw="kW", interval_hours="h")
    with pytest.raises(SchemaValidationError, match="simultaneously"):
        validate_record(row, "simulation_state", units=units, metadata=metadata)


def test_explicit_currency_and_no_fx_conversion():
    row, units, metadata = sample("price")
    del metadata["currency_code"]
    with pytest.raises(SchemaValidationError, match="explicit currency_code"):
        validate_record(row, "price", units=units, metadata=metadata)
    assert validate_record(row, "price", units=units, metadata=metadata, currency_code="BDT")
    with pytest.raises(SchemaValidationError, match="incompatible unit"):
        validate_record(row, "price", units=units, metadata=metadata, currency_code="USD")
    metadata["currency_code"] = "BDT"
    with pytest.raises(SchemaValidationError, match="Conflicting currency"):
        validate_record(row, "price", units=units, metadata=metadata, currency_code="USD")
    row["export_price"] = 11
    with pytest.raises(SchemaValidationError, match="export_price must not exceed"):
        validate_record(row, "price", units=units, metadata=metadata)


def test_currency_can_be_configured_in_yaml(tmp_path):
    schema = load_schema()
    schema["currency_code"] = "USD"
    path = tmp_path / "schema.yaml"
    path.write_text(yaml.safe_dump(schema), encoding="utf-8")
    row, units, metadata = sample("price")
    del metadata["currency_code"]
    units.update(import_price="USD/kWh", export_price="USD/kWh")
    result = validate_record(row, "price", units=units, metadata=metadata, schema_path=path)
    assert result["metadata"]["currency_code"] == "USD"


@pytest.mark.parametrize("change", ["missing_unit", "extra_unit", "missing_metadata", "wrong_origin"])
def test_invalid_units_and_metadata(change):
    row, units, metadata = sample()
    if change == "missing_unit":
        del units["building_load_kw"]
    elif change == "extra_unit":
        units["unknown"] = "kW"
    elif change == "missing_metadata":
        del metadata["source_metadata"]
    else:
        metadata["origin"] = "assumed_measured"
    with pytest.raises(SchemaValidationError):
        validate_record(row, "demand", units=units, metadata=metadata)


def test_forecasts_require_derived_provenance():
    row, units, metadata = sample("forecast")
    metadata["origin"] = "measured"
    with pytest.raises(SchemaValidationError, match="origin: derived"):
        validate_record(row, "forecast", units=units, metadata=metadata)


def test_empty_dataset_unknown_type_and_missing_schema():
    row, units, metadata = sample()
    with pytest.raises(SchemaValidationError, match="at least one record"):
        validate_dataset([], "demand", units=units, metadata=metadata)
    with pytest.raises(SchemaValidationError, match="Unknown record type"):
        validate_record(row, "unknown", units=units, metadata=metadata)
    with pytest.raises(SchemaValidationError, match="Cannot read data schema"):
        load_schema(Path(__file__).parent / "missing_schema.yaml")
