"""Synthetic offline examples for conservative, auditable cleaning."""

from copy import deepcopy
import hashlib
import json

import pandas as pd
import pytest

from src.data.cleaning import (
    CleaningError,
    clean_dataframe,
    load_and_clean_csv,
    load_cleaning_config,
)


@pytest.fixture
def metadata():
    return {"source_id": "test_only", "origin": "simulated",
            "source_metadata": {"note": "Synthetic test rows; no real campus data"}}


def demand(values, start="2026-01-01T00:00:00Z", building="B01"):
    return pd.DataFrame({
        "timestamp_utc": pd.date_range(start, periods=len(values), freq="h").astype(str),
        "building_id": building, "interval_hours": 1, "building_load_kw": values,
    })


def clean(frame, metadata, config=None, **kwargs):
    return clean_dataframe(frame, "demand", metadata=metadata, config=config,
                           units={"interval_hours": "h", "building_load_kw": "kW"}, **kwargs)


def kinds(result):
    return {entry["kind"] for entry in result["report"]["issues"]}


def fill_policy(method, **kwargs):
    config = load_cleaning_config()
    config["missing"]["columns"]["building_load_kw"] = {"method": method, **kwargs}
    return config


def test_valid_records_and_input_frame_are_preserved(metadata):
    frame = demand([1.0, 2.0])
    before, before_meta = frame.copy(deep=True), deepcopy(metadata)
    result = clean(frame, metadata)
    pd.testing.assert_frame_equal(frame, before)
    assert metadata == before_meta
    assert result["report"]["canonical_validation"]["status"] == "passed"
    assert result["metadata"]["source_metadata"] == before_meta["source_metadata"]
    json.dumps(result["report"], allow_nan=False)


def test_duplicates_are_flagged_and_retained_by_default(metadata):
    frame = demand([1, 2])
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    result = clean(frame, metadata)
    assert "duplicate_keys" in kinds(result)
    assert len(result["data"]) == 3


def test_drop_exact_preserves_conflicting_duplicate_measurements(metadata):
    first = demand([1])
    conflict = demand([99])
    frame = pd.concat([first, first, conflict], ignore_index=True)
    config = load_cleaning_config()
    config["duplicates"] = "drop_exact"
    result = clean(frame, metadata, config)
    assert list(result["data"]["building_load_kw"]) == [1, 99]
    assert any(op["operation"] == "drop_exact_duplicate" for op in result["report"]["operations"])
    assert result["report"]["canonical_validation"]["status"] == "failed"


def test_duplicate_error_attaches_report(metadata):
    config = load_cleaning_config()
    config["duplicates"] = "error"
    with pytest.raises(CleaningError, match="Duplicate") as error:
        clean(pd.concat([demand([1]), demand([1])]), metadata, config)
    assert error.value.report["issues"]


def test_missing_values_reported_without_zero_fill(metadata):
    result = clean(demand([1, None, " ", 4]), metadata)
    assert "missing_values" in kinds(result)
    assert result["data"]["building_load_kw"].isna().sum() == 2
    assert result["report"]["remaining_missing_by_column"]["building_load_kw"] == 2


def test_invalid_values_flagged_and_original_numeric_values_retained(metadata):
    result = clean(demand([-2, "bad", float("inf"), 4]), metadata)
    assert {"physically_invalid", "invalid_numeric"} <= kinds(result)
    assert result["data"].iloc[0]["building_load_kw"] == -2
    assert result["data"]["building_load_kw"].isna().sum() == 2
    json.dumps(result["report"], allow_nan=False)


def test_negative_price_and_price_spike_preserved(metadata):
    frame = pd.DataFrame({
        "timestamp_utc": pd.date_range("2026-01-01", periods=10, freq="h", tz="UTC").astype(str),
        "grid_id": "GRID01", "interval_hours": 1,
        "import_price": [-20] + [5] * 8 + [10000], "export_price": [-30] + [0] * 9,
    })
    result = clean_dataframe(frame, "price", metadata=metadata, currency_code="USD",
                             units={"interval_hours": "h", "import_price": "USD/MWh", "export_price": "USD/MWh"})
    assert result["data"]["import_price"].iloc[0] == -0.02
    assert result["data"]["import_price"].iloc[-1] == 10
    assert "baseline_tariff_incompatible_preserved" in kinds(result)
    assert "suspicious_outlier_preserved" in kinds(result)
    assert "physically_invalid" not in kinds(result)
    assert result["report"]["canonical_validation"]["status"] == "failed"


def test_demand_spike_flagged_without_clipping(metadata):
    result = clean(demand([5] * 9 + [1000]), metadata)
    assert "suspicious_outlier_preserved" in kinds(result)
    assert result["data"]["building_load_kw"].iloc[-1] == 1000


def test_explicit_mapping_and_power_conversion(metadata):
    frame = demand([1000, 2500]).rename(columns={"building_load_kw": "meter_w"})
    config = load_cleaning_config()
    config["column_mapping"] = {"meter_w": "building_load_kw"}
    result = clean_dataframe(frame, "demand", metadata=metadata, config=config,
                             units={"interval_hours": "h", "meter_w": "W"})
    assert list(result["data"]["building_load_kw"]) == [1, 2.5]
    assert result["units"]["building_load_kw"] == "kW"
    assert {op["operation"] for op in result["report"]["operations"]} >= {"rename_column", "convert_unit"}


def test_documented_solar_profile(metadata):
    frame = pd.DataFrame({"timestamp_utc": ["2026-01-01T00:00:00Z"],
                          "interval_hours": [1], "asset_id": ["PV01"],
                          "generation_type": ["solar"], "pv_available_kw": [3]})
    result = clean_dataframe(frame, "generation", metadata=metadata, profile="campus_solar",
                             units={"interval_hours": "h", "pv_available_kw": "kW"})
    assert result["data"]["available_generation_kw"].iloc[0] == 3
    assert result["report"]["canonical_validation"]["status"] == "passed"


@pytest.mark.parametrize("unit", ["kWh", "unsupported"])
def test_incompatible_units_fail_without_guessing(unit, metadata):
    with pytest.raises(CleaningError, match="Unsupported unit conversion") as error:
        clean_dataframe(demand([1]), "demand", metadata=metadata,
                         units={"interval_hours": "h", "building_load_kw": unit})
    assert error.value.report is not None


@pytest.mark.parametrize("column,input_unit,value,expected", [
    ("air_temperature_c", "K", 273.15, 0),
    ("air_temperature_c", "degF", 32, 0),
    ("wind_speed_m_s", "km/h", 36, 10),
    ("irradiance_w_m2", "kW/m^2", 0.5, 500),
    ("relative_humidity_fraction", "%", 50, 0.5),
])
def test_weather_unit_conversion(column, input_unit, value, expected, metadata):
    frame = pd.DataFrame({"timestamp_utc": ["2026-01-01T00:00:00Z"],
                          "interval_hours": [1], "site_id": ["SITE01"], column: [value]})
    result = clean_dataframe(frame, "weather", metadata=metadata,
                             units={"interval_hours": "h", column: input_unit})
    assert result["data"][column].iloc[0] == pytest.approx(expected)


def test_energy_conversion_does_not_change_power_dimension(metadata):
    frame = demand([1])
    frame["demand_energy_kwh"] = 1000
    result = clean_dataframe(frame, "demand", metadata=metadata,
                             units={"interval_hours": "h", "building_load_kw": "kW", "demand_energy_kwh": "Wh"})
    assert result["data"]["demand_energy_kwh"].iloc[0] == 1
    assert result["report"]["canonical_validation"]["status"] == "passed"


def test_aware_and_naive_timezones(metadata):
    frame = demand([1])
    frame["timestamp_utc"] = "2026-01-01T06:00:00+06:00"
    assert clean(frame, metadata)["data"]["timestamp_utc"].iloc[0] == "2026-01-01T00:00:00Z"
    frame["timestamp_utc"] = "2026-01-01 06:00:00"
    result = clean(frame, metadata)
    assert "invalid_timestamp" in kinds(result)
    config = load_cleaning_config()
    config["source_timezone"] = "Asia/Dhaka"
    assert clean(frame, metadata, config)["data"]["timestamp_utc"].iloc[0] == "2026-01-01T00:00:00Z"


def test_explicit_non_iso_format_and_dst_error(metadata):
    frame = demand([1])
    frame["timestamp_utc"] = "01/02/2026 06:00"
    assert "invalid_timestamp" in kinds(clean(frame, metadata))
    config = load_cleaning_config()
    config.update(source_timezone="UTC", timestamp_format="%d/%m/%Y %H:%M")
    assert clean(frame, metadata, config)["data"]["timestamp_utc"].iloc[0] == "2026-02-01T06:00:00Z"
    config.update(source_timezone="America/New_York", timestamp_format=None)
    frame["timestamp_utc"] = "2026-11-01T01:00:00"
    assert "invalid_timestamp" in kinds(clean(frame, metadata, config))


def test_short_interpolation_is_explicit_and_audited(metadata):
    result = clean(demand([1, None, 3]), metadata, fill_policy("interpolate", max_gap_hours=1))
    assert result["data"]["building_load_kw"].iloc[1] == 2
    assert any(op.get("uses_future_endpoint") for op in result["report"]["operations"])


def test_long_missing_run_is_not_partially_interpolated(metadata):
    result = clean(demand([1, None, None, None, 5]), metadata,
                   fill_policy("interpolate", max_gap_hours=2))
    assert result["data"]["building_load_kw"].isna().sum() == 3
    assert "fill_skipped_gap_or_boundary" in kinds(result)


def test_missing_timestamps_block_interpolation(metadata):
    frame = demand([1, None, 4]).copy()
    frame.at[2, "timestamp_utc"] = "2026-01-01T04:00:00Z"
    result = clean(frame, metadata, fill_policy("interpolate", max_gap_hours=3))
    assert pd.isna(result["data"]["building_load_kw"].iloc[1])
    assert "missing_time_intervals" in kinds(result)


def test_fills_do_not_cross_buildings(metadata):
    frame = pd.concat([demand([1, None]), demand([None, 9], building="B02")], ignore_index=True)
    result = clean(frame, metadata, fill_policy("forward_fill", max_gap_hours=1))
    assert result["data"]["building_load_kw"].iloc[1] == 1
    assert pd.isna(result["data"]["building_load_kw"].iloc[2])


def test_invalid_endpoints_and_parse_errors_are_not_imputed(metadata):
    result = clean(demand([-5, None, 3]), metadata, fill_policy("interpolate", max_gap_hours=1))
    assert pd.isna(result["data"]["building_load_kw"].iloc[1])
    result = clean(demand([1, "bad", 3]), metadata,
                   fill_policy("constant", value=0, reason="Explicit synthetic fixture"))
    assert pd.isna(result["data"]["building_load_kw"].iloc[1])


def test_fill_requires_gap_limit_and_constant_requires_reason(metadata):
    with pytest.raises(CleaningError, match="max_gap_hours"):
        clean(demand([1, None, 3]), metadata, fill_policy("interpolate"))
    with pytest.raises(CleaningError, match="documented reason"):
        clean(demand([1, None]), metadata, fill_policy("constant", value=0))


def test_explicit_constant_drop_and_error_policies(metadata):
    frame = demand([1, None])
    result = clean(frame, metadata, fill_policy("constant", value=0, reason="Test-only assumption"))
    assert result["data"]["building_load_kw"].iloc[1] == 0
    assert len(clean(frame, metadata, fill_policy("drop"))["data"]) == 1
    with pytest.raises(CleaningError, match="Missing values"):
        clean(frame, metadata, fill_policy("error"))


def test_mapping_collision_is_rejected(metadata):
    frame = demand([1])
    frame["other"] = 2
    config = load_cleaning_config()
    config["column_mapping"] = {"other": "building_load_kw"}
    with pytest.raises(CleaningError, match="collision"):
        clean(frame, metadata, config)


def test_raw_csv_is_byte_preserved_and_load_is_audited(tmp_path, metadata):
    path = tmp_path / "raw.csv"
    original = b"timestamp_utc,interval_hours,building_id,building_load_kw\r\n2026-01-01T00:00:00Z,1,B01,1000\r\n"
    path.write_bytes(original)
    result = load_and_clean_csv(path, "demand", metadata=metadata,
                                units={"interval_hours": "h", "building_load_kw": "W"})
    assert path.read_bytes() == original
    assert result["data"]["building_load_kw"].iloc[0] == 1
    assert result["report"]["operations"][0]["operation"] == "load_csv_read_only"
    assert result["metadata"]["cleaning"]["raw_file"]["sha256"] == hashlib.sha256(original).hexdigest()


def test_missing_file_error_is_useful(tmp_path, metadata):
    with pytest.raises(CleaningError, match="Cannot load raw CSV"):
        load_and_clean_csv(tmp_path / "missing.csv", "demand", units={}, metadata=metadata)
