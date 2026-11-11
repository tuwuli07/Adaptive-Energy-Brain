"""Small offline calculation fixtures; no production datasets are generated."""

from copy import deepcopy
import json

import pandas as pd
import pytest
import yaml

from scripts.build_dataset import main
from src.data.integration import (
    IntegrationError,
    integrate_datasets,
    save_integrated_dataset,
)


def demand_source(values=(4, 8), *, times=None, durations=None,
                  building="B01", source_id="test_demand"):
    count = len(values)
    return {
        "source_id": source_id, "record_type": "demand",
        "measurement_kind": "interval_mean",
        "data": pd.DataFrame({
            "timestamp_utc": times if times is not None else pd.date_range(
                "2026-01-01", periods=count, freq="h", tz="UTC").astype(str),
            "interval_hours": durations if durations is not None else [1] * count,
            "building_id": [building] * count, "building_load_kw": list(values),
        }),
        "units": {"interval_hours": "h", "building_load_kw": "kW"},
        "metadata": {"source_id": source_id, "origin": "simulated",
                     "source_metadata": {"note": "Unit-test calculation fixture only"}},
    }


def issue_kinds(result):
    return {issue["kind"] for issue in result["report"]["issues"]}


def test_hourly_utc_alignment_and_input_preservation():
    source = demand_source()
    before = deepcopy(source)
    result = integrate_datasets([source])
    assert result["data"]["timestamp_utc"].tolist() == [
        "2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z",
    ]
    assert result["report"]["complete"]
    pd.testing.assert_frame_equal(source["data"], before["data"])
    assert source["metadata"] == before["metadata"]
    assert result["report"]["sources"]["test_demand"]["metadata"] == before["metadata"]
    json.dumps(result["report"], allow_nan=False)


def test_duration_weighted_power_and_summed_interval_energy():
    source = demand_source([4, 8], times=["2026-01-01T00:00:00Z", "2026-01-01T00:15:00Z"], durations=[0.25, 0.75])
    source["data"]["demand_energy_kwh"] = [1, 6]
    source["units"]["demand_energy_kwh"] = "kWh"
    result = integrate_datasets([source])
    assert result["data"].iloc[0]["building_load_kw"] == 7
    assert result["data"].iloc[0]["demand_energy_kwh"] == 7
    assert len(result["report"]["provenance"]) == 4


def test_power_is_averaged_not_summed():
    source = demand_source([10, 10, 10, 10], times=pd.date_range(
        "2026-01-01", periods=4, freq="15min", tz="UTC").astype(str), durations=[0.25] * 4)
    assert integrate_datasets([source])["data"].iloc[0]["building_load_kw"] == 10


def test_energy_splitting_requires_explicit_uniform_policy():
    source = demand_source([4], durations=[2])
    source["mean_allocation"] = "constant"
    source["data"]["demand_energy_kwh"] = [8]
    source["units"]["demand_energy_kwh"] = "kWh"
    with pytest.raises(IntegrationError, match="energy_allocation"):
        integrate_datasets([source])
    source["energy_allocation"] = "uniform"
    result = integrate_datasets([source])
    assert result["data"]["demand_energy_kwh"].tolist() == [4, 4]
    assert result["data"]["building_load_kw"].tolist() == [4, 4]


def test_coarse_means_are_not_silently_disaggregated():
    source = demand_source([4], durations=[2])
    with pytest.raises(IntegrationError, match="mean_allocation"):
        integrate_datasets([source])
    source["mean_allocation"] = "constant"
    result = integrate_datasets([source])
    assert result["data"]["building_load_kw"].tolist() == [4, 4]
    assert result["report"]["sources"]["test_demand"]["mean_allocation"] == "constant"


def test_weather_weighted_mean():
    source = demand_source([0, 20], times=["2026-01-01T00:00:00Z", "2026-01-01T00:30:00Z"], durations=[0.5, 0.5])
    source["record_type"] = "weather"
    source["data"] = source["data"].rename(columns={"building_id": "site_id", "building_load_kw": "air_temperature_c"})
    source["units"] = {"interval_hours": "h", "air_temperature_c": "degC"}
    assert integrate_datasets([source])["data"].iloc[0]["air_temperature_c"] == 10


def test_negative_prices_are_preserved_and_time_weighted():
    source = demand_source([-2, 6], times=["2026-01-01T00:00:00Z", "2026-01-01T00:30:00Z"], durations=[0.5, 0.5])
    source["record_type"] = "price"
    source["data"] = source["data"].rename(columns={"building_id": "grid_id", "building_load_kw": "import_price"})
    source["data"]["export_price"] = [-3, -1]
    source["units"] = {"interval_hours": "h", "import_price": "USD/kWh", "export_price": "USD/kWh"}
    source["metadata"]["currency_code"] = "USD"
    result = integrate_datasets([source])
    assert result["data"].iloc[0]["import_price"] == 2
    assert result["data"].iloc[0]["export_price"] == -2


def test_two_buildings_preserved_without_duplicate_building_keys():
    result = integrate_datasets([
        demand_source(building="B01"), demand_source(building="B02", source_id="other"),
    ])
    assert len(result["data"]) == 4
    assert set(result["data"]["building_id"]) == {"B01", "B02"}
    assert not result["data"].duplicated(["timestamp_utc", "building_id"]).any()


def test_shared_generation_not_multiplied_by_building_count():
    generation = demand_source([3, 5], source_id="test_solar")
    generation["record_type"] = "generation"
    generation["data"] = generation["data"].rename(columns={"building_id": "asset_id", "building_load_kw": "available_generation_kw"})
    generation["data"]["asset_id"] = "PV01"
    generation["data"]["generation_type"] = "solar"
    generation["units"] = {"interval_hours": "h", "available_generation_kw": "kW"}
    result = integrate_datasets([demand_source(), demand_source(building="B02", source_id="d2"), generation])
    assert len(result["data"]) == 6
    assert result["data"]["available_generation_kw"].sum() == 8


def test_full_outer_window_does_not_discard_unmatched_timestamps():
    first = demand_source([1], building="B01")
    second = demand_source([2], times=["2026-01-01T02:00:00Z"], building="B02", source_id="other")
    result = integrate_datasets([first, second])
    assert len(result["data"]) == 6
    assert result["report"]["expected_hours"] == 3
    assert not result["report"]["complete"]
    assert "missing_entity_interval" in issue_kinds(result)
    assert result["data"]["building_load_kw"].notna().sum() == 2


def test_wholly_missing_expected_building_is_flagged():
    result = integrate_datasets([demand_source()], expected_entities={"demand": ["B01", "B02"]})
    assert "entire_entity_missing" in issue_kinds(result)
    assert len(result["data"]) == 4


def test_partial_hour_not_extrapolated_or_filled():
    source = demand_source([12], durations=[0.5])
    result = integrate_datasets([source])
    assert pd.isna(result["data"].iloc[0]["building_load_kw"])
    assert "incomplete_hour" in issue_kinds(result)
    assert not result["report"]["complete"]


def test_offset_and_naive_timezone_normalization():
    source = demand_source([1], times=["2026-01-01T06:00:00+06:00"])
    assert integrate_datasets([source])["data"].iloc[0]["timestamp_utc"] == "2026-01-01T00:00:00Z"
    source["data"]["timestamp_utc"] = "2026-01-01T06:00:00"
    with pytest.raises(IntegrationError, match="source_timezone"):
        integrate_datasets([source])
    source["source_timezone"] = "Asia/Dhaka"
    assert integrate_datasets([source])["data"].iloc[0]["timestamp_utc"] == "2026-01-01T00:00:00Z"


def test_dst_repeated_local_hour_with_explicit_offsets_is_distinct():
    source = demand_source([1, 2], times=["2026-11-01T01:00:00-04:00", "2026-11-01T01:00:00-05:00"])
    assert integrate_datasets([source])["data"]["timestamp_utc"].tolist() == [
        "2026-11-01T05:00:00Z", "2026-11-01T06:00:00Z",
    ]


@pytest.mark.parametrize("local_time", ["2026-11-01T01:00:00", "2026-03-08T02:00:00"])
def test_ambiguous_and_nonexistent_dst_times_are_rejected(local_time):
    source = demand_source([1], times=[local_time])
    source["source_timezone"] = "America/New_York"
    with pytest.raises(IntegrationError, match="Cannot normalize"):
        integrate_datasets([source])


def test_duplicate_and_overlapping_observations_fail():
    source = demand_source([1, 2], times=["2026-01-01T00:00:00Z"] * 2)
    with pytest.raises(IntegrationError, match="duplicate"):
        integrate_datasets([source])
    source = demand_source([1, 2], durations=[2, 1])
    with pytest.raises(IntegrationError, match="overlapping"):
        integrate_datasets([source])


def test_conflicting_sources_are_not_silently_prioritized():
    with pytest.raises(IntegrationError, match="Conflicting sources"):
        integrate_datasets([demand_source(), demand_source(source_id="other")])


def test_instantaneous_samples_are_not_assumed_to_be_interval_means():
    source = demand_source()
    source["measurement_kind"] = "instantaneous"
    with pytest.raises(IntegrationError, match="instantaneous"):
        integrate_datasets([source])


def test_unit_mismatch_and_inconsistent_power_energy_rejected():
    source = demand_source()
    source["units"]["building_load_kw"] = "kWh"
    with pytest.raises(IntegrationError, match="expected kW"):
        integrate_datasets([source])
    source = demand_source()
    source["data"]["demand_energy_kwh"] = [99, 99]
    source["units"]["demand_energy_kwh"] = "kWh"
    with pytest.raises(IntegrationError, match="power times hours"):
        integrate_datasets([source])


def test_requested_window_clipping_is_explicitly_reported():
    result = integrate_datasets([demand_source()], start="2026-01-01T01:00:00Z", end="2026-01-01T02:00:00Z")
    assert len(result["data"]) == 1
    assert "outside_requested_window" in issue_kinds(result)


def test_save_new_bundle_and_prevent_overwrite(tmp_path):
    result = integrate_datasets([demand_source()])
    saved = save_integrated_dataset(result, "test_bundle", processed_root=tmp_path)
    original = (saved / "dataset.csv").read_bytes()
    report = json.loads((saved / "integration_report.json").read_text())
    assert report["dataset_sha256"]
    with pytest.raises(IntegrationError, match="Cannot create new output"):
        save_integrated_dataset(result, "test_bundle", processed_root=tmp_path)
    assert (saved / "dataset.csv").read_bytes() == original


def test_incomplete_output_requires_explicit_save_approval(tmp_path):
    result = integrate_datasets([demand_source([1], durations=[0.5])])
    with pytest.raises(IntegrationError, match="Incomplete"):
        save_integrated_dataset(result, "test_bundle", processed_root=tmp_path)
    saved = save_integrated_dataset(result, "test_bundle", processed_root=tmp_path, allow_incomplete=True)
    assert json.loads((saved / "integration_report.json").read_text())["saved_with_incomplete_approval"]


def test_empty_inputs_and_bad_windows_fail():
    with pytest.raises(IntegrationError, match="No cleaned datasets"):
        integrate_datasets([])
    with pytest.raises(IntegrationError, match="hour boundaries"):
        integrate_datasets([demand_source()], start="2026-01-01T00:30:00Z")


def test_cli_report_only_uses_existing_cleaned_csv_without_modifying_it(tmp_path, capsys):
    source = demand_source()
    csv_path = tmp_path / "cleaned.csv"
    source.pop("data").to_csv(csv_path, index=False)
    original = csv_path.read_bytes()
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(yaml.safe_dump({"datasets": [{**source, "path": "cleaned.csv"}]}))
    assert main(["--manifest", str(manifest), "--report-only"]) == 0
    assert '"complete": true' in capsys.readouterr().out
    assert csv_path.read_bytes() == original
    assert sorted(path.name for path in tmp_path.iterdir()) == ["cleaned.csv", "manifest.yaml"]


def test_cli_missing_manifest_fails_usefully(tmp_path, capsys):
    assert main(["--manifest", str(tmp_path / "missing.yaml")]) == 1
    assert "Integration error" in capsys.readouterr().err
