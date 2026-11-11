"""Reproducibility, provenance and physical bounds of hypothetical profiles."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.data.integration import integrate_datasets
from src.data.synthetic import (
    SyntheticDataError,
    assess_missing_variables,
    generate_synthetic,
    load_scenario,
    save_synthetic,
    validate_scenario,
)


@pytest.fixture
def setup(tmp_path):
    root = Path(__file__).resolve().parents[1]
    sources = yaml.safe_load((root / "configs/data_sources.yaml").read_text())
    for source in sources["sources"]:
        source.update(status="pending", local_path=None, download_url=None)
    source_path = tmp_path / "sources.yaml"
    source_path.write_text(yaml.safe_dump(sources))
    config = load_scenario()
    config["hours"] = 168
    return config, {"sources_path": source_path, "processed_root": tmp_path / "processed"}


def categories(result):
    return {source["category"]: source for source in result["sources"]}


def test_assessment_identifies_missing_variables_before_generation(setup):
    _, paths = setup
    report = assess_missing_variables(**paths)
    assert set(report["categories"]) == {"demand", "solar", "wind", "weather", "grid_prices"}
    assert all(item["status"] == "unavailable" and item["variables"] for item in report["categories"].values())
    assert not paths["processed_root"].exists()


def test_same_seed_reproduces_all_frames_and_metadata(setup):
    config, paths = setup
    first, second = generate_synthetic(config, **paths), generate_synthetic(config, **paths)
    for a, b in zip(first["sources"], second["sources"]):
        pd.testing.assert_frame_equal(a["data"], b["data"], check_exact=True)
        assert a["metadata"] == b["metadata"]
    assert first["report"] == second["report"]


def test_different_seed_changes_stochastic_profiles(setup):
    config, paths = setup
    first = categories(generate_synthetic(config, **paths))
    config["seed"] += 1
    second = categories(generate_synthetic(config, **paths))
    assert not first["demand"]["data"].equals(second["demand"]["data"])


def test_continuous_hourly_timestamps_and_unique_entity_keys(setup):
    config, paths = setup
    result = generate_synthetic(config, **paths)
    for source in result["sources"]:
        frame = source["data"]
        id_column = {"demand": "building_id", "generation": "asset_id", "weather": "site_id", "price": "grid_id"}[source["record_type"]]
        assert not frame.duplicated(["timestamp_utc", id_column]).any()
        for _, group in frame.groupby(id_column):
            times = pd.to_datetime(group["timestamp_utc"], utc=True)
            assert len(times) == config["hours"]
            assert (times.diff().dropna() == pd.Timedelta(hours=1)).all()


def test_power_energy_units_and_bounds(setup):
    config, paths = setup
    result = categories(generate_synthetic(config, **paths))
    for category in ("solar", "wind"):
        source = result[category]
        frame = source["data"]
        assert frame["available_generation_kw"].between(0, config[category]["capacity_kw"]).all()
        np.testing.assert_allclose(frame["available_generation_energy_kwh"], frame["available_generation_kw"] * frame["interval_hours"])
        assert source["units"]["available_generation_kw"] == "kW"
        assert source["units"]["available_generation_energy_kwh"] == "kWh"
    demand = result["demand"]["data"]
    for building in config["buildings"]:
        values = demand[demand["building_id"] == building["id"]]["building_load_kw"]
        assert values.between(building["min_kw"], building["max_kw"]).all()
    np.testing.assert_allclose(demand["demand_energy_kwh"], demand["building_load_kw"] * demand["interval_hours"])
    assert sum(item["max_kw"] for item in config["buildings"]) <= config["grid"]["import_limit_kw"]


def test_nighttime_pv_and_wind_cutout_are_zero(setup):
    config, paths = setup
    result = categories(generate_synthetic(config, **paths))
    weather = result["weather"]["data"]
    solar = result["solar"]["data"]
    assert (solar.loc[weather["irradiance_w_m2"] == 0, "available_generation_kw"] == 0).all()
    config["weather"].update(wind_mean_m_s=30, wind_seasonal_amplitude_m_s=0, wind_noise_std_m_s=0)
    result = categories(generate_synthetic(config, **paths))
    assert (result["wind"]["data"]["available_generation_kw"] == 0).all()


def test_daily_and_seasonal_patterns_are_present(setup):
    config, paths = setup
    config["hours"] = 8760
    config["demand"].update(uncertainty_std=0, surge_probability=0)
    result = categories(generate_synthetic(config, **paths))
    weather = result["weather"]["data"]
    local = pd.to_datetime(weather["timestamp_utc"], utc=True).dt.tz_convert(config["campus_timezone"])
    summer = weather.loc[local.dt.month == 7, "air_temperature_c"].mean()
    winter = weather.loc[local.dt.month == 1, "air_temperature_c"].mean()
    assert summer > winter
    assert weather.loc[local.dt.hour == 12, "irradiance_w_m2"].mean() > weather.loc[local.dt.hour == 0, "irradiance_w_m2"].mean()


def test_every_variable_is_synthetically_labeled(setup):
    config, paths = setup
    result = generate_synthetic(config, **paths)
    for source in result["sources"]:
        metadata = source["metadata"]
        assert metadata["origin"] == "simulated"
        assert metadata["synthetic"] is True
        assert "NOT_MEASURED" in metadata["dataset_label"]
        origins = metadata["source_metadata"]["variable_origins"]
        assert set(origins) == set(source["data"].columns)
        assert all(item["origin"] == "synthetic" for item in origins.values())
        assert metadata["source_metadata"]["seed"] == config["seed"]


def test_optional_surges_and_spikes_are_bounded_and_logged(setup):
    config, paths = setup
    config["demand"].update(surge_probability=1, surge_multiplier=100)
    config["prices"].update(spike_probability=1, spike_multiplier=100)
    result = generate_synthetic(config, **paths)
    prices = categories(result)["grid_prices"]["data"]
    assert (prices["import_price"] == config["prices"]["max_import_per_kwh"]).all()
    assert (prices["export_price"] <= prices["import_price"]).all()
    assert all(len(event["surge_hours"]) == config["hours"] for event in result["report"]["events"] if event["kind"] == "demand_assumptions")
    config["demand"]["surge_probability"] = 0
    config["prices"]["spike_probability"] = 0
    result = generate_synthetic(config, **paths)
    assert all(not event.get("surge_hours", []) and not event.get("hours", []) for event in result["report"]["events"])


def test_verified_source_is_protected_without_claiming_coverage(setup):
    config, paths = setup
    sources = yaml.safe_load(paths["sources_path"].read_text())
    sources["sources"][0].update(status="verified", local_path="existing.csv", time_zone="UTC", license="Test-only license")
    paths["sources_path"].write_text(yaml.safe_dump(sources))
    result = generate_synthetic(config, **paths)
    assert "solar" not in categories(result)
    assert result["report"]["assessment"]["categories"]["solar"]["status"] == "protected_requires_coverage_review"


def test_existing_partial_integration_provenance_prevents_replacement(setup):
    config, paths = setup
    report_dir = paths["processed_root"] / "existing"
    report_dir.mkdir(parents=True)
    original = json.dumps({"sources": {}, "provenance": [
        {"source_id": "campus_wind", "record_type": "generation"},
    ]})
    report_path = report_dir / "integration_report.json"
    report_path.write_text(original)
    result = generate_synthetic(config, **paths)
    assert "wind" not in categories(result)
    assert report_path.read_text() == original


def test_invalid_existing_report_fails_closed(setup):
    config, paths = setup
    report_dir = paths["processed_root"] / "existing"
    report_dir.mkdir(parents=True)
    (report_dir / "integration_report.json").write_text("not JSON")
    with pytest.raises(SyntheticDataError, match="Cannot assess"):
        generate_synthetic(config, **paths)


def test_weather_dependency_needs_explicit_latent_scenario_policy(setup):
    config, paths = setup
    config["generate_categories"] = ["solar"]
    with pytest.raises(SyntheticDataError, match="allow_latent_weather"):
        generate_synthetic(config, **paths)
    config["allow_latent_weather"] = True
    assert set(categories(generate_synthetic(config, **paths))) == {"solar"}


def test_integration_compatibility_and_complete_output(setup):
    config, paths = setup
    config["hours"] = 24
    generated = generate_synthetic(config, **paths)
    integrated = integrate_datasets(generated["sources"])
    assert integrated["report"]["complete"]
    assert integrated["report"]["entity_count"] == 9
    assert integrated["report"]["output_rows"] == 24 * 9


@pytest.mark.parametrize("section,key,value", [
    (None, "seed", -1), (None, "hours", 0),
    ("wind", "rated_m_s", 1), ("battery", "initial_soc", 1.5),
    ("grid", "import_limit_kw", 1), ("weather", "autocorrelation", 1),
    ("prices", "max_import_per_kwh", float("inf")),
    ("demand", "surge_probability", -0.5),
])
def test_invalid_assumptions_rejected(setup, section, key, value):
    config, _ = setup
    target = config if section is None else config[section]
    target[key] = value
    with pytest.raises(SyntheticDataError):
        validate_scenario(config)


def test_save_labels_checksums_manifest_and_no_overwrites(setup):
    config, paths = setup
    config["hours"] = 24
    result = generate_synthetic(config, **paths)
    destination = save_synthetic(result, "test", processed_root=paths["processed_root"])
    assert destination.name.startswith("synthetic_")
    report = json.loads((destination / "generation_report.json").read_text())
    assert len(report["file_checksums"]) == 5
    assert len(yaml.safe_load((destination / "integration_manifest.yaml").read_text())["datasets"]) == 5
    original = (destination / "synthetic_demand.csv").read_bytes()
    with pytest.raises(SyntheticDataError, match="already exists"):
        save_synthetic(result, "test", processed_root=paths["processed_root"])
    assert (destination / "synthetic_demand.csv").read_bytes() == original


def test_config_and_source_metadata_not_modified(setup):
    config, paths = setup
    original = deepcopy(config)
    source_bytes = paths["sources_path"].read_bytes()
    generate_synthetic(config, **paths)
    assert config == original
    assert paths["sources_path"].read_bytes() == source_bytes
