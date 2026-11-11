"""Simulator configuration validation without battery or dispatch behavior."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict
from pathlib import Path

import pytest
import yaml

from src.simulation.config import (
    SimulationConfigError,
    load_simulation_config,
    validate_simulation_config,
)


@pytest.fixture
def config():
    return asdict(load_simulation_config())


def test_valid_default_loads_and_matches_labeled_synthetic_parameters():
    loaded = load_simulation_config()
    reference = Path(__file__).resolve().parents[1] / "configs/synthetic_scenario.yaml"
    synthetic = yaml.safe_load(reference.read_text(encoding="utf-8"))
    assert loaded.parameter_origin == "hypothetical_example"
    assert loaded.time_step_hours == 1
    assert asdict(loaded.battery) == synthetic["battery"]
    assert loaded.grid.import_limit_kw == synthetic["grid"]["import_limit_kw"]
    assert loaded.grid.export_limit_kw == synthetic["grid"]["export_limit_kw"]
    assert loaded.renewable_curtailment.enabled is True
    assert loaded.unserved_load.handling == "error"


@pytest.mark.parametrize("section,field", [
    (None, "battery"), (None, "time_step_hours"), ("battery", "capacity_kwh"),
    ("battery", "initial_soc"), ("battery", "charge_efficiency"),
    ("grid", "import_limit_kw"), ("renewable_curtailment", "enabled"),
    ("unserved_load", "handling"),
])
def test_missing_required_fields(config, section, field):
    del (config if section is None else config[section])[field]
    with pytest.raises(SimulationConfigError, match=f"missing required parameters.*{field}"):
        validate_simulation_config(config)


@pytest.mark.parametrize("value", [0, -1, True, "500", None, float("nan"), float("inf")])
def test_invalid_capacity(config, value):
    config["battery"]["capacity_kwh"] = value
    with pytest.raises(SimulationConfigError, match="battery.capacity_kwh"):
        validate_simulation_config(config)


@pytest.mark.parametrize("field", ["charge_efficiency", "discharge_efficiency"])
@pytest.mark.parametrize("value", [0, -0.1, 1.01, True, "0.95", float("inf")])
def test_invalid_efficiency(config, field, value):
    config["battery"][field] = value
    with pytest.raises(SimulationConfigError, match=field):
        validate_simulation_config(config)


@pytest.mark.parametrize("minimum,maximum", [(0.9, 0.1), (0.5, 0.5)])
def test_soc_bounds_must_be_strictly_ordered(config, minimum, maximum):
    config["battery"].update(soc_min=minimum, soc_max=maximum)
    with pytest.raises(SimulationConfigError, match="soc_min.*strictly less"):
        validate_simulation_config(config)


@pytest.mark.parametrize("value", [0.05, 0.95, -1, 50, float("nan")])
def test_initial_soc_must_be_fraction_inside_bounds(config, value):
    config["battery"]["initial_soc"] = value
    with pytest.raises(SimulationConfigError, match="initial_soc"):
        validate_simulation_config(config)


@pytest.mark.parametrize("value", [0.1, 0.9])
def test_initial_soc_accepts_boundaries(config, value):
    config["battery"]["initial_soc"] = value
    assert validate_simulation_config(config).battery.initial_soc == value


@pytest.mark.parametrize("field", ["soc_min", "soc_max"])
@pytest.mark.parametrize("value", [-0.01, 1.01, False])
def test_invalid_soc_limits(config, field, value):
    config["battery"][field] = value
    with pytest.raises(SimulationConfigError, match=field):
        validate_simulation_config(config)


@pytest.mark.parametrize("section,field", [
    ("battery", "charge_limit_kw"), ("battery", "discharge_limit_kw"),
    ("grid", "import_limit_kw"), ("grid", "export_limit_kw"),
])
def test_negative_power_limits_rejected(config, section, field):
    config[section][field] = -1
    with pytest.raises(SimulationConfigError, match=field):
        validate_simulation_config(config)


def test_zero_power_limits_and_perfect_efficiency_are_valid_settings(config):
    config["battery"].update(charge_limit_kw=0, discharge_limit_kw=0,
                              charge_efficiency=1, discharge_efficiency=1)
    config["grid"].update(import_limit_kw=0, export_limit_kw=0)
    config["renewable_curtailment"]["enabled"] = False
    loaded = validate_simulation_config(config)
    assert loaded.grid.export_limit_kw == 0
    assert not loaded.renewable_curtailment.enabled


@pytest.mark.parametrize("value", [0, -1, True, "1", float("inf")])
def test_time_step_must_be_positive_finite_numeric(config, value):
    config["time_step_hours"] = value
    with pytest.raises(SimulationConfigError, match="time_step_hours"):
        validate_simulation_config(config)


@pytest.mark.parametrize("value", [0.25, 0.5, 2])
def test_component_time_step_is_configurable_without_resampling_data(config, value):
    config["time_step_hours"] = value
    assert validate_simulation_config(config).time_step_hours == value


@pytest.mark.parametrize("section,field,value", [
    ("renewable_curtailment", "enabled", "false"),
    ("renewable_curtailment", "enabled", 1),
    ("renewable_curtailment", "allocation", "cost_optimal"),
    ("unserved_load", "handling", "silently_shed"),
    ("grid", "asset_id", "BAT01"),
])
def test_invalid_policy_types_and_relationships(config, section, field, value):
    config[section][field] = value
    with pytest.raises(SimulationConfigError, match=field):
        validate_simulation_config(config)


def test_unknown_parameters_are_not_silently_ignored(config):
    config["battery"]["capacity_kw"] = 500
    with pytest.raises(SimulationConfigError, match="unknown parameters.*capacity_kw"):
        validate_simulation_config(config)


@pytest.mark.parametrize("value", [True, 1.0, "1", 2])
def test_version_requires_supported_integer(config, value):
    config["version"] = value
    with pytest.raises(SimulationConfigError, match="version"):
        validate_simulation_config(config)


def test_configuration_is_immutable_and_input_unchanged(config):
    before = deepcopy(config)
    loaded = validate_simulation_config(config)
    assert config == before
    with pytest.raises(FrozenInstanceError):
        loaded.battery.capacity_kwh = 1
    with pytest.raises(FrozenInstanceError):
        loaded.grid = None


def test_loader_supports_custom_file(config, tmp_path):
    path = tmp_path / "simulation.yaml"
    config["grid"]["export_limit_kw"] = 0
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert load_simulation_config(path).grid.export_limit_kw == 0


@pytest.mark.parametrize("text", ["", "[]", "battery: [", "version: 1\nversion: 2\n"])
def test_invalid_yaml_or_structure_rejected(tmp_path, text):
    path = tmp_path / "bad.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(SimulationConfigError):
        load_simulation_config(path)


def test_nested_duplicate_yaml_keys_rejected(tmp_path):
    path = tmp_path / "duplicate.yaml"
    path.write_text("battery:\n  capacity_kwh: 500\n  capacity_kwh: 100\n")
    with pytest.raises(SimulationConfigError, match="Duplicate YAML key.*capacity_kwh"):
        load_simulation_config(path)


def test_missing_file_has_useful_error(tmp_path):
    with pytest.raises(SimulationConfigError, match="Cannot load simulation configuration"):
        load_simulation_config(tmp_path / "missing.yaml")
