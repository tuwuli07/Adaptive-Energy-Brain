"""Campus demand aggregation and immutable per-building observations."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from src.simulation.demand import CampusDemand, ComponentValidationError


STAMP = "2026-01-01T00:00:00Z"


def provenance(origin="simulated", **extra):
    return {"source_id": "fixture_demand", "origin": origin,
            "source_metadata": {"note": "Unit-test readings only"}, **extra}


def campus(**kwargs):
    return CampusDemand(provenance(), building_ids=["B01", "B02"], **kwargs)


def test_building_aggregation_preserves_values_and_order():
    loads = {"B02": 20, "B01": 10}
    result = campus().evaluate(loads, timestamp_utc=STAMP)
    assert result.campus_load_kw == 30
    assert list(result.building_loads_kw) == ["B01", "B02"]
    assert dict(result.building_loads_kw) == loads
    assert result.metadata["origin"] == "derived"
    assert result.metadata["source_metadata"]["unit"] == "kW"


def test_zero_demand_is_valid_not_a_missing_measurement():
    result = campus().evaluate({"B01": 0, "B02": 0}, timestamp_utc=STAMP)
    assert result.campus_load_kw == 0
    assert len(result.building_loads_kw) == 2


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True, "10", None])
def test_invalid_building_power_rejected(value):
    with pytest.raises(ComponentValidationError, match="B01.building_load_kw"):
        campus().evaluate({"B01": value, "B02": 20}, timestamp_utc=STAMP)


@pytest.mark.parametrize("loads", [{"B01": 10}, {"B01": 10, "B02": 20, "CAMPUS_TOTAL": 30}])
def test_missing_or_aggregate_rows_are_not_silently_summed(loads):
    with pytest.raises(ComponentValidationError, match="registry exactly"):
        campus().evaluate(loads, timestamp_utc=STAMP)


def test_duplicate_registry_rejected():
    with pytest.raises(ComponentValidationError, match="double-count"):
        CampusDemand(provenance(), building_ids=["B01", "B01"])


def test_optional_limits_are_enforced_and_labeled():
    component = campus(building_limits_kw={"B01": 15}, campus_limit_kw=40,
                       limits_origin="hypothetical_example")
    result = component.evaluate({"B01": 10, "B02": 20}, timestamp_utc=STAMP)
    assert result.metadata["source_metadata"]["limits"]["origin"] == "hypothetical_example"
    with pytest.raises(ComponentValidationError, match="B01.*exceeds"):
        component.evaluate({"B01": 16, "B02": 20}, timestamp_utc=STAMP)
    with pytest.raises(ComponentValidationError, match="campus_load_kw exceeds"):
        component.evaluate({"B01": 10, "B02": 35}, timestamp_utc=STAMP)
    with pytest.raises(ComponentValidationError, match="Parameter origin"):
        campus(building_limits_kw={"B01": 15})


def test_aggregate_only_mode_does_not_invent_building_allocations():
    component = CampusDemand(provenance("measured"))
    result = component.evaluate_total(100, timestamp_utc=STAMP)
    assert result.campus_load_kw == 100
    assert dict(result.building_loads_kw) == {}
    assert result.metadata["source_metadata"]["input_origins"] == {"campus": "measured"}
    with pytest.raises(ComponentValidationError, match="Do not mix"):
        campus().evaluate_total(100, timestamp_utc=STAMP)


def test_mixed_per_building_provenance_is_retained():
    sources = {"B01": provenance("measured"), "B02": provenance("derived", derived_kind="model")}
    result = CampusDemand(sources, building_ids=["B01", "B02"]).evaluate({"B01": 10, "B02": 20}, timestamp_utc=STAMP)
    assert result.metadata["source_metadata"]["input_provenance"] == sources
    assert result.metadata["source_metadata"]["input_origins"] == {"B01": "measured", "B02": "modeled"}


def test_data_loader_observation_read_checks_aggregate_consistency():
    observation = SimpleNamespace(timestamp_utc=STAMP, building_loads_kw={"B01": 10, "B02": 20}, campus_load_kw=30)
    assert campus().read(observation).campus_load_kw == 30
    observation.campus_load_kw = 999
    with pytest.raises(ComponentValidationError, match="disagrees"):
        campus().read(observation)
    with pytest.raises(ComponentValidationError, match="preserve"):
        CampusDemand(provenance()).read(observation)


def test_output_and_input_metadata_are_defensively_preserved():
    metadata = provenance()
    before = deepcopy(metadata)
    component = CampusDemand(metadata, building_ids=["B01"])
    metadata["origin"] = "measured"
    loads = {"B01": 10}
    result = component.evaluate(loads, timestamp_utc=STAMP)
    loads["B01"] = 999
    assert result.building_loads_kw["B01"] == 10
    assert result.metadata["source_metadata"]["input_provenance"]["B01"] == before
    exposed = result.metadata
    exposed["source_metadata"]["input_origins"]["B01"] = "measured"
    assert result.metadata["source_metadata"]["input_origins"]["B01"] == "synthetic"
    with pytest.raises(TypeError):
        result.building_loads_kw["B01"] = 1
    with pytest.raises(FrozenInstanceError):
        result.campus_load_kw = 0


def test_no_implicit_upper_bound_on_valid_demand_spikes():
    # A bound is enforced only when configured, never inferred from typical load.
    assert campus().evaluate({"B01": 10000, "B02": 20}, timestamp_utc=STAMP).campus_load_kw == 10020
