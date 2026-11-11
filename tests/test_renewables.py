"""Renewable readings/curtailment only; no weather models or dispatch."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from src.simulation.renewables import (
    ComponentValidationError,
    SolarPV,
    WindGeneration,
    combine_renewables,
)


STAMP = "2026-01-01T00:00:00Z"


def provenance(origin="simulated", **extra):
    return {"source_id": "fixture_generation", "origin": origin,
            "source_metadata": {"note": "Unit-test provenance only"}, **extra}


def solar(metadata=None):
    return SolarPV("PV01", 300, provenance() if metadata is None else metadata,
                   capacity_origin="hypothetical_example")


def wind():
    return WindGeneration("W01", 150, provenance(), capacity_origin="hypothetical_example")


def test_solar_wind_and_curtailment_combine_correctly():
    pv = solar().evaluate(200, timestamp_utc=STAMP, curtailment_kw=25)
    turbine = wind().evaluate(100, timestamp_utc=STAMP, curtailment_kw=10)
    total = combine_renewables([pv, turbine])
    assert total.total_available_kw == 300
    assert total.total_curtailed_kw == 35
    assert total.total_used_kw == 265
    assert pv.used_generation_kw == 175
    assert turbine.used_generation_kw == 90
    assert [item.asset_id for item in total.by_asset] == ["PV01", "W01"]
    assert total.metadata["origin"] == "derived"


def test_zero_and_full_curtailment_conditions():
    total = combine_renewables([solar().evaluate(0, timestamp_utc=STAMP), wind().evaluate(0, timestamp_utc=STAMP)])
    assert total.total_used_kw == total.total_available_kw == total.total_curtailed_kw == 0
    assert solar().evaluate(100, timestamp_utc=STAMP, curtailment_kw=100).used_generation_kw == 0


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True, "10", None])
def test_invalid_physical_power_rejected(value):
    with pytest.raises(ComponentValidationError, match="available_generation_kw"):
        solar().evaluate(value, timestamp_utc=STAMP)


@pytest.mark.parametrize("value", [-1, 101, float("inf"), True])
def test_invalid_curtailment_rejected_without_clipping(value):
    with pytest.raises(ComponentValidationError, match="curtailment"):
        solar().evaluate(100, timestamp_utc=STAMP, curtailment_kw=value)


@pytest.mark.parametrize("component,value", [(solar, 301), (wind, 151)])
def test_nameplate_capacity_enforced(component, value):
    with pytest.raises(ComponentValidationError, match="exceeds capacity_kw"):
        component().evaluate(value, timestamp_utc=STAMP)


@pytest.mark.parametrize("capacity", [0, -1, float("nan"), True])
def test_invalid_capacity_rejected(capacity):
    with pytest.raises(ComponentValidationError, match="capacity_kw"):
        SolarPV("PV01", capacity, provenance(), capacity_origin="hypothetical_example")


@pytest.mark.parametrize("metadata,expected", [
    (provenance("measured"), "measured"),
    (provenance("derived", derived_kind="model"), "modeled"),
    (provenance("simulated", synthetic=True), "synthetic"),
    (provenance("derived", derived_kind="model", synthetic=True), "synthetic"),
])
def test_input_origin_preserved_and_output_identified_as_derived(metadata, expected):
    result = solar(metadata).evaluate(10, timestamp_utc=STAMP)
    assert result.metadata["origin"] == "derived"
    assert result.metadata["derived_kind"] == "simulator"
    details = result.metadata["source_metadata"]
    assert details["input_provenance"] == metadata
    assert details["available_power_origin"] == expected
    assert details["rating"]["origin"] == "hypothetical_example"


def test_provenance_and_output_are_defensively_copied():
    metadata = provenance()
    before = deepcopy(metadata)
    component = solar(metadata)
    metadata["source_metadata"]["note"] = "changed"
    result = component.evaluate(10, timestamp_utc=STAMP)
    assert result.metadata["source_metadata"]["input_provenance"] == before
    returned = result.metadata
    returned["source_metadata"]["input_provenance"]["origin"] = "measured"
    assert component.provenance == before
    with pytest.raises(FrozenInstanceError):
        result.used_generation_kw = 0


def test_existing_hourly_asset_data_used_without_redistribution():
    observation = SimpleNamespace(timestamp_utc=STAMP,
                                  pv_asset_powers_kw={"PV01": 30, "PV02": 40},
                                  wind_asset_powers_kw={"W01": 20})
    assert solar().read(observation).available_generation_kw == 30
    assert wind().read(observation, curtailment_kw=5).used_generation_kw == 15
    assert observation.pv_asset_powers_kw == {"PV01": 30, "PV02": 40}


def test_missing_power_does_not_trigger_weather_conversion_or_zero_fill():
    observation = SimpleNamespace(timestamp_utc=STAMP, pv_asset_powers_kw={}, irradiance_w_m2=1000)
    with pytest.raises(ComponentValidationError, match="Missing observed solar power"):
        solar().read(observation)


def test_combination_rejects_duplicate_assets_and_mixed_timestamps():
    first = solar().evaluate(10, timestamp_utc=STAMP)
    with pytest.raises(ComponentValidationError, match="double-count"):
        combine_renewables([first, first])
    second = wind().evaluate(5, timestamp_utc="2026-01-01T01:00:00Z")
    with pytest.raises(ComponentValidationError, match="share one hourly timestamp"):
        combine_renewables([first, second])


@pytest.mark.parametrize("stamp", ["2026-01-01", "2026-01-01T00:30:00Z", "2026-01-01T00:00:00", "2026-02-30T00:00:00Z"])
def test_invalid_timestamps_rejected(stamp):
    with pytest.raises(ComponentValidationError, match="timestamp_utc"):
        solar().evaluate(5, timestamp_utc=stamp)


def test_missing_or_contradictory_provenance_rejected():
    with pytest.raises(ComponentValidationError):
        solar({})
    with pytest.raises(ComponentValidationError, match="simultaneously"):
        solar(provenance("measured", synthetic=True))
    with pytest.raises(ComponentValidationError, match="Parameter origin"):
        SolarPV("PV01", 300, provenance(), capacity_origin="unknown")
