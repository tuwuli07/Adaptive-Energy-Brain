"""Executed-flow accounting, economic assumptions, and episode metrics."""

from dataclasses import replace

import pytest
import yaml

from src.simulation.config import load_simulation_config
from src.simulation.data_loader import HourlyObservation
from src.simulation.demand import CampusDemand
from src.simulation.dispatch import DispatchEngine
from src.simulation.metrics import (
    CostConfig, MetricsTracker, calculate_step_metrics, load_cost_config,
)
from src.simulation.renewables import SolarPV, WindGeneration


def engine(**changes):
    config = load_simulation_config()
    config = replace(config, unserved_load=replace(config.unserved_load, handling="record"),
                     grid=replace(config.grid, **changes))
    provenance = {"source_id": "metrics_fixture", "origin": "simulated", "synthetic": True,
                  "source_metadata": {"note": "Unit test only"}}
    return DispatchEngine([
        SolarPV("PV01", 300, provenance, capacity_origin="hypothetical_example"),
        WindGeneration("W01", 150, provenance, capacity_origin="hypothetical_example"),
    ], CampusDemand(provenance, building_ids=["B01"]), config)


def row(load=0, solar=0, wind=0, hour=0, price=0.1):
    return HourlyObservation(f"2026-01-01T{hour:02}:00:00Z", solar, wind, load, price, 0.04,
                             {"B01": load}, {"PV01": solar}, {"W01": wind})


def test_default_cost_configuration_is_explicit():
    costs = load_cost_config()
    assert costs.currency_code == "USD"
    assert costs.parameter_origin == "hypothetical_example"
    assert costs.battery_degradation_per_kwh_throughput == 0.02
    assert costs.unserved_demand_per_kwh == 0


@pytest.mark.parametrize("changes", [
    {"currency_code": "usd"}, {"version": True}, {"version": 2},
    {"parameter_origin": "measured"}, {"unserved_demand_per_kwh": -1},
    {"battery_degradation_per_kwh_throughput": float("nan")},
    {"battery_degradation_per_kwh_throughput": True},
])
def test_invalid_economic_assumptions_rejected(changes):
    with pytest.raises(ValueError):
        replace(load_cost_config(), **changes)


def test_yaml_loading_missing_unknown_and_duplicate_fields(tmp_path):
    path = tmp_path / "costs.yaml"
    valid = {"version": 1, "currency_code": "EUR", "parameter_origin": "hypothetical_example",
             "battery_degradation_per_kwh_throughput": 0.03, "unserved_demand_per_kwh": 1}
    path.write_text(yaml.safe_dump(valid))
    assert load_cost_config(path).currency_code == "EUR"
    for bad in ({}, {**valid, "extra": 1}):
        path.write_text(yaml.safe_dump(bad))
        with pytest.raises(ValueError, match="configuration"):
            load_cost_config(path)
    path.write_text(yaml.safe_dump(valid) + "version: 1\n")
    with pytest.raises(ValueError, match="Duplicate"):
        load_cost_config(path)


def test_zero_activity_and_undefined_utilization():
    tracker = MetricsTracker()
    assert tracker.totals.renewable_utilization is None
    metrics = tracker.record(engine().step(row()))
    assert all(value == 0 for value in vars(metrics.values).values())
    assert metrics.values.renewable_utilization is None
    assert metrics.currency_code == "USD"
    assert not metrics.violations


def test_import_degradation_and_net_cost_use_actual_energy():
    result = engine().step(row(load=500, solar=100), 100)
    metrics = calculate_step_metrics(result)
    values = metrics.values
    assert values.grid_import_energy_kwh == 300
    assert values.import_cost == 30
    assert values.battery_throughput_kwh == 100
    assert values.battery_degradation_cost == 2
    assert values.net_operating_cost == 32
    assert values.renewable_utilization == 1
    assert not metrics.violations


def test_export_and_charging_do_not_double_count_renewables():
    system = engine()
    first = system.step(row(solar=300, wind=100), -100)
    metrics = calculate_step_metrics(first).values
    assert metrics.solar_available_energy_kwh == 300
    assert metrics.wind_available_energy_kwh == 100
    assert metrics.renewable_utilized_energy_kwh == 400
    assert metrics.grid_export_energy_kwh == 300
    assert metrics.export_revenue == 12
    assert metrics.net_operating_cost == -10
    second = system.step(row(load=100, hour=1), 100)
    next_metrics = calculate_step_metrics(second).values
    assert next_metrics.renewable_available_energy_kwh == 0
    assert next_metrics.renewable_utilized_energy_kwh == 0
    assert next_metrics.battery_degradation_cost == 2


def test_curtailment_and_cumulative_energy_weighted_utilization():
    tracker = MetricsTracker()
    system = engine(export_limit_kw=0)
    first = tracker.record(system.step(row(solar=300), 0))
    assert first.values.renewable_curtailed_energy_kwh == pytest.approx(150)
    assert first.values.renewable_utilization == pytest.approx(0.5)
    second = tracker.record(system.step(row(load=100, solar=100, hour=1)))
    assert second.values.renewable_utilization == 1
    assert tracker.totals.renewable_available_energy_kwh == 400
    assert tracker.totals.renewable_utilization == pytest.approx(250 / 400)
    assert tracker.constraint_records
    assert all(not s.violations for s in tracker.steps)


def test_unserved_demand_cost_is_separate_configurable_assumption():
    result = engine(import_limit_kw=0).step(row(load=500))
    costs = replace(load_cost_config(), unserved_demand_per_kwh=2)
    values = calculate_step_metrics(result, costs).values
    assert values.unserved_demand_energy_kwh == 350
    assert values.served_demand_energy_kwh == 150
    assert values.unserved_demand_cost == 700
    assert values.battery_degradation_cost == 3
    assert values.net_operating_cost == 703


def test_proxy_increases_linearly_with_both_throughput_directions():
    costs = replace(load_cost_config(), battery_degradation_per_kwh_throughput=0.5)
    small = calculate_step_metrics(engine().step(row(load=200), 20), costs).values
    large = calculate_step_metrics(engine().step(row(solar=300), -100), costs).values
    assert small.battery_degradation_cost == 10
    assert large.battery_degradation_cost == 50


def test_duplicate_gap_currency_and_tampering_rejected_atomically():
    system, tracker = engine(), MetricsTracker()
    first = system.step(row(load=100))
    tracker.record(first)
    before = tracker.totals
    for result in (first, replace(first, timestamp_utc="2026-01-01T03:00:00Z"),
                   replace(first, grid=replace(first.grid, imported_energy_kwh=999))):
        with pytest.raises(ValueError):
            tracker.record(result)
        assert tracker.totals == before
        assert len(tracker.steps) == 1
    with pytest.raises(ValueError, match="currencies"):
        calculate_step_metrics(first, replace(load_cost_config(), currency_code="EUR"))


def test_negative_tariffs_preserve_signed_costs():
    result = engine(allow_negative_prices=True).step(row(load=100, price=-0.1))
    values = calculate_step_metrics(result).values
    assert values.import_cost == values.net_operating_cost == -10


def test_constraint_adjustments_are_not_physical_violations():
    result = engine().step(row(load=300), 1000)
    metrics = calculate_step_metrics(result)
    assert metrics.adjustments == result.adjustments
    assert metrics.adjustments
    assert not metrics.violations
    # Audit a result with a stricter recorded connection limit: flows are kept,
    # and the actual limit violation is distinct from an adjustment record.
    metadata = result.metadata
    metadata["config"]["grid"]["import_limit_kw"] = 0
    audited = calculate_step_metrics(replace(result, _metadata=metadata))
    assert "grid_import_limit" in {v.code for v in audited.violations}


def test_cumulative_cost_identity_and_input_preservation():
    system, tracker = engine(), MetricsTracker()
    original = system.state
    for hour in range(4):
        result = system.step(row(load=300, solar=100, hour=hour), -50 if hour % 2 else 50)
        before = system.state
        tracker.record(result)
        assert system.state == before
    total = tracker.totals
    assert total.net_operating_cost == pytest.approx(
        total.import_cost - total.export_revenue + total.battery_degradation_cost + total.unserved_demand_cost)
    assert total.battery_throughput_kwh == 200
    assert total.battery_degradation_cost == 4
    assert system.state != original
