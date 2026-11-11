"""Deterministic feasibility dispatch and complete AC/stored-energy accounting."""

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import random

import pytest

from src.simulation.config import load_simulation_config
from src.simulation.data_loader import HourlyObservation
from src.simulation.demand import CampusDemand
from src.simulation.dispatch import DispatchEngine, DispatchError, DispatchInfeasibleError
from src.simulation.grid import Grid, GridError
from src.simulation.renewables import SolarPV, WindGeneration


def provenance():
    return {"source_id": "dispatch_fixture", "origin": "simulated", "synthetic": True,
            "source_metadata": {"note": "Hypothetical unit-test observations only"}}


def make_engine(*, imports=2000, exports=500, soc=0.5, handling="error",
                curtailment=True, missing_export="error"):
    config = load_simulation_config()
    config = replace(config, battery=replace(config.battery, initial_soc=soc),
                     grid=replace(config.grid, import_limit_kw=imports, export_limit_kw=exports,
                                  missing_export_price_handling=missing_export),
                     unserved_load=replace(config.unserved_load, handling=handling),
                     renewable_curtailment=replace(config.renewable_curtailment, enabled=curtailment))
    # Larger renewable ratings are labeled fixture assumptions, not real ratings.
    components = [SolarPV("PV01", 1000, provenance(), capacity_origin="hypothetical_example"),
                  WindGeneration("W01", 500, provenance(), capacity_origin="hypothetical_example")]
    demand = CampusDemand(provenance(), building_ids=["B01", "B02"])
    return DispatchEngine(components, demand, config)


def observation(load=100, pv=0, wind=0, *, hour=0, export_price=0.04):
    stamp = (datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(hours=hour)).isoformat().replace("+00:00", "Z")
    return HourlyObservation(stamp, pv, wind, load, 0.1, export_price,
                             {"B01": load * 0.4, "B02": load * 0.6}, {"PV01": pv}, {"W01": wind})


def assert_conservation(result):
    assert abs(result.balance_residual_kw) <= 1e-6
    assert abs(result.energy_balance_residual_kwh) <= 1e-6
    assert result.demand.campus_load_kw == pytest.approx(result.served_demand_kw + result.unserved_demand_kw)
    supply = result.grid.import_power_kw + result.renewables.total_used_kw + result.battery.discharge_power_kw
    consumed = result.served_demand_kw + result.battery.charge_power_kw + result.grid.export_power_kw
    assert supply == pytest.approx(consumed, abs=1e-6)
    assert not (result.grid.import_power_kw > 0 and result.grid.export_power_kw > 0)
    assert not (result.battery.charge_power_kw > 0 and result.battery.discharge_power_kw > 0)
    assert result.renewables.total_used_kw + result.renewables.total_curtailed_kw == pytest.approx(result.renewables.total_available_kw)


def test_renewable_surplus_charges_requested_battery_and_exports_remainder():
    engine = make_engine()
    result = engine.step(observation(load=100, pv=300, wind=100), -150)
    assert result.requested_battery_power_kw == result.actual_battery_power_kw == -150
    assert result.battery.charge_power_kw == 150
    assert result.grid.export_power_kw == 150
    assert result.battery.state_after.battery_energy_kwh == 392.5
    assert result.renewables.total_curtailed_kw == 0
    assert_conservation(result)


def test_deficit_supplied_by_requested_battery_then_grid():
    result = make_engine().step(observation(load=500, pv=80, wind=20), 150)
    assert result.battery.discharge_power_kw == 150
    assert result.grid.import_power_kw == 250
    assert result.battery.efficiency_loss_kwh == pytest.approx(150 / 0.95 - 150)
    assert_conservation(result)


def test_zero_generation_and_zero_flow_cases():
    result = make_engine().step(observation(load=100))
    assert result.grid.import_power_kw == 100
    assert result.battery.actual_power_kw == 0
    assert_conservation(result)
    zero = make_engine().step(observation(load=0))
    assert zero.grid.actual_power_kw == zero.battery.actual_power_kw == 0
    assert zero.battery.state_after.battery_energy_kwh == 250
    assert_conservation(zero)


def test_requested_power_saturates_and_adjustments_recorded():
    result = make_engine().step(observation(load=500), 1000)
    assert result.requested_battery_power_kw == 1000
    assert result.actual_battery_power_kw == 150
    assert "discharge_power_limit" in {item.code for item in result.adjustments}
    assert_conservation(result)


def test_charging_request_reversed_only_when_grid_limit_requires_discharge():
    result = make_engine(imports=200).step(observation(load=300), -100)
    assert result.actual_battery_power_kw == 100
    assert result.grid.import_power_kw == 200
    assert result.unserved_demand_kw == 0
    assert "battery_adjusted_for_grid_feasibility" in {item.code for item in result.adjustments}
    assert_conservation(result)


def test_export_limit_reduces_discharge_before_curtailment_is_needed():
    result = make_engine(exports=100).step(observation(load=0, pv=100), 150)
    assert result.actual_battery_power_kw == 0
    assert result.grid.export_power_kw == 100
    assert result.renewables.total_curtailed_kw == 0
    assert_conservation(result)


def test_full_battery_excess_is_curtailed_proportionally():
    result = make_engine(exports=100, soc=0.9).step(observation(load=0, pv=300, wind=150))
    assert result.actual_battery_power_kw == 0
    assert result.grid.export_power_kw == pytest.approx(100)
    assert result.renewables.total_curtailed_kw == pytest.approx(350)
    outputs = {item.asset_id: item for item in result.renewable_outputs}
    assert outputs["PV01"].curtailed_kw == pytest.approx(350 * 2 / 3)
    assert outputs["W01"].curtailed_kw == pytest.approx(350 / 3)
    assert_conservation(result)


def test_zero_grid_limits_can_force_charging_and_curtailment():
    result = make_engine(imports=0, exports=0, soc=0.1).step(observation(load=100, pv=300))
    assert result.actual_battery_power_kw == -150
    assert result.grid.actual_power_kw == 0
    assert result.renewables.total_curtailed_kw == 50
    assert_conservation(result)


def test_curtailment_restored_before_declaring_shortfall():
    result = make_engine(imports=0).step(observation(load=400, pv=300, wind=100),
                                       curtailment_by_asset_kw={"PV01": 300, "W01": 100})
    assert result.unserved_demand_kw == 0
    assert result.actual_battery_power_kw == 150
    assert result.renewables.total_curtailed_kw == 150
    assert "curtailment_restored_for_supply" in {item.code for item in result.adjustments}
    assert_conservation(result)


def test_curtailment_request_clipped_to_available_with_explicit_trace():
    result = make_engine().step(observation(load=100, pv=300, wind=100), curtailment_by_asset_kw={"PV01": 1000})
    assert result.requested_curtailment_by_asset_kw["PV01"] == 1000
    assert result.renewables.total_curtailed_kw == 300
    assert "curtailment_limited_by_availability" in {item.code for item in result.adjustments}
    assert_conservation(result)


def test_true_shortfall_default_error_records_diagnostic_without_commit():
    engine = make_engine(imports=100)
    before = engine.state
    with pytest.raises(DispatchInfeasibleError) as error:
        engine.step(observation(load=500, pv=80, wind=20))
    assert error.value.unserved_demand_kw == 150
    assert engine.state == before and not engine.results


def test_record_mode_reports_unserved_without_fictitious_supply_or_charging():
    engine = make_engine(imports=100, handling="record")
    result = engine.step(observation(load=500, pv=80, wind=20), -100)
    assert result.grid.import_power_kw == 100
    assert result.actual_battery_power_kw == 150
    assert result.served_demand_kw == 350
    assert result.unserved_demand_kw == result.unserved_energy_kwh == 150
    assert result.demand.campus_load_kw == 500
    assert engine.state.unserved_energy_kwh == 150
    assert result.step_cost == 10 # No arbitrary shortage cost/reward is invented.
    assert_conservation(result)


def test_empty_battery_and_disconnected_grid_record_full_unserved_demand():
    result = make_engine(imports=0, exports=0, soc=0.1, handling="record").step(observation(load=100))
    assert result.unserved_demand_kw == 100
    assert result.served_demand_kw == result.grid.actual_power_kw == result.actual_battery_power_kw == 0
    assert_conservation(result)


def test_disabled_curtailment_rejects_unresolvable_surplus_atomically():
    engine = make_engine(exports=0, soc=0.9, curtailment=False)
    before = engine.state
    with pytest.raises(DispatchInfeasibleError, match="curtailment is disabled"):
        engine.step(observation(load=0, pv=300))
    assert engine.state == before


@pytest.mark.parametrize("policy", ["disable_export", "zero_compensation", "error"])
def test_missing_export_policy_is_respected_before_energy_balance_commit(policy):
    engine = make_engine(missing_export=policy)
    before = engine.state
    if policy == "error":
        with pytest.raises(DispatchError, match="export_price"):
            engine.step(observation(load=100, pv=300, wind=100, export_price=None))
        assert engine.state == before
    else:
        result = engine.step(observation(load=100, pv=300, wind=100, export_price=None))
        if policy == "disable_export":
            assert result.grid.export_power_kw == 0
            assert result.battery.charge_power_kw == 150
            assert result.renewables.total_curtailed_kw == 150
        else:
            assert result.grid.export_power_kw == 300
            assert result.grid.export_price_assumed_zero
        assert_conservation(result)


def test_simultaneous_battery_requests_and_invalid_values_leave_engine_unchanged():
    engine = make_engine()
    before = engine.state
    with pytest.raises(DispatchError, match="Simultaneous"):
        engine.step(observation(), charge_kw=10, discharge_kw=10)
    with pytest.raises(DispatchError, match="finite"):
        engine.step(observation(), float("nan"))
    with pytest.raises(DispatchError, match="capacity"):
        engine.step(observation(pv=1001))
    assert engine.state == before and not engine.results


def test_bad_aggregate_or_unknown_asset_is_not_silently_ignored():
    engine = make_engine()
    bad = observation(pv=20)
    bad = replace(bad, pv_available_kw=30)
    with pytest.raises(DispatchError, match="aggregate"):
        engine.step(bad)
    bad = replace(observation(), pv_asset_powers_kw={"PV01": 0, "PV02": 0})
    with pytest.raises(DispatchError, match="registry"):
        engine.step(bad)


def test_timestamp_gap_and_bad_currency_rejected_without_advancing():
    engine = make_engine()
    engine.step(observation())
    before = engine.state
    with pytest.raises(DispatchError, match="timestamp"):
        engine.step(observation(hour=2))
    with pytest.raises(DispatchError, match="currency"):
        engine.step(observation(hour=1), currency_code="EUR")
    assert engine.state == before and len(engine.results) == 1


def test_grid_commit_failure_does_not_half_advance_battery(monkeypatch):
    engine = make_engine()
    before = engine.state
    def fail(self, transaction):
        raise GridError("injected commit failure")
    monkeypatch.setattr(Grid, "commit", fail)
    with pytest.raises(DispatchError, match="injected"):
        engine.step(observation(load=300), 100)
    assert engine.state == before and not engine.results


def test_corrupt_candidate_fails_conservation_before_commit(monkeypatch):
    engine = make_engine()
    before = engine.state
    original = Grid.preview
    def corrupt(self, requested_power_kw=None, **kwargs):
        candidate = original(self, requested_power_kw, **kwargs)
        return replace(candidate, import_power_kw=candidate.import_power_kw + 1) if requested_power_kw else candidate
    monkeypatch.setattr(Grid, "preview", corrupt)
    with pytest.raises(DispatchError, match="Power balance"):
        engine.step(observation(load=100))
    assert engine.state == before


def test_determinism_and_input_preservation():
    first, second = make_engine(), make_engine()
    for hour in range(20):
        sample = observation(load=300, pv=100, wind=50, hour=hour)
        before = deepcopy(sample)
        request = -100 if hour % 2 == 0 else 150
        assert first.step(sample, request) == second.step(sample, request)
        assert sample == before
    assert first.state == second.state


def test_random_stress_requests_keep_bounds_and_energy_accounting():
    engine = make_engine(imports=200, exports=100, handling="record")
    rng = random.Random(42)
    unserved = 0
    for hour in range(200):
        sample = observation(load=rng.uniform(0, 1500), pv=rng.uniform(0, 800), wind=rng.uniform(0, 300), hour=hour)
        result = engine.step(sample, rng.uniform(-500, 500),
                             curtailment_by_asset_kw={"PV01": sample.pv_available_kw * rng.random(), "W01": sample.wind_available_kw * rng.random()})
        assert 0 <= result.grid.import_power_kw <= 200
        assert 0 <= result.grid.export_power_kw <= 100
        assert 50 <= result.battery.state_after.battery_energy_kwh <= 450
        assert 0 <= result.unserved_demand_kw <= sample.campus_load_kw
        assert_conservation(result)
        unserved += result.unserved_energy_kwh
    assert engine.state.unserved_energy_kwh == pytest.approx(unserved)


def test_reset_is_atomic_and_clears_episode_counters():
    engine = make_engine(imports=0, handling="record")
    engine.step(observation(load=500))
    before = engine.state
    with pytest.raises(ValueError):
        # Invalid reset must preserve both component states and episode counters.
        engine.reset(initial_soc=2)
    assert engine.state == before
    engine.reset()
    assert engine.state.step_count == engine.state.unserved_energy_kwh == 0
    assert engine.state.battery.battery_energy_kwh == 250
    assert engine.state.grid.step_count == 0
    assert not engine.results
