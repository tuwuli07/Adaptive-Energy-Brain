"""Battery AC/internal energy accounting, saturation and atomic state changes."""

from dataclasses import FrozenInstanceError, asdict, replace
import random

import pytest
import yaml

from src.simulation.battery import Battery, BatteryError
from src.simulation.config import load_simulation_config


def configured(**battery_changes):
    config = load_simulation_config()
    return replace(config, battery=replace(config.battery, **battery_changes))


def assert_accounting(result):
    expected = (result.state_before.battery_energy_kwh + result.charge_energy_kwh
                - result.discharge_energy_kwh - result.efficiency_loss_kwh
                + result.rounding_adjustment_kwh)
    assert result.state_after.battery_energy_kwh == pytest.approx(expected, abs=1e-9)
    assert result.stored_energy_change_kwh == pytest.approx(
        result.state_after.battery_energy_kwh - result.state_before.battery_energy_kwh)
    assert not (result.charge_power_kw > 0 and result.discharge_power_kw > 0)
    assert result.actual_power_kw == result.discharge_power_kw - result.charge_power_kw
    assert result.efficiency_loss_kwh >= 0


def test_loads_default_configuration_and_initial_state():
    battery = Battery()
    assert battery.energy_kwh == 250
    assert battery.soc == 0.5
    assert battery.throughput_kwh == 0
    result = battery.step()
    assert result.metadata["parameter_origin"] == "hypothetical_example"
    assert result.metadata["source_reference"] == "configs/synthetic_scenario.yaml"
    assert result.metadata["origin"] == "derived"


def test_charge_increases_energy_with_correct_efficiency():
    battery = Battery()
    result = battery.charge(150)
    assert result.requested_power_kw == result.actual_power_kw == -150
    assert battery.energy_kwh == 392.5
    assert result.updated_soc == 0.785
    assert result.charge_energy_kwh == 150
    assert result.efficiency_loss_kwh == 7.5
    assert battery.state.charge_throughput_kwh == 150
    assert battery.state.internal_energy_throughput_kwh == 142.5
    assert_accounting(result)


def test_discharge_decreases_energy_with_correct_efficiency():
    battery = Battery()
    result = battery.discharge(150)
    assert result.requested_power_kw == result.actual_power_kw == 150
    assert battery.energy_kwh == pytest.approx(250 - 150 / 0.95)
    assert result.efficiency_loss_kwh == pytest.approx(150 / 0.95 - 150)
    assert battery.state.discharge_throughput_kwh == 150
    assert_accounting(result)


def test_different_charge_and_discharge_efficiencies():
    battery = Battery(configured(charge_efficiency=0.8, discharge_efficiency=0.9))
    result = battery.charge(100)
    assert battery.energy_kwh == 330
    assert result.efficiency_loss_kwh == 20
    result = battery.discharge(100)
    assert battery.energy_kwh == pytest.approx(330 - 100 / 0.9)
    assert result.efficiency_loss_kwh == pytest.approx(100 / 0.9 - 100)
    assert_accounting(result)


def test_signed_and_separate_direction_commands_are_equivalent():
    first, second = Battery(), Battery()
    assert first.step(-100).state_after == second.step(charge_kw=100, discharge_kw=0).state_after
    assert first.step(100).state_after == second.step(charge_kw=0, discharge_kw=100).state_after


@pytest.mark.parametrize("power_request,expected,reason", [
    (-10000, -150, "charge_power_limit"), (10000, 150, "discharge_power_limit"),
    (-1e308, -150, "charge_power_limit"), (1e308, 150, "discharge_power_limit"),
])
def test_requested_and_actual_power_show_rating_saturation(power_request, expected, reason):
    result = Battery().step(power_request)
    assert result.requested_power_kw == power_request
    assert result.actual_power_kw == expected
    assert result.power_limited
    assert reason in result.limiting_reasons
    assert_accounting(result)


def test_charge_limited_by_maximum_soc():
    battery = Battery(configured(initial_soc=0.89))
    result = battery.charge(150)
    assert result.actual_power_kw == pytest.approx(-5 / 0.95)
    assert battery.energy_kwh == 450
    assert battery.soc == 0.9
    assert "max_soc" in result.limiting_reasons
    assert battery.charge(150).actual_power_kw == 0
    assert_accounting(result)


def test_discharge_limited_by_minimum_soc():
    battery = Battery(configured(initial_soc=0.11))
    result = battery.discharge(150)
    assert result.actual_power_kw == pytest.approx(5 * 0.95)
    assert battery.energy_kwh == 50
    assert battery.soc == 0.1
    assert "min_soc" in result.limiting_reasons
    assert battery.discharge(150).actual_power_kw == 0
    assert_accounting(result)


@pytest.mark.parametrize("dt", [0.25, 0.5, 2])
def test_time_step_override_scales_energy_not_power(dt):
    battery = Battery()
    result = battery.charge(100, time_step_hours=dt)
    assert result.actual_power_kw == -100
    assert result.charge_energy_kwh == 100 * dt
    assert battery.energy_kwh == pytest.approx(250 + 0.95 * 100 * dt)
    assert battery.state.elapsed_hours == dt
    assert_accounting(result)


def test_configured_half_hour_duration_and_file_loading(tmp_path):
    config = asdict(load_simulation_config())
    config["time_step_hours"] = 0.5
    path = tmp_path / "simulation.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    battery = Battery.from_config(path)
    assert battery.charge(100).time_step_hours == 0.5
    assert battery.energy_kwh == 297.5


def test_two_hour_discharge_uses_efficiency_correct_headroom():
    battery = Battery()
    limits = battery.feasible_limits(2)
    assert limits.discharge_limit_kw == 95
    result = battery.discharge(150, time_step_hours=2)
    assert result.actual_power_kw == 95
    assert battery.energy_kwh == 50
    assert_accounting(result)


def test_round_trip_and_throughput_bases_are_explicit():
    battery = Battery()
    charge = battery.charge(150)
    discharge = battery.discharge(150 * 0.95 * 0.95)
    assert battery.energy_kwh == pytest.approx(250)
    assert battery.throughput_kwh == pytest.approx(150 + 135.375)
    assert battery.state.internal_energy_throughput_kwh == pytest.approx(285)
    assert charge.efficiency_loss_kwh + discharge.efficiency_loss_kwh == pytest.approx(14.625)
    assert_accounting(charge)
    assert_accounting(discharge)


def test_perfect_efficiency_and_disabled_power_directions():
    battery = Battery(configured(charge_efficiency=1, discharge_efficiency=1))
    assert battery.charge(100).efficiency_loss_kwh == 0
    assert battery.discharge(100).efficiency_loss_kwh == 0
    assert battery.energy_kwh == 250
    disabled = Battery(configured(charge_limit_kw=0, discharge_limit_kw=0))
    assert disabled.charge(100).actual_power_kw == 0
    assert disabled.discharge(100).actual_power_kw == 0
    assert disabled.energy_kwh == 250


def test_zero_command_keeps_energy_soc_and_throughput_unchanged():
    battery = Battery(configured(initial_soc=0.3141592653589793))
    before = battery.state
    result = battery.step(0)
    assert battery.energy_kwh == before.battery_energy_kwh
    assert battery.soc == before.battery_soc
    assert battery.throughput_kwh == 0
    assert result.efficiency_loss_kwh == 0
    assert battery.state.step_count == 1
    assert battery.state.elapsed_hours == 1


@pytest.mark.parametrize("power_request", [True, "10", float("nan"), float("inf"), -float("inf"), []])
def test_invalid_signed_commands_leave_state_unchanged(power_request):
    battery = Battery()
    before = battery.state
    with pytest.raises(BatteryError):
        battery.step(power_request)
    assert battery.state == before


@pytest.mark.parametrize("dt", [0, -1, True, "1", float("nan"), float("inf")])
def test_invalid_durations_leave_state_unchanged(dt):
    battery = Battery()
    before = battery.state
    with pytest.raises(BatteryError, match="time_step_hours"):
        battery.charge(100, time_step_hours=dt)
    assert battery.state == before


def test_simultaneous_and_ambiguous_commands_rejected_atomically():
    battery = Battery()
    before = battery.state
    with pytest.raises(BatteryError, match="Simultaneous"):
        battery.step(charge_kw=50, discharge_kw=50)
    with pytest.raises(BatteryError, match="Choose either"):
        battery.step(0, charge_kw=50)
    with pytest.raises(BatteryError, match="nonnegative"):
        battery.charge(-10)
    assert battery.state == before


def test_preview_commit_stale_and_altered_transition_handling():
    battery = Battery()
    before = battery.state
    candidate = battery.preview(-50)
    assert battery.state == before
    with pytest.raises(BatteryError, match="Altered"):
        battery.commit(replace(candidate, actual_power_kw=-100))
    assert battery.state == before
    battery.commit(candidate)
    assert battery.state == candidate.state_after
    with pytest.raises(BatteryError, match="Stale"):
        battery.commit(candidate)
    candidate = battery.preview(50)
    battery.reset()
    with pytest.raises(BatteryError, match="Stale"):
        battery.commit(candidate)


def test_reset_clears_counters_and_invalid_reset_is_atomic():
    battery = Battery()
    battery.charge(100)
    before = battery.state
    with pytest.raises(BatteryError, match="initial_soc"):
        battery.reset(2)
    assert battery.state == before
    battery.reset()
    assert battery.energy_kwh == 250
    assert battery.throughput_kwh == battery.state.elapsed_hours == battery.state.step_count == 0


def test_invalid_direct_configuration_is_revalidated():
    with pytest.raises(BatteryError, match="capacity_kwh"):
        Battery(configured(capacity_kwh=0))


def test_random_request_sequence_respects_bounds_and_conservation():
    battery = Battery()
    rng = random.Random(42)
    total = 0
    for _ in range(1000):
        result = battery.step(rng.uniform(-500, 500), time_step_hours=rng.choice([0.25, 0.5, 1, 2]))
        assert 50 <= battery.energy_kwh <= 450
        assert 0.1 <= battery.soc <= 0.9
        assert 0 <= result.charge_power_kw <= 150
        assert 0 <= result.discharge_power_kw <= 150
        assert_accounting(result)
        total += result.throughput_increment_kwh
    assert battery.throughput_kwh == pytest.approx(total)


def test_state_and_metadata_are_protected():
    battery = Battery()
    result = battery.charge(100)
    with pytest.raises(FrozenInstanceError):
        battery.state.battery_soc = 2
    changed = result.metadata
    changed["battery_parameters"]["capacity_kwh"] = 1
    assert result.metadata["battery_parameters"]["capacity_kwh"] == 500


def test_numeric_overflow_does_not_commit_partial_state():
    battery = Battery()
    battery.step(0, time_step_hours=1e308)
    before = battery.state
    with pytest.raises(BatteryError, match="Cumulative"):
        battery.step(0, time_step_hours=1e308)
    assert battery.state == before
