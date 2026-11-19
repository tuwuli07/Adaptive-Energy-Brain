"""Comprehensive unit tests for the standalone MPC Linear Programming solver.

Tests cover:
  1. Feasibility on standard 24-hour profile.
  2. Correct energy balance (power balance at each step).
  3. Correct battery state transitions (DC energy dynamics).
  4. SOC bounds enforcement.
  5. Charging and discharging power limits.
  6. Efficiency losses.
  7. Degradation accounting.
  8. Import and export cost calculations.
  9. Terminal energy valuation sensitivity.
  10. Flat-tariff behavior (zero unprofitable cycling).
  11. Profitable time-of-use arbitrage.
  12. Infeasible-input handling.
  13. Solver failure handling.
  14. Simultaneous-flow detection and verification.
  15. Agreement with simulator physics for a feasible one-step dispatch.
  16. Solver runtime measurement.
"""

import math
import time
import numpy as np
import pytest

from src.control.mpc_solver import (
    MPCInputProfile,
    MPCSolverConfig,
    load_mpc_config,
    solve_mpc_linear_program,
)
from src.simulation.actions import BatteryAction
from src.simulation.config import load_simulation_config
from src.simulation.data_loader import HourlyObservation, SimulationData
from src.simulation.demand import CampusDemand
from src.simulation.environment import MicrogridEnvironment
from src.simulation.metrics import load_cost_config
from src.simulation.renewables import SolarPV, WindGeneration


@pytest.fixture
def sim_config():
    return load_simulation_config("configs/simulation.yaml")


@pytest.fixture
def cost_config():
    return load_cost_config("configs/simulation_costs.yaml")


@pytest.fixture
def mpc_config():
    return load_mpc_config("configs/mpc_controller.yaml")


@pytest.fixture
def standard_24h_profile():
    """Standard 24-hour synthetic profile with TOU pricing."""
    h = 24
    # Peak price 0.18 between 11:00 and 16:00 (hours 11-15), 0.12 otherwise
    prices = [0.18 if 11 <= i <= 15 else 0.12 for i in range(h)]
    export_prices = [0.04] * h
    # Moderate daytime solar profile
    solar = [0.0] * 6 + [50.0, 100.0, 150.0, 200.0, 220.0, 200.0, 150.0, 100.0, 50.0] + [0.0] * 9
    wind = [30.0] * h
    demand = [300.0] * h
    return MPCInputProfile(
        initial_energy_kwh=250.0,
        demand_kw=tuple(demand),
        solar_generation_kw=tuple(solar),
        wind_generation_kw=tuple(wind),
        import_price=tuple(prices),
        export_price=tuple(export_prices),
    )


def test_1_feasibility_standard_24h(standard_24h_profile, sim_config, cost_config, mpc_config):
    """1. Test that the solver finds a feasible optimal solution on a standard 24h profile."""
    res = solve_mpc_linear_program(standard_24h_profile, sim_config, cost_config, mpc_config)
    assert res.success is True
    assert res.diagnostics.success is True
    assert res.diagnostics.status == 0
    assert len(res.planned_charge_kw) == 24
    assert len(res.planned_discharge_kw) == 24
    assert len(res.planned_energy_kwh) == 25


def test_2_correct_power_balance(standard_24h_profile, sim_config, cost_config, mpc_config):
    """2. Test that power balance is strictly satisfied at every hour in the horizon."""
    res = solve_mpc_linear_program(standard_24h_profile, sim_config, cost_config, mpc_config)
    assert res.success is True
    h = standard_24h_profile.horizon_hours
    for k in range(h):
        gen = standard_24h_profile.solar_generation_kw[k] + standard_24h_profile.wind_generation_kw[k]
        p_c = res.planned_charge_kw[k]
        p_d = res.planned_discharge_kw[k]
        p_imp = res.planned_grid_import_kw[k]
        p_exp = res.planned_grid_export_kw[k]
        p_curt = res.planned_curtailment_kw[k]
        load = standard_24h_profile.demand_kw[k]

        bus_lhs = gen + p_imp + p_d
        bus_rhs = load + p_c + p_exp + p_curt
        assert abs(bus_lhs - bus_rhs) < 1e-4, f"Power balance residual at step {k} exceeds tolerance"


def test_3_correct_battery_state_transitions(standard_24h_profile, sim_config, cost_config, mpc_config):
    """3. Test that DC stored energy transition matches the physical battery formula."""
    res = solve_mpc_linear_program(standard_24h_profile, sim_config, cost_config, mpc_config)
    assert res.success is True
    dt = mpc_config.time_step_hours
    eta_c = sim_config.battery.charge_efficiency
    eta_d = sim_config.battery.discharge_efficiency

    energies = res.planned_energy_kwh
    assert energies[0] == standard_24h_profile.initial_energy_kwh

    for k in range(standard_24h_profile.horizon_hours):
        e_k = energies[k]
        e_next_expected = e_k + (eta_c * res.planned_charge_kw[k] - res.planned_discharge_kw[k] / eta_d) * dt
        assert abs(energies[k + 1] - e_next_expected) < 1e-4


def test_4_soc_bounds_enforced(standard_24h_profile, sim_config, cost_config, mpc_config):
    """4. Test that battery energy and SOC remain strictly within physical limits."""
    res = solve_mpc_linear_program(standard_24h_profile, sim_config, cost_config, mpc_config)
    assert res.success is True
    e_min = sim_config.battery.capacity_kwh * sim_config.battery.soc_min
    e_max = sim_config.battery.capacity_kwh * sim_config.battery.soc_max

    for e in res.planned_energy_kwh:
        assert e >= e_min - 1e-5
        assert e <= e_max + 1e-5

    for soc in res.planned_soc:
        assert soc >= sim_config.battery.soc_min - 1e-6
        assert soc <= sim_config.battery.soc_max + 1e-6


def test_5_power_limits_enforced(standard_24h_profile, sim_config, cost_config, mpc_config):
    """5. Test that charging and discharging AC powers respect maximum inverter ratings."""
    res = solve_mpc_linear_program(standard_24h_profile, sim_config, cost_config, mpc_config)
    assert res.success is True
    p_c_max = sim_config.battery.charge_limit_kw
    p_d_max = sim_config.battery.discharge_limit_kw

    for p_c in res.planned_charge_kw:
        assert 0.0 <= p_c <= p_c_max + 1e-5
    for p_d in res.planned_discharge_kw:
        assert 0.0 <= p_d <= p_d_max + 1e-5


def test_6_efficiency_losses_penalized(sim_config, cost_config, mpc_config):
    """6. Test that round-trip efficiency losses discourage useless cycling."""
    h = 24
    # Identical flat tariff throughout: no arbitrage possible
    flat_price = [0.15] * h
    flat_demand = [100.0] * h
    profile = MPCInputProfile(
        initial_energy_kwh=250.0,
        demand_kw=tuple(flat_demand),
        solar_generation_kw=tuple([0.0] * h),
        wind_generation_kw=tuple([0.0] * h),
        import_price=tuple(flat_price),
        export_price=tuple([0.0] * h),
    )
    res = solve_mpc_linear_program(profile, sim_config, cost_config, mpc_config)
    assert res.success is True
    # With flat prices and zero renewable surplus, optimal battery action is 0 kW idle at all steps
    assert max(res.planned_charge_kw) < 1e-5
    assert max(res.planned_discharge_kw) < 1e-5
    assert abs(res.first_requested_power_kw) < 1e-5


def test_7_degradation_accounting(standard_24h_profile, sim_config, cost_config, mpc_config):
    """7. Test that degradation cost equals c_deg * total throughput."""
    res = solve_mpc_linear_program(standard_24h_profile, sim_config, cost_config, mpc_config)
    assert res.success is True
    c_deg = cost_config.battery_degradation_per_kwh_throughput
    dt = mpc_config.time_step_hours
    total_tp = sum(res.planned_charge_kw) * dt + sum(res.planned_discharge_kw) * dt
    expected_deg_cost = c_deg * total_tp
    assert abs(res.degradation_cost_usd - expected_deg_cost) < 1e-4


def test_8_import_export_cost_calculations(standard_24h_profile, sim_config, cost_config, mpc_config):
    """8. Test that grid import costs and export revenues match planned exchanges."""
    res = solve_mpc_linear_program(standard_24h_profile, sim_config, cost_config, mpc_config)
    assert res.success is True
    dt = mpc_config.time_step_hours
    exp_imp = sum(p * price for p, price in zip(res.planned_grid_import_kw, standard_24h_profile.import_price)) * dt
    exp_rev = sum(p * price for p, price in zip(res.planned_grid_export_kw, standard_24h_profile.export_price)) * dt
    assert abs(res.import_cost_usd - exp_imp) < 1e-4
    assert abs(res.export_revenue_usd - exp_rev) < 1e-4
    assert abs(res.total_cost_usd - (res.import_cost_usd - res.export_revenue_usd + res.degradation_cost_usd)) < 1e-4


def test_9_terminal_energy_valuation_sensitivity(standard_24h_profile, sim_config, cost_config, mpc_config):
    """9. Test sensitivity of terminal battery energy to terminal valuation v_term."""
    # When v_term = 0.0, the battery has no incentive to preserve energy and will dump at peak
    res_zero = solve_mpc_linear_program(
        standard_24h_profile, sim_config, cost_config, mpc_config, override_terminal_value=0.0
    )
    assert res_zero.success is True

    # When v_term = 0.20 (higher than peak price 0.18), the battery strongly values retention
    res_high = solve_mpc_linear_program(
        standard_24h_profile, sim_config, cost_config, mpc_config, override_terminal_value=0.20
    )
    assert res_high.success is True

    # High terminal value must retain at least as much terminal energy as zero terminal value
    assert res_high.planned_energy_kwh[-1] >= res_zero.planned_energy_kwh[-1] - 1e-5


def test_10_flat_tariff_behavior(sim_config, cost_config, mpc_config):
    """10. Test that under a flat tariff without renewable surplus, battery remains idle."""
    h = 24
    profile = MPCInputProfile(
        initial_energy_kwh=250.0,
        demand_kw=tuple([150.0] * h),
        solar_generation_kw=tuple([0.0] * h),
        wind_generation_kw=tuple([0.0] * h),
        import_price=tuple([0.14] * h),
        export_price=tuple([0.04] * h),
    )
    res = solve_mpc_linear_program(profile, sim_config, cost_config, mpc_config)
    assert res.success is True
    # All battery power requests should be identically zero
    assert max(res.planned_charge_kw) < 1e-5
    assert max(res.planned_discharge_kw) < 1e-5
    assert abs(res.first_requested_power_kw) < 1e-5


def test_11_profitable_time_of_use_arbitrage(sim_config, cost_config, mpc_config):
    """11. Test that when price difference exceeds round-trip degradation cost, TOU arbitrage occurs."""
    h = 24
    # First 4 hours very cheap (0.05 $/kWh), next 4 hours very expensive (0.35 $/kWh)
    prices = [0.05] * 4 + [0.35] * 4 + [0.15] * 16
    profile = MPCInputProfile(
        initial_energy_kwh=100.0,  # low initial SOC to allow charging
        demand_kw=tuple([200.0] * h),
        solar_generation_kw=tuple([0.0] * h),
        wind_generation_kw=tuple([0.0] * h),
        import_price=tuple(prices),
        export_price=tuple([0.0] * h),
    )
    res = solve_mpc_linear_program(profile, sim_config, cost_config, mpc_config)
    assert res.success is True
    # Off-peak period (hours 0-3 at 0.05 $/kWh) must charge to full capacity
    offpeak_charge = sum(res.planned_charge_kw[:4])
    assert offpeak_charge > 300.0
    # Battery must hit maximum storage (450 kWh) by the start of the peak period (hour 4)
    assert abs(res.planned_energy_kwh[4] - 450.0) < 1e-4
    # Peak period (hours 4-7 at 0.35 $/kWh) must discharge fully to reserve
    peak_discharge = sum(res.planned_discharge_kw[4:8])
    assert peak_discharge > 300.0
    assert abs(res.planned_energy_kwh[7] - 50.0) < 1e-4


def test_12_infeasible_input_handling(sim_config, cost_config, mpc_config):
    """12. Test graceful handling when demand exceeds total possible supply capacity."""
    h = 24
    # Huge demand of 5000 kW per hour, exceeding grid import (2000 kW) + battery discharge (150 kW)
    profile = MPCInputProfile(
        initial_energy_kwh=250.0,
        demand_kw=tuple([5000.0] * h),
        solar_generation_kw=tuple([0.0] * h),
        wind_generation_kw=tuple([0.0] * h),
        import_price=tuple([0.15] * h),
        export_price=tuple([0.0] * h),
    )
    res = solve_mpc_linear_program(profile, sim_config, cost_config, mpc_config)
    assert res.success is False
    assert res.diagnostics.success is False
    assert res.first_action.requested_power_kw == 0.0
    assert np.isnan(res.net_objective_usd)


def test_13_solver_failure_handling(sim_config, cost_config, mpc_config):
    """13. Test that invalid inputs (e.g. empty or non-matching profile) raise clear errors before solve."""
    with pytest.raises(ValueError, match="Profile length mismatch"):
        MPCInputProfile(
            initial_energy_kwh=250.0,
            demand_kw=(100.0, 100.0),
            solar_generation_kw=(0.0,),  # mismatched length
            wind_generation_kw=(0.0, 0.0),
            import_price=(0.12, 0.12),
            export_price=(0.04, 0.04),
        )


def test_14_simultaneous_flow_detection(standard_24h_profile, sim_config, cost_config, mpc_config):
    """14. Test simultaneous flow diagnostics under standard conditions and edge cases."""
    # (a) Standard TOU profile: flows must be strictly 0.0
    res = solve_mpc_linear_program(standard_24h_profile, sim_config, cost_config, mpc_config)
    assert res.success is True
    assert res.diagnostics.has_simultaneous_battery_flow is False
    assert res.diagnostics.max_simultaneous_battery_kw < 1e-5
    assert res.diagnostics.has_simultaneous_grid_flow is False
    assert res.diagnostics.max_simultaneous_grid_kw < 1e-5

    # (b) Boundary saturation with extreme renewable surplus: flows must remain 0.0
    surplus_prof = MPCInputProfile(
        initial_energy_kwh=450.0,
        demand_kw=tuple([50.0] * 24),
        solar_generation_kw=tuple([800.0] * 24),
        wind_generation_kw=tuple([0.0] * 24),
        import_price=tuple([0.12] * 24),
        export_price=tuple([0.0] * 24),
    )
    res_surplus = solve_mpc_linear_program(surplus_prof, sim_config, cost_config, mpc_config)
    assert res_surplus.success is True
    assert res_surplus.diagnostics.has_simultaneous_battery_flow is False
    assert res_surplus.diagnostics.has_simultaneous_grid_flow is False

    # (c) Export arbitrage edge case (p_export > p_import): triggers grid simultaneous flow diagnostic
    arb_prof = MPCInputProfile(
        initial_energy_kwh=250.0,
        demand_kw=tuple([200.0] * 24),
        solar_generation_kw=tuple([0.0] * 24),
        wind_generation_kw=tuple([0.0] * 24),
        import_price=tuple([0.10] * 24),
        export_price=tuple([0.20] * 24),
    )
    res_arb = solve_mpc_linear_program(arb_prof, sim_config, cost_config, mpc_config)
    assert res_arb.success is True
    assert res_arb.diagnostics.has_simultaneous_grid_flow is True
    assert res_arb.diagnostics.max_simultaneous_grid_kw > 10.0

    # (d) Deeply negative import price edge case (p_import = -1.0 $/kWh): triggers battery simultaneous flow
    neg_prof = MPCInputProfile(
        initial_energy_kwh=450.0,
        demand_kw=tuple([0.0] * 24),
        solar_generation_kw=tuple([0.0] * 24),
        wind_generation_kw=tuple([0.0] * 24),
        import_price=tuple([-1.0] * 24),
        export_price=tuple([0.0] * 24),
    )
    # Under strict_exclusivity=True (default), simultaneous battery flow causes solver rejection and safe idle fallback
    res_neg = solve_mpc_linear_program(neg_prof, sim_config, cost_config, mpc_config)
    assert res_neg.success is False
    assert res_neg.first_action.requested_power_kw == 0.0
    assert res_neg.diagnostics.has_simultaneous_battery_flow is True
    assert "Strict exclusivity violation" in res_neg.diagnostics.message

    # Under permissive strict_exclusivity=False, solver succeeds and nets out flow
    from dataclasses import replace
    permissive_cfg = replace(mpc_config, strict_exclusivity=False)
    res_permissive = solve_mpc_linear_program(neg_prof, sim_config, cost_config, permissive_cfg)
    assert res_permissive.success is True
    assert res_permissive.diagnostics.has_simultaneous_battery_flow is True
    assert isinstance(res_permissive.first_action, BatteryAction)
    assert math.isfinite(res_permissive.first_action.requested_power_kw)


def test_15_agreement_with_simulator_physics_one_step(sim_config, cost_config, mpc_config):
    """15. Test that the LP predicted one-step transition agrees with actual MicrogridEnvironment physics using a synthetic fixture."""
    prov = {
        "source_id": "synthetic_mpc_fixture",
        "origin": "simulated",
        "synthetic": True,
        "source_metadata": {"note": "Synthetic test fixture for MPC physical validation"},
    }
    # Create fully synthetic hourly observations
    rows = [
        HourlyObservation(
            timestamp_utc=f"2026-01-01T{h:02d}:00:00Z",
            pv_available_kw=50.0,
            wind_available_kw=25.0,
            campus_load_kw=180.0,
            import_price=0.12,
            export_price=0.04,
            building_loads_kw={"B01": 180.0},
            pv_asset_powers_kw={"PV01": 50.0},
            wind_asset_powers_kw={"W01": 25.0},
        )
        for h in range(4)
    ]
    dataset = SimulationData(
        rows,
        {
            "currency_code": "USD",
            "units": {"power": "kW", "prices": "USD/kWh"},
            "sources": {"price": {"record_type": "price", "metadata": prov}},
        },
    )
    solar = SolarPV("PV01", 300, prov, capacity_origin="hypothetical_example")
    wind = WindGeneration("W01", 150, prov, capacity_origin="hypothetical_example")
    demand = CampusDemand(prov, building_ids=["B01"])
    env = MicrogridEnvironment(dataset, [solar, wind], demand, sim_config, cost_config)

    obs = env.reset()
    f = obs.features()

    # Formulate 1-hour horizon from current observation
    profile = MPCInputProfile(
        initial_energy_kwh=f["battery_energy_kwh"],
        demand_kw=(f["campus_demand_kw"],),
        solar_generation_kw=(f["solar_generation_kw"],),
        wind_generation_kw=(f["wind_generation_kw"],),
        import_price=(f["grid_import_price"],),
        export_price=(0.04,),
    )
    res = solve_mpc_linear_program(profile, sim_config, cost_config, mpc_config)
    assert res.success is True

    # Step the simulator with the solver's first action
    step_result = env.step(res.first_action)
    dispatch = step_result.dispatch

    # Verify battery power executed in simulator matches LP planned power
    assert abs(dispatch.battery.actual_power_kw - res.first_requested_power_kw) < 1e-4
    # Verify next SOC matches
    assert abs(dispatch.battery.state_after.battery_soc - res.planned_soc[1]) < 1e-4
    # Verify zero violations in physical simulator
    assert len(step_result.metrics.violations) == 0


def test_16_solver_runtime_measurement(standard_24h_profile, sim_config, cost_config, mpc_config):
    """16. Empirically measure solver runtime over 20 repeated 24-step solves."""
    runtimes = []
    for _ in range(20):
        t0 = time.perf_counter()
        res = solve_mpc_linear_program(standard_24h_profile, sim_config, cost_config, mpc_config)
        elapsed = time.perf_counter() - t0
        assert res.success is True
        runtimes.append(elapsed)

    mean_rt = float(np.mean(runtimes))
    max_rt = float(np.max(runtimes))
    # Report measured timing
    print(f"\nMeasured 24-step HiGHS solve time: mean = {mean_rt * 1000:.2f} ms, max = {max_rt * 1000:.2f} ms")
    assert mean_rt > 0.0
    assert np.isfinite(mean_rt)
