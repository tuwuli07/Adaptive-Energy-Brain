"""Unit and integration tests for OracleMPCController."""

import math
from pathlib import Path
import pytest

from src.baselines.base import BaselineController
from src.control.oracle_mpc import OracleMPCController, OracleStepDiagnostic
from src.control.mpc_solver import MPCSolverConfig, load_mpc_config
from src.simulation.actions import BatteryAction
from src.simulation.config import load_simulation_config
from src.simulation.data_loader import HourlyObservation, SimulationData
from src.simulation.demand import CampusDemand
from src.simulation.environment import MicrogridEnvironment
from src.simulation.metrics import load_cost_config
from src.simulation.renewables import SolarPV, WindGeneration


@pytest.fixture
def synthetic_48h_dataset():
    """Create a synthetic 48-hour SimulationData fixture with realistic TOU price variations."""
    prov = {
        "source_id": "synthetic_oracle_test",
        "origin": "simulated",
        "synthetic": True,
        "source_metadata": {"scenario": {"buildings": [{"id": "BLD01"}], "solar": {"asset_id": "PV01"}, "wind": {"asset_id": "WT01"}}},
    }
    meta = {
        "currency_code": "USD",
        "units": {"power": "kW", "prices": "USD/kWh"},
        "sources": {
            "load": {"record_type": "load", "metadata": prov, "entity_ids": ["BLD01"]},
            "solar": {"record_type": "solar", "metadata": prov, "entity_ids": ["PV01"]},
            "wind": {"record_type": "wind", "metadata": prov, "entity_ids": ["WT01"]},
            "price": {"record_type": "price", "metadata": prov},
        },
        "source_origin_summary": {"load": "simulated", "solar": "simulated", "wind": "simulated", "price": "simulated"},
    }
    base_ts = "2026-06-01T"
    rows = []
    for h in range(48):
        day = 1 + h // 24
        hour = h % 24
        ts = f"2026-06-{day:02d}T{hour:02d}:00:00Z"
        # TOU pricing: peak 16-20 (0.24 $/kWh), off-peak (0.12 $/kWh)
        price = 0.24 if 16 <= hour <= 20 else 0.12
        # Solar peak midday, wind steady
        solar = 100.0 if 10 <= hour <= 15 else 0.0
        wind = 30.0
        demand = 200.0 if 14 <= hour <= 21 else 80.0
        obs = HourlyObservation(
            timestamp_utc=ts,
            pv_available_kw=solar,
            wind_available_kw=wind,
            campus_load_kw=demand,
            import_price=price,
            export_price=0.04,
            building_loads_kw={"BLD01": demand},
            pv_asset_powers_kw={"PV01": solar},
            wind_asset_powers_kw={"WT01": wind},
        )
        rows.append(obs)
    return SimulationData(rows, meta)


def test_oracle_controller_interface_and_reset(synthetic_48h_dataset):
    sim_cfg = load_simulation_config()
    cost_cfg = load_cost_config()
    mpc_cfg = load_mpc_config()

    ctrl = OracleMPCController(synthetic_48h_dataset, sim_cfg, cost_cfg, mpc_cfg)
    assert isinstance(ctrl, BaselineController)
    assert len(ctrl.solve_times) == 0
    assert len(ctrl.step_diagnostics) == 0


def test_oracle_controller_simulation_run(synthetic_48h_dataset):
    sim_cfg = load_simulation_config()
    cost_cfg = load_cost_config()
    mpc_cfg = load_mpc_config()

    prov = synthetic_48h_dataset.metadata["sources"]["solar"]["metadata"]
    solar = SolarPV("PV01", 150.0, prov, capacity_origin="hypothetical_example")
    wind = WindGeneration("WT01", 50.0, prov, capacity_origin="hypothetical_example")
    demand = CampusDemand(prov, building_ids=["BLD01"])

    env = MicrogridEnvironment(
        synthetic_48h_dataset,
        [solar, wind],
        demand,
        sim_cfg,
        cost_cfg,
    )

    ctrl = OracleMPCController(synthetic_48h_dataset, sim_cfg, cost_cfg, mpc_cfg)
    obs = env.reset()
    assert obs is not None

    step_count = 0
    while not env.terminated:
        action = ctrl.select_action(env.observation)
        step = env.step(action)
        step_count += 1

    assert step_count == 47  # 48 rows yield 47 steps
    assert len(ctrl.solve_times) == 47
    assert len(ctrl.step_diagnostics) == 47
    assert ctrl.fallback_count == 0
    assert all(d.success for d in ctrl.step_diagnostics)
    assert all(not d.has_simultaneous_battery_flow for d in ctrl.step_diagnostics)
    # Check that tail steps have shrinking horizons
    assert ctrl.step_diagnostics[-1].horizon_hours == 1
    assert ctrl.step_diagnostics[-2].horizon_hours == 2
    assert ctrl.step_diagnostics[0].horizon_hours == 24


def test_oracle_controller_fallback_on_missing_timestamp(synthetic_48h_dataset):
    sim_cfg = load_simulation_config()
    cost_cfg = load_cost_config()
    mpc_cfg = load_mpc_config()

    ctrl = OracleMPCController(synthetic_48h_dataset, sim_cfg, cost_cfg, mpc_cfg)
    prov = synthetic_48h_dataset.metadata["sources"]["solar"]["metadata"]
    solar = SolarPV("PV01", 150.0, prov, capacity_origin="hypothetical_example")
    wind = WindGeneration("WT01", 50.0, prov, capacity_origin="hypothetical_example")
    demand = CampusDemand(prov, building_ids=["BLD01"])

    env = MicrogridEnvironment(
        synthetic_48h_dataset,
        [solar, wind],
        demand,
        sim_cfg,
        cost_cfg,
    )
    obs = env.reset()
    # Modify observation timestamp to an unknown timestamp
    from dataclasses import replace
    unknown_obs = replace(obs, decision_timestamp_utc="2099-01-01T00:00:00Z")
    action = ctrl.select_action(unknown_obs)
    assert action.requested_power_kw == 0.0
    assert ctrl.fallback_count == 1


def test_oracle_controller_economic_arbitrage_behavior(synthetic_48h_dataset):
    """Verify that Oracle MPC performs active charging and discharging in response to TOU pricing."""
    sim_cfg = load_simulation_config()
    cost_cfg = load_cost_config()
    mpc_cfg = load_mpc_config()

    prov = synthetic_48h_dataset.metadata["sources"]["solar"]["metadata"]
    solar = SolarPV("PV01", 150.0, prov, capacity_origin="hypothetical_example")
    wind = WindGeneration("WT01", 50.0, prov, capacity_origin="hypothetical_example")
    demand = CampusDemand(prov, building_ids=["BLD01"])

    env = MicrogridEnvironment(
        synthetic_48h_dataset,
        [solar, wind],
        demand,
        sim_cfg,
        cost_cfg,
    )
    ctrl = OracleMPCController(synthetic_48h_dataset, sim_cfg, cost_cfg, mpc_cfg)
    obs = env.reset()

    charges = 0
    discharges = 0
    idles = 0
    while not env.terminated:
        action = ctrl.select_action(env.observation)
        p = action.requested_power_kw
        if p < -1.0:
            charges += 1
        elif p > 1.0:
            discharges += 1
        else:
            idles += 1
        env.step(action)

    assert charges > 0, "Oracle MPC must learn/plan charging during off-peak periods"
    assert discharges > 0, "Oracle MPC must learn/plan discharging during peak periods"

