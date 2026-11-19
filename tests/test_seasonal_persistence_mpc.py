"""Mandatory preflight and causality tests for SeasonalPersistenceMPCController."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import pytest

from src.baselines.base import BaselineController
from src.control.mpc_solver import load_mpc_config
from src.control.seasonal_persistence_mpc import (
    SeasonalPersistenceMPCController,
    get_scheduled_tariff,
)
from src.simulation.actions import BatteryAction
from src.simulation.config import load_simulation_config
from src.simulation.data_loader import HourlyObservation, SimulationData, load_simulation_split
from src.simulation.demand import CampusDemand
from src.simulation.environment import MicrogridEnvironment
from src.simulation.metrics import load_cost_config
from src.simulation.renewables import SolarPV, WindGeneration


@pytest.fixture
def synthetic_72h_dataset():
    """Create a 72-hour synthetic dataset (first 24h as training history, next 48h as episode)."""
    prov = {
        "source_id": "synthetic_sp_test",
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
    rows = []
    base_dt = datetime(2026, 6, 1, 0, 0, 0, tzinfo=timezone.utc)
    for h in range(72):
        dt = base_dt + timedelta(hours=h)
        ts = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        hour = dt.hour
        price = 0.18 if 11 <= hour < 16 else 0.12
        # Introduce an unannounced stochastic price spike at h=35
        if h == 35:
            price = 0.72
        solar = 120.0 if 10 <= hour <= 14 else 0.0
        wind = 25.0
        demand = 220.0 if 11 <= hour <= 16 else 90.0
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


def test_1_forecast_source_timestamps_precede_decision_time(synthetic_72h_dataset):
    """1. Verify that every forecast source timestamp strictly precedes the decision time."""
    sim_cfg = load_simulation_config()
    cost_cfg = load_cost_config()
    mpc_cfg = load_mpc_config()

    history = synthetic_72h_dataset[:24]
    episode_data = SimulationData(synthetic_72h_dataset[23:], synthetic_72h_dataset.metadata)

    ctrl = SeasonalPersistenceMPCController(history, sim_cfg, cost_cfg, mpc_cfg)

    prov = synthetic_72h_dataset.metadata["sources"]["solar"]["metadata"]
    solar = SolarPV("PV01", 150.0, prov, capacity_origin="hypothetical_example")
    wind = WindGeneration("WT01", 50.0, prov, capacity_origin="hypothetical_example")
    demand = CampusDemand(prov, building_ids=["BLD01"])
    env = MicrogridEnvironment(episode_data, [solar, wind], demand, sim_cfg, cost_cfg)

    obs = env.reset()
    ctrl.select_action(obs)

    diag = ctrl.step_diagnostics[0]
    dec_dt = datetime.fromisoformat(diag.decision_timestamp_utc.replace("Z", "+00:00"))

    for src_ts in diag.source_timestamps_utc:
        src_dt = datetime.fromisoformat(src_ts.replace("Z", "+00:00"))
        assert src_dt < dec_dt, f"Source time {src_ts} must strictly precede decision time {diag.decision_timestamp_utc}"


def test_2_future_validation_perturbations_cannot_change_current_decision(synthetic_72h_dataset):
    """2. Verify that perturbing future realized rows cannot alter the causal controller's current decision."""
    sim_cfg = load_simulation_config()
    cost_cfg = load_cost_config()
    mpc_cfg = load_mpc_config()

    history = synthetic_72h_dataset[:24]

    # Baseline environment run
    episode_data_base = SimulationData(synthetic_72h_dataset[23:48], synthetic_72h_dataset.metadata)
    prov = synthetic_72h_dataset.metadata["sources"]["solar"]["metadata"]
    solar = SolarPV("PV01", 150.0, prov, capacity_origin="hypothetical_example")
    wind = WindGeneration("WT01", 50.0, prov, capacity_origin="hypothetical_example")
    demand = CampusDemand(prov, building_ids=["BLD01"])
    env_base = MicrogridEnvironment(episode_data_base, [solar, wind], demand, sim_cfg, cost_cfg)

    ctrl_base = SeasonalPersistenceMPCController(history, sim_cfg, cost_cfg, mpc_cfg)
    obs_base = env_base.reset()
    action_base = ctrl_base.select_action(obs_base)

    # Perturbed environment run: drastically alter all future rows (demand x10, solar x0, prices x10)
    perturbed_rows = list(synthetic_72h_dataset[23:48])
    for j in range(1, len(perturbed_rows)):
        r = perturbed_rows[j]
        new_demand = r.campus_load_kw * 10.0
        from dataclasses import replace
        perturbed_rows[j] = replace(
            r,
            campus_load_kw=new_demand,
            building_loads_kw={"BLD01": new_demand},
            pv_available_kw=0.0,
            pv_asset_powers_kw={"PV01": 0.0},
            import_price=1.00,
        )
    episode_data_pert = SimulationData(perturbed_rows, synthetic_72h_dataset.metadata)
    env_pert = MicrogridEnvironment(episode_data_pert, [solar, wind], demand, sim_cfg, cost_cfg)

    ctrl_pert = SeasonalPersistenceMPCController(history, sim_cfg, cost_cfg, mpc_cfg)
    obs_pert = env_pert.reset()
    action_pert = ctrl_pert.select_action(obs_pert)

    # Future perturbations must produce ZERO change in the causal decision
    assert action_base.requested_power_kw == pytest.approx(action_pert.requested_power_kw, abs=1e-6)


def test_3_training_validation_history_continuity():
    """3. Verify that training and validation partitions connect with strictly continuous hourly timestamps."""
    root = Path(__file__).resolve().parents[1] / "data" / "processed" / "splits_hypothetical_campus_v1"
    train = load_simulation_split(root, "train")
    val = load_simulation_split(root, "validation")

    train_end_dt = datetime.fromisoformat(train[-1].timestamp_utc.replace("Z", "+00:00"))
    val_start_dt = datetime.fromisoformat(val[0].timestamp_utc.replace("Z", "+00:00"))

    assert val_start_dt - train_end_dt == timedelta(hours=1), "Training end and validation start must be exactly 1 hour apart"


def test_4_no_future_stochastic_tariff_spikes_exposed(synthetic_72h_dataset):
    """4. Verify that unannounced stochastic tariff spikes are NOT exposed to the planning LP."""
    sim_cfg = load_simulation_config()
    cost_cfg = load_cost_config()
    mpc_cfg = load_mpc_config()

    # Step at h=34, where next hour h=35 has a 0.72 $/kWh stochastic spike in realized dataset
    history = synthetic_72h_dataset[:24]
    ctrl = SeasonalPersistenceMPCController(history, sim_cfg, cost_cfg, mpc_cfg)

    # Decision at 2026-06-02T10:00:00Z (h=34), so h=35 is target step k=1
    target_ts = "2026-06-02T11:00:00Z"
    p_imp, p_exp = get_scheduled_tariff(target_ts)
    # The scheduled tariff must be 0.18 (peak), NOT 0.72 (stochastic spike)
    assert p_imp == 0.18
    assert p_imp != 0.72


def test_5_missing_history_triggers_documented_fallback():
    """5. Verify that missing history triggers the documented 1-step persistence fallback without failure."""
    sim_cfg = load_simulation_config()
    cost_cfg = load_cost_config()
    mpc_cfg = load_mpc_config()

    # Cold start: empty initial history
    ctrl = SeasonalPersistenceMPCController(initial_history=None, sim_config=sim_cfg, cost_config=cost_cfg, mpc_config=mpc_cfg)

    prov = {
        "source_id": "test",
        "origin": "simulated",
        "synthetic": True,
        "source_metadata": {"note": "test"},
    }
    meta = {
        "currency_code": "USD",
        "units": {"power": "kW", "prices": "USD/kWh"},
        "sources": {
            "load": {"record_type": "load", "metadata": prov, "entity_ids": ["B01"]},
            "solar": {"record_type": "solar", "metadata": prov, "entity_ids": ["PV01"]},
            "wind": {"record_type": "wind", "metadata": prov, "entity_ids": ["W01"]},
            "price": {"record_type": "price", "metadata": prov},
        },
        "source_origin_summary": {"load": "simulated", "solar": "simulated", "wind": "simulated", "price": "simulated"},
    }
    rows = [
        HourlyObservation(
            timestamp_utc=f"2026-01-01T{h:02d}:00:00Z",
            pv_available_kw=50.0,
            wind_available_kw=20.0,
            campus_load_kw=180.0,
            import_price=0.12,
            export_price=0.04,
            building_loads_kw={"B01": 180.0},
            pv_asset_powers_kw={"PV01": 50.0},
            wind_asset_powers_kw={"W01": 20.0},
        )
        for h in range(4)
    ]
    data = SimulationData(rows, meta)
    solar = SolarPV("PV01", 150.0, prov, capacity_origin="hypothetical_example")
    wind = WindGeneration("W01", 50.0, prov, capacity_origin="hypothetical_example")
    demand = CampusDemand(prov, building_ids=["B01"])
    env = MicrogridEnvironment(data, [solar, wind], demand, sim_cfg, cost_cfg)

    obs = env.reset()
    action = ctrl.select_action(obs)

    # Controller must successfully produce an action and report fallback usage
    assert isinstance(action, BatteryAction)
    assert ctrl.step_diagnostics[0].fallback_used is True
    # Rows from before 2026-01-01 are missing, so 23 out of 24 horizon steps fall back to 1-step persistence
    assert ctrl.step_diagnostics[0].fallback_hours_count == 23


def test_6_causal_controller_cannot_access_oracle_future_rows():
    """6. Verify that SeasonalPersistenceMPCController is completely decoupled from Oracle future rows."""
    ctrl = SeasonalPersistenceMPCController()
    assert not hasattr(ctrl, "_rows")
    assert not hasattr(ctrl, "partition_data")


def test_7_forecast_horizon_and_timestamps_match_lp_horizon(synthetic_72h_dataset):
    """7. Verify that forecast horizon length and timestamps align with the LP horizon."""
    sim_cfg = load_simulation_config()
    cost_cfg = load_cost_config()
    mpc_cfg = load_mpc_config()

    ctrl = SeasonalPersistenceMPCController(synthetic_72h_dataset[:24], sim_cfg, cost_cfg, mpc_cfg)

    prov = synthetic_72h_dataset.metadata["sources"]["solar"]["metadata"]
    solar = SolarPV("PV01", 150.0, prov, capacity_origin="hypothetical_example")
    wind = WindGeneration("WT01", 50.0, prov, capacity_origin="hypothetical_example")
    demand = CampusDemand(prov, building_ids=["BLD01"])
    env = MicrogridEnvironment(SimulationData(synthetic_72h_dataset[23:], synthetic_72h_dataset.metadata), [solar, wind], demand, sim_cfg, cost_cfg)

    obs = env.reset()
    ctrl.select_action(obs)

    diag = ctrl.step_diagnostics[0]
    assert diag.horizon_hours == 24
    assert len(diag.forecast_demand_kw) == 24
    assert len(diag.forecast_solar_kw) == 24
    assert len(diag.forecast_wind_kw) == 24
    assert len(diag.target_timestamps_utc) == 24


def test_8_first_planned_action_agrees_with_simulator_execution(synthetic_72h_dataset):
    """8. Verify that the planned first action agrees with simulator execution within tolerance."""
    sim_cfg = load_simulation_config()
    cost_cfg = load_cost_config()
    mpc_cfg = load_mpc_config()

    ctrl = SeasonalPersistenceMPCController(synthetic_72h_dataset[:24], sim_cfg, cost_cfg, mpc_cfg)

    prov = synthetic_72h_dataset.metadata["sources"]["solar"]["metadata"]
    solar = SolarPV("PV01", 150.0, prov, capacity_origin="hypothetical_example")
    wind = WindGeneration("WT01", 50.0, prov, capacity_origin="hypothetical_example")
    demand = CampusDemand(prov, building_ids=["BLD01"])
    env = MicrogridEnvironment(SimulationData(synthetic_72h_dataset[23:], synthetic_72h_dataset.metadata), [solar, wind], demand, sim_cfg, cost_cfg)

    obs = env.reset()
    action = ctrl.select_action(obs)
    step = env.step(action)

    assert abs(step.dispatch.battery.actual_power_kw - action.requested_power_kw) < 1e-4
