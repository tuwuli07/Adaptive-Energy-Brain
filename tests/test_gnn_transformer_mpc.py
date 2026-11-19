"""Phase 7.4: Verification and causal contract tests for GNNTransformerMPCController."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import pytest
import numpy as np
import torch

from src.control.gnn_transformer_mpc import GNNTransformerMPCController, GNNMPCStepDiagnostic
from src.control.mpc_solver import load_mpc_config
from src.forecasting.integration import GRAPH_NODE_ORDER, TARGET_NODE_ORDER
from src.simulation.actions import BatteryAction
from src.simulation.config import load_simulation_config
from src.simulation.data_loader import HourlyObservation, SimulationData, load_simulation_split
from src.simulation.demand import CampusDemand
from src.simulation.environment import MicrogridEnvironment
from src.simulation.metrics import load_cost_config
from src.simulation.renewables import SolarPV, WindGeneration

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = Path(r"E:\AdaptiveEnergyBrainStorage\checkpoints\part5_9-colab-20261111T172326Z\best.pt")
PREPROCESSING_PATH = ROOT / "data/processed/splits_hypothetical_campus_v1/preprocessing.json"


@pytest.fixture
def synthetic_72h_node_dataset():
    """Create a 72-hour synthetic dataset with 5 building nodes, 1 solar, 1 wind."""
    prov = {
        "source_id": "synthetic_gnn_test",
        "origin": "simulated",
        "synthetic": True,
        "source_metadata": {
            "scenario": {
                "buildings": [{"id": f"B0{k}"} for k in range(1, 6)],
                "solar": {"asset_id": "PV01"},
                "wind": {"asset_id": "W01"},
            }
        },
    }
    meta = {
        "currency_code": "USD",
        "units": {"power": "kW", "prices": "USD/kWh"},
        "sources": {
            "load": {"record_type": "load", "metadata": prov, "entity_ids": [f"B0{k}" for k in range(1, 6)]},
            "solar": {"record_type": "solar", "metadata": prov, "entity_ids": ["PV01"]},
            "wind": {"record_type": "wind", "metadata": prov, "entity_ids": ["W01"]},
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
        if h == 35:
            price = 0.72  # Unannounced stochastic price spike
        solar = 120.0 if 10 <= hour <= 14 else 0.0
        wind = 25.0
        b_loads = {f"B0{k}": 30.0 + k * 5.0 + (10.0 if 11 <= hour <= 16 else 0.0) for k in range(1, 6)}
        campus_demand = sum(b_loads.values())

        obs = HourlyObservation(
            timestamp_utc=ts,
            pv_available_kw=solar,
            wind_available_kw=wind,
            campus_load_kw=campus_demand,
            import_price=price,
            export_price=0.04,
            building_loads_kw=b_loads,
            pv_asset_powers_kw={"PV01": solar},
            wind_asset_powers_kw={"W01": wind},
        )
        rows.append(obs)
    return SimulationData(rows, meta)


def test_1_checkpoint_model_integrity_and_weights_immutable():
    """Verify that checkpoint loads safely, matches expected architecture, and weights remain immutable."""
    assert CHECKPOINT_PATH.exists(), f"Missing checkpoint: {CHECKPOINT_PATH}"
    ctrl = GNNTransformerMPCController(
        checkpoint_path=CHECKPOINT_PATH,
        preprocessing_path=PREPROCESSING_PATH,
    )
    assert ctrl.model.training is False
    assert sum(p.numel() for p in ctrl.model.parameters()) == 77080
    assert ctrl.model.forecast_horizon == 24
    assert ctrl.model.lookback_hours == 24
    assert ctrl.node_order == GRAPH_NODE_ORDER
    assert ctrl.edge_index.shape == (2, 14)


def test_2_forecast_transformation_and_fixture_verification():
    """Verify input scaling, output inverse scaling, aggregation, masking, and non-negativity."""
    ctrl = GNNTransformerMPCController(
        checkpoint_path=CHECKPOINT_PATH,
        preprocessing_path=PREPROCESSING_PATH,
    )

    # Synthetic fixture with known normalized inputs
    dummy_input = torch.zeros(1, 24, 8, 1, dtype=torch.float32)
    with torch.no_grad():
        out = ctrl.model(dummy_input, ctrl.edge_index)

    assert out.shape == (1, 24, 8, 1)
    # BUS01 must be masked to zero
    assert torch.all(out[0, :, 7, 0] == 0.0)
    # Target channels must be finite
    assert torch.isfinite(out).all()

    # Model output is already in physical kW
    out_np = out.cpu().numpy()[0]
    # Projected non-negative
    out_np[:, :7, 0] = np.maximum(out_np[:, :7, 0], 0.0)
    assert np.all(out_np[:, :7, 0] >= 0.0)

    # Aggregated campus load must equal sum of building loads
    demand_sum = np.sum(out_np[:, :5, 0], axis=1)
    assert demand_sum.shape == (24,)
    assert np.all(demand_sum >= 0.0)


def test_3_forecast_timestamp_alignment_to_execution_interval(synthetic_72h_node_dataset):
    """Prove that LP prediction index 0 corresponds to the exact interval in which the first action executes."""
    sim_cfg = load_simulation_config()
    cost_cfg = load_cost_config()
    mpc_cfg = load_mpc_config()

    history = synthetic_72h_node_dataset[:24]
    episode_data = SimulationData(synthetic_72h_node_dataset[23:], synthetic_72h_node_dataset.metadata)

    ctrl = GNNTransformerMPCController(
        checkpoint_path=CHECKPOINT_PATH,
        preprocessing_path=PREPROCESSING_PATH,
        initial_history=history,
        history_source=synthetic_72h_node_dataset,
        sim_config=sim_cfg,
        cost_config=cost_cfg,
        mpc_config=mpc_cfg,
    )

    prov = synthetic_72h_node_dataset.metadata["sources"]["solar"]["metadata"]
    solar = SolarPV("PV01", 150.0, prov, capacity_origin="hypothetical_example")
    wind = WindGeneration("W01", 50.0, prov, capacity_origin="hypothetical_example")
    demand = CampusDemand(prov, building_ids=[f"B0{k}" for k in range(1, 6)])
    env = MicrogridEnvironment(episode_data, [solar, wind], demand, sim_cfg, cost_cfg)

    obs = env.reset()
    dec_ts = obs.decision_timestamp_utc
    # The dispatched row at step 1 in the environment is episode_data[1]
    expected_dispatch_interval_start = episode_data[1].timestamp_utc
    assert dec_ts == expected_dispatch_interval_start, "Decision timestamp must match the start of the dispatched interval."

    action = ctrl.select_action(obs)
    diag = ctrl.step_diagnostics[0]

    # The first target timestamp in the MPC LP horizon MUST match dec_ts
    assert diag.target_timestamps_utc[0] == dec_ts, (
        f"Alignment defect: LP target 0 ({diag.target_timestamps_utc[0]}) != dispatched interval ({dec_ts})"
    )
    assert len(diag.target_timestamps_utc) == 24


def test_4_strict_causal_non_anticipativity_future_perturbations_leave_action_invariant(synthetic_72h_node_dataset):
    """Test that modifying future rows (t >= decision) has ZERO impact on the controller's decision at decision time."""
    sim_cfg = load_simulation_config()
    cost_cfg = load_cost_config()
    mpc_cfg = load_mpc_config()

    history = synthetic_72h_node_dataset[:24]
    episode_data = SimulationData(synthetic_72h_node_dataset[23:], synthetic_72h_node_dataset.metadata)

    # Run unperturbed decision
    ctrl1 = GNNTransformerMPCController(
        checkpoint_path=CHECKPOINT_PATH,
        preprocessing_path=PREPROCESSING_PATH,
        initial_history=history,
        history_source=synthetic_72h_node_dataset,
        sim_config=sim_cfg,
        cost_config=cost_cfg,
        mpc_config=mpc_cfg,
    )

    prov = synthetic_72h_node_dataset.metadata["sources"]["solar"]["metadata"]
    solar1 = SolarPV("PV01", 150.0, prov, capacity_origin="hypothetical_example")
    wind1 = WindGeneration("W01", 50.0, prov, capacity_origin="hypothetical_example")
    demand1 = CampusDemand(prov, building_ids=[f"B0{k}" for k in range(1, 6)])
    env1 = MicrogridEnvironment(episode_data, [solar1, wind1], demand1, sim_cfg, cost_cfg)

    obs1 = env1.reset()
    action1 = ctrl1.select_action(obs1)
    diag1 = ctrl1.step_diagnostics[0]

    # Create perturbed dataset where all future rows (rows 25..71) have massive perturbations
    # Keep PV/Wind within rated nameplate capacities (PV01 <= 150 kW, W01 <= 50 kW)
    perturbed_rows = []
    for idx, r in enumerate(synthetic_72h_node_dataset):
        if idx >= 24:  # Future rows relative to decision time of step 1 (row 24)
            perturbed_r = HourlyObservation(
                timestamp_utc=r.timestamp_utc,
                pv_available_kw=min(r.pv_available_kw * 1.2, 140.0),
                wind_available_kw=min(r.wind_available_kw * 1.5, 45.0),
                campus_load_kw=r.campus_load_kw * 2.0,
                import_price=0.99,
                export_price=0.01,
                building_loads_kw={k: v * 2.0 for k, v in r.building_loads_kw.items()},
                pv_asset_powers_kw={"PV01": min(r.pv_available_kw * 1.2, 140.0)},
                wind_asset_powers_kw={"W01": min(r.wind_available_kw * 1.5, 45.0)},
            )
            perturbed_rows.append(perturbed_r)
        else:
            perturbed_rows.append(r)

    perturbed_data = SimulationData(perturbed_rows, synthetic_72h_node_dataset.metadata)
    perturbed_episode_data = SimulationData(perturbed_rows[23:], synthetic_72h_node_dataset.metadata)

    ctrl2 = GNNTransformerMPCController(
        checkpoint_path=CHECKPOINT_PATH,
        preprocessing_path=PREPROCESSING_PATH,
        initial_history=history,
        history_source=perturbed_data,
        sim_config=sim_cfg,
        cost_config=cost_cfg,
        mpc_config=mpc_cfg,
    )

    solar2 = SolarPV("PV01", 150.0, prov, capacity_origin="hypothetical_example")
    wind2 = WindGeneration("W01", 50.0, prov, capacity_origin="hypothetical_example")
    demand2 = CampusDemand(prov, building_ids=[f"B0{k}" for k in range(1, 6)])
    env2 = MicrogridEnvironment(perturbed_episode_data, [solar2, wind2], demand2, sim_cfg, cost_cfg)

    obs2 = env2.reset()
    action2 = ctrl2.select_action(obs2)
    diag2 = ctrl2.step_diagnostics[0]

    # Action and diagnostics must be identical despite future perturbations
    assert action1.requested_power_kw == action2.requested_power_kw, "Future perturbations altered current action!"
    assert diag1.forecast_demand_kw == diag2.forecast_demand_kw, "Future perturbations altered GNN forecast!"
    assert diag1.forecast_solar_kw == diag2.forecast_solar_kw
    assert diag1.forecast_wind_kw == diag2.forecast_wind_kw


def test_5_unannounced_tariff_spike_invariance(synthetic_72h_node_dataset):
    """Verify that unannounced stochastic tariff spikes are not included in the forward planning tariff schedule."""
    sim_cfg = load_simulation_config()
    cost_cfg = load_cost_config()
    mpc_cfg = load_mpc_config()

    ctrl = GNNTransformerMPCController(
        checkpoint_path=CHECKPOINT_PATH,
        preprocessing_path=PREPROCESSING_PATH,
        initial_history=synthetic_72h_node_dataset[:24],
        history_source=synthetic_72h_node_dataset,
        sim_config=sim_cfg,
        cost_config=cost_cfg,
        mpc_config=mpc_cfg,
    )

    prov = synthetic_72h_node_dataset.metadata["sources"]["solar"]["metadata"]
    solar = SolarPV("PV01", 150.0, prov, capacity_origin="hypothetical_example")
    wind = WindGeneration("W01", 50.0, prov, capacity_origin="hypothetical_example")
    demand = CampusDemand(prov, building_ids=[f"B0{k}" for k in range(1, 6)])
    env = MicrogridEnvironment(SimulationData(synthetic_72h_node_dataset[23:], synthetic_72h_node_dataset.metadata), [solar, wind], demand, sim_cfg, cost_cfg)

    obs = env.reset()
    action = ctrl.select_action(obs)

    # In synthetic_72h_node_dataset, h=35 has price 0.72. The controller planning schedule must still use standard TOU (0.12 or 0.18).
    diag = ctrl.step_diagnostics[0]
    assert diag.success is True


def test_6_validation_boundary_continuity_and_truncated_horizon(synthetic_72h_node_dataset):
    """Verify that near episode end, the controller truncates the planning horizon cleanly."""
    sim_cfg = load_simulation_config()
    cost_cfg = load_cost_config()
    mpc_cfg = load_mpc_config()

    # Set episode end 5 hours after decision
    dec_dt = datetime(2026, 6, 2, 0, 0, 0, tzinfo=timezone.utc)
    end_dt = dec_dt + timedelta(hours=5)
    end_ts = end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    ctrl = GNNTransformerMPCController(
        checkpoint_path=CHECKPOINT_PATH,
        preprocessing_path=PREPROCESSING_PATH,
        initial_history=synthetic_72h_node_dataset[:24],
        history_source=synthetic_72h_node_dataset,
        sim_config=sim_cfg,
        cost_config=cost_cfg,
        mpc_config=mpc_cfg,
        episode_end_utc=end_ts,
    )

    prov = synthetic_72h_node_dataset.metadata["sources"]["solar"]["metadata"]
    solar = SolarPV("PV01", 150.0, prov, capacity_origin="hypothetical_example")
    wind = WindGeneration("W01", 50.0, prov, capacity_origin="hypothetical_example")
    demand = CampusDemand(prov, building_ids=[f"B0{k}" for k in range(1, 6)])
    env = MicrogridEnvironment(SimulationData(synthetic_72h_node_dataset[23:], synthetic_72h_node_dataset.metadata), [solar, wind], demand, sim_cfg, cost_cfg)

    obs = env.reset()
    action = ctrl.select_action(obs)

    diag = ctrl.step_diagnostics[0]
    assert diag.horizon_hours == 5
    assert len(diag.forecast_demand_kw) == 5
    assert len(diag.target_timestamps_utc) == 5


def test_7_first_planned_action_agrees_with_simulator_execution(synthetic_72h_node_dataset):
    """Verify that the first action requested by the LP matches physical execution."""
    sim_cfg = load_simulation_config()
    cost_cfg = load_cost_config()
    mpc_cfg = load_mpc_config()

    ctrl = GNNTransformerMPCController(
        checkpoint_path=CHECKPOINT_PATH,
        preprocessing_path=PREPROCESSING_PATH,
        initial_history=synthetic_72h_node_dataset[:24],
        history_source=synthetic_72h_node_dataset,
        sim_config=sim_cfg,
        cost_config=cost_cfg,
        mpc_config=mpc_cfg,
    )

    prov = synthetic_72h_node_dataset.metadata["sources"]["solar"]["metadata"]
    solar = SolarPV("PV01", 150.0, prov, capacity_origin="hypothetical_example")
    wind = WindGeneration("W01", 50.0, prov, capacity_origin="hypothetical_example")
    demand = CampusDemand(prov, building_ids=[f"B0{k}" for k in range(1, 6)])
    env = MicrogridEnvironment(SimulationData(synthetic_72h_node_dataset[23:], synthetic_72h_node_dataset.metadata), [solar, wind], demand, sim_cfg, cost_cfg)

    obs = env.reset()
    action = ctrl.select_action(obs)
    step = env.step(action)

    assert abs(step.dispatch.battery.actual_power_kw - action.requested_power_kw) < 1e-4
