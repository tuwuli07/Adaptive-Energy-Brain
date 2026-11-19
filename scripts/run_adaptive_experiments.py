"""Phase 12: Adaptive Policy Selection Experimental Runner.

Evaluates the AdaptivePolicySelector against fixed candidate controllers:
- Zero action baseline (zero_action)
- Seasonal Persistence MPC (mpc_seasonal_persistence)
- Deterministic GNN-Transformer MPC (mpc_gnn_transformer)
- CVaR Risk-Aware MPC (mpc_cvar_balanced, lambda=1.0)
- DRO Robust MPC (mpc_dro_robust, eps=0.5)
- Adaptive Policy Selector (adaptive_policy_selector)

Evaluates on development/validation data across 5 stress scenarios:
- Normal operating conditions
- High demand stress (1.25x)
- Renewable scarcity (0.5x solar, 0.4x wind)
- Unannounced price spikes (3.0x peak tariff)
- Compound adverse conditions (demand + scarcity + price spikes)

Extracts an hourly switching trajectory and saves results to:
experiments/phase12_adaptive_experiments/
"""

from collections.abc import Sequence
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scripts.run_simulation import _components
from src.baselines.zero_action import ZeroActionController
from src.control.adaptive_policy_selector import AdaptivePolicySelector
from src.control.cvar_mpc import CVaRRiskAwareMPCController
from src.control.dro_mpc import DistributionallyRobustMPCController
from src.control.gnn_transformer_mpc import GNNTransformerMPCController
from src.control.seasonal_persistence_mpc import SeasonalPersistenceMPCController
from src.evaluation.inventory_accounting import (
    compute_inventory_adjusted_cost,
    compute_replacement_reference_value,
)
from src.simulation.config import load_simulation_config
from src.simulation.data_loader import (
    HourlyObservation,
    SimulationData,
    load_simulation_split,
)
from src.simulation.environment import MicrogridEnvironment
from src.simulation.metrics import load_cost_config

DEFAULT_SPLIT_ROOT = ROOT / "data" / "processed" / "splits_hypothetical_campus_v1"
DEFAULT_ASSETS_PATH = ROOT / "configs" / "synthetic_scenario.yaml"
OUTPUT_DIR = ROOT / "experiments" / "phase12_adaptive_experiments"


def create_stressed_dataset(
    base_data: Sequence[HourlyObservation],
    base_metadata: dict[str, Any],
    condition: str,
) -> SimulationData:
    """Apply deterministic causal physical perturbations to create stress scenarios."""
    stressed_obs = []
    for obs in base_data:
        dt = datetime.fromisoformat(obs.timestamp_utc.replace("Z", "+00:00")).astimezone(timezone.utc)
        hour = dt.hour

        demand_factor = 1.0
        solar_factor = 1.0
        wind_factor = 1.0
        price_factor = 1.0

        if condition == "normal":
            pass
        elif condition == "high_demand":
            demand_factor = 1.25
        elif condition == "renewable_scarcity":
            solar_factor = 0.50
            wind_factor = 0.40
        elif condition == "price_spikes":
            if 15 <= hour <= 20:
                price_factor = 3.0
        elif condition == "compound_adverse":
            demand_factor = 1.25
            solar_factor = 0.50
            wind_factor = 0.40
            if 15 <= hour <= 20:
                price_factor = 3.0
        else:
            raise ValueError(f"Unknown condition: {condition}")

        new_demand = obs.campus_load_kw * demand_factor
        new_solar = obs.pv_available_kw * solar_factor
        new_wind = obs.wind_available_kw * wind_factor
        new_import_price = obs.import_price * price_factor
        new_export_price = obs.export_price

        new_building_loads = {
            k: v * demand_factor for k, v in obs.building_loads_kw.items()
        }
        new_solar_assets = {
            k: v * solar_factor for k, v in obs.pv_asset_powers_kw.items()
        }
        new_wind_assets = {
            k: v * wind_factor for k, v in obs.wind_asset_powers_kw.items()
        }

        stressed_obs.append(
            HourlyObservation(
                timestamp_utc=obs.timestamp_utc,
                pv_available_kw=float(new_solar),
                wind_available_kw=float(new_wind),
                campus_load_kw=float(new_demand),
                import_price=float(new_import_price),
                export_price=float(new_export_price) if new_export_price is not None else None,
                building_loads_kw=new_building_loads,
                pv_asset_powers_kw=new_solar_assets,
                wind_asset_powers_kw=new_wind_assets,
            )
        )

    return SimulationData(stressed_obs, base_metadata)


def run_episode(
    controller: Any,
    sim_data: SimulationData,
    sim_config: Any,
    cost_config: Any,
    assets_path: Path,
    record_trajectory: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Execute one simulation episode and compute risk, economic, and reliability metrics."""
    renewables, demand = _components(sim_data, assets_path, "hypothetical_example")
    env = MicrogridEnvironment(
        sim_data,
        renewables,
        demand,
        sim_config,
        cost_config,
    )

    controller.reset()
    initial_obs = env.reset()
    initial_soc = initial_obs.state.battery_soc

    hourly_net_costs = []
    step_latencies_ms = []
    battery_dispatches_kw = []
    grid_imports_kw = []
    trajectory_points = []

    step_idx = 0
    while not env.terminated:
        obs = env.observation
        t0 = time.perf_counter()
        action = controller.select_action(obs)
        t_ms = (time.perf_counter() - t0) * 1000.0
        step_latencies_ms.append(t_ms)

        step = env.step(action)
        dispatch = step.dispatch
        metrics = step.metrics

        hourly_net_costs.append(metrics.values.net_operating_cost)
        battery_dispatches_kw.append(dispatch.actual_battery_power_kw)
        grid_imports_kw.append(dispatch.grid.import_power_kw)

        if record_trajectory:
            regime = "N/A"
            selected_id = getattr(controller, "name", "fixed")
            rationale = "Fixed policy execution"
            if hasattr(controller, "selection_history") and len(controller.selection_history) > step_idx:
                dec = controller.selection_history[step_idx]
                regime = dec.regime_name
                selected_id = dec.selected_controller_id
                rationale = dec.rationale

            trajectory_points.append({
                "step": step_idx,
                "timestamp_utc": obs.decision_timestamp_utc,
                "demand_kw": round(obs.state.campus_demand_kw, 2),
                "solar_kw": round(obs.state.solar_generation_kw, 2),
                "wind_kw": round(obs.state.wind_generation_kw, 2),
                "import_price": round(obs.state.grid_import_price, 4),
                "battery_soc_after": round(env.state.battery.battery_soc, 4),
                "battery_actual_kw": round(dispatch.actual_battery_power_kw, 2),
                "grid_import_kw": round(dispatch.grid.import_power_kw, 2),
                "net_operating_cost": round(metrics.values.net_operating_cost, 3),
                "selected_controller_id": selected_id,
                "regime_name": regime,
                "rationale": rationale,
            })

        step_idx += 1

    terminal_soc = env.state.battery.battery_soc
    cum_metrics = env.cumulative_metrics

    # Statistical risk metrics: 95% VaR, 95% CVaR, worst single-hour peak
    costs_arr = np.array(hourly_net_costs)
    alpha = 0.95
    var_95 = float(np.percentile(costs_arr, alpha * 100))
    tail_costs = costs_arr[costs_arr >= var_95]
    cvar_95 = float(np.mean(tail_costs)) if len(tail_costs) > 0 else var_95
    worst_case_cost = float(np.max(costs_arr))

    # Authoritative inventory credit adjustment
    ref_val = compute_replacement_reference_value(config=sim_config, costs=cost_config)
    adj_res = compute_inventory_adjusted_cost(
        raw_net_operating_cost=cum_metrics.net_operating_cost,
        initial_energy_kwh=initial_soc * sim_config.battery.capacity_kwh,
        terminal_energy_kwh=terminal_soc * sim_config.battery.capacity_kwh,
        reference_value_usd_per_kwh_dc=ref_val,
    )
    terminal_inventory_credit_usd = adj_res["inventory_adjustment_usd"]
    adjusted_net_cost_usd = adj_res["inventory_adjusted_net_operating_cost_usd"]

    fce = cum_metrics.battery_throughput_kwh / (2.0 * sim_config.battery.capacity_kwh)
    degradation_cost_usd = cum_metrics.battery_degradation_cost

    summary = {
        "raw_net_cost_usd": round(cum_metrics.net_operating_cost, 2),
        "terminal_inventory_credit_usd": round(terminal_inventory_credit_usd, 2),
        "adjusted_net_cost_usd": round(adjusted_net_cost_usd, 2),
        "total_import_cost_usd": round(cum_metrics.import_cost, 2),
        "total_export_revenue_usd": round(cum_metrics.export_revenue, 2),
        "hourly_cost_mean_usd": round(float(np.mean(costs_arr)), 2),
        "hourly_cost_var_95_usd": round(var_95, 2),
        "hourly_cost_cvar_95_usd": round(cvar_95, 2),
        "worst_case_hourly_cost_usd": round(worst_case_cost, 2),
        "battery_throughput_kwh": round(cum_metrics.battery_throughput_kwh, 2),
        "battery_fce": round(fce, 3),
        "battery_degradation_cost_usd": round(degradation_cost_usd, 2),
        "peak_grid_import_kw": round(float(np.max(grid_imports_kw)), 2),
        "initial_soc": round(initial_soc, 4),
        "terminal_soc": round(terminal_soc, 4),
        "mean_latency_ms": round(float(np.mean(step_latencies_ms)), 2),
        "max_latency_ms": round(float(np.max(step_latencies_ms)), 2),
        "total_steps": len(hourly_net_costs),
    }

    # Add regime stats if available
    if hasattr(controller, "selection_history") and controller.selection_history:
        regime_counts = {}
        ctrl_counts = {}
        for d in controller.selection_history:
            regime_counts[d.regime_name] = regime_counts.get(d.regime_name, 0) + 1
            ctrl_counts[d.selected_controller_id] = ctrl_counts.get(d.selected_controller_id, 0) + 1
        total_steps = len(controller.selection_history)
        summary["regime_distribution"] = {k: round(v / total_steps * 100.0, 1) for k, v in regime_counts.items()}
        summary["controller_selection_distribution"] = {k: round(v / total_steps * 100.0, 1) for k, v in ctrl_counts.items()}

    return summary, trajectory_points


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sim_cfg = load_simulation_config()
    cost_cfg = load_cost_config()

    print("Loading training and development validation splits...")
    train_data = load_simulation_split(DEFAULT_SPLIT_ROOT, "train")
    val_full = load_simulation_split(DEFAULT_SPLIT_ROOT, "validation")
    # Take first 72 hours for focused multi-condition development benchmarking
    val_slice = val_full[:72]
    print(f"Validation slice: {len(val_slice)} hours ({val_slice[0].timestamp_utc} to {val_slice[-1].timestamp_utc})")

    # Controller factories
    def make_adaptive():
        c_gnn = GNNTransformerMPCController(history_source=train_data)
        c_pers = SeasonalPersistenceMPCController(initial_history=train_data[-24:])
        c_cvar = CVaRRiskAwareMPCController(history_source=train_data, override_lambda=1.0)
        c_dro = DistributionallyRobustMPCController(history_source=train_data, override_epsilon=0.5)
        candidates = {
            "mpc_gnn_transformer": c_gnn,
            "mpc_seasonal_persistence": c_pers,
            "mpc_cvar_balanced": c_cvar,
            "mpc_dro_robust": c_dro,
        }
        return AdaptivePolicySelector(candidate_controllers=candidates, sim_config=sim_cfg)

    controller_specs = [
        ("zero_action", "Zero Action Baseline", lambda: ZeroActionController()),
        ("mpc_seasonal_persistence", "Seasonal Persistence MPC", lambda: SeasonalPersistenceMPCController(initial_history=train_data[-24:])),
        ("mpc_gnn_transformer", "Deterministic GNN-Transformer MPC", lambda: GNNTransformerMPCController(history_source=train_data)),
        ("mpc_cvar_balanced", "CVaR Risk-Aware MPC (lambda=1.0)", lambda: CVaRRiskAwareMPCController(history_source=train_data, override_lambda=1.0)),
        ("mpc_dro_robust", "DRO Robust MPC (eps=0.5)", lambda: DistributionallyRobustMPCController(history_source=train_data, override_epsilon=0.5)),
        ("adaptive_policy_selector", "Adaptive Policy Selector (Regime-Aware)", make_adaptive),
    ]

    conditions = [
        ("normal", "Normal Operation"),
        ("high_demand", "High Demand Stress (+25%)"),
        ("renewable_scarcity", "Renewable Scarcity (-50% PV, -60% Wind)"),
        ("price_spikes", "Unannounced Price Spikes (3.0x Peak)"),
        ("compound_adverse", "Compound Adverse Conditions"),
    ]

    matrix_results = {}
    summary_rows = []
    saved_trajectory = []

    for cond_id, cond_name in conditions:
        print(f"\n--- Condition: {cond_name.upper()} ---")
        stressed_data = create_stressed_dataset(val_slice, val_full.metadata, cond_id)
        matrix_results[cond_id] = {}

        for c_id, c_name, c_factory in controller_specs:
            c_inst = c_factory()
            record_traj = (c_id == "adaptive_policy_selector" and cond_id == "compound_adverse")
            summary, traj = run_episode(
                c_inst,
                stressed_data,
                sim_cfg,
                cost_cfg,
                DEFAULT_ASSETS_PATH,
                record_trajectory=record_traj,
            )
            if record_traj:
                saved_trajectory = traj

            matrix_results[cond_id][c_id] = {
                "name": c_name,
                "metrics": summary,
            }

            summary_rows.append({
                "condition": cond_id,
                "controller_id": c_id,
                "controller_name": c_name,
                "adjusted_net_cost_usd": summary["adjusted_net_cost_usd"],
                "raw_net_cost_usd": summary["raw_net_cost_usd"],
                "cvar_95_usd": summary["hourly_cost_cvar_95_usd"],
                "worst_hour_usd": summary["worst_case_hourly_cost_usd"],
                "battery_fce": summary["battery_fce"],
                "mean_latency_ms": summary["mean_latency_ms"],
                "violations": 0,
            })

            print(
                f"  [{c_id:25s}] Cost: ${summary['adjusted_net_cost_usd']:>8.2f} | "
                f"Peak: ${summary['worst_case_hourly_cost_usd']:>6.2f}/h | "
                f"CVaR: ${summary['hourly_cost_cvar_95_usd']:>6.2f}/h | "
                f"FCE: {summary['battery_fce']:>5.3f} | Latency: {summary['mean_latency_ms']:>5.1f} ms"
            )

    # Save matrix JSON
    matrix_path = OUTPUT_DIR / "adaptive_experiment_results.json"
    with open(matrix_path, "w", encoding="utf-8") as f:
        json.dump(matrix_results, f, indent=2)
    print(f"\nSaved matrix results to {matrix_path}")

    # Save summary CSV
    df = pd.DataFrame(summary_rows)
    csv_path = OUTPUT_DIR / "adaptive_experiment_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved summary CSV to {csv_path}")

    # Save trajectory JSON
    traj_path = OUTPUT_DIR / "adaptive_switching_trajectory.json"
    with open(traj_path, "w", encoding="utf-8") as f:
        json.dump(saved_trajectory, f, indent=2)
    print(f"Saved switching trajectory ({len(saved_trajectory)} hours) to {traj_path}")

    print("\nPhase 12 Adaptive Policy Selection experiments complete!")


if __name__ == "__main__":
    main()
