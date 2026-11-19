"""Phase 10: CVaR Risk-Aware MPC Experimental Runner and Trade-off Analysis.

Evaluates multi-scenario CVaR MPC against deterministic baselines across stress scenarios:
- Normal operating conditions
- High demand stress (1.25x)
- Renewable scarcity (0.5x solar, 0.4x wind)
- Unannounced peak price spikes (3.0x peak tariff)
- Compound adverse conditions (stress demand + scarcity + price spikes)

Also computes the Pareto efficiency curve across risk-aversion parameter lambda in [0.0, 5.0].
Outputs structured metrics to experiments/phase10_cvar_experiments/.
"""

from collections.abc import Sequence
from dataclasses import asdict
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
from src.control.cvar_mpc import CVaRRiskAwareMPCController
from src.control.gnn_transformer_mpc import GNNTransformerMPCController
from src.control.seasonal_persistence_mpc import SeasonalPersistenceMPCController
from src.evaluation.inventory_accounting import compute_inventory_adjusted_cost
from src.simulation.config import load_simulation_config
from src.simulation.data_loader import (
    HourlyObservation,
    SimulationData,
    load_simulation_split,
)
from src.simulation.environment import MicrogridEnvironment
from src.simulation.metrics import load_cost_config

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT_ROOT = ROOT / "data" / "processed" / "splits_hypothetical_campus_v1"
DEFAULT_ASSETS_PATH = ROOT / "configs" / "synthetic_scenario.yaml"
OUTPUT_DIR = ROOT / "experiments" / "phase10_cvar_experiments"


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
            if 15 <= hour <= 20:  # Peak evening hours
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
) -> dict[str, Any]:
    """Execute one simulation episode and compute risk and economic metrics."""
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

    terminal_soc = env.state.battery.battery_soc
    cum_metrics = env.cumulative_metrics

    # Calculate VaR and CVaR at 95% confidence on hourly net operating cost distribution
    costs_arr = np.array(hourly_net_costs)
    alpha = 0.95
    var_95 = float(np.percentile(costs_arr, alpha * 100))
    tail_costs = costs_arr[costs_arr >= var_95]
    cvar_95 = float(np.mean(tail_costs)) if len(tail_costs) > 0 else var_95
    worst_case_cost = float(np.max(costs_arr))

    from src.evaluation.inventory_accounting import compute_replacement_reference_value
    ref_val = compute_replacement_reference_value(config=sim_config, costs=cost_config)
    adj_res = compute_inventory_adjusted_cost(
        raw_net_operating_cost=cum_metrics.net_operating_cost,
        initial_energy_kwh=initial_soc * sim_config.battery.capacity_kwh,
        terminal_energy_kwh=terminal_soc * sim_config.battery.capacity_kwh,
        reference_value_usd_per_kwh_dc=ref_val,
    )
    terminal_inventory_credit_usd = adj_res["inventory_adjustment_usd"]
    adjusted_net_cost_usd = adj_res["inventory_adjusted_net_operating_cost_usd"]

    # Battery throughput and FCE
    fce = cum_metrics.battery_throughput_kwh / (2.0 * sim_config.battery.capacity_kwh)
    degradation_cost_usd = cum_metrics.battery_degradation_cost

    return {
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


def main():
    print("=" * 70)
    print("Adaptive Energy Brain — Phase 10: CVaR Risk-Aware Experiments")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sim_config = load_simulation_config()
    cost_config = load_cost_config()

    print("[1/4] Loading training and validation partition data...")
    train_data = load_simulation_split(DEFAULT_SPLIT_ROOT, "train")
    val_full = load_simulation_split(DEFAULT_SPLIT_ROOT, "validation")

    # Select representative 72-hour validation slice (3 days)
    horizon_slice = 72
    val_slice = val_full[:horizon_slice]
    print(f"  Selected evaluation slice: {len(val_slice)} hours ({val_slice[0].timestamp_utc} to {val_slice[-1].timestamp_utc})")

    conditions = [
        "normal",
        "high_demand",
        "renewable_scarcity",
        "price_spikes",
        "compound_adverse",
    ]

    controller_specs = [
        ("zero_action", "Zero Action Baseline", lambda: ZeroActionController()),
        ("mpc_seasonal_persistence", "Seasonal Persistence MPC", lambda: SeasonalPersistenceMPCController(initial_history=train_data[-24:])),
        ("mpc_gnn_transformer", "Deterministic GNN-Transformer MPC", lambda: GNNTransformerMPCController(initial_history=train_data[-24:], history_source=train_data)),
        ("mpc_cvar_neutral", "CVaR MPC (lambda=0.0 Risk-Neutral)", lambda: CVaRRiskAwareMPCController(history_source=train_data, override_lambda=0.0)),
        ("mpc_cvar_balanced", "CVaR MPC (lambda=1.0 Balanced)", lambda: CVaRRiskAwareMPCController(history_source=train_data, override_lambda=1.0)),
        ("mpc_cvar_averse", "CVaR MPC (lambda=3.0 Risk-Averse)", lambda: CVaRRiskAwareMPCController(history_source=train_data, override_lambda=3.0)),
    ]

    print(f"[2/4] Executing comparative matrix across {len(conditions)} conditions x {len(controller_specs)} controllers...")
    results_matrix: dict[str, dict[str, Any]] = {}
    summary_rows = []

    for cond in conditions:
        print(f"\n--- Condition: {cond.upper()} ---")
        stressed_sim_data = create_stressed_dataset(val_slice, val_full.metadata, cond)
        results_matrix[cond] = {}

        for c_id, c_name, c_factory in controller_specs:
            t_start = time.perf_counter()
            ctrl = c_factory()
            res = run_episode(ctrl, stressed_sim_data, sim_config, cost_config, DEFAULT_ASSETS_PATH)
            elapsed = time.perf_counter() - t_start

            results_matrix[cond][c_id] = {
                "name": c_name,
                "metrics": res,
            }

            row = {
                "condition": cond,
                "controller_id": c_id,
                "controller_name": c_name,
                **res,
            }
            summary_rows.append(row)

            print(f"  [{c_id:<24}] AdjCost: ${res['adjusted_net_cost_usd']:>8.2f} | 95% CVaR: ${res['hourly_cost_cvar_95_usd']:>6.2f}/h | Max: ${res['worst_case_hourly_cost_usd']:>6.2f}/h | Latency: {res['mean_latency_ms']:>5.1f}ms ({elapsed:.1f}s)")

    # Save comprehensive results JSON
    results_json_path = OUTPUT_DIR / "cvar_experiment_results.json"
    results_json_path.write_text(json.dumps(results_matrix, indent=2), encoding="utf-8")
    print(f"\nSaved matrix results -> {results_json_path}")

    # Save summary CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = OUTPUT_DIR / "cvar_experiment_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Saved summary CSV -> {summary_csv_path}")

    print("\n[3/4] Generating Pareto Risk-Cost Frontier sweep across lambda in [0.0, 5.0] on compound adverse scenario...")
    lambdas = [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0]
    compound_data = create_stressed_dataset(val_slice, val_full.metadata, "compound_adverse")

    frontier_curve = []
    for lam in lambdas:
        ctrl = CVaRRiskAwareMPCController(history_source=train_data, override_lambda=lam)
        res = run_episode(ctrl, compound_data, sim_config, cost_config, DEFAULT_ASSETS_PATH)
        point = {
            "risk_aversion_lambda": lam,
            "adjusted_net_cost_usd": res["adjusted_net_cost_usd"],
            "raw_net_cost_usd": res["raw_net_cost_usd"],
            "hourly_cost_cvar_95_usd": res["hourly_cost_cvar_95_usd"],
            "worst_case_hourly_cost_usd": res["worst_case_hourly_cost_usd"],
            "battery_fce": res["battery_fce"],
            "battery_degradation_cost_usd": res["battery_degradation_cost_usd"],
            "mean_latency_ms": res["mean_latency_ms"],
        }
        frontier_curve.append(point)
        print(f"  lambda={lam:>4.2f} -> Cost: ${point['adjusted_net_cost_usd']:>8.2f} | 95% CVaR: ${point['hourly_cost_cvar_95_usd']:>6.2f}/h | Worst: ${point['worst_case_hourly_cost_usd']:>6.2f}/h")

    tradeoff_path = OUTPUT_DIR / "cvar_tradeoff_curve.json"
    tradeoff_path.write_text(json.dumps(frontier_curve, indent=2), encoding="utf-8")
    print(f"Saved Pareto curve -> {tradeoff_path}")

    print("\n[4/4] Phase 10 evaluation experiments completed successfully.")


if __name__ == "__main__":
    main()
