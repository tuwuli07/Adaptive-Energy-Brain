"""Phase 7.4: Causal GNN-Transformer MPC validation evaluation script.

Evaluates GNNTransformerMPCController using the Part 5.9 trained checkpoint
on the 1,313-hour validation split and outputs comparative metrics against:
  - Zero-Action
  - Rule-Based Continuous Price-Aware
  - Rule-Based Discrete Price-Aware
  - PPO Best Diagnostic (Discount 0.9995)
  - Seasonal Persistence MPC (Phase 7.3)
  - 24-Hour Perfect-Foresight Oracle MPC (Phase 7.2)
"""

from collections import defaultdict
from dataclasses import asdict
import json
import math
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_simulation import _components
from src.control.gnn_transformer_mpc import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_PREPROCESSING_PATH,
    GNNTransformerMPCController,
)
from src.control.mpc_solver import load_mpc_config
from src.evaluation.baseline_runner import _run_controller
from src.evaluation.inventory_accounting import (
    compute_inventory_adjusted_cost,
    compute_replacement_reference_value,
)
from src.simulation.config import load_simulation_config
from src.simulation.data_loader import load_simulation_split
from src.simulation.environment import MicrogridEnvironment
from src.simulation.metrics import load_cost_config


def run_gnn_mpc_evaluation():
    split_root = ROOT / "data" / "processed" / "splits_hypothetical_campus_v1"
    assets_path = ROOT / "configs" / "synthetic_scenario.yaml"
    rating_origin = "hypothetical_example"

    print("Loading train partition for causal initial history (trailing 24 hours)...")
    train_data = load_simulation_split(split_root, "train")
    initial_history = train_data[-24:]
    print(
        f"Initial history loaded: 24 rows ({initial_history[0].timestamp_utc} to {initial_history[-1].timestamp_utc})"
    )

    print("Loading validation partition...")
    val_data = load_simulation_split(split_root, "validation")
    print(f"Validation dataset loaded: {len(val_data)} rows ({len(val_data) - 1} dispatch steps)")

    # Verify continuity at split boundary
    train_end = train_data[-1].timestamp_utc
    val_start = val_data[0].timestamp_utc
    print(f"Boundary check: train_end={train_end}, val_start={val_start}")

    sim_config = load_simulation_config()
    cost_config = load_cost_config()
    mpc_config = load_mpc_config()

    print(f"MPC Config strict_exclusivity: {mpc_config.strict_exclusivity}")
    print(f"MPC Horizon: {mpc_config.horizon_hours} hours")
    print(f"Model Checkpoint: {DEFAULT_CHECKPOINT_PATH}")

    renewables, demand = _components(val_data, assets_path, rating_origin)
    env = MicrogridEnvironment(
        val_data,
        renewables,
        demand,
        sim_config,
        cost_config,
    )

    episode_end_utc = "2026-11-07T06:00:00Z"
    controller = GNNTransformerMPCController(
        checkpoint_path=DEFAULT_CHECKPOINT_PATH,
        preprocessing_path=DEFAULT_PREPROCESSING_PATH,
        initial_history=initial_history,
        history_source=train_data,
        sim_config=sim_config,
        cost_config=cost_config,
        mpc_config=mpc_config,
        episode_end_utc=episode_end_utc,
    )

    print("Starting 1,313-hour causal GNN-Transformer MPC simulation replay...")
    start_time = time.perf_counter()
    summary, hourly = _run_controller(
        "gnn_transformer_mpc",
        controller,
        env,
        synthetic=True,
        record_observation_context=True,
    )
    total_elapsed = time.perf_counter() - start_time
    print(f"Replay completed in {total_elapsed:.2f} seconds.")

    # Calculate inventory-adjusted cost
    capacity = sim_config.battery.capacity_kwh
    init_e = summary["initial_soc"] * capacity
    term_e = summary["terminal_soc"] * capacity
    raw_cost = summary["cumulative_metrics"]["net_operating_cost"]

    ref_value = compute_replacement_reference_value(
        sim_config, cost_config, baseline_config=ROOT / "configs" / "baselines.yaml"
    )
    adj_data = compute_inventory_adjusted_cost(raw_cost, init_e, term_e, ref_value)
    summary.update(adj_data)

    # Solve and Inference time statistics (in milliseconds)
    solve_times_ms = [t * 1000.0 for t in controller.solve_times]
    mean_solve_ms = statistics.mean(solve_times_ms) if solve_times_ms else 0.0
    median_solve_ms = statistics.median(solve_times_ms) if solve_times_ms else 0.0
    max_solve_ms = max(solve_times_ms) if solve_times_ms else 0.0
    min_solve_ms = min(solve_times_ms) if solve_times_ms else 0.0

    inf_times_ms = [t * 1000.0 for t in controller.inference_times]
    mean_inf_ms = statistics.mean(inf_times_ms) if inf_times_ms else 0.0
    median_inf_ms = statistics.median(inf_times_ms) if inf_times_ms else 0.0
    max_inf_ms = max(inf_times_ms) if inf_times_ms else 0.0
    min_inf_ms = min(inf_times_ms) if inf_times_ms else 0.0

    # Action counts
    charge_count = sum(1 for row in hourly if row["battery_actual_kw"] < -1.0)
    discharge_count = sum(1 for row in hourly if row["battery_actual_kw"] > 1.0)
    idle_count = len(hourly) - charge_count - discharge_count

    # Aggregate physical flows
    cum = summary["cumulative_metrics"]
    import_energy_kwh = cum["grid_import_energy_kwh"]
    export_energy_kwh = cum["grid_export_energy_kwh"]
    curtailment_kwh = cum["renewable_curtailed_energy_kwh"]
    throughput_kwh = cum["battery_throughput_kwh"]
    degradation_cost = cum["battery_degradation_cost"]

    # Hourly profile behavior (hour of day 0..23 UTC)
    hourly_soc = [[] for _ in range(24)]
    hourly_action = [[] for _ in range(24)]
    for row in hourly:
        hour_int = int(row["timestamp_utc"][11:13])
        hourly_soc[hour_int].append(row["battery_soc_after"])
        hourly_action[hour_int].append(row["battery_actual_kw"])

    hourly_mean_soc = {h: statistics.mean(hourly_soc[h]) for h in range(24)}
    hourly_mean_action = {h: statistics.mean(hourly_action[h]) for h in range(24)}

    # Period classifications (UTC)
    tariff_peak_hours = list(range(11, 16))   # 11:00 to 15:00 UTC (peak price 0.18 - 0.72 $/kWh)
    overnight_hours = list(range(0, 7))        # 00:00 to 06:00 UTC (off-peak 0.12 $/kWh)
    prepeak_hours = list(range(7, 11))         # 07:00 to 10:00 UTC (pre-peak off-peak 0.12 $/kWh)

    tariff_peak_discharges = sum(
        1 for row in hourly
        if int(row["timestamp_utc"][11:13]) in tariff_peak_hours and row["battery_actual_kw"] > 1.0
    )
    overnight_charges = sum(
        1 for row in hourly
        if int(row["timestamp_utc"][11:13]) in overnight_hours and row["battery_actual_kw"] < -1.0
    )
    prepeak_charges = sum(
        1 for row in hourly
        if int(row["timestamp_utc"][11:13]) in prepeak_hours and row["battery_actual_kw"] < -1.0
    )

    # Simultaneous flow and fallback checks
    simult_battery_count = sum(1 for d in controller.step_diagnostics if d.has_simultaneous_battery_flow)
    solver_fallbacks = controller.fallback_count
    history_fallback_count = sum(1 for d in controller.step_diagnostics if d.fallback_used)

    # Forecast Accuracy Analysis:
    # 1. 1-Hour-Ahead Forecast Accuracy (step k=0 of each LP plan vs realized)
    load_errors_1h = []
    solar_errors_1h = []
    wind_errors_1h = []
    for i, row in enumerate(hourly):
        diag = controller.step_diagnostics[i]
        realized_load = row["campus_demand_kw"]
        realized_solar = row["solar_available_kw"]
        realized_wind = row["wind_available_kw"]

        pred_load = diag.forecast_demand_kw[0]
        pred_solar = diag.forecast_solar_kw[0]
        pred_wind = diag.forecast_wind_kw[0]

        load_errors_1h.append(pred_load - realized_load)
        solar_errors_1h.append(pred_solar - realized_solar)
        wind_errors_1h.append(pred_wind - realized_wind)

    load_1h_mae = statistics.mean(abs(e) for e in load_errors_1h)
    load_1h_rmse = math.sqrt(statistics.mean(e**2 for e in load_errors_1h))
    solar_1h_mae = statistics.mean(abs(e) for e in solar_errors_1h)
    solar_1h_rmse = math.sqrt(statistics.mean(e**2 for e in solar_errors_1h))
    wind_1h_mae = statistics.mean(abs(e) for e in wind_errors_1h)
    wind_1h_rmse = math.sqrt(statistics.mean(e**2 for e in wind_errors_1h))

    # 2. Multi-step forecast accuracy across all horizons k in 1..24
    # Map realized values by timestamp
    realized_by_ts = {
        row["timestamp_utc"]: {
            "load": row["campus_demand_kw"],
            "solar": row["solar_available_kw"],
            "wind": row["wind_available_kw"],
        }
        for row in hourly
    }
    # Also add the final observation row
    final_obs = val_data[-1]
    realized_by_ts[final_obs.timestamp_utc] = {
        "load": final_obs.campus_load_kw,
        "solar": final_obs.pv_available_kw,
        "wind": final_obs.wind_available_kw,
    }

    horizon_load_errors = defaultdict(list)
    horizon_solar_errors = defaultdict(list)
    horizon_wind_errors = defaultdict(list)

    for diag in controller.step_diagnostics:
        for k, target_ts in enumerate(diag.target_timestamps_utc):
            h_step = k + 1  # 1-indexed horizon
            if target_ts in realized_by_ts:
                real = realized_by_ts[target_ts]
                e_load = diag.forecast_demand_kw[k] - real["load"]
                e_solar = diag.forecast_solar_kw[k] - real["solar"]
                e_wind = diag.forecast_wind_kw[k] - real["wind"]

                horizon_load_errors[h_step].append(e_load)
                horizon_solar_errors[h_step].append(e_solar)
                horizon_wind_errors[h_step].append(e_wind)

    horizon_metrics = {}
    for h_step in sorted(horizon_load_errors.keys()):
        l_errs = horizon_load_errors[h_step]
        s_errs = horizon_solar_errors[h_step]
        w_errs = horizon_wind_errors[h_step]

        horizon_metrics[h_step] = {
            "load_mae_kw": statistics.mean(abs(e) for e in l_errs),
            "load_rmse_kw": math.sqrt(statistics.mean(e**2 for e in l_errs)),
            "solar_mae_kw": statistics.mean(abs(e) for e in s_errs),
            "solar_rmse_kw": math.sqrt(statistics.mean(e**2 for e in s_errs)),
            "wind_mae_kw": statistics.mean(abs(e) for e in w_errs),
            "wind_rmse_kw": math.sqrt(statistics.mean(e**2 for e in w_errs)),
        }

    # Overall 24-hour mean MAE and RMSE across all horizons
    all_l_errs = [e for errs in horizon_load_errors.values() for e in errs]
    all_s_errs = [e for errs in horizon_solar_errors.values() for e in errs]
    all_w_errs = [e for errs in horizon_wind_errors.values() for e in errs]

    overall_mae_load = statistics.mean(abs(e) for e in all_l_errs)
    overall_rmse_load = math.sqrt(statistics.mean(e**2 for e in all_l_errs))
    overall_mae_solar = statistics.mean(abs(e) for e in all_s_errs)
    overall_rmse_solar = math.sqrt(statistics.mean(e**2 for e in all_s_errs))
    overall_mae_wind = statistics.mean(abs(e) for e in all_w_errs)
    overall_rmse_wind = math.sqrt(statistics.mean(e**2 for e in all_w_errs))

    report = {
        "controller": "gnn_transformer_mpc_24h",
        "checkpoint": str(DEFAULT_CHECKPOINT_PATH),
        "horizon_hours": mpc_config.horizon_hours,
        "steps": summary["steps"],
        "initial_soc": summary["initial_soc"],
        "initial_energy_kwh": init_e,
        "terminal_soc": summary["terminal_soc"],
        "terminal_energy_kwh": term_e,
        "delta_energy_kwh": term_e - init_e,
        "replacement_reference_value_usd_per_kwh": ref_value,
        "terminal_inventory_credit_usd": adj_data["inventory_adjustment_usd"],
        "raw_net_operating_cost_usd": raw_cost,
        "inventory_adjusted_cost_usd": adj_data["inventory_adjusted_net_operating_cost_usd"],
        "battery_throughput_kwh": throughput_kwh,
        "degradation_cost_usd": degradation_cost,
        "grid_import_energy_kwh": import_energy_kwh,
        "grid_export_energy_kwh": export_energy_kwh,
        "curtailment_energy_kwh": curtailment_kwh,
        "physical_violation_count": summary["physical_violation_count"],
        "adjustment_count": summary["adjustment_count"],
        "solver_fallbacks": solver_fallbacks,
        "history_fallback_steps": history_fallback_count,
        "simultaneous_battery_flow_steps": simult_battery_count,
        "timing_ms": {
            "solver": {
                "mean": mean_solve_ms,
                "median": median_solve_ms,
                "max": max_solve_ms,
                "min": min_solve_ms,
                "total_seconds": sum(controller.solve_times),
            },
            "inference": {
                "mean": mean_inf_ms,
                "median": median_inf_ms,
                "max": max_inf_ms,
                "min": min_inf_ms,
                "total_seconds": sum(controller.inference_times),
            },
            "total_replay_elapsed_s": total_elapsed,
        },
        "forecast_accuracy": {
            "1h_ahead": {
                "load_mae_kw": load_1h_mae,
                "load_rmse_kw": load_1h_rmse,
                "solar_mae_kw": solar_1h_mae,
                "solar_rmse_kw": solar_1h_rmse,
                "wind_mae_kw": wind_1h_mae,
                "wind_rmse_kw": wind_1h_rmse,
            },
            "overall_24h": {
                "load_mae_kw": overall_mae_load,
                "load_rmse_kw": overall_rmse_load,
                "solar_mae_kw": overall_mae_solar,
                "solar_rmse_kw": overall_rmse_solar,
                "wind_mae_kw": overall_mae_wind,
                "wind_rmse_kw": overall_rmse_wind,
            },
            "by_horizon": horizon_metrics,
        },
        "action_distribution": {
            "charge_steps": charge_count,
            "discharge_steps": discharge_count,
            "idle_steps": idle_count,
            "overnight_charge_steps_00_06h": overnight_charges,
            "prepeak_charge_steps_07_10h": prepeak_charges,
            "peak_discharge_steps_11_15h": tariff_peak_discharges,
        },
        "hourly_mean_soc": hourly_mean_soc,
        "hourly_mean_action_kw": hourly_mean_action,
    }

    local_dir = ROOT / "results" / "part7_4_gnn_mpc"
    local_dir.mkdir(parents=True, exist_ok=True)
    out_file = local_dir / "gnn_mpc_validation_summary.json"
    out_file.write_text(json.dumps(report, indent=2))
    print(f"Results written to {out_file}")

    hourly_file = local_dir / "gnn_mpc_validation_hourly.json"
    hourly_file.write_text(json.dumps(hourly, indent=2))
    print(f"Hourly replay log written to {hourly_file}")

    print("\n" + "=" * 80)
    print("PHASE 7.4 GNN-TRANSFORMER MPC VALIDATION RESULTS SUMMARY")
    print("=" * 80)
    print(f"Steps dispatched: {report['steps']}")
    print(f"Raw Net Operating Cost: ${report['raw_net_operating_cost_usd']:,.2f}")
    print(f"Terminal Inventory Credit: ${report['terminal_inventory_credit_usd']:,.2f}")
    print(f"Inventory-Adjusted Cost: ${report['inventory_adjusted_cost_usd']:,.2f}")
    print(f"Initial SOC: {report['initial_soc']:.4f} -> Terminal SOC: {report['terminal_soc']:.4f}")
    print(f"Battery Throughput: {report['battery_throughput_kwh']:,.2f} kWh AC")
    print(f"Degradation Cost: ${report['degradation_cost_usd']:,.2f}")
    print(f"Grid Import Energy: {report['grid_import_energy_kwh']:,.2f} kWh")
    print(f"Grid Export Energy: {report['grid_export_energy_kwh']:,.2f} kWh")
    print(f"Physical Violations: {report['physical_violation_count']}")
    print(f"Adjustments: {report['adjustment_count']}")
    print(f"Solver Fallbacks: {report['solver_fallbacks']}")
    print(f"Mean Inference Time: {mean_inf_ms:.2f} ms")
    print(f"Mean Solve Time: {mean_solve_ms:.2f} ms")
    print("\nForecast Accuracy (1-Hour-Ahead):")
    print(f"  Load MAE: {load_1h_mae:.2f} kW | RMSE: {load_1h_rmse:.2f} kW")
    print(f"  Solar MAE: {solar_1h_mae:.2f} kW | RMSE: {solar_1h_rmse:.2f} kW")
    print(f"  Wind MAE: {wind_1h_mae:.2f} kW | RMSE: {wind_1h_rmse:.2f} kW")
    print("\nForecast Accuracy (24-Hour Horizon Overall):")
    print(f"  Load MAE: {overall_mae_load:.2f} kW | RMSE: {overall_rmse_load:.2f} kW")
    print(f"  Solar MAE: {overall_mae_solar:.2f} kW | RMSE: {overall_rmse_solar:.2f} kW")
    print(f"  Wind MAE: {overall_mae_wind:.2f} kW | RMSE: {overall_rmse_wind:.2f} kW")
    print("=" * 80)


if __name__ == "__main__":
    run_gnn_mpc_evaluation()
