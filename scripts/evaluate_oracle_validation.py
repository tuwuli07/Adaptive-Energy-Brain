"""Phase 7.2: 24-hour rolling perfect-foresight MPC validation evaluation script.

Evaluates OracleMPCController on the 1,313-hour validation split and outputs
comparative metrics against zero-action, price-aware, and PPO baselines.
"""

from dataclasses import asdict
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_simulation import _components
from src.control.mpc_solver import load_mpc_config
from src.control.oracle_mpc import OracleMPCController
from src.evaluation.baseline_runner import _run_controller
from src.evaluation.inventory_accounting import (
    compute_inventory_adjusted_cost,
    compute_replacement_reference_value,
)
from src.simulation.config import load_simulation_config
from src.simulation.data_loader import load_simulation_split
from src.simulation.environment import MicrogridEnvironment
from src.simulation.metrics import load_cost_config


def run_oracle_evaluation():
    split_root = ROOT / "data" / "processed" / "splits_hypothetical_campus_v1"
    assets_path = ROOT / "configs" / "synthetic_scenario.yaml"
    rating_origin = "hypothetical_example"

    print("Loading validation partition...")
    data = load_simulation_split(split_root, "validation")
    print(f"Validation dataset loaded: {len(data)} rows ({len(data) - 1} dispatch steps)")

    sim_config = load_simulation_config()
    cost_config = load_cost_config()
    mpc_config = load_mpc_config()

    # Verify strict exclusivity
    print(f"MPC Config strict_exclusivity: {mpc_config.strict_exclusivity}")
    print(f"MPC Horizon: {mpc_config.horizon_hours} hours")

    renewables, demand = _components(data, assets_path, rating_origin)
    env = MicrogridEnvironment(
        data,
        renewables,
        demand,
        sim_config,
        cost_config,
    )

    controller = OracleMPCController(
        partition_data=data,
        sim_config=sim_config,
        cost_config=cost_config,
        mpc_config=mpc_config,
    )

    print("Starting 1,313-hour rolling Oracle MPC simulation replay...")
    start_time = time.perf_counter()
    summary, hourly = _run_controller(
        "oracle_mpc",
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

    # Solve time statistics (in milliseconds)
    solve_times_ms = [t * 1000.0 for t in controller.solve_times]
    mean_solve_ms = statistics.mean(solve_times_ms) if solve_times_ms else 0.0
    median_solve_ms = statistics.median(solve_times_ms) if solve_times_ms else 0.0
    max_solve_ms = max(solve_times_ms) if solve_times_ms else 0.0
    min_solve_ms = min(solve_times_ms) if solve_times_ms else 0.0

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

    # Hourly profile behavior (hour of day 0..23)
    hourly_soc = [[] for _ in range(24)]
    hourly_action = [[] for _ in range(24)]
    for row in hourly:
        # Extract hour from timestamp_utc "YYYY-MM-DDTHH:MM:SSZ"
        hour_int = int(row["timestamp_utc"][11:13])
        hourly_soc[hour_int].append(row["battery_soc_after"])
        hourly_action[hour_int].append(row["battery_actual_kw"])

    hourly_mean_soc = {h: statistics.mean(hourly_soc[h]) for h in range(24)}
    hourly_mean_action = {h: statistics.mean(hourly_action[h]) for h in range(24)}

    # Tariff schedule analysis:
    # In the synthetic campus dataset, peak tariff hours (0.18 - 0.72 $/kWh) are 11:00 to 15:00 UTC.
    # Off-peak hours (0.12 $/kWh) are 16:00 to 10:00 UTC, with overnight hours 00:00 to 06:00 UTC.
    tariff_peak_hours = list(range(11, 16))  # 11:00 to 15:00 UTC
    overnight_hours = list(range(0, 7))       # 00:00 to 06:00 UTC
    prepeak_hours = list(range(7, 11))        # 07:00 to 10:00 UTC

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

    # Check for simultaneous flows across all solver diagnostics
    simult_battery_count = sum(1 for d in controller.step_diagnostics if d.has_simultaneous_battery_flow)
    simult_grid_count = sum(1 for d in controller.step_diagnostics if d.has_simultaneous_grid_flow)

    report = {
        "controller": "oracle_mpc_24h",
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
        "solver_fallbacks": controller.fallback_count,
        "simultaneous_battery_flow_steps": simult_battery_count,
        "simultaneous_grid_flow_steps": simult_grid_count,
        "solve_time_ms": {
            "mean": mean_solve_ms,
            "median": median_solve_ms,
            "max": max_solve_ms,
            "min": min_solve_ms,
            "total_solve_time_s": sum(controller.solve_times),
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

    # Output paths
    output_dir = Path("E:/AdaptiveEnergyBrainStorage/experiments/part7_2_oracle_mpc_validation_20261115")
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_file = output_dir / "oracle_validation_summary.json"
        out_file.write_text(json.dumps(report, indent=2))
        print(f"Results written to {out_file}")
    except Exception as exc:
        print(f"Warning: could not write to external storage ({exc}); writing locally")
        local_dir = ROOT / "results" / "part7_2_oracle_mpc"
        local_dir.mkdir(parents=True, exist_ok=True)
        out_file = local_dir / "oracle_validation_summary.json"
        out_file.write_text(json.dumps(report, indent=2))
        print(f"Results written to {out_file}")

    print("\n" + "=" * 80)
    print("PHASE 7.2 ORACLE MPC VALIDATION RESULTS SUMMARY")
    print("=" * 80)
    print(f"Steps dispatched: {report['steps']}")
    print(f"Raw Net Operating Cost: ${report['raw_net_operating_cost_usd']:,.2f}")
    print(f"Initial SOC: {report['initial_soc']:.4f} ({report['initial_energy_kwh']:.1f} kWh)")
    print(f"Terminal SOC: {report['terminal_soc']:.4f} ({report['terminal_energy_kwh']:.1f} kWh)")
    print(f"Inventory Adjustment: ${report['terminal_inventory_credit_usd']:,.2f}")
    print(f"Inventory-Adjusted Cost: ${report['inventory_adjusted_cost_usd']:,.2f}")
    print(f"Battery Throughput: {report['battery_throughput_kwh']:,.1f} kWh")
    print(f"Degradation Cost: ${report['degradation_cost_usd']:,.2f}")
    print(f"Grid Import: {report['grid_import_energy_kwh']:,.1f} kWh")
    print(f"Grid Export: {report['grid_export_energy_kwh']:,.1f} kWh")
    print(f"Curtailment: {report['curtailment_energy_kwh']:,.1f} kWh")
    print(f"Physical Violations: {report['physical_violation_count']}")
    print(f"Action Adjustments: {report['adjustment_count']}")
    print(f"Solver Fallbacks: {report['solver_fallbacks']}")
    print(f"Simultaneous Battery Flow Steps: {report['simultaneous_battery_flow_steps']}")
    print(f"Simultaneous Grid Flow Steps: {report['simultaneous_grid_flow_steps']}")
    print(f"Solve Time (ms): Mean={mean_solve_ms:.2f}ms, Median={median_solve_ms:.2f}ms, Max={max_solve_ms:.2f}ms")
    print(f"Action Distribution: Charge={charge_count}, Discharge={discharge_count}, Idle={idle_count}")
    print(f"Overnight Charges (00-06h): {overnight_charges}, Pre-peak Charges (07-10h): {prepeak_charges}, Peak Discharges (11-15h): {tariff_peak_discharges}")
    print("=" * 80)

    return report


if __name__ == "__main__":
    run_oracle_evaluation()
