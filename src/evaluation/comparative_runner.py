"""Unified Comparative Evaluation Runner for Microgrid Controllers.

Orchestrates identical-episode replays across rule-based, DRL, and MPC controllers
over the validation partition, computing standardized comparative metrics and verifying
exact parity against historical benchmark artifacts.
"""

from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import pandas as pd

from scripts.run_simulation import _components
from src.evaluation.baseline_runner import _run_controller
from src.evaluation.comparative_metrics import (
    ComparativeEvaluationMetrics,
    calculate_comparative_metrics,
)
from src.evaluation.controller_registry import (
    CONTROLLER_REGISTRY,
    BenchmarkStatus,
    CausalityType,
    ControllerCategory,
    ControllerRegistryEntry,
    get_controller_entry,
    get_primary_benchmark_ids,
    list_controllers,
)
from src.simulation.config import load_simulation_config
from src.simulation.data_loader import load_simulation_split
from src.simulation.environment import MicrogridEnvironment
from src.simulation.metrics import load_cost_config


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLIT_ROOT = ROOT / "data" / "processed" / "splits_hypothetical_campus_v1"
DEFAULT_ASSETS_PATH = ROOT / "configs" / "synthetic_scenario.yaml"
DEFAULT_RATING_ORIGIN = "hypothetical_example"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "part8_comparative_benchmark"


class ComparativeEvaluationError(Exception):
    """Raised when an evaluation constraint or verification fails."""


def import_historical_benchmark(controller_id: str) -> dict[str, Any]:
    """Import and parse the verified historical benchmark summary for a controller.

    Args:
        controller_id: Unique registered controller identifier.

    Returns:
        Dictionary containing historical summary metrics and artifact provenance.
    """
    entry = get_controller_entry(controller_id)
    artifact_rel = entry.validation_benchmark_artifact
    artifact_path = ROOT / artifact_rel

    if not artifact_path.exists():
        # Check in external storage if path begins with experiments/
        external_root = Path(r"E:\AdaptiveEnergyBrainStorage")
        alt_path = external_root / artifact_rel
        if alt_path.exists():
            artifact_path = alt_path
        else:
            raise FileNotFoundError(
                f"Historical benchmark artifact not found at {artifact_path} or {alt_path}"
            )

    data = json.loads(artifact_path.read_text(encoding="utf-8"))

    # Map artifact structure depending on controller paradigm
    if "results" in data and controller_id in data["results"]:
        # Standard baseline runner structure
        summary = data["results"][controller_id]["summary"]
    elif "evaluation_summary" in data:
        # PPO validation experiment structure
        summary = data["evaluation_summary"]
    else:
        # Standalone summary JSON (MPC or specific baseline)
        summary = data

    return {
        "controller_id": controller_id,
        "mode": "imported_historical",
        "artifact_path": str(artifact_path),
        "raw_cost_usd": entry.authoritative_raw_cost_usd,
        "inventory_adjustment_usd": entry.authoritative_inventory_credit_usd,
        "adjusted_cost_usd": entry.authoritative_adjusted_cost_usd,
        "summary": summary,
    }


def reproduce_controller_replay(
    controller_id: str,
    *,
    partition: str = "validation",
    split_root: Path | str = DEFAULT_SPLIT_ROOT,
    assets_path: Path | str = DEFAULT_ASSETS_PATH,
    rating_origin: str = DEFAULT_RATING_ORIGIN,
    output_dir: Path | str | None = None,
    allow_held_out_test: bool = False,
) -> tuple[ComparativeEvaluationMetrics, list[dict[str, Any]], dict[str, Any]]:
    """Execute a deterministic replay of a registered controller on an approved partition.

    Args:
        controller_id: Unique registered controller identifier.
        partition: Partition name (strictly 'validation' unless explicit approval granted).
        split_root: Path to processed chronological splits.
        assets_path: Path to scenario asset definitions.
        rating_origin: Provenance string for asset ratings.
        output_dir: Destination directory for artifacts (optional).
        allow_held_out_test: Explicit safety gate required for 'test' partition.

    Returns:
        Tuple of (ComparativeEvaluationMetrics, hourly_records, provenance_dict).
    """
    if partition == "test" and not allow_held_out_test:
        raise PermissionError(
            "Access to held-out test partition is strictly blocked without explicit approval."
        )

    if partition not in {"validation", "train"}:
        if partition != "test":
            raise ValueError(f"Unknown or unsupported partition: {partition}")

    entry = get_controller_entry(controller_id)
    split_root_p = Path(split_root)
    assets_path_p = Path(assets_path)

    # 1. Load partition observations
    data = load_simulation_split(split_root_p, partition)

    # 2. Build physical simulation environment
    sim_config = load_simulation_config()
    cost_config = load_cost_config()
    renewables, demand = _components(data, assets_path_p, rating_origin)

    env = MicrogridEnvironment(
        data,
        renewables,
        demand,
        sim_config,
        cost_config,
    )

    # 3. Instantiate controller
    if controller_id == "mpc_oracle_rolling_24h":
        controller = entry.factory(partition_data=data)
    elif controller_id == "mpc_gnn_transformer":
        if partition == "validation":
            train_data = load_simulation_split(split_root_p, "train")
            controller = entry.factory(initial_history=train_data[-24:], history_source=train_data)
        elif partition == "test":
            val_data = load_simulation_split(split_root_p, "validation")
            controller = entry.factory(initial_history=val_data[-24:], history_source=val_data)
        else:
            controller = entry.factory()
    elif controller_id == "mpc_seasonal_persistence":
        if partition == "validation":
            train_data = load_simulation_split(split_root_p, "train")
            controller = entry.factory(initial_history=train_data[-24:])
        elif partition == "test":
            val_data = load_simulation_split(split_root_p, "validation")
            controller = entry.factory(initial_history=val_data[-24:])
        else:
            controller = entry.factory()
    else:
        controller = entry.factory()

    # 4. Execute deterministic replay with high-resolution per-step timing
    step_latencies_ms: list[float] = []

    controller.reset()
    initial_obs = env.reset()
    initial_soc = initial_obs.state.battery_soc

    hourly: list[dict[str, Any]] = []

    while not env.terminated:
        obs = env.observation
        t0 = time.perf_counter()
        action = controller.select_action(obs)
        t_ms = (time.perf_counter() - t0) * 1000.0
        step_latencies_ms.append(t_ms)

        step = env.step(action)
        dispatch = step.dispatch
        values = asdict(step.metrics.values)
        cumulative = asdict(step.cumulative_metrics)

        record = {
            "timestamp_utc": dispatch.timestamp_utc,
            "interval_hours": step.metrics.time_step_hours,
            "currency_code": step.metrics.currency_code,
            "synthetic_inputs_present": True,
            "solar_available_kw": sum(
                item.available_generation_kw
                for item in dispatch.renewable_outputs
                if item.generation_type == "solar"
            ),
            "wind_available_kw": sum(
                item.available_generation_kw
                for item in dispatch.renewable_outputs
                if item.generation_type == "wind"
            ),
            "renewable_used_kw": dispatch.renewables.total_used_kw,
            "campus_demand_kw": dispatch.demand.campus_load_kw,
            "served_demand_kw": dispatch.served_demand_kw,
            "unserved_demand_kw": dispatch.unserved_demand_kw,
            "battery_soc_before": dispatch.battery.state_before.battery_soc,
            "battery_soc_after": dispatch.battery.state_after.battery_soc,
            "battery_requested_kw": dispatch.requested_battery_power_kw,
            "battery_actual_kw": dispatch.actual_battery_power_kw,
            "battery_charge_kw": dispatch.battery.charge_power_kw,
            "battery_discharge_kw": dispatch.battery.discharge_power_kw,
            "grid_import_kw": dispatch.grid.import_power_kw,
            "grid_export_kw": dispatch.grid.export_power_kw,
            "curtailment_kw": dispatch.renewables.total_curtailed_kw,
            **values,
            "renewable_utilization": step.metrics.values.renewable_utilization,
            "power_balance_residual_kw": dispatch.balance_residual_kw,
            "energy_balance_residual_kwh": dispatch.energy_balance_residual_kwh,
            "adjustment_count": len(step.metrics.adjustments),
            "violation_count": len(step.metrics.violations),
            "step_latency_ms": t_ms,
        }
        hourly.append(record)

    # 5. Extract fallback counts if tracked
    fallback_count = getattr(controller, "fallback_count", 0)

    # 6. Compute standardized metrics
    metrics = calculate_comparative_metrics(
        hourly,
        sim_config=sim_config,
        cost_config=cost_config,
        solver_fallback_count=fallback_count,
        timing_records_ms=step_latencies_ms,
    )

    provenance = {
        "controller_id": controller_id,
        "mode": "fresh_replay_reproduced",
        "partition": partition,
        "steps": len(hourly),
        "initial_soc": initial_soc,
        "terminal_soc": metrics.terminal_soc,
        "authoritative_benchmark_adjusted_cost_usd": entry.authoritative_adjusted_cost_usd,
        "reproduced_adjusted_cost_usd": metrics.inventory_adjusted_cost_usd,
        "discrepancy_usd": (
            abs(metrics.inventory_adjusted_cost_usd - entry.authoritative_adjusted_cost_usd)
            if entry.authoritative_adjusted_cost_usd is not None
            else None
        ),
        "controller_metadata": entry.to_dict(),
    }

    # 7. Write isolated outputs if output_dir specified
    if output_dir is not None:
        ctrl_out = Path(output_dir) / controller_id
        ctrl_out.mkdir(parents=True, exist_ok=True)

        summary_file = ctrl_out / "summary.json"
        summary_file.write_text(
            json.dumps(metrics.to_dict(), indent=2), encoding="utf-8"
        )

        provenance_file = ctrl_out / "provenance.json"
        provenance_file.write_text(
            json.dumps(provenance, indent=2), encoding="utf-8"
        )

        hourly_file = ctrl_out / "hourly_records.json"
        hourly_file.write_text(
            json.dumps(hourly, indent=2), encoding="utf-8"
        )

    return metrics, hourly, provenance


def run_comparative_benchmark_suite(
    controller_ids: Sequence[str] | None = None,
    *,
    partition: str = "validation",
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    tolerance_usd: float = 0.01,
    allow_held_out_test: bool = False,
) -> dict[str, Any]:
    """Execute complete comparative benchmark suite across registered controllers.

    Args:
        controller_ids: Subset of controller IDs (defaults to all primary benchmarks).
        partition: Partition name (strictly 'validation' or explicit 'test').
        output_dir: Root directory for output artifacts.
        tolerance_usd: Numerical tolerance for parity against authoritative benchmark.
        allow_held_out_test: Explicit safety gate required for 'test' partition.

    Returns:
        Consolidated benchmark report dictionary.
    """
    if controller_ids is None:
        target_ids = get_primary_benchmark_ids()
    else:
        target_ids = list(controller_ids)

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    suite_results: dict[str, Any] = {}
    benchmark_table: list[dict[str, Any]] = []

    all_passed = True

    for cid in target_ids:
        entry = get_controller_entry(cid)
        print(f"\nEvaluating controller: {entry.display_name} ({cid})...")
        t_start = time.perf_counter()

        metrics, hourly, prov = reproduce_controller_replay(
            cid,
            partition=partition,
            output_dir=out_root,
            allow_held_out_test=allow_held_out_test,
        )
        elapsed = time.perf_counter() - t_start

        # Check reconciliation tolerance
        expected_adj = entry.authoritative_adjusted_cost_usd
        actual_adj = metrics.inventory_adjusted_cost_usd

        if expected_adj is not None:
            diff = abs(actual_adj - expected_adj)
            is_match = diff <= tolerance_usd
            if not is_match:
                all_passed = False
                status_str = f"FAILED (diff: ${diff:+.4f})"
            else:
                status_str = f"PASSED (exact match, diff: ${diff:.4f})"
        else:
            diff = 0.0
            is_match = True
            status_str = "NEW BASELINE (no prior benchmark)"

        print(f"  -> Raw Cost: ${metrics.raw_net_operating_cost_usd:,.2f}")
        print(f"  -> Inventory Credit: ${metrics.terminal_inventory_credit_usd:,.2f}")
        print(f"  -> Adjusted Cost: ${metrics.inventory_adjusted_cost_usd:,.2f} | {status_str}")
        print(f"  -> Latency: {metrics.latency_ms_mean:.2f} ms/step (total {elapsed:.1f}s)")

        row = {
            "controller_id": cid,
            "display_name": entry.display_name,
            "category": entry.category.value,
            "causality": entry.causality.value,
            "status": entry.benchmark_status.value,
            "raw_cost_usd": metrics.raw_net_operating_cost_usd,
            "terminal_inventory_credit_usd": metrics.terminal_inventory_credit_usd,
            "inventory_adjusted_cost_usd": metrics.inventory_adjusted_cost_usd,
            "historical_benchmark_usd": expected_adj,
            "discrepancy_usd": diff,
            "parity_verified": is_match,
            "battery_throughput_kwh": metrics.battery_throughput_kwh_ac,
            "battery_degradation_cost_usd": metrics.battery_degradation_cost_usd,
            "grid_import_kwh": metrics.grid_import_energy_kwh,
            "grid_export_kwh": metrics.grid_export_energy_kwh,
            "curtailment_kwh": metrics.renewable_curtailment_energy_kwh,
            "terminal_soc": metrics.terminal_soc,
            "physical_violations": metrics.physical_violation_count,
            "simulator_adjustments": metrics.simulator_adjustment_count,
            "simultaneous_flow_steps": metrics.simultaneous_flow_steps,
            "mean_latency_ms": metrics.latency_ms_mean,
            "total_elapsed_seconds": elapsed,
        }
        benchmark_table.append(row)
        suite_results[cid] = {
            "metrics": metrics.to_dict(),
            "provenance": prov,
        }

    # Save consolidated machine-readable benchmark table
    table_file = out_root / "benchmark_table.json"
    table_file.write_text(json.dumps(benchmark_table, indent=2), encoding="utf-8")

    # Generate Markdown table
    md_lines = [
        "# Unified Comparative Benchmark Table (Validation Split)",
        "",
        "| Controller | Category | Causality | Raw Cost ($) | Inventory Credit ($) | Adjusted Cost ($) | Historical Benchmark ($) | Discrepancy ($) | Parity | AC Throughput (kWh) | Latency (ms) |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for r in benchmark_table:
        md_lines.append(
            f"| **{r['display_name']}** | {r['category']} | {r['causality']} | "
            f"${r['raw_cost_usd']:,.2f} | ${r['terminal_inventory_credit_usd']:,.2f} | "
            f"**${r['inventory_adjusted_cost_usd']:,.2f}** | "
            f"${r['historical_benchmark_usd']:,.2f} | "
            f"${r['discrepancy_usd']:.4f} | "
            f"{'PASS' if r['parity_verified'] else 'FAIL'} | "
            f"{r['battery_throughput_kwh']:,.1f} | "
            f"{r['mean_latency_ms']:.2f} ms |"
        )

    summary_md = out_root / "benchmark_summary.md"
    summary_md.write_text("\n".join(md_lines), encoding="utf-8")

    report = {
        "suite_status": "PASSED" if all_passed else "FAILED_DISCREPANCY",
        "evaluated_partition": partition,
        "total_controllers": len(benchmark_table),
        "tolerance_usd": tolerance_usd,
        "controllers": suite_results,
        "benchmark_table": benchmark_table,
    }
    consolidated_file = out_root / "comparative_suite_report.json"
    consolidated_file.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nComparative suite finished. All parity verified: {all_passed}")
    print(f"Consolidated results saved to: {out_root}")

    return report
