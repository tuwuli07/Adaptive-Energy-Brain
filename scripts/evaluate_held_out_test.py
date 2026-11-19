"""Phase 8.5: One-Shot Comparative Evaluation on Held-Out Test Partition.

CRITICAL GOVERNANCE RULES:
1. This script must ONLY be executed after explicit human written authorization.
2. The held-out test split serves strictly as a single-pass confirmation of generalization.
3. No retraining, hyperparameter tuning, heuristic adjustment, or LP re-formulation
   is permitted following test-set evaluation.
4. Automatic retries are strictly forbidden. Any failure invalidates the run and
   requires renewed human authorization.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure project root is in python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.comparative_runner import (
    DEFAULT_ASSETS_PATH,
    DEFAULT_SPLIT_ROOT,
    reproduce_controller_replay,
)
from src.evaluation.controller_registry import (
    get_controller_entry,
    get_primary_benchmark_ids,
)

TEST_OUTPUT_DIR = Path("results/part8_comparative_benchmark/test_evaluation")
FAILURES_DIR = TEST_OUTPUT_DIR / "failures"


def compute_file_sha256(path: Path) -> str:
    """Compute hex SHA-256 digest of a file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def run_held_out_test_evaluation(
    *,
    confirm_unsealing: bool = False,
    controller_ids: list[str] | None = None,
    output_dir: Path = TEST_OUTPUT_DIR,
    partition: str = "test",
) -> dict[str, Any]:
    """Execute single-pass comparative evaluation on the held-out test partition.

    Args:
        confirm_unsealing: Explicit gate confirming human authorization to unseal test set.
        controller_ids: Subset of controllers (defaults to all 7 primary benchmarks).
        output_dir: Output directory for test evaluation artifacts.
        partition: Partition name (defaults strictly to 'test').

    Returns:
        Consolidated benchmark results dictionary.
    """
    if not confirm_unsealing:
        raise PermissionError(
            "Access to the held-out test partition is strictly locked. "
            "Execution requires explicit human authorization and the --confirm-test-unsealing flag."
        )

    if output_dir.exists():
        existing_artifacts = [p for p in output_dir.iterdir() if p.name != "failures"]
        if existing_artifacts:
            raise FileExistsError(
                f"Test output directory '{output_dir}' already exists and is non-empty. "
                "Silent overwrite of test evaluation results is strictly prohibited. "
                "Any re-execution requires explicit human authorization and manual archiving."
            )

    if controller_ids is None:
        controller_ids = get_primary_benchmark_ids()

    output_dir.mkdir(parents=True, exist_ok=True)
    failures_dir = output_dir / "failures"
    failures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PHASE 8.5: HELD-OUT TEST EVALUATION (ONE-SHOT EXECUTION)")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print(f"Output Directory: {output_dir}")
    print(f"Target Controllers ({len(controller_ids)}): {', '.join(controller_ids)}")
    print(f"Partition: {partition.upper()} (One-shot generalization assessment)")
    print("=" * 80)

    test_results: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    generated_files: list[Path] = []
    all_succeeded = True

    for cid in controller_ids:
        entry = get_controller_entry(cid)
        print(f"\n[TEST EVALUATION] Executing replay: {entry.display_name} ({cid})...")
        t0 = time.perf_counter()

        try:
            metrics, hourly, prov = reproduce_controller_replay(
                cid,
                partition=partition,
                output_dir=output_dir,
                allow_held_out_test=(partition == "test"),
            )
            elapsed = time.perf_counter() - t0

            if len(hourly) != 1313:
                raise ValueError(
                    f"Integrity violation: expected exactly 1313 hourly records for test partition, got {len(hourly)}"
                )

            metrics_dict = metrics.to_dict()
            test_results[cid] = {
                "status": "COMPLETED",
                "elapsed_seconds": round(elapsed, 2),
                "metrics": metrics_dict,
                "provenance": prov,
            }

            row = {
                "controller_id": cid,
                "display_name": entry.display_name,
                "category": entry.category.value,
                "causality": entry.causality.value,
                "raw_cost_usd": metrics.raw_net_operating_cost_usd,
                "terminal_credit_usd": metrics.terminal_inventory_credit_usd,
                "adjusted_cost_usd": metrics.inventory_adjusted_cost_usd,
                "throughput_kwh": metrics.battery_throughput_kwh_ac,
                "degradation_usd": metrics.battery_degradation_cost_usd,
                "import_kwh": metrics.grid_import_energy_kwh,
                "export_kwh": metrics.grid_export_energy_kwh,
                "curtailment_kwh": metrics.renewable_curtailment_energy_kwh,
                "terminal_soc": metrics.terminal_soc,
                "physical_violations": metrics.physical_violation_count,
                "simulator_adjustments": metrics.simulator_adjustment_count,
                "solver_fallbacks": metrics.solver_fallback_count,
                "simultaneous_flow_steps": metrics.simultaneous_flow_steps,
                "mean_latency_ms": metrics.latency_ms_mean,
            }
            summary_rows.append(row)

            ctrl_out = output_dir / cid
            if (ctrl_out / "summary.json").is_file():
                generated_files.extend([
                    ctrl_out / "summary.json",
                    ctrl_out / "provenance.json",
                    ctrl_out / "hourly_records.json",
                ])

            print(
                f"  -> SUCCESS ({elapsed:.1f}s) | "
                f"Adjusted Cost: ${metrics.inventory_adjusted_cost_usd:,.2f} | "
                f"Throughput: {metrics.battery_throughput_kwh_ac:,.1f} kWh"
            )

        except Exception as exc:
            all_succeeded = False
            elapsed = time.perf_counter() - t0
            print(f"  -> FATAL FAILURE ({elapsed:.1f}s) on controller {cid}: {exc}")

            crash_record = {
                "controller_id": cid,
                "failure_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
                "policy": "NO_AUTO_RETRY_AWAITING_EXPLICIT_HUMAN_INSPECTION",
            }
            crash_path = failures_dir / f"{cid}_crash.json"
            crash_path.write_text(json.dumps(crash_record, indent=2), encoding="utf-8")
            generated_files.append(crash_path)

            test_results[cid] = {
                "status": "FAILED",
                "elapsed_seconds": round(elapsed, 2),
                "crash_log": str(crash_path),
            }

            # Immediate abort on any failure: atomic one-shot rule
            print("\n[CRITICAL ERROR] Single-pass test execution halted due to fatal failure.")
            print("Automatic retries are prohibited by governance protocol.")
            break

    # Save test_benchmark_results.json
    results_path = output_dir / "test_benchmark_results.json"
    results_payload = {
        "benchmark_partition": partition,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "all_succeeded": all_succeeded,
        "total_controllers_evaluated": len(test_results),
        "results": test_results,
    }
    results_path.write_text(json.dumps(results_payload, indent=2), encoding="utf-8")
    generated_files.append(results_path)

    # Save test_summary_table.csv
    csv_path = output_dir / "test_summary_table.csv"
    if summary_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        generated_files.append(csv_path)

    # Save test_evaluation_metadata.json
    meta_path = output_dir / "test_evaluation_metadata.json"
    metadata_payload = {
        "execution_type": "held_out_test_evaluation_one_shot",
        "python_version": sys.version,
        "platform": sys.platform,
        "split_root": str(DEFAULT_SPLIT_ROOT),
        "assets_path": str(DEFAULT_ASSETS_PATH),
        "total_generated_files": len(generated_files),
        "all_controllers_succeeded": all_succeeded,
    }
    meta_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")
    generated_files.append(meta_path)

    # Save test_sha256_manifest.json
    manifest_path = output_dir / "test_sha256_manifest.json"
    manifest: dict[str, str] = {}
    for p in output_dir.glob("**/*"):
        if p.is_file() and p.name != manifest_path.name:
            rel_name = str(p.relative_to(output_dir)).replace("\\", "/")
            manifest[rel_name] = compute_file_sha256(p)

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n[TEST EVALUATION] Execution complete. SHA-256 manifest recorded ({len(manifest)} files).")
    print(f"Results directory: {output_dir}")

    return {
        "all_succeeded": all_succeeded,
        "results_path": str(results_path),
        "manifest_path": str(manifest_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute single-pass test set evaluation.")
    parser.add_argument(
        "--confirm-test-unsealing",
        action="store_true",
        help="Explicit confirmation that human authorization has been granted to unseal test set.",
    )
    args = parser.parse_args()

    if not args.confirm_test_unsealing:
        print("[ERROR] Execution blocked: missing --confirm-test-unsealing flag.")
        print("Access to held-out test set requires explicit human approval.")
        sys.exit(1)

    res = run_held_out_test_evaluation(confirm_unsealing=True)
    if not res["all_succeeded"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
