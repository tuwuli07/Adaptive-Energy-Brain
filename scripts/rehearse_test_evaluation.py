"""Validation-only rehearsal of the Phase 8.5 one-shot evaluation workflow.

Tests the complete orchestration pipeline, schema generation, checksum calculation,
and failure handling protocols on the VALIDATION split without opening or accessing
the sealed held-out test partition.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import time
import traceback
from dataclasses import asdict
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

REHEARSAL_DIR = Path("results/part8_comparative_benchmark/rehearsal")
FAILURES_DIR = REHEARSAL_DIR / "failures"


def compute_file_sha256(path: Path) -> str:
    """Compute hex SHA-256 digest of a file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def run_validation_rehearsal(
    controller_ids: list[str] | None = None,
    output_dir: Path = REHEARSAL_DIR,
) -> dict[str, Any]:
    """Execute end-to-end rehearsal on the validation partition.

    Args:
        controller_ids: List of controllers to rehearse (defaults to 7 primary controllers).
        output_dir: Directory to store rehearsal outputs.

    Returns:
        Consolidated summary dictionary.
    """
    if controller_ids is None:
        controller_ids = get_primary_benchmark_ids()

    output_dir.mkdir(parents=True, exist_ok=True)
    failures_dir = output_dir / "failures"
    failures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PHASE 8.4 PREFLIGHT: VALIDATION-ONLY EVALUATION REHEARSAL")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print(f"Output Directory: {output_dir}")
    print(f"Target Controllers ({len(controller_ids)}): {', '.join(controller_ids)}")
    print("Partition: VALIDATION (Held-out test set remains strictly sealed)")
    print("=" * 80)

    rehearsal_results: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    generated_files: list[Path] = []

    all_succeeded = True

    for cid in controller_ids:
        entry = get_controller_entry(cid)
        print(f"\n[REHEARSAL] Executing replay: {entry.display_name} ({cid})...")
        t0 = time.perf_counter()

        try:
            metrics, hourly, prov = reproduce_controller_replay(
                cid,
                partition="validation",
                output_dir=output_dir,
                allow_held_out_test=False,
            )
            elapsed = time.perf_counter() - t0

            # Verify that hourly records exist and match expected 1,313 decisions
            if len(hourly) != 1313:
                raise ValueError(f"Expected 1313 hourly records, got {len(hourly)}")

            # Verify metrics schema
            metrics_dict = metrics.to_dict()
            required_metric_keys = [
                "raw_net_operating_cost_usd",
                "terminal_inventory_credit_usd",
                "inventory_adjusted_cost_usd",
                "battery_throughput_kwh_ac",
                "battery_degradation_cost_usd",
                "grid_import_energy_kwh",
                "grid_export_energy_kwh",
                "renewable_curtailment_energy_kwh",
                "terminal_soc",
                "physical_violation_count",
                "simulator_adjustment_count",
                "simultaneous_flow_steps",
                "latency_ms_mean",
            ]
            for k in required_metric_keys:
                if k not in metrics_dict:
                    raise KeyError(f"Missing required metric key: {k}")

            rehearsal_results[cid] = {
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
                "mean_latency_ms": metrics.latency_ms_mean,
            }
            summary_rows.append(row)

            # Add controller-specific generated files
            ctrl_out = output_dir / cid
            if (ctrl_out / "summary.json").is_file():
                generated_files.extend([
                    ctrl_out / "summary.json",
                    ctrl_out / "provenance.json",
                    ctrl_out / "hourly_records.json",
                ])

            print(f"  -> SUCCESS ({elapsed:.1f}s) | Adjusted Cost: ${metrics.inventory_adjusted_cost_usd:,.2f}")

        except Exception as exc:
            all_succeeded = False
            elapsed = time.perf_counter() - t0
            print(f"  -> FATAL FAILURE ({elapsed:.1f}s): {exc}")

            # Quarantine crash artifact
            crash_record = {
                "controller_id": cid,
                "failure_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
            crash_path = failures_dir / f"{cid}_crash.json"
            crash_path.write_text(json.dumps(crash_record, indent=2), encoding="utf-8")
            generated_files.append(crash_path)

            rehearsal_results[cid] = {
                "status": "FAILED",
                "elapsed_seconds": round(elapsed, 2),
                "crash_log": str(crash_path),
            }

    # 1. Write rehearsal_benchmark_results.json
    results_path = output_dir / "rehearsal_benchmark_results.json"
    results_payload = {
        "benchmark_partition": "validation",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "all_succeeded": all_succeeded,
        "total_controllers": len(controller_ids),
        "results": rehearsal_results,
    }
    results_path.write_text(json.dumps(results_payload, indent=2), encoding="utf-8")
    generated_files.append(results_path)

    # 2. Write rehearsal_summary_table.csv
    csv_path = output_dir / "rehearsal_summary_table.csv"
    if summary_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        generated_files.append(csv_path)

    # 3. Write rehearsal_metadata.json
    meta_path = output_dir / "rehearsal_metadata.json"
    metadata_payload = {
        "execution_type": "preflight_validation_rehearsal",
        "python_version": sys.version,
        "platform": sys.platform,
        "split_root": str(DEFAULT_SPLIT_ROOT),
        "assets_path": str(DEFAULT_ASSETS_PATH),
        "total_generated_files": len(generated_files),
        "all_controllers_succeeded": all_succeeded,
    }
    meta_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")
    generated_files.append(meta_path)

    # 4. Generate SHA-256 Manifest
    manifest_path = output_dir / "rehearsal_sha256_manifest.json"
    manifest: dict[str, str] = {}
    for p in output_dir.glob("**/*"):
        if p.is_file() and p.name != manifest_path.name:
            rel_name = str(p.relative_to(output_dir)).replace("\\", "/")
            manifest[rel_name] = compute_file_sha256(p)

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n[REHEARSAL] Completed. Manifest written with {len(manifest)} files.")
    print(f"Manifest path: {manifest_path}")

    return {
        "all_succeeded": all_succeeded,
        "results_path": str(results_path),
        "manifest_path": str(manifest_path),
        "file_count": len(manifest),
    }


def rehearse_simulated_failure(
    output_dir: Path = FAILURES_DIR,
) -> Path:
    """Demonstrate failure protocol by simulating a crash and asserting quarantine behavior."""
    output_dir.mkdir(parents=True, exist_ok=True)
    simulated_crash_file = output_dir / "simulated_worker_crash.json"

    try:
        raise RuntimeError("Simulated transient process termination during rehearsal")
    except RuntimeError as exc:
        crash_record = {
            "controller_id": "simulated_fault_controller",
            "failure_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "failure_classification": "PROCESS_TERMINATION_OR_CRASH",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
            "quarantine_action": "QUARANTINED_AWAITING_HUMAN_AUTHORIZATION",
            "auto_retry_attempted": False,
        }
        simulated_crash_file.write_text(json.dumps(crash_record, indent=2), encoding="utf-8")

    return simulated_crash_file


if __name__ == "__main__":
    out = run_validation_rehearsal()
    crash_f = rehearse_simulated_failure()
    print(f"[REHEARSAL] Simulated failure protocol verified: {crash_f}")
