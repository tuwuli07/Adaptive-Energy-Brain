"""Validate and compare saved baseline experiment outputs."""

from dataclasses import fields
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

from src.simulation.metrics import MetricValues


_METRIC_FIELDS = tuple(field.name for field in fields(MetricValues))
_SHARED_FIELDS = (
    "currency_code",
    "input_rows",
    "steps",
    "initial_soc",
    "interval",
    "input_files",
    "simulation_config",
    "economic_assumptions",
    "episode_policy",
)


class BaselineComparisonError(ValueError):
    """Experiment artifacts are incomplete, inconsistent, or incomparable."""


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _close(actual, expected, label):
    if not math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-8):
        raise BaselineComparisonError(
            f"{label} does not match simulator-recorded cumulative metrics."
        )


def cost_improvement(baseline, candidate):
    """Return absolute savings and signed percent improvement for a cost.

    Lower costs are better. A negative baseline uses its absolute magnitude as
    denominator; a zero baseline has no defined percentage.
    """
    baseline = float(baseline)
    candidate = float(candidate)
    if not math.isfinite(baseline) or not math.isfinite(candidate):
        raise BaselineComparisonError("Comparison costs must be finite numbers.")
    savings = baseline - candidate
    if baseline == 0:
        return {
            "absolute_improvement": savings,
            "percentage_improvement": None,
            "percentage_status": "undefined_zero_baseline",
        }
    return {
        "absolute_improvement": savings,
        "percentage_improvement": savings / abs(baseline) * 100,
        "percentage_status": "defined",
    }


def _load_experiment(experiment_dir):
    root = Path(experiment_dir).resolve()
    summary_path = root / "experiment_summary.json"
    if not summary_path.is_file():
        raise BaselineComparisonError(
            f"Missing experiment summary: {summary_path}"
        )
    try:
        experiment = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BaselineComparisonError(f"Cannot read experiment summary: {exc}") from exc
    if experiment.get("status") != "complete":
        raise BaselineComparisonError("Only complete baseline experiments can be compared.")
    controllers = experiment.get("controllers")
    results = experiment.get("results")
    if not isinstance(controllers, list) or not isinstance(results, dict):
        raise BaselineComparisonError("Experiment summary has no controller results.")
    if "zero_action" not in controllers or "zero_action" not in results:
        raise BaselineComparisonError(
            "Include the zero_action controller to calculate baseline improvements."
        )
    if set(controllers) != set(results):
        raise BaselineComparisonError("Experiment controller list and result map differ.")

    loaded = {}
    first_shared = None
    reference_timestamps = None
    for name in controllers:
        entry = results[name]
        try:
            summary_path = (root / entry["summary"]).resolve()
            hourly_path = (root / entry["hourly_results"]).resolve()
            summary_path.relative_to(root)
            hourly_path.relative_to(root)
            if _sha256(summary_path) != entry["summary_sha256"]:
                raise BaselineComparisonError(f"{name}: summary checksum mismatch.")
            if _sha256(hourly_path) != entry["hourly_results_sha256"]:
                raise BaselineComparisonError(f"{name}: hourly checksum mismatch.")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            frame = pd.read_csv(hourly_path)
        except BaselineComparisonError:
            raise
        except (OSError, KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise BaselineComparisonError(
                f"{name}: incomplete or unreadable result artifacts: {exc}"
            ) from exc

        shared = summary.get("shared_experiment")
        if not isinstance(shared, dict):
            raise BaselineComparisonError(f"{name}: missing shared experiment metadata.")
        if first_shared is None:
            first_shared = shared
        else:
            for key in _SHARED_FIELDS:
                if shared.get(key) != first_shared.get(key):
                    raise BaselineComparisonError(
                        f"Controller results differ in shared experiment field {key!r}."
                    )
        for key in _SHARED_FIELDS:
            if shared.get(key) != experiment.get(key):
                raise BaselineComparisonError(
                    f"{name}: shared metadata {key!r} disagrees with experiment summary."
                )
        if summary.get("controller") != name:
            raise BaselineComparisonError(f"{name}: controller summary name mismatch.")
        if summary.get("currency_code") != experiment.get("currency_code"):
            raise BaselineComparisonError(f"{name}: currency mismatch.")

        required_columns = {
            "timestamp_utc",
            "interval_hours",
            "grid_import_kw",
            "grid_export_kw",
            "adjustment_count",
            "violation_count",
            *_METRIC_FIELDS,
        }
        missing = required_columns - set(frame.columns)
        if missing:
            raise BaselineComparisonError(
                f"{name}: hourly results missing fields {sorted(missing)}."
            )
        timestamps = tuple(frame["timestamp_utc"].astype(str))
        if len(timestamps) != summary.get("steps") or len(frame) != experiment.get("steps"):
            raise BaselineComparisonError(f"{name}: episode length mismatch.")
        if reference_timestamps is None:
            reference_timestamps = timestamps
        elif timestamps != reference_timestamps:
            raise BaselineComparisonError(
                "Controller results do not contain identical timestamp sequences."
            )
        if frame["interval_hours"].nunique() != 1:
            raise BaselineComparisonError(f"{name}: mixed time-step durations.")
        if not math.isclose(
            float(frame["interval_hours"].iloc[0]),
            float(experiment["simulation_config"]["time_step_hours"]),
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise BaselineComparisonError(f"{name}: hourly interval disagrees with config.")
        if not math.isclose(
            float(summary["initial_soc"]),
            float(experiment["initial_soc"]),
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise BaselineComparisonError(f"{name}: initial SOC mismatch.")

        totals = summary.get("cumulative_metrics")
        if not isinstance(totals, dict):
            raise BaselineComparisonError(f"{name}: missing cumulative simulator metrics.")
        for metric in _METRIC_FIELDS:
            if metric not in totals:
                raise BaselineComparisonError(f"{name}: missing cumulative metric {metric}.")
            _close(frame[metric].sum(), totals[metric], f"{name}.{metric}")
        if "renewable_utilization" in totals:
            actual_utilization = totals["renewable_utilization"]
            if actual_utilization is None:
                if totals["renewable_available_energy_kwh"] != 0:
                    raise BaselineComparisonError(
                        f"{name}: utilization is undefined despite renewable availability."
                    )
            elif totals["renewable_available_energy_kwh"] == 0:
                raise BaselineComparisonError(
                    f"{name}: utilization should be undefined with zero availability."
                )
            else:
                expected_utilization = (
                    totals["renewable_utilized_energy_kwh"]
                    / totals["renewable_available_energy_kwh"]
                )
                _close(actual_utilization, expected_utilization, f"{name}.renewable_utilization")

        violation_count = int(frame["violation_count"].sum())
        adjustment_count = int(frame["adjustment_count"].sum())
        if violation_count != summary.get("physical_violation_count"):
            raise BaselineComparisonError(f"{name}: violation counts do not reconcile.")
        if adjustment_count != summary.get("adjustment_count"):
            raise BaselineComparisonError(f"{name}: adjustment counts do not reconcile.")

        loaded[name] = {
            "summary": summary,
            "frame": frame,
            "timestamps": timestamps,
            "violation_count": violation_count,
            "adjustment_count": adjustment_count,
        }
    return root, experiment, loaded


def _comparison_metrics(name, result, baseline):
    summary = result["summary"]
    values = summary["cumulative_metrics"]
    baseline_values = baseline["summary"]["cumulative_metrics"]
    grid_transaction_cost = values["import_cost"] - values["export_revenue"]
    baseline_grid_transaction_cost = (
        baseline_values["import_cost"] - baseline_values["export_revenue"]
    )
    frame = result["frame"]
    row = {
        "controller": name,
        "currency_code": summary["currency_code"],
        "steps": summary["steps"],
        "first_dispatch_timestamp_utc": (
            result["timestamps"][0] if result["timestamps"] else None
        ),
        "last_dispatch_timestamp_utc": (
            result["timestamps"][-1] if result["timestamps"] else None
        ),
        "grid_import_cost": values["import_cost"],
        "grid_export_revenue": values["export_revenue"],
        "net_grid_transaction_cost": grid_transaction_cost,
        "net_operating_cost": values["net_operating_cost"],
        "renewable_utilization": values["renewable_utilization"],
        "renewable_curtailment_energy_kwh": values["renewable_curtailed_energy_kwh"],
        "battery_energy_throughput_kwh": values["battery_throughput_kwh"],
        "estimated_battery_degradation_cost": values["battery_degradation_cost"],
        "grid_import_energy_kwh": values["grid_import_energy_kwh"],
        "grid_export_energy_kwh": values["grid_export_energy_kwh"],
        "peak_grid_import_kw": (
            float(frame["grid_import_kw"].max()) if not frame.empty else 0.0
        ),
        "unserved_demand_energy_kwh": values["unserved_demand_energy_kwh"],
        "constraint_violations": result["violation_count"],
        "action_adjustments": result["adjustment_count"],
        "reliability_worse_than_zero_action": (
            values["unserved_demand_energy_kwh"]
            > baseline_values["unserved_demand_energy_kwh"] + 1e-9
            or result["violation_count"] > baseline["violation_count"]
        ),
        "zero_action_baseline": name == "zero_action",
    }
    for metric, baseline_value, candidate_value in (
        ("grid_import_cost", baseline_values["import_cost"], values["import_cost"]),
        (
            "net_grid_transaction_cost",
            baseline_grid_transaction_cost,
            grid_transaction_cost,
        ),
        (
            "net_operating_cost",
            baseline_values["net_operating_cost"],
            values["net_operating_cost"],
        ),
    ):
        improvement = cost_improvement(baseline_value, candidate_value)
        row[f"{metric}_absolute_improvement"] = improvement["absolute_improvement"]
        row[f"{metric}_percentage_improvement"] = improvement["percentage_improvement"]
        row[f"{metric}_percentage_status"] = improvement["percentage_status"]
    row["cost_superiority_claim_allowed"] = not row[
        "reliability_worse_than_zero_action"
    ]
    return row


def compare_baseline_experiment(experiment_dir, output=None):
    """Compare saved runner outputs against zero-action on identical periods.

    Results are kept separate from the original experiment directory. If no
    output path is supplied, a sibling directory named ``<experiment>_comparison``
    is exclusively created.
    """
    root, experiment, loaded = _load_experiment(experiment_dir)
    destination = (
        Path(output)
        if output is not None
        else root.with_name(f"{root.name}_comparison")
    )
    if destination.exists():
        raise FileExistsError(
            f"Comparison output already exists: {destination}; choose a fresh directory."
        )

    baseline = loaded["zero_action"]
    rows = [
        _comparison_metrics(name, result, baseline)
        for name, result in loaded.items()
    ]
    zero_row = next(row for row in rows if row["controller"] == "zero_action")
    for row in rows:
        if row["controller"] == "zero_action":
            continue
        if row["reliability_worse_than_zero_action"]:
            row["cost_superiority_claim_allowed"] = False

    report = {
        "status": "complete",
        "experiment_directory": str(root),
        "partition": experiment.get("partition"),
        "currency_code": experiment["currency_code"],
        "identical_episode": {
            "steps": experiment["steps"],
            "first_dispatch_timestamp_utc": (
                baseline["timestamps"][0] if baseline["timestamps"] else None
            ),
            "last_dispatch_timestamp_utc": (
                baseline["timestamps"][-1] if baseline["timestamps"] else None
            ),
            "initial_soc": experiment["initial_soc"],
            "input_files": experiment["input_files"],
        },
        "metric_definitions": {
            "grid_import_cost": "Simulator-recorded import_cost, in declared currency.",
            "grid_export_revenue": "Simulator-recorded export_revenue, in declared currency.",
            "net_grid_transaction_cost": "grid_import_cost minus grid_export_revenue; excludes modeled degradation and unserved-demand costs.",
            "net_operating_cost": "Simulator-recorded total: import cost - export revenue + battery degradation cost + unserved-demand cost.",
            "renewable_utilization": "Simulator MetricValues ratio of utilized renewable energy to available energy; null when availability is zero.",
            "renewable_curtailment_energy_kwh": "Simulator-recorded curtailed renewable energy.",
            "battery_energy_throughput_kwh": "Simulator-recorded AC charge plus discharge energy.",
            "estimated_battery_degradation_cost": "Simulator's configured linear throughput-cost proxy; not validated battery aging.",
            "peak_grid_import_kw": "Maximum recorded hourly grid_import_kw in the common episode.",
            "reliability_worse_than_zero_action": "True when unserved energy or physical constraint violation count exceeds zero-action.",
            "cost_improvement_percentage": "100 * (zero_action_cost - candidate_cost) / abs(zero_action_cost); undefined when zero-action cost is zero.",
        },
        "zero_action_reliability": {
            "unserved_demand_energy_kwh": zero_row["unserved_demand_energy_kwh"],
            "constraint_violations": zero_row["constraint_violations"],
        },
        "controllers": rows,
        "limitations": [
            "Cost improvement does not account for terminal SOC value; compare terminal SOC from controller summaries before interpreting cost-only outcomes.",
            "A controller with worse unserved demand or more physical violations is not marked as eligible for a cost-superiority claim.",
            "Battery degradation is the simulator's modeled linear throughput proxy, not measured aging.",
        ],
    }
    destination.mkdir(parents=True, exist_ok=False)
    csv_path = destination / "performance_comparison.csv"
    with csv_path.open("x", encoding="utf-8", newline="") as stream:
        pd.DataFrame(rows).to_csv(stream, index=False, float_format="%.12g")
    report["comparison_csv"] = csv_path.name
    json_path = destination / "performance_comparison.json"
    with json_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report
