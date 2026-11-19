"""Frozen full-partition validation, shared baseline replay, and joint screening."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import re
import time
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO

from scripts.run_simulation import _components
from src.baselines import available_controllers
from src.baselines.base import BaselineController
from src.evaluation.baseline_runner import _controller, _run_controller
from src.evaluation.inventory_accounting import (
    compute_inventory_adjusted_cost,
    compute_replacement_reference_value,
)
from src.rl.environment import DISCRETE_ACTION_MAP, map_normalized_action
from src.rl.episodes import load_verified_partition, sha256
from src.rl.observations import ObservationEncoder, SCHEMA
from src.rl.training import ROOT, source_snapshot, verify_checkpoint, write_json
from src.simulation.config import load_simulation_config
from src.simulation.environment import MicrogridEnvironment
from src.simulation.metrics import load_cost_config


def policy_state_hash(model):
    """Fingerprint policy and optimizer tensors and scalar state semantically."""
    digest = hashlib.sha256()
    def visit(value):
        if isinstance(value, torch.Tensor):
            array = value.detach().cpu().contiguous().numpy()
            digest.update(str((array.dtype, array.shape)).encode())
            digest.update(array.tobytes())
        elif isinstance(value, dict):
            for key in sorted(value, key=str):
                digest.update(str(key).encode())
                visit(value[key])
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)
        else:
            digest.update(repr(value).encode())
    visit(model.get_parameters())
    return digest.hexdigest()


class FrozenPPOController(BaselineController):
    """Use the identical causal encoder and action mapping as the Gym adapter."""

    def __init__(self, model, config):
        self.model = model
        self.encoder = ObservationEncoder(config)
        self.config = config
        model.policy.set_training_mode(False)

    def _select_action(self, observation):
        numeric = self.encoder.encode(observation)
        action, _ = self.model.predict(numeric, deterministic=True)
        if hasattr(self.model, "action_space") and hasattr(self.model.action_space, "n"):
            action = DISCRETE_ACTION_MAP[int(action)]
        return map_normalized_action(action, self.config)[1]


def eligible(record):
    return (record.get("status") == "complete" and record.get("physical_violation_count") == 0
            and record.get("cumulative_metrics", {}).get("unserved_demand_energy_kwh") == 0
            and record.get("steps") == record.get("expected_steps")
            and record.get("maximum_absolute_power_balance_residual_kw", float("inf")) <= 1e-6
            and record.get("maximum_absolute_energy_balance_residual_kwh", float("inf")) <= 1e-6)


def dominates(left, right, cost_tolerance=1e-6, energy_tolerance=1e-6):
    """Numerical tolerances only, with no terminal energy price or scalarization."""
    lc = left["cumulative_metrics"]["net_operating_cost"]
    rc = right["cumulative_metrics"]["net_operating_cost"]
    le, re = left["terminal_energy_kwh"], right["terminal_energy_kwh"]
    return (lc <= rc + cost_tolerance and le >= re - energy_tolerance
            and (lc < rc - cost_tolerance or le > re + energy_tolerance))


def aggregate_records(records, expected_seeds, required_timesteps, *, pilot=False,
                      cost_tolerance=1e-6, energy_tolerance=1e-6):
    """Aggregate seeds and retain failures. Pilot screening carries no winner."""
    expected_seeds = tuple(expected_seeds)
    if not pilot and required_timesteps != 32768:
        raise ValueError("The planned study selects only the fixed 32768-transition checkpoint.")
    if not expected_seeds or len(set(expected_seeds)) != len(expected_seeds):
        raise ValueError("Expected seeds must be unique and nonempty.")
    groups = {}
    for record in records:
        if record.get("kind") == "ppo":
            group = groups.setdefault(record["configuration"], {})
            if record["seed"] in group:
                raise ValueError("Duplicate configuration/seed record.")
            group[record["seed"]] = record
    summaries = {}
    for name, group in groups.items():
        valid = [r for r in group.values() if eligible(r) and r.get("checkpoint_timesteps") == required_timesteps]
        complete = set(group) == set(expected_seeds) and len(valid) == len(expected_seeds)
        summary = {"recorded_seeds": sorted(group), "expected_seeds": list(expected_seeds),
                   "failed_or_ineligible_seeds": sorted(seed for seed in expected_seeds
                                                        if seed not in group or group[seed] not in valid),
                   "selection_eligible": complete, "successful_seed_count": len(valid)}
        for field in ("net_operating_cost", "terminal_energy_kwh"):
            values = [r["cumulative_metrics"][field] if field == "net_operating_cost" else r[field] for r in valid]
            summary[field] = {"mean": float(np.mean(values)) if values else None,
                              "std_across_seeds": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                              "min": min(values) if values else None, "max": max(values) if values else None}
        summaries[name] = summary
    paired = []
    dominated = set()
    for left in sorted(groups):
        for right in sorted(groups):
            if left == right or not summaries[left]["selection_eligible"] or not summaries[right]["selection_eligible"]:
                continue
            deltas = []
            all_no_worse = True
            any_strict = False
            for seed in expected_seeds:
                a, b = groups[left][seed], groups[right][seed]
                dc = a["cumulative_metrics"]["net_operating_cost"] - b["cumulative_metrics"]["net_operating_cost"]
                de = a["terminal_energy_kwh"] - b["terminal_energy_kwh"]
                all_no_worse &= dc <= cost_tolerance and de >= -energy_tolerance
                any_strict |= dc < -cost_tolerance or de > energy_tolerance
                deltas.append({"seed": seed, "cost_difference": dc, "terminal_energy_difference_kwh": de})
            relation = bool(all_no_worse and any_strict)
            if relation:
                dominated.add(right)
            paired.append({"left": left, "right": right, "paired_differences": deltas,
                           "dominates_all_paired_seeds": relation})
    frontier = sorted(name for name in groups if summaries[name]["selection_eligible"] and name not in dominated)
    return {"schema_version": "1.0", "stage": "pilot" if pilot else "planned_study",
            "unit_of_variability": "training_seed", "required_checkpoint_timesteps": required_timesteps,
            "configurations": summaries, "paired_comparisons": paired, "pareto_set": frontier,
            "selected_configuration": None if pilot or len(frontier) != 1 else frontier[0],
            "research_winner": None,
            "decision": "pilot_descriptive_only" if pilot else ("unique_pareto_candidate" if len(frontier) == 1 else "no_unique_winner")}


def _enhance(summary, environment, config, costs=None, baseline_config=None):
    results = environment.dispatch_results
    requested = np.array([r.requested_battery_power_kw for r in results])
    executed = np.array([r.actual_battery_power_kw for r in results])
    socs = np.array([r.battery.state_after.battery_soc for r in results])
    initial_energy = results[0].battery.state_before.battery_energy_kwh
    terminal_energy = results[-1].battery.state_after.battery_energy_kwh
    energy_change = terminal_energy - initial_energy
    ref_value = compute_replacement_reference_value(
        config, costs, baseline_config=baseline_config
    )
    raw_cost = summary["cumulative_metrics"]["net_operating_cost"]
    adj_data = compute_inventory_adjusted_cost(
        raw_cost, initial_energy, terminal_energy, ref_value
    )
    summary.update({"expected_steps": environment.metadata["environment"]["action_steps"],
                    "initial_energy_kwh": initial_energy, "terminal_energy_kwh": terminal_energy,
                    "energy_change_kwh": energy_change,
                    "reference_value_usd_per_kwh_dc": ref_value,
                    "inventory_adjustment_usd": adj_data["inventory_adjustment_usd"],
                    "inventory_adjusted_net_operating_cost_usd": adj_data["inventory_adjusted_net_operating_cost_usd"],
                    "battery_efficiency_loss_kwh": sum(r.battery.efficiency_loss_kwh for r in results),
                    "peak_grid_import_kw": max(r.grid.import_power_kw for r in results),
                    "behavior": {
                        "requested_charge_fraction": float(np.mean(requested < -1e-6)),
                        "requested_discharge_fraction": float(np.mean(requested > 1e-6)),
                        "requested_idle_fraction": float(np.mean(np.abs(requested) <= 1e-6)),
                        "executed_charge_fraction": float(np.mean(executed < -1e-6)),
                        "executed_discharge_fraction": float(np.mean(executed > 1e-6)),
                        "mean_absolute_requested_kw": float(np.mean(np.abs(requested))),
                        "mean_absolute_executed_kw": float(np.mean(np.abs(executed))),
                        "mean_absolute_action_adjustment_kw": float(np.mean(np.abs(requested - executed))),
                        "battery_adjusted_intervals": int(np.sum(np.abs(requested - executed) > 1e-6)),
                        "all_adjusted_intervals": sum(bool(s.adjustments) for s in environment.step_metrics),
                        "post_step_soc_min_fraction": float(np.mean(socs <= config.battery.soc_min + 1e-9)),
                        "post_step_soc_max_fraction": float(np.mean(socs >= config.battery.soc_max - 1e-9)),
                    }})
    summary["selection_eligible"] = eligible(summary)


def _plots(output, frames, records):
    valid = [r for r in records if r.get("status") == "complete"]
    figure, axis = plt.subplots(figsize=(9, 5), layout="constrained")
    markers = ("o", "s", "^", "D", "v", "P")
    for index, record in enumerate(valid):
        axis.scatter(record["cumulative_metrics"]["net_operating_cost"], record["terminal_energy_kwh"],
                     label=record["controller"], marker=markers[index % len(markers)],
                     s=65, facecolors="none", edgecolors=f"C{index % 10}", linewidths=1.5)
    if valid:
        axis.legend(loc="best")
    currency = valid[0]["currency_code"] if valid else "declared currency"
    axis.set(xlabel=f"Net operating cost ({currency})", ylabel="Terminal stored energy (kWh)",
             title="Pilot validation: cost and retained battery energy")
    figure.savefig(output / "cost_terminal_energy.png", dpi=150)
    plt.close(figure)
    figure, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, layout="constrained")
    for name, frame in frames.items():
        hours = np.arange(len(frame))
        axes[0].plot(hours, frame["battery_soc_after"], label=name, linewidth=1)
        axes[1].plot(hours, frame["battery_actual_kw"], label=name, linewidth=.7, alpha=.8)
        axes[2].plot(hours, frame["cumulative_net_operating_cost"], label=name)
    for axis, label in zip(axes, ("SOC", "Executed battery power (kW)", "Cumulative operating cost")):
        axis.set_ylabel(label)
        axis.grid(alpha=.2)
    axes[0].legend(ncol=3)
    axes[-1].set_xlabel("Scored hour in fixed validation episode")
    figure.savefig(output / "validation_trajectories.png", dpi=150)
    plt.close(figure)


def validate_policies(*, split_root, assets, rating_origin, output, checkpoints,
                      expected_rows=1314, initial_soc=.5, expected_timesteps=4096,
                      simulation_config=ROOT / "configs/simulation.yaml",
                      costs_config=ROOT / "configs/simulation_costs.yaml",
                      baseline_config=ROOT / "configs/baselines.yaml"):
    """Evaluate checkpoint records and all four baselines on one immutable replay."""
    started = time.perf_counter()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Validation output already exists: {output}")
    data, provenance = load_verified_partition(split_root, "validation")
    config, costs = load_simulation_config(simulation_config), load_cost_config(costs_config)
    if len(data) != expected_rows or config.battery.initial_soc != initial_soc:
        raise ValueError("Validation rows or initial SOC disagree with the approved fixed episode.")
    renewables, demand = _components(data, assets, rating_origin)
    identifiers = [r["id"] for r in checkpoints]
    if (len(set(identifiers)) != len(identifiers) or any(
            not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", name) or name in available_controllers()
            for name in identifiers)):
        raise ValueError("Checkpoint identifiers must be unique safe names distinct from baselines.")
    output.mkdir(parents=True)
    records, frames = [], {}
    report = {"schema_version": "1.0", "stage": "pilot", "status": "running",
              "data": provenance, "simulation": asdict(config), "costs": asdict(costs),
              "asset_sha256": sha256(assets), "baseline_config_sha256": sha256(baseline_config),
              "source": source_snapshot(output), "controllers": records,
              "common_timestamps": {"history": data[0].timestamp_utc,
                                    "first_scored": data[1].timestamp_utc,
                                    "end_exclusive": provenance["boundaries"]["end_exclusive_utc"]}}
    jobs = [{"id": name, "kind": "baseline"} for name in available_controllers()]
    jobs += [{**record, "kind": "ppo"} for record in checkpoints]
    previous_threads = torch.get_num_threads()
    saved_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    try:
        torch.set_num_threads(1)
        for job in jobs:
            run_started = time.perf_counter()
            environment = MicrogridEnvironment(data, renewables, demand, config, costs)
            environment.reset()
            record = {"controller": job["id"], "kind": job["kind"], "status": "failed",
                      "configuration": job.get("configuration"), "seed": job.get("seed")}
            try:
                if job["kind"] == "baseline":
                    controller = _controller(job["id"], baseline_config=baseline_config,
                                             cost_config=costs, currency_code=config.grid.currency_code)
                else:
                    bundle = Path(job["path"])
                    manifest = verify_checkpoint(bundle)
                    contract = manifest["contract"]
                    if (manifest["timesteps"] != expected_timesteps
                            or contract["observation_schema"] != SCHEMA
                            or contract["simulation"] != asdict(config) or contract["costs"] != asdict(costs)
                            or contract["asset_config_sha256"] != sha256(assets)
                            or contract["rating_origin"] != rating_origin
                            or contract["settings"]["seed"] != job["seed"]
                            or contract["data"]["split_manifest_sha256"] != provenance["split_manifest_sha256"]):
                        raise ValueError("Checkpoint contract disagrees with the common validation scenario.")
                    model = PPO.load(bundle / "model.zip", device="cpu")
                    before = policy_state_hash(model)
                    controller = FrozenPPOController(model, config)
                    record.update({"checkpoint": str(bundle.resolve()),
                                   "training_settings": contract["settings"],
                                   "checkpoint_manifest_sha256": sha256(bundle / "manifest.json"),
                                   "checkpoint_timesteps": manifest["timesteps"]})
                summary, hourly = _run_controller(
                    job["id"],
                    controller,
                    environment,
                    True,
                    record_observation_context=job["kind"] == "ppo",
                )
                total_cost = sum(row["net_operating_cost"] for row in hourly)
                if abs(total_cost - summary["cumulative_metrics"]["net_operating_cost"]) > 1e-6:
                    raise ValueError("Hourly accounting does not reconcile with committed totals.")
                _enhance(summary, environment, config, costs=costs, baseline_config=baseline_config)
                record.update(summary)
                if job["kind"] == "ppo":
                    record["frozen_state_verified"] = before == policy_state_hash(model)
                    if not record["frozen_state_verified"]:
                        raise ValueError("Policy or optimizer state changed during validation.")
                folder = output / job["id"]
                folder.mkdir()
                frame = pd.DataFrame(hourly)
                expected = [row.timestamp_utc for row in tuple(data)[1:]]
                if frame["timestamp_utc"].tolist() != expected:
                    raise ValueError("Scored timestamps disagree with fixed validation episode.")
                frame.to_csv(folder / "hourly.csv", index=False)
                frames[job["id"]] = frame
            except BaseException as exc:
                record.update({"status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", "selection_eligible": False,
                               "exception": type(exc).__name__, "message": str(exc),
                               "traceback": traceback.format_exc(),
                               "committed_steps": len(environment.step_metrics),
                               "partial_metrics": asdict(environment.cumulative_metrics) if environment.step_metrics else {}})
                if not isinstance(exc, Exception):
                    raise
            finally:
                record["validation_runtime_seconds"] = time.perf_counter() - run_started
                records.append(record)
                write_json(output / "validation.json", report)
        report["status"] = "completed" if all(r["status"] == "complete" for r in records) else "completed_with_failures"
        candidates = [record for record in records if eligible(record)]
        report["descriptive_cost_energy_frontier"] = [
            record["controller"] for record in candidates
            if not any(dominates(other, record) for other in candidates if other is not record)]
        report["research_winner"] = None
        flat = []
        for record in records:
            item = {key: record.get(key) for key in ("controller", "kind", "status", "configuration", "seed",
                    "terminal_soc", "terminal_energy_kwh", "energy_change_kwh", "validation_runtime_seconds",
                    "physical_violation_count", "adjustment_count", "steps", "initial_energy_kwh",
                    "initial_soc", "minimum_soc", "maximum_soc", "battery_efficiency_loss_kwh",
                    "reference_value_usd_per_kwh_dc", "inventory_adjustment_usd",
                    "inventory_adjusted_net_operating_cost_usd",
                    "selection_eligible", "peak_grid_import_kw")}
            item.update(record.get("cumulative_metrics", {}))
            item.update(record.get("behavior", {}))
            flat.append(item)
        pd.DataFrame(flat).to_csv(output / "comparison.csv", index=False)
        _plots(output, frames, records)
    except BaseException as exc:
        report["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        report["diagnostics"] = {"exception": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        random.setstate(saved_rng[0])
        np.random.set_state(saved_rng[1])
        torch.set_rng_state(saved_rng[2])
        torch.set_num_threads(previous_threads)
        report["runtime_seconds"] = time.perf_counter() - started
        report["artifacts"] = {p.relative_to(output).as_posix(): sha256(p) for p in output.rglob("*")
                               if p.is_file() and p.name != "validation.json"}
        write_json(output / "validation.json", report)
    return report
