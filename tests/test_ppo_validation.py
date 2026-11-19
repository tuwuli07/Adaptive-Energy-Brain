"""Frozen validation, shared physical replay, and seed-based selection rules."""

import builtins
from copy import deepcopy
import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

pytest.importorskip("stable_baselines3")
from stable_baselines3 import PPO

from test_rl_episodes import training_inputs
from src.baselines import available_controllers
from src.evaluation.baseline_runner import _controller, _run_controller
from src.evaluation.ppo_validation import (
    FrozenPPOController, aggregate_records, dominates, policy_state_hash, validate_policies,
)
from src.rl.environment import MicrogridGymEnvironment
from src.rl.episodes import load_verified_partition
from src.rl.experiments import load_experiment_config, run_pilot
from src.rl.training import load_training_config, train_ppo, verify_checkpoint, ROOT
from src.simulation.config import load_simulation_config
from src.simulation.environment import MicrogridEnvironment
from src.simulation.metrics import load_cost_config
from scripts.run_simulation import _components


def access_guard(monkeypatch, root, allowed):
    opened = []
    original_io, original_builtin = io.open, builtins.open
    def guard(original):
        def open_file(file, *args, **kwargs):
            if isinstance(file, (str, Path)):
                path = Path(file).resolve()
                if root.resolve() in path.parents:
                    relative = path.relative_to(root.resolve())
                    assert relative.parts[0] not in ({"train", "validation", "test"} - {allowed}), relative
                    opened.append(relative.as_posix())
            return original(file, *args, **kwargs)
        return open_file
    monkeypatch.setattr(io, "open", guard(original_io))
    monkeypatch.setattr(builtins, "open", guard(original_builtin))
    return opened


@pytest.fixture
def pilot_fixture(training_inputs, tmp_path):
    config = load_training_config()
    config.update(version=2, purpose="validation_pilot", total_timesteps=32, n_steps=16,
                  batch_size=8, n_epochs=1, policy_layers=[16, 16], episode_hours=8,
                  checkpoint_interval=64)
    path = tmp_path / "training.yaml"
    path.write_text(yaml.safe_dump(config))
    output = tmp_path / "training"
    result = train_ppo(**training_inputs, output=output, training_config=path)
    return training_inputs, output / "checkpoints/step_000000032", result


def test_validation_reads_only_validation_and_baselines_match(pilot_fixture, tmp_path, monkeypatch):
    inputs, checkpoint, training = pilot_fixture
    assert training["schema_version"] == "1.1" and training["phase"] == "6.4"
    opened = access_guard(monkeypatch, inputs["split_root"], "validation")
    jobs = [{"id": "PPO_A", "configuration": "A", "seed": 0, "path": str(checkpoint)}]
    output = tmp_path / "validation"
    report = validate_policies(**inputs, output=output, checkpoints=jobs,
                               expected_rows=7, expected_timesteps=32)
    assert report["status"] == "completed"
    assert any(name.startswith("validation/") for name in opened)
    assert len(report["controllers"]) == 5
    assert report["data"]["partition"] == "validation"
    timestamps = []
    data, _ = load_verified_partition(inputs["split_root"], "validation")
    config, costs = load_simulation_config(), load_cost_config()
    renewable, demand = _components(data, inputs["assets"], inputs["rating_origin"])
    for record in report["controllers"]:
        assert record["steps"] == 6 and record["initial_soc"] == .5
        assert record["initial_energy_kwh"] == 250
        assert record["terminal_energy_kwh"] == pytest.approx(record["terminal_soc"] * 500)
        assert record["energy_change_kwh"] == pytest.approx(record["terminal_energy_kwh"] - 250)
        assert record["selection_eligible"]
        import pandas as pd
        timestamps.append(pd.read_csv(output / record["controller"] / "hourly.csv")["timestamp_utc"].tolist())
        if record["kind"] == "baseline":
            core = MicrogridEnvironment(data, renewable, demand, config, costs)
            controller = _controller(record["controller"], baseline_config=ROOT / "configs/baselines.yaml",
                                     cost_config=costs, currency_code="USD")
            expected, _ = _run_controller(record["controller"], controller, core, True)
            assert record["cumulative_metrics"] == expected["cumulative_metrics"]
            assert record["terminal_soc"] == expected["terminal_soc"]
        else:
            assert record["frozen_state_verified"]
            ppo_hourly = pd.read_csv(output / record["controller"] / "hourly.csv")
            assert "observed_import_price_per_kwh" in ppo_hourly
    assert all(values == timestamps[0] for values in timestamps)
    assert (output / "cost_terminal_energy.png").is_file()
    assert (output / "validation_trajectories.png").is_file()
    manifest = verify_checkpoint(checkpoint)
    assert manifest["timesteps"] == 32
    assert not (checkpoint.parent / "step_000000016").exists()
    assert (checkpoint.parent / "step_000000000").exists()


def test_policy_mapping_and_frozen_inference_match_gym(training_inputs):
    data, _ = load_verified_partition(training_inputs["split_root"], "validation")
    config, costs = load_simulation_config(), load_cost_config()
    renewable, demand = _components(data, training_inputs["assets"], training_inputs["rating_origin"])
    gym = MicrogridGymEnvironment(data, renewable, demand, config, costs)
    model = PPO("MlpPolicy", gym, n_steps=4, batch_size=2, device="cpu", seed=0)
    controller = FrozenPPOController(model, config)
    core = MicrogridEnvironment(data, renewable, demand, config, costs)
    native = core.reset()
    numeric, _ = gym.reset()
    before = policy_state_hash(model)
    while not core.terminated:
        action = model.predict(numeric, deterministic=True)[0]
        first = controller.select_action(native)
        assert first == controller.select_action(native)
        expected = core.step(first)
        numeric, _, _, _, info = gym.step(action)
        assert info["requested_power_kw"] == first.requested_power_kw
        assert info["executed_power_kw"] == expected.dispatch.actual_battery_power_kw
        native = expected.observation
        assert core.cumulative_metrics == gym._core.cumulative_metrics
    assert policy_state_hash(model) == before


def test_validation_failure_retained_and_wrong_window_rejected(training_inputs, tmp_path):
    jobs = [{"id": "PPO_bad", "configuration": "A", "seed": 0, "path": str(tmp_path / "missing")}]
    report = validate_policies(**training_inputs, output=tmp_path / "failed", checkpoints=jobs,
                               expected_rows=7, expected_timesteps=32)
    assert report["status"] == "completed_with_failures"
    assert report["controllers"][-1]["status"] == "failed"
    assert "traceback" in report["controllers"][-1]
    assert len(report["controllers"]) == 5
    with pytest.raises(ValueError, match="rows"):
        validate_policies(**training_inputs, output=tmp_path / "wrong", checkpoints=[], expected_rows=8)
    with pytest.raises(ValueError, match="Only train or validation"):
        load_verified_partition(training_inputs["split_root"], "test")


def metric_record(config, seed, cost, energy, status="complete", steps=32768):
    return {"kind": "ppo", "configuration": config, "seed": seed, "status": status,
            "physical_violation_count": 0, "cumulative_metrics": {"net_operating_cost": cost,
            "unserved_demand_energy_kwh": 0}, "steps": 6, "expected_steps": 6,
            "maximum_absolute_power_balance_residual_kw": 0,
            "maximum_absolute_energy_balance_residual_kwh": 0,
            "terminal_energy_kwh": energy, "checkpoint_timesteps": steps}


def test_pareto_tradeoffs_failures_fixed_checkpoint_and_seed_uncertainty():
    records = [metric_record("A", seed, 10 + seed, 200) for seed in range(5)]
    records += [metric_record("B", seed, 9 + seed, 190) for seed in range(5)]
    report = aggregate_records(records, range(5), 32768)
    assert report["pareto_set"] == ["A", "B"] and report["selected_configuration"] is None
    assert report["configurations"]["A"]["net_operating_cost"]["std_across_seeds"] == pytest.approx(np.std(range(10, 15), ddof=1))
    for record in records[5:]:
        record["terminal_energy_kwh"] = 210
    assert aggregate_records(records, range(5), 32768)["selected_configuration"] == "B"
    records[-1]["status"] = "failed"
    failed = aggregate_records(records, range(5), 32768)
    assert failed["configurations"]["B"]["failed_or_ineligible_seeds"] == [4]
    assert not failed["configurations"]["B"]["selection_eligible"]
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate_records(records + [records[0]], range(5), 32768)
    with pytest.raises(ValueError, match="32768"):
        aggregate_records(records, range(5), 16384)
    pilot = aggregate_records([metric_record("A", 0, 10, 200, steps=4096),
                               metric_record("B", 0, 9, 210, steps=4096)], [0], 4096, pilot=True)
    assert pilot["selected_configuration"] is None and pilot["research_winner"] is None
    assert pilot["configurations"]["A"]["net_operating_cost"]["std_across_seeds"] is None


def test_pilot_keeps_failed_runs_and_does_not_retry(training_inputs, tmp_path, monkeypatch):
    config = load_experiment_config()
    config.update(pilot_timesteps=32, episode_hours=8, checkpoint_interval=32, validation_rows=7)
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(config))
    calls = []
    def fail(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("synthetic training failure")
    monkeypatch.setattr("src.rl.experiments.train_ppo", fail)
    report = run_pilot(**training_inputs, output=tmp_path / "pilot", experiment_config=path)
    assert len(calls) == 2 and len(report["runs"]) == 2
    assert report["status"] == "completed_with_failures"
    assert all(r["status"] == "failed" for r in report["runs"])
    assert not report["planned_study_executed"]
    assert set(report["screening"]["configurations"]) == {"A", "B"}


def test_small_complete_pilot_records_six_controllers_and_both_configs(training_inputs, tmp_path):
    config = load_experiment_config()
    config.update(pilot_timesteps=32, episode_hours=8, checkpoint_interval=32, validation_rows=7)
    path = tmp_path / "pilot.yaml"
    path.write_text(yaml.safe_dump(config))
    base = load_training_config()
    base.update(n_steps=16, batch_size=8, n_epochs=1, policy_layers=[16, 16])
    base_path = tmp_path / "base.yaml"
    base_path.write_text(yaml.safe_dump(base))
    output = tmp_path / "pilot_success"
    report = run_pilot(**training_inputs, output=output, experiment_config=path, base_training_config=base_path)
    assert report["status"] == "completed"
    assert len(report["runs"]) == 2
    assert [record["learning_rate"] for record in report["runs"]] == [3e-4, 1e-4]
    validation = json.loads((output / "validation/validation.json").read_text())
    assert len(validation["controllers"]) == 6
    assert report["screening"]["selected_configuration"] is None
    assert (output / "training_diagnostics.png").exists()
    for relative, expected in report["artifacts"].items():
        from src.rl.episodes import sha256
        assert sha256(output / relative) == expected
    configs = [yaml.safe_load((output / ("training_" + name + ".yaml")).read_text()) for name in ["A", "B"]]
    assert {key for key in configs[0] if configs[0][key] != configs[1][key]} == {"learning_rate"}
