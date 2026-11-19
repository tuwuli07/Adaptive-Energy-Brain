"""Synthetic bounded telemetry verification; no real-data training in tests."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

pytest.importorskip("stable_baselines3")
from stable_baselines3 import PPO

from test_rl_episodes import training_inputs, forbid_heldout_reads
from src.evaluation.ppo_validation import policy_state_hash
from src.rl.diagnostic_experiment import load_diagnostic_config, paired_preflight, run_diagnostic
from src.rl.diagnostics import PPODiagnostics
from src.rl.training import ROOT, load_training_config, train_ppo, verify_checkpoint


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def small_config(tmp_path, reward_scale=None):
    settings = load_training_config()
    settings.update(version=2, purpose="gamma_diagnostic", total_timesteps=32, n_steps=16,
                    batch_size=8, n_epochs=1, policy_layers=[16, 16], episode_hours=8,
                    checkpoint_interval=16)
    if reward_scale is not None:
        settings["reward_scale"] = reward_scale
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(settings))
    return path, settings


def test_telemetry_preserves_weights_rng_rewards_and_captures_exact_hooks(training_inputs, tmp_path, monkeypatch):
    scale = .0025
    path, settings = small_config(tmp_path, reward_scale=scale)
    accesses = forbid_heldout_reads(monkeypatch, training_inputs["split_root"])
    plain, observed = tmp_path / "plain", tmp_path / "observed"
    train_ppo(**training_inputs, output=plain, training_config=path)
    result = train_ppo(**training_inputs, output=observed, training_config=path, diagnostics=True)
    assert result["status"] == "completed"
    assert any(p.startswith("train/") for p in accesses)
    for steps in (0, 16, 32):
        first = plain / "checkpoints" / f"step_{steps:09d}"
        second = observed / "checkpoints" / f"step_{steps:09d}"
        assert verify_checkpoint(first)["optimizer_steps"] == verify_checkpoint(second)["optimizer_steps"]
        assert policy_state_hash(PPO.load(first / "model.zip")) == policy_state_hash(PPO.load(second / "model.zip"))
        rng1 = torch.load(first / "state.pt", weights_only=False)
        rng2 = torch.load(second / "state.pt", weights_only=False)
        assert torch.equal(rng1["torch"], rng2["torch"])
        assert rng1["sampler"] == rng2["sampler"] and rng1["python"] == rng2["python"]
        np.testing.assert_array_equal(rng1["numpy"][1], rng2["numpy"][1])
    rows = read_lines(observed / "telemetry/rollouts.jsonl")
    gradients = read_lines(observed / "telemetry/gradients.jsonl")
    updates = read_lines(observed / "telemetry/parameter_updates.jsonl")
    assert len(rows) == 32 and len(gradients) == 4
    assert len(updates) == len(gradients)
    assert len(read_lines(observed / "telemetry/transitions.jsonl")) == 32
    assert sum(row["truncated"] for row in rows) == 4
    for row in rows:
        assert row["bounded_action"] == np.clip(row["sampled_action"], -1, 1)
        assert row["requested_kw"] == pytest.approx(150 * row["bounded_action"])
        assert row["raw_reward_float32"] == float(np.float32(-row["net_operating_cost"]))
        assert row["scaled_reward_float32"] == pytest.approx(
            float(np.float32(row["raw_reward_float32"] * scale)), abs=1e-8
        )
        assert row["value_target"] == pytest.approx(row["advantage"] + row["value_prediction"], abs=1e-4)
        assert row["dollar_equivalent_value_prediction"] == pytest.approx(
            row["value_prediction"] / scale
        )
        assert row["dollar_equivalent_value_target"] == pytest.approx(
            row["value_target"] / scale
        )
        expected = row["scaled_reward_float32"]
        if row["truncated"]:
            assert row["terminal_bootstrap_value"] is not None
            assert row["dollar_equivalent_terminal_bootstrap_value"] == pytest.approx(
                row["terminal_bootstrap_value"] / scale
            )
            expected += .99 * row["terminal_bootstrap_value"]
            m = row["episode_metrics"]
            end_soc = (250 + .95*m["battery_charge_energy_kwh"] - m["battery_discharge_energy_kwh"]/.95)/500
            assert row["soc_after"] == pytest.approx(end_soc, abs=1e-7)
        else:
            assert row["terminal_bootstrap_value"] is None
        assert row["buffer_reward"] == pytest.approx(expected, abs=1e-4)
        assert row["dollar_equivalent_buffer_reward"] == pytest.approx(
            row["buffer_reward"] / scale
        )
    for g in gradients:
        assert g["global_pre_norm"]**2 == pytest.approx(g["actor_pre_norm"]**2 + g["critic_pre_norm"]**2)
        assert g["global_post_norm"] <= .50001
        clip_scale = min(1, .5 / (g["global_pre_norm"] + 1e-6))
        assert g["observed_global_scale"] == pytest.approx(clip_scale, rel=1e-5)
        assert g["actor_post_norm"] == pytest.approx(g["actor_pre_norm"]*clip_scale, rel=1e-5)
    for update in updates:
        assert update["global_update_norm"]**2 == pytest.approx(
            update["actor_update_norm"]**2 + update["critic_update_norm"]**2,
            rel=1e-5,
        )
        assert all(np.isfinite(value) for key, value in update.items() if key.endswith("_norm"))
    assert read_lines(plain / "episodes.jsonl") == read_lines(observed / "episodes.jsonl")


def test_nonfinite_diagnostic_refused_and_hooks_removed(tmp_path):
    import gymnasium as gym
    model = PPO("MlpPolicy", gym.make("Pendulum-v1"), n_steps=4, batch_size=2, seed=0)
    telemetry = PPODiagnostics(tmp_path / "telemetry")
    telemetry.attach(model)
    assert telemetry.handles
    with pytest.raises(ValueError):
        telemetry.emit("invalid", {"value": float("nan")})
    telemetry.close()
    assert not telemetry.handles
    assert all(not p._backward_hooks for p in model.policy.parameters())
    assert not model.policy.optimizer._optimizer_step_pre_hooks
    assert not model.policy.optimizer._optimizer_step_post_hooks
    model.env.close()


def test_diagnostic_failure_keeps_checkpoint_and_transition_evidence(training_inputs, tmp_path, monkeypatch):
    path, _ = small_config(tmp_path)
    original = PPODiagnostics.emit
    def fail_gradient(self, name, record):
        if name == "gradients":
            raise OSError("injected telemetry failure")
        return original(self, name, record)
    monkeypatch.setattr(PPODiagnostics, "emit", fail_gradient)
    output = tmp_path / "failure"
    with pytest.raises(OSError, match="telemetry failure"):
        train_ppo(**training_inputs, output=output, training_config=path, diagnostics=True)
    record = json.loads((output / "experiment.json").read_text())
    assert record["status"] == "failed" and record["progress"]["checkpoint_timesteps"] == 0
    assert verify_checkpoint(output / "checkpoints/step_000000000")["timesteps"] == 0
    assert len(read_lines(output / "telemetry/transitions.jsonl")) == 16


def test_diagnostic_pairing_budget_and_end_to_end(training_inputs, tmp_path, monkeypatch):
    path, base = small_config(tmp_path)
    # The base loader accepts diagnostic v2; runner replaces gamma and purpose.
    protocol = load_diagnostic_config(ROOT / "configs/ppo_gamma_diagnostic.yaml")
    protocol.update(total_timesteps=32, episode_hours=8, checkpoint_interval=16,
                    validation_checkpoints=[16, 32], validation_rows=7)
    experiment = tmp_path / "protocol.yaml"
    experiment.write_text(yaml.safe_dump(protocol))
    inputs = {**training_inputs, "simulation_config": ROOT / "configs/simulation.yaml",
              "costs_config": ROOT / "configs/simulation_costs.yaml"}
    configs = {"control": {**base, "gamma": .99}, "treatment": {**base, "gamma": .999}}
    preflight = paired_preflight(configs, inputs)
    assert preflight["verified"] and len(preflight["windows"]) == 5
    bad = {**configs, "treatment": {**configs["treatment"], "learning_rate": .0001}}
    with pytest.raises(ValueError, match="only in gamma"):
        paired_preflight(bad, inputs)
    # Deny held-out test data even while the runner alternates train/validation.
    import builtins, io
    for module in (builtins, io):
        original = module.open
        def guarded(file, *args, _original=original, **kwargs):
            if isinstance(file, (str, Path)):
                assert (training_inputs["split_root"] / "test").resolve() not in Path(file).resolve().parents
            return _original(file, *args, **kwargs)
        monkeypatch.setattr(module, "open", guarded)
    output = tmp_path / "diagnostic"
    result = run_diagnostic(**training_inputs, output=output, base_training_config=path, experiment_config=experiment)
    assert result["status"] == "completed" and result["research_winner"] is None
    assert all(r["actual_pairing_verified"] for r in result["runs"])
    assert [v["timesteps"] for v in result["validations"]] == [16, 32]
    for step in (16, 32):
        report = json.loads((output / f"validation_{step}/validation.json").read_text())
        assert len(report["controllers"]) == 6
        assert all(r["frozen_state_verified"] for r in report["controllers"] if r["kind"] == "ppo")
    with pytest.raises(FileExistsError):
        run_diagnostic(**training_inputs, output=output, base_training_config=path, experiment_config=experiment)
    protocol["total_timesteps"] = 32768
    experiment.write_text(yaml.safe_dump(protocol))
    with pytest.raises(ValueError):
        load_diagnostic_config(experiment)
