"""Small CPU PPO updates, checkpoint integrity, and recoverable failures."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

pytest.importorskip("stable_baselines3")
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from test_rl_episodes import training_inputs, episode_environment, forbid_heldout_reads
from src.rl import training


@pytest.fixture
def tiny_config(tmp_path):
    settings = training.load_training_config()
    settings.update(episode_hours=8, total_timesteps=32, n_steps=16, batch_size=8,
                    n_epochs=1, policy_layers=[16, 16])
    path = tmp_path / "tiny.yaml"
    path.write_text(yaml.safe_dump(settings))
    return path


def run(inputs, config, output, **kwargs):
    return training.train_ppo(**inputs, training_config=config, output=output, **kwargs)


def test_tiny_training_reload_and_continuation(training_inputs, tiny_config, tmp_path, monkeypatch):
    forbid_heldout_reads(monkeypatch, training_inputs["split_root"])
    output = tmp_path / "first"
    result = run(training_inputs, tiny_config, output)
    assert result["status"] == "completed"
    assert result["progress"]["collected_timesteps"] == 32
    assert result["progress"]["completed_rollout_updates"] == 2
    assert result["progress"]["optimizer_steps"] == 4
    assert all(result["verification"].values())
    assert isinstance(result["source"]["dirty_worktree"], bool)
    assert (output / "logs/progress.csv").exists()
    assert (output / "episodes.jsonl").exists()
    assert (output / "source_snapshot.zip").exists()
    training.validate_metadata(result)
    checkpoint = output / training.read_json(output / "latest_checkpoint.json")["path"]
    manifest = training.verify_checkpoint(checkpoint)
    assert manifest["timesteps"] == 32
    state = torch.load(checkpoint / "state.pt", weights_only=False)
    assert {"python", "numpy", "torch", "sampler"} == set(state)
    a, b = PPO.load(checkpoint / "model.zip"), PPO.load(checkpoint / "model.zip")
    env = episode_environment(training_inputs)
    obs, _ = env.reset(seed=9)
    np.testing.assert_array_equal(a.predict(obs, deterministic=True)[0], b.predict(obs, deterministic=True)[0])
    continued = run(training_inputs, tiny_config, tmp_path / "continued", resume=checkpoint)
    assert continued["progress"]["checkpoint_timesteps"] == 64
    assert continued["progress"]["completed_rollout_updates"] == 4
    assert continued["lineage"]["restored_optimizer_steps"] == 4
    assert continued["progress"]["optimizer_steps"] == 8
    assert continued["lineage"]["fresh_episode_continuation"]
    records = [json.loads(line) for line in (tmp_path / "continued/episodes.jsonl").read_text().splitlines()]
    probe = episode_environment(training_inputs)
    probe.restore_sampler(state["sampler"])
    probe.reset()
    assert records[0]["start_index"] == probe.current_window["start_index"]
    assert records[0]["child_seed"] == probe.current_window["child_seed"]
    assert training.optimizer_steps(PPO.load(checkpoint / "model.zip")) == 4
    initial_continuation = tmp_path / "continued/checkpoints/step_000000032"
    resumed_model = PPO.load(initial_continuation / "model.zip")
    for key, parameter in a.policy.state_dict().items():
        torch.testing.assert_close(parameter, resumed_model.policy.state_dict()[key], rtol=0, atol=0)
    original_optimizer = a.policy.optimizer.state_dict()
    resumed_optimizer = resumed_model.policy.optimizer.state_dict()
    assert original_optimizer["param_groups"] == resumed_optimizer["param_groups"]
    for key, parameter_state in original_optimizer["state"].items():
        for name, value in parameter_state.items():
            torch.testing.assert_close(value, resumed_optimizer["state"][key][name], rtol=0, atol=0)
    resumed_state = torch.load(initial_continuation / "state.pt", weights_only=False)
    assert resumed_state["python"] == state["python"]
    np.testing.assert_array_equal(resumed_state["numpy"][1], state["numpy"][1])
    torch.testing.assert_close(resumed_state["torch"], state["torch"], rtol=0, atol=0)
    assert resumed_state["sampler"] == state["sampler"]
    # Identical seeds and inputs reproduce the tiny training result locally.
    repeated = tmp_path / "repeated"
    run(training_inputs, tiny_config, repeated)
    repeated_model = PPO.load(repeated / "checkpoints/step_000000032/model.zip")
    for key, parameter in a.policy.state_dict().items():
        torch.testing.assert_close(parameter, repeated_model.policy.state_dict()[key], rtol=0, atol=0)
    (checkpoint / "state.pt").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="hash mismatch"):
        training.verify_checkpoint(checkpoint)
    with pytest.raises(FileExistsError):
        run(training_inputs, tiny_config, output)


def test_vector_truncation_preserves_terminal_observation(training_inputs):
    vector = DummyVecEnv([lambda: episode_environment(training_inputs)])
    vector.seed(3)
    vector.reset()
    for _ in range(8):
        obs, reward, done, info = vector.step(np.zeros((1, 1), dtype=np.float32))
    assert done[0] and info[0]["TimeLimit.truncated"]
    assert info[0]["terminal_observation"].shape == (15,)
    assert info[0]["decision_timestamp_utc"] != vector.reset_infos[0]["decision_timestamp_utc"]
    vector.close()


@pytest.mark.parametrize("exception,status", [(RuntimeError("injected update failure"), "failed"),
                                              (KeyboardInterrupt(), "interrupted")])
def test_last_valid_checkpoint_survives_training_failure(training_inputs, tiny_config, tmp_path,
                                                        monkeypatch, exception, status):
    original = PPO.learn
    calls = 0
    def failing_learn(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise exception
        return original(self, *args, **kwargs)
    monkeypatch.setattr(PPO, "learn", failing_learn)
    output = tmp_path / status
    with pytest.raises(type(exception)):
        run(training_inputs, tiny_config, output)
    result = training.read_json(output / "experiment.json")
    assert result["status"] == status
    checkpoint = output / training.read_json(output / "latest_checkpoint.json")["path"]
    assert training.verify_checkpoint(checkpoint)["timesteps"] == 16
    assert result["diagnostics"]["exception"] == type(exception).__name__


def test_failed_checkpoint_publication_preserves_previous_bundle(training_inputs, tiny_config, tmp_path, monkeypatch):
    original = PPO.save
    calls = 0
    def fail(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected checkpoint write failure")
        return original(self, *args, **kwargs)
    monkeypatch.setattr(PPO, "save", fail)
    output = tmp_path / "publication_failure"
    with pytest.raises(OSError, match="checkpoint write"):
        run(training_inputs, tiny_config, output)
    checkpoint = output / training.read_json(output / "latest_checkpoint.json")["path"]
    assert training.verify_checkpoint(checkpoint)["timesteps"] == 0
    assert training.read_json(output / "experiment.json")["status"] == "failed"


def test_config_and_metadata_validation(tiny_config):
    settings = yaml.safe_load(tiny_config.read_text())
    settings["batch_size"] = 7
    tiny_config.write_text(yaml.safe_dump(settings))
    with pytest.raises(ValueError, match="Batch"):
        training.load_training_config(tiny_config)
    with pytest.raises(ValueError, match="metadata"):
        training.validate_metadata({})


@pytest.mark.parametrize("scale", [0, -0.1, float("nan"), float("inf"), True])
def test_reward_scale_must_be_finite_and_positive(tiny_config, scale):
    settings = yaml.safe_load(tiny_config.read_text())
    settings["reward_scale"] = scale
    tiny_config.write_text(yaml.safe_dump(settings))
    with pytest.raises(ValueError, match="reward_scale"):
        training.load_training_config(tiny_config)


def test_partial_optimizer_update_does_not_publish_checkpoint(training_inputs, tiny_config, tmp_path, monkeypatch):
    original = PPO.train
    calls = 0
    def interrupted_update(self):
        nonlocal calls
        calls += 1
        original(self)
        if calls == 2:
            raise KeyboardInterrupt()
    monkeypatch.setattr(PPO, "train", interrupted_update)
    output = tmp_path / "partial_update"
    with pytest.raises(KeyboardInterrupt):
        run(training_inputs, tiny_config, output)
    latest = output / training.read_json(output / "latest_checkpoint.json")["path"]
    manifest = training.verify_checkpoint(latest)
    assert manifest["timesteps"] == 16 and manifest["optimizer_steps"] == 2
    result = training.read_json(output / "experiment.json")
    assert result["status"] == "interrupted"
    assert result["progress"]["collected_timesteps"] == 32
    assert result["progress"]["checkpoint_timesteps"] == 16
