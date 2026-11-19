"""Reward-scale protocol pairing and train-only preflight."""

import yaml

from test_rl_episodes import forbid_heldout_reads, training_inputs
from src.rl.diagnostic_experiment import paired_preflight
from src.rl.reward_scale_experiment import ROOT, load_reward_scale_config
from src.rl.training import load_training_config


def test_reward_scale_protocol_matches_approval():
    settings = load_reward_scale_config()
    assert settings["gamma"] == .99
    assert settings["reward_scales"] == {"control": 1.0, "treatment": .0025}
    assert settings["seed"] == 0
    assert settings["total_timesteps"] == 8192
    assert settings["validation_checkpoints"] == [4096, 8192]


def test_reward_scale_preflight_pairs_policy_and_train_windows(training_inputs, monkeypatch):
    forbid_heldout_reads(monkeypatch, training_inputs["split_root"])
    base = load_training_config()
    base.update(
        version=2,
        purpose="reward_scale_diagnostic",
        total_timesteps=32,
        episode_hours=8,
        n_steps=16,
        batch_size=8,
        n_epochs=1,
        policy_layers=[16, 16],
        checkpoint_interval=16,
        reward_scale=1.0,
    )
    configs = {
        "control": dict(base),
        "treatment": {**base, "reward_scale": .0025},
    }
    inputs = {
        **training_inputs,
        "simulation_config": ROOT / "configs/simulation.yaml",
        "costs_config": ROOT / "configs/simulation_costs.yaml",
    }

    result = paired_preflight(configs, inputs, differing_setting="reward_scale")

    assert result["verified"]
    assert result["settings_differ_only_in"] == ["reward_scale"]
    assert len(result["windows"]) == 5
    assert result["test_partition_access"] is False
