"""Discounting protocol pairing and train-only preflight."""

import yaml

from test_rl_episodes import forbid_heldout_reads, training_inputs
from src.rl.diagnostic_experiment import paired_preflight
from src.rl.discounting_experiment import ROOT, load_discounting_config
from src.rl.training import load_training_config


def test_discounting_protocol_matches_approval():
    settings = load_discounting_config()
    assert settings["gammas"] == {"control": .99, "treatment": .999}
    assert settings["reward_scale"] == .0025
    assert settings["seed"] == 0
    assert settings["total_timesteps"] == 16384
    assert settings["validation_checkpoints"] == [4096, 8192, 16384]


def test_discounting_preflight_pairs_policy_and_train_windows(training_inputs, monkeypatch):
    forbid_heldout_reads(monkeypatch, training_inputs["split_root"])
    base = load_training_config()
    base.update(
        version=2,
        purpose="discounting_diagnostic",
        total_timesteps=32,
        episode_hours=8,
        n_steps=16,
        batch_size=8,
        n_epochs=1,
        policy_layers=[16, 16],
        checkpoint_interval=16,
        reward_scale=.0025,
    )
    configs = {
        "control": {**base, "gamma": .99},
        "treatment": {**base, "gamma": .999},
    }
    inputs = {
        **training_inputs,
        "simulation_config": ROOT / "configs/simulation.yaml",
        "costs_config": ROOT / "configs/simulation_costs.yaml",
    }

    result = paired_preflight(configs, inputs, differing_setting="gamma")

    assert result["verified"]
    assert result["settings_differ_only_in"] == ["gamma"]
    assert len(result["windows"]) == 5
    assert result["test_partition_access"] is False
