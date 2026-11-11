"""Tests for masked training, validation selection, and checkpoint state."""

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from src.data.graph_builder import build_graph
from src.forecasting.dataloader import (
    build_forecasting_train_validation_dataloaders,
    load_forecasting_train_validation_dataloaders,
)
from src.forecasting.models import GNNTransformerForecaster
from src.forecasting.training import (
    EarlyStopping,
    ForecastTrainingError,
    TrainingConfig,
    fit_forecaster,
    load_forecaster_for_evaluation,
    masked_mse_loss,
    seed_everything,
    write_json,
)


NODE_ORDER = ("B01", "B02", "B03", "B04", "B05", "PV01", "W01", "BUS01")
TARGET_NODES = NODE_ORDER[:-1]


def _partition(partition: str, start: str, samples: int = 5):
    lookback, horizon, nodes = 4, 3, len(NODE_ORDER)
    sample_offsets = np.arange(samples)[:, None]
    start_time = np.datetime64(start, "ns")
    input_times = start_time + (sample_offsets + np.arange(lookback)) * np.timedelta64(1, "h")
    target_times = (
        start_time
        + (sample_offsets + lookback + np.arange(horizon)) * np.timedelta64(1, "h")
    )
    end = start_time + np.timedelta64(samples + lookback + horizon - 1, "h")
    boundaries = {
        "start_utc": f"{start}Z",
        "end_exclusive_utc": np.datetime_as_string(end, unit="s") + "Z",
    }
    mask = np.ones((nodes, 1), dtype=bool)
    mask[-1] = False
    inputs = np.arange(samples * lookback * nodes, dtype=np.float32).reshape(
        samples, lookback, nodes, 1
    ) / 100
    targets = np.arange(samples * horizon * nodes, dtype=np.float32).reshape(
        samples, horizon, nodes, 1
    ) / 10
    inputs[:, :, -1] = 0
    targets[:, :, -1] = 0
    return {
        "arrays": {
            "X": inputs,
            "global_X": np.zeros((samples, lookback, 3), dtype=np.float32),
            "y": targets,
            "input_timestamps": input_times.astype("datetime64[ns]"),
            "target_timestamps": target_times.astype("datetime64[ns]"),
            "node_input_mask": mask,
            "target_mask": mask.copy(),
        },
        "graph": build_graph(),
        "metadata": {
            "partition": partition,
            "boundaries": boundaries,
            "config": {
                "lookback_hours": lookback,
                "horizon_hours": horizon,
                "stride_hours": 1,
            },
            "node_order": list(NODE_ORDER),
            "target_unit": "kW",
            "source_metadata": {
                "synthetic_fixture_demand": {
                    "origin": "simulated",
                    "split": {
                        "name": partition,
                        "fitted_on": "train",
                        "boundaries": boundaries,
                    },
                }
            },
        },
    }


@pytest.fixture
def partition_results():
    return {
        "train": _partition("train", "2026-01-01T00:00:00"),
        "validation": _partition("validation", "2026-02-01T00:00:00"),
    }


@pytest.fixture
def loaders(partition_results):
    return build_forecasting_train_validation_dataloaders(
        partition_results, batch_size=2, seed=7
    )


def _model_config():
    return {
        "version": 1,
        "data": {"lookback_hours": 4, "horizon_hours": 3},
        "graph": {"node_order": list(NODE_ORDER)},
        "targets": {"nodes": list(TARGET_NODES), "channels": ["power_kw"]},
        "model": {
            "architecture": "gnn_transformer",
            "spatial_encoder": "graph_mean_message_passing",
            "temporal_encoder": "per_node_transformer",
            "lookback_hours": 4,
            "forecast_horizon_hours": 3,
            "target_scope": "node_level",
            "system_pooling": None,
            "gnn_encoder": {
                "input_dim": 1,
                "hidden_dim": 8,
                "num_layers": 1,
                "dropout": 0.0,
                "activation": "relu",
                "aggregation": "weighted_mean",
                "edge_direction": "source_to_target",
                "self_loop_policy": "reject_explicit_use_self_projection",
                "isolated_node_policy": "self_projection_only",
            },
            "transformer_encoder": {
                "d_model": 8,
                "num_heads": 2,
                "num_layers": 1,
                "dim_feedforward": 16,
                "dropout": 0.0,
                "activation": "relu",
                "max_seq_len": 8,
                "positional_encoding": "sinusoidal",
                "attention_strategy": "bidirectional_within_historical_window",
                "padding_mask_convention": "true_means_padding",
            },
            "forecast_head": {"type": "direct_multi_step", "dropout": 0.0},
        },
    }


def _training_config(**updates):
    values = {
        "batch_size": 2,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "max_epochs": 2,
        "patience": 3,
        "min_delta": 0.0,
        "seed": 23,
        "optimizer": "adamw",
        "loss": "masked_mse",
        "gradient_clip_norm": 1.0,
        "device": "cpu",
        "deterministic": False,
    }
    values.update(updates)
    return TrainingConfig.from_mapping(values)


def _fit(model, loaders, tmp_path, config=None, **kwargs):
    preprocessing_artifacts = kwargs.pop(
        "preprocessing_artifacts",
        {"preprocessing": "fixture/preprocessing.json"},
    )
    return fit_forecaster(
        model,
        train_loader=loaders.train,
        validation_loader=loaders.validation,
        graph=loaders.graph,
        config=config or _training_config(),
        experiment_id="test-training-run",
        checkpoint_path=tmp_path / "best.pt",
        preprocessing_artifacts=preprocessing_artifacts,
        **kwargs,
    )


def test_masked_mse_uses_only_valid_targets_and_excludes_bus_placeholders():
    predictions = torch.zeros((2, 2, 8, 1), requires_grad=True)
    targets = torch.ones_like(predictions)
    predictions.data[:, :, 7] = 1_000_000
    targets[:, :, 7] = -1_000_000
    mask = torch.tensor([True, True, True, True, True, True, True, False]).view(8, 1)
    loss = masked_mse_loss(predictions, targets, mask)
    assert loss.item() == 1.0
    loss.backward()
    assert torch.count_nonzero(predictions.grad[:, :, 7]) == 0
    assert torch.isfinite(predictions.grad).all()


def test_masked_mse_handles_batched_channel_masks():
    predictions = torch.tensor([[[[1.0, 100.0], [3.0, 200.0]]]])
    targets = torch.zeros_like(predictions)
    mask = torch.tensor([[[True, False], [True, False]]])
    assert masked_mse_loss(predictions, targets, mask).item() == 5.0


def test_masked_mse_rejects_all_invalid_targets():
    with pytest.raises(ForecastTrainingError, match="at least one valid"):
        masked_mse_loss(
            torch.zeros((1, 2, 8, 1)),
            torch.zeros((1, 2, 8, 1)),
            torch.zeros((1, 8, 1), dtype=torch.bool),
        )


def test_training_updates_parameters_and_aggregates_incomplete_batches(loaders, tmp_path):
    torch.manual_seed(3)
    model = GNNTransformerForecaster.from_config(_model_config())
    initial = next(model.parameters()).detach().clone()
    result = _fit(model, loaders, tmp_path)
    assert result.device == "cpu"
    assert len(result.history) == 2
    assert result.history[0]["training_valid_targets"] == 5 * 3 * 7
    assert result.history[0]["validation_valid_targets"] == 5 * 3 * 7
    assert not torch.equal(initial, next(model.parameters()).detach())
    assert all(
        np.isfinite(row["training_loss"]) and np.isfinite(row["validation_loss"])
        for row in result.history
    )
    assert Path(result.checkpoint_path).is_file()
    assert Path(result.resume_checkpoint_path).is_file()


def test_training_and_validation_losses_use_valid_element_weighting(loaders, tmp_path):
    model = GNNTransformerForecaster.from_config(_model_config())
    result = _fit(model, loaders, tmp_path, _training_config(max_epochs=1))
    model.eval()
    square_sum, count = 0.0, 0
    with torch.inference_mode():
        for batch in loaders.validation:
            predictions = model(
                batch["X"],
                loaders.graph.edge_index,
                loaders.graph.edge_weight,
                target_mask=batch["target_mask"],
                node_order=loaders.node_order,
            )
            valid = batch["target_mask"][:, None].expand_as(batch["y"])
            square_sum += torch.square(predictions[valid] - batch["y"][valid]).sum().item()
            count += int(valid.sum().item())
    assert result.history[0]["validation_loss"] == pytest.approx(square_sum / count)


def test_early_stopping_uses_validation_only_and_records_reason():
    stopper = EarlyStopping(patience=2, min_delta=0.01)
    assert stopper.update(1, 1.0) is True
    assert stopper.update(2, 0.995) is True
    assert not stopper.should_stop
    assert stopper.update(3, 0.996) is False
    assert stopper.should_stop
    assert stopper.best_epoch == 2
    assert stopper.stop_reason == "validation_loss_not_improved_for_patience_epochs"


def test_best_checkpoint_matches_best_validation_epoch(loaders, tmp_path):
    model = GNNTransformerForecaster.from_config(_model_config())
    result = _fit(
        model,
        loaders,
        tmp_path,
        _training_config(max_epochs=4, patience=2, min_delta=100.0),
    )
    best = load_forecaster_for_evaluation(result.checkpoint_path)
    payload = torch.load(result.checkpoint_path, map_location="cpu", weights_only=True)
    state = payload["training_state"]
    assert state["epoch"] == result.best_epoch
    assert result.best_validation_loss == min(
        row["validation_loss"] for row in result.history
    )
    assert best.training is False
    assert state["selection_partition"] == "validation"
    assert state["test_partition_used_for_selection"] is False


def test_evaluation_checkpoint_round_trip_and_overwrite_protection(loaders, tmp_path):
    model = GNNTransformerForecaster.from_config(_model_config())
    result = _fit(model, loaders, tmp_path, _training_config(max_epochs=1))
    restored = load_forecaster_for_evaluation(result.checkpoint_path)
    batch = next(iter(loaders.validation))
    expected = model.eval()(
        batch["X"],
        loaders.graph.edge_index,
        target_mask=batch["target_mask"],
        node_order=loaders.node_order,
    )
    actual = restored(
        batch["X"],
        loaders.graph.edge_index,
        target_mask=batch["target_mask"],
        node_order=loaders.node_order,
    )
    torch.testing.assert_close(expected, actual)
    with pytest.raises(FileExistsError, match="choose a new experiment"):
        _fit(
            GNNTransformerForecaster.from_config(_model_config()),
            loaders,
            tmp_path,
            _training_config(max_epochs=1),
        )


def test_resume_restores_optimizer_early_stopping_rng_and_history(loaders, tmp_path):
    model = GNNTransformerForecaster.from_config(_model_config())
    first = _fit(model, loaders, tmp_path, _training_config(max_epochs=1))
    resumed_model = GNNTransformerForecaster.from_config(_model_config())
    resumed = _fit(
        resumed_model,
        loaders,
        tmp_path,
        _training_config(max_epochs=2),
        resume_from=first.resume_checkpoint_path,
        resume_checkpoint_path=first.resume_checkpoint_path,
    )
    assert [row["epoch"] for row in resumed.history] == [1, 2]
    payload = torch.load(resumed.resume_checkpoint_path, map_location="cpu", weights_only=True)
    state = payload["training_state"]
    assert state["optimizer_state_dict"]["state"]
    assert state["early_stopping_state"]["best_epoch"] is not None
    assert "torch_cpu" in state["rng_state"]
    assert state["target_order"] == list(TARGET_NODES)


def test_resume_rejects_graph_or_preprocessing_mismatch(loaders, tmp_path):
    first = _fit(
        GNNTransformerForecaster.from_config(_model_config()),
        loaders,
        tmp_path,
        _training_config(max_epochs=1),
    )
    with pytest.raises(ForecastTrainingError, match="preprocessing references"):
        _fit(
            GNNTransformerForecaster.from_config(_model_config()),
            loaders,
            tmp_path,
            _training_config(max_epochs=2),
            resume_from=first.resume_checkpoint_path,
            resume_checkpoint_path=first.resume_checkpoint_path,
            preprocessing_artifacts={"preprocessing": "different.json"},
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("batch_size", 0),
        ("learning_rate", 0),
        ("weight_decay", -1),
        ("max_epochs", 0),
        ("patience", 0),
        ("min_delta", -0.1),
        ("seed", -1),
        ("optimizer", "sgd"),
        ("loss", "mse"),
        ("gradient_clip_norm", 0),
        ("device", "mps"),
    ],
)
def test_training_configuration_rejects_invalid_values(field, value):
    with pytest.raises(ForecastTrainingError):
        TrainingConfig.from_mapping({field: value})


def test_seeded_model_initialization_is_reproducible():
    seed_everything(77)
    first = GNNTransformerForecaster.from_config(_model_config())
    seed_everything(77)
    second = GNNTransformerForecaster.from_config(_model_config())
    for left, right in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(left, right)


def test_nonfinite_valid_values_fail_explicitly():
    predictions = torch.tensor([[[[float("nan")]]]])
    with pytest.raises(ForecastTrainingError, match="must be finite"):
        masked_mse_loss(
            predictions,
            torch.zeros_like(predictions),
            torch.ones((1, 1, 1), dtype=torch.bool),
        )


def test_test_partition_cannot_be_passed_as_training_loader(partition_results):
    from src.forecasting.dataset import ForecastDatasetError
    from src.forecasting.dataloader import ForecastDataLoaderError
    test_partition = _partition("test", "2026-03-01T00:00:00")
    with pytest.raises(ForecastDatasetError, match="Expected 'validation' data"):
        build_forecasting_train_validation_dataloaders(
            {
                "train": partition_results["train"],
                "validation": test_partition,
            },
            batch_size=2,
        )
    with pytest.raises(ForecastDataLoaderError, match="exactly the train and validation"):
        build_forecasting_train_validation_dataloaders(
            {**partition_results, "test": test_partition}, batch_size=2
        )


def test_disk_loader_reads_only_train_and_validation(monkeypatch, partition_results, tmp_path):
    import src.forecasting.dataloader as module

    calls = []

    def fake_load(*args, **kwargs):
        partition = args[1]
        calls.append((partition, tuple(kwargs["verify_partitions"])))
        return partition_results[partition]

    def fake_prepare(sources, config):
        assert config is not None
        return sources

    monkeypatch.setattr(module, "load_partition_sources", fake_load)
    monkeypatch.setattr(module, "prepare_sequences", fake_prepare)
    loaders = load_forecasting_train_validation_dataloaders(
        tmp_path, batch_size=2, sequence_config={"synthetic_fixture": True}
    )
    assert [partition for partition, _ in calls] == ["train", "validation"]
    assert all(selected == ("train", "validation") for _, selected in calls)
    assert loaders.node_order == NODE_ORDER


def test_training_rejects_nonfinite_loss_and_wrong_partition(loaders, tmp_path):
    class NonFiniteModel(GNNTransformerForecaster):
        def forward(self, *args, **kwargs):
            output = super().forward(*args, **kwargs)
            return output + torch.tensor(float("nan"), device=output.device)

    model = NonFiniteModel.from_config(_model_config())
    with pytest.raises(ForecastTrainingError, match="must be finite"):
        _fit(model, loaders, tmp_path, _training_config(max_epochs=1))


def test_training_rejects_masks_that_mark_bus_as_a_valid_target(loaders, tmp_path):
    loaders.train.dataset._target_mask[-1, 0] = True
    with pytest.raises(ForecastTrainingError, match="non-target/context"):
        _fit(
            GNNTransformerForecaster.from_config(_model_config()),
            loaders,
            tmp_path,
            _training_config(max_epochs=1),
        )


def test_checked_in_training_yaml_and_experiment_metadata_serialize(tmp_path):
    import json

    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (root / "configs" / "forecasting_training.yaml").read_text(encoding="utf-8")
    )
    assert TrainingConfig.from_mapping(config).optimizer == "adamw"
    schema = json.loads(
        (root / "configs" / "experiment_metadata.schema.json").read_text(encoding="utf-8")
    )
    assert schema["properties"]["data"]["additionalProperties"] is False
    assert "graph_topology_sha256" in schema["properties"]["data"]["properties"]
    digest = "a" * 64
    metadata = {
        "schema_version": "1.0",
        "experiment_id": "part58-test",
        "phase": "Part_5.8_training_validation",
        "status": "completed",
        "created_at_utc": "2026-11-11T00:00:00Z",
        "seed": 42,
        "data": {
            "dataset_manifest_sha256": digest,
            "source_ids": ["synthetic_fixture"],
            "source_origins": ["simulated"],
            "training_partition_sha256": digest,
            "validation_partition_sha256": digest,
            "feature_columns": ["historical_power_scaled"],
            "target_column": "power_kw",
            "lookback_hours": 4,
            "horizon_hours": 3,
            "preprocessing_fit_partition": "train",
            "graph_topology_sha256": digest,
            "graph_reference": "configs/forecasting_data.yaml",
            "target_nodes": list(TARGET_NODES),
            "target_channels": ["power_kw"],
            "target_unit": "kW",
            "preprocessing_artifacts": {"path": "preprocessing.json", "sha256": digest},
            "split_root": "data/processed/synthetic",
        },
        "model": {
            "name": "GNN-Transformer",
            "architecture": {"type": "gnn_transformer"},
            "configuration": _model_config(),
        },
        "training": {
            "epochs": 2,
            "batch_size": 2,
            "optimizer": "adamw",
            "loss": "masked_mse",
            "selection_partition": "validation",
            "test_partition_used_for_selection": False,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "patience": 8,
            "min_delta": 0.0001,
            "gradient_clip_norm": 1.0,
            "max_epochs": 2,
            "best_epoch": 1,
            "best_validation_loss": 1.0,
            "duration_seconds": 1.0,
            "stop_reason": "max_epochs_reached",
            "checkpoint_path": "best.pt",
            "resume_checkpoint_path": "last.pt",
            "history_path": "training_history.json",
        },
        "software": {
            "python_version": "3.12",
            "framework": "PyTorch",
            "framework_version": "2.7.0+cpu",
            "device": "cpu",
            "deterministic": False,
            "package_versions": {"numpy": "2.0"},
        },
        "artifacts": {"best_checkpoint_sha256": digest},
    }
    destination = write_json(tmp_path / "experiment.json", metadata)
    assert json.loads(destination.read_text(encoding="utf-8")) == metadata
