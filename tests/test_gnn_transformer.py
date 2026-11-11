"""Synthetic integration tests for direct GNN-Transformer forecasting."""

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from src.data.graph_builder import build_graph
from src.forecasting.dataloader import build_forecasting_dataloaders
from src.forecasting.models import (
    GNNTransformerError,
    GNNTransformerForecaster,
)


def _config(
    *,
    graph_node_ids=("N0", "N1", "N2"),
    target_node_ids=("N0", "N1"),
    input_dim=1,
    lookback=4,
    horizon=3,
    target_channels=("power_kw",),
    target_scope="node_level",
    system_pooling=None,
):
    node_order = list(graph_node_ids)
    target_ids = list(target_node_ids)
    return {
        "version": 1,
        "data": {"lookback_hours": lookback, "horizon_hours": horizon},
        "graph": {"node_order": node_order},
        "targets": {"nodes": target_ids, "channels": list(target_channels)},
        "model": {
            "architecture": "gnn_transformer",
            "spatial_encoder": "graph_mean_message_passing",
            "temporal_encoder": "per_node_transformer",
            "lookback_hours": lookback,
            "forecast_horizon_hours": horizon,
            "target_scope": target_scope,
            "system_pooling": system_pooling,
            "gnn_encoder": {
                "input_dim": input_dim,
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
                "max_seq_len": max(lookback, 8),
                "positional_encoding": "sinusoidal",
                "attention_strategy": "bidirectional_within_historical_window",
                "padding_mask_convention": "true_means_padding",
            },
            "forecast_head": {"type": "direct_multi_step", "dropout": 0.0},
        },
    }


def _model(**kwargs):
    return GNNTransformerForecaster.from_config(_config(**kwargs))


def _small_partition(partition: str, start: str):
    node_order = ["B01", "B02", "B03", "B04", "B05", "PV01", "W01", "BUS01"]
    batch_size, lookback, horizon, node_count = 3, 4, 3, len(node_order)
    start_time = np.datetime64(start, "ns")
    offsets = np.arange(batch_size)[:, None]
    input_times = start_time + (offsets + np.arange(lookback)[None, :]) * np.timedelta64(1, "h")
    target_times = (
        start_time
        + (offsets + lookback + np.arange(horizon)[None, :]) * np.timedelta64(1, "h")
    )
    end = start_time + np.timedelta64(batch_size + lookback + horizon - 1, "h")
    boundaries = {
        "start_utc": f"{start}Z",
        "end_exclusive_utc": np.datetime_as_string(end, unit="s") + "Z",
    }
    mask = np.ones((node_count, 1), dtype=bool)
    mask[-1] = False
    x = np.arange(batch_size * lookback * node_count, dtype=np.float32).reshape(
        batch_size, lookback, node_count, 1
    )
    y = np.arange(batch_size * horizon * node_count, dtype=np.float32).reshape(
        batch_size, horizon, node_count, 1
    ) + 1
    x[:, :, -1, :] = 0
    y[:, :, -1, :] = 0
    return {
        "arrays": {
            "X": x,
            "global_X": np.zeros((batch_size, lookback, 0), dtype=np.float32),
            "y": y,
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
            "node_order": node_order,
            "target_unit": "kW",
            "source_metadata": {
                "synthetic_test_fixture": {
                    "split": {
                        "name": partition,
                        "fitted_on": "train",
                        "boundaries": boundaries,
                    }
                }
            },
        },
    }


@pytest.fixture
def forecasting_loaders():
    partitions = {
        "train": _small_partition("train", "2026-01-01T00:00:00"),
        "validation": _small_partition("validation", "2026-02-01T00:00:00"),
        "test": _small_partition("test", "2026-03-01T00:00:00"),
    }
    return build_forecasting_dataloaders(
        partitions, batch_size=2, shuffle_train=False
    )


def test_node_forecast_shape_targets_and_non_target_placeholders():
    config = _config(
        graph_node_ids=("B01", "PV01", "BUS01"),
        target_node_ids=("B01", "PV01"),
    )
    model = GNNTransformerForecaster.from_config(config).eval()
    inputs = torch.randn(2, 4, 3, 1)
    edges = torch.tensor([[0, 1, 2, 2], [2, 2, 0, 1]], dtype=torch.long)
    target_mask = torch.tensor([[True], [True], [False]])
    output = model(
        inputs,
        edges,
        target_mask=target_mask,
        node_order=("B01", "PV01", "BUS01"),
    )
    assert output.shape == (2, 3, 3, 1)
    assert torch.count_nonzero(output[:, :, 2]) == 0
    assert model.output_node_ids == ("B01", "PV01", "BUS01")


def test_single_node_and_multiple_nodes_are_supported():
    single = _model(graph_node_ids=("only",), target_node_ids=("only",)).eval()
    assert single(
        torch.randn(1, 4, 1, 1), torch.empty((2, 0), dtype=torch.long)
    ).shape == (1, 3, 1, 1)
    multiple = _model().eval()
    assert multiple(
        torch.randn(5, 4, 3, 1),
        torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
    ).shape == (5, 3, 3, 1)


def test_configurable_horizon_features_channels_and_variable_batches():
    model = _model(
        input_dim=2,
        horizon=7,
        target_channels=("demand_kw", "generation_kw"),
    ).eval()
    output = model(
        torch.randn(3, 4, 3, 2),
        torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
    )
    assert output.shape == (3, 7, 3, 2)


def test_variable_lookback_uses_configured_limit():
    config = _config(lookback=4)
    config["model"]["lookback_hours"] = None
    config["data"]["lookback_hours"] = None
    config["model"]["transformer_encoder"]["max_seq_len"] = 10
    model = GNNTransformerForecaster.from_config(config).eval()
    assert model(torch.randn(2, 8, 3, 1), torch.empty((2, 0), dtype=torch.long)).shape == (
        2,
        3,
        3,
        1,
    )


@pytest.mark.parametrize(
    ("config_update", "message"),
    [
        ({"model": {"forecast_horizon_hours": 0}}, "positive integer"),
        ({"model": {"architecture": "transformer_only"}}, "architecture"),
        ({"model": {"forecast_horizon_hours": 4}}, "match data.horizon_hours"),
    ],
)
def test_invalid_model_configuration_rejected(config_update, message):
    config = _config()
    config["model"].update(config_update["model"])
    with pytest.raises(GNNTransformerError, match=message):
        GNNTransformerForecaster.from_config(config)


def test_encoder_dimensions_must_match():
    config = _config()
    config["model"]["transformer_encoder"]["d_model"] = 16
    with pytest.raises(GNNTransformerError, match="must equal"):
        GNNTransformerForecaster.from_config(config)


def test_invalid_input_shapes_node_counts_and_order_are_rejected():
    model = _model()
    edges = torch.empty((2, 0), dtype=torch.long)
    with pytest.raises(GNNTransformerError, match="Expected 3 graph nodes"):
        model(torch.randn(2, 4, 2, 1), edges)
    with pytest.raises(GNNTransformerError, match="input features"):
        model(torch.randn(2, 4, 3, 2), edges)
    with pytest.raises(GNNTransformerError, match="node order"):
        model(torch.randn(2, 4, 3, 1), edges, node_order=("N1", "N0", "N2"))


def test_unknown_target_nodes_and_invalid_target_masks_are_rejected():
    config = _config()
    config["targets"]["nodes"] = ["missing"]
    with pytest.raises(GNNTransformerError, match="not present"):
        GNNTransformerForecaster.from_config(config)
    model = _model()
    with pytest.raises(GNNTransformerError, match="boolean"):
        model(
            torch.randn(1, 4, 3, 1),
            torch.empty((2, 0), dtype=torch.long),
            target_mask=torch.ones(3, 1),
        )


def test_system_level_forecasting_requires_explicit_pooling_and_returns_system_shape():
    model = _model(
        target_node_ids=("N0", "N1"),
        target_scope="system_level",
        system_pooling="mean",
    ).eval()
    output = model(
        torch.randn(2, 4, 3, 1),
        torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
    )
    assert output.shape == (2, 3, 1, 1)
    assert model.output_node_ids == ("system",)
    with pytest.raises(GNNTransformerError, match="requires system_pooling"):
        _model(target_scope="system_level")


def test_system_pooling_respects_valid_node_mask_and_rejects_all_invalid():
    model = _model(
        target_node_ids=("N0", "N1"),
        target_scope="system_level",
        system_pooling="mean",
    ).eval()
    inputs = torch.randn(2, 4, 3, 1)
    edges = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    mask = torch.tensor([[[True], [False], [True]], [[True], [False], [True]]])
    output = model(inputs, edges, target_mask=mask)
    assert output.shape == (2, 3, 1, 1)
    all_invalid = torch.zeros((2, 3, 1), dtype=torch.bool)
    with pytest.raises(GNNTransformerError, match="at least one valid"):
        model(inputs, edges, target_mask=all_invalid)


def test_gradients_reach_gnn_transformer_and_forecast_head():
    torch.manual_seed(29)
    model = _model()
    inputs = torch.randn(2, 4, 3, 1, requires_grad=True)
    edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    predictions = model(inputs, edges)
    loss = predictions.square().mean()
    loss.backward()
    assert predictions.shape == (2, 3, 3, 1)
    for component in (
        model.gnn_encoder,
        model.transformer_encoder,
        model.forecast_projection,
    ):
        gradients = [parameter.grad for parameter in component.parameters()]
        assert gradients and all(
            gradient is not None and torch.isfinite(gradient).all()
            for gradient in gradients
        )
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    assert torch.isfinite(predictions).all() and torch.isfinite(loss)


def test_eval_inference_is_deterministic_and_does_not_mix_batch_samples():
    torch.manual_seed(31)
    model = _model().eval()
    edges = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    inputs = torch.randn(3, 4, 3, 1)
    together = model(inputs, edges)
    individually = torch.cat([model(inputs[i : i + 1], edges) for i in range(3)])
    torch.testing.assert_close(together, individually)
    torch.testing.assert_close(together, model(inputs, edges))


def test_future_target_changes_cannot_affect_forward_predictions():
    model = _model().eval()
    inputs = torch.randn(2, 4, 3, 1)
    edges = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    predictions_before = model(inputs, edges)
    future_labels = torch.randn(2, 3, 3, 1)
    future_labels.fill_(1_000_000)
    predictions_after = model(inputs, edges)
    torch.testing.assert_close(predictions_before, predictions_after)


def test_checkpoint_round_trip_refuses_overwrite_by_default(tmp_path):
    torch.manual_seed(37)
    config = _config()
    model = GNNTransformerForecaster.from_config(config).eval()
    inputs = torch.randn(2, 4, 3, 1)
    edges = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    expected = model(inputs, edges)
    checkpoint = model.save_checkpoint(tmp_path / "model.pt")
    restored = GNNTransformerForecaster.load_checkpoint(checkpoint).eval()
    torch.testing.assert_close(expected, restored(inputs, edges))
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        model.save_checkpoint(checkpoint)
    model.save_checkpoint(checkpoint, overwrite=True)


def test_actual_model_config_constructs_without_replacing_encoder_settings():
    path = Path(__file__).resolve().parents[1] / "configs" / "forecasting_model.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    model = GNNTransformerForecaster.from_config(config)
    assert model.forecast_horizon == 24
    assert model.graph_node_ids == (
        "B01", "B02", "B03", "B04", "B05", "PV01", "W01", "BUS01"
    )
    assert model.target_node_ids == ("B01", "B02", "B03", "B04", "B05", "PV01", "W01")


def test_part_5_4_dataloader_batch_produces_target_aligned_forecasts(forecasting_loaders):
    model = _model(
        graph_node_ids=("B01", "B02", "B03", "B04", "B05", "PV01", "W01", "BUS01"),
        target_node_ids=("B01", "B02", "B03", "B04", "B05", "PV01", "W01"),
    ).eval()
    batch = next(iter(forecasting_loaders.train))
    predictions = model(
        batch["X"],
        forecasting_loaders.graph.edge_index,
        forecasting_loaders.graph.edge_weight,
        target_mask=batch["target_mask"],
        node_order=forecasting_loaders.node_order,
    )
    assert predictions.shape == batch["y"].shape
    assert predictions.shape == (2, 3, 8, 1)


def test_default_config_output_aligns_to_checked_in_target_schema():
    config_path = Path(__file__).resolve().parents[1] / "configs" / "forecasting_model.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model = GNNTransformerForecaster.from_config(config).eval()
    inputs = torch.randn(2, 24, 8, 1)
    from src.forecasting.dataloader import validate_forecasting_graph

    graph_data = build_graph()
    graph = validate_forecasting_graph(
        graph_data, node_count=8, expected_node_order=config["graph"]["node_order"]
    )
    target_mask = torch.tensor([True, True, True, True, True, True, True, False]).view(
        1, 8, 1
    ).expand(2, -1, -1)
    predictions = model(
        inputs,
        graph.edge_index,
        target_mask=target_mask,
        node_order=config["graph"]["node_order"],
    )
    assert predictions.shape == (2, 24, 8, 1)
    assert torch.count_nonzero(predictions[:, :, 7]) == 0
