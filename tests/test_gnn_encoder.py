"""Small synthetic graph fixtures for the message-passing encoder."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from src.forecasting.models.gnn_encoder import (
    GNNEncoderError,
    GraphMessagePassingEncoder,
)


def _identity_encoder() -> GraphMessagePassingEncoder:
    model = GraphMessagePassingEncoder(
        input_dim=1,
        hidden_dim=1,
        num_layers=1,
        dropout=0,
        activation="identity",
    )
    with torch.no_grad():
        layer = model.layers[0]
        layer.self_projection.weight.fill_(1)
        layer.self_projection.bias.zero_()
        layer.neighbor_projection.weight.fill_(1)
        layer.neighbor_projection.bias.zero_()
    return model


def test_output_shape_for_multiple_batch_time_node_and_feature_dimensions():
    model = GraphMessagePassingEncoder(3, hidden_dim=7, num_layers=2, dropout=0)
    inputs = torch.randn(4, 9, 6, 3)
    edges = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    assert model(inputs, edges).shape == (4, 9, 6, 7)


def test_single_node_without_edges_uses_self_projection():
    model = _identity_encoder()
    inputs = torch.tensor([[[[2.0]]]], dtype=torch.float32)
    output = model(inputs, torch.empty((2, 0), dtype=torch.long))
    torch.testing.assert_close(output, torch.tensor([[[[2.0]]]]))


def test_disconnected_nodes_keep_self_features_and_empty_edges_are_supported():
    model = _identity_encoder()
    inputs = torch.tensor([[[[1.0], [3.0], [5.0]]]], dtype=torch.float32)
    output = model(inputs, torch.empty((2, 0), dtype=torch.long))
    torch.testing.assert_close(output, inputs)


def test_directed_connected_edge_aggregates_source_into_target_only():
    model = _identity_encoder()
    inputs = torch.tensor([[[[1.0], [3.0]]]], dtype=torch.float32)
    output = model(inputs, torch.tensor([[0], [1]], dtype=torch.long))
    torch.testing.assert_close(output, torch.tensor([[[[1.0], [4.0]]]]))


def test_neighbor_aggregation_is_weighted_mean():
    model = _identity_encoder()
    inputs = torch.tensor([[[[2.0], [6.0], [8.0]]]], dtype=torch.float32)
    edges = torch.tensor([[0, 1], [2, 2]], dtype=torch.long)
    weights = torch.tensor([1.0, 3.0])
    output = model(inputs, edges, weights)
    expected_target = 8.0 + (1.0 * 2.0 + 3.0 * 6.0) / 4.0
    torch.testing.assert_close(output[0, 0, :, 0], torch.tensor([2.0, 6.0, expected_target]))


def test_variable_batch_sizes_and_sequence_lengths_are_supported():
    model = GraphMessagePassingEncoder(1, hidden_dim=4, num_layers=2, dropout=0)
    edges = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    assert model(torch.randn(1, 1, 2, 1), edges).shape == (1, 1, 2, 4)
    assert model(torch.randn(5, 11, 2, 1), edges).shape == (5, 11, 2, 4)


def test_no_mixing_between_independent_batch_samples():
    model = _identity_encoder()
    edges = torch.tensor([[0], [1]], dtype=torch.long)
    inputs = torch.tensor(
        [[[[1.0], [3.0]]], [[[20.0], [30.0]]]], dtype=torch.float32
    )
    outputs = model(inputs, edges)
    torch.testing.assert_close(outputs[0], torch.tensor([[[1.0], [4.0]]]))
    torch.testing.assert_close(outputs[1], torch.tensor([[[20.0], [50.0]]]))


def test_no_mixing_between_historical_time_steps():
    model = _identity_encoder()
    edges = torch.tensor([[0], [1]], dtype=torch.long)
    inputs = torch.tensor([[[[1.0], [3.0]], [[100.0], [200.0]]]])
    outputs = model(inputs, edges)
    torch.testing.assert_close(outputs[0, 0], torch.tensor([[1.0], [4.0]]))
    torch.testing.assert_close(outputs[0, 1], torch.tensor([[100.0], [300.0]]))


def test_graph_connectivity_changes_connected_node_embedding():
    model = _identity_encoder()
    inputs = torch.tensor([[[[2.0], [5.0]]]], dtype=torch.float32)
    no_edges = model(inputs, torch.empty((2, 0), dtype=torch.long))
    connected = model(inputs, torch.tensor([[0], [1]], dtype=torch.long))
    torch.testing.assert_close(no_edges[0, 0, 1], torch.tensor([5.0]))
    torch.testing.assert_close(connected[0, 0, 1], torch.tensor([7.0]))


def test_edge_weights_change_aggregation_and_numpy_graph_indices_are_supported():
    model = _identity_encoder()
    inputs = torch.tensor([[[[2.0], [4.0], [8.0]]]], dtype=torch.float32)
    edges = np.array([[0, 1], [2, 2]], dtype=np.int64)
    weighted = model(inputs, edges, np.array([3.0, 1.0], dtype=np.float32))
    unweighted = model(inputs, edges)
    torch.testing.assert_close(weighted[0, 0, 2], torch.tensor([10.5]))
    torch.testing.assert_close(unweighted[0, 0, 2], torch.tensor([11.0]))


@pytest.mark.parametrize(
    ("edges", "weights", "message"),
    [
        (torch.tensor([[0], [2]], dtype=torch.long), None, "in \\[0, 1\\]"),
        (torch.tensor([[0], [1]], dtype=torch.int32), None, "int64"),
        (torch.tensor([[0], [0]], dtype=torch.long), None, "self-loops"),
        (torch.tensor([[0, 0], [1, 1]], dtype=torch.long), None, "Duplicate"),
        (torch.tensor([[0], [1]], dtype=torch.long), torch.tensor([-1.0]), "nonnegative"),
        (torch.tensor([[0], [1]], dtype=torch.long), torch.tensor([float("nan")]), "finite"),
    ],
)
def test_invalid_graph_indices_self_loops_duplicates_or_weights_rejected(
    edges, weights, message
):
    model = _identity_encoder()
    inputs = torch.ones(1, 1, 2, 1)
    with pytest.raises(GNNEncoderError, match=message):
        model(inputs, edges, weights)


@pytest.mark.parametrize(
    "inputs",
    [
        torch.ones(2, 3, 1),
        torch.ones(1, 2, 2, 2),
        torch.ones(1, 2, 2, 1, dtype=torch.float64),
        torch.full((1, 2, 2, 1), float("inf")),
        torch.empty((1, 0, 2, 1)),
    ],
)
def test_invalid_input_dimensions_features_types_and_values_rejected(inputs):
    model = _identity_encoder()
    edges = torch.empty((2, 0), dtype=torch.long)
    with pytest.raises(GNNEncoderError):
        model(inputs, edges)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"input_dim": 0}, "input_dim"),
        ({"input_dim": 1, "hidden_dim": 0}, "hidden_dim"),
        ({"input_dim": 1, "num_layers": 0}, "num_layers"),
        ({"input_dim": 1, "dropout": -0.1}, "dropout"),
        ({"input_dim": 1, "dropout": 1.0}, "dropout"),
        ({"input_dim": 1, "activation": "attention"}, "activation"),
    ],
)
def test_invalid_encoder_configuration_rejected(kwargs, message):
    with pytest.raises(GNNEncoderError, match=message):
        GraphMessagePassingEncoder(**kwargs)


def test_backward_gradients_are_finite_and_reach_trainable_parameters():
    torch.manual_seed(7)
    model = GraphMessagePassingEncoder(2, hidden_dim=5, num_layers=2, dropout=0)
    inputs = torch.randn(3, 4, 3, 2, requires_grad=True)
    edges = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    loss = model(inputs, edges).square().mean()
    loss.backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_eval_mode_is_deterministic_and_outputs_are_finite():
    model = GraphMessagePassingEncoder(1, hidden_dim=3, dropout=0.5).eval()
    inputs = torch.randn(2, 4, 3, 1)
    edges = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    first = model(inputs, edges)
    second = model(inputs, edges)
    torch.testing.assert_close(first, second)
    assert torch.isfinite(first).all()


def test_config_defaults_construct_encoder_without_requiring_unselected_transformer():
    config_path = Path(__file__).resolve().parents[1] / "configs" / "forecasting_model.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    encoder = GraphMessagePassingEncoder.from_config(config["model"]["gnn_encoder"])
    assert encoder.input_dim == 1
    assert encoder.hidden_dim == 64
    assert encoder.num_layers == 2


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("aggregation", "attention", "weighted_mean"),
        ("edge_direction", "target_to_source", "source_to_target"),
        ("self_loop_policy", "add_self_loops", "self_loop_policy"),
    ],
)
def test_config_rejects_unsupported_message_passing_contract(field, value, message):
    config_path = Path(__file__).resolve().parents[1] / "configs" / "forecasting_model.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))["model"]["gnn_encoder"]
    invalid = deepcopy(config)
    invalid[field] = value
    with pytest.raises(GNNEncoderError, match=message):
        GraphMessagePassingEncoder.from_config(invalid)
