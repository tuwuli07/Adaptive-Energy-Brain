"""Synthetic tests for per-node temporal Transformer encoding."""

from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

from src.forecasting.models import (
    GraphMessagePassingEncoder,
    SinusoidalPositionalEncoding,
    TransformerEncoderError,
    TransformerTemporalEncoder,
)


def _encoder(
    *,
    d_model: int = 8,
    max_seq_len: int = 12,
    dropout: float = 0,
) -> TransformerTemporalEncoder:
    return TransformerTemporalEncoder(
        d_model=d_model,
        num_heads=2,
        num_layers=2,
        dim_feedforward=16,
        dropout=dropout,
        activation="relu",
        max_seq_len=max_seq_len,
    )


def test_output_shape_for_multiple_samples_nodes_and_history_lengths():
    model = _encoder().eval()
    for batch_size, lookback, node_count in ((1, 1, 1), (3, 5, 4), (2, 9, 7)):
        inputs = torch.randn(batch_size, lookback, node_count, 8)
        assert model(inputs).shape == inputs.shape


def test_cpu_forward_and_finite_outputs():
    model = _encoder().cpu().eval()
    output = model(torch.randn(2, 6, 3, 8, device="cpu"))
    assert output.device.type == "cpu"
    assert torch.isfinite(output).all()


def test_sinusoidal_encoding_distinguishes_positions_without_changing_dimension():
    positional = SinusoidalPositionalEncoding(d_model=7, max_seq_len=10)
    encoded = positional(torch.zeros(1, 5, 7))
    assert encoded.shape == (1, 5, 7)
    assert not torch.equal(encoded[:, 0], encoded[:, 1])
    assert torch.isfinite(encoded).all()


def test_eval_mode_is_deterministic():
    model = _encoder(dropout=0.5).eval()
    inputs = torch.randn(2, 4, 3, 8)
    torch.testing.assert_close(model(inputs), model(inputs))


def test_attention_does_not_mix_independent_batch_samples():
    torch.manual_seed(3)
    model = _encoder().eval()
    inputs = torch.randn(2, 6, 3, 8)
    together = model(inputs)
    individually = torch.cat([model(inputs[index : index + 1]) for index in range(2)])
    torch.testing.assert_close(together, individually)


def test_attention_does_not_mix_different_nodes():
    torch.manual_seed(5)
    model = _encoder().eval()
    inputs = torch.randn(1, 6, 3, 8)
    together = model(inputs)
    individually = torch.cat(
        [model(inputs[:, :, index : index + 1]) for index in range(3)], dim=2
    )
    torch.testing.assert_close(together, individually)
    changed = inputs.clone()
    changed[:, :, 1] += 100
    updated = model(changed)
    torch.testing.assert_close(together[:, :, 0], updated[:, :, 0])
    torch.testing.assert_close(together[:, :, 2], updated[:, :, 2])


def test_bidirectional_temporal_attention_uses_only_provided_history():
    torch.manual_seed(7)
    model = _encoder().eval()
    history = torch.randn(1, 5, 2, 8)
    changed_history = history.clone()
    changed_history[:, 4] += 10
    before = model(history)
    after = model(changed_history)
    assert not torch.allclose(before[:, 0], after[:, 0])


def test_padding_mask_ignores_padded_keys_and_zeros_padded_outputs():
    torch.manual_seed(11)
    model = _encoder().eval()
    inputs = torch.randn(2, 6, 3, 8)
    mask = torch.tensor(
        [[False, False, False, True, True, True], [False] * 6], dtype=torch.bool
    )
    changed = inputs.clone()
    changed[0, 3:] = 10_000
    first = model(inputs, padding_mask=mask)
    second = model(changed, padding_mask=mask)
    torch.testing.assert_close(first[0, :3], second[0, :3])
    assert torch.count_nonzero(first[0, 3:]) == 0


def test_node_specific_padding_mask_is_supported():
    model = _encoder().eval()
    inputs = torch.randn(1, 5, 2, 8)
    mask = torch.tensor(
        [[[False, False, True, True, True], [False, False, False, False, False]]]
    )
    output = model(inputs, padding_mask=mask)
    assert output.shape == inputs.shape
    assert torch.count_nonzero(output[:, 2:4, 0]) == 0
    assert torch.count_nonzero(output[:, :, 1]) > 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"d_model": 0}, "d_model"),
        ({"d_model": 8, "num_heads": 0}, "num_heads"),
        ({"d_model": 8, "num_heads": 3}, "divisible"),
        ({"d_model": 8, "num_layers": 0}, "num_layers"),
        ({"d_model": 8, "dim_feedforward": 0}, "dim_feedforward"),
        ({"d_model": 8, "dropout": -0.01}, "dropout"),
        ({"d_model": 8, "dropout": 1.0}, "dropout"),
        ({"d_model": 8, "activation": "tanh"}, "activation"),
        ({"d_model": 8, "max_seq_len": 0}, "max_seq_len"),
        ({"d_model": 8, "positional_encoding": "learned"}, "positional_encoding"),
    ],
)
def test_invalid_transformer_configuration_rejected(kwargs, message):
    with pytest.raises(TransformerEncoderError, match=message):
        TransformerTemporalEncoder(**kwargs)


def test_sequences_longer_than_configured_maximum_are_rejected():
    model = _encoder(max_seq_len=4)
    with pytest.raises(TransformerEncoderError, match="exceeds configured maximum"):
        model(torch.randn(1, 5, 2, 8))


@pytest.mark.parametrize(
    "inputs",
    [
        torch.ones(1, 4, 8),
        torch.ones(1, 4, 2, 4),
        torch.ones(1, 4, 2, 8, dtype=torch.float64),
        torch.full((1, 4, 2, 8), float("nan")),
        torch.empty((1, 0, 2, 8)),
    ],
)
def test_invalid_input_shape_dimension_dtype_or_finiteness_rejected(inputs):
    with pytest.raises(TransformerEncoderError):
        _encoder()(inputs)


@pytest.mark.parametrize(
    "mask",
    [
        torch.ones(1, 4, dtype=torch.float32),
        torch.ones(1, 2, 4, dtype=torch.int64),
        torch.ones(1, 2),
        torch.ones(1, 2, 4, dtype=torch.bool),
    ],
)
def test_invalid_or_all_padding_masks_are_rejected(mask):
    inputs = torch.randn(1, 4, 2, 8)
    with pytest.raises(TransformerEncoderError):
        _encoder()(inputs, padding_mask=mask)


def test_backpropagation_reaches_inputs_and_parameters_with_finite_gradients():
    torch.manual_seed(13)
    model = _encoder()
    inputs = torch.randn(2, 5, 3, 8, requires_grad=True)
    output = model(inputs)
    output.square().mean().backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_gnn_encoder_output_is_compatible_with_temporal_encoder():
    gnn = GraphMessagePassingEncoder(
        input_dim=1, hidden_dim=8, num_layers=1, dropout=0
    ).eval()
    temporal = _encoder().eval()
    inputs = torch.randn(2, 7, 4, 1)
    edges = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    spatial = gnn(inputs, edges)
    output = temporal(spatial)
    assert spatial.shape == (2, 7, 4, 8)
    assert output.shape == spatial.shape


def test_config_defaults_match_gnn_dimension_and_construct_temporal_encoder():
    config_path = Path(__file__).resolve().parents[1] / "configs" / "forecasting_model.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))["model"]
    assert config["gnn_encoder"]["hidden_dim"] == config["transformer_encoder"]["d_model"]
    model = TransformerTemporalEncoder.from_config(config["transformer_encoder"])
    assert model.d_model == 64
    assert model.num_heads == 4
    assert model.max_seq_len == 168


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("attention_strategy", "causal_autoregressive", "bidirectional"),
        ("padding_mask_convention", "false_means_padding", "padding mask"),
    ],
)
def test_unsupported_attention_or_mask_config_is_rejected(field, value, message):
    config_path = Path(__file__).resolve().parents[1] / "configs" / "forecasting_model.yaml"
    config = deepcopy(yaml.safe_load(config_path.read_text(encoding="utf-8"))["model"]["transformer_encoder"])
    config[field] = value
    with pytest.raises(TransformerEncoderError, match=message):
        TransformerTemporalEncoder.from_config(config)
