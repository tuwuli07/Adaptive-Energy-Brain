"""Time-step-independent graph mean message passing for energy histories.

Edges use ``source -> target`` indexing: each source sends its projected node
features to the target. Neighbor messages are averaged (or edge-weighted
averaged); every node also receives a separate self projection, so isolated
nodes remain representable without adding self-loops.
"""

from collections.abc import Mapping
from numbers import Integral, Real
from typing import Any

import numpy as np
import torch
from torch import nn


class GNNEncoderError(ValueError):
    """Invalid encoder configuration, features, graph indices, or edge weights."""


_ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
    "silu": nn.SiLU,
    "identity": nn.Identity,
}


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise GNNEncoderError(f"{name} must be a positive integer.")
    return int(value)


def _activation_layer(name: str) -> nn.Module:
    if not isinstance(name, str) or name not in _ACTIVATIONS:
        raise GNNEncoderError(
            f"activation must be one of {', '.join(_ACTIVATIONS)}."
        )
    return _ACTIVATIONS[name]()


class GraphMeanMessagePassingLayer(nn.Module):
    """One differentiable directed weighted-mean message-passing layer."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        self.self_projection = nn.Linear(input_dim, output_dim)
        self.neighbor_projection = nn.Linear(input_dim, output_dim)
        self.activation = _activation_layer(activation)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_features: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        self_features = self.self_projection(node_features)
        if source.numel() == 0:
            neighbor_features = torch.zeros_like(self_features)
        else:
            messages = self.neighbor_projection(node_features[:, source, :])
            degree = torch.zeros(
                node_features.shape[1],
                dtype=torch.float64,
                device=node_features.device,
            ).index_add(0, target, edge_weight)
            if not torch.isfinite(degree).all():
                raise GNNEncoderError("Incoming edge-weight sums must remain finite.")
            normalized_weights = (
                edge_weight
                / degree[target].clamp_min(torch.finfo(torch.float64).tiny)
            ).to(dtype=node_features.dtype)
            weighted_messages = messages * normalized_weights.view(1, -1, 1)
            neighbor_features = torch.zeros_like(self_features).index_add(
                1, target, weighted_messages
            )
        updated = self_features + neighbor_features
        return self.dropout(self.activation(updated))


class GraphMessagePassingEncoder(nn.Module):
    """Encode ``[batch, lookback, nodes, features]`` into spatial embeddings.

    The encoder operates independently on every batch/time slice and returns
    ``[batch, lookback, nodes, hidden_dim]``. Edges are directed from
    ``edge_index[0]`` to ``edge_index[1]``. The default aggregation is the
    nonnegative edge-weighted mean over incoming neighbors; omitted weights
    mean unit weights. Duplicate edges and explicit self-loops are rejected.
    A node without incoming edges uses only its own learned projection.

    ``activation="identity"`` is available for controlled experiments; other
    supported activations are relu, gelu, tanh, elu, and silu.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.input_dim = _positive_integer(input_dim, "input_dim")
        self.hidden_dim = _positive_integer(hidden_dim, "hidden_dim")
        self.num_layers = _positive_integer(num_layers, "num_layers")
        if (
            isinstance(dropout, bool)
            or not isinstance(dropout, Real)
            or not np.isfinite(dropout)
            or not 0 <= dropout < 1
        ):
            raise GNNEncoderError("dropout must be finite and in the interval [0, 1).")
        if not isinstance(activation, str) or activation not in _ACTIVATIONS:
            raise GNNEncoderError(
                f"activation must be one of {', '.join(_ACTIVATIONS)}."
            )
        self.dropout_probability = float(dropout)
        self.activation_name = activation
        dimensions = [self.input_dim] + [self.hidden_dim] * self.num_layers
        self.layers = nn.ModuleList(
            GraphMeanMessagePassingLayer(
                dimensions[index],
                dimensions[index + 1],
                dropout=self.dropout_probability,
                activation=self.activation_name,
            )
            for index in range(self.num_layers)
        )

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "GraphMessagePassingEncoder":
        """Construct from the ``model.gnn_encoder`` mapping in model YAML."""
        if not isinstance(config, Mapping):
            raise GNNEncoderError("GNN encoder configuration must be a mapping.")
        expected = {
            "input_dim",
            "hidden_dim",
            "num_layers",
            "dropout",
            "activation",
            "aggregation",
            "edge_direction",
            "self_loop_policy",
            "isolated_node_policy",
        }
        missing = expected - config.keys()
        if missing:
            raise GNNEncoderError(
                f"GNN encoder configuration is missing {sorted(missing)}."
            )
        if config["aggregation"] != "weighted_mean":
            raise GNNEncoderError("Only weighted_mean neighbor aggregation is supported.")
        if config["edge_direction"] != "source_to_target":
            raise GNNEncoderError("edge_direction must be source_to_target.")
        if config["self_loop_policy"] != "reject_explicit_use_self_projection":
            raise GNNEncoderError("Unsupported self_loop_policy.")
        if config["isolated_node_policy"] != "self_projection_only":
            raise GNNEncoderError("Unsupported isolated_node_policy.")
        return cls(
            input_dim=config["input_dim"],
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
            activation=config["activation"],
        )

    def _graph_tensors(
        self,
        edge_index: torch.Tensor | np.ndarray,
        edge_weight: torch.Tensor | np.ndarray | None,
        *,
        node_count: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(edge_index, np.ndarray):
            if edge_index.dtype != np.int64:
                raise GNNEncoderError("edge_index NumPy values must use int64.")
            edge_index = torch.from_numpy(edge_index)
        if (
            not isinstance(edge_index, torch.Tensor)
            or edge_index.dtype != torch.int64
            or edge_index.ndim != 2
            or edge_index.shape[0] != 2
        ):
            raise GNNEncoderError("edge_index must be int64 with shape [2, edges].")
        edge_index = edge_index.to(device=device)
        source, target = edge_index[0], edge_index[1]
        if source.numel() and (
            int(edge_index.min().item()) < 0
            or int(edge_index.max().item()) >= node_count
        ):
            raise GNNEncoderError(
                f"edge_index values must be in [0, {node_count - 1}]."
            )
        if torch.any(source == target):
            raise GNNEncoderError(
                "Explicit self-loops are not allowed; each layer includes a self projection."
            )
        if source.numel():
            edge_keys = source * node_count + target
            if torch.unique(edge_keys).numel() != edge_keys.numel():
                raise GNNEncoderError("Duplicate directed edges are not allowed.")

        if edge_weight is None:
            weights = torch.ones(source.shape[0], device=device, dtype=torch.float64)
        else:
            if isinstance(edge_weight, np.ndarray):
                edge_weight = torch.from_numpy(edge_weight)
            if (
                not isinstance(edge_weight, torch.Tensor)
                or not edge_weight.is_floating_point()
                or edge_weight.ndim != 1
                or edge_weight.shape[0] != source.shape[0]
            ):
                raise GNNEncoderError(
                    "edge_weight must be a floating vector with one value per edge."
                )
            if not torch.isfinite(edge_weight).all():
                raise GNNEncoderError("edge_weight must contain only finite values.")
            if torch.any(edge_weight < 0):
                raise GNNEncoderError("edge_weight values must be nonnegative.")
            weights = edge_weight.to(device=device, dtype=torch.float64)
            if not torch.isfinite(weights).all():
                raise GNNEncoderError("edge_weight cannot be represented as finite values.")
        return source, target, weights

    def forward(
        self,
        inputs: torch.Tensor,
        edge_index: torch.Tensor | np.ndarray,
        edge_weight: torch.Tensor | np.ndarray | None = None,
    ) -> torch.Tensor:
        if not isinstance(inputs, torch.Tensor) or inputs.ndim != 4:
            raise GNNEncoderError(
                "inputs must be a torch tensor [batch, lookback, nodes, features]."
            )
        if not inputs.is_floating_point() or inputs.dtype != torch.float32:
            raise GNNEncoderError("inputs must use float32 values.")
        if any(size < 1 for size in inputs.shape):
            raise GNNEncoderError("batch, lookback, node, and feature dimensions must be nonempty.")
        if inputs.shape[-1] != self.input_dim:
            raise GNNEncoderError(
                f"Expected {self.input_dim} input features, received {inputs.shape[-1]}."
            )
        if not torch.isfinite(inputs).all():
            raise GNNEncoderError("inputs must contain only finite values.")
        parameter = next(self.parameters())
        if inputs.device != parameter.device:
            raise GNNEncoderError(
                f"inputs are on {inputs.device}, but encoder parameters are on {parameter.device}."
            )

        batch, lookback, node_count, _ = inputs.shape
        source, target, weights = self._graph_tensors(
            edge_index,
            edge_weight,
            node_count=node_count,
            device=inputs.device,
        )
        flattened = inputs.reshape(batch * lookback, node_count, self.input_dim)
        for layer in self.layers:
            flattened = layer(flattened, source, target, weights)
        return flattened.reshape(batch, lookback, node_count, self.hidden_dim)
