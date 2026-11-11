"""Integrated direct multi-step forecasting from historical node observations."""

from collections.abc import Mapping, Sequence
from copy import deepcopy
from io import BytesIO
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from src.forecasting.models.gnn_encoder import (
    GNNEncoderError,
    GraphMessagePassingEncoder,
)
from src.forecasting.models.transformer_encoder import (
    TransformerEncoderError,
    TransformerTemporalEncoder,
)


class GNNTransformerError(ValueError):
    """Invalid integrated model configuration or input contract."""


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise GNNTransformerError(f"{name} must be a positive integer.")
    return int(value)


def _ordered_ids(values: Any, name: str) -> tuple[str, ...]:
    if (
        not isinstance(values, Sequence)
        or isinstance(values, (str, bytes))
        or not values
        or any(not isinstance(value, str) or not value.strip() for value in values)
        or len(set(values)) != len(values)
    ):
        raise GNNTransformerError(f"{name} must be a nonempty sequence of unique IDs.")
    return tuple(values)


class GNNTransformerForecaster(nn.Module):
    """Forecast configured targets using spatial, temporal, then direct heads.

    Node-level outputs use ``[batch, horizon, graph_nodes, target_channels]``.
    Nodes not present in ``target_node_ids`` are zero placeholders and are
    expected to be masked during loss/metrics. System-level mode is opt-in and
    pools the explicitly listed target nodes with a configured ``mean`` or
    ``sum`` before a direct horizon projection. The current audited dataset
    supports seven node-level targets and does not provide a system label.

    The forward method accepts only historical inputs and graph metadata; no
    future observations or labels are model inputs.
    """

    CHECKPOINT_FORMAT_VERSION = 1

    def __init__(
        self,
        *,
        gnn_config: Mapping[str, Any],
        transformer_config: Mapping[str, Any],
        graph_node_ids: Sequence[str],
        target_node_ids: Sequence[str],
        forecast_horizon: int,
        target_channels: Sequence[str] = ("power_kw",),
        target_scope: str = "node_level",
        system_pooling: str | None = None,
        head_dropout: float = 0.1,
        lookback_hours: int | None = None,
    ) -> None:
        super().__init__()
        try:
            self.gnn_encoder = GraphMessagePassingEncoder.from_config(gnn_config)
            self.transformer_encoder = TransformerTemporalEncoder.from_config(
                transformer_config
            )
        except (GNNEncoderError, TransformerEncoderError) as exc:
            raise GNNTransformerError(f"Invalid encoder configuration: {exc}") from exc

        if self.gnn_encoder.hidden_dim != self.transformer_encoder.d_model:
            raise GNNTransformerError(
                "GNN hidden_dim must equal Transformer d_model; implicit projections are not used."
            )
        self.graph_node_ids = _ordered_ids(graph_node_ids, "graph_node_ids")
        self.target_node_ids = _ordered_ids(target_node_ids, "target_node_ids")
        unknown = set(self.target_node_ids) - set(self.graph_node_ids)
        if unknown:
            raise GNNTransformerError(
                f"target_node_ids are not present in graph_node_ids: {sorted(unknown)}."
            )
        self.target_node_indices = tuple(
            self.graph_node_ids.index(node) for node in self.target_node_ids
        )
        self.forecast_horizon = _positive_integer(forecast_horizon, "forecast_horizon")
        self.target_channels = _ordered_ids(target_channels, "target_channels")
        if target_scope not in {"node_level", "system_level"}:
            raise GNNTransformerError(
                "target_scope must be 'node_level' or 'system_level'."
            )
        if target_scope == "system_level" and system_pooling not in {"mean", "sum"}:
            raise GNNTransformerError(
                "system_level forecasting requires system_pooling='mean' or 'sum'."
            )
        if target_scope == "node_level" and system_pooling is not None:
            raise GNNTransformerError(
                "system_pooling is only valid for system_level forecasting."
            )
        if (
            isinstance(head_dropout, bool)
            or not isinstance(head_dropout, Real)
            or not np.isfinite(head_dropout)
            or not 0 <= head_dropout < 1
        ):
            raise GNNTransformerError(
                "head_dropout must be finite and in the interval [0, 1)."
            )
        self.target_scope = target_scope
        self.system_pooling = system_pooling
        self.head_dropout_probability = float(head_dropout)
        self.lookback_hours = (
            None
            if lookback_hours is None
            else _positive_integer(lookback_hours, "lookback_hours")
        )
        if (
            self.lookback_hours is not None
            and self.lookback_hours > self.transformer_encoder.max_seq_len
        ):
            raise GNNTransformerError(
                "lookback_hours cannot exceed the Transformer max_seq_len."
            )

        self.head_dropout = nn.Dropout(self.head_dropout_probability)
        self.forecast_projection = nn.Linear(
            self.transformer_encoder.d_model,
            self.forecast_horizon * len(self.target_channels),
        )
        self._configuration = {
            "model": {
                "architecture": "gnn_transformer",
                "spatial_encoder": "graph_mean_message_passing",
                "temporal_encoder": "per_node_transformer",
                "lookback_hours": self.lookback_hours,
                "forecast_horizon_hours": self.forecast_horizon,
                "target_scope": self.target_scope,
                "system_pooling": self.system_pooling,
                "gnn_encoder": deepcopy(dict(gnn_config)),
                "transformer_encoder": deepcopy(dict(transformer_config)),
                "forecast_head": {
                    "type": "direct_multi_step",
                    "dropout": self.head_dropout_probability,
                },
            },
            "graph": {"node_order": list(self.graph_node_ids)},
            "data": {
                "lookback_hours": self.lookback_hours,
                "horizon_hours": self.forecast_horizon,
            },
            "targets": {
                "nodes": list(self.target_node_ids),
                "channels": list(self.target_channels),
            },
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "GNNTransformerForecaster":
        """Create a model from the versioned forecasting_model YAML structure."""
        if not isinstance(config, Mapping):
            raise GNNTransformerError("model configuration must be a mapping.")
        try:
            model_config = config["model"]
            graph_config = config["graph"]
            data_config = config["data"]
            target_config = config["targets"]
            gnn_config = model_config["gnn_encoder"]
            transformer_config = model_config["transformer_encoder"]
            head_config = model_config["forecast_head"]
            graph_nodes = graph_config["node_order"]
            target_nodes = target_config["nodes"]
            target_channels = target_config["channels"]
            forecast_horizon = model_config["forecast_horizon_hours"]
            lookback = model_config["lookback_hours"]
        except (KeyError, TypeError) as exc:
            raise GNNTransformerError(
                f"Forecasting configuration is missing a required field: {exc}."
            ) from exc
        forecast_horizon = _positive_integer(
            forecast_horizon, "model.forecast_horizon_hours"
        )
        if lookback is not None:
            lookback = _positive_integer(lookback, "model.lookback_hours")
        if model_config.get("architecture") != "gnn_transformer":
            raise GNNTransformerError("model.architecture must be gnn_transformer.")
        if model_config.get("spatial_encoder") != "graph_mean_message_passing":
            raise GNNTransformerError("Unsupported configured spatial encoder.")
        if model_config.get("temporal_encoder") != "per_node_transformer":
            raise GNNTransformerError("Unsupported configured temporal encoder.")
        if head_config.get("type") != "direct_multi_step":
            raise GNNTransformerError("Only direct_multi_step forecasting is supported.")
        if lookback != data_config.get("lookback_hours"):
            raise GNNTransformerError(
                "model.lookback_hours must match data.lookback_hours."
            )
        if forecast_horizon != data_config.get("horizon_hours"):
            raise GNNTransformerError(
                "model.forecast_horizon_hours must match data.horizon_hours."
            )
        return cls(
            gnn_config=gnn_config,
            transformer_config=transformer_config,
            graph_node_ids=graph_nodes,
            target_node_ids=target_nodes,
            forecast_horizon=forecast_horizon,
            target_channels=target_channels,
            target_scope=model_config.get("target_scope", "node_level"),
            system_pooling=model_config.get("system_pooling"),
            head_dropout=head_config.get("dropout", 0.0),
            lookback_hours=lookback,
        )

    @property
    def output_node_ids(self) -> tuple[str, ...]:
        if self.target_scope == "system_level":
            return ("system",)
        return self.graph_node_ids

    def _validate_target_mask(
        self,
        target_mask: torch.Tensor | None,
        *,
        batch_size: int,
        node_count: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if target_mask is None:
            return None
        if not isinstance(target_mask, torch.Tensor) or target_mask.dtype != torch.bool:
            raise GNNTransformerError("target_mask must be a boolean torch tensor.")
        if target_mask.device != device:
            raise GNNTransformerError("target_mask and inputs must be on the same device.")
        channels = len(self.target_channels)
        if target_mask.shape == (node_count, 1):
            mask = target_mask.unsqueeze(0).expand(batch_size, -1, -1)
        elif target_mask.shape == (batch_size, node_count, 1):
            mask = target_mask
        elif target_mask.shape == (batch_size, node_count, channels):
            mask = target_mask
        else:
            raise GNNTransformerError(
                "target_mask must have shape [nodes, 1], [batch, nodes, 1], "
                "or [batch, nodes, target_channels]."
            )
        return mask.expand(batch_size, node_count, channels)

    def _last_valid_embeddings(
        self,
        temporal: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, lookback, node_count, embedding_dim = temporal.shape
        per_node_mask = None
        if padding_mask is not None:
            if padding_mask.shape == (batch_size, lookback):
                per_node_mask = padding_mask[:, None, :].expand(
                    batch_size, node_count, lookback
                )
            else:
                per_node_mask = padding_mask
        by_node = temporal.permute(0, 2, 1, 3)
        if per_node_mask is None:
            return by_node[:, :, -1, :]
        positions = torch.arange(lookback, device=temporal.device).view(1, 1, -1)
        last_valid = positions.expand(batch_size, node_count, -1).masked_fill(
            per_node_mask, -1
        ).max(dim=-1).values
        gather_indices = last_valid.view(batch_size, node_count, 1, 1).expand(
            batch_size, node_count, 1, embedding_dim
        )
        return by_node.gather(2, gather_indices).squeeze(2)

    def forward(
        self,
        inputs: torch.Tensor,
        edge_index: torch.Tensor | np.ndarray,
        edge_weight: torch.Tensor | np.ndarray | None = None,
        *,
        target_mask: torch.Tensor | None = None,
        node_order: Sequence[str] | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return direct forecasts without accessing future realized values."""
        if not isinstance(inputs, torch.Tensor) or inputs.ndim != 4:
            raise GNNTransformerError(
                "inputs must be [batch, lookback, nodes, input_features]."
            )
        batch_size, lookback, node_count, feature_count = inputs.shape
        if node_count != len(self.graph_node_ids):
            raise GNNTransformerError(
                f"Expected {len(self.graph_node_ids)} graph nodes, received {node_count}."
            )
        if feature_count != self.gnn_encoder.input_dim:
            raise GNNTransformerError(
                f"Expected {self.gnn_encoder.input_dim} input features, received {feature_count}."
            )
        if self.lookback_hours is not None and lookback != self.lookback_hours:
            raise GNNTransformerError(
                f"Expected lookback {self.lookback_hours}, received {lookback}."
            )
        if node_order is not None and tuple(node_order) != self.graph_node_ids:
            raise GNNTransformerError(
                "Input node order does not match configured graph_node_ids."
            )
        if not inputs.is_floating_point() or inputs.dtype != torch.float32:
            raise GNNTransformerError("inputs must use float32 values.")
        if not torch.isfinite(inputs).all():
            raise GNNTransformerError("inputs must contain only finite values.")
        mask = self._validate_target_mask(
            target_mask,
            batch_size=batch_size,
            node_count=node_count,
            device=inputs.device,
        )
        spatial = self.gnn_encoder(inputs, edge_index, edge_weight)
        temporal = self.transformer_encoder(spatial, padding_mask=padding_mask)
        final_embeddings = self._last_valid_embeddings(temporal, padding_mask)
        selected_embeddings = final_embeddings.index_select(
            1,
            torch.tensor(self.target_node_indices, dtype=torch.long, device=inputs.device),
        )

        if self.target_scope == "system_level":
            selected_mask = None
            if mask is not None:
                selected_mask = mask.index_select(1, torch.tensor(
                    self.target_node_indices, dtype=torch.long, device=inputs.device
                ))
                if selected_mask.shape[-1] != 1:
                    raise GNNTransformerError(
                        "System-level pooling currently requires one shared validity mask per node."
                    )
                selected_mask = selected_mask[..., 0]
            if selected_mask is None:
                pooled = (
                    selected_embeddings.sum(dim=1, keepdim=True)
                    if self.system_pooling == "sum"
                    else selected_embeddings.mean(dim=1, keepdim=True)
                )
            else:
                weights = selected_mask.unsqueeze(-1).to(selected_embeddings.dtype)
                pooled_values = (selected_embeddings * weights).sum(dim=1, keepdim=True)
                if self.system_pooling == "mean":
                    denominator = weights.sum(dim=1, keepdim=True)
                    if torch.any(denominator == 0):
                        raise GNNTransformerError(
                            "System-level pooling requires at least one valid target node per sample."
                        )
                    pooled = pooled_values / denominator
                else:
                    pooled = pooled_values
            forecast = self.forecast_projection(self.head_dropout(pooled))
            return forecast.reshape(
                batch_size,
                1,
                self.forecast_horizon,
                len(self.target_channels),
            ).permute(0, 2, 1, 3)

        node_forecasts = self.forecast_projection(
            self.head_dropout(selected_embeddings)
        ).reshape(
            batch_size,
            len(self.target_node_ids),
            self.forecast_horizon,
            len(self.target_channels),
        ).permute(0, 2, 1, 3)
        node_indices = torch.tensor(
            self.target_node_indices, dtype=torch.long, device=inputs.device
        )
        predictions = inputs.new_zeros(
            batch_size,
            self.forecast_horizon,
            node_count,
            len(self.target_channels),
        ).index_copy(2, node_indices, node_forecasts)
        if mask is not None:
            predictions = predictions * mask[:, None].to(predictions.dtype)
        return predictions

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        overwrite: bool = False,
        training_state: Mapping[str, Any] | None = None,
    ) -> Path:
        """Save weights/config; graph topology and Part 2 preprocessing stay external.

        Inference must supply the matching graph edges, node order, and any
        scaler artifacts used to prepare historical inputs. Existing paths are
        protected unless ``overwrite=True`` is explicitly requested.
        """
        destination = Path(path)
        if type(overwrite) is not bool:
            raise GNNTransformerError("overwrite must be a boolean.")
        if training_state is not None and not isinstance(training_state, Mapping):
            raise GNNTransformerError("training_state must be a mapping when provided.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": self.CHECKPOINT_FORMAT_VERSION,
            "config": deepcopy(self._configuration),
            "state_dict": self.state_dict(),
        }
        if training_state is not None:
            payload["training_state"] = deepcopy(dict(training_state))
        buffer = BytesIO()
        torch.save(payload, buffer)
        mode = "wb" if overwrite else "xb"
        try:
            with destination.open(mode) as stream:
                stream.write(buffer.getvalue())
        except FileExistsError:
            raise FileExistsError(
                f"Refusing to overwrite existing model checkpoint: {destination}"
            ) from None
        except OSError:
            if destination.exists() and not overwrite:
                destination.unlink(missing_ok=True)
            raise
        return destination

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        *,
        map_location: torch.device | str = "cpu",
    ) -> "GNNTransformerForecaster":
        """Restore model configuration and state without loading executable objects."""
        try:
            payload = torch.load(
                Path(path), map_location=map_location, weights_only=True
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise GNNTransformerError(f"Cannot load model checkpoint: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise GNNTransformerError("Checkpoint payload must be a mapping.")
        if payload.get("format_version") != cls.CHECKPOINT_FORMAT_VERSION:
            raise GNNTransformerError("Unsupported model checkpoint format version.")
        config, state_dict = payload.get("config"), payload.get("state_dict")
        if not isinstance(config, Mapping) or not isinstance(state_dict, Mapping):
            raise GNNTransformerError("Checkpoint requires model config and state_dict.")
        model = cls.from_config(config)
        try:
            model.load_state_dict(state_dict, strict=True)
        except (RuntimeError, TypeError) as exc:
            raise GNNTransformerError(f"Checkpoint state is incompatible: {exc}") from exc
        return model.to(map_location)
