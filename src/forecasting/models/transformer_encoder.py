"""Per-node temporal Transformer encoder with sinusoidal positional encoding."""

from collections.abc import Mapping
from numbers import Integral, Real
from typing import Any

import numpy as np
import torch
from torch import nn


class TransformerEncoderError(ValueError):
    """Invalid Transformer configuration, embeddings, or padding mask."""


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise TransformerEncoderError(f"{name} must be a positive integer.")
    return int(value)


class SinusoidalPositionalEncoding(nn.Module):
    """Add fixed sinusoidal encodings for positions within the historical window."""

    def __init__(self, d_model: int, max_seq_len: int) -> None:
        super().__init__()
        d_model = _positive_integer(d_model, "d_model")
        max_seq_len = _positive_integer(max_seq_len, "max_seq_len")
        positions = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-np.log(10000.0) / d_model)
        )
        encodings = torch.zeros(max_seq_len, d_model, dtype=torch.float32)
        angles = positions * frequencies.unsqueeze(0)
        encodings[:, 0::2] = torch.sin(angles)
        if d_model > 1:
            encodings[:, 1::2] = torch.cos(angles[:, : encodings[:, 1::2].shape[1]])
        self.register_buffer("encodings", encodings.unsqueeze(0), persistent=True)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.ndim != 3:
            raise TransformerEncoderError(
                "Positional encoding expects [sequence_batch, time, d_model]."
            )
        if sequence.shape[1] > self.encodings.shape[1]:
            raise TransformerEncoderError(
                f"Sequence length {sequence.shape[1]} exceeds configured maximum "
                f"{self.encodings.shape[1]}."
            )
        if sequence.shape[2] != self.encodings.shape[2]:
            raise TransformerEncoderError(
                f"Expected embedding dimension {self.encodings.shape[2]}, "
                f"received {sequence.shape[2]}."
            )
        return sequence + self.encodings[:, : sequence.shape[1]].to(
            device=sequence.device, dtype=sequence.dtype
        )


class TransformerTemporalEncoder(nn.Module):
    """Apply self-attention over time independently for each node.

    Input and output have shape ``[batch, lookback, nodes, d_model]``. The
    batch and node dimensions are folded into Transformer batch entries, so
    attention never mixes samples or nodes. By default, attention is
    bidirectional within the supplied historical window; this is valid for
    direct multi-step forecasting because the entire window predates forecast
    issuance. No future targets, timestamps, or covariates are accepted.

    Optional ``padding_mask`` uses PyTorch semantics (``True`` means padding)
    and accepts ``[batch, lookback]`` or ``[batch, nodes, lookback]``. Padded
    positions are excluded as keys and their output embeddings are set to zero.
    All-padding node sequences are rejected.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        activation: str = "relu",
        max_seq_len: int = 168,
        positional_encoding: str = "sinusoidal",
    ) -> None:
        super().__init__()
        self.d_model = _positive_integer(d_model, "d_model")
        self.num_heads = _positive_integer(num_heads, "num_heads")
        self.num_layers = _positive_integer(num_layers, "num_layers")
        self.dim_feedforward = _positive_integer(dim_feedforward, "dim_feedforward")
        self.max_seq_len = _positive_integer(max_seq_len, "max_seq_len")
        if self.d_model % self.num_heads:
            raise TransformerEncoderError(
                "d_model must be divisible by num_heads."
            )
        if (
            isinstance(dropout, bool)
            or not isinstance(dropout, Real)
            or not np.isfinite(dropout)
            or not 0 <= dropout < 1
        ):
            raise TransformerEncoderError(
                "dropout must be finite and in the interval [0, 1)."
            )
        if activation not in {"relu", "gelu"}:
            raise TransformerEncoderError("activation must be either 'relu' or 'gelu'.")
        if positional_encoding != "sinusoidal":
            raise TransformerEncoderError(
                "positional_encoding must be 'sinusoidal'."
            )
        self.dropout_probability = float(dropout)
        self.activation_name = activation
        self.positional_encoding_name = positional_encoding
        self.positional_encoding = SinusoidalPositionalEncoding(
            self.d_model, self.max_seq_len
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout_probability,
            activation=self.activation_name,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=self.num_layers,
            enable_nested_tensor=False,
        )

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "TransformerTemporalEncoder":
        """Construct from the ``model.transformer_encoder`` YAML mapping."""
        if not isinstance(config, Mapping):
            raise TransformerEncoderError(
                "Transformer encoder configuration must be a mapping."
            )
        expected = {
            "d_model",
            "num_heads",
            "num_layers",
            "dim_feedforward",
            "dropout",
            "activation",
            "max_seq_len",
            "positional_encoding",
            "attention_strategy",
            "padding_mask_convention",
        }
        missing = expected - config.keys()
        if missing:
            raise TransformerEncoderError(
                f"Transformer configuration is missing {sorted(missing)}."
            )
        if config["attention_strategy"] != "bidirectional_within_historical_window":
            raise TransformerEncoderError(
                "Only bidirectional attention within the historical window is supported."
            )
        if config["padding_mask_convention"] != "true_means_padding":
            raise TransformerEncoderError("Unsupported padding mask convention.")
        return cls(
            d_model=config["d_model"],
            num_heads=config["num_heads"],
            num_layers=config["num_layers"],
            dim_feedforward=config["dim_feedforward"],
            dropout=config["dropout"],
            activation=config["activation"],
            max_seq_len=config["max_seq_len"],
            positional_encoding=config["positional_encoding"],
        )

    def _validate_padding_mask(
        self,
        padding_mask: torch.Tensor | None,
        *,
        batch_size: int,
        lookback: int,
        node_count: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if padding_mask is None:
            return None
        if not isinstance(padding_mask, torch.Tensor) or padding_mask.dtype != torch.bool:
            raise TransformerEncoderError("padding_mask must be a boolean torch tensor.")
        if padding_mask.device != device:
            raise TransformerEncoderError(
                "padding_mask and embeddings must be on the same device."
            )
        if padding_mask.shape == (batch_size, lookback):
            per_node_mask = padding_mask[:, None, :].expand(
                batch_size, node_count, lookback
            )
        elif padding_mask.shape == (batch_size, node_count, lookback):
            per_node_mask = padding_mask
        else:
            raise TransformerEncoderError(
                "padding_mask must have shape [batch, lookback] or "
                "[batch, nodes, lookback]."
            )
        flattened_mask = per_node_mask.reshape(batch_size * node_count, lookback)
        if flattened_mask.all(dim=1).any():
            raise TransformerEncoderError(
                "Each batch/node sequence must include at least one unpadded position."
            )
        return flattened_mask

    def forward(
        self,
        spatial_embeddings: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not isinstance(spatial_embeddings, torch.Tensor) or spatial_embeddings.ndim != 4:
            raise TransformerEncoderError(
                "spatial_embeddings must be a tensor [batch, lookback, nodes, d_model]."
            )
        if (
            not spatial_embeddings.is_floating_point()
            or spatial_embeddings.dtype != torch.float32
        ):
            raise TransformerEncoderError("spatial_embeddings must use float32 values.")
        if any(size < 1 for size in spatial_embeddings.shape):
            raise TransformerEncoderError(
                "batch, lookback, node, and embedding dimensions must be nonempty."
            )
        batch_size, lookback, node_count, embedding_dim = spatial_embeddings.shape
        if embedding_dim != self.d_model:
            raise TransformerEncoderError(
                f"Expected embedding dimension {self.d_model}, received {embedding_dim}."
            )
        if lookback > self.max_seq_len:
            raise TransformerEncoderError(
                f"Sequence length {lookback} exceeds configured maximum "
                f"{self.max_seq_len}."
            )
        if not torch.isfinite(spatial_embeddings).all():
            raise TransformerEncoderError(
                "spatial_embeddings must contain only finite values."
            )
        parameter = next(self.parameters())
        if spatial_embeddings.device != parameter.device:
            raise TransformerEncoderError(
                f"embeddings are on {spatial_embeddings.device}, but encoder parameters "
                f"are on {parameter.device}."
            )
        flattened_mask = self._validate_padding_mask(
            padding_mask,
            batch_size=batch_size,
            lookback=lookback,
            node_count=node_count,
            device=spatial_embeddings.device,
        )
        # Fold batch and node axes; each Transformer sequence is one node's history.
        sequences = spatial_embeddings.permute(0, 2, 1, 3).reshape(
            batch_size * node_count, lookback, self.d_model
        )
        encoded = self.encoder(
            self.positional_encoding(sequences),
            src_key_padding_mask=flattened_mask,
        )
        if flattened_mask is not None:
            encoded = encoded.masked_fill(flattened_mask.unsqueeze(-1), 0.0)
        return encoded.reshape(batch_size, node_count, lookback, self.d_model).permute(
            0, 2, 1, 3
        )
