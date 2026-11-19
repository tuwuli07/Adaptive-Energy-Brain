"""Adapt node-level forecasts to the simulator's causal observation contract."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
from typing import Literal

import numpy as np
import torch

from src.simulation.state import AgentObservation, ForecastFeature


GRAPH_NODE_ORDER = ("B01", "B02", "B03", "B04", "B05", "PV01", "W01", "BUS01")
TARGET_NODE_ORDER = ("B01", "B02", "B03", "B04", "B05", "PV01", "W01")
TARGET_MASK = (True, True, True, True, True, True, True, False)


class ForecastIntegrationError(ValueError):
    """Invalid forecast data or provenance at the simulator boundary."""


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ForecastIntegrationError(f"{name} must be a nonempty identifier.")
    return value


def _timestamp_utc(value: object, name: str) -> tuple[str, datetime]:
    if not isinstance(value, str):
        raise ForecastIntegrationError(f"{name} must be an hourly UTC timestamp.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ForecastIntegrationError(
            f"{name} must be an hourly UTC timestamp."
        ) from exc
    if (
        parsed.tzinfo is None
        or parsed.utcoffset() != timedelta(0)
        or parsed.minute != 0
        or parsed.second != 0
        or parsed.microsecond != 0
    ):
        raise ForecastIntegrationError(f"{name} must be an hourly UTC timestamp.")
    normalized = parsed.astimezone(timezone.utc)
    return normalized.isoformat(timespec="seconds").replace("+00:00", "Z"), normalized


@dataclass(frozen=True)
class ForecastProvenance:
    """Run and artifact references kept alongside simulator observations."""

    experiment_id: str
    method_id: str
    method_type: Literal["model", "baseline"]
    graph_reference: str
    preprocessing_reference: str
    negative_prediction_policy: Literal["reject", "project_to_zero"] = "reject"

    def __post_init__(self) -> None:
        for name in (
            "experiment_id",
            "method_id",
            "graph_reference",
            "preprocessing_reference",
        ):
            _identifier(getattr(self, name), name)
        if not isinstance(self.method_type, str) or self.method_type not in {
            "model",
            "baseline",
        }:
            raise ForecastIntegrationError(
                "method_type must be 'model' or 'baseline'."
            )
        if (
            not isinstance(self.negative_prediction_policy, str)
            or self.negative_prediction_policy
            not in {"reject", "project_to_zero"}
        ):
            raise ForecastIntegrationError(
                "negative_prediction_policy must be 'reject' or 'project_to_zero'."
            )
        if self.method_type == "model" and (
            self.graph_reference == "not_applicable"
            or self.preprocessing_reference == "not_applicable"
        ):
            raise ForecastIntegrationError(
                "Model forecasts require graph and preprocessing artifact references."
            )


@dataclass(frozen=True)
class ForecastIntegration:
    """Causal observation plus the provenance needed to reproduce its forecasts."""

    observation: AgentObservation
    provenance: ForecastProvenance
    node_order: tuple[str, ...]
    target_mask: tuple[bool, ...]
    target_timestamps_utc: tuple[str, ...]
    units: str = "kW"

    @property
    def horizon_hours(self) -> int:
        return len(self.target_timestamps_utc)


def adapt_node_forecasts(
    observation: AgentObservation,
    predictions_kw: np.ndarray | torch.Tensor,
    *,
    node_order: Sequence[str],
    target_mask: np.ndarray | torch.Tensor,
    target_timestamps_utc: Sequence[str],
    issued_at_utc: str,
    data_cutoff_utc: str,
    provenance: ForecastProvenance,
    units: str = "kW",
) -> ForecastIntegration:
    """Attach ordered [horizon, nodes, 1] forecasts to an existing observation.

    A singleton leading batch dimension is also accepted. Five building
    predictions are summed to the simulator's campus-demand feature; PV01 and
    W01 map to solar and wind features. BUS01 is always excluded by the exact
    target-mask contract. The default negative-value policy is strict rejection.
    ``project_to_zero`` must be explicitly selected in ``provenance`` and
    projects valid predictions at this boundary only. It never mutates the
    supplied predictions or changes evaluation artifacts.
    """
    if not isinstance(observation, AgentObservation):
        raise ForecastIntegrationError("observation must be an AgentObservation.")
    if not isinstance(provenance, ForecastProvenance):
        raise ForecastIntegrationError("provenance must be a ForecastProvenance.")
    if units != "kW":
        raise ForecastIntegrationError("Forecast predictions must use physical kW.")
    if (
        isinstance(node_order, (str, bytes))
        or not isinstance(node_order, Sequence)
        or tuple(node_order) != GRAPH_NODE_ORDER
    ):
        raise ForecastIntegrationError(
            f"node_order must exactly match {GRAPH_NODE_ORDER}."
        )

    if isinstance(target_mask, torch.Tensor):
        mask_array = target_mask.detach().cpu().numpy()
    else:
        mask_array = np.asarray(target_mask)
    if mask_array.dtype != np.bool_:
        raise ForecastIntegrationError("target_mask must contain boolean values.")
    if mask_array.shape == (len(GRAPH_NODE_ORDER), 1):
        mask_array = mask_array[:, 0]
    if mask_array.shape != (len(GRAPH_NODE_ORDER),):
        raise ForecastIntegrationError("target_mask must have shape [8] or [8, 1].")
    if tuple(bool(value) for value in mask_array) != TARGET_MASK:
        raise ForecastIntegrationError(
            "target_mask must include B01-B05, PV01 and W01 and exclude BUS01."
        )

    if isinstance(predictions_kw, torch.Tensor):
        if predictions_kw.is_complex() or not predictions_kw.is_floating_point():
            raise ForecastIntegrationError(
                "Torch forecast predictions must use a real floating-point dtype."
            )
        prediction_array = (
            predictions_kw.detach().to(device="cpu", dtype=torch.float64).numpy()
        )
    else:
        prediction_array = np.asarray(predictions_kw)
    if prediction_array.ndim == 4 and prediction_array.shape[0] == 1:
        prediction_array = prediction_array[0]
    if (
        prediction_array.ndim != 3
        or prediction_array.shape[1:] != (len(GRAPH_NODE_ORDER), 1)
        or prediction_array.shape[0] < 1
    ):
        raise ForecastIntegrationError(
            "predictions_kw must have shape [horizon, 8, 1] or [1, horizon, 8, 1]."
        )
    if not np.issubdtype(prediction_array.dtype, np.number) or np.issubdtype(
        prediction_array.dtype, np.complexfloating
    ):
        raise ForecastIntegrationError("Forecast predictions must be real numeric values.")
    valid_predictions = prediction_array[:, : len(TARGET_NODE_ORDER), 0]
    if not np.isfinite(valid_predictions).all():
        raise ForecastIntegrationError("Valid target predictions must be finite.")
    integration_predictions = prediction_array.copy()
    has_negative_predictions = bool(np.any(valid_predictions < 0))
    if (
        has_negative_predictions
        and provenance.negative_prediction_policy == "reject"
    ):
        raise ForecastIntegrationError(
            "Physical power forecasts contain negative values; select "
            "negative_prediction_policy='project_to_zero' explicitly to project "
            "them at the integration boundary."
        )
    if (
        has_negative_predictions
        and provenance.negative_prediction_policy == "project_to_zero"
    ):
        integration_predictions[:, : len(TARGET_NODE_ORDER), :] = np.maximum(
            integration_predictions[:, : len(TARGET_NODE_ORDER), :], 0
        )

    if (
        isinstance(target_timestamps_utc, (str, bytes))
        or not isinstance(target_timestamps_utc, Sequence)
        or len(target_timestamps_utc) != prediction_array.shape[0]
    ):
        raise ForecastIntegrationError(
            "target_timestamps_utc length must match the forecast horizon."
        )
    normalized_targets = tuple(
        _timestamp_utc(value, "target_timestamps_utc")[0]
        for value in target_timestamps_utc
    )
    target_datetimes = tuple(
        _timestamp_utc(value, "target_timestamps_utc")[1]
        for value in normalized_targets
    )
    decision, decision_datetime = _timestamp_utc(
        observation.decision_timestamp_utc, "decision_timestamp_utc"
    )
    issued, issued_datetime = _timestamp_utc(issued_at_utc, "issued_at_utc")
    cutoff, cutoff_datetime = _timestamp_utc(data_cutoff_utc, "data_cutoff_utc")
    if issued_datetime > decision_datetime:
        raise ForecastIntegrationError("Forecast issue time must not follow the decision.")
    if cutoff_datetime > issued_datetime:
        raise ForecastIntegrationError("Historical data cutoff must not follow issue time.")
    if not target_datetimes or target_datetimes[0] != decision_datetime + timedelta(hours=1):
        raise ForecastIntegrationError(
            "The first forecast target must be the hour immediately after the decision."
        )
    if any(
        current - previous != timedelta(hours=1)
        for previous, current in zip(target_datetimes, target_datetimes[1:])
    ):
        raise ForecastIntegrationError(
            "Forecast target timestamps must be consecutive hourly UTC values."
        )

    features: list[ForecastFeature] = []
    for step, target_timestamp in enumerate(normalized_targets):
        try:
            building_demand = math.fsum(
                float(integration_predictions[step, index, 0]) for index in range(5)
            )
        except OverflowError as exc:
            raise ForecastIntegrationError(
                "Aggregated campus-demand forecast is not finite."
            ) from exc
        if not math.isfinite(building_demand):
            raise ForecastIntegrationError("Aggregated campus-demand forecast is not finite.")
        values = (
            ("campus_demand_kw", building_demand),
            ("solar_generation_kw", float(integration_predictions[step, 5, 0])),
            ("wind_generation_kw", float(integration_predictions[step, 6, 0])),
        )
        features.extend(
            ForecastFeature(
                variable=variable,
                value_kw=value,
                target_timestamp_utc=target_timestamp,
                issued_at_utc=issued,
                data_cutoff_utc=cutoff,
                model_id=f"{provenance.method_id}@{provenance.experiment_id}",
            )
            for variable, value in values
        )

    try:
        adapted_observation = AgentObservation(
            state=observation.state,
            decision_timestamp_utc=decision,
            forecasts=observation.forecasts + tuple(features),
        )
    except ValueError as exc:
        raise ForecastIntegrationError(
            f"Adapted forecasts are incompatible with the observation: {exc}"
        ) from exc

    return ForecastIntegration(
        observation=adapted_observation,
        provenance=provenance,
        node_order=GRAPH_NODE_ORDER,
        target_mask=TARGET_MASK,
        target_timestamps_utc=normalized_targets,
    )
