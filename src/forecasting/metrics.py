"""Physical-unit forecasting metrics with optional broadcastable target masks."""

import numpy as np
from numpy.typing import ArrayLike


class ForecastingMetricError(ValueError):
    """Invalid forecast values, target dimensions, or target mask."""


def forecast_metrics(
    predictions: ArrayLike, targets: ArrayLike, *, mask: ArrayLike | None = None
) -> dict[str, float]:
    """Return MAE and RMSE in the input's physical unit (normally kW).

    Predictions and targets must have exactly the same shape. An optional
    boolean mask may be broadcast to that shape; masked entries are excluded
    from both metrics. Every unmasked value must be finite.
    """
    predicted = np.asarray(predictions)
    actual = np.asarray(targets)
    if predicted.shape != actual.shape:
        raise ForecastingMetricError(
            f"Prediction shape {predicted.shape} does not match target shape "
            f"{actual.shape}."
        )
    for name, values in (("predictions", predicted), ("targets", actual)):
        if (
            not np.issubdtype(values.dtype, np.number)
            or np.issubdtype(values.dtype, np.complexfloating)
        ):
            raise ForecastingMetricError(f"{name} must contain real numeric values.")

    if mask is None:
        valid = np.ones(predicted.shape, dtype=bool)
    else:
        supplied_mask = np.asarray(mask)
        if supplied_mask.dtype != np.bool_:
            raise ForecastingMetricError("mask must contain boolean values.")
        try:
            valid = np.broadcast_to(supplied_mask, predicted.shape)
        except ValueError as exc:
            raise ForecastingMetricError(
                f"Mask shape {supplied_mask.shape} cannot broadcast to "
                f"{predicted.shape}."
            ) from exc
    if not valid.any():
        raise ForecastingMetricError("At least one valid target is required.")
    if not np.isfinite(predicted[valid]).all() or not np.isfinite(actual[valid]).all():
        raise ForecastingMetricError("Unmasked predictions and targets must be finite.")
    errors = predicted[valid].astype(np.float64) - actual[valid].astype(np.float64)
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
    }
