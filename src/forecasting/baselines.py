"""Causal, dependency-light forecasting baselines for physical power histories.

Inputs contain only observations available at the forecast origin. The time
axis is replaced by the requested forecast horizon; every other dimension
(batch, node, target) is preserved. Values and returned predictions use the
same physical unit, normally kW.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray


class ForecastingBaselineError(ValueError):
    """Invalid history, forecast horizon, or seasonal context."""


def _validate_history(
    history: ArrayLike, time_axis: int
) -> tuple[NDArray[np.float64], int]:
    values = np.asarray(history)
    if values.ndim == 0:
        raise ForecastingBaselineError("history must include a time dimension.")
    if not np.issubdtype(values.dtype, np.number) or np.issubdtype(
        values.dtype, np.complexfloating
    ):
        raise ForecastingBaselineError("history must contain real numeric values.")
    if not np.isfinite(values).all():
        raise ForecastingBaselineError("history must contain only finite observations.")
    if isinstance(time_axis, bool) or not isinstance(time_axis, (int, np.integer)):
        raise ForecastingBaselineError("time_axis must be an integer dimension.")
    axis = int(time_axis)
    if axis < 0:
        axis += values.ndim
    if axis < 0 or axis >= values.ndim:
        raise ForecastingBaselineError(
            f"time_axis {time_axis} is invalid for {values.ndim}-D history."
        )
    if values.shape[axis] < 1:
        raise ForecastingBaselineError("history must contain at least one observation.")
    return values.astype(np.float64, copy=False), axis


def _validate_horizon(horizon: int) -> int:
    if (
        isinstance(horizon, bool)
        or not isinstance(horizon, (int, np.integer))
        or horizon < 1
    ):
        raise ForecastingBaselineError("horizon must be a positive integer.")
    return int(horizon)


def persistence_forecast(
    history: ArrayLike, horizon: int, *, time_axis: int = 0
) -> NDArray[np.float64]:
    """Repeat the last observed value for every step of the forecast horizon."""
    values, axis = _validate_history(history, time_axis)
    steps = _validate_horizon(horizon)
    last = np.take(values, [-1], axis=axis)
    return np.repeat(last, steps, axis=axis)


def seasonal_persistence_forecast(
    history: ArrayLike,
    horizon: int,
    *,
    time_axis: int = 0,
    season_period: int = 24,
) -> NDArray[np.float64]:
    """Repeat the latest complete seasonal cycle for the requested horizon.

    For hourly data the default period is 24, so each forecast hour uses the
    observation from the corresponding hour of the latest complete day. A
    complete cycle is required; incomplete context is rejected explicitly.
    """
    values, axis = _validate_history(history, time_axis)
    steps = _validate_horizon(horizon)
    if (
        isinstance(season_period, bool)
        or not isinstance(season_period, (int, np.integer))
        or season_period < 1
    ):
        raise ForecastingBaselineError("season_period must be a positive integer.")
    period = int(season_period)
    observed = values.shape[axis]
    if observed < period:
        raise ForecastingBaselineError(
            f"Seasonal persistence requires at least {period} observations; "
            f"received {observed}."
        )
    indices = observed - period + (np.arange(steps) % period)
    return np.take(values, indices, axis=axis)


def historical_average_forecast(
    history: ArrayLike, horizon: int, *, time_axis: int = 0
) -> NDArray[np.float64]:
    """Repeat the mean of observations supplied from before the forecast origin."""
    values, axis = _validate_history(history, time_axis)
    steps = _validate_horizon(horizon)
    average = np.mean(values, axis=axis, keepdims=True)
    return np.repeat(average, steps, axis=axis)
