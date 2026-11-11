"""Causality, alignment, shapes, and physical-unit metrics for forecast baselines."""

import numpy as np
import pytest

from src.forecasting.baselines import (
    ForecastingBaselineError,
    historical_average_forecast,
    persistence_forecast,
    seasonal_persistence_forecast,
)
from src.forecasting.metrics import ForecastingMetricError, forecast_metrics


def test_persistence_repeats_last_value_for_each_horizon_step():
    history = np.array([4.0, 7.0, 9.0])
    np.testing.assert_array_equal(persistence_forecast(history, 4), [9.0] * 4)


def test_persistence_supports_node_level_targets_and_batched_windows():
    history = np.arange(2 * 5 * 3 * 1, dtype=float).reshape(2, 5, 3, 1)
    predictions = persistence_forecast(history, 4, time_axis=1)
    assert predictions.shape == (2, 4, 3, 1)
    np.testing.assert_array_equal(predictions, np.repeat(history[:, -1:, :, :], 4, axis=1))


def test_seasonal_persistence_aligns_each_hour_with_previous_day():
    history = np.arange(48, dtype=float)
    predictions = seasonal_persistence_forecast(history, 24)
    np.testing.assert_array_equal(predictions, np.arange(24, 48))


def test_seasonal_persistence_repeats_last_cycle_for_long_horizon():
    history = np.arange(30, dtype=float).reshape(30, 1)
    predictions = seasonal_persistence_forecast(history, 30, season_period=24)
    np.testing.assert_array_equal(
        predictions[:, 0], np.tile(np.arange(6, 30), 2)[:30]
    )


def test_seasonal_persistence_supports_batched_node_targets():
    history = np.arange(2 * 30 * 4, dtype=float).reshape(2, 30, 4)
    predictions = seasonal_persistence_forecast(history, 5, time_axis=1)
    assert predictions.shape == (2, 5, 4)
    np.testing.assert_array_equal(predictions, history[:, 6:11, :])


def test_seasonal_persistence_rejects_insufficient_history():
    with pytest.raises(ForecastingBaselineError, match="requires at least 24"):
        seasonal_persistence_forecast(np.arange(23), 2)


def test_historical_average_uses_only_provided_history_and_repeats_horizon():
    history = np.array([2.0, 4.0, 9.0])
    predictions = historical_average_forecast(history, 5)
    np.testing.assert_allclose(predictions, np.full(5, 5.0))


def test_historical_average_supports_batched_node_targets():
    history = np.arange(2 * 4 * 3, dtype=float).reshape(2, 4, 3)
    predictions = historical_average_forecast(history, 2, time_axis=1)
    assert predictions.shape == (2, 2, 3)
    np.testing.assert_allclose(predictions, np.repeat(history.mean(axis=1, keepdims=True), 2, axis=1))


@pytest.mark.parametrize(
    ("forecast", "expected", "kwargs"),
    [
        (persistence_forecast, np.array([2.0, 2.0, 2.0]), {}),
        (seasonal_persistence_forecast, np.array([1.0, 2.0, 1.0]), {"season_period": 2}),
        (historical_average_forecast, np.array([1.5, 1.5, 1.5]), {}),
    ],
)
def test_baselines_never_read_future_values(forecast, expected, kwargs):
    observed_and_future = np.array(
        [1.0, 2.0, 1_000_000.0, -1_000_000.0, 9_999_999.0]
    )
    observed_history = observed_and_future[:2]
    future_values = len(observed_and_future) - len(observed_history)
    before = forecast(observed_history, future_values, **kwargs)
    observed_and_future[2:] *= -1
    after = forecast(observed_history, future_values, **kwargs)
    np.testing.assert_array_equal(before, expected)
    np.testing.assert_array_equal(after, expected)


def test_persistence_requires_nonempty_finite_real_history():
    with pytest.raises(ForecastingBaselineError, match="at least one"):
        persistence_forecast(np.array([]), 1)
    with pytest.raises(ForecastingBaselineError, match="finite"):
        historical_average_forecast(np.array([1.0, np.nan]), 1)
    with pytest.raises(ForecastingBaselineError, match="positive integer"):
        persistence_forecast(np.array([1.0]), 0)


def test_forecast_metric_values_match_manual_mae_and_rmse():
    predictions = np.array([2.0, 4.0, 8.0])
    targets = np.array([1.0, 6.0, 5.0])
    metrics = forecast_metrics(predictions, targets)
    assert metrics["mae"] == pytest.approx((1 + 2 + 3) / 3)
    assert metrics["rmse"] == pytest.approx(np.sqrt((1 + 4 + 9) / 3))


def test_forecast_metrics_support_node_masks_and_ignore_masked_placeholders():
    predictions = np.array([[[2.0], [999.0]], [[4.0], [-999.0]]])
    targets = np.array([[[1.0], [0.0]], [[6.0], [0.0]]])
    mask = np.array([[True], [False]])
    metrics = forecast_metrics(predictions, targets, mask=mask)
    assert metrics["mae"] == pytest.approx(1.5)
    assert metrics["rmse"] == pytest.approx(np.sqrt(2.5))


def test_forecast_metrics_reject_incompatible_shapes_and_empty_masks():
    with pytest.raises(ForecastingMetricError, match="does not match"):
        forecast_metrics(np.zeros((2, 3)), np.zeros((2, 2)))
    with pytest.raises(ForecastingMetricError, match="At least one"):
        forecast_metrics(np.zeros(2), np.zeros(2), mask=np.zeros(2, dtype=bool))


def test_forecast_metrics_reject_nonfinite_unmasked_values():
    with pytest.raises(ForecastingMetricError, match="must be finite"):
        forecast_metrics(np.array([np.inf]), np.array([0.0]))


def test_forecasting_baselines_are_independent_of_torch():
    # These deterministic baselines operate on NumPy/array-like values only.
    history = np.array([3.0, 6.0])
    assert persistence_forecast(history, 2).shape == (2,)
