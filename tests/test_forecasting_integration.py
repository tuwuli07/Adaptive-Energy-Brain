"""Offline contract tests for attaching forecasts to simulator observations."""

import numpy as np
import pytest

from src.forecasting.baselines import seasonal_persistence_forecast
from src.forecasting.integration import (
    GRAPH_NODE_ORDER,
    ForecastIntegrationError,
    ForecastProvenance,
    adapt_node_forecasts,
)
from src.simulation.actions import BatteryAction
from src.simulation.config import load_simulation_config
from src.simulation.state import ForecastFeature, SimulationState


DECISION = "2026-01-01T01:00:00Z"
TARGETS = ("2026-01-01T02:00:00Z", "2026-01-01T03:00:00Z")
MASK = np.array([[True], [True], [True], [True], [True], [True], [True], [False]])
PROVENANCE = ForecastProvenance(
    experiment_id="exp-001",
    method_id="gnn_transformer",
    method_type="model",
    graph_reference="configs/forecasting_data.yaml#graph",
    preprocessing_reference="split_bundle/preprocessing.json",
)


def observation(forecasts=()):
    state = SimulationState(
        timestamp_utc="2026-01-01T00:00:00Z",
        available_at_utc=DECISION,
        solar_generation_kw=10,
        wind_generation_kw=5,
        campus_demand_kw=300,
        battery_soc=0.5,
        grid_import_price=0.1,
        grid_export_price=None,
        config=load_simulation_config(),
    )
    return state.observation(decision_timestamp_utc=DECISION, forecasts=forecasts)


def predictions():
    values = np.zeros((2, 8, 1), dtype=np.float32)
    values[:, :5, 0] = np.array([[10, 20, 30, 40, 50], [11, 21, 31, 41, 51]])
    values[:, 5, 0] = [7, 8]
    values[:, 6, 0] = [9, 10]
    values[:, 7, 0] = 100_000
    return values


def adapt(**overrides):
    arguments = {
        "observation": observation(),
        "predictions_kw": predictions(),
        "node_order": GRAPH_NODE_ORDER,
        "target_mask": MASK,
        "target_timestamps_utc": TARGETS,
        "issued_at_utc": DECISION,
        "data_cutoff_utc": "2026-01-01T00:00:00Z",
        "provenance": PROVENANCE,
    }
    arguments.update(overrides)
    return adapt_node_forecasts(**arguments)


def test_adapter_aggregates_node_forecasts_and_excludes_masked_bus():
    result = adapt()
    forecast_values = {
        (feature.target_timestamp_utc, feature.variable): feature.value_kw
        for feature in result.observation.forecasts
    }
    assert forecast_values[(TARGETS[0], "campus_demand_kw")] == 150
    assert forecast_values[(TARGETS[0], "solar_generation_kw")] == 7
    assert forecast_values[(TARGETS[0], "wind_generation_kw")] == 9
    assert len(forecast_values) == 6
    assert result.target_mask == (True, True, True, True, True, True, True, False)
    assert result.horizon_hours == 2
    assert result.units == "kW"
    assert result.provenance == PROVENANCE
    assert {feature.model_id for feature in result.observation.forecasts} == {
        "gnn_transformer@exp-001"
    }


def test_adapter_accepts_singleton_model_batch_and_preserves_existing_forecasts():
    prior = ForecastFeature(
        "campus_demand_kw",
        1,
        "2026-01-01T04:00:00Z",
        DECISION,
        DECISION,
        "prior",
    )
    result = adapt(
        observation=observation((prior,)),
        predictions_kw=predictions()[None, ...],
    )
    assert prior in result.observation.forecasts
    assert len(result.observation.forecasts) == 7
    assert isinstance(BatteryAction(0), BatteryAction)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("node_order", tuple(reversed(GRAPH_NODE_ORDER)), "node_order"),
        ("target_mask", np.ones((8, 1), dtype=bool), "exclude BUS01"),
        ("target_mask", np.ones((8, 1), dtype=np.int64), "boolean"),
        ("units", "normalized", "physical kW"),
        ("predictions_kw", np.zeros((2, 7, 1)), "shape"),
        ("predictions_kw", np.zeros((2, 8, 2)), "shape"),
        ("predictions_kw", np.full((2, 8, 1), -1.0), "negative values"),
    ],
)
def test_adapter_rejects_incompatible_shape_order_mask_units_and_values(
    field, value, message
):
    with pytest.raises(ForecastIntegrationError, match=message):
        adapt(**{field: value})


def test_adapter_rejects_nonfinite_valid_targets_but_ignores_masked_bus_value():
    values = predictions()
    values[0, 0, 0] = np.nan
    with pytest.raises(ForecastIntegrationError, match="finite"):
        adapt(predictions_kw=values)
    values = predictions()
    values[0, 7, 0] = np.nan
    assert adapt(predictions_kw=values).observation.forecasts


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("issued_at_utc", "2026-01-01T02:00:00Z", "issue time"),
        ("data_cutoff_utc", "2026-01-01T02:00:00Z", "cutoff"),
        ("target_timestamps_utc", ("2026-01-01T01:00:00Z", TARGETS[1]), "immediately"),
        (
            "target_timestamps_utc",
            (TARGETS[0], "2026-01-01T04:00:00Z"),
            "consecutive",
        ),
        ("target_timestamps_utc", ("2026-01-01T02:00:00", TARGETS[1]), "hourly UTC"),
        ("target_timestamps_utc", (TARGETS[0],), "length"),
    ],
)
def test_adapter_rejects_future_data_or_misaligned_timestamps(field, value, message):
    with pytest.raises(ForecastIntegrationError, match=message):
        adapt(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("experiment_id", ""),
        ("method_id", " "),
        ("graph_reference", ""),
        ("preprocessing_reference", "not_applicable"),
    ],
)
def test_model_provenance_requires_explicit_identifiers_and_artifact_references(
    field, value
):
    arguments = {
        "experiment_id": "exp",
        "method_id": "model",
        "method_type": "model",
        "graph_reference": "graph.json",
        "preprocessing_reference": "preprocessing.json",
    }
    arguments[field] = value
    with pytest.raises(ForecastIntegrationError):
        ForecastProvenance(**arguments)


@pytest.mark.parametrize("policy", ["clip", "", None, []])
def test_provenance_rejects_unsupported_negative_prediction_policy(policy):
    with pytest.raises(ForecastIntegrationError, match="negative_prediction_policy"):
        ForecastProvenance(
            experiment_id="exp",
            method_id="gnn_transformer",
            method_type="model",
            graph_reference="graph.json",
            preprocessing_reference="preprocessing.json",
            negative_prediction_policy=policy,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("node_order", None, "node_order"),
        ("target_mask", None, "boolean"),
        ("target_timestamps_utc", None, "length"),
        ("issued_at_utc", None, "hourly UTC"),
        ("provenance", None, "ForecastProvenance"),
    ],
)
def test_adapter_rejects_missing_integration_metadata(field, value, message):
    with pytest.raises(ForecastIntegrationError, match=message):
        adapt(**{field: value})


def test_baseline_provenance_may_explicitly_mark_unused_artifacts():
    baseline = ForecastProvenance(
        experiment_id="baseline-run",
        method_id="seasonal_persistence",
        method_type="baseline",
        graph_reference="not_applicable",
        preprocessing_reference="not_applicable",
    )
    result = adapt(provenance=baseline)
    assert result.provenance.method_type == "baseline"
    assert {feature.model_id for feature in result.observation.forecasts} == {
        "seasonal_persistence@baseline-run"
    }


def test_forecasting_methods_with_shared_experiment_keep_distinct_feature_ids():
    seasonal = ForecastProvenance(
        experiment_id=PROVENANCE.experiment_id,
        method_id="seasonal_persistence",
        method_type="baseline",
        graph_reference="not_applicable",
        preprocessing_reference="not_applicable",
    )
    model_result = adapt()
    seasonal_result = adapt(provenance=seasonal)
    model_ids = {feature.model_id for feature in model_result.observation.forecasts}
    seasonal_ids = {
        feature.model_id for feature in seasonal_result.observation.forecasts
    }
    assert model_ids == {"gnn_transformer@exp-001"}
    assert seasonal_ids == {"seasonal_persistence@exp-001"}
    assert model_ids.isdisjoint(seasonal_ids)
    assert model_result.provenance.experiment_id == seasonal_result.provenance.experiment_id


def test_seasonal_persistence_output_is_accepted_by_the_same_adapter():
    history = np.arange(24 * 8, dtype=np.float64).reshape(24, 8, 1)
    history[:, 7, :] = 0
    seasonal_predictions = seasonal_persistence_forecast(
        history, horizon=2, time_axis=0
    )
    seasonal = ForecastProvenance(
        experiment_id=PROVENANCE.experiment_id,
        method_id="seasonal_persistence",
        method_type="baseline",
        graph_reference="not_applicable",
        preprocessing_reference="not_applicable",
    )
    result = adapt(
        predictions_kw=seasonal_predictions,
        provenance=seasonal,
    )
    assert result.horizon_hours == 2
    assert {feature.model_id for feature in result.observation.forecasts} == {
        "seasonal_persistence@exp-001"
    }


def test_explicit_projection_is_recorded_and_does_not_mutate_model_predictions():
    values = predictions()
    values[0, 0, 0] = -10
    values[0, 1, 0] = -20
    values[0, 5, 0] = -7
    values[0, 6, 0] = -9
    original = values.copy()
    projected_provenance = ForecastProvenance(
        experiment_id=PROVENANCE.experiment_id,
        method_id=PROVENANCE.method_id,
        method_type=PROVENANCE.method_type,
        graph_reference=PROVENANCE.graph_reference,
        preprocessing_reference=PROVENANCE.preprocessing_reference,
        negative_prediction_policy="project_to_zero",
    )
    result = adapt(
        predictions_kw=values,
        provenance=projected_provenance,
    )
    by_variable = {
        feature.variable: feature.value_kw
        for feature in result.observation.forecasts
        if feature.target_timestamp_utc == TARGETS[0]
    }
    assert by_variable == {
        "campus_demand_kw": 120,
        "solar_generation_kw": 0,
        "wind_generation_kw": 0,
    }
    assert result.provenance.negative_prediction_policy == "project_to_zero"
    np.testing.assert_array_equal(values, original)
    assert values[0, 0, 0] == -10
    assert result.provenance.experiment_id == PROVENANCE.experiment_id
    assert result.provenance.method_id == PROVENANCE.method_id


def test_projection_policy_still_rejects_nonfinite_valid_predictions():
    values = predictions()
    values[0, 0, 0] = np.nan
    projected_provenance = ForecastProvenance(
        experiment_id="exp",
        method_id="gnn_transformer",
        method_type="model",
        graph_reference="graph.json",
        preprocessing_reference="preprocessing.json",
        negative_prediction_policy="project_to_zero",
    )
    with pytest.raises(ForecastIntegrationError, match="finite"):
        adapt(predictions_kw=values, provenance=projected_provenance)


def test_adapter_rejects_duplicate_forecast_features_for_same_target():
    prior = ForecastFeature(
        "campus_demand_kw",
        1,
        TARGETS[0],
        DECISION,
        DECISION,
        "other_method",
    )
    with pytest.raises(ForecastIntegrationError, match="Duplicate forecast"):
        adapt(observation=observation((prior,)))


def test_adapter_returns_observation_accepted_by_existing_controller_contract():
    from src.baselines.zero_action import ZeroActionController

    result = adapt()
    action = ZeroActionController().select_action(result.observation)
    assert action == BatteryAction(0)
