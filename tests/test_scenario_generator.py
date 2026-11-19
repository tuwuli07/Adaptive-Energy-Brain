"""Unit tests for the CausalScenarioGenerator."""

import pytest
import numpy as np

from src.control.scenario_generator import CausalScenarioGenerator, UncertaintyScenario


def test_scenario_generator_probabilities_sum_to_one():
    gen = CausalScenarioGenerator(seed=42)
    h = 24
    scenarios = gen.generate_scenarios(
        nominal_demand_kw=[250.0] * h,
        nominal_solar_kw=[80.0] * h,
        nominal_wind_kw=[20.0] * h,
        scheduled_import_price=[0.12] * h,
        scheduled_export_price=[0.04] * h,
        num_scenarios=5,
    )

    assert len(scenarios) == 5
    tot_prob = sum(s.probability for s in scenarios)
    assert tot_prob == pytest.approx(1.0, abs=1e-7)

    for s in scenarios:
        assert 0.0 < s.probability <= 1.0
        assert s.horizon_hours == 24


def test_scenario_generator_reproducibility():
    gen1 = CausalScenarioGenerator(seed=123)
    gen2 = CausalScenarioGenerator(seed=123)
    h = 24

    args = {
        "nominal_demand_kw": [200.0 + i for i in range(h)],
        "nominal_solar_kw": [max(0.0, 100.0 - 5 * abs(i - 12)) for i in range(h)],
        "nominal_wind_kw": [15.0] * h,
        "scheduled_import_price": [0.12 if i < 11 or i > 19 else 0.18 for i in range(h)],
        "scheduled_export_price": [0.04] * h,
        "num_scenarios": 5,
    }

    sc1 = gen1.generate_scenarios(**args)
    sc2 = gen2.generate_scenarios(**args)

    for s1, s2 in zip(sc1, sc2):
        assert s1.name == s2.name
        assert s1.probability == pytest.approx(s2.probability)
        assert s1.demand_kw == s2.demand_kw
        assert s1.solar_generation_kw == s2.solar_generation_kw
        assert s1.import_price == s2.import_price


def test_scenario_physical_bounds():
    gen = CausalScenarioGenerator(seed=42, pv_capacity_kw=300.0, wind_capacity_kw=150.0)
    h = 24

    scenarios = gen.generate_scenarios(
        nominal_demand_kw=[500.0] * h,
        nominal_solar_kw=[400.0] * h,  # exceeds 300 kW capacity
        nominal_wind_kw=[200.0] * h,   # exceeds 150 kW capacity
        scheduled_import_price=[0.18] * h,
        scheduled_export_price=[0.04] * h,
        num_scenarios=5,
    )

    for s in scenarios:
        assert all(0.0 <= val <= 300.0 for val in s.solar_generation_kw)
        assert all(0.0 <= val <= 150.0 for val in s.wind_generation_kw)
        assert all(val >= 0.0 for val in s.demand_kw)
