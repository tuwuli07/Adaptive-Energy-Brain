"""Grid exchange limits, explicit tariff policies and atomic transaction ledger."""

from dataclasses import FrozenInstanceError, asdict, replace
import random

import pytest
import yaml

from src.simulation.config import load_simulation_config
from src.simulation.grid import Grid, GridError


def configured(**changes):
    config = load_simulation_config()
    return replace(config, grid=replace(config.grid, **changes))


def test_import_cost_and_export_revenue_and_ledger():
    grid = Grid()
    imported = grid.import_power(150, import_price=0.12)
    exported = grid.export_power(100, export_price=0.04)
    assert imported.actual_power_kw == 150
    assert imported.imported_energy_kwh == 150
    assert imported.import_cost == 18
    assert imported.export_revenue == 0
    assert exported.actual_power_kw == -100
    assert exported.exported_energy_kwh == 100
    assert exported.export_revenue == 4
    assert exported.net_cost == -4
    assert grid.state.imported_energy_kwh == 150
    assert grid.state.exported_energy_kwh == 100
    assert grid.state.net_cost == 14
    assert grid.transactions == (imported, exported)
    assert imported.currency_code == "USD"
    assert imported.price_unit == "USD/kWh"


@pytest.mark.parametrize("power,actual,reason", [
    (2500, 2000, "import_power_limit"), (-600, -500, "export_power_limit"),
])
def test_connection_limits_report_requested_actual_and_unfulfilled(power, actual, reason):
    result = Grid().step(power, import_price=0.12, export_price=0.04)
    assert result.requested_power_kw == power
    assert result.actual_power_kw == actual
    assert result.unfulfilled_power_kw == power - actual
    assert result.power_limited and reason in result.limiting_reasons


def test_zero_limits_prevent_flows_without_requiring_unused_tariffs():
    grid = Grid(configured(import_limit_kw=0, export_limit_kw=0))
    assert grid.import_power(100).actual_power_kw == 0
    assert grid.export_power(100).actual_power_kw == 0
    assert grid.state.imported_energy_kwh == grid.state.exported_energy_kwh == 0
    assert grid.state.net_cost == 0


def test_idle_step_is_recorded_with_zero_cost_and_no_price_assumptions():
    grid = Grid()
    result = grid.step()
    assert result.actual_power_kw == 0
    assert result.import_price is result.provided_export_price is result.effective_export_price is None
    assert not result.export_price_assumed_zero
    assert result.import_cost == result.export_revenue == result.net_cost == 0
    assert grid.state.elapsed_hours == grid.state.step_count == len(grid.transactions) == 1


def test_signed_and_separate_requests_agree():
    first, second = Grid(), Grid()
    assert first.step(100, import_price=0.1).state_after == second.step(import_kw=100, import_price=0.1).state_after
    assert first.step(-50, export_price=0.02).state_after == second.step(export_kw=50, export_price=0.02).state_after


def test_simultaneous_negative_and_mixed_requests_fail_atomically():
    grid = Grid()
    before = grid.state
    with pytest.raises(GridError, match="Simultaneous"):
        grid.step(import_kw=10, export_kw=10, import_price=0.1, export_price=0.1)
    with pytest.raises(GridError, match="Choose either"):
        grid.step(0, import_kw=10)
    with pytest.raises(GridError, match="nonnegative"):
        grid.import_power(-10)
    assert grid.state == before and not grid.transactions


@pytest.mark.parametrize("value", [True, "0.1", float("nan"), float("inf"), -float("inf")])
def test_invalid_provided_prices_fail_even_in_unused_direction(value):
    grid = Grid()
    before = grid.state
    with pytest.raises(GridError, match="export_price"):
        grid.step(0, export_price=value)
    assert grid.state == before and not grid.transactions


def test_missing_import_tariff_only_required_when_importing():
    grid = Grid()
    with pytest.raises(GridError, match="import_price is required"):
        grid.import_power(100)
    assert not grid.transactions
    assert grid.export_power(100, export_price=0.04).export_revenue == 4


def test_missing_export_price_error_policy_is_explicit_default():
    grid = Grid()
    before = grid.state
    with pytest.raises(GridError, match="configured error policy"):
        grid.export_power(100)
    assert grid.state == before and not grid.transactions


def test_explicit_zero_compensation_records_assumed_tariff():
    grid = Grid(configured(missing_export_price_handling="zero_compensation"))
    result = grid.export_power(100)
    assert result.actual_power_kw == -100
    assert result.provided_export_price is None
    assert result.effective_export_price == result.export_revenue == 0
    assert result.export_price_assumed_zero
    assert result.metadata["source_metadata"]["missing_export_price_handling"] == "zero_compensation"


def test_missing_export_price_can_explicitly_disable_export():
    grid = Grid(configured(missing_export_price_handling="disable_export"))
    result = grid.export_power(100)
    assert result.actual_power_kw == result.exported_energy_kwh == result.net_cost == 0
    assert result.unfulfilled_power_kw == -100
    assert result.effective_export_price is None
    assert not result.export_price_assumed_zero
    assert "missing_export_price_disables_export" in result.limiting_reasons


def test_negative_prices_rejected_by_default_without_clipping():
    grid = Grid()
    with pytest.raises(GridError, match="allow_negative_prices"):
        grid.import_power(100, import_price=-0.02)
    assert not grid.transactions


def test_negative_market_prices_require_opt_in_and_preserve_signs():
    grid = Grid(configured(allow_negative_prices=True))
    assert grid.import_power(100, import_price=-0.02).import_cost == -2
    export = grid.export_power(50, export_price=-0.01)
    assert export.export_revenue == -0.5
    assert export.net_cost == 0.5
    assert grid.state.net_cost == -1.5


@pytest.mark.parametrize("dt", [0.25, 0.5, 2])
def test_energy_and_cost_scale_with_duration(dt):
    result = Grid().import_power(100, import_price=0.12, time_step_hours=dt)
    assert result.imported_energy_kwh == 100 * dt
    assert result.import_cost == 12 * dt
    assert result.state_after.elapsed_hours == dt


@pytest.mark.parametrize("dt", [0, -1, True, "1", float("nan"), float("inf")])
def test_invalid_duration_does_not_commit(dt):
    grid = Grid()
    before = grid.state
    with pytest.raises(GridError, match="time_step_hours"):
        grid.step(0, time_step_hours=dt)
    assert grid.state == before and not grid.transactions


def test_currency_and_price_energy_units_are_explicit():
    grid = Grid()
    with pytest.raises(GridError, match="no FX conversion"):
        grid.import_power(100, import_price=0.1, currency_code="EUR")
    with pytest.raises(GridError, match="convert explicitly upstream"):
        grid.import_power(100, import_price=100, price_unit="USD/MWh")
    assert not grid.transactions
    euro = Grid(configured(currency_code="EUR"))
    assert euro.import_power(100, import_price=0.1, price_unit="EUR/kWh").currency_code == "EUR"


def test_hourly_prices_and_timestamp_order_are_preserved():
    grid = Grid(start_timestamp_utc="2026-01-01T00:00:00Z")
    assert grid.import_power(100, import_price=0.1).timestamp_utc == "2026-01-01T00:00:00Z"
    assert grid.import_power(100, import_price=0.2).timestamp_utc == "2026-01-01T01:00:00Z"
    assert grid.state.import_cost == 30
    before = grid.state
    with pytest.raises(GridError, match="does not follow"):
        grid.step(0, timestamp_utc="2026-01-01T01:00:00Z")
    assert grid.state == before


def test_optional_clock_can_start_with_first_transaction():
    grid = Grid()
    first = grid.step(0, timestamp_utc="2026-01-01T00:00:00+00:00", time_step_hours=0.5)
    assert first.state_after.next_timestamp_utc == "2026-01-01T00:30:00Z"
    assert grid.step(0, time_step_hours=0.5).timestamp_utc == "2026-01-01T00:30:00Z"
    unanchored = Grid()
    unanchored.step()
    with pytest.raises(GridError, match="mid-episode"):
        unanchored.step(timestamp_utc="2026-01-01T01:00:00Z")


def test_preview_commit_rejects_altered_stale_or_reset_transactions():
    grid = Grid()
    candidate = grid.preview(100, import_price=0.1)
    assert not grid.transactions and grid.state.step_count == 0
    with pytest.raises(GridError, match="Altered"):
        grid.commit(replace(candidate, import_cost=0))
    with pytest.raises(GridError, match="malformed"):
        grid.commit(replace(candidate, _metadata={}))
    grid.commit(candidate)
    with pytest.raises(GridError, match="Stale"):
        grid.commit(candidate)
    pending = grid.preview(0)
    grid.reset()
    with pytest.raises(GridError, match="Stale"):
        grid.commit(pending)
    assert not grid.transactions and grid.state.step_count == 0


def test_provenance_is_preserved_and_transactions_immutable():
    provenance = {"source_id": "fixture_price", "origin": "simulated", "synthetic": True,
                  "source_metadata": {"note": "Unit-test tariff only"}}
    grid = Grid()
    result = grid.import_power(100, import_price=0.1, price_provenance=provenance)
    provenance["origin"] = "measured"
    assert result.metadata["source_metadata"]["price_provenance"]["origin"] == "simulated"
    with pytest.raises(FrozenInstanceError):
        result.import_cost = 0
    with pytest.raises(FrozenInstanceError):
        grid.state.net_cost = 0
    assert isinstance(grid.transactions, tuple)


@pytest.mark.parametrize("field,value", [
    ("currency_code", "usd"), ("currency_code", "US"),
    ("missing_export_price_handling", "guess"), ("allow_negative_prices", "false"),
])
def test_grid_configuration_policies_are_validated(field, value):
    with pytest.raises(GridError, match=field):
        Grid(configured(**{field: value}))


def test_configuration_file_can_select_currency_policy_and_duration(tmp_path):
    config = asdict(load_simulation_config())
    config["time_step_hours"] = 0.5
    config["grid"].update(currency_code="BDT", missing_export_price_handling="zero_compensation")
    path = tmp_path / "simulation.yaml"
    path.write_text(yaml.safe_dump(config))
    result = Grid.from_config(path).export_power(100)
    assert result.exported_energy_kwh == 50
    assert result.currency_code == "BDT" and result.export_price_assumed_zero


def test_numerical_overflow_is_atomic():
    grid = Grid()
    before = grid.state
    with pytest.raises(GridError, match="finite"):
        grid.import_power(2000, import_price=1e308)
    assert grid.state == before and not grid.transactions


def test_many_transactions_have_consistent_energy_cost_and_counters():
    grid = Grid()
    rng = random.Random(42)
    for _ in range(1000):
        result = grid.step(rng.uniform(-1000, 3000), import_price=0.12, export_price=0.04,
                           time_step_hours=rng.choice([0.25, 0.5, 1, 2]))
        assert 0 <= result.import_power_kw <= 2000
        assert 0 <= result.export_power_kw <= 500
        assert not (result.import_power_kw > 0 and result.export_power_kw > 0)
        assert result.imported_energy_kwh == result.import_power_kw * result.time_step_hours
        assert result.exported_energy_kwh == result.export_power_kw * result.time_step_hours
        assert result.net_cost == pytest.approx(result.imported_energy_kwh * 0.12 - result.exported_energy_kwh * 0.04)
    assert len(grid.transactions) == grid.state.step_count == 1000
    assert grid.state.net_cost == pytest.approx(sum(item.net_cost for item in grid.transactions))
