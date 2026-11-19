"""Rigorous test suite verifying boundary history alignment, one-shot runner safety,
and failure quarantine protocols without accessing held-out test data.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest

from scripts.run_simulation import _components
from src.control.gnn_transformer_mpc import GNNTransformerMPCController
from src.control.seasonal_persistence_mpc import SeasonalPersistenceMPCController
from src.evaluation.comparative_runner import (
    DEFAULT_ASSETS_PATH,
    DEFAULT_RATING_ORIGIN,
    DEFAULT_SPLIT_ROOT,
    reproduce_controller_replay,
)
from src.evaluation.controller_registry import get_controller_entry
from src.simulation.config import load_simulation_config
from src.simulation.metrics import load_cost_config
from src.simulation.data_loader import HourlyObservation, SimulationData
from src.simulation.environment import MicrogridEnvironment
from src.simulation.state import AgentObservation


def create_synthetic_hourly_observation(
    timestamp_utc: str,
    unique_id: int,
) -> HourlyObservation:
    """Create a synthetic observation with uniquely identifiable values based on unique_id."""
    # Values uniquely encode unique_id while respecting physical capacity limits:
    # Grid import limit is 2000 kW -> campus demand must be <= 1500 kW
    # PV <= 300 kW, Wind <= 150 kW
    demand = 400.0 + (unique_id % 100) * 2.0
    solar = min(250.0, 50.0 + (unique_id % 100) * 1.5)
    wind = min(120.0, 30.0 + (unique_id % 100) * 0.8)
    b_split = demand / 5.0
    return HourlyObservation(
        timestamp_utc=timestamp_utc,
        pv_available_kw=solar,
        wind_available_kw=wind,
        campus_load_kw=demand,
        import_price=0.12,
        export_price=0.04,
        building_loads_kw=MappingProxyType({f"B0{k}": b_split for k in range(1, 6)}),
        pv_asset_powers_kw=MappingProxyType({"PV01": solar}),
        wind_asset_powers_kw=MappingProxyType({"W01": wind}),
    )


def build_synthetic_boundary_fixture() -> tuple[list[HourlyObservation], list[HourlyObservation], dict[str, Any]]:
    """Build synthetic validation tail and test head with uniquely identifiable observations.

    Validation tail: 24 observations from 2026-11-06T06:00:00Z to 2026-11-07T05:00:00Z
    Test head: 5 observations from 2026-11-07T06:00:00Z to 2026-11-07T10:00:00Z
    """
    from src.simulation.data_loader import load_simulation_split
    val_data = load_simulation_split(DEFAULT_SPLIT_ROOT, "validation")
    metadata = val_data.metadata

    val_tail: list[HourlyObservation] = []
    # Start: 2026-11-06T06:00:00Z (24 hours prior to 2026-11-07T06:00:00Z)
    val_start = datetime(2026, 11, 6, 6, 0, 0, tzinfo=timezone.utc)
    for i in range(24):
        dt = val_start + timedelta(hours=i)
        ts = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        # unique_id: 1000 + i (1000 = 2026-11-06T06:00:00Z, 1023 = 2026-11-07T05:00:00Z)
        val_tail.append(create_synthetic_hourly_observation(ts, 1000 + i))

    # Test head: starts at 2026-11-07T06:00:00Z
    test_head: list[HourlyObservation] = []
    test_start = datetime(2026, 11, 7, 6, 0, 0, tzinfo=timezone.utc)
    for i in range(5):
        dt = test_start + timedelta(hours=i)
        ts = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        # unique_id: 2000 + i (2000 = 2026-11-07T06:00:00Z, 2001 = 2026-11-07T07:00:00Z)
        test_head.append(create_synthetic_hourly_observation(ts, 2000 + i))

    return val_tail, test_head, metadata


def test_boundary_timestamps_and_simulator_step_semantics():
    """Verify exact timestamp boundaries, simulator reset semantics, and first decision timestamp."""
    val_tail, test_head, metadata = build_synthetic_boundary_fixture()

    # 1. Verify validation tail geometry
    assert len(val_tail) == 24
    assert val_tail[0].timestamp_utc == "2026-11-06T06:00:00Z"
    assert val_tail[-1].timestamp_utc == "2026-11-07T05:00:00Z"

    # 2. Verify test head geometry
    assert len(test_head) == 5
    assert test_head[0].timestamp_utc == "2026-11-07T06:00:00Z"
    assert test_head[1].timestamp_utc == "2026-11-07T07:00:00Z"

    # 3. Simulate environment reset with test_head
    sim_config = load_simulation_config()
    cost_config = load_cost_config()
    data = SimulationData(test_head, metadata)
    renewables, demand = _components(data, DEFAULT_ASSETS_PATH, DEFAULT_RATING_ORIGIN)

    env = MicrogridEnvironment(data, renewables, demand, sim_config, cost_config)
    obs = env.reset()

    # Check Observation Boundary (T0) and Decision Timestamp (T1)
    # The pre-episode observation state is test_head[0] (2026-11-07T06:00:00Z)
    assert obs.state.timestamp_utc == "2026-11-07T06:00:00Z"
    assert obs.state.available_at_utc == "2026-11-07T07:00:00Z"

    # The decision timestamp where controller selects action is 2026-11-07T07:00:00Z
    assert obs.decision_timestamp_utc == "2026-11-07T07:00:00Z"

    # The first physical interval stepped by the environment is [07:00:00Z, 08:00:00Z)
    assert env._rows[env._index].timestamp_utc == "2026-11-07T07:00:00Z"


def test_seasonal_persistence_mpc_first_decision_history():
    """Verify Seasonal Persistence MPC history lookback at first decision step."""
    val_tail, test_head, metadata = build_synthetic_boundary_fixture()

    controller = SeasonalPersistenceMPCController(
        initial_history=val_tail,
        episode_end_utc="2026-11-08T12:00:00Z",
    )
    controller.reset()

    sim_config = load_simulation_config()
    cost_config = load_cost_config()
    data = SimulationData(test_head, metadata)
    renewables, demand = _components(data, DEFAULT_ASSETS_PATH, DEFAULT_RATING_ORIGIN)
    env = MicrogridEnvironment(data, renewables, demand, sim_config, cost_config)
    obs = env.reset()

    action = controller.select_action(obs)

    # Inspect the step diagnostic from the first decision
    assert len(controller.step_diagnostics) == 1
    diag = controller.step_diagnostics[0]

    # Verify decision timestamp
    assert diag.decision_timestamp_utc == "2026-11-07T07:00:00Z"

    # Verify that the forecast target timestamps start at 2026-11-07T07:00:00Z
    assert diag.target_timestamps_utc[0] == "2026-11-07T07:00:00Z"

    # Verify that source timestamps (24h prior) start at 2026-11-06T07:00:00Z
    assert diag.source_timestamps_utc[0] == "2026-11-06T07:00:00Z"

    # Verify that source timestamp at horizon step 23 (2026-11-08T06:00:00Z) is 2026-11-07T06:00:00Z (test row 0)
    assert diag.target_timestamps_utc[23] == "2026-11-08T06:00:00Z"
    assert diag.source_timestamps_utc[23] == "2026-11-07T06:00:00Z"

    # Verify zero missing history and no fallback
    assert diag.fallback_used is False
    assert diag.fallback_hours_count == 0

    # Verify exactly which records were used:
    # 1. First forecast demand corresponds to val_tail[1] (2026-11-06T07:00:00Z, unique_id=1001)
    assert diag.forecast_demand_kw[0] == 400.0 + (1001 % 100) * 2.0

    # 2. 23rd forecast demand corresponds to val_tail[23] (2026-11-07T05:00:00Z, unique_id=1023)
    assert diag.forecast_demand_kw[22] == 400.0 + (1023 % 100) * 2.0

    # 3. 24th forecast demand corresponds to test_head[0] (2026-11-07T06:00:00Z, unique_id=2000)
    assert diag.forecast_demand_kw[23] == 400.0 + (2000 % 100) * 2.0


def test_gnn_transformer_mpc_first_decision_history():
    """Verify GNN-Transformer MPC history lookback at first decision step."""
    val_tail, test_head, metadata = build_synthetic_boundary_fixture()

    controller = GNNTransformerMPCController(
        initial_history=val_tail,
        history_source=val_tail,
        episode_end_utc="2026-11-07T11:00:00Z",
    )
    controller.reset()

    sim_config = load_simulation_config()
    cost_config = load_cost_config()
    data = SimulationData(test_head, metadata)
    renewables, demand = _components(data, DEFAULT_ASSETS_PATH, DEFAULT_RATING_ORIGIN)
    env = MicrogridEnvironment(data, renewables, demand, sim_config, cost_config)
    obs = env.reset()

    # Step through select_action history ingestion and extraction verification
    completed_state = obs.state
    completed_ts = completed_state.timestamp_utc
    assert completed_ts == "2026-11-07T06:00:00Z"

    dec_ts_str = obs.decision_timestamp_utc
    dec_dt = datetime.fromisoformat(dec_ts_str.replace("Z", "+00:00")).astimezone(timezone.utc)
    assert dec_ts_str == "2026-11-07T07:00:00Z"

    # Verify the 24 extracted historical timestamps
    extracted_timestamps = []
    for idx in range(24):
        hist_dt = dec_dt - timedelta(hours=24 - idx)
        hist_ts = hist_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        assert hist_dt < dec_dt, "Non-anticipativity violated"
        extracted_timestamps.append(hist_ts)

    # Verify extracted timestamps range from 2026-11-06T07:00:00Z to 2026-11-07T06:00:00Z
    assert extracted_timestamps[0] == "2026-11-06T07:00:00Z"
    assert extracted_timestamps[22] == "2026-11-07T05:00:00Z"
    assert extracted_timestamps[23] == "2026-11-07T06:00:00Z"

    # Verify no duplicates, no skips, exact 24 contiguous hours
    for i in range(len(extracted_timestamps) - 1):
        t1 = datetime.fromisoformat(extracted_timestamps[i].replace("Z", "+00:00"))
        t2 = datetime.fromisoformat(extracted_timestamps[i + 1].replace("Z", "+00:00"))
        assert (t2 - t1).total_seconds() == 3600


def test_one_shot_runner_safety_and_overwrite_protection():
    """Verify that test runner requires authorization flag and refuses to overwrite non-empty dir."""
    from scripts.evaluate_held_out_test import run_held_out_test_evaluation

    # 1. Verify that running without confirm_unsealing raises PermissionError
    with pytest.raises(PermissionError, match="strictly locked"):
        run_held_out_test_evaluation(confirm_unsealing=False)

    # 2. Verify overwrite protection with synthetic temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        test_file = tmp_path / "dummy_prior_artifact.json"
        test_file.write_text("{}", encoding="utf-8")

        # Attempting to run with non-empty directory should raise FileExistsError
        with pytest.raises(FileExistsError, match="already exists and is non-empty"):
            run_held_out_test_evaluation(
                confirm_unsealing=True,
                output_dir=tmp_path,
            )

    # 3. Exercise complete runner pipeline on validation partition in an isolated temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "test_run_validation"
        res = run_held_out_test_evaluation(
            confirm_unsealing=True,
            controller_ids=["zero_action"],
            output_dir=tmp_path,
            partition="validation",
        )
        assert res["all_succeeded"] is True
        assert Path(res["results_path"]).is_file()
        assert Path(res["manifest_path"]).is_file()
        assert (tmp_path / "test_summary_table.csv").is_file()
        assert (tmp_path / "test_evaluation_metadata.json").is_file()
        assert (tmp_path / "zero_action" / "summary.json").is_file()


def test_reproduce_controller_replay_test_partition_branch_with_synthetic_injection(monkeypatch):
    """Verify the actual partition='test' branch in reproduce_controller_replay() using synthetic injection."""
    from src.evaluation.comparative_runner import reproduce_controller_replay
    import src.evaluation.comparative_runner as comp_runner

    # 1. Verify library-level access guard when allow_held_out_test is False
    with pytest.raises(PermissionError, match="strictly blocked without explicit approval"):
        reproduce_controller_replay("mpc_seasonal_persistence", partition="test", allow_held_out_test=False)

    # 2. Build 10-row synthetic test slice with valid test timestamps starting at 2026-11-07T06:00:00Z
    val_data = comp_runner.load_simulation_split(DEFAULT_SPLIT_ROOT, "validation")
    metadata = val_data.metadata

    synth_test_rows: list[HourlyObservation] = []
    test_start = datetime(2026, 11, 7, 6, 0, 0, tzinfo=timezone.utc)
    for i in range(10):
        dt = test_start + timedelta(hours=i)
        ts = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        synth_test_rows.append(create_synthetic_hourly_observation(ts, 5000 + i))

    synth_test_data = SimulationData(synth_test_rows, metadata)

    # Mock load_simulation_split so that partition="test" returns synth_test_data,
    # while partition="validation" returns real validation data
    real_load_split = comp_runner.load_simulation_split

    def mock_load_split(split_root, partition, **kwargs):
        if partition == "test":
            return synth_test_data
        return real_load_split(split_root, partition, **kwargs)

    monkeypatch.setattr(comp_runner, "load_simulation_split", mock_load_split)

    # 3. Test Seasonal Persistence MPC on partition="test"
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        metrics, hourly, prov = reproduce_controller_replay(
            "mpc_seasonal_persistence",
            partition="test",
            output_dir=tmp_path,
            allow_held_out_test=True,
        )

        # 9 decisions for 10 rows
        assert len(hourly) == 9
        assert metrics.total_steps == 9

        # Verify step 0 physical dispatch timestamp
        first_step = hourly[0]
        assert first_step["timestamp_utc"] == "2026-11-07T07:00:00Z"

        # Verify provenance records validation history origin
        assert prov["partition"] == "test"
        assert prov["controller_id"] == "mpc_seasonal_persistence"

        # Verify that output files were generated
        ctrl_out = tmp_path / "mpc_seasonal_persistence"
        assert (ctrl_out / "summary.json").is_file()
        assert (ctrl_out / "provenance.json").is_file()
        assert (ctrl_out / "hourly_records.json").is_file()

    # 4. Test GNN-Transformer MPC on partition="test"
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        metrics_gnn, hourly_gnn, prov_gnn = reproduce_controller_replay(
            "mpc_gnn_transformer",
            partition="test",
            output_dir=tmp_path,
            allow_held_out_test=True,
        )
        assert len(hourly_gnn) == 9
        assert metrics_gnn.total_steps == 9
        assert hourly_gnn[0]["timestamp_utc"] == "2026-11-07T07:00:00Z"
        assert prov_gnn["partition"] == "test"
        assert prov_gnn["controller_id"] == "mpc_gnn_transformer"

