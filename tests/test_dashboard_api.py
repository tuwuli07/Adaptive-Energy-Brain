import json
import pytest
from src.dashboard.api import DATA_SOURCE, APIRequestHandler

def test_api_data_source_loads_registry():
    reg = DATA_SOURCE.get_registry()
    assert "zero_action" in reg
    assert "mpc_gnn_transformer" in reg
    assert reg["mpc_gnn_transformer"]["causality"] == "strictly_causal"

def test_api_data_source_loads_test_benchmark():
    test_res = DATA_SOURCE.get_test_benchmark()
    assert test_res.get("all_succeeded") is True
    results = test_res.get("results", {})
    assert len(results) == 7
    assert "mpc_gnn_transformer" in results
    gnn_m = results["mpc_gnn_transformer"]["metrics"]
    assert gnn_m["inventory_adjusted_cost_usd"] == pytest.approx(36532.73, abs=0.01)

def test_api_data_source_loads_val_benchmark():
    val_res = DATA_SOURCE.get_val_benchmark()
    assert len(val_res) == 7
    assert "mpc_gnn_transformer" in val_res
    gnn_val = val_res["mpc_gnn_transformer"]["metrics"]
    assert gnn_val["inventory_adjusted_cost_usd"] == pytest.approx(43702.87, abs=0.01)

def test_api_data_source_loads_hourly_records():
    hourly = DATA_SOURCE.get_hourly_records("mpc_gnn_transformer", partition="test")
    assert hourly is not None
    assert len(hourly) == 1313
    first_step = hourly[0]
    assert "campus_demand_kw" in first_step
    assert "battery_actual_kw" in first_step
    assert "battery_soc_after" in first_step


def test_api_data_source_loads_cvar_results():
    cvar_res = DATA_SOURCE.get_cvar_results()
    assert "normal" in cvar_res
    assert "compound_adverse" in cvar_res
    assert "mpc_cvar_neutral" in cvar_res["normal"]
    assert "mpc_cvar_averse" in cvar_res["compound_adverse"]

    tradeoff = DATA_SOURCE.get_cvar_tradeoff()
    assert len(tradeoff) >= 5
    assert tradeoff[0]["risk_aversion_lambda"] == 0.0


def test_api_data_source_loads_dro_results():
    dro_res = DATA_SOURCE.get_dro_results()
    assert "normal" in dro_res
    assert "compound_adverse" in dro_res
    assert "mpc_dro_robust" in dro_res["normal"]
    assert "mpc_dro_robust" in dro_res["compound_adverse"]

    sweep = DATA_SOURCE.get_dro_ambiguity_sweep()
    assert len(sweep) >= 5
    assert sweep[0]["ambiguity_radius_epsilon"] == 0.0


def test_api_data_source_loads_adaptive_results():
    adaptive_res = DATA_SOURCE.get_adaptive_results()
    assert "normal" in adaptive_res
    assert "compound_adverse" in adaptive_res
    assert "adaptive_policy_selector" in adaptive_res["normal"]
    assert "adaptive_policy_selector" in adaptive_res["compound_adverse"]

    traj = DATA_SOURCE.get_adaptive_trajectory()
    assert len(traj) > 0
    first_step = traj[0]
    assert "selected_controller_id" in first_step
    assert "regime_name" in first_step
    assert "rationale" in first_step


