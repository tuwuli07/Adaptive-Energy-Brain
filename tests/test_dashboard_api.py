import json
import builtins
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


def test_assistant_orchestrator_integration():
    from src.assistant.orchestrator import ASSISTANT_ORCHESTRATOR
    res = ASSISTANT_ORCHESTRATOR.answer_query("What is the battery inverter power limit?")
    assert res["status"] in ("success", "offline_fallback", "fallback_on_error")
    assert "150" in res["answer"]
    assert len(res["sources"]) > 0
    assert "data_provenance" in res


def test_assistant_status_payload():
    from src.assistant.gemini_service import GEMINI_SERVICE
    from src.assistant.mcp_server import list_registered_tools
    from src.assistant.rag import RAG_PIPELINE

    tools = list_registered_tools()
    assert len(tools) == 10
    chunks = RAG_PIPELINE.get_collection().count()
    assert chunks >= 200


def test_assistant_status_does_not_import_optional_research_dependencies(monkeypatch):
    handler = APIRequestHandler.__new__(APIRequestHandler)
    payload = {}
    monkeypatch.setattr(handler, "send_json", lambda value, status=200: payload.update(value))
    real_import = builtins.__import__

    def reject_optional_imports(name, *args, **kwargs):
        if name in {"src.assistant.mcp_server", "src.assistant.rag"}:
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_optional_imports)
    handler.handle_assistant_status()
    assert payload["status"] == "online"
    assert payload["default_provider"] == "gemini"



