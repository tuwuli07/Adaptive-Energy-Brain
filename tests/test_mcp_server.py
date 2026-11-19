"""Unit tests for Model Context Protocol (MCP) server tools."""

import pytest

from src.assistant.mcp_server import (
    list_registered_tools,
    execute_tool,
    READ_ONLY_TOOLS,
    get_system_overview,
    get_battery_status,
    get_controller_comparison,
    get_risk_results,
    get_dro_results,
    get_adaptive_policy_results,
    get_forecast_metrics,
    get_simulation_metadata,
    get_research_limitations,
    get_energy_summary,
)


def test_mcp_tools_registration():
    tools = list_registered_tools()
    assert len(tools) == 10
    tool_names = {t["name"] for t in tools}
    expected = {
        "get_system_overview",
        "get_energy_summary",
        "get_forecast_metrics",
        "get_battery_status",
        "get_controller_comparison",
        "get_risk_results",
        "get_dro_results",
        "get_adaptive_policy_results",
        "get_simulation_metadata",
        "get_research_limitations",
    }
    assert tool_names == expected


def test_mcp_system_overview():
    data = get_system_overview()
    assert data["assets"]["battery_storage"]["max_charge_power_kw"] == 150.0
    assert data["assets"]["battery_storage"]["capacity_kwh"] == 500.0
    assert len(data["available_controllers"]) >= 7


def test_mcp_battery_status():
    status = get_battery_status()
    assert status["capacity_kwh"] == 500.0
    assert status["power_limits_kw"]["charge_max_kw"] == 150.0
    assert status["power_limits_kw"]["discharge_max_kw"] == 150.0
    assert status["soc_limits"]["min_soc"] == 0.10
    assert status["soc_limits"]["max_soc"] == 0.90


def test_mcp_controller_comparison_test_partition():
    res = get_controller_comparison(partition="test")
    assert res["partition"].startswith("test")
    assert len(res["controllers"]) == 7
    
    # Verify GNN-MPC metrics
    gnn = next(c for c in res["controllers"] if c["controller_id"] == "mpc_gnn_transformer")
    assert gnn["adjusted_cost_usd"] == pytest.approx(36532.73, abs=0.01)
    assert gnn["battery_throughput_kwh"] == pytest.approx(31540.80, abs=0.01)


def test_mcp_risk_and_dro_results():
    risk = get_risk_results(condition="all")
    assert "conditions" in risk
    assert "pareto_frontier" in risk

    dro = get_dro_results(condition="all")
    assert "conditions" in dro
    assert "ambiguity_sweep" in dro


def test_mcp_adaptive_policy_results():
    adaptive = get_adaptive_policy_results()
    assert "regime_architecture" in adaptive
    assert "switching_profile_compound_stress" in adaptive
    assert adaptive["switching_profile_compound_stress"]["physical_violations"] == 0


def test_mcp_forecast_metrics():
    metrics = get_forecast_metrics()
    assert "horizons" in metrics
    assert "24h" in metrics["horizons"]
    assert metrics["horizons"]["24h"]["rmse_kw"] > 0.0


def test_mcp_research_limitations():
    limits = get_research_limitations()
    assert len(limits["limitations"]) >= 4
    assert any("synthetic" in l.lower() for l in limits["limitations"])
    assert any("150" in l or "linear" in l.lower() for l in limits["limitations"])


def test_mcp_execute_tool_invalid_name():
    with pytest.raises(ValueError, match="Unknown tool"):
        execute_tool("nonexistent_tool")


def test_mcp_execute_tool_extra_arguments():
    # Extra arguments should be safely ignored
    res = execute_tool("get_battery_status", arguments={"extra_param": 1234})
    assert res["status"] == "success"
    assert res["data"]["capacity_kwh"] == 500.0
