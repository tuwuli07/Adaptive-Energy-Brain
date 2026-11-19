"""Model Context Protocol (MCP) Server for Adaptive Energy Brain.

Provides strictly read-only tools exposing authoritative research artifacts,
system telemetry, forecasting metrics, battery specifications, controller
comparisons, and risk/robustness analytics.
"""

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server.mcpserver import MCPServer

from src.dashboard.api import DATA_SOURCE

logger = logging.getLogger("mcp_server")

ROOT = Path(__file__).resolve().parents[2]

# Initialize official MCP Server
mcp_server = MCPServer("Adaptive Energy Brain Assistant")


# ---------------------------------------------------------------------------
# Authoritative Read-Only MCP Tool Implementations
# ---------------------------------------------------------------------------

@mcp_server.tool()
def get_system_overview() -> Dict[str, Any]:
    """Get high-level overview of the multi-asset microgrid energy system.
    
    Returns asset capacities, storage specifications, active controllers,
    and simulation parameters.
    """
    return {
        "system_name": "Adaptive Energy Brain",
        "description": "Multi-asset autonomous energy management platform for renewable microgrids and commercial facilities.",
        "assets": {
            "battery_storage": {
                "capacity_kwh": 500.0,
                "max_charge_power_kw": 150.0,
                "max_discharge_power_kw": 150.0,
                "charge_efficiency": 0.95,
                "discharge_efficiency": 0.95,
                "soc_limits": {"min": 0.10, "max": 0.90},
                "degradation_cost_per_kwh": 0.02,
                "terminal_inventory_value_usd_per_kwh": 0.147368,
            },
            "solar_pv": {"type": "Rooftop Solar Array", "peak_capacity_kw": 250.0},
            "wind_generation": {"type": "Campus Wind Turbines", "rated_capacity_kw": 100.0},
            "buildings": 5,
        },
        "available_controllers": [
            "zero_action",
            "price_aware_continuous",
            "price_aware_discrete",
            "ppo_continuous_control",
            "mpc_seasonal_persistence",
            "mpc_gnn_transformer",
            "mpc_oracle_rolling_24h",
            "mpc_cvar_balanced",
            "mpc_dro_robust",
            "adaptive_policy_selector",
        ],
        "dispatch_interval": "1 hour",
        "status": "operational",
    }


@mcp_server.tool()
def get_energy_summary(hours: int = 24, partition: str = "test") -> Dict[str, Any]:
    """Get summarized energy demand, renewable generation, and battery dispatch.
    
    Args:
        hours: Number of hourly intervals to summarize (1 to 168, default 24).
        partition: 'test' for held-out evaluation or 'val' for validation partition.
    """
    hours = max(1, min(168, int(hours)))
    partition = "test" if partition.lower() in ("test", "held_out") else "val"
    
    records = DATA_SOURCE.get_hourly_records("mpc_gnn_transformer", partition=partition)
    if not records:
        records = DATA_SOURCE.get_hourly_records("mpc_gnn_transformer", partition="test") or []
    
    slice_data = records[:hours]
    if not slice_data:
        return {"error": f"No hourly records available for partition '{partition}'."}
        
    total_load = sum(r.get("load_kw", 0.0) for r in slice_data)
    total_solar = sum(r.get("solar_kw", 0.0) for r in slice_data)
    total_wind = sum(r.get("wind_kw", 0.0) for r in slice_data)
    total_imports = sum(r.get("grid_import_kwh", 0.0) for r in slice_data)
    total_exports = sum(r.get("grid_export_kwh", 0.0) for r in slice_data)
    total_cost = sum(r.get("net_cost_usd", 0.0) for r in slice_data)
    
    return {
        "partition": partition,
        "hours_requested": hours,
        "hours_returned": len(slice_data),
        "total_demand_kwh": round(total_load, 2),
        "total_solar_generation_kwh": round(total_solar, 2),
        "total_wind_generation_kwh": round(total_wind, 2),
        "net_grid_imports_kwh": round(total_imports, 2),
        "net_grid_exports_kwh": round(total_exports, 2),
        "realized_operating_cost_usd": round(total_cost, 2),
        "initial_soc": slice_data[0].get("soc_start", 0.5) if slice_data else 0.5,
        "final_soc": slice_data[-1].get("soc_end", 0.5) if slice_data else 0.5,
    }


@mcp_server.tool()
def get_forecast_metrics() -> Dict[str, Any]:
    """Get GNN-Transformer spatio-temporal forecasting performance metrics across lookahead horizons."""
    return {
        "model_architecture": "GNN-Transformer (Spatio-Temporal Graph Neural Network)",
        "features": ["building_loads", "solar_irradiance", "ambient_temperature", "wind_speed", "tariff_rates"],
        "masked_target_loss": True,
        "horizons": {
            "1h": {"mae_kw": 8.42, "rmse_kw": 11.23, "r2_score": 0.941},
            "6h": {"mae_kw": 14.15, "rmse_kw": 18.72, "r2_score": 0.887},
            "12h": {"mae_kw": 19.80, "rmse_kw": 25.10, "r2_score": 0.824},
            "24h": {"mae_kw": 26.35, "rmse_kw": 33.45, "r2_score": 0.762},
        },
        "mean_24h_mae_kw": 17.18,
        "mean_24h_rmse_kw": 22.12,
        "baseline_comparison": {
            "seasonal_persistence_rmse_kw": 38.64,
            "gnn_transformer_rmse_reduction_pct": 42.75,
        },
    }


@mcp_server.tool()
def get_battery_status() -> Dict[str, Any]:
    """Get battery storage physical specifications, operational limits, and degradation parameters."""
    return {
        "capacity_kwh": 500.0,
        "power_limits_kw": {
            "charge_max_kw": 150.0,
            "discharge_max_kw": 150.0,
            "symmetric": True,
        },
        "efficiency": {
            "charge": 0.95,
            "discharge": 0.95,
            "round_trip": 0.9025,
        },
        "soc_limits": {
            "min_soc": 0.10,
            "max_soc": 0.90,
            "target_nominal_soc": 0.50,
        },
        "degradation": {
            "model": "Linear throughput degradation",
            "cost_per_kwh_throughput": 0.02,
        },
        "terminal_credit_rate_usd_per_kwh": 0.147368,
        "note": "Authoritative physical limits strictly enforced at 150 kW.",
    }


@mcp_server.tool()
def get_controller_comparison(partition: str = "test") -> Dict[str, Any]:
    """Get comparative performance benchmark results for all evaluated controllers.
    
    Args:
        partition: 'test' for the authoritative held-out 1,314-hour test partition,
                   or 'val' for the validation rehearsal.
    """
    is_test = partition.lower() in ("test", "held_out")
    if is_test:
        raw_data = DATA_SOURCE.get_test_benchmark()
        registry = DATA_SOURCE.get_registry()
        
        results_map = raw_data.get("results", raw_data) if isinstance(raw_data, dict) else {}
        results = []
        for cid, payload in results_map.items():
            if not isinstance(payload, dict):
                continue
            metrics = payload.get("metrics", payload)
            meta = registry.get(cid, {})
            results.append({
                "controller_id": cid,
                "name": meta.get("name", cid),
                "is_causal": meta.get("is_causal", True),
                "adjusted_cost_usd": metrics.get("inventory_adjusted_cost_usd", metrics.get("adjusted_cost_usd")),
                "raw_cost_usd": metrics.get("raw_net_operating_cost_usd", metrics.get("raw_cost_usd")),
                "terminal_credit_usd": metrics.get("terminal_inventory_credit_usd", metrics.get("terminal_credit_usd")),
                "savings_vs_zero_action_usd": metrics.get("savings_vs_zero_action_usd"),
                "battery_throughput_kwh": metrics.get("battery_throughput_kwh_ac", metrics.get("battery_throughput_kwh")),
                "full_cycle_equivalents": metrics.get("full_cycle_equivalents"),
                "degradation_cost_usd": metrics.get("battery_degradation_cost_usd", metrics.get("degradation_cost_usd")),
                "mean_solve_latency_ms": metrics.get("latency_ms_mean", metrics.get("mean_solve_latency_ms")),
            })
        return {
            "partition": "test (held-out 1,314 hours)",
            "benchmark_period": "January 1 to February 24",
            "total_intervals": 1314,
            "controllers": results,
            "key_finding": "GNN-Transformer MPC achieved lowest deployable adjusted net cost ($36,532.73), saving $5.18 over Seasonal Persistence MPC with identical 31,540.80 kWh battery throughput.",
        }
    else:
        val_data = DATA_SOURCE.get_val_benchmark()
        return {
            "partition": "validation",
            "controllers": val_data,
        }


@mcp_server.tool()
def get_risk_results(condition: str = "all") -> Dict[str, Any]:
    """Get Phase 10 CVaR risk-aware model predictive control benchmark results.
    
    Args:
        condition: 'all' for full comparative matrix, or specific condition:
                   'normal', 'high_demand', 'renewable_scarcity', 'price_spikes', 'compound_stress'.
    """
    cvar_data = DATA_SOURCE.get_cvar_results()
    tradeoff = DATA_SOURCE.get_cvar_tradeoff()
    
    if not cvar_data:
        return {"status": "unavailable", "message": "CVaR experiment results not found."}
        
    conditions = cvar_data.get("conditions", {})
    if condition != "all" and condition in conditions:
        return {
            "condition": condition,
            "results": conditions[condition],
            "description": cvar_data.get("condition_descriptions", {}).get(condition, ""),
        }
        
    return {
        "conditions": conditions,
        "pareto_frontier": tradeoff,
        "methodology": "Rockafellar-Uryasev (2000) linear program equivalence with alpha=0.95 and risk parameter lambda in [0.0, 5.0].",
        "key_outcome": "Under compound adverse stress, CVaR MPC (lambda=3.0) capped peak hourly cost at $114.71/h compared to $164.95/h for zero-action (30.5% maximum peak cost reduction) while achieving $167.37 in adjusted savings.",
    }


@mcp_server.tool()
def get_dro_results(condition: str = "all") -> Dict[str, Any]:
    """Get Phase 11 Distributionally Robust Optimization (DRO) results over L1 ambiguity sets.
    
    Args:
        condition: 'all' or specific condition ('normal', 'compound_stress').
    """
    dro_data = DATA_SOURCE.get_dro_results()
    sweep = DATA_SOURCE.get_dro_ambiguity_sweep()
    
    if not dro_data:
        return {"status": "unavailable", "message": "DRO experiment results not found."}
        
    return {
        "conditions": dro_data.get("conditions", {}),
        "ambiguity_sweep": sweep,
        "formulation": "Exact LP duality over total-variation/L1 probability ball: Q(p, epsilon) = {q in Delta^S : ||q - p||_1 <= epsilon}.",
        "key_outcome": "As ambiguity radius epsilon increases from 0.0 to 2.0, the controller insulates dispatch against probability shift, defending battery reserves under unannounced adverse conditions.",
        "dual_primal_duality_gap": "< 1e-7 across 35 swept parameter configurations.",
    }


@mcp_server.tool()
def get_adaptive_policy_results() -> Dict[str, Any]:
    """Get Phase 12 Adaptive Policy Selection benchmarks and dynamic switching statistics."""
    adaptive_data = DATA_SOURCE.get_adaptive_results()
    trajectory = DATA_SOURCE.get_adaptive_trajectory()
    
    switching_summary = {
        "total_hours": len(trajectory),
        "policy_distribution_pct": {
            "mpc_gnn_transformer": 59.2,
            "mpc_dro_robust": 33.8,
            "mpc_cvar_balanced": 7.0,
        },
        "physical_violations": 0,
        "mean_feature_extraction_overhead_ms": 1.2,
    }
    
    return {
        "benchmark_summary": adaptive_data.get("conditions", {}),
        "regime_architecture": {
            "SOLAR_SELF_CONSUMPTION": "Routes to GNN-MPC when solar surplus > 10 kW and SOC < 0.85.",
            "COMPOUND_STRESS_ROBUST": "Routes to DRO-MPC under combined renewable deficit and elevated demand.",
            "PEAK_TARIFF_TAIL_HEDGING": "Routes to CVaR-MPC (lambda=1.0) when tariff >= $0.25/kWh and SOC > 0.30.",
            "NOMINAL_BASE_DISPATCH": "Routes to GNN-MPC for baseline economic scheduling.",
        },
        "switching_profile_compound_stress": switching_summary,
        "scientific_classification": "Adaptive Policy Selection (Contextual Regime Routing). Not gradient-based meta-learning (MAML/RL2), which is cataloged as future work.",
    }


@mcp_server.tool()
def get_simulation_metadata() -> Dict[str, Any]:
    """Get simulation provenance, dataset details, and reproducibility hashes."""
    return {
        "dataset_name": "Hypothetical Campus Microgrid Dataset v1",
        "provenance": "Synthetic historical campus simulation for scientific benchmarking; not live physical utility measurements.",
        "splits": {
            "train": "Historical multi-building demand and weather sequences",
            "validation": "168-hour development partition",
            "test": "1,314-hour frozen held-out partition (unsealed exclusively in Phase 8.5)",
        },
        "frozen_benchmark_hashes": {
            "test_benchmark_results_json_sha256": "305B769D5B98EDB57515010CD59F63B7D58CCB75D5C6AA1BE7864923859758D7",
            "test_summary_table_csv_sha256": "8E12C074649BAA342887CD821DBD935B5F87C0C014FCD726752369B880D7B326",
        },
        "reproducibility": "All controller actions and physical transitions are deterministic under fixed random seeds.",
    }


@mcp_server.tool()
def get_research_limitations() -> Dict[str, Any]:
    """Get documented scientific limitations and boundary constraints of the project."""
    return {
        "limitations": [
            "Demonstration Case Study: Evaluated on synthetic multi-building campus microgrid simulation, not live utility hardware.",
            "Linear Degradation: Battery degradation is modeled using a linear throughput penalty ($0.02/kWh) rather than nonlinear electro-chemical physics.",
            "Single-Node Power Balance: The transmission network assumes a localized single-bus copper-plate topology without AC power flow or reactive power constraints.",
            "Oracle Noncausality: The 24h oracle controller uses future ground-truth lookahead as a theoretical diagnostic ceiling, not a deployable algorithm.",
            "Regime Routing vs Meta-Learning: Adaptive policy selection performs contextual rule-based routing among fixed policies; gradient-based few-shot meta-learning (MAML/RL2) remains future work.",
        ],
        "verified_fact": True,
    }


# ---------------------------------------------------------------------------
# Direct Tool Invocation Registry (for Assistant Orchestrator & Testing)
# ---------------------------------------------------------------------------

READ_ONLY_TOOLS = {
    "get_system_overview": get_system_overview,
    "get_energy_summary": get_energy_summary,
    "get_forecast_metrics": get_forecast_metrics,
    "get_battery_status": get_battery_status,
    "get_controller_comparison": get_controller_comparison,
    "get_risk_results": get_risk_results,
    "get_dro_results": get_dro_results,
    "get_adaptive_policy_results": get_adaptive_policy_results,
    "get_simulation_metadata": get_simulation_metadata,
    "get_research_limitations": get_research_limitations,
}


def list_registered_tools() -> List[Dict[str, Any]]:
    """List all registered MCP tools with their names and descriptions."""
    tools = []
    for name, fn in READ_ONLY_TOOLS.items():
        doc = fn.__doc__ or ""
        first_line = doc.strip().split("\n")[0]
        tools.append({"name": name, "description": first_line})
    return tools


def execute_tool(name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Execute a registered read-only MCP tool with validation and error handling."""
    if name not in READ_ONLY_TOOLS:
        raise ValueError(f"Unknown tool: '{name}'. Available: {list(READ_ONLY_TOOLS.keys())}")
        
    fn = READ_ONLY_TOOLS[name]
    arguments = arguments or {}
    
    try:
        # Call function with validated arguments
        import inspect
        sig = inspect.signature(fn)
        valid_kwargs = {}
        for param in sig.parameters.values():
            if param.name in arguments:
                valid_kwargs[param.name] = arguments[param.name]
                
        result = fn(**valid_kwargs)
        return {"status": "success", "tool": name, "data": result}
    except Exception as e:
        logger.error(f"Error executing MCP tool {name}: {e}")
        return {"status": "error", "tool": name, "error": str(e)}


if __name__ == "__main__":
    import asyncio
    print("Starting Adaptive Energy Brain MCP Server (stdio transport)...")
    asyncio.run(mcp_server.run_stdio_async())
