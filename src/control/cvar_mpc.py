"""Phase 10: CVaR Risk-Aware Model Predictive Control Controller.

Integrates causal scenario generation and the multi-scenario Rockafellar-Uryasev
CVaR LP solver with the physical microgrid simulator.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Any
import numpy as np
import torch
import yaml

from src.baselines.base import BaselineController
from src.control.cvar_mpc_solver import (
    CVaRSolverConfig,
    CVaRSolverResult,
    solve_cvar_mpc_linear_program,
)
from src.control.scenario_generator import CausalScenarioGenerator, UncertaintyScenario
from src.control.seasonal_persistence_mpc import get_scheduled_tariff
from src.data.graph_builder import build_graph
from src.forecasting.integration import GRAPH_NODE_ORDER, TARGET_NODE_ORDER
from src.forecasting.models.gnn_transformer import GNNTransformerForecaster
from src.simulation.actions import BatteryAction
from src.simulation.config import SimulationConfig, load_simulation_config, _UniqueKeyLoader
from src.simulation.data_loader import HourlyObservation
from src.simulation.metrics import CostConfig, load_cost_config
from src.simulation.state import AgentObservation

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CVAR_CONFIG_PATH = ROOT / "configs" / "cvar_mpc_controller.yaml"
DEFAULT_CHECKPOINT_PATH = Path(
    r"E:\AdaptiveEnergyBrainStorage\checkpoints\part5_9-colab-20261111T172326Z\best.pt"
)
DEFAULT_PREPROCESSING_PATH = ROOT / "data" / "processed" / "splits_hypothetical_campus_v1" / "preprocessing.json"


@dataclass(frozen=True)
class CVaRControllerConfig:
    version: int
    horizon_hours: int
    time_step_hours: float
    solver_method: str
    alpha: float
    risk_aversion_lambda: float
    num_scenarios: int
    terminal_inventory_value_usd_per_kwh_dc: float
    power_balance_tolerance_kw: float
    simultaneous_flow_tolerance_kw: float
    allow_curtailment: bool
    price_spike_multiplier: float
    demand_stress_multiplier: float
    renewable_scarcity_multiplier: float
    random_seed: int


def load_cvar_controller_config(path=None) -> CVaRControllerConfig:
    config_path = Path(path) if path is not None else DEFAULT_CVAR_CONFIG_PATH
    raw = yaml.load(config_path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    return CVaRControllerConfig(**raw)


@dataclass(frozen=True)
class CVaRStepDiagnostic:
    decision_timestamp_utc: str
    success: bool
    first_requested_power_kw: float
    expected_cost_usd: float
    cvar_usd: float
    var_usd: float
    worst_case_cost_usd: float
    solve_time_seconds: float
    num_scenarios: int
    fallback_used: bool


class CVaRRiskAwareMPCController(BaselineController):
    """Scenario-based CVaR Risk-Aware Model Predictive Controller (mpc_cvar_risk_aware)."""

    def __init__(
        self,
        cvar_config_path: Path | str | None = None,
        checkpoint_path: Path | str | None = None,
        preprocessing_path: Path | str | None = None,
        sim_config: SimulationConfig | None = None,
        cost_config: CostConfig | None = None,
        history_source: Mapping[str, HourlyObservation] | None = None,
        episode_end_utc: str | None = None,
        device: str = "cpu",
        override_lambda: float | None = None,
    ):
        self.config = load_cvar_controller_config(cvar_config_path)
        self.sim_config = sim_config if sim_config is not None else load_simulation_config()
        self.cost_config = cost_config if cost_config is not None else load_cost_config()
        self.history_source = history_source or {}
        self.episode_end_utc = episode_end_utc
        self.device = torch.device(device)

        # Allow programmatic lambda override for trade-off exploration
        self.risk_lambda = override_lambda if override_lambda is not None else self.config.risk_aversion_lambda

        # Checkpoint resolution
        ckpt_p = Path(checkpoint_path) if checkpoint_path is not None else DEFAULT_CHECKPOINT_PATH
        if not ckpt_p.exists():
            # Check secondary local path
            sec_p = ROOT / "checkpoints" / "part5_9-colab-20261111T172326Z" / "best.pt"
            if sec_p.exists():
                ckpt_p = sec_p
            else:
                raise FileNotFoundError(f"GNN-Transformer checkpoint not found at {ckpt_p}")

        # Load GNN-Transformer model
        self.model = GNNTransformerForecaster.load_checkpoint(ckpt_p, map_location=self.device)
        self.model.eval()

        # Build graph topology
        self.graph = build_graph()
        self.edge_index = torch.tensor(self.graph["edge_index"], dtype=torch.long, device=self.device)

        # Preprocessing normalization scalers
        prep_p = Path(preprocessing_path) if preprocessing_path is not None else DEFAULT_PREPROCESSING_PATH
        with open(prep_p, "r", encoding="utf-8") as f:
            prep_data = json.load(f)
        self._load_scalers(prep_data)

        # Scenario generator
        self.scenario_gen = CausalScenarioGenerator(
            seed=self.config.random_seed,
            demand_stress_multiplier=self.config.demand_stress_multiplier,
            renewable_scarcity_multiplier=self.config.renewable_scarcity_multiplier,
            price_spike_multiplier=self.config.price_spike_multiplier,
            pv_capacity_kw=self.sim_config.solar.capacity_kw if hasattr(self.sim_config, "solar") else 300.0,
            wind_capacity_kw=self.sim_config.wind.capacity_kw if hasattr(self.sim_config, "wind") else 150.0,
        )

        # Solver config
        self.solver_config = CVaRSolverConfig(
            alpha=self.config.alpha,
            risk_aversion_lambda=self.risk_lambda,
            solver_method=self.config.solver_method,
            time_step_hours=self.config.time_step_hours,
            terminal_inventory_value_usd_per_kwh_dc=self.config.terminal_inventory_value_usd_per_kwh_dc,
            power_balance_tolerance_kw=self.config.power_balance_tolerance_kw,
            allow_curtailment=self.config.allow_curtailment,
        )

        self._history: dict[str, Any] = {}
        self._latest_record: Any = None
        self.step_diagnostics: list[CVaRStepDiagnostic] = []

    def _load_scalers(self, prep_json: dict[str, Any]) -> None:
        """Extract train-fitted scaler parameters for node power features."""
        scalers = prep_json.get("scalers", {})
        demand_entities = scalers.get("synthetic_hypothetical_campus_v1_demand", {}).get("entities", {})
        solar_entities = scalers.get("synthetic_hypothetical_campus_v1_solar", {}).get("entities", {})
        wind_entities = scalers.get("synthetic_hypothetical_campus_v1_wind", {}).get("entities", {})

        means: list[float] = []
        stds: list[float] = []

        for node_id in GRAPH_NODE_ORDER:
            if node_id in {"B01", "B02", "B03", "B04", "B05"}:
                p = demand_entities[node_id]["building_load_kw"]
                means.append(float(p["mean"]))
                stds.append(float(p["scale"]))
            elif node_id == "PV01":
                p = solar_entities["PV01"]["available_generation_kw"]
                means.append(float(p["mean"]))
                stds.append(float(p["scale"]))
            elif node_id == "W01":
                p = wind_entities["W01"]["available_generation_kw"]
                means.append(float(p["mean"]))
                stds.append(float(p["scale"]))
            elif node_id == "BUS01":
                means.append(0.0)
                stds.append(1.0)
            else:
                raise ValueError(f"Unknown node ID: {node_id}")

        self._node_means_t = torch.tensor(means, dtype=torch.float32, device=self.device).reshape(1, 1, 8, 1)
        self._node_stds_t = torch.tensor(stds, dtype=torch.float32, device=self.device).reshape(1, 1, 8, 1)

    def _reset(self) -> None:
        self._history.clear()
        self._latest_record = None
        self.step_diagnostics.clear()

    def _select_action(self, observation: AgentObservation) -> BatteryAction:
        completed_state = getattr(observation, "state", None)
        if completed_state is not None:
            completed_ts = completed_state.timestamp_utc
            if completed_ts in self.history_source:
                row = self.history_source[completed_ts]
                record = {
                    "timestamp_utc": completed_ts,
                    "building_loads_kw": dict(row.building_loads_kw),
                    "solar_generation_kw": float(row.pv_available_kw),
                    "wind_generation_kw": float(row.wind_available_kw),
                    "campus_demand_kw": float(row.campus_load_kw),
                }
            else:
                split_b = completed_state.campus_demand_kw / 5.0
                record = {
                    "timestamp_utc": completed_ts,
                    "building_loads_kw": {f"B0{k}": split_b for k in range(1, 6)},
                    "solar_generation_kw": float(completed_state.solar_generation_kw),
                    "wind_generation_kw": float(completed_state.wind_generation_kw),
                    "campus_demand_kw": float(completed_state.campus_demand_kw),
                }
            self._history[completed_ts] = record
            self._latest_record = record

        dec_ts_str = observation.decision_timestamp_utc
        dec_dt = datetime.fromisoformat(dec_ts_str.replace("Z", "+00:00")).astimezone(timezone.utc)

        # Determine horizon
        max_h = self.config.horizon_hours
        if self.episode_end_utc is not None:
            end_dt = datetime.fromisoformat(self.episode_end_utc.replace("Z", "+00:00")).astimezone(timezone.utc)
            rem_hours = int((end_dt - dec_dt).total_seconds() // 3600)
            horizon = max(1, min(max_h, rem_hours))
        else:
            horizon = max_h

        # Extract 24 completed history hours
        input_array = np.zeros((1, 24, 8, 1), dtype=np.float32)
        for idx in range(24):
            hist_dt = dec_dt - timedelta(hours=24 - idx)
            hist_ts = hist_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

            if hist_ts in self._history:
                h_rec = self._history[hist_ts]
            elif hist_ts in self.history_source:
                row = self.history_source[hist_ts]
                h_rec = {
                    "building_loads_kw": dict(row.building_loads_kw),
                    "solar_generation_kw": float(row.pv_available_kw),
                    "wind_generation_kw": float(row.wind_available_kw),
                    "campus_demand_kw": float(row.campus_load_kw),
                }
            else:
                h_rec = self._latest_record

            if h_rec is not None:
                b_loads = h_rec.get("building_loads_kw", {}) if isinstance(h_rec, dict) else getattr(h_rec, "building_loads_kw", {})
                for n_idx, node_id in enumerate(GRAPH_NODE_ORDER):
                    if node_id in b_loads:
                        val = b_loads[node_id]
                    elif node_id == "PV01":
                        val = h_rec.get("solar_generation_kw", 0.0) if isinstance(h_rec, dict) else getattr(h_rec, "solar_generation_kw", 0.0)
                    elif node_id == "W01":
                        val = h_rec.get("wind_generation_kw", 0.0) if isinstance(h_rec, dict) else getattr(h_rec, "wind_generation_kw", 0.0)
                    elif node_id == "BUS01":
                        val = 0.0
                    else:
                        val = 50.0
                    input_array[0, idx, n_idx, 0] = float(val)
            else:
                input_array[0, idx, :5, 0] = 50.0

        # GNN-Transformer forward pass
        input_tensor = torch.tensor(input_array, dtype=torch.float32, device=self.device)
        scaled_input = (input_tensor - self._node_means_t) / self._node_stds_t
        with torch.no_grad():
            predictions = self.model(scaled_input, self.edge_index)

        pred_kw = predictions.cpu().numpy()[0]
        pred_kw = np.maximum(pred_kw, 0.0)

        # Aggregate nominal forecasts
        nominal_demands = []
        nominal_solars = []
        nominal_winds = []
        scheduled_import_prices = []
        scheduled_export_prices = []

        for k in range(horizon):
            target_dt = dec_dt + timedelta(hours=k)
            target_ts = target_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

            # Campus load is sum of B01-B05
            b_load = float(np.sum(pred_kw[k, :5, 0]))
            sol = float(pred_kw[k, 5, 0])
            wnd = float(pred_kw[k, 6, 0])

            p_imp, p_exp = get_scheduled_tariff(target_ts)

            nominal_demands.append(b_load)
            nominal_solars.append(sol)
            nominal_winds.append(wnd)
            scheduled_import_prices.append(p_imp)
            scheduled_export_prices.append(p_exp)

        # Generate uncertainty scenarios
        scenarios = self.scenario_gen.generate_scenarios(
            nominal_demand_kw=nominal_demands,
            nominal_solar_kw=nominal_solars,
            nominal_wind_kw=nominal_winds,
            scheduled_import_price=scheduled_import_prices,
            scheduled_export_price=scheduled_export_prices,
            num_scenarios=self.config.num_scenarios,
        )

        soc = observation.state.battery_soc if hasattr(observation, "state") and hasattr(observation.state, "battery_soc") else getattr(observation, "battery_soc", 0.5)
        initial_energy_kwh = soc * self.sim_config.battery.capacity_kwh

        # Solve CVaR LP
        res = solve_cvar_mpc_linear_program(
            initial_energy_kwh=initial_energy_kwh,
            scenarios=scenarios,
            sim_config=self.sim_config,
            cost_config=self.cost_config,
            cvar_config=self.solver_config,
        )

        fallback_used = False
        if not res.success:
            fallback_used = True
            # Infeasibility fallback: zero action
            action = BatteryAction(0.0)
        else:
            action = res.first_action

        # Record diagnostics
        diag = CVaRStepDiagnostic(
            decision_timestamp_utc=dec_ts_str,
            success=res.success,
            first_requested_power_kw=res.first_requested_power_kw,
            expected_cost_usd=res.expected_operating_cost_usd,
            cvar_usd=res.cvar_loss_usd,
            var_usd=res.var_loss_usd,
            worst_case_cost_usd=res.worst_case_scenario_cost_usd,
            solve_time_seconds=res.diagnostics.solve_time_seconds,
            num_scenarios=len(scenarios),
            fallback_used=fallback_used,
        )
        self.step_diagnostics.append(diag)

        return action
