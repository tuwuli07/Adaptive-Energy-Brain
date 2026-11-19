"""Phase 7.4: GNN-Transformer MPC controller for microgrid energy management.

Integrates the Part 5.9 trained GNN-Transformer forecasting checkpoint with the
Phase 7.1 linear programming MPC solver to optimize battery dispatch over a rolling
24-hour horizon.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.baselines.base import BaselineController
from src.control.mpc_solver import (
    MPCInputProfile,
    MPCSolverConfig,
    MPCSolverResult,
    load_mpc_config,
    solve_mpc_linear_program,
)
from src.control.seasonal_persistence_mpc import get_scheduled_tariff
from src.data.graph_builder import build_graph
from src.forecasting.integration import GRAPH_NODE_ORDER, TARGET_NODE_ORDER
from src.forecasting.models.gnn_transformer import (
    GNNTransformerError,
    GNNTransformerForecaster,
)
from src.simulation.actions import BatteryAction
from src.simulation.config import SimulationConfig, load_simulation_config
from src.simulation.data_loader import HourlyObservation
from src.simulation.metrics import CostConfig, load_cost_config
from src.simulation.state import AgentObservation

DEFAULT_CHECKPOINT_PATH = Path(
    r"E:\AdaptiveEnergyBrainStorage\checkpoints\part5_9-colab-20261111T172326Z\best.pt"
)
DEFAULT_PREPROCESSING_PATH = Path(
    "data/processed/splits_hypothetical_campus_v1/preprocessing.json"
)


@dataclass(frozen=True)
class NodeHistoricalRecord:
    """Historical completed node observations retained causally."""

    timestamp_utc: str
    building_loads_kw: Mapping[str, float]
    solar_generation_kw: float
    wind_generation_kw: float
    campus_demand_kw: float


@dataclass(frozen=True)
class GNNMPCStepDiagnostic:
    """Per-step diagnostics for the causal GNN-Transformer MPC controller."""

    decision_timestamp_utc: str
    success: bool
    status: int
    solve_time_seconds: float
    inference_time_seconds: float
    first_requested_power_kw: float
    horizon_hours: int
    fallback_used: bool
    fallback_hours_count: int
    has_simultaneous_battery_flow: bool
    forecast_demand_kw: tuple[float, ...]
    forecast_solar_kw: tuple[float, ...]
    forecast_wind_kw: tuple[float, ...]
    target_timestamps_utc: tuple[str, ...]


class GNNTransformerMPCController(BaselineController):
    """24-Hour rolling MPC controller driven by causal GNN-Transformer forecasts.

    Strict causality rules:
      - Uses completed historical intervals only: observations completed at or before t-1.
      - Never receives or accesses future realized validation partition rows.
      - Never accesses future unannounced stochastic price spikes (plans with published TOU).
      - Applies train-fitted standard scalers for input normalization and inverse transformation.
      - Projects negative predictions to 0.0 kW at the boundary.
      - Aggregates B01-B05 predictions into campus demand.
      - First forecast step (k=0) corresponds to interval [t, t+1h) in which the first action executes.
    """

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        preprocessing_path: str | Path | None = None,
        initial_history: Sequence[HourlyObservation] | None = None,
        history_source: Mapping[str, HourlyObservation] | Sequence[HourlyObservation] | None = None,
        sim_config: SimulationConfig | None = None,
        cost_config: CostConfig | None = None,
        mpc_config: MPCSolverConfig | None = None,
        *,
        device: str | torch.device = "cpu",
        episode_end_utc: str | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.sim_config = load_simulation_config() if sim_config is None else sim_config
        self.cost_config = load_cost_config() if cost_config is None else cost_config
        self.mpc_config = load_mpc_config() if mpc_config is None else mpc_config
        self.episode_end_utc = episode_end_utc

        # 1. Load model checkpoint safely in eval mode
        ckpt_path = Path(checkpoint_path) if checkpoint_path is not None else DEFAULT_CHECKPOINT_PATH
        if not ckpt_path.exists():
            raise FileNotFoundError(f"GNN-Transformer checkpoint not found: {ckpt_path}")
        self.checkpoint_path = ckpt_path
        self.model = GNNTransformerForecaster.load_checkpoint(ckpt_path, map_location=self.device)
        self.model.eval()

        # 2. Build graph and edge index
        graph_data = build_graph()
        self.edge_index = torch.tensor(
            graph_data["edge_index"], dtype=torch.long, device=self.device
        )
        self.node_order = tuple(graph_data["metadata"]["node_order"])
        if self.node_order != GRAPH_NODE_ORDER:
            raise GNNTransformerError(f"Unexpected graph node order: {self.node_order}")

        # 3. Load preprocessing scalers
        prep_path = Path(preprocessing_path) if preprocessing_path is not None else DEFAULT_PREPROCESSING_PATH
        if not prep_path.exists():
            raise FileNotFoundError(f"Preprocessing artifact not found: {prep_path}")
        self.preprocessing_path = prep_path
        prep_json = json.loads(prep_path.read_text(encoding="utf-8"))
        self._load_scalers(prep_json)

        # 4. History storage
        self._history_source: dict[str, HourlyObservation] = {}
        if history_source is not None:
            if isinstance(history_source, Mapping):
                self._history_source = dict(history_source)
            else:
                for r in history_source:
                    self._history_source[r.timestamp_utc] = r

        self._initial_history_records: dict[str, NodeHistoricalRecord] = {}
        if initial_history is not None:
            for r in initial_history:
                self._initial_history_records[r.timestamp_utc] = NodeHistoricalRecord(
                    timestamp_utc=r.timestamp_utc,
                    building_loads_kw=dict(r.building_loads_kw),
                    solar_generation_kw=float(r.pv_available_kw),
                    wind_generation_kw=float(r.wind_available_kw),
                    campus_demand_kw=float(r.campus_load_kw),
                )
                self._history_source[r.timestamp_utc] = r

        self._history: dict[str, NodeHistoricalRecord] = {}
        self._latest_record: NodeHistoricalRecord | None = None

        # 5. Telemetry & diagnostics
        self.step_diagnostics: list[GNNMPCStepDiagnostic] = []
        self.solve_times: list[float] = []
        self.inference_times: list[float] = []
        self.step_actions: list[BatteryAction] = []
        self.fallback_count: int = 0
        self.total_history_fallback_steps: int = 0

        self._reset()

    def _load_scalers(self, prep_json: dict[str, Any]) -> None:
        """Extract train-fitted scaler parameters for node power features."""
        scalers = prep_json.get("scalers", {})
        demand_entities = scalers.get("synthetic_hypothetical_campus_v1_demand", {}).get("entities", {})
        solar_entities = scalers.get("synthetic_hypothetical_campus_v1_solar", {}).get("entities", {})
        wind_entities = scalers.get("synthetic_hypothetical_campus_v1_wind", {}).get("entities", {})

        self._node_means: list[float] = []
        self._node_stds: list[float] = []

        for node_id in GRAPH_NODE_ORDER:
            if node_id in {"B01", "B02", "B03", "B04", "B05"}:
                p = demand_entities[node_id]["building_load_kw"]
                self._node_means.append(float(p["mean"]))
                self._node_stds.append(float(p["scale"]))
            elif node_id == "PV01":
                p = solar_entities["PV01"]["available_generation_kw"]
                self._node_means.append(float(p["mean"]))
                self._node_stds.append(float(p["scale"]))
            elif node_id == "W01":
                p = wind_entities["W01"]["available_generation_kw"]
                self._node_means.append(float(p["mean"]))
                self._node_stds.append(float(p["scale"]))
            elif node_id == "BUS01":
                self._node_means.append(0.0)
                self._node_stds.append(1.0)
            else:
                raise ValueError(f"Unknown node ID: {node_id}")

        self._node_means_t = torch.tensor(self._node_means, dtype=torch.float32, device=self.device).reshape(1, 1, 8, 1)
        self._node_stds_t = torch.tensor(self._node_stds, dtype=torch.float32, device=self.device).reshape(1, 1, 8, 1)

    def _reset(self) -> None:
        """Reset episode state, restoring initial history buffer."""
        self._history = dict(self._initial_history_records)
        if self._initial_history_records:
            latest_ts = max(self._initial_history_records)
            self._latest_record = self._initial_history_records[latest_ts]
        else:
            self._latest_record = None

        self.step_diagnostics.clear()
        self.solve_times.clear()
        self.inference_times.clear()
        self.step_actions.clear()
        self.fallback_count = 0
        self.total_history_fallback_steps = 0

    def _select_action(self, observation: AgentObservation) -> BatteryAction:
        """Causal decision step: runs GNN-Transformer inference and solves MPC linear program."""
        import time

        # 1. Ingest completed state from observation
        completed_state = observation.state
        completed_ts = completed_state.timestamp_utc

        # Check if detailed node observation exists in history_source
        if completed_ts in self._history_source:
            row = self._history_source[completed_ts]
            record = NodeHistoricalRecord(
                timestamp_utc=completed_ts,
                building_loads_kw=dict(row.building_loads_kw),
                solar_generation_kw=float(row.pv_available_kw),
                wind_generation_kw=float(row.wind_available_kw),
                campus_demand_kw=float(row.campus_load_kw),
            )
        else:
            # Fallback when individual building loads are not available
            split_b = completed_state.campus_demand_kw / 5.0
            record = NodeHistoricalRecord(
                timestamp_utc=completed_ts,
                building_loads_kw={f"B0{k}": split_b for k in range(1, 6)},
                solar_generation_kw=float(completed_state.solar_generation_kw),
                wind_generation_kw=float(completed_state.wind_generation_kw),
                campus_demand_kw=float(completed_state.campus_demand_kw),
            )

        self._history[completed_ts] = record
        self._latest_record = record

        # 2. Decision timestamp
        dec_ts_str = observation.decision_timestamp_utc
        dec_dt = datetime.fromisoformat(dec_ts_str.replace("Z", "+00:00")).astimezone(timezone.utc)

        # 3. Horizon determination
        max_h = self.mpc_config.horizon_hours
        if self.episode_end_utc is not None:
            end_dt = datetime.fromisoformat(self.episode_end_utc.replace("Z", "+00:00")).astimezone(timezone.utc)
            rem_hours = int((end_dt - dec_dt).total_seconds() // 3600)
            horizon = max(1, min(max_h, rem_hours))
        else:
            horizon = max_h

        # 4. Extract 24 completed historical hours [dec_dt - 24h, ..., dec_dt - 1h]
        input_array = np.zeros((1, 24, 8, 1), dtype=np.float32)
        missing_count = 0

        for idx in range(24):
            hist_dt = dec_dt - timedelta(hours=24 - idx)
            hist_ts = hist_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

            # Strict non-anticipativity check: historical timestamp must strictly precede decision time
            if hist_dt >= dec_dt:
                raise RuntimeError(f"Causality violation: historical timestamp {hist_ts} >= decision {dec_ts_str}")

            if hist_ts in self._history:
                h_rec = self._history[hist_ts]
            elif hist_ts in self._history_source:
                row = self._history_source[hist_ts]
                h_rec = NodeHistoricalRecord(
                    timestamp_utc=hist_ts,
                    building_loads_kw=dict(row.building_loads_kw),
                    solar_generation_kw=float(row.pv_available_kw),
                    wind_generation_kw=float(row.wind_available_kw),
                    campus_demand_kw=float(row.campus_load_kw),
                )
                self._history[hist_ts] = h_rec
            else:
                missing_count += 1
                h_rec = self._latest_record

            if h_rec is not None:
                for n_idx, node_id in enumerate(GRAPH_NODE_ORDER):
                    if node_id in {"B01", "B02", "B03", "B04", "B05"}:
                        val = h_rec.building_loads_kw.get(node_id, h_rec.campus_demand_kw / 5.0)
                    elif node_id == "PV01":
                        val = h_rec.solar_generation_kw
                    elif node_id == "W01":
                        val = h_rec.wind_generation_kw
                    else:
                        val = 0.0
                    input_array[0, idx, n_idx, 0] = val
            else:
                # Cold-start fallback: nominal 50 kW per building, 0 solar/wind
                input_array[0, idx, :5, 0] = 50.0

        if missing_count > 0:
            self.total_history_fallback_steps += 1

        # 5. Scale input tensor and run GNN-Transformer forward pass
        t_inf_start = time.perf_counter()
        input_tensor = torch.tensor(input_array, dtype=torch.float32, device=self.device)
        scaled_input = (input_tensor - self._node_means_t) / self._node_stds_t

        with torch.no_grad():
            predictions = self.model(scaled_input, self.edge_index)

        # 6. Model output is directly in physical kW (target_unit: kW).
        # Project nonphysical negatives to zero.
        pred_kw = predictions.cpu().numpy()[0]  # [24, 8, 1]
        pred_kw[:, :7, 0] = np.maximum(pred_kw[:, :7, 0], 0.0)
        t_inf = time.perf_counter() - t_inf_start
        self.inference_times.append(t_inf)

        # 7. Aggregate forecasts across nodes
        # Index 0 corresponds to [dec_dt, dec_dt + 1h)
        target_timestamps: list[str] = []
        forecast_demands: list[float] = []
        forecast_solars: list[float] = []
        forecast_winds: list[float] = []

        for k in range(horizon):
            target_dt = dec_dt + timedelta(hours=k)
            target_timestamps.append(target_dt.strftime("%Y-%m-%dT%H:%M:%SZ"))

            campus_load = float(np.sum(pred_kw[k, :5, 0]))
            solar_power = float(pred_kw[k, 5, 0])
            wind_power = float(pred_kw[k, 6, 0])

            forecast_demands.append(campus_load)
            forecast_solars.append(solar_power)
            forecast_winds.append(wind_power)

        # 8. Tariff schedule for planning horizon
        import_prices = []
        export_prices = []
        for target_ts in target_timestamps:
            p_imp, p_exp = get_scheduled_tariff(target_ts)
            import_prices.append(p_imp)
            export_prices.append(p_exp)

        # 9. Formulate MPC input profile
        capacity_kwh = self.sim_config.battery.capacity_kwh
        initial_energy_kwh = observation.state.battery_soc * capacity_kwh

        profile = MPCInputProfile(
            initial_energy_kwh=initial_energy_kwh,
            demand_kw=tuple(forecast_demands),
            solar_generation_kw=tuple(forecast_solars),
            wind_generation_kw=tuple(forecast_winds),
            import_price=tuple(import_prices),
            export_price=tuple(export_prices),
        )

        # 10. Solve linear program
        t_solve_start = time.perf_counter()
        result: MPCSolverResult = solve_mpc_linear_program(
            profile=profile,
            sim_config=self.sim_config,
            mpc_config=self.mpc_config,
            cost_config=self.cost_config,
        )
        t_solve = time.perf_counter() - t_solve_start
        self.solve_times.append(t_solve)

        # 11. Extract first action
        fallback_used = False
        if result.success and len(result.planned_discharge_kw) > 0:
            first_discharge = result.planned_discharge_kw[0]
            first_charge = result.planned_charge_kw[0]
            action_kw = first_discharge - first_charge
            status_code = result.diagnostics.status
            has_sim_battery = result.diagnostics.has_simultaneous_battery_flow
        else:
            self.fallback_count += 1
            fallback_used = True
            action_kw = 0.0
            status_code = -1
            has_sim_battery = False

        action = BatteryAction(requested_power_kw=action_kw)
        self.step_actions.append(action)

        # 12. Record diagnostic
        diag = GNNMPCStepDiagnostic(
            decision_timestamp_utc=dec_ts_str,
            success=result.success,
            status=status_code,
            solve_time_seconds=t_solve,
            inference_time_seconds=t_inf,
            first_requested_power_kw=action_kw,
            horizon_hours=horizon,
            fallback_used=fallback_used,
            fallback_hours_count=missing_count,
            has_simultaneous_battery_flow=has_sim_battery,
            forecast_demand_kw=tuple(forecast_demands),
            forecast_solar_kw=tuple(forecast_solars),
            forecast_wind_kw=tuple(forecast_winds),
            target_timestamps_utc=tuple(target_timestamps),
        )
        self.step_diagnostics.append(diag)

        return action
