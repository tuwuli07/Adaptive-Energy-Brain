"""Phase 12: Adaptive Policy Selection for Multi-Regime Energy Management.

Dynamically selects among specialized deployable controllers (GNN-Transformer MPC,
Seasonal Persistence MPC, CVaR Risk-Aware MPC, and DRO Robust MPC) based exclusively
on causally observable system conditions at decision timestamp t.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time
from typing import Any

import numpy as np
import yaml

from src.baselines.base import BaselineController
from src.control.cvar_mpc import CVaRRiskAwareMPCController
from src.control.dro_mpc import DistributionallyRobustMPCController
from src.control.gnn_transformer_mpc import GNNTransformerMPCController
from src.control.seasonal_persistence_mpc import SeasonalPersistenceMPCController
from src.simulation.actions import BatteryAction
from src.simulation.config import SimulationConfig, load_simulation_config, _UniqueKeyLoader
from src.simulation.data_loader import HourlyObservation
from src.simulation.state import AgentObservation

logger = logging.getLogger("adaptive_policy")
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "adaptive_policy_selector.yaml"


@dataclass(frozen=True)
class AdaptiveSelectorConfig:
    version: int
    method: str
    meta_learning_status: str
    peak_tariff_threshold_usd: float
    extreme_price_spike_usd: float
    solar_capture_min_kw: float
    solar_window_start_hour: int
    solar_window_end_hour: int
    peak_window_start_hour: int
    peak_window_end_hour: int
    min_discharge_soc: float
    max_charge_soc: float
    compound_deficit_kw: float
    fallback_order: list[str]


def load_adaptive_selector_config(path: Path | str | None = None) -> AdaptiveSelectorConfig:
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=_UniqueKeyLoader)
    thresholds = data.get("thresholds", {})
    return AdaptiveSelectorConfig(
        version=int(data.get("version", 1)),
        method=str(data.get("method", "adaptive_regime_routing")),
        meta_learning_status=str(data.get("meta_learning_status", "future_work")),
        peak_tariff_threshold_usd=float(thresholds.get("peak_tariff_threshold_usd", 0.16)),
        extreme_price_spike_usd=float(thresholds.get("extreme_price_spike_usd", 0.25)),
        solar_capture_min_kw=float(thresholds.get("solar_capture_min_kw", 35.0)),
        solar_window_start_hour=int(thresholds.get("solar_window_start_hour", 8)),
        solar_window_end_hour=int(thresholds.get("solar_window_end_hour", 16)),
        peak_window_start_hour=int(thresholds.get("peak_window_start_hour", 15)),
        peak_window_end_hour=int(thresholds.get("peak_window_end_hour", 20)),
        min_discharge_soc=float(thresholds.get("min_discharge_soc", 0.20)),
        max_charge_soc=float(thresholds.get("max_charge_soc", 0.85)),
        compound_deficit_kw=float(thresholds.get("compound_deficit_kw", 180.0)),
        fallback_order=list(data.get("fallback_order", ["mpc_gnn_transformer", "mpc_seasonal_persistence", "zero_action"])),
    )


@dataclass
class PolicySelectionDecision:
    step_index: int
    timestamp_utc: str
    selected_controller_id: str
    regime_name: str
    rationale: str
    observable_features: dict[str, Any]
    requested_power_kw: float
    fallback_used: bool
    solve_latency_ms: float


class AdaptivePolicySelector(BaselineController):
    """Contextual regime-aware controller selector.
    
    Observes causal state at decision time t, determines the operational regime,
    and dispatches execution to the optimal candidate controller.
    """

    def __init__(
        self,
        config: AdaptiveSelectorConfig | None = None,
        candidate_controllers: dict[str, BaselineController] | None = None,
        sim_config: SimulationConfig | None = None,
        history_source: Mapping[str, HourlyObservation] | None = None,
    ):
        self.config = config or load_adaptive_selector_config()
        self.sim_config = sim_config or load_simulation_config()
        self.history_source = history_source

        # Initialized or lazy-loaded controllers
        self._candidate_controllers: dict[str, BaselineController] = candidate_controllers or {}
        self._initialized = candidate_controllers is not None

        self.selection_history: list[PolicySelectionDecision] = []
        self._step_counter = 0

    def _ensure_controllers_initialized(self):
        if self._initialized:
            return

        logger.info("Initializing candidate controllers for Adaptive Policy Selector...")
        try:
            gnn = GNNTransformerMPCController(history_source=self.history_source)
            self._candidate_controllers["mpc_gnn_transformer"] = gnn
        except Exception as e:
            logger.warning(f"Could not load GNN-Transformer MPC: {e}")

        try:
            pers = SeasonalPersistenceMPCController(history_source=self.history_source)
            self._candidate_controllers["mpc_seasonal_persistence"] = pers
        except Exception as e:
            logger.warning(f"Could not load Seasonal Persistence MPC: {e}")

        try:
            cvar = CVaRRiskAwareMPCController(history_source=self.history_source)
            self._candidate_controllers["mpc_cvar_balanced"] = cvar
        except Exception as e:
            logger.warning(f"Could not load CVaR MPC: {e}")

        try:
            dro = DistributionallyRobustMPCController(history_source=self.history_source)
            self._candidate_controllers["mpc_dro_robust"] = dro
        except Exception as e:
            logger.warning(f"Could not load DRO Robust MPC: {e}")

        self._initialized = True

    def _reset(self) -> None:
        self.selection_history.clear()
        self._step_counter = 0
        for ctrl in self._candidate_controllers.values():
            if hasattr(ctrl, "reset"):
                ctrl.reset()

    def extract_observable_features(self, observation: AgentObservation) -> dict[str, Any]:
        """Extract purely causal, observable features available at decision timestamp t."""
        if hasattr(observation, "decision_timestamp_utc"):
            ts_str = str(observation.decision_timestamp_utc)
        elif hasattr(observation, "timestamp"):
            ts_str = str(observation.timestamp)
        else:
            ts_str = "2026-11-10T12:00:00Z"

        # Parse timestamp safely
        try:
            dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            hour = dt.hour
        except Exception:
            hour = getattr(observation, "current_hour", 12)

        # Extract state fields
        if hasattr(observation, "state") and observation.state is not None:
            soc = float(getattr(observation.state, "battery_soc", 0.5))
            import_price = float(getattr(observation.state, "grid_import_price", 0.12))
            export_price = float(getattr(observation.state, "grid_export_price", 0.04) or 0.04)
            solar_kw = float(getattr(observation.state, "solar_generation_kw", 0.0))
            wind_kw = float(getattr(observation.state, "wind_generation_kw", 0.0))
            demand_kw = float(getattr(observation.state, "campus_demand_kw", 0.0))
        else:
            soc = float(getattr(observation, "battery_soc", 0.5))
            import_price = float(getattr(observation, "current_tariff", 0.12))
            export_price = float(getattr(observation, "export_tariff", 0.04))
            solar_kw = float(getattr(observation, "solar_kw", 0.0))
            wind_kw = float(getattr(observation, "wind_kw", 0.0))
            demand_kw = float(getattr(observation, "demand_kw", 0.0))

        if demand_kw == 0.0 and hasattr(observation, "recent_demand_history") and observation.recent_demand_history:
            demand_kw = float(observation.recent_demand_history[-1])
        if solar_kw == 0.0 and hasattr(observation, "recent_solar_history") and observation.recent_solar_history:
            solar_kw = float(observation.recent_solar_history[-1])
        if wind_kw == 0.0 and hasattr(observation, "recent_wind_history") and observation.recent_wind_history:
            wind_kw = float(observation.recent_wind_history[-1])

        is_peak_tariff = bool(getattr(observation, "is_peak_tariff", (15 <= hour < 20)))

        renewables_kw = solar_kw + wind_kw
        net_deficit_kw = demand_kw - renewables_kw

        # Stress indicators
        is_extreme_price = import_price >= self.config.extreme_price_spike_usd
        is_high_tariff = is_peak_tariff or (import_price >= self.config.peak_tariff_threshold_usd)
        is_solar_window = self.config.solar_window_start_hour <= hour <= self.config.solar_window_end_hour
        has_abundant_solar = is_solar_window and (solar_kw >= self.config.solar_capture_min_kw)

        return {
            "timestamp_utc": ts_str,
            "hour": hour,
            "soc": round(soc, 4),
            "import_price_usd": round(import_price, 4),
            "export_price_usd": round(export_price, 4),
            "demand_kw": round(demand_kw, 2),
            "solar_kw": round(solar_kw, 2),
            "wind_kw": round(wind_kw, 2),
            "renewables_kw": round(renewables_kw, 2),
            "net_deficit_kw": round(net_deficit_kw, 2),
            "is_peak_tariff": is_high_tariff,
            "is_extreme_price": is_extreme_price,
            "is_solar_window": is_solar_window,
            "has_abundant_solar": has_abundant_solar,
        }

    def determine_regime_and_policy(self, features: dict[str, Any]) -> tuple[str, str, str]:
        """Classify operational context and select the optimal controller.
        
        Returns:
            (selected_controller_id, regime_name, rationale)
        """
        # Regime 1: Extreme Price Spike / Compound Adverse Stress -> DRO Robust MPC
        if features["is_extreme_price"] or (
            features["is_peak_tariff"]
            and features["net_deficit_kw"] >= self.config.compound_deficit_kw
            and features["renewables_kw"] < 25.0
        ):
            return (
                "mpc_dro_robust",
                "COMPOUND_STRESS_ROBUST",
                f"Severe net deficit ({features['net_deficit_kw']:.1f} kW) or price surge (${features['import_price_usd_per_kwh' if 'import_price_usd_per_kwh' in features else 'import_price_usd']:.2f}/kWh). "
                "DRO hedges against adverse probability redistribution and protects battery reserves.",
            )

        # Regime 2: Peak Tariff Shaving & Tail Risk Hedging -> CVaR Risk-Aware MPC
        if features["is_peak_tariff"] and features["soc"] >= self.config.min_discharge_soc:
            return (
                "mpc_cvar_balanced",
                "PEAK_TARIFF_TAIL_HEDGING",
                f"Peak tariff window (${features['import_price_usd']:.2f}/kWh) with battery reserve (SOC {features['soc']:.2f}). "
                "CVaR minimizes tail purchase risk and shaves peak grid imports.",
            )

        # Regime 3: Solar Capture & Renewable Utilization -> GNN-Transformer MPC
        if features["has_abundant_solar"] and features["soc"] <= self.config.max_charge_soc:
            return (
                "mpc_gnn_transformer",
                "SOLAR_SELF_CONSUMPTION",
                f"Daytime solar surplus ({features['solar_kw']:.1f} kW) with battery charging headroom (SOC {features['soc']:.2f}). "
                "GNN spatial-temporal lookahead coordinates battery charge to absorb clean power.",
            )

        # Regime 4: Nominal Base Operations -> GNN-Transformer MPC (or Seasonal Persistence)
        return (
            "mpc_gnn_transformer",
            "NOMINAL_BASE_DISPATCH",
            f"Standard off-peak operating conditions (${features['import_price_usd']:.2f}/kWh). "
            "Deterministic lookahead manages off-peak pre-charging and economic arbitrage.",
        )

    def _select_action(self, observation: AgentObservation) -> BatteryAction:
        self._ensure_controllers_initialized()
        t0 = time.perf_counter()

        features = self.extract_observable_features(observation)
        primary_id, regime, rationale = self.determine_regime_and_policy(features)

        # Execution with fallback cascade
        selected_id = primary_id
        fallback_used = False
        action = None

        candidate = self._candidate_controllers.get(primary_id)
        if candidate is not None:
            try:
                action = candidate.select_action(observation)
            except Exception as e:
                logger.warning(f"Controller {primary_id} failed: {e}. Executing fallback sequence.")
                fallback_used = True

        if action is None:
            fallback_used = True
            for fallback_id in self.config.fallback_order:
                fb_ctrl = self._candidate_controllers.get(fallback_id)
                if fb_ctrl is not None:
                    try:
                        action = fb_ctrl.select_action(observation)
                        selected_id = fallback_id
                        break
                    except Exception:
                        continue

        if action is None:
            # Final physical safety fallback: zero dispatch
            action = BatteryAction(0.0)
            selected_id = "zero_action_safety_fallback"

        # Enforce physical inverter limits
        p_max = float(self.sim_config.battery.charge_limit_kw)
        clamped_power = float(np.clip(action.requested_power_kw, -p_max, p_max))
        final_action = BatteryAction(clamped_power)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        decision = PolicySelectionDecision(
            step_index=self._step_counter,
            timestamp_utc=features["timestamp_utc"],
            selected_controller_id=selected_id,
            regime_name=regime,
            rationale=rationale,
            observable_features=features,
            requested_power_kw=clamped_power,
            fallback_used=fallback_used,
            solve_latency_ms=round(elapsed_ms, 2),
        )
        self.selection_history.append(decision)
        self._step_counter += 1

        return final_action
