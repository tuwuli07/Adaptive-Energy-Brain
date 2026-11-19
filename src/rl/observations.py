"""Causal drl_observation_v1 encoding; never reads replay rows or fits statistics."""

from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from gymnasium.spaces import Box

from src.simulation.config import _UniqueKeyLoader
from src.simulation.state import AgentObservation


FEATURE_NAMES = (
    "battery_soc", "charge_limit_kw", "discharge_limit_kw", "campus_demand_kw",
    "solar_generation_kw", "wind_generation_kw", "grid_import_price",
    "grid_export_price_value", "grid_export_price_available",
    "grid_import_limit_kw", "grid_export_limit_kw", "hour_sin", "hour_cos",
    "weekday_sin", "weekday_cos",
)
SCHEMA = "drl_observation_v1"
POWER_REFERENCE_KW = 1000.0
PRICE_REFERENCE_PER_KWH = 1.0
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs/rl_environment.yaml"


def validate_adapter_config(path=DEFAULT_CONFIG):
    """Require the approved representation; changes require a new decision."""
    values = yaml.load(Path(path).read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    expected = {"version": 1, "observation_schema": SCHEMA,
                "power_reference_kw": POWER_REFERENCE_KW,
                "price_reference_per_kwh": PRICE_REFERENCE_PER_KWH}
    if (not isinstance(values, dict) or values != expected
            or type(values.get("version")) is not int
            or any(isinstance(values.get(k), bool) for k in
                   ("power_reference_kw", "price_reference_per_kwh"))):
        raise ValueError("Adapter configuration must match approved drl_observation_v1 scales.")


class ObservationEncoder:
    """Fixed feature order, kW/1000 and prices/1; no clipping or learned scaling.

    SOC/mask bounds are [0,1], calendar [-1,1], powers [0,inf]. Prices
    are [0,inf] or [-inf,inf] according to the physical negative-price policy.
    Returned values must always be finite and representable as float32.
    """

    def __init__(self, config):
        self.config = config
        low = np.zeros(15, dtype=np.float32)
        high = np.full(15, np.inf, dtype=np.float32)
        high[[0, 8]] = 1
        low[11:] = -1
        high[11:] = 1
        if config.grid.allow_negative_prices:
            low[6:8] = -np.inf
        self.space = Box(low, high, dtype=np.float32)

    def encode(self, observation):
        if not isinstance(observation, AgentObservation):
            raise ValueError("Expected AgentObservation.")
        if observation.state.config != self.config:
            raise ValueError("Observation equipment configuration changed.")
        if observation.forecasts:
            raise ValueError("Forecasts are disabled for drl_observation_v1.")
        f = observation.features()
        stamp = datetime.fromisoformat(observation.decision_timestamp_utc.replace("Z", "+00:00"))
        hour = 2 * np.pi * stamp.hour / 24
        weekday = 2 * np.pi * stamp.weekday() / 7
        export = f["grid_export_price"]
        values = np.array([
            f["battery_soc"], f["charge_limit_kw"] / POWER_REFERENCE_KW,
            f["discharge_limit_kw"] / POWER_REFERENCE_KW,
            f["campus_demand_kw"] / POWER_REFERENCE_KW,
            f["solar_generation_kw"] / POWER_REFERENCE_KW,
            f["wind_generation_kw"] / POWER_REFERENCE_KW,
            f["grid_import_price"] / PRICE_REFERENCE_PER_KWH,
            0 if export is None else export / PRICE_REFERENCE_PER_KWH,
            float(export is not None), f["grid_import_limit_kw"] / POWER_REFERENCE_KW,
            f["grid_export_limit_kw"] / POWER_REFERENCE_KW,
            np.sin(hour), np.cos(hour), np.sin(weekday), np.cos(weekday),
        ], dtype=np.float64)
        if (not np.isfinite(values).all()
                or np.any(np.abs(values) > np.finfo(np.float32).max)):
            raise ValueError("Observation has nonfinite or unrepresentable float32 values.")
        result = values.astype(np.float32)
        if np.any((values != 0) & (result == 0)):
            raise ValueError("Observation has unrepresentable float32 underflow.")
        if not self.space.contains(result):
            raise ValueError("Observation lies outside its declared space.")
        return result
