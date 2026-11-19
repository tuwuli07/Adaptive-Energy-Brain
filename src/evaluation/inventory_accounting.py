"""Secondary inventory-adjusted cost convention for microgrid evaluation.

This module provides a secondary economic reporting metric that accounts for
changes in stored battery energy inventory across an evaluation horizon.

Convention:
    adjusted_cost = raw_net_operating_cost - (terminal_energy_kwh - initial_energy_kwh) * reference_value

For the synthetic campus scenario, reference_value is the off-peak replacement cost:
    reference_value = (off_peak_tariff + degradation_per_ac_kwh) / charge_efficiency
    expected = (0.12 + 0.02) / 0.95 = 0.14736842105263157 USD/kWh DC

This is an off-peak replacement-cost convention, not a universal electricity-market valuation.
Primary evaluation dimensions (raw net operating cost, terminal energy, terminal SOC,
and Pareto dominance) remain unchanged.
"""

import math
from numbers import Real
from pathlib import Path

from src.baselines.price_aware import load_price_aware_config
from src.simulation.actions import finite_number
from src.simulation.config import SimulationConfig, load_simulation_config
from src.simulation.metrics import CostConfig, load_cost_config

ROOT = Path(__file__).resolve().parents[2]


def compute_replacement_reference_value(
    config=None,
    costs=None,
    *,
    baseline_config=None,
    off_peak_price_per_kwh=None,
) -> float:
    """Calculate the off-peak replacement cost per stored DC kWh.

    Inputs are derived dynamically from validated configuration contracts:
      - off_peak_price_per_kwh: from baseline price_aware low_price_threshold or parameter
      - degradation_per_ac_kwh: from CostConfig battery_degradation_per_kwh_throughput
      - charge_efficiency: from SimulationConfig battery charge_efficiency

    Formula:
        (p_offpeak + c_degradation) / eta_charge
    Units:
        currency / kWh DC
    """
    if config is None:
        config = load_simulation_config()
    elif not isinstance(config, SimulationConfig):
        raise TypeError("config must be a SimulationConfig.")

    if costs is None:
        costs = load_cost_config()
    elif not isinstance(costs, CostConfig):
        raise TypeError("costs must be a CostConfig.")

    if off_peak_price_per_kwh is None:
        baseline_path = baseline_config if baseline_config is not None else ROOT / "configs/baselines.yaml"
        price_cfg = load_price_aware_config(baseline_path)
        p_off = price_cfg.low_price_threshold
    else:
        p_off = finite_number(off_peak_price_per_kwh, "off_peak_price_per_kwh")
        if p_off < 0:
            raise ValueError("off_peak_price_per_kwh must be nonnegative.")

    c_deg = costs.battery_degradation_per_kwh_throughput
    eta_c = config.battery.charge_efficiency

    if eta_c <= 0 or eta_c > 1.0:
        raise ValueError(f"Invalid charge_efficiency: {eta_c}")

    reference_value = (p_off + c_deg) / eta_c
    return float(reference_value)


def compute_inventory_adjusted_cost(
    raw_net_operating_cost: float,
    initial_energy_kwh: float,
    terminal_energy_kwh: float,
    reference_value_usd_per_kwh_dc: float,
) -> dict[str, float]:
    """Calculate secondary inventory adjustment and adjusted net operating cost.

    Sign convention:
      - If terminal_energy > initial_energy (inventory accumulated):
        adjustment is positive (credit), adjusted_cost is LOWER than raw cost.
      - If terminal_energy < initial_energy (inventory depleted):
        adjustment is negative (debit), adjusted_cost is HIGHER than raw cost.
      - If terminal_energy == initial_energy:
        adjustment is zero, adjusted_cost EQUALS raw cost.
    """
    raw_cost = finite_number(raw_net_operating_cost, "raw_net_operating_cost")
    init_e = finite_number(initial_energy_kwh, "initial_energy_kwh")
    term_e = finite_number(terminal_energy_kwh, "terminal_energy_kwh")
    ref_val = finite_number(reference_value_usd_per_kwh_dc, "reference_value_usd_per_kwh_dc")

    if init_e < 0 or term_e < 0:
        raise ValueError("Stored energy must be nonnegative.")
    if ref_val < 0:
        raise ValueError("Reference value must be nonnegative.")

    energy_change_kwh = term_e - init_e
    inventory_adjustment_usd = energy_change_kwh * ref_val
    adjusted_cost = raw_cost - inventory_adjustment_usd

    return {
        "initial_energy_kwh": float(init_e),
        "terminal_energy_kwh": float(term_e),
        "energy_change_kwh": float(energy_change_kwh),
        "reference_value_usd_per_kwh_dc": float(ref_val),
        "inventory_adjustment_usd": float(inventory_adjustment_usd),
        "inventory_adjusted_net_operating_cost_usd": float(adjusted_cost),
    }
