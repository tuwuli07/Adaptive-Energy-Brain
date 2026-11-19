"""Unified Comparative Evaluation Metrics for Microgrid Control.

Calculates comprehensive economic, physical, degradation, and reliability metrics
directly from physical simulator records, maintaining exact mathematical consistency
with the established project contracts.
"""

from dataclasses import asdict, dataclass
import math
from numbers import Real
from typing import Any, Mapping, Sequence

from src.evaluation.inventory_accounting import (
    compute_inventory_adjusted_cost,
    compute_replacement_reference_value,
)
from src.simulation.actions import finite_number
from src.simulation.config import SimulationConfig, load_simulation_config
from src.simulation.metrics import CostConfig, load_cost_config


@dataclass(frozen=True)
class ComparativeEvaluationMetrics:
    """Comprehensive standardized metrics for microgrid controller evaluation."""

    # Economic metrics (USD)
    raw_net_operating_cost_usd: float
    grid_import_cost_usd: float
    grid_export_revenue_usd: float
    battery_degradation_cost_usd: float
    unserved_demand_cost_usd: float

    # Battery inventory accounting (USD and kWh DC)
    initial_energy_kwh_dc: float
    terminal_energy_kwh_dc: float
    energy_change_kwh_dc: float
    inventory_reference_value_usd_per_kwh_dc: float
    terminal_inventory_credit_usd: float
    inventory_adjusted_cost_usd: float

    # Energy flows (kWh)
    grid_import_energy_kwh: float
    grid_export_energy_kwh: float
    campus_demand_energy_kwh: float
    solar_available_energy_kwh: float
    wind_available_energy_kwh: float
    renewable_curtailment_energy_kwh: float
    renewable_utilized_energy_kwh: float

    # Battery physical metrics
    battery_charge_energy_kwh_ac: float
    battery_discharge_energy_kwh_ac: float
    battery_throughput_kwh_ac: float
    full_cycle_equivalents: float
    initial_soc: float
    terminal_soc: float
    min_soc_observed: float
    max_soc_observed: float

    # Operational safety & reliability
    total_steps: int
    physical_violation_count: int
    simulator_adjustment_count: int
    solver_fallback_count: int
    simultaneous_flow_steps: int
    unserved_demand_steps: int

    # Execution timing (ms) - optional, 0.0 if not measured
    latency_ms_mean: float
    latency_ms_median: float
    latency_ms_max: float
    latency_ms_total: float

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to a standard serializable dictionary."""
        return asdict(self)


def calculate_comparative_metrics(
    records: Sequence[Mapping[str, Any]],
    sim_config: SimulationConfig | None = None,
    cost_config: CostConfig | None = None,
    *,
    solver_fallback_count: int = 0,
    timing_records_ms: Sequence[float] | None = None,
    simultaneous_flow_tolerance_kw: float = 1e-5,
) -> ComparativeEvaluationMetrics:
    """Compute comprehensive comparative evaluation metrics from step records.

    Args:
        records: Sequence of hourly step records from simulation environment replay.
        sim_config: Simulation configuration (defaults to loaded configs/simulation.yaml).
        cost_config: Cost configuration (defaults to loaded configs/simulation_costs.yaml).
        solver_fallback_count: Count of solver fallbacks recorded by controller.
        timing_records_ms: Optional sequence of per-step execution latencies in ms.
        simultaneous_flow_tolerance_kw: Numerical threshold for simultaneous charge/discharge.

    Returns:
        ComparativeEvaluationMetrics instance.
    """
    if not records:
        raise ValueError("Cannot compute metrics from empty records sequence.")

    if sim_config is None:
        sim_config = load_simulation_config()
    if cost_config is None:
        cost_config = load_cost_config()

    # Derived reference replacement rate: (0.12 + 0.02) / 0.95 = 0.14736842...
    v_ref = compute_replacement_reference_value(sim_config, cost_config)

    # Battery parameters
    capacity_kwh = float(sim_config.battery.capacity_kwh)
    c_deg = float(cost_config.battery_degradation_per_kwh_throughput)

    # State variables across time
    initial_soc = float(records[0]["battery_soc_before"])
    terminal_soc = float(records[-1]["battery_soc_after"])
    initial_energy_kwh = initial_soc * capacity_kwh
    terminal_energy_kwh = terminal_soc * capacity_kwh

    soc_values = [float(r["battery_soc_before"]) for r in records] + [terminal_soc]
    min_soc = min(soc_values)
    max_soc = max(soc_values)

    # Accumulate energy and costs
    grid_import_kwh = 0.0
    grid_export_kwh = 0.0
    demand_kwh = 0.0
    solar_kwh = 0.0
    wind_kwh = 0.0
    curtailment_kwh = 0.0
    renewable_used_kwh = 0.0

    charge_kwh_ac = 0.0
    discharge_kwh_ac = 0.0

    import_cost_usd = 0.0
    export_revenue_usd = 0.0
    unserved_cost_usd = 0.0

    physical_violations = 0
    simulator_adjustments = 0
    simultaneous_flows = 0
    unserved_steps = 0

    for r in records:
        imp_kwh = float(r.get("grid_import_energy_kwh", 0.0))
        exp_kwh = float(r.get("grid_export_energy_kwh", 0.0))
        grid_import_kwh += imp_kwh
        grid_export_kwh += exp_kwh

        demand_kwh += float(r.get("demand_energy_kwh", 0.0))
        solar_kwh += float(r.get("solar_available_energy_kwh", 0.0))
        wind_kwh += float(r.get("wind_available_energy_kwh", 0.0))
        curtailment_kwh += float(r.get("renewable_curtailed_energy_kwh", 0.0))
        renewable_used_kwh += float(r.get("renewable_utilized_energy_kwh", 0.0))

        chg_ac = float(r.get("battery_charge_energy_kwh", 0.0))
        dis_ac = float(r.get("battery_discharge_energy_kwh", 0.0))
        charge_kwh_ac += chg_ac
        discharge_kwh_ac += dis_ac

        import_cost_usd += float(r.get("import_cost", 0.0))
        export_revenue_usd += float(r.get("export_revenue", 0.0))
        unserved_c = float(r.get("unserved_demand_cost", 0.0))
        unserved_cost_usd += unserved_c
        if unserved_c > 0.0 or float(r.get("unserved_demand_kw", 0.0)) > 0.0:
            unserved_steps += 1

        physical_violations += int(r.get("violation_count", 0))
        simulator_adjustments += int(r.get("adjustment_count", 0))

        # Check for simultaneous charging and discharging flows
        chg_kw = float(r.get("battery_charge_kw", 0.0))
        dis_kw = float(r.get("battery_discharge_kw", 0.0))
        if chg_kw > simultaneous_flow_tolerance_kw and dis_kw > simultaneous_flow_tolerance_kw:
            simultaneous_flows += 1

    # Total AC throughput & degradation
    throughput_kwh_ac = charge_kwh_ac + discharge_kwh_ac
    degradation_cost_usd = throughput_kwh_ac * c_deg

    # Raw net operating cost = import_cost - export_revenue + degradation_cost + unserved_cost
    raw_net_operating_cost = (
        import_cost_usd - export_revenue_usd + degradation_cost_usd + unserved_cost_usd
    )

    # Full Cycle Equivalents: total AC throughput / (2 * usable capacity AC)
    # where nominal capacity is 500 kWh DC
    fce = throughput_kwh_ac / (2.0 * capacity_kwh)

    # Inventory adjustment
    adj = compute_inventory_adjusted_cost(
        raw_net_operating_cost=raw_net_operating_cost,
        initial_energy_kwh=initial_energy_kwh,
        terminal_energy_kwh=terminal_energy_kwh,
        reference_value_usd_per_kwh_dc=v_ref,
    )

    # Latency timing aggregation
    if timing_records_ms and len(timing_records_ms) > 0:
        sorted_times = sorted(timing_records_ms)
        n_times = len(sorted_times)
        lat_mean = float(sum(sorted_times) / n_times)
        lat_median = (
            float(sorted_times[n_times // 2])
            if n_times % 2 == 1
            else float((sorted_times[n_times // 2 - 1] + sorted_times[n_times // 2]) / 2.0)
        )
        lat_max = float(sorted_times[-1])
        lat_total = float(sum(sorted_times))
    else:
        lat_mean = 0.0
        lat_median = 0.0
        lat_max = 0.0
        lat_total = 0.0

    return ComparativeEvaluationMetrics(
        raw_net_operating_cost_usd=raw_net_operating_cost,
        grid_import_cost_usd=import_cost_usd,
        grid_export_revenue_usd=export_revenue_usd,
        battery_degradation_cost_usd=degradation_cost_usd,
        unserved_demand_cost_usd=unserved_cost_usd,
        initial_energy_kwh_dc=initial_energy_kwh,
        terminal_energy_kwh_dc=terminal_energy_kwh,
        energy_change_kwh_dc=adj["energy_change_kwh"],
        inventory_reference_value_usd_per_kwh_dc=v_ref,
        terminal_inventory_credit_usd=adj["inventory_adjustment_usd"],
        inventory_adjusted_cost_usd=adj["inventory_adjusted_net_operating_cost_usd"],
        grid_import_energy_kwh=grid_import_kwh,
        grid_export_energy_kwh=grid_export_kwh,
        campus_demand_energy_kwh=demand_kwh,
        solar_available_energy_kwh=solar_kwh,
        wind_available_energy_kwh=wind_kwh,
        renewable_curtailment_energy_kwh=curtailment_kwh,
        renewable_utilized_energy_kwh=renewable_used_kwh,
        battery_charge_energy_kwh_ac=charge_kwh_ac,
        battery_discharge_energy_kwh_ac=discharge_kwh_ac,
        battery_throughput_kwh_ac=throughput_kwh_ac,
        full_cycle_equivalents=fce,
        initial_soc=initial_soc,
        terminal_soc=terminal_soc,
        min_soc_observed=min_soc,
        max_soc_observed=max_soc,
        total_steps=len(records),
        physical_violation_count=physical_violations,
        simulator_adjustment_count=simulator_adjustments,
        solver_fallback_count=solver_fallback_count,
        simultaneous_flow_steps=simultaneous_flows,
        unserved_demand_steps=unserved_steps,
        latency_ms_mean=lat_mean,
        latency_ms_median=lat_median,
        latency_ms_max=lat_max,
        latency_ms_total=lat_total,
    )
