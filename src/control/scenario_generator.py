"""Causal uncertainty scenario generator for risk-aware microgrid control.

Generates physically plausible, causal uncertainty scenarios representing forecast
errors in demand, solar, wind, and unannounced electricity price spikes over
the planning horizon H.
"""

from dataclasses import dataclass
import math
from typing import Sequence
import numpy as np

from src.simulation.actions import finite_number


@dataclass(frozen=True)
class UncertaintyScenario:
    """A single discrete uncertainty trajectory over the planning horizon H."""

    scenario_id: int
    name: str
    probability: float
    demand_kw: tuple[float, ...]
    solar_generation_kw: tuple[float, ...]
    wind_generation_kw: tuple[float, ...]
    import_price: tuple[float, ...]
    export_price: tuple[float, ...]
    description: str

    def __post_init__(self):
        p = finite_number(self.probability, "probability")
        if p <= 0.0 or p > 1.0:
            raise ValueError(f"Scenario probability must be in (0, 1], got {p}")
        object.__setattr__(self, "probability", p)

        h = len(self.demand_kw)
        if h <= 0:
            raise ValueError("Scenario demand profile cannot be empty.")

        for name in ("solar_generation_kw", "wind_generation_kw", "import_price", "export_price"):
            seq = getattr(self, name)
            if len(seq) != h:
                raise ValueError(f"Profile length mismatch in scenario {self.scenario_id}: {name} len {len(seq)} != {h}")

        object.__setattr__(self, "demand_kw", tuple(max(0.0, finite_number(x, "demand_kw")) for x in self.demand_kw))
        object.__setattr__(self, "solar_generation_kw", tuple(max(0.0, finite_number(x, "solar_generation_kw")) for x in self.solar_generation_kw))
        object.__setattr__(self, "wind_generation_kw", tuple(max(0.0, finite_number(x, "wind_generation_kw")) for x in self.wind_generation_kw))
        object.__setattr__(self, "import_price", tuple(max(0.0, finite_number(x, "import_price")) for x in self.import_price))
        object.__setattr__(self, "export_price", tuple(max(0.0, finite_number(x, "export_price")) for x in self.export_price))

    @property
    def horizon_hours(self) -> int:
        return len(self.demand_kw)


class CausalScenarioGenerator:
    """Generates structured, reproducible uncertainty scenarios from a nominal forecast.

    Enforces:
      - Strictly causal information usage (derived from nominal point forecast at t0).
      - Conservation of scheduled non-spike tariffs unless spike scenario is selected.
      - Physical clamping to generation capacities (PV 300 kW, Wind 150 kW).
      - Exact probability normalization (sum(p_s) == 1.0).
    """

    def __init__(
        self,
        seed: int = 42,
        demand_stress_multiplier: float = 1.20,
        renewable_scarcity_multiplier: float = 0.50,
        price_spike_multiplier: float = 3.0,
        pv_capacity_kw: float = 300.0,
        wind_capacity_kw: float = 150.0,
    ):
        self.seed = int(seed)
        self.demand_stress_mult = finite_number(demand_stress_multiplier, "demand_stress_multiplier")
        self.renewable_scarcity_mult = finite_number(renewable_scarcity_multiplier, "renewable_scarcity_multiplier")
        self.price_spike_mult = finite_number(price_spike_multiplier, "price_spike_multiplier")
        self.pv_capacity_kw = finite_number(pv_capacity_kw, "pv_capacity_kw")
        self.wind_capacity_kw = finite_number(wind_capacity_kw, "wind_capacity_kw")
        self._rng = np.random.default_rng(self.seed)

    def generate_scenarios(
        self,
        nominal_demand_kw: Sequence[float],
        nominal_solar_kw: Sequence[float],
        nominal_wind_kw: Sequence[float],
        scheduled_import_price: Sequence[float],
        scheduled_export_price: Sequence[float],
        num_scenarios: int = 5,
    ) -> list[UncertaintyScenario]:
        """Generate structured uncertainty scenarios with normalized probabilities."""
        h = len(nominal_demand_kw)
        if h <= 0:
            raise ValueError("Horizon must be positive.")

        dem = np.array(nominal_demand_kw, dtype=np.float64)
        sol = np.array(nominal_solar_kw, dtype=np.float64)
        wnd = np.array(nominal_wind_kw, dtype=np.float64)
        imp = np.array(scheduled_import_price, dtype=np.float64)
        exp = np.array(scheduled_export_price, dtype=np.float64)

        scenarios = []

        # Scenario 0: Nominal Baseline (Expected conditions)
        scenarios.append(
            UncertaintyScenario(
                scenario_id=0,
                name="Nominal",
                probability=0.40,
                demand_kw=tuple(dem.tolist()),
                solar_generation_kw=tuple(np.clip(sol, 0.0, self.pv_capacity_kw).tolist()),
                wind_generation_kw=tuple(np.clip(wnd, 0.0, self.wind_capacity_kw).tolist()),
                import_price=tuple(imp.tolist()),
                export_price=tuple(exp.tolist()),
                description="Nominal neural/seasonal point forecast profile",
            )
        )

        if num_scenarios >= 2:
            # Scenario 1: Demand Surge (+20% load surge during daytime/working hours)
            dem_surge = dem.copy()
            # Daytime hours 7-19 have highest surge
            surge_factors = 1.0 + (self.demand_stress_mult - 1.0) * np.linspace(0.8, 1.2, h)
            dem_surge = dem_surge * surge_factors
            scenarios.append(
                UncertaintyScenario(
                    scenario_id=1,
                    name="High Demand",
                    probability=0.20,
                    demand_kw=tuple(dem_surge.tolist()),
                    solar_generation_kw=tuple(np.clip(sol, 0.0, self.pv_capacity_kw).tolist()),
                    wind_generation_kw=tuple(np.clip(wnd, 0.0, self.wind_capacity_kw).tolist()),
                    import_price=tuple(imp.tolist()),
                    export_price=tuple(exp.tolist()),
                    description="Elevated building electricity demand stress (+20%)",
                )
            )

        if num_scenarios >= 3:
            # Scenario 2: Renewable Scarcity (Sudden cloud cover and wind lull)
            sol_scarce = sol * self.renewable_scarcity_mult
            wnd_scarce = wnd * max(0.2, self.renewable_scarcity_mult * 0.8)
            scenarios.append(
                UncertaintyScenario(
                    scenario_id=2,
                    name="Renewable Scarcity",
                    probability=0.20,
                    demand_kw=tuple(dem.tolist()),
                    solar_generation_kw=tuple(np.clip(sol_scarce, 0.0, self.pv_capacity_kw).tolist()),
                    wind_generation_kw=tuple(np.clip(wnd_scarce, 0.0, self.wind_capacity_kw).tolist()),
                    import_price=tuple(imp.tolist()),
                    export_price=tuple(exp.tolist()),
                    description="Cloud cover and wind lull (-50% solar, -60% wind)",
                )
            )

        if num_scenarios >= 4:
            # Scenario 3: Real-Time Price Spike during peak tariff hours
            imp_spike = imp.copy()
            # Spikes apply to hours where scheduled tariff is high (e.g. on-peak hours >= median price)
            med_price = np.median(imp)
            for k in range(h):
                if imp[k] >= med_price:
                    imp_spike[k] = imp[k] * self.price_spike_mult
            scenarios.append(
                UncertaintyScenario(
                    scenario_id=3,
                    name="Price Spike",
                    probability=0.10,
                    demand_kw=tuple(dem.tolist()),
                    solar_generation_kw=tuple(np.clip(sol, 0.0, self.pv_capacity_kw).tolist()),
                    wind_generation_kw=tuple(np.clip(wnd, 0.0, self.wind_capacity_kw).tolist()),
                    import_price=tuple(imp_spike.tolist()),
                    export_price=tuple(exp.tolist()),
                    description="Unannounced dynamic grid tariff spike during peak hours (3x tariff)",
                )
            )

        if num_scenarios >= 5:
            # Scenario 4: Compound Adverse Stress (Demand + Scarcity + Price Spike)
            dem_compound = dem * (1.0 + (self.demand_stress_mult - 1.0) * 0.75)
            sol_compound = sol * (self.renewable_scarcity_mult * 1.2)
            wnd_compound = wnd * (self.renewable_scarcity_mult * 1.2)
            imp_compound = imp.copy()
            med_price = np.median(imp)
            for k in range(h):
                if imp[k] >= med_price:
                    imp_compound[k] = imp[k] * (self.price_spike_mult * 0.8)

            scenarios.append(
                UncertaintyScenario(
                    scenario_id=4,
                    name="Compound Stress",
                    probability=0.10,
                    demand_kw=tuple(dem_compound.tolist()),
                    solar_generation_kw=tuple(np.clip(sol_compound, 0.0, self.pv_capacity_kw).tolist()),
                    wind_generation_kw=tuple(np.clip(wnd_compound, 0.0, self.wind_capacity_kw).tolist()),
                    import_price=tuple(imp_compound.tolist()),
                    export_price=tuple(exp.tolist()),
                    description="Combined adverse event: +15% demand, -40% renewables, 2.4x peak tariff",
                )
            )

        # Normalize probabilities so they sum strictly to 1.0
        tot_prob = sum(s.probability for s in scenarios)
        normalized = []
        for s in scenarios:
            normalized.append(
                UncertaintyScenario(
                    scenario_id=s.scenario_id,
                    name=s.name,
                    probability=s.probability / tot_prob,
                    demand_kw=s.demand_kw,
                    solar_generation_kw=s.solar_generation_kw,
                    wind_generation_kw=s.wind_generation_kw,
                    import_price=s.import_price,
                    export_price=s.export_price,
                    description=s.description,
                )
            )

        return normalized
