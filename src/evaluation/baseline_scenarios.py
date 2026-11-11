"""Deterministic stress scenarios for same-condition baseline comparisons."""

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

from src.baselines import available_controllers
from src.evaluation.baseline_runner import (
    DEFAULT_SIMULATION_CONFIG,
    _sha256,
    run_baseline_experiments,
)
from src.simulation.config import SimulationConfig, load_simulation_config
from src.simulation.data_loader import (
    HourlyObservation,
    SimulationData,
    load_simulation_data,
    load_simulation_split,
)


_SCENARIO_DEFINITIONS = {
    "normal_operation": {
        "scenario_id": "normal_operation_v1",
        "label": "Normal operation",
        "modification": "No changes to input observations or simulation limits.",
    },
    "low_renewable": {
        "scenario_id": "low_renewable_v1",
        "label": "Low renewable generation",
        "renewable_scale": 0.25,
    },
    "high_campus_demand": {
        "scenario_id": "high_campus_demand_v1",
        "label": "High campus demand",
        "demand_scale": 1.25,
    },
    "electricity_price_spikes": {
        "scenario_id": "electricity_price_spikes_v1",
        "label": "Electricity price spikes",
        "price_multiplier": 4.0,
        "minimum_spike_price": 1.0,
    },
    "limited_grid_import": {
        "scenario_id": "limited_grid_import_v1",
        "label": "Limited grid import capacity",
        "grid_import_fraction": 0.1,
        "grid_import_cap_kw": 100.0,
        "unserved_load_handling": "record",
    },
}


def available_scenarios():
    """Return supported deterministic scenario names in their stable order."""
    return tuple(_SCENARIO_DEFINITIONS)


def _scenario_description(data, name):
    definition = deepcopy(_SCENARIO_DEFINITIONS[name])
    definition["scenario_name"] = name
    definition["modified_or_synthetic"] = name != "normal_operation"
    definition["source_data_synthetic"] = any(
        summary in {"synthetic", "simulated"}
        for summary in data.metadata["source_origin_summary"].values()
    )
    return definition


def apply_baseline_scenario(data, name, simulation_config):
    """Return independent scenario observations and physical configuration.

    The validated input snapshot is never modified. Scenario transformations
    are deterministic and are recorded as explicit metadata.
    """
    if not isinstance(data, SimulationData):
        raise TypeError("data must be validated SimulationData.")
    if name not in _SCENARIO_DEFINITIONS:
        raise ValueError(
            f"Unknown scenario {name!r}; available: {', '.join(available_scenarios())}."
        )
    if not isinstance(simulation_config, SimulationConfig):
        raise TypeError("simulation_config must be a validated SimulationConfig.")

    scenario = _scenario_description(data, name)
    if name == "electricity_price_spikes":
        start = max(1, len(data) // 2)
        duration = max(1, (len(data) - 1 + 9) // 10)
        stop = min(len(data), start + duration)
        scenario["spike_start_index"] = start
        scenario["spike_end_index_exclusive"] = stop
        scenario["spike_timestamps_utc"] = [
            data[index].timestamp_utc for index in range(start, stop)
        ]

    rows = []
    for index, row in enumerate(data):
        pv = dict(row.pv_asset_powers_kw)
        wind = dict(row.wind_asset_powers_kw)
        buildings = dict(row.building_loads_kw)
        import_price = row.import_price
        if name == "low_renewable":
            scale = scenario["renewable_scale"]
            pv = {asset: value * scale for asset, value in pv.items()}
            wind = {asset: value * scale for asset, value in wind.items()}
        elif name == "high_campus_demand":
            scale = scenario["demand_scale"]
            buildings = {building: value * scale for building, value in buildings.items()}
        elif name == "electricity_price_spikes" and start <= index < stop:
            import_price = max(
                import_price * scenario["price_multiplier"],
                scenario["minimum_spike_price"],
            )
        rows.append(HourlyObservation(
            timestamp_utc=row.timestamp_utc,
            pv_available_kw=sum(pv.values()),
            wind_available_kw=sum(wind.values()),
            campus_load_kw=sum(buildings.values()),
            import_price=import_price,
            export_price=row.export_price,
            building_loads_kw=MappingProxyType(buildings),
            pv_asset_powers_kw=MappingProxyType(pv),
            wind_asset_powers_kw=MappingProxyType(wind),
        ))

    metadata = data.metadata
    metadata["scenario"] = deepcopy(scenario)
    transformed = SimulationData(rows, metadata)
    config = simulation_config
    if name == "limited_grid_import":
        original_limit = simulation_config.grid.import_limit_kw
        constrained_limit = min(
            original_limit * scenario["grid_import_fraction"],
            scenario["grid_import_cap_kw"],
        )
        scenario["original_grid_import_limit_kw"] = original_limit
        scenario["grid_import_limit_kw"] = constrained_limit
        metadata["scenario"] = deepcopy(scenario)
        transformed = SimulationData(rows, metadata)
        config = replace(
            simulation_config,
            grid=replace(simulation_config.grid, import_limit_kw=constrained_limit),
            unserved_load=replace(
                simulation_config.unserved_load,
                handling=scenario["unserved_load_handling"],
            ),
        )
    return transformed, config, scenario


def run_baseline_scenarios(
    *,
    assets,
    rating_origin,
    output,
    manifest=None,
    split_root=None,
    partition=None,
    scenarios=None,
    controllers=None,
    start=None,
    end=None,
    simulation_config=None,
    costs=None,
    baseline_config=None,
):
    """Run every selected controller under identical deterministic scenarios."""
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(
            f"Scenario output already exists: {destination}; choose a fresh directory."
        )
    if (manifest is None) == (split_root is None):
        raise ValueError("Supply exactly one of manifest or split_root.")
    if split_root is not None and partition not in {"train", "validation", "test"}:
        raise ValueError("split_root requires partition: train, validation or test.")
    if manifest is not None and partition is not None:
        raise ValueError("partition is valid only when loading from split_root.")
    selected_scenarios = tuple(
        available_scenarios() if scenarios is None else scenarios
    )
    if (
        not selected_scenarios
        or len(selected_scenarios) != len(set(selected_scenarios))
        or set(selected_scenarios) - set(available_scenarios())
    ):
        raise ValueError(
            "Select one or more unique scenarios from: "
            + ", ".join(available_scenarios())
            + "."
        )
    if controllers is not None:
        selected_controllers = (
            (controllers,) if isinstance(controllers, str) else tuple(controllers)
        )
        if (
            not selected_controllers
            or len(selected_controllers) != len(set(selected_controllers))
            or set(selected_controllers) - set(available_controllers())
            or "zero_action" not in selected_controllers
        ):
            raise ValueError(
                "Scenario comparisons require unique registered controllers, "
                "including zero_action."
            )

    if manifest is not None:
        source_data = load_simulation_data(manifest, start=start, end=end)
        input_source = {
            "kind": "manifest",
            "path": str(Path(manifest).resolve()),
            "sha256": _sha256(manifest),
        }
    else:
        source_data = load_simulation_split(
            split_root, partition, start=start, end=end
        )
        split_manifest = Path(split_root) / "split_manifest.json"
        input_source = {
            "kind": "split_partition",
            "path": str(Path(split_root).resolve()),
            "partition": partition,
            "sha256": _sha256(split_manifest),
        }
    config_source = (
        simulation_config
        if simulation_config is not None
        else DEFAULT_SIMULATION_CONFIG
    )
    base_config = (
        config_source
        if isinstance(config_source, SimulationConfig)
        else load_simulation_config(config_source)
    )
    runs = []
    destination.mkdir(parents=True, exist_ok=False)
    for name in selected_scenarios:
        transformed, config, scenario = apply_baseline_scenario(
            source_data, name, base_config
        )
        scenario["base_input_sha256"] = input_source["sha256"]
        scenario["source_partition"] = partition
        scenario["data_interval"] = source_data.metadata["interval"]
        run_path = destination / scenario["scenario_id"]
        run_baseline_experiments(
            assets=assets,
            rating_origin=rating_origin,
            output=run_path,
            controllers=controllers,
            simulation_config=config,
            costs=costs,
            baseline_config=baseline_config,
            data=transformed,
            scenario_metadata=scenario,
        )
        experiment = json.loads(
            (run_path / "experiment_summary.json").read_text(encoding="utf-8")
        )
        runs.append({
            "scenario_name": name,
            "scenario_id": scenario["scenario_id"],
            "label": scenario["label"],
            "experiment_directory": scenario["scenario_id"],
            "experiment_summary_sha256": _sha256(
                run_path / "experiment_summary.json"
            ),
            "controllers": experiment["controllers"],
            "steps": experiment["steps"],
        })

    index = {
        "status": "complete",
        "suite_id": "baseline_scenarios_v1",
        "input_source": input_source,
        "scenario_runs": runs,
    }
    index_path = destination / "scenario_index.json"
    with index_path.open("x", encoding="utf-8") as stream:
        json.dump(index, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return index
