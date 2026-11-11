"""Reproducible scenario profiles, never represented as measured campus data.

python -m src.data.synthetic --dry-run
python -m src.data.synthetic --output hypothetical_campus_v1

Availability is assessed before generation. A verified source or integration
provenance protects its entire category; partial series are not silently filled.
Daily/seasonal curves, correlated weather, a cubic wind proxy and temperature-
adjusted PV proxy are stylized assumptions, not calibrated physical models.
Seed reproducibility is for a fixed configuration and NumPy/generator version.
"""

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd
import yaml

from src.data.acquisition import load_sources
from src.data.schema import validate_dataset


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/synthetic_scenario.yaml"
CATEGORY_VARIABLES = {
    "demand": ["building_load_kw", "demand_energy_kwh"],
    "solar": ["available_generation_kw", "available_generation_energy_kwh"],
    "wind": ["available_generation_kw", "available_generation_energy_kwh"],
    "weather": ["irradiance_w_m2", "air_temperature_c", "wind_speed_m_s"],
    "grid_prices": ["import_price", "export_price"],
}


class SyntheticDataError(ValueError):
    """Unsafe assumptions, unverified availability or output collisions."""


def load_scenario(path=DEFAULT_CONFIG):
    try:
        config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        validate_scenario(config)
        return config
    except (OSError, yaml.YAMLError) as exc:
        raise SyntheticDataError(f"Cannot load synthetic scenario: {exc}") from exc


def validate_scenario(config):
    """Validate physical ranges and baseline fallback before creating profiles."""
    try:
        def finite_tree(value):
            if isinstance(value, dict):
                for child in value.values():
                    finite_tree(child)
            elif isinstance(value, list):
                for child in value:
                    finite_tree(child)
            elif isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isfinite(value):
                raise SyntheticDataError("All numeric assumptions must be finite.")

        finite_tree(config)
        if config["version"] != 1:
            raise SyntheticDataError("Unsupported scenario version.")
        if not isinstance(config["seed"], int) or isinstance(config["seed"], bool) or config["seed"] < 0:
            raise SyntheticDataError("seed must be a nonnegative integer.")
        if not isinstance(config["hours"], int) or isinstance(config["hours"], bool) or not 1 <= config["hours"] <= 87840:
            raise SyntheticDataError("hours must be an integer in [1, 87840].")
        start = pd.Timestamp(config["start_utc"])
        if pd.isna(start) or start.tzinfo is None or start.utcoffset().total_seconds() != 0 or start != start.floor("h"):
            raise SyntheticDataError("start_utc must be an explicit UTC hour boundary.")
        start.tz_convert(config["campus_timezone"])
        if not re.fullmatch(r"[A-Z]{3}", config["currency_code"]):
            raise SyntheticDataError("currency_code must contain three uppercase letters.")
        if not config["scenario_id"] or not isinstance(config["allow_latent_weather"], bool):
            raise SyntheticDataError("Scenario identity and boolean latent-weather policy are required.")
        categories = config["generate_categories"]
        if not isinstance(categories, list) or len(categories) != len(set(categories)) or not set(categories) <= CATEGORY_VARIABLES.keys():
            raise SyntheticDataError("generate_categories must contain unique supported category names.")

        def number(value, low=None, high=None):
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise SyntheticDataError("Assumptions must be finite numeric values.")
            if (low is not None and value < low) or (high is not None and value > high):
                raise SyntheticDataError(f"Assumption {value} is outside [{low}, {high}].")

        def positive(value):
            number(value, 0)
            if value == 0:
                raise SyntheticDataError("Capacity/scale assumptions must be positive.")

        buildings = config["buildings"]
        if not isinstance(buildings, list) or not buildings:
            raise SyntheticDataError("At least one hypothetical building is required.")
        identifiers = []
        for building in buildings:
            identifiers.append(building["id"])
            positive(building["base_kw"])
            number(building["min_kw"], 0, building["base_kw"])
            number(building["max_kw"], building["base_kw"])
        identifiers += [config[k]["asset_id"] for k in ("solar", "wind", "battery")]
        identifiers += [config["weather"]["site_id"], config["prices"]["grid_id"]]
        if not all(isinstance(item, str) and item.strip() for item in identifiers) or len(identifiers) != len(set(identifiers)):
            raise SyntheticDataError("Building/asset/site/grid IDs must be nonempty and unique.")
        demand, weather, solar, wind, prices, battery, grid = [config[key] for key in
            ("demand", "weather", "solar", "wind", "prices", "battery", "grid")]
        for key in ("night_fraction", "weekend_factor", "seasonal_amplitude", "surge_probability"):
            number(demand[key], 0, 1)
        for key in ("temperature_sensitivity_per_c", "uncertainty_std"):
            number(demand[key], 0, 1)
        number(demand["temperature_reference_c"], -273.15)
        number(demand["surge_multiplier"], 1)
        number(demand["seasonal_peak_day"], 1, 366)
        for key in ("temperature_mean_c", "temperature_min_c", "temperature_max_c"):
            number(weather[key], -273.15)
        if not weather["temperature_min_c"] <= weather["temperature_mean_c"] <= weather["temperature_max_c"]:
            raise SyntheticDataError("Temperature bounds must contain the mean.")
        for key in ("temperature_seasonal_amplitude_c", "temperature_daily_amplitude_c", "temperature_noise_std_c", "wind_mean_m_s", "wind_seasonal_amplitude_m_s", "wind_noise_std_m_s", "cloud_std"):
            number(weather[key], 0)
        number(weather["cloud_mean"], 0, 1)
        number(weather["autocorrelation"], 0, 0.999)
        number(weather["daylight_mean_hours"], 1, 23)
        number(weather["daylight_seasonal_amplitude_hours"], 0)
        if not 0 < weather["daylight_mean_hours"] - weather["daylight_seasonal_amplitude_hours"] <= weather["daylight_mean_hours"] + weather["daylight_seasonal_amplitude_hours"] < 24:
            raise SyntheticDataError("Daylight duration must remain between 0 and 24 hours.")
        for key in ("seasonal_peak_day", "daylight_peak_day"):
            number(weather[key], 1, 366)
        for value in (weather["wind_max_m_s"], weather["irradiance_peak_w_m2"], solar["capacity_kw"], solar["reference_irradiance_w_m2"], wind["capacity_kw"]):
            positive(value)
        number(solar["temperature_derating_per_c"], 0, 1)
        if not 0 <= wind["cut_in_m_s"] < wind["rated_m_s"] < wind["cut_out_m_s"] <= weather["wind_max_m_s"]:
            raise SyntheticDataError("Wind thresholds require cut-in < rated < cut-out <= weather maximum.")
        number(prices["spike_probability"], 0, 1)
        number(prices["spike_multiplier"], 1)
        if not 0 <= prices["export_per_kwh"] <= prices["base_import_per_kwh"] <= prices["peak_import_per_kwh"] <= prices["max_import_per_kwh"]:
            raise SyntheticDataError("Baseline tariff bounds must be ordered and nonnegative.")
        if not 0 <= prices["peak_start_hour"] < prices["peak_end_hour"] <= 24:
            raise SyntheticDataError("Tariff peak hours must lie in [0, 24].")
        for key in ("capacity_kwh", "charge_limit_kw", "discharge_limit_kw"):
            positive(battery[key])
        for key in ("charge_efficiency", "discharge_efficiency"):
            positive(battery[key])
            number(battery[key], 0, 1)
        if not 0 <= battery["soc_min"] <= battery["initial_soc"] <= battery["soc_max"] <= 1 or battery["soc_min"] == battery["soc_max"]:
            raise SyntheticDataError("Battery initial SOC must lie inside ordered SOC bounds.")
        number(grid["import_limit_kw"], sum(item["max_kw"] for item in buildings))
        number(grid["export_limit_kw"], 0)
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, SyntheticDataError):
            raise
        raise SyntheticDataError(f"Invalid or missing scenario assumption: {exc}") from exc


def assess_missing_variables(*, sources_path=ROOT / "configs/data_sources.yaml",
                             processed_root=ROOT / "data/processed"):
    """Conservatively assess categories; approval is not proof of complete data.

    Existing integration provenance protects categories even for partial or
    modeled data. No per-cell gap filling or overwrite is authorized by this
    report. Unknown generation provenance protects both renewable categories.
    """
    sources = load_sources(sources_path)
    state = {category: {"variables": columns, "status": "unavailable", "evidence": []}
             for category, columns in CATEGORY_VARIABLES.items()}
    by_id = {source["id"]: source["category"] for source in sources}
    for source in sources:
        if source["status"] == "verified":
            entry = state[source["category"]]
            entry["status"] = "protected_requires_coverage_review"
            entry["evidence"].append({"source_id": source["id"], "reason": "verified configuration; import/coverage not assumed"})
    paths = sorted(Path(processed_root).glob("*/integration_report.json"))
    for path in paths:
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(report.get("sources"), dict) or not isinstance(report.get("provenance"), list):
                raise ValueError("missing sources/provenance")
            for record in report["provenance"]:
                category = by_id.get(record["source_id"])
                categories = [category] if category else {
                    "demand": ["demand"], "generation": ["solar", "wind"],
                    "weather": ["weather"], "price": ["grid_prices"],
                }.get(record["record_type"], [])
                for category in categories:
                    state[category]["status"] = "protected_requires_coverage_review"
                    evidence = {"report": str(path.resolve()), "reason": "existing integration provenance; not proof of full coverage"}
                    if evidence not in state[category]["evidence"]:
                        state[category]["evidence"].append(evidence)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise SyntheticDataError(f"Cannot assess existing integration report {path}: {exc}") from exc
    return {"categories": state, "integration_reports_inspected": [str(path.resolve()) for path in paths],
            "note": "Unavailable means no verified source or integration evidence found, not a claim about external data availability."}


def _rng(seed, name):
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    key = [int.from_bytes(digest[index:index + 4], "little") for index in range(0, 16, 4)]
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed, spawn_key=key)))


def _correlated(rng, size, rho):
    values = rng.normal(size=size)
    for index in range(1, size):
        values[index] = rho * values[index - 1] + np.sqrt(1 - rho ** 2) * values[index]
    return values


def generate_synthetic(config=None, *, sources_path=ROOT / "configs/data_sources.yaml",
                       processed_root=ROOT / "data/processed"):
    """Assess missing inputs, then return schema-valid synthetic source payloads.

    Does not write files or merge generated values into measured records. The
    synthetic origin also applies to all deterministic model outputs generated
    from synthetic weather. Existing categories are conservatively skipped.
    """
    config = deepcopy(config) if config is not None else load_scenario()
    validate_scenario(config)
    assessment = assess_missing_variables(sources_path=sources_path, processed_root=processed_root)
    selected = [category for category in config["generate_categories"]
                if assessment["categories"][category]["status"] == "unavailable"]
    if not selected:
        return {"sources": [], "report": {"assessment": assessment, "generated_categories": []}}
    if "weather" not in selected and set(selected) & {"demand", "solar", "wind"} and not config["allow_latent_weather"]:
        raise SyntheticDataError("Dependent profiles need synthetic weather; existing/excluded weather requires explicit allow_latent_weather for an independent scenario.")
    hours = config["hours"]
    stamps = pd.date_range(config["start_utc"], periods=hours, freq="h").tz_convert("UTC")
    local = stamps.tz_convert(config["campus_timezone"])
    hour = local.hour.to_numpy() + 0.5
    day = local.day_of_year.to_numpy()
    seasonal = lambda peak: np.cos(2 * np.pi * (day - peak) / 365.25)
    weather = config["weather"]
    rng = _rng(config["seed"], "weather")
    cloud = np.clip(weather["cloud_mean"] + weather["cloud_std"] * _correlated(rng, hours, weather["autocorrelation"]), 0, 1)
    daylight = weather["daylight_mean_hours"] + weather["daylight_seasonal_amplitude_hours"] * seasonal(weather["daylight_peak_day"])
    # Four subhour samples approximate interval-average daylight, not an instant.
    irradiance = np.zeros(hours)
    for offset in (0.125, 0.375, 0.625, 0.875):
        phase = (local.hour.to_numpy() + offset - (12 - daylight / 2)) / daylight
        irradiance += np.where((phase > 0) & (phase < 1), np.sin(np.pi * phase), 0) / 4
    irradiance *= cloud * weather["irradiance_peak_w_m2"]
    temperature = np.clip(weather["temperature_mean_c"] + weather["temperature_seasonal_amplitude_c"] * seasonal(weather["seasonal_peak_day"])
                          + weather["temperature_daily_amplitude_c"] * np.cos(2 * np.pi * (hour - 14) / 24)
                          + weather["temperature_noise_std_c"] * _correlated(rng, hours, weather["autocorrelation"]),
                          weather["temperature_min_c"], weather["temperature_max_c"])
    speed = np.clip(weather["wind_mean_m_s"] + weather["wind_seasonal_amplitude_m_s"] * seasonal(weather["seasonal_peak_day"])
                    + weather["wind_noise_std_m_s"] * _correlated(rng, hours, weather["autocorrelation"]), 0, weather["wind_max_m_s"])
    base = {"timestamp_utc": stamps.strftime("%Y-%m-%dT%H:%M:%SZ"), "interval_hours": np.ones(hours)}
    sources, events = [], []
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()

    def add(category, kind, frame, units, model):
        source_id = f"synthetic_{config['scenario_id']}_{category}"
        metadata = {
            "source_id": source_id, "origin": "simulated", "synthetic": True,
            "dataset_label": "SYNTHETIC_SCENARIO_NOT_MEASURED", "currency_code": config["currency_code"],
            "source_metadata": {"scenario": config, "config_sha256": config_hash,
                                "seed": config["seed"], "rng": "PCG64", "numpy_version": np.__version__,
                                "generator_version": 1, "model": model,
                                "variable_origins": {column: {"origin": "synthetic", "method": model} for column in frame.columns},
                                "latent_weather": "synthetic; never substituted or described as measured weather"},
        }
        validate_dataset(frame.to_dict("records"), kind, units=units, metadata=metadata)
        sources.append({"source_id": source_id, "record_type": kind, "category": category,
                        "measurement_kind": "interval_mean", "data": frame,
                        "units": units, "metadata": metadata})

    if "weather" in selected:
        add("weather", "weather", pd.DataFrame({**base, "site_id": weather["site_id"],
            "irradiance_w_m2": irradiance, "air_temperature_c": temperature, "wind_speed_m_s": speed}),
            {"interval_hours": "h", "irradiance_w_m2": "W/m^2", "air_temperature_c": "degC", "wind_speed_m_s": "m/s"},
            "stylized seasonal/daily curves and bounded correlated noise; irradiance plane is illustrative")
    for category in ("solar", "wind"):
        if category not in selected:
            continue
        asset = config[category]
        if category == "solar":
            power = asset["capacity_kw"] * irradiance / asset["reference_irradiance_w_m2"] * np.maximum(0, 1 - asset["temperature_derating_per_c"] * (temperature - 25))
        else:
            fraction = np.clip((speed ** 3 - asset["cut_in_m_s"] ** 3) / (asset["rated_m_s"] ** 3 - asset["cut_in_m_s"] ** 3), 0, 1)
            power = asset["capacity_kw"] * np.where(speed >= asset["cut_out_m_s"], 0, fraction)
        power = np.clip(power, 0, asset["capacity_kw"])
        add(category, "generation", pd.DataFrame({**base, "asset_id": asset["asset_id"],
            "generation_type": category, "available_generation_kw": power,
            "available_generation_energy_kwh": power * 1.0, "generation_capacity_kw": asset["capacity_kw"]}),
            {"interval_hours": "h", "available_generation_kw": "kW", "available_generation_energy_kwh": "kWh", "generation_capacity_kw": "kW"},
            "simplified PV temperature proxy" if category == "solar" else "simplified cubic wind power curve; synthetic hub-height-equivalent wind")
    if "demand" in selected:
        demand = config["demand"]
        frames = []
        for building in config["buildings"]:
            rng = _rng(config["seed"], building["id"])
            occupied = np.exp(-0.5 * ((hour - 13) / 3.5) ** 2)
            daily = demand["night_fraction"] + (1 - demand["night_fraction"]) * occupied
            weekly = np.where(local.day_of_week.to_numpy() >= 5, demand["weekend_factor"], 1)
            uncertainty = np.exp(demand["uncertainty_std"] * _correlated(rng, hours, 0.5) - 0.5 * demand["uncertainty_std"] ** 2)
            power = building["base_kw"] * daily * weekly * (1 + demand["seasonal_amplitude"] * seasonal(demand["seasonal_peak_day"]))
            power *= (1 + demand["temperature_sensitivity_per_c"] * np.maximum(temperature - demand["temperature_reference_c"], 0)) * uncertainty
            surge = rng.random(hours) < demand["surge_probability"]
            power *= np.where(surge, demand["surge_multiplier"], 1)
            clipped = (power < building["min_kw"]) | (power > building["max_kw"])
            power = np.clip(power, building["min_kw"], building["max_kw"])
            events.append({"kind": "demand_assumptions", "building_id": building["id"],
                           "surge_hours": stamps[surge].strftime("%Y-%m-%dT%H:%M:%SZ").tolist(),
                           "clipped_hours": int(clipped.sum())})
            frames.append(pd.DataFrame({**base, "building_id": building["id"],
                                        "building_load_kw": power, "demand_energy_kwh": power * 1.0}))
        add("demand", "demand", pd.concat(frames, ignore_index=True),
            {"interval_hours": "h", "building_load_kw": "kW", "demand_energy_kwh": "kWh"},
            "assumed building profiles, seasonal/weather effects and bounded stochastic demand")
    if "grid_prices" in selected:
        prices = config["prices"]
        peak = (local.hour >= prices["peak_start_hour"]) & (local.hour < prices["peak_end_hour"])
        buy = np.where(peak, prices["peak_import_per_kwh"], prices["base_import_per_kwh"])
        spike = _rng(config["seed"], "prices").random(hours) < prices["spike_probability"]
        buy = np.minimum(buy * np.where(spike, prices["spike_multiplier"], 1), prices["max_import_per_kwh"])
        events.append({"kind": "price_spikes", "hours": stamps[spike].strftime("%Y-%m-%dT%H:%M:%SZ").tolist()})
        add("grid_prices", "price", pd.DataFrame({**base, "grid_id": prices["grid_id"],
            "import_price": buy, "export_price": prices["export_per_kwh"]}),
            {"interval_hours": "h", "import_price": f"{config['currency_code']}/kWh", "export_price": f"{config['currency_code']}/kWh"},
            "artificial time-of-use tariff with optional bounded spikes; not an observed utility tariff")
    return {"sources": sources, "report": {
        "label": "SYNTHETIC_SCENARIO_NOT_MEASURED", "assessment": assessment,
        "generated_categories": selected, "scenario": config, "config_sha256": config_hash,
        "events": events, "note": "No battery dispatch, energy balance controller or real-world validation is implemented.",
    }}


def save_synthetic(result, name, *, processed_root=ROOT / "data/processed"):
    """Write only a new explicitly synthetic bundle; never update existing data."""
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", name):
        raise SyntheticDataError("Output name must be a safe basename.")
    if not result["sources"]:
        raise SyntheticDataError("No missing categories were generated; nothing to save.")
    root = Path(processed_root)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"synthetic_{name}"
    try:
        destination.mkdir()
    except OSError as exc:
        raise SyntheticDataError(f"Output already exists or is inaccessible: {destination}") from exc
    created, manifest = [], {"datasets": []}
    try:
        report = deepcopy(result["report"])
        report["file_checksums"] = {}
        for source in result["sources"]:
            if source["category"] not in CATEGORY_VARIABLES:
                raise SyntheticDataError("Unknown synthetic category; refusing unsafe output path.")
            if source["metadata"].get("synthetic") is not True:
                raise SyntheticDataError("Refusing to label non-synthetic data as synthetic.")
            validate_dataset(source["data"].to_dict("records"), source["record_type"], units=source["units"], metadata=source["metadata"])
            path = destination / f"synthetic_{source['category']}.csv"
            created.append(path)
            source["data"].to_csv(path, index=False, mode="x")
            report["file_checksums"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest["datasets"].append({key: source[key] for key in ("source_id", "record_type", "measurement_kind", "units", "metadata")})
            manifest["datasets"][-1]["path"] = path.name
        for filename, text in (("generation_report.json", json.dumps(report, indent=2, allow_nan=False)),
                               ("integration_manifest.yaml", yaml.safe_dump(manifest, sort_keys=False))):
            path = destination / filename
            created.append(path)
            with path.open("x", encoding="utf-8") as stream:
                stream.write(text + "\n")
        return destination
    except Exception as exc:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        destination.rmdir()
        raise SyntheticDataError(f"Synthetic bundle not saved: {exc}") from exc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", help="New bundle name; stored with synthetic_ prefix")
    args = parser.parse_args(argv)
    try:
        config = load_scenario(args.config)
        if args.dry_run:
            print(json.dumps(assess_missing_variables(), indent=2))
        elif args.output:
            result = generate_synthetic(config)
            print(save_synthetic(result, args.output))
        else:
            parser.error("Choose --dry-run or --output NAME.")
        return 0
    except (ValueError, OSError) as exc:
        print(f"Synthetic generation error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
