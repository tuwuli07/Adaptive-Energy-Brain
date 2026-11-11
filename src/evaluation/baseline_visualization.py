"""Comparison plots from checksum-validated baseline experiment artifacts."""

from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import pandas as pd

from src.evaluation.baseline_comparison import (
    _comparison_metrics,
    _load_experiment,
)


_PLOT_NAMES = (
    "net_operating_cost_comparison.png",
    "renewable_utilization_comparison.png",
    "battery_soc_over_time.png",
    "grid_import_export_over_time.png",
    "battery_throughput_comparison.png",
    "renewable_curtailment.png",
    "unserved_demand.png",
)


def _title(experiment):
    scenario = experiment.get("scenario")
    if scenario:
        descriptor = scenario.get("label", scenario.get("scenario_id", "Scenario"))
        if scenario.get("modified_or_synthetic"):
            descriptor += " (modified stress scenario)"
    else:
        descriptor = "Baseline experiment"
    if experiment.get("synthetic_inputs_present"):
        descriptor += " - synthetic inputs"
    return descriptor


def _figure():
    figure = Figure(figsize=(11, 6), layout="constrained")
    FigureCanvasAgg(figure)
    return figure


def _write(figure, path):
    with path.open("xb") as stream:
        figure.savefig(stream, format="png", dpi=130)
    figure.clear()


def plot_baseline_experiment(experiment_dir, output_dir=None):
    """Create seven labelled PNGs without overwriting existing files.

    The source experiment is checksum-, metric-, and episode-validated before
    plotting. Values are read only from its saved simulator results.
    """
    root, experiment, loaded = _load_experiment(experiment_dir)
    destination = (
        Path(output_dir)
        if output_dir is not None
        else root.with_name(f"{root.name}_plots")
    )
    existing = [destination / name for name in _PLOT_NAMES if (destination / name).exists()]
    if existing:
        raise FileExistsError(
            f"Plot outputs already exist: {existing[0]}; choose a fresh directory."
        )
    destination.mkdir(parents=True, exist_ok=True)

    baseline = loaded["zero_action"]
    comparison = [
        _comparison_metrics(name, result, baseline)
        for name, result in loaded.items()
    ]
    labels = [row["controller"] for row in comparison]
    provenance = _title(experiment)
    currency = experiment["currency_code"]
    figures = []
    paths = [destination / name for name in _PLOT_NAMES]

    figure = _figure()
    axis = figure.subplots()
    costs = [row["net_operating_cost"] for row in comparison]
    axis.bar(labels, costs)
    axis.set_ylabel(f"Net operating cost ({currency})")
    axis.set_title(f"{provenance}\nNet operating cost by controller")
    axis.grid(axis="y", alpha=0.25)
    axis.tick_params(axis="x", rotation=20)
    figures.append(figure)

    figure = _figure()
    axis = figure.subplots()
    utilization = [row["renewable_utilization"] for row in comparison]
    valid = [(label, value) for label, value in zip(labels, utilization) if value is not None]
    if valid:
        axis.bar([item[0] for item in valid], [item[1] for item in valid])
    else:
        axis.text(0.5, 0.5, "Not defined: no renewable generation available",
                  ha="center", va="center", transform=axis.transAxes)
    axis.set_ylabel("Renewable utilization (fraction)")
    axis.set_ylim(0, 1)
    axis.set_title(f"{provenance}\nRenewable utilization by controller")
    axis.grid(axis="y", alpha=0.25)
    axis.tick_params(axis="x", rotation=20)
    figures.append(figure)

    figure = _figure()
    axis = figure.subplots()
    for name, result in loaded.items():
        frame = result["frame"]
        times = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="raise")
        interval = pd.to_timedelta(float(frame["interval_hours"].iloc[0]), unit="h")
        soc_times = list(times) + [times.iloc[-1] + interval]
        soc_values = list(frame["battery_soc_before"]) + [
            float(frame["battery_soc_after"].iloc[-1])
        ]
        axis.step(soc_times, soc_values, where="post", label=name)
    axis.set_ylabel("Battery state of charge (fraction)")
    axis.set_ylim(0, 1)
    axis.set_xlabel("Interval start (UTC)")
    axis.set_title(f"{provenance}\nBattery SOC over time")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.autofmt_xdate()
    figures.append(figure)

    figure = _figure()
    imports_axis, exports_axis = figure.subplots(2, 1, sharex=True)
    for name, result in loaded.items():
        frame = result["frame"]
        times = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="raise")
        imports_axis.plot(times, frame["grid_import_kw"], label=name)
        exports_axis.plot(times, frame["grid_export_kw"], label=name)
    imports_axis.set_ylabel("Grid import (kW)")
    exports_axis.set_ylabel("Grid export (kW)")
    exports_axis.set_xlabel("Interval start (UTC)")
    imports_axis.set_title(f"{provenance}\nGrid import and export")
    imports_axis.grid(alpha=0.25)
    exports_axis.grid(alpha=0.25)
    imports_axis.legend()
    exports_axis.legend()
    figure.autofmt_xdate()
    figures.append(figure)

    figure = _figure()
    axis = figure.subplots()
    axis.bar(
        labels,
        [row["battery_energy_throughput_kwh"] for row in comparison],
    )
    axis.set_ylabel("Battery energy throughput (kWh)")
    axis.set_title(f"{provenance}\nBattery throughput by controller")
    axis.grid(axis="y", alpha=0.25)
    axis.tick_params(axis="x", rotation=20)
    figures.append(figure)

    figure = _figure()
    axis = figure.subplots()
    for name, result in loaded.items():
        frame = result["frame"]
        times = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="raise")
        axis.plot(times, frame["curtailment_kw"], label=name)
    axis.set_ylabel("Renewable curtailment (kW)")
    axis.set_xlabel("Interval start (UTC)")
    axis.set_title(f"{provenance}\nRenewable curtailment over time")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.autofmt_xdate()
    figures.append(figure)

    figure = _figure()
    axis = figure.subplots()
    for name, result in loaded.items():
        frame = result["frame"]
        times = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="raise")
        axis.plot(times, frame["unserved_demand_kw"], label=name)
    axis.set_ylabel("Unserved demand (kW)")
    axis.set_xlabel("Interval start (UTC)")
    axis.set_title(f"{provenance}\nUnserved demand over time")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.autofmt_xdate()
    figures.append(figure)

    for figure, path in zip(figures, paths):
        _write(figure, path)
    return paths
