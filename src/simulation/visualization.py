"""Headless Matplotlib figures from executed hourly simulation results."""

from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import pandas as pd


def plot_simulation(frame, output_dir, *, currency_code, label):
    """Write three PNGs without interactive windows or global backend changes.

    Power is kW; SOC is a fraction. Costs include configured economic proxies.
    label must describe measured/synthetic provenance, not imply field validation.
    Existing plot paths are never overwritten.
    """
    if frame.empty:
        raise ValueError("Cannot plot an empty episode.")
    destination = Path(output_dir)
    paths = [destination / name for name in ("energy_flows.png", "battery.png", "costs.png")]
    if any(path.exists() for path in paths):
        raise FileExistsError("Plot outputs already exist; choose a fresh output directory.")
    times = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="raise")
    panels = [
        [("solar_available_kw", "Solar available"), ("wind_available_kw", "Wind available"),
         ("campus_demand_kw", "Demand"), ("grid_import_kw", "Grid import"),
         ("grid_export_kw", "Grid export"), ("curtailment_kw", "Curtailment"),
         ("unserved_demand_kw", "Unserved demand")],
        [("battery_charge_kw", "Charge"), ("battery_discharge_kw", "Discharge")],
        [("net_operating_cost", "Per-step net cost"),
         ("cumulative_net_operating_cost", "Cumulative net cost")],
    ]
    destination.mkdir(parents=True, exist_ok=True)
    for index, (path, columns) in enumerate(zip(paths, panels)):
        figure = Figure(figsize=(11, 6), layout="constrained")
        FigureCanvasAgg(figure)
        axis = figure.subplots()
        for column, title in columns:
            axis.plot(times, frame[column], label=title)
        axis.set_ylabel(currency_code if index == 2 else "Power (kW)")
        axis.set_xlabel("Interval start (UTC)")
        axis.set_title(f"{label}\n{path.stem.replace('_', ' ').title()}")
        axis.grid(alpha=0.25)
        axis.legend(loc="upper left")
        if index == 1:
            soc_axis = axis.twinx()
            soc_axis.plot(times, frame["battery_soc_after"], color="black", label="SOC after step")
            soc_axis.set_ylabel("SOC (fraction)")
            soc_axis.set_ylim(0, 1)
            soc_axis.legend(loc="upper right")
        figure.autofmt_xdate()
        with path.open("xb") as stream:
            figure.savefig(stream, format="png", dpi=130)
        figure.clear()
    return paths
