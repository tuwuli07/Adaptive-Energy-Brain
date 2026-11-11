"""Audit Part 3 with offline tests, explicit edge fixtures and two episode replays.

Defaults use the EXISTING synthetic bundle; no data is acquired or generated.
The selected episode uses at most 168 action hours plus one history row. Fresh
output/report paths are required. Failures remain failures in the saved report;
readiness requires every check and the full pytest suite to pass.
"""

import argparse
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_simulation import run_simulation, sha256
from src.simulation.config import load_simulation_config
from src.simulation.data_loader import HourlyObservation, load_simulation_data
from src.simulation.demand import CampusDemand
from src.simulation.dispatch import DispatchEngine
from src.simulation.metrics import calculate_step_metrics
from src.simulation.renewables import SolarPV, WindGeneration


LIMITATIONS = [
    "Available default source data are synthetic, not measured campus observations. Hardware ratings and USD tariffs are hypothetical; real-campus validity is unverified.",
    "Hourly single-bus balance omits electrical network flows, voltage/frequency, sub-hourly transients, outages, ramp rates, storage self-discharge and detailed losses outside configured battery efficiencies.",
    "Observations use the previous completed hour, including lagged prices. One row supplies history and is excluded from episode metrics. Internal settlement uses current realized inputs, not future agent features.",
    "Zero requested battery power is a deterministic verification baseline. Safety dispatch may adjust it; it is not an optimized energy-management controller.",
    "Battery wear is an uncalibrated linear AC-throughput cost proxy, not validated electrochemical aging. Default unmet-demand accounting cost is zero; it is not an estimated value of lost load.",
    "No terminal SOC target/value, building-specific load shedding priorities, renewable storage-origin attribution or campus self-consumption calculation is implemented. Renewable utilization includes charging and export and does not recount battery discharge.",
    "The main replay is a bounded selected episode, not a full-year endurance benchmark. Edge cases use separately labeled deterministic test fixtures.",
    "The loader retains the current Part 2 nonnegative tariff schema even though standalone grid components can explicitly allow negative prices. Real manifests need explicit entity metadata/registries and asset ratings; no topology is inferred.",
    "Ledgers are copied for transactional step isolation. Long episodes may need profiling before large-scale training; no performance guarantee is claimed.",
    "Gymnasium, Part 4 policies, forecasting, RL, meta-learning and DRO/CVaR are outside this implementation and verification.",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def inventory(path):
    return {str(p.resolve()): sha256(p) for p in Path(path).rglob("*") if p.is_file()}


def edge_cases():
    """Synthetic TEST FIXTURES, independent of campus data or controller tuning."""
    base = load_simulation_config()
    provenance = {"source_id": "part3_verification_fixture", "origin": "simulated", "synthetic": True,
                  "source_metadata": {"note": "Deterministic edge fixture; not campus observations"}}
    cases = [
        # name, load, PV, wind, initial SOC, import cap, export cap, request, expected
        ("zero renewable generation", 100, 0, 0, .5, 2000, 500, 0, {"import": 100}),
        ("zero demand", 0, 0, 0, .5, 2000, 500, 0, {"import": 0, "export": 0}),
        ("fully charged battery", 0, 300, 0, .9, 2000, 500, -150, {"battery": 0, "export": 300}),
        ("battery at minimum SOC", 100, 0, 0, .1, 2000, 500, 150, {"battery": 0, "import": 100}),
        ("maximum grid import reached", 250, 0, 0, .5, 100, 500, 0, {"import": 100, "battery": 150}),
        ("maximum grid export reached", 0, 100, 0, .5, 2000, 100, 0, {"export": 100}),
        ("high demand with insufficient supply", 500, 0, 0, .1, 100, 500, 0, {"unserved": 400, "import": 100}),
        ("surplus exceeding storage and export", 100, 600, 300, .9, 2000, 100, 0, {"curtailment": 700, "export": 100}),
        ("battery charging efficiency", 0, 100, 0, .5, 2000, 500, -100, {"energy_after": 345, "loss": 5}),
        ("battery discharging efficiency", 100, 0, 0, .5, 2000, 500, 100, {"energy_after": 250 - 100 / .95, "loss": 100 / .95 - 100}),
    ]
    records = []
    for name, load, solar, wind, soc, imports, exports, request, expected in cases:
        config = replace(base, battery=replace(base.battery, initial_soc=soc),
                         grid=replace(base.grid, import_limit_kw=imports, export_limit_kw=exports),
                         unserved_load=replace(base.unserved_load, handling="record"))
        engine = DispatchEngine([
            SolarPV("PV01", 1000, provenance, capacity_origin="hypothetical_example"),
            WindGeneration("W01", 500, provenance, capacity_origin="hypothetical_example"),
        ], CampusDemand(provenance, building_ids=["B01"]), config)
        row = HourlyObservation("2026-01-01T00:00:00Z", solar, wind, load, .1, .04,
                                {"B01": load}, {"PV01": solar}, {"W01": wind})
        result = engine.step(row, request)
        metrics = calculate_step_metrics(result)
        actual = {"import": result.grid.import_power_kw, "export": result.grid.export_power_kw,
                  "battery": result.actual_battery_power_kw, "unserved": result.unserved_demand_kw,
                  "curtailment": result.renewables.total_curtailed_kw,
                  "energy_after": result.battery.state_after.battery_energy_kwh,
                  "loss": result.battery.efficiency_loss_kwh}
        require(all(abs(actual[key] - value) <= 1e-6 for key, value in expected.items()), f"Edge case failed: {name}")
        require(abs(result.balance_residual_kw) <= 1e-6 and abs(result.energy_balance_residual_kwh) <= 1e-6,
                f"Edge balance failed: {name}")
        require(not metrics.violations, f"Edge physical constraint failed: {name}")
        records.append({"name": name, "status": "passed", "expected": expected, "actual": actual,
                        "power_residual_kw": result.balance_residual_kw,
                        "energy_residual_kwh": result.energy_balance_residual_kwh})
    return {"origin": "explicit synthetic verification fixtures", "cases": records}


def replay(manifest, assets, rating_origin, config, costs, hours, output):
    data = load_simulation_data(manifest)
    require(len(data) >= 2, "Need a history row and at least one action row.")
    count = min(hours, len(data) - 1)
    start = data[0].timestamp_utc
    end = (datetime.fromisoformat(data[count].timestamp_utc.replace("Z", "+00:00"))
           + timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    kwargs = dict(manifest=manifest, assets=assets, rating_origin=rating_origin,
                  config=config, costs=costs, start=start, end=end)
    first = run_simulation(**kwargs, output=output / "replay_a")
    second = run_simulation(**kwargs, output=output / "replay_b")
    a = pd.read_csv(output / "replay_a/hourly_results.csv")
    b = pd.read_csv(output / "replay_b/hourly_results.csv")
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    require(first == second, "Repeat summaries, CSV/plot hashes or provenance differ.")
    require(len(a) == count, "Episode step count differs from selected data coverage.")
    require(a["timestamp_utc"].is_unique, "Duplicate output intervals.")
    stamps = pd.to_datetime(a["timestamp_utc"], utc=True)
    require(stamps.diff().dropna().eq(pd.Timedelta(hours=1)).all(), "Output intervals are not hourly.")
    power = float(a["power_balance_residual_kw"].abs().max())
    energy = float(a["energy_balance_residual_kwh"].abs().max())
    require(power <= 1e-6 and energy <= 1e-6, "Replay energy balance exceeds tolerance.")
    soc_min = float(a[["battery_soc_before", "battery_soc_after"]].min().min())
    soc_max = float(a[["battery_soc_before", "battery_soc_after"]].max().max())
    battery = first["simulation_config"]["battery"]
    require(soc_min >= battery["soc_min"] - 1e-9 and soc_max <= battery["soc_max"] + 1e-9, "Replay SOC out of range.")
    require(first["physical_violation_count"] == 0, "Physical violations in replay.")
    totals = first["cumulative_metrics"]
    expected_cost = totals["import_cost"] - totals["export_revenue"] + totals["battery_degradation_cost"] + totals["unserved_demand_cost"]
    require(abs(totals["net_operating_cost"] - expected_cost) <= 1e-6, "Net cost accounting differs.")
    return {"steps": count, "duration_hours": count, "resolution_hours": 1,
            "dataset_hours_available": len(data), "history_rows": 1,
            "history_start_utc": start, "dispatch_start_utc": str(a["timestamp_utc"].iloc[0]),
            "end_exclusive_utc": end, "label": first["label"],
            "max_power_residual_kw": power, "max_energy_residual_kwh": energy,
            "soc_min": soc_min, "soc_max": soc_max, "currency_code": first["currency_code"],
            "cumulative_metrics": totals, "physical_violations": 0,
            "adjustment_count": int(a["adjustment_count"].sum()),
            "reproducible": True, "source_provenance": first["source_provenance"],
            "input_files": first["input_files"], "simulation_config": first["simulation_config"],
            "economic_assumptions": first["economic_assumptions"],
            "rating_origin": rating_origin, "renewable_ratings": first["renewable_ratings"],
            "replay_outputs": [str(output / "replay_a"), str(output / "replay_b")]}


def full_tests(output):
    xml = output / "pytest.xml"
    command = [sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider", "--tb=short", f"--junitxml={xml}"]
    process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=600)
    log = process.stdout + process.stderr
    (output / "pytest.log").write_text(log, encoding="utf-8")
    require(process.returncode == 0 and xml.is_file(), f"Full test suite failed; inspect {output / 'pytest.log'}")
    cases = ET.parse(xml).getroot().findall(".//testcase")
    require(bool(cases) and all(c.find("failure") is None and c.find("error") is None and c.find("skipped") is None for c in cases),
            "Full suite must contain passing, unskipped tests.")
    modules = {}
    for case in cases:
        name = case.get("classname", "unknown").split(".")[-1]
        modules[name] = modules.get(name, 0) + 1
    required = ["test_simulation_config", "test_simulation_data_loader", "test_renewables", "test_demand",
                "test_battery", "test_grid", "test_dispatch", "test_simulation_state",
                "test_simulation_metrics", "test_simulation_environment", "test_simulation_runner"]
    require(all(modules.get(name, 0) for name in required), "A required simulation test module is missing.")
    return {"passed_count": len(cases), "exit_code": process.returncode,
            "command": command, "module_counts": modules, "log": str(output / "pytest.log")}


def validate(*, manifest, assets, rating_origin, config, costs, hours, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    before = {name: inventory(ROOT / "data" / name) for name in ("raw", "processed")}
    result = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "checks": [],
              "limitations": LIMITATIONS, "output": str(output.resolve())}

    def check(name, operation):
        print(f"Verifying: {name}", flush=True)
        try:
            details = operation()
            result["checks"].append({"name": name, "status": "passed", "details": details})
        except Exception as exc:
            result["checks"].append({"name": name, "status": "failed", "error": str(exc)})

    check("Required edge cases and battery efficiency", edge_cases)
    check("End-to-end episode and exact reproducibility", lambda: replay(
        manifest, assets, rating_origin, config, costs, hours, output))
    check("Complete automated test suite", lambda: full_tests(output))

    def unchanged():
        after = {name: inventory(ROOT / "data" / name) for name in before}
        require(after == before, "Raw or processed source files changed during verification.")
        return {name: len(files) for name, files in before.items()}

    check("Original raw and processed datasets unchanged", unchanged)
    result["passed"] = all(c["status"] == "passed" for c in result["checks"])
    result["assessment"] = (
        "READY for Part 4: Rule-Based Energy Management development using the verified simulation abstractions and explicitly labeled scenario assumptions. Real-campus validation remains outstanding."
        if result["passed"] else "NOT VERIFIED for Part 4: resolve failed checks before declaring readiness."
    )
    return result


def render_report(result):
    lines = ["# Part 3 completion report", "", f"Verification time (UTC): {result['timestamp_utc']}", "",
             "## Verification results", "", "| Check | Result |", "|---|---|"]
    for check in result["checks"]:
        lines.append(f"| {check['name']} | {check['status']} |")
        if check["status"] != "passed":
            lines.extend(["", f"Failure: {check['error']}", ""])
    details = {c["name"]: c.get("details") for c in result["checks"] if c["status"] == "passed"}
    tests = details.get("Complete automated test suite")
    if tests:
        lines += ["", "## Automated tests", "", f"**{tests['passed_count']} passed; no failed or skipped tests.**", "",
                  "| Module | Passed cases |", "|---|---:|"]
        lines.extend(f"| {name} | {count} |" for name, count in sorted(tests["module_counts"].items()))
        lines += ["", f"Full test log: `{tests['log']}`."]
    edges = details.get("Required edge cases and battery efficiency")
    if edges:
        lines += ["", "## Edge cases", "", "All values below are deterministic verification fixtures, not campus observations.", "",
                  "| Case | Status |", "|---|---|"]
        lines.extend(f"| {case['name']} | {case['status']} |" for case in edges["cases"])
    episode = details.get("End-to-end episode and exact reproducibility")
    if episode:
        totals = episode["cumulative_metrics"]
        lines += ["", "## End-to-end episode", "", f"**{episode['label']}**", "",
                  f"Two zero-request episodes each executed **{episode['steps']} steps / {episode['duration_hours']} hours**, at **{episode['resolution_hours']} hour** resolution.",
                  f"Dispatched coverage: `{episode['dispatch_start_utc']}` to `{episode['end_exclusive_utc']}` (end exclusive). History row: `{episode['history_start_utc']}`; excluded from costs and energy totals.",
                  f"The input bundle has {episode['dataset_hours_available']} hourly observations. This audit selects a bounded episode; it does not claim a full-year replay.",
                  "Repeated CSV values, cumulative metrics, source/configuration reports and plot hashes matched exactly in this environment.", "",
                  "| Measurement | Result |", "|---|---:|",
                  f"| Maximum absolute power balance residual | {episode['max_power_residual_kw']:.12g} kW |",
                  f"| Maximum absolute energy balance residual | {episode['max_energy_residual_kwh']:.12g} kWh |",
                  "| Absolute balance tolerance | 1e-6 kW / 1e-6 kWh |",
                  f"| Battery SOC range, including initial state | {episode['soc_min']:.8g} to {episode['soc_max']:.8g} |",
                  f"| Physical violations | {episode['physical_violations']} |",
                  f"| Constraint adjustments | {episode['adjustment_count']} |"]
        for name, value in totals.items():
            unit = "kWh" if name.endswith("kwh") else ("fraction (bus utilization)" if name == "renewable_utilization" else episode["currency_code"])
            display = "undefined (zero availability)" if value is None else f"{value:.12g} {unit}"
            lines.append(f"| {name} | {display} |")
        lines += ["", "## Configuration and provenance", "",
                  f"Hardware/rating origin: `{episode['simulation_config']['parameter_origin']}` / `{episode['rating_origin']}`. Economic origin: `{episode['economic_assumptions']['parameter_origin']}`.",
                  "Simulation outputs are model-derived; source measurement origins remain recorded separately. Synthetic weather-derived generation remains synthetic, not measured.", "",
                  "| Source | Origin | Input SHA-256 |", "|---|---|---|"]
        provenance = episode["source_provenance"]
        for source, item in provenance["sources"].items():
            digest = (item.get("input_file") or {}).get("sha256", "see source metadata")
            lines.append(f"| {source} | {provenance['source_origin_summary'][source]} | {digest} |")
        lines += ["", "Ignored weather sources: " + ", ".join(provenance.get("ignored_weather_sources", [])) + ".",
                  "Hardware/cost configuration snapshots, explicit asset ratings, manifest/config hashes, source paths and source metadata are saved in `validation.json` and each replay's `summary.json`.",
                  "Battery proxy cost = AC charge-plus-discharge throughput times the configured currency/kWh coefficient. Net cost = purchase cost - export revenue + battery proxy + configured unserved cost. Renewable utilization includes export/charging and excludes later battery discharge.",
                  "", "Replay outputs:"]
        lines.extend(f"- `{path}` (CSV, three PNG charts, summary/provenance)" for path in episode["replay_outputs"])
    lines += ["", "## Review and limitations", "",
              "Reviewed configuration, data loading, renewable/demand components, battery/grid accounting, dispatch, actions/observations, metrics, episode lifecycle and CLI/plots. No Part 4 policy or AI implementation was added.",
              "Corrected the historical Part 3.1 documents' status text to point to current implementation evidence, while preserving their original design proposals. No physical-model behavior was changed during this audit.",
              "The automated suite covers validation failures, units, building aggregation, charging/discharging losses and limits, tariffs, curtailed/unserved power, causal observations, atomic failure handling, reset/termination and reproducibility.", ""]
    lines.extend(f"- {limitation}" for limitation in result["limitations"])
    lines += ["", "## Repeat verification", "", "Run from the project root; choose fresh output/report names:", "", "```powershell",
              ".\\.venv\\Scripts\\python.exe scripts/validate_simulation.py `",
              "  --output results/part3_recheck `",
              "  --report docs/part3_recheck.md", "```", "",
              "## Part 4 readiness", "", f"**{result['assessment']}**", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/processed/synthetic_hypothetical_campus_v1/integration_manifest.yaml")
    parser.add_argument("--assets", type=Path, default=ROOT / "configs/synthetic_scenario.yaml")
    parser.add_argument("--rating-origin", choices=("hypothetical_example", "documented_specification", "mixed"), default="hypothetical_example")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/simulation.yaml")
    parser.add_argument("--costs", type=Path, default=ROOT / "configs/simulation_costs.yaml")
    parser.add_argument("--hours", type=int, default=168)
    parser.add_argument("--output", type=Path, default=ROOT / "results/part3_12_validation")
    parser.add_argument("--report", type=Path, default=ROOT / "docs/part3_completion_report.md")
    args = parser.parse_args(argv)
    if args.hours < 1:
        parser.error("--hours must be positive.")
    if args.output.exists() or args.report.exists():
        parser.error("Refusing to overwrite existing output/report; choose fresh paths.")
    report = args.report
    kwargs = vars(args).copy()
    kwargs.pop("report")
    result = validate(**kwargs)
    with (args.output / "validation.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("x", encoding="utf-8") as stream:
        stream.write(render_report(result))
    print(json.dumps({"passed": result["passed"], "assessment": result["assessment"],
                      "checks": [{k: v for k, v in c.items() if k != "details"} for c in result["checks"]]}, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
