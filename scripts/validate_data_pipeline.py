r"""Audit Part 2 artifacts and replay an explicitly synthetic offline pipeline.

Run from the project root:
  .\.venv\Scripts\python.exe scripts/validate_data_pipeline.py \
      --report docs/part2_completion_report.md \
      --json-output data/processed/part2_validation.json

Existing dataset bundles are read-only. Replay files live in a temporary
directory and are removed afterward. Reports use exclusive creation; choose
new report paths for subsequent runs. --skip-tests records tests as not run.
No real downloads or models are executed. Numerical replay comparisons use
explicit tolerances where CSV serialization is involved; timestamps, masks and
graph indices must match exactly. Full metadata/NPZ byte identity is not claimed
because acquisition clocks, paths and archive timestamps may differ.
"""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.acquisition import AcquisitionError, acquire_dataset, load_sources
from src.data.cleaning import load_and_clean_csv
from src.data.graph_builder import build_graph, load_forecasting_config
from src.data.integration import integrate_datasets, save_integrated_dataset
from src.data.schema import validate_dataset
from src.data.sequences import prepare_sequences
from src.data.splitting import PARTITIONS, save_splits, split_datasets, verify_split_bundle
from src.data.synthetic import generate_synthetic, load_scenario, save_synthetic


SYNTHETIC = ROOT / "data/processed/synthetic_hypothetical_campus_v1"
SPLITS = ROOT / "data/processed/splits_hypothetical_campus_v1"
FORECASTING = ROOT / "data/processed/forecasting_hypothetical_campus_v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inventory(root):
    return {str(path.resolve()): sha256(path) for path in Path(root).rglob("*") if path.is_file()}


def read_bundle():
    manifest = yaml.safe_load((SYNTHETIC / "integration_manifest.yaml").read_text())
    report = json.loads((SYNTHETIC / "generation_report.json").read_text())
    sources = []
    for item in manifest["datasets"]:
        path = SYNTHETIC / item["path"]
        require(sha256(path) == report["file_checksums"][path.name], f"Changed synthetic file: {path.name}")
        source = deepcopy(item)
        source.pop("path")
        source["data"] = pd.read_csv(path, dtype={"building_id": str, "asset_id": str, "site_id": str, "grid_id": str})
        validate_dataset(source["data"].to_dict("records"), source["record_type"], units=source["units"], metadata=source["metadata"])
        require(source["metadata"].get("synthetic") is True, "Synthetic provenance label missing")
        sources.append(source)
    return sources, report


def smoke_replay(root, config):
    """Generate explicit fixtures, acquire them locally, then replay every stage."""
    generated = generate_synthetic(config, processed_root=root / "empty_evidence")
    origin = save_synthetic(generated, "smoke", processed_root=root / "origin")
    cleaned_sources = []
    for source in generated["sources"]:
        path = origin / f"synthetic_{source['category']}.csv"
        spec = {
            "id": source["source_id"], "category": source["category"],
            "description": "Explicit synthetic smoke fixture; not a selected real dataset",
            "type": "local_csv", "status": "verified", "local_path": str(path),
            "download_url": None, "filename": path.name,
            "expected_columns": list(source["data"].columns),
            "units": {column: source["units"].get(column, "identifier or UTC datetime") for column in source["data"]},
            "time_resolution": "hourly", "time_zone": "UTC",
            "license": "Generated test scenario; no external dataset license asserted",
            "sha256": sha256(path),
        }
        dry_root = root / "must_not_be_created"
        require(acquire_dataset(spec, dry_root, dry_run=True)["dry_run"], "Dry-run did not report intent")
        require(not dry_root.exists(), "Dry-run wrote files")
        acquired = acquire_dataset(spec, root / "raw_fixture")
        original_hash = sha256(path)
        try:
            acquire_dataset(spec, root / "raw_fixture")
        except AcquisitionError:
            pass
        else:
            raise ValueError("Acquisition overwrote an existing dataset")
        result = load_and_clean_csv(acquired["destination"], source["record_type"],
                                    units=source["units"], metadata=source["metadata"])
        require(result["report"]["canonical_validation"]["status"] == "passed", "Smoke cleaning failed canonical validation")
        require(sha256(path) == original_hash == sha256(acquired["destination"]), "Raw fixture changed")
        payload = {key: deepcopy(value) for key, value in source.items() if key != "data"}
        payload.update(result)
        cleaned_sources.append(payload)
    raw_before = inventory(root / "raw_fixture")
    integrated = integrate_datasets(cleaned_sources, expected_entities={
        "demand": [building["id"] for building in config["buildings"]],
        "generation": [config["solar"]["asset_id"], config["wind"]["asset_id"]],
        "weather": [config["weather"]["site_id"]], "price": [config["prices"]["grid_id"]],
    })
    require(integrated["report"]["complete"], "Synthetic smoke integration has incomplete joins")
    save_integrated_dataset(integrated, "smoke_integration", processed_root=root / "processed")
    split = split_datasets(cleaned_sources)
    saved = save_splits(split, "smoke", processed_root=root / "processed")
    require(verify_split_bundle(saved), "Saved smoke split integrity failed")
    sequences = {name: prepare_sequences(split["partitions"][name]) for name in PARTITIONS}
    require(inventory(root / "raw_fixture") == raw_before, "Replay changed acquired raw fixtures")
    return {"sources": cleaned_sources, "integrated": integrated, "split": split, "sequences": sequences}


def validate(skip_tests=False):
    result = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "mode": "EXPLICIT_SYNTHETIC_SMOKE_TEST; no real-data end-to-end claim",
              "checks": [], "limitations": [
                  "This verification uses synthetic artifacts only; real-source acquisition, campus coverage and external licenses are not validated.",
                  "Live HTTP downloads were not attempted. Public-download failures/redirects are tested with mocks.",
                  "Raw preservation is checked for this run; no historical raw baseline exists because permanent raw data is empty.",
                  "Synthetic patterns and capacities are uncalibrated assumptions; they do not demonstrate real-world forecasting or control performance.",
                  "Graph wiring is an assumed shared bus, not measured campus electrical topology.",
                  "Negative prices survive cleaning/integration, but the initial canonical price schema still rejects them.",
                  "Upstream unreported leakage cannot be detected. Test holdout protection uses workflow, hashes and no-overwrite rules, not access control.",
                  "A single synthetic year and seasonal chronological splits are not evidence of robust out-of-distribution performance.",
                  "Before simulator experiments, specify episode duration, terminal SOC treatment, action-feasibility handling and reward accounting.",
                  "No energy simulator, forecasting model or RL agent is implemented by this verification.",
              ]}
    raw_before = inventory(ROOT / "data/raw")
    processed_before = inventory(ROOT / "data/processed")

    def check(name, action):
        print(f"Checking: {name}", flush=True)
        try:
            details = action()
            result["checks"].append({"name": name, "status": "passed", "details": details})
            print(f"Passed: {name}", flush=True)
            return details
        except Exception as exc:
            result["checks"].append({"name": name, "status": "failed", "error": str(exc)})
            print(f"Failed: {name}: {exc}", flush=True)
            return None

    def requirements():
        files = ["docs/data_requirements.md", "docs/microgrid_scenario.md", "docs/dataset_acquisition.md"]
        require(all((ROOT / name).is_file() for name in files), "Required documentation missing")
        sources = load_sources(ROOT / "configs/data_sources.yaml")
        return {"documents": files, "source_statuses": {source["id"]: source["status"] for source in sources},
                "source_selection_document_present": (ROOT / "docs/dataset_sources.md").exists()}

    check("Requirements and source configuration", requirements)

    def artifacts():
        sources, generation = read_bundle()
        stats = []
        for source in sources:
            frame = source["data"]
            stats.append({"source_id": source["source_id"], "record_type": source["record_type"],
                          "rows": len(frame), "start_utc": frame["timestamp_utc"].min(),
                          "last_timestamp_utc": frame["timestamp_utc"].max(),
                          "missing_by_column": {column: int(frame[column].isna().sum()) for column in frame},
                          "units": source["units"], "origin": source["metadata"]["origin"], "synthetic": True})
        require(verify_split_bundle(SPLITS), "Existing split checksums failed")
        manifest = json.loads((FORECASTING / "manifest.json").read_text())
        for relative, digest in manifest["checksums"].items():
            path = (FORECASTING / relative).resolve()
            require(FORECASTING.resolve() in path.parents and sha256(path) == digest, "Changed forecasting artifact")
        graph = build_graph()
        with np.load(FORECASTING / "graph.npz", allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["edge_index"], graph["edge_index"])
            np.testing.assert_array_equal(saved["node_attributes"], graph["node_attributes"])
            require(saved["node_ids"].tolist() == graph["metadata"]["node_order"], "Node ordering mismatch")
        preprocessing = json.loads((SPLITS / "preprocessing.json").read_text())
        reconstructed = split_datasets(sources)
        require(reconstructed["preprocessing"]["scalers"] == preprocessing["scalers"], "Stored scalers do not reproduce from training-only fitting")
        shapes, partition_rows = {}, {}
        previous_end = None
        for name in PARTITIONS:
            bounds = preprocessing["boundaries"][name]
            if previous_end is not None:
                require(previous_end == bounds["start_utc"], "Split boundaries have a gap/overlap")
            previous_end = bounds["end_exclusive_utc"]
            partition_rows[name] = sum(len(item["data"]) for item in reconstructed["partitions"][name])
            replay = prepare_sequences(reconstructed["partitions"][name])
            with np.load(FORECASTING / f"{name}.npz", allow_pickle=False) as saved:
                for key, expected in replay["arrays"].items():
                    if expected.dtype.kind == "f":
                        np.testing.assert_allclose(saved[key], expected, rtol=1e-6, atol=1e-6)
                    else:
                        np.testing.assert_array_equal(saved[key], expected)
                require((saved["input_timestamps"][:, -1] < saved["target_timestamps"][:, 0]).all(), "Future input leakage")
                require(saved["target_timestamps"].min() >= np.datetime64(bounds["start_utc"].removesuffix("Z")), "Targets precede split")
                require(saved["target_timestamps"].max() < np.datetime64(bounds["end_exclusive_utc"].removesuffix("Z")), "Targets cross split end")
                shapes[name] = {key: list(saved[key].shape) for key in ("X", "y", "global_X")}
        return {"sources": stats, "total_records": sum(item["rows"] for item in stats),
                "measured_records": 0, "synthetic_records": sum(item["rows"] for item in stats),
                "missing_total": sum(sum(item["missing_by_column"].values()) for item in stats),
                "coverage_end_exclusive_utc": preprocessing["boundaries"]["test"]["end_exclusive_utc"],
                "splits": preprocessing["boundaries"], "partition_rows": partition_rows,
                "graph": {"nodes": len(graph["metadata"]["nodes"]), "directed_edges": graph["edge_index"].shape[1],
                          "node_order": graph["metadata"]["node_order"], "basis": "modeling_assumption"},
                "sequences": shapes, "seed": generation["scenario"]["seed"],
                "raw_dataset_files": len(list((ROOT / "data/raw").glob("*.csv")))}

    check("Existing canonical data, provenance, splits, scalers, graph and tensors", artifacts)

    def smoke():
        with tempfile.TemporaryDirectory(prefix="aeb-part2-validation-") as temporary:
            work = Path(temporary)
            scenario = load_scenario()
            scenario["hours"] = 336
            first = smoke_replay(work / "first", scenario)
            second = smoke_replay(work / "second", scenario)
            for left, right in zip(first["sources"], second["sources"]):
                pd.testing.assert_frame_equal(left["data"], right["data"], check_exact=True)
            pd.testing.assert_frame_equal(first["integrated"]["data"], second["integrated"]["data"], check_exact=True)
            require(first["split"]["preprocessing"]["scalers"] == second["split"]["preprocessing"]["scalers"], "Replay scaler mismatch")
            for name in PARTITIONS:
                for key in first["sequences"][name]["arrays"]:
                    np.testing.assert_array_equal(first["sequences"][name]["arrays"][key], second["sequences"][name]["arrays"][key])
            # Replay the saved full-year scenario independently; CSV bytes are
            # deterministic here, unlike clock/path-bearing metadata and ZIP files.
            saved = json.loads((SYNTHETIC / "generation_report.json").read_text())
            regenerated = generate_synthetic(saved["scenario"], processed_root=work / "unobserved")
            for source in regenerated["sources"]:
                name = f"synthetic_{source['category']}.csv"
                digest = hashlib.sha256(source["data"].to_csv(index=False).encode("utf-8")).hexdigest()
                require(digest == saved["file_checksums"][name], f"Full-year CSV reproduction failed: {name}")
            require(len(regenerated["sources"]) == 5, "Full-year synthetic replay was incomplete")
            return {"label": "SYNTHETIC ONLY", "hours": 336, "seed": scenario["seed"], "replays": 2,
                    "source_records": sum(len(source["data"]) for source in first["sources"]),
                    "integrated_records": len(first["integrated"]["data"]),
                    "integration_complete": first["integrated"]["report"]["complete"],
                    "integration_issues": len(first["integrated"]["report"]["issues"]),
                    "full_year_csv_byte_reproduction": True,
                    "stages": ["synthetic generation", "local acquisition/dry-run/overwrite protection", "raw preservation",
                               "cleaning", "schema", "UTC integration", "chronological splits", "training-only scaling", "graph sequences"],
                    "reproducibility": "Exact smoke frames/scalers/tensors; existing saved float32 tensors compared at rtol=atol=1e-6."}

    check("Offline end-to-end synthetic smoke and output reproducibility", smoke)
    if skip_tests:
        result["tests"] = {"status": "not_run", "reason": "--skip-tests selected"}
    else:
        print("Running offline pytest regression suite", flush=True)
        command = [sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider", "--tb=short"]
        process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=300)
        output = process.stdout + process.stderr
        match = re.search(r"(\d+) passed", output)
        result["tests"] = {"status": "passed" if process.returncode == 0 and match else "failed",
                           "passed_count": int(match.group(1)) if match else None,
                           "exit_code": process.returncode, "command": " ".join(command), "output": output}
    check("Permanent raw and existing processed files unchanged", lambda: (
        require(inventory(ROOT / "data/raw") == raw_before, "Permanent raw files changed"),
        require(all(Path(path).is_file() and sha256(path) == digest for path, digest in processed_before.items()), "Existing processed artifacts changed"),
        {"raw_files_checked": len(raw_before), "processed_files_checked": len(processed_before)},
    )[-1])
    result["passed"] = all(item["status"] == "passed" for item in result["checks"]) and result["tests"]["status"] == "passed"
    result["part3_assessment"] = (
        "READY for Part 3 development using the explicitly synthetic scenario; not ready for real-campus validation. "
        "Define episode duration, terminal SOC treatment and feasibility/reward rules before simulator experiments."
        if result["passed"] else "NOT VERIFIED: resolve failed checks or run the skipped tests before claiming Part 3 readiness."
    )
    return result


def render_report(result):
    lines = ["# Part 2 completion report", "", f"Validation time: {result['timestamp_utc']}", "",
             f"Mode: **{result['mode']}**", "", "## Verification results", "",
             "| Check | Result |", "|---|---|"]
    for item in result["checks"]:
        lines.append(f"| {item['name']} | {item['status']} |")
        if item["status"] == "failed":
            lines.append(f"\nFailure: {item['error']}\n")
    artifact = next((item.get("details") for item in result["checks"] if item["name"].startswith("Existing canonical") and item["status"] == "passed"), None)
    if artifact:
        lines += ["", "## Dataset inventory", "", f"Total source records: **{artifact['total_records']:,}**. Measured records: **0**. All records are synthetic.",
                  f"Missing cells across canonical source CSVs: **{artifact['missing_total']}**. This is not a missingness assessment of real campus data.", "",
                  "| Source | Records | Origin |", "|---|---:|---|"]
        lines += [f"| {item['source_id']} | {item['rows']:,} | explicitly synthetic |" for item in artifact["sources"]]
        lines += ["", "Coverage: **2026-01-01 00:00 UTC to 2027-01-01 00:00 UTC, end exclusive**; 8,760 hours per entity.",
                  f"Permanent `data/raw/` contains {artifact['raw_dataset_files']} dataset CSVs. Raw immutability is assessed against this run's initial file hashes.",
                  "Source provenance, units and per-column missing counts are included in the JSON audit and existing source manifests.", "",
                  "## Chronological partitions", "", "| Partition | Hours/entity | Total source rows | Start UTC | End UTC (exclusive) |", "|---|---:|---:|---|---|"]
        for name in PARTITIONS:
            bounds = artifact["splits"][name]
            lines.append(f"| {name} | {bounds['hours']:,} | {artifact['partition_rows'][name]:,} | {bounds['start_utc']} | {bounds['end_exclusive_utc']} |")
        lines += ["", "Scaling is fitted on training rows only, separately per source/entity; saved parameters reproduce. Original physical columns remain alongside dimensionless scaled features. No model selection or test-based tuning was performed.",
                  "", "## Graph and forecasting tensors", "",
                  f"Nodes: **{artifact['graph']['nodes']}**. Directed edges: **{artifact['graph']['directed_edges']}**, all modeling assumptions.",
                  "Node order: `" + ", ".join(artifact["graph"]["node_order"]) + "`.",
                  "The bus is a masked context node; seven nodes have demand/generation targets. Static capacities are hypothetical.", "",
                  "| Partition | X | y | Historical weather X |", "|---|---|---|---|"]
        for name, shapes in artifact["sequences"].items():
            lines.append(f"| {name} | {shapes['X']} | {shapes['y']} | {shapes['global_X']} |")
        lines += ["", "Tensor axes: sample, time, node, feature. Lookback and horizon are both 24 hours. Targets are kW; inputs use training-fitted scaling. Input times precede targets, and complete target horizons stay within their own split. NPZ files load with `allow_pickle=False`; no PyTorch or PyG dependency is required."]
    smoke = next((item.get("details") for item in result["checks"] if item["name"].startswith("Offline end-to-end") and item["status"] == "passed"), None)
    if smoke:
        lines += ["", "## End-to-end replay", "",
                  f"Two isolated, explicitly synthetic {smoke['hours']}-hour runs used seed {smoke['seed']}. Each processed {smoke['source_records']:,} source records and {smoke['integrated_records']:,} entity-hour records, with {smoke['integration_issues']} integration issues.",
                  "Stages: " + ", ".join(smoke["stages"]) + ".",
                  "Both runs reproduced frames, fitted parameters and tensors exactly. All five saved full-year source CSV byte hashes reproduced from the recorded scenario. Saved floating-point forecasting tensors match numerical replay within rtol=atol=1e-6; full metadata/NPZ byte identity is not claimed because paths, acquisition clocks and archive timestamps can differ."]
    requirement_details = next((item.get("details") for item in result["checks"] if item["name"] == "Requirements and source configuration" and item["status"] == "passed"), {})
    lines += ["", "## Sources, assumptions and provenance", "",
              "Configured source statuses: " + ", ".join(f"`{name}`: {status}" for name, status in requirement_details.get("source_statuses", {}).items()) + ".",
              f"`docs/dataset_sources.md` present: {requirement_details.get('source_selection_document_present', False)}. No real-data download, external dataset license verification or measured campus topology is claimed by this audit.",
              "Demand, demand energy, irradiance, temperature, wind speed, available solar/wind power and energy, and grid prices are synthetic. Renewable power is modeled from synthetic weather and remains synthetic; it is not modeled from measured observations. Timestamps, IDs, units, seed and scenario assumptions are recorded in source metadata.",
              "Assumptions include five buildings; 300 kW PV; 150 kW wind; 500 kWh battery with 150 kW charge/discharge limits, 0.95 efficiencies and SOC bounds 0.1–0.9; initial SOC 0.5; grid import/export limits 2,000/500 kW; artificial USD tariffs; and an Asia/Dhaka scheduling time zone. These are not observations. Maximum building demand totals 1,350 kW, below grid import capacity. No battery state trajectory or dispatch is generated.",
              "", "## Automated tests", "", f"Status: **{result['tests']['status']}**. Passed count: **{result['tests'].get('passed_count', 'not run')}**."]
    if "output" in result["tests"]:
        lines += ["", "```text", result["tests"]["output"].strip(), "```"]
    lines += ["", "## Unresolved limitations", ""] + [f"- {item}" for item in result["limitations"]]
    lines += ["", "## Repeat verification", "", "```powershell",
              "./.venv/Scripts/python.exe scripts/validate_data_pipeline.py --report docs/part2_recheck.md --json-output data/processed/part2_recheck.json",
              "```", "", "Use fresh report names: existing reports and datasets are never overwritten. Validation replays use temporary directories and do not replace production artifacts.",
              "", "## Part 3 readiness", "", "**" + result["part3_assessment"] + "**", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args(argv)
    for path in (args.report, args.json_output):
        if path is not None and path.exists():
            parser.error(f"Refusing to overwrite existing report: {path}")
    result = validate(args.skip_tests)
    for path, text in ((args.report, render_report(result)),
                       (args.json_output, json.dumps(result, indent=2, allow_nan=False))):
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8") as stream:
                stream.write(text + "\n")
    print(json.dumps({"passed": result["passed"], "checks": [
        {key: value for key, value in item.items() if key != "details"} for item in result["checks"]], "tests": result["tests"],
                      "part3_assessment": result["part3_assessment"]}, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
