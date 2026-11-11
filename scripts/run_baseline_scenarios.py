"""Run registered baseline controllers across deterministic stress scenarios."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.baseline_scenarios import (
    available_scenarios,
    run_baseline_scenarios,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path)
    source.add_argument("--split-root", type=Path)
    parser.add_argument(
        "--partition",
        choices=("train", "validation", "test"),
        help="Required with --split-root; only that partition is loaded.",
    )
    parser.add_argument("--assets", required=True, type=Path)
    parser.add_argument(
        "--rating-origin",
        required=True,
        choices=("hypothetical_example", "documented_specification", "mixed"),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--scenario",
        action="append",
        choices=available_scenarios(),
        help="Scenario to run; repeat to select multiple. Defaults to all.",
    )
    parser.add_argument("--controller", action="append")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--simulation-config", type=Path)
    parser.add_argument("--costs", type=Path)
    parser.add_argument("--baseline-config", type=Path)
    args = parser.parse_args(argv)
    try:
        index = run_baseline_scenarios(
            manifest=args.manifest,
            split_root=args.split_root,
            partition=args.partition,
            assets=args.assets,
            rating_origin=args.rating_origin,
            output=args.output,
            scenarios=args.scenario,
            controllers=args.controller,
            start=args.start,
            end=args.end,
            simulation_config=args.simulation_config,
            costs=args.costs,
            baseline_config=args.baseline_config,
        )
        print(json.dumps({
            "status": index["status"],
            "scenario_count": len(index["scenario_runs"]),
            "output": str(args.output.resolve()),
            "scenario_ids": [
                item["scenario_id"] for item in index["scenario_runs"]
            ],
        }, indent=2))
        return 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"Baseline scenario error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
