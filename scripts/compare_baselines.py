"""Compare saved baseline experiment outputs against zero-action."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.baseline_comparison import compare_baseline_experiment
from src.evaluation.baseline_visualization import plot_baseline_experiment


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="Fresh comparison output directory; defaults to a sibling directory.",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Also generate seven comparison plots from the validated experiment.",
    )
    args = parser.parse_args(argv)
    try:
        report = compare_baseline_experiment(args.experiment, args.output)
        plot_paths = []
        if args.plots:
            plot_output = (
                args.output
                if args.output is not None
                else args.experiment.resolve().with_name(
                    f"{args.experiment.name}_comparison"
                )
            )
            plot_paths = plot_baseline_experiment(args.experiment, plot_output)
        print(json.dumps({
            "controllers": len(report["controllers"]),
            "currency_code": report["currency_code"],
            "comparison_csv": report["comparison_csv"],
            "plots": [path.name for path in plot_paths],
            "status": report["status"],
        }, indent=2))
        return 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"Baseline comparison error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
