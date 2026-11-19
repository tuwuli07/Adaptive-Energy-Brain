"""Execute the approved two-run Part 6.4 pilot and fixed validation replay."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/ppo_validation.yaml")
    parser.add_argument("--assets", type=Path, default=ROOT / "configs/synthetic_scenario.yaml")
    parser.add_argument("--rating-origin", default="hypothetical_example",
                        choices=("hypothetical_example", "documented_specification", "mixed"))
    args = parser.parse_args(argv)
    from src.rl.experiments import run_pilot
    result = run_pilot(split_root=args.split_root, output=args.output, assets=args.assets,
                       rating_origin=args.rating_origin, experiment_config=args.config)
    print(json.dumps({"status": result["status"], "wall_seconds": result["wall_seconds"],
                      "output": str(args.output.resolve()), "screening": result["screening"]}, indent=2))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
