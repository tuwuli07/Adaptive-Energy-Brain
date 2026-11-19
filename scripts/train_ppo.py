"""Run the Part 6.3 CPU PPO infrastructure demonstration on verified train data."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, default=os.environ.get("AEB_SPLIT_ROOT"))
    parser.add_argument("--assets", type=Path, default=ROOT / "configs/synthetic_scenario.yaml")
    parser.add_argument("--rating-origin", choices=("hypothetical_example", "documented_specification", "mixed"),
                        default="hypothetical_example")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/ppo_training.yaml")
    parser.add_argument("--simulation-config", type=Path, default=ROOT / "configs/simulation.yaml")
    parser.add_argument("--costs", type=Path, default=ROOT / "configs/simulation_costs.yaml")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path, help="Trusted completed-update checkpoint bundle")
    args = parser.parse_args(argv)
    if args.split_root is None:
        parser.error("Provide --split-root or AEB_SPLIT_ROOT.")
    output = args.output
    if output is None:
        base = Path(os.environ.get("AEB_EXPERIMENTS_ROOT", ROOT / "results/ppo"))
        output = base / ("part6_3-ppo-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    from src.rl.training import train_ppo
    try:
        result = train_ppo(split_root=args.split_root, assets=args.assets,
                           rating_origin=args.rating_origin, output=output,
                           training_config=args.config, simulation_config=args.simulation_config,
                           costs_config=args.costs, resume=args.resume)
        print(json.dumps({"status": result["status"], "output": str(output.resolve()),
                          "progress": result["progress"], "verification": result["verification"]}, indent=2))
        return 0
    except KeyboardInterrupt:
        print(f"Training interrupted. Recovery artifacts: {output}", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"PPO training failed: {exc}. Inspect output: {output}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
