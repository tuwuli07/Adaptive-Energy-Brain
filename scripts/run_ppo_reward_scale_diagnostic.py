"""Run the approved paired PPO reward-scale diagnostic."""

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
    args = parser.parse_args(argv)
    from src.rl.reward_scale_experiment import run_reward_scale_diagnostic

    result = run_reward_scale_diagnostic(
        split_root=args.split_root,
        output=args.output,
        assets=ROOT / "configs/synthetic_scenario.yaml",
        rating_origin="hypothetical_example",
    )
    print(json.dumps(
        {key: result[key] for key in ("status", "wall_seconds", "runs", "validations")},
        indent=2,
    ))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
