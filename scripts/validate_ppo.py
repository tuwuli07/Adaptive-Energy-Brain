"""Replay pilot checkpoints and all four baselines on the validation partition."""

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
    parser.add_argument("--checkpoint-records", type=Path, required=True,
                        help="JSON list with id/configuration/seed/path for each checkpoint")
    parser.add_argument("--assets", type=Path, default=ROOT / "configs/synthetic_scenario.yaml")
    args = parser.parse_args(argv)
    from src.evaluation.ppo_validation import validate_policies
    records = json.loads(args.checkpoint_records.read_text(encoding="utf-8"))
    report = validate_policies(split_root=args.split_root, assets=args.assets,
                               rating_origin="hypothetical_example", output=args.output, checkpoints=records)
    print(json.dumps({"status": report["status"], "runtime_seconds": report["runtime_seconds"]}))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
