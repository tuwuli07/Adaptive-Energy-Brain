"""Command-line entry point for configured dataset acquisition."""

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.acquisition import AcquisitionError, acquire_dataset, load_sources


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/data_sources.yaml")
    parser.add_argument("--source", action="append", help="Source id; repeat to select multiple sources")
    parser.add_argument("--dry-run", action="store_true", help="Validate and report without network or writes")
    parser.add_argument("--allow-large", action="store_true", help="Explicitly approve selected files exceeding 50 MiB")
    args = parser.parse_args(argv)
    try:
        sources = load_sources(args.config)
        if args.source:
            unknown = set(args.source) - {source["id"] for source in sources}
            if unknown:
                raise AcquisitionError(f"Unknown source ids: {sorted(unknown)}")
            sources = [source for source in sources if source["id"] in args.source]
        for source in sources:
            result = acquire_dataset(
                source, PROJECT_ROOT / "data/raw",
                base_dir=args.config.resolve().parent,
                dry_run=args.dry_run, allow_large=args.allow_large,
            )
            print(json.dumps(result, indent=2))
        return 0
    except AcquisitionError as exc:
        print(f"Acquisition error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
