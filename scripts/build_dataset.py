"""Build an hourly dataset from an explicit manifest of already-cleaned CSVs.

Manifest YAML fields: datasets (list), optional start/end (UTC, end exclusive),
expected_entities (mapping of record type to ID list). Each dataset entry needs
source_id, record_type, path, units, metadata, measurement_kind: interval_mean.
Optional source_timezone, energy_allocation: uniform and mean_allocation:
constant are explicit policies. Splitting coarse/straddling observations needs
the corresponding allocation policy; no measured hourly detail is invented.
Paths resolve relative to the manifest. Original metadata must include source_id,
origin, source_metadata; prices additionally need currency_code. Raw source
approval configuration is not an integration manifest. No acquisition, cleaning,
synthetic profile generation or training occurs here.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.integration import IntegrationError, integrate_datasets, save_integrated_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--name", default="hourly_dataset")
    parser.add_argument("--report-only", action="store_true", help="Inspect integration without saving files")
    parser.add_argument("--allow-incomplete", action="store_true", help="Explicitly save flagged gaps as missing values")
    args = parser.parse_args(argv)
    try:
        manifest = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or not isinstance(manifest.get("datasets"), list):
            raise IntegrationError("Manifest requires a datasets list.")
        if set(manifest) - {"datasets", "start", "end", "expected_entities"}:
            raise IntegrationError("Unknown manifest options; use datasets/start/end/expected_entities.")
        sources = []
        for item in manifest["datasets"]:
            if not isinstance(item, dict) or "path" not in item:
                raise IntegrationError("Each manifest source requires a cleaned CSV path.")
            source = dict(item)
            path = Path(source.pop("path"))
            if not path.is_absolute():
                path = args.manifest.resolve().parent / path
            path = path.resolve()
            if PROJECT_ROOT.joinpath("data/raw").resolve() in path.parents:
                raise IntegrationError("Manifest must reference cleaned CSVs, not files under data/raw/.")
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
                stream.seek(0)
                source["data"] = pd.read_csv(stream, dtype=object, keep_default_na=False, na_values=[""])
            if not isinstance(source.get("metadata"), dict):
                raise IntegrationError("Each source requires provenance metadata.")
            source["metadata"] = {**source["metadata"], "integration_input": {"path": str(path), "sha256": digest}}
            sources.append(source)
        result = integrate_datasets(sources, start=manifest.get("start"), end=manifest.get("end"),
                                    expected_entities=manifest.get("expected_entities"))
        print(json.dumps(result["report"], indent=2, allow_nan=False))
        if not args.report_only:
            destination = save_integrated_dataset(result, args.name, allow_incomplete=args.allow_incomplete)
            print(f"Saved validated hourly bundle: {destination}")
        return 0
    except (IntegrationError, OSError, ValueError, TypeError, yaml.YAMLError) as exc:
        print(f"Integration error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
