# Dataset acquisition

The pipeline copies or downloads CSV bytes into `data/raw/` without preprocessing.
`configs/data_sources.yaml` currently contains five **pending placeholders**;
no real dataset URL, campus observation, license or source approval is asserted.
`docs/dataset_sources.md` was absent when the pipeline was created. Requirements
come from `data_requirements.md` and `microgrid_scenario.md`.

## Configuration contract

The top-level `sources` list contains unique `id` and `filename` values. Entries
declare `category` (solar, wind, demand, weather, grid_prices), `description`,
`type` (local_csv or http), `local_path`, `download_url`, `filename`,
`expected_columns`, `units`, `time_resolution`, `time_zone`, `license`, and
`status` (pending or verified). An optional `sha256` supplies an expected digest.

Pending entries are reported as skipped, even without dry-run; they do not cause
network requests or destination directory creation. Their column/unit entries
are provisional data requirements. Actual source headers and units must be
documented after selection; acquisition does not validate or rename CSV columns.
Time zone and license are null until known. Hourly resolution describes the
initial collection target, not a verified source property.

After reviewing an actual source and its licensing/access conditions, set
`status: verified` and document resolution, time zone and license. For a local
CSV, provide `local_path` and leave `download_url: null`. Relative input paths
resolve from the YAML file's directory. For HTTP/HTTPS, provide a reviewed public
`download_url` and leave `local_path: null`. Never put credentials or expiring
secret URLs in configuration. A verified flag records human review; the software
cannot verify a publisher's legitimacy or legal permissions.

## PowerShell commands

Run from the project root using the existing virtual environment:

```powershell
# Safe with current pending placeholders; reports five skips.
.\.venv\Scripts\python.exe scripts/download_data.py --dry-run

# Once the corresponding source is genuinely verified:
.\.venv\Scripts\python.exe scripts/download_data.py --source campus_demand --dry-run
.\.venv\Scripts\python.exe scripts/download_data.py --source campus_demand

# Approve a selected source over 50 MiB only after reviewing its size.
.\.venv\Scripts\python.exe scripts/download_data.py --source campus_demand --allow-large

# Offline test suite; disables optional pytest disk caching.
.\.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider
```

`--config path/to/sources.yaml` selects another configuration. Repeating
`--source` selects multiple IDs; omitting it processes all configured entries.
Any acquisition failure produces a useful stderr message and nonzero exit code;
earlier successful files in a batch are retained. The batch is not transactional.

## Integrity and operational behavior

- Each successful acquisition has a `<filename>.metadata.json` sidecar containing
  the source configuration/location, UTC acquisition timestamp, byte size,
  SHA-256, optional expected digest, and final HTTP URL.
- Existing dataset or sidecar paths are refused, including dangling symlinks.
  Choose a different versioned filename explicitly when a new snapshot is needed.
- Local originals are opened read-only. No newline, encoding, schema, unit, or
  timestamp conversion occurs.
- Transfers are staged under `data/raw/`. Empty responses, interrupted reads,
  mismatched Content-Length, expected-checksum failures and exceeded size limits
  are rejected and temporary files removed on handled errors.
- A transfer without Content-Length is capped during streaming, but completeness
  cannot be independently established from EOF alone. A publisher checksum is
  recommended; the recorded checksum alone does not prove publisher authenticity.
- Without `--allow-large`, each file is limited to 50 MiB, including local copies
  and unknown-length HTTP streams. This is a per-file limit, not a batch budget.
- Dry-run validates configuration and local file availability/size without writes,
  DNS or HTTP requests. HTTP availability, size and actual integrity cannot be
  checked in dry-run.
- Public-address checks apply to direct hosts and redirect targets; private hosts,
  embedded URL credentials and HTTPS downgrade redirects are rejected. Normal
  certificate verification remains enabled. Environment proxies are ignored.
  Use reviewed sources: DNS checks are not a complete hostile-server sandbox.
- Requests use a 30-second socket timeout and fail clearly without automatic
  retries. A timeout is not a total transfer-duration limit.
- Publishing uses same-filesystem hard links to avoid replacement races and
  expose only completed payloads. Unsupported filesystems fail safely. Dataset
  and metadata publication are individually atomic, not a single crash-atomic
  transaction; after an abrupt process/system crash, inspect any orphan sidecar,
  dataset or `.acquire-*` directory before retrying. No automatic overwrite occurs.

No new dependency is needed beyond Python's standard library and existing
PyYAML/pytest. No real data is acquired by the offline tests.
