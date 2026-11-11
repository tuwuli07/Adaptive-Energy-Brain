# Part 2 completion report

Validation time: 2026-11-09T19:34:12.628228+00:00

Mode: **EXPLICIT_SYNTHETIC_SMOKE_TEST; no real-data end-to-end claim**

## Verification results

| Check | Result |
|---|---|
| Requirements and source configuration | passed |
| Existing canonical data, provenance, splits, scalers, graph and tensors | passed |
| Offline end-to-end synthetic smoke and output reproducibility | passed |
| Permanent raw and existing processed files unchanged | passed |

## Dataset inventory

Total source records: **78,840**. Measured records: **0**. All records are synthetic.
Missing cells across canonical source CSVs: **0**. This is not a missingness assessment of real campus data.

| Source | Records | Origin |
|---|---:|---|
| synthetic_hypothetical_campus_v1_weather | 8,760 | explicitly synthetic |
| synthetic_hypothetical_campus_v1_solar | 8,760 | explicitly synthetic |
| synthetic_hypothetical_campus_v1_wind | 8,760 | explicitly synthetic |
| synthetic_hypothetical_campus_v1_demand | 43,800 | explicitly synthetic |
| synthetic_hypothetical_campus_v1_grid_prices | 8,760 | explicitly synthetic |

Coverage: **2026-01-01 00:00 UTC to 2027-01-01 00:00 UTC, end exclusive**; 8,760 hours per entity.
Permanent `data/raw/` contains 0 dataset CSVs. Raw immutability is assessed against this run's initial file hashes.
Source provenance, units and per-column missing counts are included in the JSON audit and existing source manifests.

## Chronological partitions

| Partition | Hours/entity | Total source rows | Start UTC | End UTC (exclusive) |
|---|---:|---:|---|---|
| train | 6,132 | 55,188 | 2026-01-01T00:00:00Z | 2026-09-13T12:00:00Z |
| validation | 1,314 | 11,826 | 2026-09-13T12:00:00Z | 2026-11-07T06:00:00Z |
| test | 1,314 | 11,826 | 2026-11-07T06:00:00Z | 2027-01-01T00:00:00Z |

Scaling is fitted on training rows only, separately per source/entity; saved parameters reproduce. Original physical columns remain alongside dimensionless scaled features. No model selection or test-based tuning was performed.

## Graph and forecasting tensors

Nodes: **8**. Directed edges: **14**, all modeling assumptions.
Node order: `B01, B02, B03, B04, B05, PV01, W01, BUS01`.
The bus is a masked context node; seven nodes have demand/generation targets. Static capacities are hypothetical.

| Partition | X | y | Historical weather X |
|---|---|---|---|
| train | [6085, 24, 8, 1] | [6085, 24, 8, 1] | [6085, 24, 3] |
| validation | [1267, 24, 8, 1] | [1267, 24, 8, 1] | [1267, 24, 3] |
| test | [1267, 24, 8, 1] | [1267, 24, 8, 1] | [1267, 24, 3] |

Tensor axes: sample, time, node, feature. Lookback and horizon are both 24 hours. Targets are kW; inputs use training-fitted scaling. Input times precede targets, and complete target horizons stay within their own split. NPZ files load with `allow_pickle=False`; no PyTorch or PyG dependency is required.

## End-to-end replay

Two isolated, explicitly synthetic 336-hour runs used seed 42. Each processed 3,024 source records and 3,024 entity-hour records, with 0 integration issues.
Stages: synthetic generation, local acquisition/dry-run/overwrite protection, raw preservation, cleaning, schema, UTC integration, chronological splits, training-only scaling, graph sequences.
Both runs reproduced frames, fitted parameters and tensors exactly. All five saved full-year source CSV byte hashes reproduced from the recorded scenario. Saved floating-point forecasting tensors match numerical replay within rtol=atol=1e-6; full metadata/NPZ byte identity is not claimed because paths, acquisition clocks and archive timestamps can differ.

## Sources, assumptions and provenance

Configured source statuses: `campus_solar`: pending, `campus_wind`: pending, `campus_demand`: pending, `campus_weather`: pending, `campus_grid_prices`: pending.
`docs/dataset_sources.md` present: False. No real-data download, external dataset license verification or measured campus topology is claimed by this audit.
Demand, demand energy, irradiance, temperature, wind speed, available solar/wind power and energy, and grid prices are synthetic. Renewable power is modeled from synthetic weather and remains synthetic; it is not modeled from measured observations. Timestamps, IDs, units, seed and scenario assumptions are recorded in source metadata.
Assumptions include five buildings; 300 kW PV; 150 kW wind; 500 kWh battery with 150 kW charge/discharge limits, 0.95 efficiencies and SOC bounds 0.1–0.9; initial SOC 0.5; grid import/export limits 2,000/500 kW; artificial USD tariffs; and an Asia/Dhaka scheduling time zone. These are not observations. Maximum building demand totals 1,350 kW, below grid import capacity. No battery state trajectory or dispatch is generated.

## Automated tests

Status: **passed**. Passed count: **212**.

```text
........................................................................ [ 33%]
........................................................................ [ 67%]
....................................................................     [100%]
212 passed in 29.76s
```

## Unresolved limitations

- This verification uses synthetic artifacts only; real-source acquisition, campus coverage and external licenses are not validated.
- Live HTTP downloads were not attempted. Public-download failures/redirects are tested with mocks.
- Raw preservation is checked for this run; no historical raw baseline exists because permanent raw data is empty.
- Synthetic patterns and capacities are uncalibrated assumptions; they do not demonstrate real-world forecasting or control performance.
- Graph wiring is an assumed shared bus, not measured campus electrical topology.
- Negative prices survive cleaning/integration, but the initial canonical price schema still rejects them.
- Upstream unreported leakage cannot be detected. Test holdout protection uses workflow, hashes and no-overwrite rules, not access control.
- A single synthetic year and seasonal chronological splits are not evidence of robust out-of-distribution performance.
- Before simulator experiments, specify episode duration, terminal SOC treatment, action-feasibility handling and reward accounting.
- No energy simulator, forecasting model or RL agent is implemented by this verification.

## Repeat verification

```powershell
./.venv/Scripts/python.exe scripts/validate_data_pipeline.py --report docs/part2_recheck.md --json-output data/processed/part2_recheck.json
```

Use fresh report names: existing reports and datasets are never overwritten. Validation replays use temporary directories and do not replace production artifacts.

## Part 3 readiness

**READY for Part 3 development using the explicitly synthetic scenario; not ready for real-campus validation. Define episode duration, terminal SOC treatment and feasibility/reward rules before simulator experiments.**

