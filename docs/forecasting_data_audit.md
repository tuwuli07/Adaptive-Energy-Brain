# Part 5.1 Forecasting Data Audit

## Audit scope and evidence

This audit examines the saved Part 2 split bundle
`data/processed/splits_hypothetical_campus_v1/` and forecasting tensors in
`data/processed/forecasting_hypothetical_campus_v1/`, along with
`configs/data_schema.yaml`, `configs/data_sources.yaml`,
`configs/splitting.yaml`, `configs/forecasting_data.yaml`,
`src/data/schema.py`, `src/data/splitting.py`,
`src/data/graph_builder.py`, and `src/data/sequences.py`.

The split bundle checksum manifest verified. The source rows in each partition
were validated against the canonical schema after excluding only generated
`*_scaled` columns (which are derived features, not canonical source columns).
The result is an audit of checked-in project artifacts, not a claim that the
dataset is measured campus data. `data/raw/` contains no CSV files, and the five
configured real-data source entries are still pending.

## Actual available targets and data fields

The only available forecasting dataset is the explicitly synthetic
`synthetic_hypothetical_campus_v1` dataset. Its source records cover 2026-01-01
00:00 UTC through 2027-01-01 00:00 UTC, end exclusive, on a complete hourly
grid.

| Target series | Entity IDs | Target column | Physical unit | Target scope |
|---|---|---|---|---|
| Building electricity demand | B01, B02, B03, B04, B05 | `building_load_kw` | kW | Five individual building nodes; no campus-total demand label is stored. |
| Solar available generation | PV01 | `available_generation_kw` | kW | One solar generation asset/node; available power before dispatch/curtailment. |
| Wind available generation | W01 | `available_generation_kw` | kW | One wind generation asset/node; available power before dispatch/curtailment. |

Thus, **seven node-level targets** are available. Campus demand could be
calculated by summing the five building loads, but that is not a separately
stored or configured system-level target. Energy columns (`demand_energy_kwh`
and `available_generation_energy_kwh`) are derived interval quantities, not
the configured forecast targets.

Canonical source columns and declared units are:

| Record type | Canonical fields | Numeric units |
|---|---|---|
| Demand | `timestamp_utc`, `interval_hours`, `building_id`, `building_load_kw`, `demand_energy_kwh` | `interval_hours`: h; `building_load_kw`: kW; `demand_energy_kwh`: kWh |
| Generation (solar and wind) | `timestamp_utc`, `interval_hours`, `asset_id`, `generation_type`, `available_generation_kw`, `available_generation_energy_kwh`, `generation_capacity_kw` | `interval_hours`: h; `available_generation_kw`: kW; `available_generation_energy_kwh`: kWh; `generation_capacity_kw`: kW |
| Weather | `timestamp_utc`, `interval_hours`, `site_id`, `irradiance_w_m2`, `air_temperature_c`, `wind_speed_m_s` | `interval_hours`: h; `irradiance_w_m2`: W/m²; `air_temperature_c`: °C (`degC` in the canonical unit declaration); `wind_speed_m_s`: m/s |
| Grid prices | `timestamp_utc`, `interval_hours`, `grid_id`, `import_price`, `export_price` | `interval_hours`: h; `import_price`, `export_price`: USD/kWh |

Timestamps are hourly UTC. Entity IDs and generation type are identifiers or
categories, not numeric model features. The sequence builder selects one
historical scaled power feature per graph node and three historical scaled
weather features; it does not feed energy columns, capacity columns, prices,
IDs, or raw timestamps into the current tensors.

## Graph and topology provenance

The configured graph has eight nodes, in authoritative order
`B01, B02, B03, B04, B05, PV01, W01, BUS01`. It is an **assumed bidirectional
shared-bus star**, not a graph measured from campus electrical records. Its
seven spokes are:

| Node | Directed configured edges |
|---|---|
| Each of B01–B05 | node → BUS01 and BUS01 → node |
| PV01 | PV01 → BUS01 and BUS01 → PV01 |
| W01 | W01 → BUS01 and BUS01 → W01 |

That yields 14 directed edges. They represent unweighted membership only:
edge weight 1 is an interpretation if an unweighted message-passing operator
needs a numeric weight; **no edge-weight array or measured edge property is
provided**. There are no self-loops. The saved edge index is `[2, 14]`.

The bus `BUS01` is a masked context node, not a measured zero-valued target.
Its target and input masks are false; the other seven node masks are true. The
saved node-attribute matrix has shape `[8, 6]`: one-hot node type
(building/solar/wind/bus), capacity divided by a 1000 kW engineering scale,
and a capacity-known flag. Configured capacities are hypothetical assumptions,
not observed ratings. The graph reference is
`docs/microgrid_scenario.md`, which describes a provisional shared bus. No
measured wiring, impedance, distance, transfer limit, or topology weights are
available. Model results using these edges must be described as experiments
with an assumed topology.

## Forecasting sequences and dimensions

The current authoritative sequence settings are in
`configs/forecasting_data.yaml`: **24-hour lookback**, **24-hour horizon**,
and stride 1 hour. These defaults are supported by the complete hourly dataset
and saved tensors for all three partitions. They are configurable positive
integer parameters, but changing them changes tensor shapes and sample counts.

| Partition | Hours | X: node history | y: node targets | global_X: weather history | Samples |
|---|---:|---|---|---|---:|
| Train | 6,132 | `[6085, 24, 8, 1]` | `[6085, 24, 8, 1]` | `[6085, 24, 3]` | 6,085 |
| Validation | 1,314 | `[1267, 24, 8, 1]` | `[1267, 24, 8, 1]` | `[1267, 24, 3]` | 1,267 |
| Test | 1,314 | `[1267, 24, 8, 1]` | `[1267, 24, 8, 1]` | `[1267, 24, 3]` | 1,267 |

`X` and `global_X` use `float32`; `y` is `float32` in original kW.
`input_timestamps` and `target_timestamps` are `[sample, 24]` UTC
`datetime64[ns]` arrays. Masks have shape `[8, 1]` and broadcast over batches
and time. With 24-hour lookback, 24-hour horizon, and stride 1, each
partition's count is `partition_hours - 24 - 24 + 1`; the saved sample counts
match this calculation.

## Splits, scaling, and leakage checks

Partitions are contiguous chronological slices of unique hourly UTC timestamps;
the checksum-verified boundaries are:

| Partition | Start UTC (inclusive) | End UTC (exclusive) | Hours |
|---|---|---|---:|
| Train | 2026-01-01T00:00:00Z | 2026-09-13T12:00:00Z | 6,132 |
| Validation | 2026-09-13T12:00:00Z | 2026-11-07T06:00:00Z | 1,314 |
| Test | 2026-11-07T06:00:00Z | 2027-01-01T00:00:00Z | 1,314 |

The allocation is 70% train, 15% validation, and 15% test by hourly
timestamps (train and validation counts floored; test receives the remainder).
All sources/entities align on the same complete UTC grid. The saved scaler
metadata says `fitted_on: train`; the scaler implementation fits per source
and entity, using population standard deviation (`ddof=0`) and scale 1 for a
constant feature. A direct recomputation from the saved training records
matched the persisted scaler parameters. Original physical columns remain
alongside dimensionless `*_scaled` columns; validation/test are transformed
with the saved training parameters without clipping or refitting.

The sequence builder reads one labeled partition at a time, requires
train-fitted preprocessing metadata, and does not fit preprocessing. It uses
past scaled power and past scaled weather only. For every generated window,
the final input timestamp precedes the first target timestamp and the full
target horizon stays inside that same split. Verification confirmed this for
the saved train, validation, and test tensors. No cross-partition context is
borrowed.

Future target values in the saved `y` arrays are supervised labels, not
permitted input features. Future realized weather and prices are present in
the historical full-year source tables but are not part of `X` or `global_X`
for a forecast made at the corresponding earlier time. No forecast-vintage or
future weather forecast dataset is available. The model input currently has
**no declared known-future covariates**. Calendar features could be derived
from timestamps in a future, explicitly documented change; no such feature is
currently supplied. Grid prices likewise must not be used as advance-known
tariffs without evidence they were available at the forecast issue time.

## Missing values and limitations

- The canonical source schema passed for the saved source records in each
  partition. No null or non-finite values were found in the saved CSV columns
  in train, validation, or test; the configured sequence observations contain
  no missing-value imputation.
- This is one synthetic hypothetical year and one synthetic site, with five
  buildings, one solar asset, and one wind asset. It is not measured demand,
  renewable output, weather, or campus topology. There are no verified source
  licenses, real campus observations, or permanent raw CSVs.
- The target values are kept in physical kW while inputs are train-standardized.
  Target distributions differ by asset and scale; the current data package does
  not scale targets or prescribe a training loss. Part 5.2 must define a
  masked loss that excludes BUS01 and report metrics by node/target type, in kW.
  Any target normalization must be fit on training targets only and inverted
  before physical-unit reporting.
- There is no existing forecasting model, forecast-specific performance
  baseline, or forecast accuracy result. Part 4's controllers are policy
  baselines, not forecasting benchmarks.
- Split hashes, code checks, and workflow conventions detect accidental
  changes; they do not enforce access control or independently establish
  truthful upstream provenance.

**Audit conclusion:** the saved synthetic artifacts are internally compatible
with configurable 24-hour lookback and horizon, seven node-level kW targets,
three historical weather inputs, and a masked assumed-topology graph. They are
adequate for development and smoke experiments only. Real-campus forecasting
readiness remains blocked on approved measured datasets and defensible
topology/provenance.
