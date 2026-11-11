# Microgrid data requirements

## Scope and provenance

This contract supports the provisional five-building scenario in
[microgrid_scenario.md](microgrid_scenario.md). It defines future inputs and
outputs, not available datasets or implemented functions.

Origin labels used below:

- **Measured (M):** historical meter, weather, or published tariff observations
  with source, license, location, timestamp semantics, and units documented.
  A tariff is an externally observed schedule rather than a physical sensor.
- **Simulated (S):** explicitly assumed parameters or generated scenario inputs.
  Historical replay remains M at source; modified/scaled profiles are S and must
  retain their source linkage and transformation record.
- **Derived (D):** deterministic aggregates, simulator states, forecasts,
  actions, residuals, or optimization outputs. Mark the subtype (calculated,
  simulator-derived, or model-derived) and its upstream provenance.

Nothing in this project currently qualifies as a measured campus observation.
Every input dataset needs a provenance manifest: `source_id`, origin label,
source URL/file and license, entity mapping, original unit and time zone,
coverage, conversion/resampling steps, missing-data flags, and synthetic
generation method/seed when applicable. These metadata are text or flags
(unit: not applicable); a seed is a dimensionless integer. Preserve raw inputs.

Requirement levels: **R** = required for the baseline simulator; **A** = required
for the indicated advanced research stage, not for baseline execution; **O** =
optional. Required outputs are computed, not extra input datasets. Power units
are AC-side interval averages unless stated otherwise. SOC and probabilities are
fractions in [0, 1], not percentages. Choose one `currency_code`; monetary units
below mean that declared currency, with no assumed exchange rate.

## Time-series inputs

| Variable | Level | Description and granularity | Unit | Origin | History or parameter; intended use |
|---|---|---|---|---|---|
| timestamp_utc | R | Start of each consecutive hourly interval | ISO 8601 UTC datetime; no physical unit | M/S | Required on all inputs; alignment, causal forecasting, simulation and risk replay |
| building_load_kw | R | Nonnegative demand for each B01-B05; excludes shared generation | kW | M or explicitly S | Historical profile or synthetic scenario series; building forecasting targets and simulator/RL demand |
| pv_available_kw | R | PV power available before curtailment, one shared PV01 series | kW | M/S, or D from weather | Historical profile or synthetic/converted series; generation forecasting, dispatch feasibility and risk |
| wind_available_kw | R | Wind power available before curtailment, one shared W01 series | kW | M/S, or D from weather | Same role as PV; never substitute dispatched power without accounting for curtailment |
| import_price | R | Energy purchase price for the campus grid connection | currency/kWh | M/S | Known schedule, possibly a constant parameter expanded hourly; operating cost, RL and risk |
| export_price | R | Energy sale price; use explicit zero when export is disabled | currency/kWh | M/S | Known schedule or constant parameter; cost and RL/risk accounting |
| irradiance_w_m2 | O; A if used for PV conversion | Irradiance with plane specified (e.g. GHI or plane-of-array); not interchangeable without a conversion model | W/m^2 | M/S | Historical weather; PV forecast covariate or generation conversion |
| air_temperature_c | O; A if required by conversion model | Ambient temperature at documented site | degrees C | M/S | Historical weather; load/PV covariate and possible PV conversion |
| wind_speed_m_s | O; A if used for wind conversion | Speed at a documented measurement height | m/s | M/S | Historical weather; wind covariate/conversion using a specified height correction and power curve |
| grid_available | O | Whether grid operation is possible during an interval | binary flag, dimensionless | M/S | Constant 1 in baseline; outage stress testing requires unmet-load accounting |
| occupancy_fraction | O | Relative building occupancy; not an identified-person record | fraction, dimensionless | M/S | Historical/synthetic building covariate for later demand studies |
| carbon_intensity | O | Emissions associated with grid imports | kg CO2e/kWh | M/S | Historical schedule or scenario parameter; later emissions evaluation/objectives |
| hour_of_day | O | Campus-local interval-start hour | integer 0-23, dimensionless | D from timestamp/time zone | Calendar forecast/RL covariate; available in advance |
| day_of_week | O | Campus-local weekday using declared numbering | integer 0-6, dimensionless | D from timestamp/time zone | Calendar forecast/RL covariate; available in advance |

Each required series must be complete and aligned for every episode. Weather is
not needed for a direct-power replay simulator. If direct available-power data
is absent, the selected conversion path must additionally provide the weather
inputs and plant/model parameters it needs; raw weather alone is insufficient.
Recorded delivered renewable output may hide curtailment and cannot automatically
be interpreted as available power.

## Static configuration and entity metadata

All values below are simulation parameters (S) unless documented asset or tariff
information is later obtained. No numeric asset ratings have been chosen.

| Variable | Level | Description | Unit | Intended use / history requirement |
|---|---|---|---|---|
| building_id, asset_id, bus_id | R | Stable entity keys; five buildings, four shared assets, one campus bus | text identifiers; no physical unit | Join series to entities and prevent duplicate asset accounting; no history |
| asset_type, connected_bus_id | R | Load/PV/wind/battery/grid role and bus membership | categorical text/identifier; no physical unit | Topology, simulation and future GNN; no history |
| time_step_hours | R | Interval duration, fixed initially at 1 | h | Convert power to energy; no history |
| campus_timezone | R | IANA time zone identifier; unresolved | text identifier; no physical unit | Local calendars and source-time conversion; no history |
| currency_code | R | One declared currency for all cost terms | text code; no physical unit | Tariff/reward comparability; no history |
| pv_capacity_kw | R | Maximum available AC PV power | kW | Validate/scale PV profile; no history |
| wind_capacity_kw | R | Maximum available AC wind power | kW | Validate/scale wind profile; no history |
| battery_capacity_kwh | R | Positive storage energy capacity | kWh | Battery state and SOC conversion; no history |
| battery_charge_limit_kw | R | Maximum AC charging power | kW | Dispatch bounds; no history |
| battery_discharge_limit_kw | R | Maximum AC discharge power | kW | Dispatch bounds; no history |
| battery_charge_efficiency | R | AC charging energy retained in storage, in (0, 1] | dimensionless ratio | Battery dynamics; no history |
| battery_discharge_efficiency | R | AC discharge energy per storage energy removed, in (0, 1] | dimensionless ratio | Battery dynamics; no history |
| soc_min, soc_max | R | Lower/upper fractions satisfying 0 <= min < max <= 1 | dimensionless fraction | State constraints; no history |
| initial_soc | R | Storage fraction at episode start, within SOC limits | dimensionless fraction | Initial state; parameter, not historical SOC requirement |
| grid_import_limit_kw | R | Maximum import power | kW | Grid bounds; feasible fallback must be checked |
| grid_export_limit_kw | R | Maximum export power; zero disables export | kW | Grid bounds; no history |
| episode_steps | R | Positive number of hourly transitions per episode | count, dimensionless | Replay window, RL evaluation and risk loss horizon |
| terminal_soc_rule | R | Explicit treatment: report free endpoint, enforce a target, or assign terminal value | categorical text; no physical unit | Fair episode/policy comparison; no history |
| terminal_soc_target | Conditional R | SOC target when target enforcement is selected | dimensionless fraction | Terminal state constraint |
| terminal_energy_value | Conditional R | Monetary value per stored energy when terminal valuation is selected | currency/kWh | Terminal cost adjustment |
| lookback_steps, forecast_horizon_steps | A: forecasting/RL | Past context length and future forecast length | count, dimensionless | Supervised windows and controller observation design; not observations |
| graph_edge_source, graph_edge_target | A: GNN | Explicit graph endpoint identifiers | text identifiers; no physical unit | Assumed topology, not measured electrical parameters |
| wind_measurement_height_m, wind_hub_height_m | Conditional A | Heights when a wind conversion model needs height correction | m | Weather-to-power conversion; metadata/parameters |
| pv_conversion_model, wind_conversion_model | Conditional A | Versioned conversion definition and its model-specific parameter schema | text/version; no physical unit | Needed only when deriving generation from weather; define units of all additional parameters before use |
| degradation_cost_per_kwh | O | Simplified cost per AC charge-plus-discharge throughput | currency/kWh | Later storage wear proxy; not a physical life model |

## Simulator, control, and evaluation series

These are D outputs, not independently required historical measurements.

| Variable | Level | Description | Unit | Intended use |
|---|---|---|---|---|
| campus_load_kw | R | Sum of the five building demands | kW | Balance, RL observation history and reporting |
| battery_energy_kwh | R | Stored energy at each boundary | kWh | Simulation state and constraint verification |
| battery_soc | R | Stored energy divided by capacity at each boundary | dimensionless fraction | RL state, storage constraints and reporting |
| battery_charge_kw, battery_discharge_kw | R | Actual nonnegative AC storage flows, mutually exclusive | kW | Action realization, energy balance and cost |
| pv_curtailed_kw, wind_curtailed_kw | R | Available generation withheld, each between zero and available power | kW | Feasible dispatch and renewable utilization |
| pv_used_kw, wind_used_kw | R | Available power minus curtailment for each plant | kW | Energy balance and reporting |
| grid_import_kw, grid_export_kw | R | Actual nonnegative grid flows, mutually exclusive | kW | Balance, capacity constraints and billing |
| step_cost | R | (import_price * grid_import_kw - export_price * grid_export_kw) * time_step_hours | currency | Baseline operating cost and evaluation |
| episode_cost | R | Sum of step costs over the declared episode, with any terminal adjustment explicitly recorded | currency | Policy comparisons and risk loss definition |
| balance_residual_kw | R | Total supply minus demand including charging | kW | Validate near-zero residual using declared numeric tolerance |
| requested_battery_power_kw | A: RL | Signed proposed AC action: positive discharge, negative charge | kW | Policy interface; distinguish requested from feasible realized action |
| requested_pv_curtailment_kw, requested_wind_curtailment_kw | A: RL | Nonnegative proposed withholding per renewable asset | kW | Policy interface; check/project feasibility before applying |
| reward | A: RL | Initially negative step_cost; record terminal contribution when applicable | currency | RL signal; any later normalization must be documented |
| action_adjusted | A: RL | Whether the feasibility layer changed an action | binary flag, dimensionless | Diagnose policy constraint handling |
| selected_policy_id | A: meta-learning | Policy chosen from a versioned policy catalog | categorical identifier; no physical unit | Policy-selection logging; no presumed ground-truth best label |
| unmet_load_kw | O | Explicit demand not served during a future infeasible/outage extension | kW | Reliability and risk; not allowed as an unreported baseline balance correction |

## Forecasts and risk inputs/outputs

All entries are A requirements for the named advanced stage. They are not
needed to run the initial replay simulator. No forecasts or risk scenarios exist
yet. Retain target IDs so a building forecast cannot be confused with a campus
aggregate. Timestamp fields use ISO 8601 UTC; identifiers have no physical unit.

| Variable | Description | Unit | Origin and intended use |
|---|---|---|---|
| forecast_issue_time, forecast_valid_time | When a prediction became available and the interval it predicts | UTC datetime; no physical unit | D/model metadata; causal forecast evaluation and controller availability |
| forecast_target_id | Building or renewable asset whose power is forecast | identifier; no physical unit | D metadata; entity alignment |
| predicted_power_kw | Forecast of demand or available generation for that target/time | kW | D/model; forecasting, RL observation and risk conditioning |
| forecast_error_kw | Realized target minus its matching issued forecast | kW | D/calculated from held-out or rolling-origin predictions; joint error/scenario calibration |
| scenario_id | Identifier for a joint future trajectory | identifier; no physical unit | S or D/model; links uncertainties across assets, buildings and hours |
| scenario_power_kw | Possible future target power indexed by scenario, target and time | kW | S or D/model; risk evaluation, conditional uncertainty and stress tests |
| scenario_probability | Nonnegative scenario weight summing to one over the scenario set | dimensionless probability | S or D/model; empirical expected loss/CVaR |
| scenario_loss | Episode operating cost under one scenario and policy | currency | D/simulator; common loss horizon for risk comparisons |
| cvar_alpha | Confidence level strictly between zero and one | dimensionless probability | S parameter; CVaR of the upper-cost tail |
| var_loss, cvar_loss | Loss threshold at alpha and mean upper-tail loss with appropriate quantile weighting | currency | D/optimization; report alpha, horizon, sample count and uncertainty |
| dro_ambiguity_type | Chosen distribution family/distance and reference distribution construction | text; no physical unit | S specification; required before claiming DRO robustness |
| dro_radius | Ambiguity-set size in the declared metric | dimensionless only for a dimensionless normalized metric | S/calibrated parameter; no portable default; specify normalization, distance and calibration first |

Preserve cross-building, generation-demand, and temporal dependence in joint
scenarios. Independently shuffled scalar errors cannot be claimed to represent
the joint distribution. Fit forecasts, scalers, scenario generators, policy
selection criteria and DRO calibration only using permitted training/calibration
periods. No final-test observations may enter controller inputs or calibration.
Price uncertainty is optional initially because tariffs are known schedules.

## Minimum dataset for a functioning simulator

A baseline replay simulator requires **no weather, historical battery SOC,
trained forecasts, RL policy, or risk model** if direct renewable power inputs
are supplied. Its minimum input bundle is:

1. **Building table:** N consecutive hourly timestamps, each with exactly one
   nonnegative `building_load_kw` for each B01-B05 (5N rows in long form).
2. **Renewable table:** the same N timestamps with one `pv_available_kw` and one
   `wind_available_kw` per interval, bounded by their configured ratings.
3. **Tariff table/configuration:** import and export prices for those intervals;
   explicit constant parameters are acceptable.
4. **Entity/topology and configuration:** all R static parameters above,
   including capacities, limits, efficiencies, initial SOC, time/currency
   conventions, episode duration and terminal SOC rule. Conditional terminal
   parameters are needed only for the chosen terminal rule.
5. **Provenance and quality metadata:** source origin and transformations for
   every series, with all simulated assumptions clearly marked.

N must cover the episode. A proposed initial software smoke-test window is
**168 hours (one week)**; this is a design choice, not a research sufficiency
claim. A shorter window can test individual transitions. All inputs may initially
be declared synthetic for software validation, but no synthetic data is generated
in this task. A separate feasible baseline controller is still needed in a later
implementation step; the dataset alone does not supply a control algorithm.

For forecasting research, obtain synchronized historical observations spanning
the relevant operating regimes, preferably a full seasonal cycle as a collection
goal rather than a guarantee of adequacy. Select actual coverage after source
inspection. Use chronological train/validation/test windows, require enough
lookback and horizon samples, and account for overlapping target windows at
split boundaries. A week-long synthetic test cannot support generalization,
real-world risk, or GNN performance claims. Risk and meta-learning stages need
additional calibrated scenarios and multiple evaluated policies/regimes.

## Validation and unresolved items

- Enforce unique `(timestamp_utc, building_id)` keys and complete entity coverage;
  renewable/tariff timestamps must be unique and align exactly to the hourly grid.
- Explicitly resolve source time zones and daylight-saving transitions. Aggregate
  finer power observations as duration-weighted means; sum interval energies and
  convert to average kW. Never sum kW readings as if they were kWh.
- Reject non-finite values, negative load/available generation, ratings violations,
  SOC/efficiency violations, invalid tariffs and insufficient interval coverage.
- Record missing observations and imputation; avoid future information when
  preparing controller inputs. Fit preprocessing only on training data.
- Confirm zero/near-zero balance residual, feasible grid fallback, SOC bounds,
  non-simultaneous flow rules and documented terminal treatment.
- Location, licenses, actual sources, ratings, tariffs, conversion details,
  forecast windows, numeric tolerances, objectives and risk calibration remain
  unresolved. They must be specified before their dependent experiments.

No dataset downloads, simulation code, new dependencies, or model training are
part of this documentation step.
