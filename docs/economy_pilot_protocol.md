# Economic feasibility pilot protocol (FOR-20)

**Status:** preregistered engineering pilot; frozen before the main matrix. This is an open diagnostic study, not train/validation/sealed testing and not a statistical estimate of bot strength or balance. It cannot pass Gate 0, Gate 1, or Gate 2.

**Baseline:** `origin/main` at `b7ab8d743b961e6d1a18b9e63951acdcb78537f3` (2026-11-03). The candidate rules are those already present there. This experiment does not edit `default_balance`, physics, `step_tick`, combat, terminal rules, `ActionV2`, `ObservationV2`, the decoder, Replay v1, version constants, or the FOR-19 seed schedule/manifest.

## Questions

1. Under the current candidate `eggs_per_lay=1`, which predeclared reproduction and forage regimes retain workers and food, and where do starvation, failed lays, or a worker plateau appear?
2. At `eggs_per_lay=2` and `4`, how do the same regimes change adult growth, starvation, source stocks, and missed lays?
3. Does the isolated economic trajectory show starvation or a worker plateau consistent with the FOR-19 `turtle–raider` timeout? This is only a mechanism comparison. It cannot establish a cause for a match outcome.
4. Does this candidate economy provide enough observable trade-off for the next Gate 1 design? This pilot may describe candidate signals and risks; it does not make a Gate decision.

## Fixed experiment matrix

The main matrix has **144 runs**: `3 eggs_per_lay × 4 lay regimes × 2 forage regimes × 2 resource profiles × 3 seeds`. Each run uses two mirrored colonies in one shared L0 state. Both sides receive the same local-frame script.

| Factor | Fixed values |
| --- | --- |
| `eggs_per_lay` | `1`, `2`, `4` |
| `lay_interval_ticks` | existing candidate value `10` |
| Lay regime | `no-lay`; `worker-first` (always worker); `soldier-focused` (always soldier); `alternating` (worker for ticks 0–119, soldier for 120–239, repeating) |
| Forage regime | `single-local`: all workers to local category N1; `split-local`: N1/N2 quantized shares `(0,32768,32767,0,0,0)` (50/50 to integer precision) |
| Resource profile | `default`: exact `default_config()` source records; `scarcity`: halve `start_food` and `regen` with integer floor for every resource record, leave `capacity` and `max_harvest` unchanged |
| Open seeds | `20261111`, `20261112`, `20261113` |
| Observation points | state after completed ticks `t=100`, `300`, `1000`; also each completed 100-tick boundary through `1000` for plateau detection |
| Other settings | existing topology, starting populations/food, caste costs, upkeep, hatch delays, `policy_interval_k=30`, `T_max=3000`, and all other `default_config()` values |

The resource profile and `eggs_per_lay` are immutable per-run `SimConfig` copies. Every full config is serialized and its `config_hash` is recorded. The scarcity profile is an explicit experimental configuration; it does not change the default or claim a new approved balance. No other values will be varied. No threshold, parameter, or strategy will be selected after seeing the outcomes.

### Scripted policies and isolation

At policy ticks only (`0, 30, 60, ...`), each side receives one `ActionV2`: the lay value from the table above, its local forage shares, `army_choice=garrison_home`, and `avoid_center=false`. During `alternating`, `worker-first` applies when `(tick // 120) % 2 == 0`, otherwise `soldier-focused`. These scripts use the existing `ScriptedController → validate_action → step_tick` path; no policy or simulation formula is added. The scripts use only own-half forage categories: A targets N1/N2 and B's mirrored local frame targets N4/N5. No center or enemy-home order is issued. Soldiers stay at home.

All runs use the ordinary full `reset → step_tick` path and all normal phases. A run is valid as an isolated run only while (a) every A unit stays at nodes A/N1/N2 or on edges whose endpoints are both in that set, (b) every B unit stays at B/N4/N5 or on edges whose endpoints are both in that set, (c) no two colonies occupy the same node or traverse the same edge, (d) queen HP does not change, and (e) no combat, wound, combat death, or queen-damage event occurs. The runner checks these conditions after every tick and records the result. A terminal caused by economic collapse is recorded as an early terminal; it is not an isolation failure.

If the default topology fails the isolation condition, stop the matrix immediately without inspecting economic outcomes. The predeclared fallback is a symmetric isolation topology retaining only edges A–N1, A–N2, B–N4, and B–N5; all four edges incident to C and the N2–N5 cross-edge are removed. Node/resource records and every other setting remain as declared above. Record this as `local-barrier`, include its config hash, and restart the **entire** 144-run matrix on that topology. These results then describe the explicit isolated-map variant, not the unmodified default map. Do not selectively keep runs from the two topologies.

`eggs_per_lay=1` is the existing candidate; `2` and `4` are experimental candidates. None changes `rules_version` or `balance_version`. This pilot does not run FOR-19 bot-vs-bot matchups or use any closed/sealed seeds.

## Measurements and definitions

Record per run, per side, at each available observation point:

* adult workers and soldiers, brood workers and soldiers, warehouse food, accessible local source stocks, and queen HP;
* cumulative harvested food, upkeep paid, missing upkeep, starvation deaths by caste, egg spend, successful batches/eggs, laid eggs and hatched adults by caste;
* current and peak starvation debt, plus **debt-ticks** (sum of each end-of-tick debt value through the point);
* current population change from the preceding 100-tick boundary, the first plateau candidate time, run duration, terminal reason, isolation checks, and data-quality status;
* the full named experimental factors, exact config and `config_hash`, seed, policy/forage schedule, protocol-file SHA-256, per-point `state_hash` and `checkpoint_hash`, and an incremental hash over the complete ordered event stream.

Food ratios are descriptive: `egg_spend / harvested_food` and `egg_spend / (starting_warehouse_food + harvested_food)`. Return `null` when a denominator is zero. Food is fungible in the existing engine, so neither ratio asserts which food unit paid for an egg.

For every tick and side, classify laying opportunity from the scheduled/current `lay_order`, `next_lay_tick`, and queen state. Count `lay_none` when the active order is none; count `cooldown` when an active lay order is before `next_lay_tick`; otherwise require exactly one `LayEvent`, and count it as `laid=true` or food-rejected `laid=false`. A missing event on an eligible request is a data-quality error. Do not infer a missed lay from `None`, and do not treat the expected absence of an event during no-lay/cooldown as a failure. Report `food-rejected / eligible attempts`, or `null` if there were no eligible attempts.

The plateau candidate is the first 100-tick boundary ending three consecutive full 100-tick windows in which each window's absolute adult-worker change is at most `max(1, ceil(5% × workers at the start of that window))`. No plateau is assigned when a boundary/checkpoint is missing or workers are zero. This is a descriptive population plateau, not evidence of stable or healthy equilibrium; interpret it alongside food and starvation metrics.

`t=100/300/1000` means the state after exactly that many complete ticks. If a full-mode terminal occurs sooner, retain the terminal state and its reason, mark later observation points unavailable (`null` with a missing reason), and never fill them with zero. Preserve all earlier checkpoints. Capture checkpoint hashes only at available points and at the final early-terminal state.

### Engineering screens (not statistical pass/fail gates)

For each unique trajectory, report these predeclared labels:

* `survives_horizon`: the colony reaches state `t=1000` with its queen alive and at least one adult worker. Otherwise report the exact earlier terminal or missing condition.
* `reproductive_loop`: for a lay-active regime, at least one successful batch and at least one matching `HatchEvent` occurred by `t=1000`; for `alternating`, report worker and soldier loops separately. This is reported separately from survival.
* `engineering_viability_candidate`: a lay-active run survives to `t=1000`, has at least one adult worker, has zero starvation debt and no starvation death during ticks 900–999, has positive warehouse-plus-local-source food at `t=1000`, and has completed the relevant reproduction loop (both caste loops for alternating). The `no-lay` control is classified only by `survives_horizon`; it is not expected to reproduce. This label is only a screen for further engineering work and is not a balance approval.

Flag these red conditions without tuning: any invalid config/action/hash/invariant; incomplete 144-row matrix; isolation failure; any data-quality error; at least 25% food-rejected eligible lay attempts when there are at least 3 such attempts; any starvation death in ticks 900–999 or nonzero debt at `t=1000`; a ≥50% adult-worker reduction from reset by `t=1000`; or all local source stocks and warehouse food at zero with no harvest during the preceding 100 ticks. Report every flag with its run key and supporting counts. A failed isolation/invariant/hash invalidates that whole topology matrix; a normal economic early-terminal does not.

The reported saturated/plateau state is always paired with worker count, accessible source stock, warehouse food, starvation debt, and deaths. A plateau alone is not a red or green viability result.

## Replication, dependence, and analysis

The three seeds are open engineering seeds, not independent replicates for statistical inference. The policies and L0 economy are deterministic; if two seeds produce the same ordered event hash and checkpoint state-hash trajectory, count and display one unique state trajectory and report how many scheduled seeds duplicated it. Do not use the number of duplicate seed rows as sample size, calculate confidence intervals, or claim a win rate, causal effect, or strength. Mirrored sides are coupled observations inside one shared engine and share nodes/rules; they are not two independent colonies. Side A/B rows must not be counted as independent samples.

Compare `eggs_per_lay` and regimes descriptively within the same fixed seed/config schedule. List all 144 raw run rows, all checkpoint rows, and deterministic aggregates. Report exact unique-trajectory counts and missingness. Any extra experiment after viewing the matrix must go in an explicitly labelled exploratory supplement with new raw outputs; it may not change this protocol, its thresholds, or the primary summary.

## Stop conditions and validity

Stop a run/matrix on a raised controller/engine error, invalid action, failed invariant, unexpected phase mode, hash mismatch on deterministic replay, missing eligible `LayEvent`, malformed config, incomplete output, or isolation failure. Diagnose and correct implementation errors without changing this frozen protocol. If a correction changes any specified input, metric, or decision rule, the affected primary results are invalid and require a separately versioned protocol before rerunning. Normal early terminal outcomes are data and do not stop other matrix rows.

The matrix is complete only with 144 distinct `(eggs_per_lay, lay_regime, forage_regime, environment_profile, seed)` keys, all expected observation rows or explicit early-terminal missingness, both side records, valid isolation, and deterministic re-run agreement. `step_tick` must return full, implemented physics on every executed tick; `partial_economy=True` is not used.

## Reproduction commands and artifacts

The harness will expose one locked primary invocation after this file is committed:

```bash
python -m formicmind.economy_pilot run --out results/economy_pilot
python -m formicmind.economy_pilot verify --input results/economy_pilot
```

The run manifest records the full generated configs, hashes, factor ordering, seed schedule, protocol-file digest, source Git SHA, command, and mode (default topology or `local-barrier`). `runs.json` contains machine-readable per-run/per-side raw summaries and checkpoints. `summary.json` contains deterministic aggregates and red-flag/viability labels. `docs/economy_pilot_findings.md` reports the results, environment, command, unique trajectories, caveats, and a recommendation with no balance changes. No Replay v1 files are emitted and event lists are consumed incrementally rather than retained for the batch.

