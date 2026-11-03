# Gate 1C feedback ablation protocol (frozen before harness and matrix)

**Protocol ID:** `formicmind.gate1-feedback/v1`  
**Issue:** FOR-23  
**Base:** `origin/main` at `af4d6ff5364e29add002aab504d2609b3d797fb8` (merge of PR #13 / FOR-22)  
**Branch:** `for-23-gate1-feedback-ablation`

This is an open diagnostic ablation of the unchanged FOR-19 policies. It asks whether live `ObservationV2` feedback improves the same fixed policy beyond its initial observation, current time, and the shared action-legality path. It does not search for an optimal open-loop plan, establish that every closed-loop controller beats every open-loop program, or pass Gate 1.

**Pre-run arithmetic clarification:** the Linear description states that the six listed factors total 864, but their product is 1,728. Before implementing the harness or running the matrix, the user selected the complete factorial schedule as authoritative. Keep all listed maps, policies, six seeds, seat assignments, and modes; the preregistered count and call ceiling below use the correct product.

## Frozen engine, maps, policies, and budget

Use the real `reset → step_tick` engine, with no changes to physics, terminal rules, default balance, action/observation contracts, decoder, or existing policy implementations and parameters. Reuse the six canonical FOR-22 configurations; build them through the unchanged FOR-22 config factory and assert their canonical hashes:

| Map | Config SHA-256 |
| --- | --- |
| `R0-T0` | `3f6aaa36a0d3d15888fa42f142df1151376794e62b553ab8e5f7659c9794145d` |
| `R0-T1` | `78e9c6979a2fb3973ae9a275f79ad5b62cb4b6ed3893a8f8469e4cc599e5fb34` |
| `R0-T2` | `b9e43b2aec34f303f371a9f9e5a2bcd8b5c5d894e608a3ca9e71386339f632f1` |
| `R1-T0` | `407bb3907099533adfe10b7c300a480b45b5d604dc019693b1ac99f47502303f` |
| `R1-T1` | `3bf8dabf6011ed88275c3fcfec3875c4da46ef849e7da7b0ac17bb2cd0e9e605` |
| `R1-T2` | `1b6f1f999246711919fb4dee9238e5fed31c5baac0dca779ea76c47afe167ceb` |

Targets are exactly `balanced:v1`, `raider:v1`, `turtle:v1`, and `rush:v1`. Opponents are exactly `random:v1`, `balanced:v1`, and `raider:v1`; random is never a target. Parameter digests are the already-published FOR-22 baseline digests and must match at run and verify:

| Policy | Parameters SHA-256 |
| --- | --- |
| `random:v1` | `254037fbbe5f089fc1f909405ebcc3567573682553d3715d9bab8cf292b07f2c` |
| `turtle:v1` | `6c9fa1c676a553294c26ffb4f3372953f93ecec73b5d8669f99a451a05e0ee4e` |
| `rush:v1` | `cfea106a71257d8ebe79cfa6ac090b1927fd3010ca905516e2e41512460ea496` |
| `balanced:v1` | `03c6760da2b2fc42108133c8b81a91b70e3c086327fbabedf9d3f30eb7dcdd0f` |
| `raider:v1` | `6d4a23b0071c6ee308b8149850debe488e83d79c7fda50f19ee5053310c39b19` |

Open evaluation seeds are `20270111` through `20270116` inclusive. Every match uses both seat assignments. No FOR-19/20/21/22 seeds, sealed seeds, training, calibration, tuning, or result-based policy selection are allowed. Each episode runs to a real engine `EpisodeOutcome`; an engine timeout is a draw, while a harness cutoff without an outcome is censored/incomplete and never a draw.

The complete schedule is `6 maps × 4 targets × 3 opponents × 6 seeds × 2 seat assignments × 2 target modes = 1,728` full matches, with a hard ceiling of `5,184,000` `step_tick` calls (`1,728 × T_max 3000`). A technical error, invalid action, control-check failure, observation leak, or nondeterministic replay invalidates the study; record the failure and stop without substituting seeds or changing the schedule.

## Matched controller modes and observation isolation

Each target uses the exact same FOR-19 policy code, version, parameters, 17-logit output, and 30-tick decision cadence in both arms. No target is tuned; the setting/tuning budget is exactly zero.

1. **Closed-loop:** use the existing `PolicyController` and `decide` path. At each policy tick the baseline receives the current, side-local, 223-float `ObservationV2` and its current `ActionMaskV2`; the shared `decode_action_logits` and `validate_action` handle the action.
2. **Frozen-observation open-loop:** at `t=0`, freeze that controller instance's own full `ObservationV2` and initial `ActionMaskV2`. On later policy ticks, construct its policy input from those frozen values, replacing only the two float32 entries in the `time` block with the corresponding two values from the current real observation. The other 221 float32 entries remain bit-for-bit those from `t=0`. The baseline logits generator receives this frozen input and the frozen initial mask; it is never passed the current observation or live mask. The generated logits are passed to the same `decode_action_logits` with the current real action mask, then the selected action is passed to the same `validate_action` against the current real observation and mask. The tick engine performs its normal validation again.

The controller-local frozen snapshot is created anew for each episode and side. It cannot read `SimState`, the opponent's private state or policy, root seed, role/side index, later ticks, or other matches. The only live inputs allowed in this arm are current `time` for the policy input and the current mask through the shared decoder/validator. This is **mask-only legality adaptation**. It does not test every possible open-loop program.

The opponent is the same unchanged FOR-19 policy in both arms, with its own policy RNG stream. Self-play occurs for `balanced` vs `balanced` and `raider` vs `raider`; target and opponent then receive distinct role-bound replicate streams (`target` replicate 0, opponent replicate 1), and those streams remain attached to identities after swapping seats. For other pairs each policy uses its normal replicate 0 stream. Stream IDs are the same in closed/frozen matched runs and follow policy identity rather than seat.

### Mandatory time-zero control check

For each map/target/opponent/seed/seat pair, before any new observation can affect the target, assert equality between closed and frozen target policy-input bytes, initial mask, logits, validated `ActionV2`, and post-first-`step_tick` `checkpoint_hash` (full physical state plus all RNG streams and run state). Any mismatch invalidates the run. This also checks that the opponent stream and the rest of the engine began identically.

## Units, primary estimand, and decision rules

The scheduled matched block is `(map, target, opponent, seed)` and contains four episodes: closed/frozen with target on A and closed/frozen with target on B. Read each score from the engine outcome and convert target-relative win/draw/loss to integer `score_x2 ∈ {2,1,0}`. First average the two seat scores within each mode; the block's paired delta is exactly:

`Δ_block = (closed_A_x2 + closed_B_x2 - frozen_A_x2 - frozen_B_x2) / 4`.

Before calculating means, collapse seat assignments into this one block. Deduplicate repeated **joint matched trajectories**, not just repeated scores: hash the four per-episode, policy-relative normalized decision traces in a fixed arm order. A policy trace includes each decision tick, policy-visible observation bytes, the mask supplied to the logits generator, the current mask actually used by the shared decoder, logits, and validated action, expressed in the policy's local frame. In frozen mode the first mask stays frozen while the decoder mask is live. The composite digest excludes root seed labels; random opponents remain distinct only when their actually used decision traces differ. Repeated deterministic seeds and mirrored seats therefore never inflate independent `N`.

For each `(target, map, opponent)` cell, compute the exact rational mean of `Δ_block` over unique complete paired blocks. The target-level estimate gives equal weight to the 18 fixed map × opponent cells, with unique blocks equally weighted only within a cell. Also report every cell, each of six map means (equal weight over the three opponent cells), each opponent-family mean, scheduled blocks, unique blocks, and duplicates. Do not calculate 95% confidence intervals or p-values from these fixed maps, mirrored seats, or duplicated trajectories.

Predeclared engineering labels, not significance claims:

* Coverage for a target is adequate only if at least two distinct paired trajectory blocks remain **within the same target × map × opponent cell** for cells spanning at least 3 of 6 maps and at least 2 of the 3 opponent families. Different opponent scenarios cannot count as repeat trajectories within one cell. Any missing/censored scheduled arm or insufficient coverage makes the target `INCONCLUSIVE`.
* `MDE = 0.10` on the target-relative 0..1 score scale.
* `FEEDBACK_BENEFIT_CANDIDATE`: adequate coverage, target-level `Δ ≥ +0.10`, and positive mean delta on at least 4 of 6 maps (zero is not positive).
* `FEEDBACK_HARM_CANDIDATE`: adequate coverage, target-level `Δ ≤ −0.10`, and negative mean delta on at least 4 of 6 maps (zero is not negative).
* Adequate coverage with neither candidate label is `NO_CLEAR_FEEDBACK_SIGNAL_IN_SCHEDULE`.

Maps are six fixed handpicked scenarios, not a random map sample. R1 is diagnostic, not approved balance. A positive candidate means only that live feedback helped this unchanged baseline pool on this fixed schedule, beyond the common live action-mask legality path. It does not show that closed-loop beats an optimized open-loop program, that learning will help, or that Gate 1 passed.

## Predeclared diagnostics

For every match, record the target-relative engine outcome, terminal reason/tick, completed step count, final canonical state/checkpoint hashes, config hash, two controller stream IDs, event counts, and integrity/decision-trace hashes. Keep event handling streaming; do not retain per-tick events or replay states.

At completed-state horizons `t=100`, `300`, and `1000`, or at an earlier terminal state, record for each side food, queen HP, adult workers/soldiers, brood by caste, and target-role cumulative harvest, starvation deaths, combat events, order counts/diversity (`lay`, forage vector, army choice). A horizon not reached is explicitly `not_measured` with terminal/censoring reason, not zero. Report engine timeout draws separately.

Across the paired closed/frozen target traces, report counts of shared decision ticks with different logits, validated actions, and live 223-float observations; the first decision tick where the target's full live observation differs; and the first physics tick after which paired physical `state_hash` values differ. The observation divergence resolution is policy cadence (30 ticks); physical-state comparison is tick-by-tick until first difference. For frozen decisions, report when live mask differs from initial mask and when decoding the same frozen logits with the live rather than frozen mask changes the legal action (`mask_only_legality_changes`). Those diagnostics never feed back into policy computation.

## Artifacts, provenance, and strict replay

* This file is the first, protocol-only commit. Its SHA-256 is embedded in the manifest.
* `src/formicmind/gate1_feedback.py`: run command `PYTHONPATH=src python -m formicmind.gate1_feedback run --out results/gate1_feedback`; strict verification command `PYTHONPATH=src python -m formicmind.gate1_feedback verify --input results/gate1_feedback`.
* `results/gate1_feedback/manifest.json`: protocol/source/base commit hashes and recursive Python source fingerprint; all six canonical config hashes; exact baseline IDs/parameter hashes and stream rules; seed/seat/mode schedule; versions, command, expected count and call ceiling; artifact hashes and run status.
* `results/gate1_feedback/matches.jsonl`: exactly one row per scheduled episode in canonical schedule order, including decision-trace hashes/hashes by decision, control-check values, divergence/legality diagnostics, horizon and terminal metrics.
* `results/gate1_feedback/summary.json`: matched four-arm blocks, exact rational cell/target/map/opponent deltas, scheduled/unique/duplicate counts, predeclared labels and diagnostics.
* `docs/gate1_feedback_findings.md`: exact findings, negative results, limitations, commands, and interpretation.

`verify` must check protocol/config/policy/schedule/provenance, replay all 1,728 rows through the full `step_tick` engine, and byte-compare recomputed per-match JSONL and summary plus artifact hashes. It fails on any missing, extra, reordered, censored, tampered, invalid, or non-deterministic row. Strict verification is offline. Documentation-only edits after the run do not alter the source fingerprint; any Python source change after the run requires a new run and artifacts.

Required independent tests cover: exact config hashes and 1,728-row schedule; frozen input changes only the two time floats; arbitrarily altered live values cannot affect frozen logits; frozen policy receives initial mask while shared decoding/validation use live mask; mask-only changes are counted; identical time-zero inputs/logits/actions and checkpoint; self-play stream distinctness and identity retention across seat swap; all engine actions validated; matched seat collapse and trajectory deduplication; censored outcomes and missing horizons remain unknown; exact `0.10`, map-consistency, zero-map, and insufficient-coverage label boundaries; deterministic rerun and tamper rejection. Run full Python tests, frontend build/Vitest, strict verify, and Windows CI at the PR HEAD.

**Out of scope:** changes to default config/balance, FOR-19 policies or parameters, tick/physics/combat/terminal logic, `ObservationV2`, `ActionV2`, the shared decoder/validator, Replay v1, golden fixtures, or any gameplay contract; model training/tuning; sealed evaluation; claims that Gate 0 or all of Gate 1 passed.
