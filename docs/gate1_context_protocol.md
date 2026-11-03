# Gate 1B: preregistered cross-map strategy context study

**Protocol ID:** formicmind.gate1-context/v1  
**Registration date:** 2026-11-03, before any FOR-22 scheduled match is run  
**Scope:** open engineering diagnosis of the five frozen FOR-19 policies on six fixed, mirrored map configurations. This is not a population sample of maps, balance approval, strategy optimization, sealed evaluation, Gate 0 performance measurement, or a Gate 1 pass.

This protocol implements the context-sensitivity question in FOR-4. It uses FOR-19 policies and the real L0 engine; FOR-20 and FOR-21 findings are historical context only. No seed from FOR-19, FOR-20, or FOR-21 is reused. No sealed seed is used. The schedule, maps, thresholds, grouping rules, metrics, and stop conditions below are fixed before the primary matrix.

## 1. Fixed maps and invariants

Create six experimental SimConfig values from default_config() using dataclasses.replace and the existing typed config objects. Never mutate default_config(), the game rules, or any global object.

The two resource families have the following per-node values, in (start_food, capacity, regen, max_harvest) order. A and B are home nodes and have no resource.

| Resource node(s) | R0: center-rich | R1: local-rich candidate |
| --- | --- | --- |
| N1 and N4 | (60, 80, 8, 8) | (70, 80, 10, 10) |
| N2 and N5 | (40, 60, 8, 8) | (50, 60, 10, 10) |
| C | (80, 120, 12, 12) | (40, 120, 4, 4) |

R0 is exactly the current default resource placement. R1 is a preregistered candidate scenario, not an approved change to game economics. Both maps have 280 initial node food, 44 food/tick regeneration, 44 maximum food/tick harvest, and 400 total capacity. Corresponding A/B-side resources are identical under the existing mirror.

The three route profiles are applied identically to both resource families:

| Profile | Changed edges |
| --- | --- |
| T0 / default | None; use all default edges and lengths. |
| T1 / fast-center | N1-C and C-N4 have length 1; N2-C and C-N5 have length 2. |
| T2 / slow-center | N1-C and C-N4 have length 3; N2-C and C-N5 have length 4. |

Every other edge, including A-N1, A-N2, N4-B, N5-B, and N2-N5, retains its default length. The six IDs are R0-T0, R0-T1, R0-T2, R1-T0, R1-T1, R1-T2. For each Ti, R0-Ti and R1-Ti differ only in resource values. Each graph remains the mirrored, connected seven-node topology.

For every generated config, require all of the following before a match is scheduled:

* validate_config and check_observation_config pass; the normal reset/observation/action path remains valid.
* Node IDs, node names, home ownership, edge set, connectivity, mirror symmetry, and 7-node contract match the default topology.
* Resource mirror equality holds; source values are exactly the table above; totals equal 280 / 44 / 44 / 400.
* All unchanged CandidateBalance values and simulation settings equal default: colony start food, workers and soldiers; all HP, costs, upkeep, brood/hatch settings, eggs_per_lay=1, lay_interval_ticks=10, policy_interval_k=30, and T_max=3000.
* Existing normalizer limits hold: edge length <=4, source capacity <=120, regen <=12, and queen_hp=240.

Record each full canonical config and its config_hash. Do not modify ObservationV2 or its normalizers to accommodate these maps.

## 2. Fixed tournament

Use the unchanged FOR-19 policies and descriptors: random:v1, turtle:v1, rush:v1, balanced:v1, raider:v1. Use the same ObservationV2, action mask, decoder, validator, policy interval, and role-keyed RNG stream semantics for every policy on every map. Do not tune or select a winner.

Run a complete round robin without self-play: all 10 unordered policy pairs from that fixed order, with both A/B seat assignments. Use exactly the six open seeds 20261221, 20261222, 20261223, 20261224, 20261225, and 20261226. These are distinct from all prior schedules. Keep each policy identity and its RNG stream attached to the policy when seats are swapped.

The schedule is the Cartesian product of:

* 6 maps;
* 10 unordered policy pairs;
* 6 seeds;
* 2 seat assignments.

Expected size is exactly 720 real matches. Run each with reset(config, seed) followed by full step_tick calls until the engine sets a terminal EpisodeOutcome. The hard ceiling is 2,160,000 step_tick calls (720 x T_max). Do not force an outcome at a harness boundary. If T_max is reached without an engine terminal, mark the row censored/incomplete, not a draw. A timeout is a draw only when the engine records its terminal timeout outcome.

Record the first technical failure and stop the matrix. Do not replace a seed, skip a failed scheduled row, or continue with a modified schedule. Save an invalid/incomplete attempt and diagnostic, and make all affected labels INCONCLUSIVE. Re-running the still-unviewed fixed schedule after a technical repair is allowed; document the invalid attempt and keep its outputs separate. Never overwrite an existing output directory.

## 3. Primary estimands and exact aggregation

For each fixed policy pair P-Q, terrain profile Ti, seed, and resource family, the paired seat score for P is the arithmetic mean of P's engine score in the two seat assignments. Engine score is 1 for a terminal win, 0.5 for an engine terminal draw, and 0 for a terminal loss. A censored or missing outcome has no score.

The primary per-profile context contrast is:

    delta(P,Q,Ti,block) = mean_seats(score_P on R1-Ti) - mean_seats(score_P on R0-Ti)

One block consists of the same matchup, Ti, seed, both resource families, and both seat assignments. Average seat assignments inside each resource-family score before computing delta; the two seats are one paired block, not two independent outcomes. Scores and contrasts are stored as exact rational/integer values, not floating-point rounded values.

For each P-Q and each Ti, report all six scheduled seed-block deltas and a descriptive mean over unique paired-trajectory blocks. Give each unique block equal weight. The equally weighted three-profile contrast for P-Q is the arithmetic mean of the three Ti means; no profile or opponent is selected by observed outcome. Preserve every scheduled row in raw data even when its trajectory duplicates another seed.

For rank reversal, use the signed head-to-head margin P minus Q in each family. First average seat scores within a block, then average unique paired-trajectory blocks within each Ti, then give the three Ti means equal weight. Report R0 and R1 margins separately. A pairwise rank reversal candidate requires one family's margin to be at least +0.10 and the other family's margin to be at most -0.10 (or vice versa). Equality at exactly 0.10 qualifies. This is descriptive for the six fixed maps only.

Also publish the 6-map by 10-matchup score matrix and descriptive policy scores by family. For an overall family score, average each policy's four opponent means equally, and average the three terrain-profile means equally. Do not calculate an unweighted global win rate across arbitrary rows or describe this fixed-map score as a population ranking.

## 4. Independent trajectory accounting

The schedule rows are not the independent sample size. Seeds are schedule labels, not independent replicates by themselves. Mirrored A/B assignments are paired seats and are never counted as two independent outcomes. Repeated deterministic trajectories across different seeds count once.

For every match, store the ordinary world-specific state_hash and checkpoint_hash at each tick in its integrity trace. Separately compute the policy-identity-relative trajectory digest used for independence accounting from the complete ordered full-step trace: canonical state and per-tick canonical event data, plus a normalized checkpoint digest. Normalize side, node, and persistent entity IDs into the first-listed policy's local frame. The normalized checkpoint digest covers that normalized game state, policy-window counters, and terminal outcome; it omits root_seed and unused RNG streams. Include the actual pre/post RNG stream state at each policy decision for random:v1 only, attached to that policy's role-keyed stream. Do not include row ID, seat label, or wall-clock fields. The raw world-specific hashes remain in every row for strict verification. This preserves genuinely different random trajectories while grouping repeated deterministic paths.

A paired-trajectory block digest is a SHA-256 of the ordered composite of all four policy-relative trajectories in that block: R0 first-seat, R0 swapped-seat, R1 first-seat, R1 swapped-seat. Include policy IDs, Ti, and the resource-family identities. The mirrored seats remain together inside this one digest. Do not collapse across different Ti or policy pairs. When paired-block digests repeat across seeds, keep all rows visible but count the digest once in unique-block summaries and labels.

Report exact counts at each level: scheduled matches, paired seed blocks, distinct per-match trajectories, distinct paired-trajectory blocks, distinct random-stream trajectories, and duplicate seed blocks by policy pair, Ti, and family. Do not compute a frequentist confidence interval or p-value: the maps are a fixed handpicked set, and duplicate deterministic runs do not create uncertainty.

For a sensitivity or rank-reversal label, require at least two distinct paired-trajectory block digests for every one of the three Ti profiles in that policy pair, complete terminal outcomes for every scheduled row involved, and no integrity failure. If a policy pair misses any requirement, its label is INCONCLUSIVE, even if the descriptive point estimate crosses an engineering threshold. Report exact estimates and coverage regardless.

## 5. Predeclared thresholds and interpretation

The engineering MDE for context sensitivity is an absolute paired score contrast of 0.15 on the 0-to-1 scale for one fixed policy pair. Label CONTEXT_SENSITIVITY_CANDIDATE only when:

1. the absolute equally weighted three-profile contrast is >=0.15; and
2. its direction is shared by at least two of the three Ti-specific means.

Direction uses the sign of the three-profile contrast. A Ti mean exactly zero is not directionally concordant. If coverage is adequate and either condition fails, label NO_CONTEXT_SIGNAL_IN_SCHEDULE. If coverage is inadequate or any involved row is incomplete, label INCONCLUSIVE.

Label PAIRWISE_RANK_REVERSAL_CANDIDATE only when the pair passes the independent-block and completeness requirements and the family margins meet the +/-0.10 reversal rule in section 3. Otherwise label NO_CONTEXT_SIGNAL_IN_SCHEDULE when coverage is adequate, or INCONCLUSIVE when it is not. Report both candidate types separately. These labels are engineering diagnostics, not p-values, not proof of generalization, and not a Gate pass.

At study level, report all pair labels. The study is INCONCLUSIVE if the fixed schedule is incomplete or every policy pair is inconclusive. If at least one pair is evaluable, summarize evaluable candidate/no-signal labels and list every inconclusive pair; do not hide partial coverage. No conclusion is made about the value of feedback or adaptation; that requires a separate open-loop/closed-loop test.

## 6. Per-match metrics and degeneracy probes

For every match, report policy-identity-keyed scores, winner, engine terminal reason, terminal tick/completion, ticks executed, final state/checkpoint hashes, per-tick trace and event digests, map/config/policy/seed/seat provenance, and terminal/censoring status.

Capture state metrics at the boundaries after 100, 300, and 1000 completed ticks when the game is still available, plus the terminal state. At each available endpoint report for both policies: food, adult workers and soldiers (stationed, in transit, and total), brood by caste, queen HP, starvation debt, upkeep remainder, node food in the policy's local frame, adults by caste/local node, and transit destinations. If a match terminates before a requested live horizon, record that horizon as null with the terminal tick/reason; do not substitute zero or a later value.

The repository's canonical JSON encoder does not admit JSON null. In canonical artifacts encode an unavailable metric as the exact string not_measured and include its reason wherever a horizon or terminal status provides one. In-memory calculations may use a missing value; it must never be converted to numeric zero.

Accumulate compact event-derived metrics online: lay attempts/successes/food rejections, egg food spent, harvest, upkeep due/paid/missing, starvation deaths, combat events and damage, queen damage, order count, retargets, and distinct forage vectors/army choices. For each policy, report lay_none fraction, home-only forage fraction, garrison/defend/raid fractions, army-choice diversity, forage-vector diversity, and event counts. Use event/state data only; unavailable fields are null/not_measured, not zero. Store event digests and counts, not full per-tick event lists.

Predeclared review flags are diagnostic counts, never balance-change triggers:

* timeout concentration: within a matchup x map group of 12 scheduled matches, flag if at least 3/12 (25%) have an engine timeout terminal.
* extended zero-combat matches: a match qualifies if at least 1000 ticks completed and it has zero CombatEvent records. Within a matchup x map group, flag if at least 3/12 scheduled rows qualify. The group denominator remains all 12 scheduled rows.
* food stockpile at unresolved timeout: a timeout match qualifies if both queens are alive, each colony has at least one adult soldier stationed or in transit, and each colony has at least 300 stored food at terminal. Flag a matchup x map group if at least 3/12 scheduled matches qualify.
* repetitive policy orders: a policy-match qualifies only with at least 10 policy decisions and either one forage vector on >=90% of decisions, one army_choice on >=90%, or lay_none on >=75%. Report exact numerator/denominator; flag a matchup x map group if >=3/12 matches qualify for either policy.

These probes do not impose a win-rate cap or assert that a strategy is dominant. Compare Turtle-Raider timeout patterns with FOR-19 and FOR-20 economic red flags as context only; do not claim causal explanation from this study.

## 7. Provenance, artifacts, and strict verification

The first commit on the FOR-22 branch contains only this protocol. Before running the matrix, commit the harness and its tests. Then run the fixed matrix once to a new output directory.

Harness module and commands:

    PYTHONPATH=src python -m formicmind.gate1_context run --out results/gate1_context
    PYTHONPATH=src python -m formicmind.gate1_context verify --input results/gate1_context

Required artifacts:

* docs/gate1_context_protocol.md — this frozen preregistration.
* src/formicmind/gate1_context.py — experiment-only harness and run/verify CLI.
* results/gate1_context/manifest.json — protocol SHA-256, base origin/main SHA, source commit SHA, recursive Python source fingerprint, full six configs and hashes, fixed seed/policy/seat schedule, version descriptors, command/runtime, budget and expected row count.
* results/gate1_context/matches.jsonl — exactly one compact row for each scheduled match in stable schedule order.
* results/gate1_context/summary.json — exact aggregation, pair/map score matrices, unique/duplicate trajectory counts, labels, event metrics, review flags, errors, and file hashes.
* docs/gate1_context_findings.md — results, limitations, recommendation, and reproduction instructions.

The recursive source fingerprint is SHA-256 over sorted relative paths and SHA-256 contents of every Python file under src/formicmind/. Manifest provenance includes policy parameter digests and stream IDs, all relevant rules/balance/observation/action/decoder/RNG/serialization versions, config hashes, protocol hash, Python version, command, source commit, and expected 720 rows.

Strict verify must reject missing, extra, reordered, or changed rows; artifact hash mismatches; changed protocol/config/policy/source provenance; invalid maps; incomplete engine outcomes; and any difference in re-executed per-match actions, event/trace/state/checkpoint hashes, terminal outcomes, endpoint metrics, or aggregates. It must replay all 720 scheduled matches through reset and the unchanged step_tick and recompute the summary; merely re-aggregating saved rows is not verify. A later documentation/results-only commit may verify when the protocol SHA and recursive Python source fingerprint still match. A changed Python source fingerprint is incompatible and must fail strict verify.

Before commit/run, confirm that results/gate1_context does not exist. Hash results after run and do not edit raw data or summary by hand. A correction to implementation requires a new run under the same still-unviewed schedule and frozen protocol, with the invalid attempt retained separately. Any exploratory work requires a separately labeled supplement/new protocol.

## 8. Boundaries and required handoff

Do not change default topology/balance, physics, tick/order/combat/terminal code, baselines, ObservationV2, ActionV2, decoder, golden tests, or Replay v1. Do not run ML/RL, tune policies, alter map/seed/threshold/budget after seeing results, or declare Gate 0/1 passed. Findings apply only to the preregistered FOR-19 policies and six fixed maps. Gate 0 Windows single-core performance remains NOT MEASURED.

After the matrix, run strict verify, the full Python suite, viewer production build and Vitest, and Windows GitHub Actions. Publish a report and limitations in FOR-22 and open a PR for review. Do not merge.
