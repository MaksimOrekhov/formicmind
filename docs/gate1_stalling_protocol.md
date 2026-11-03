# Gate 1D: preregistered anti-stalling audit and one-shot timeout probes

**Protocol ID:** `formicmind.gate1-stalling/v1`  
**Registered:** 2026-11-03, before implementation or execution of the FOR-24 schedule  
**Base:** `9b09e7e7e7229868a7c861aeafebbc69bbb322a3` (`origin/main`, including merge of FOR-23)  
**Scope:** descriptive audit of fixed FOR-19 baseline matches and legal, one-decision counterfactuals using the real L0 engine. This does not decide that Gate 1 passed, measure Gate 0 Windows performance, or authorize a gameplay/balance change.

## 1. Questions and interpretation limits

This study separates engine terminal timeouts from economic collapse, repeated orders, passive defense, and potentially useful or harmful one-shot actions. Every live episode reaching `T_max=3000` is still scored by the unchanged engine terminal contract as a `0.5/0.5` timeout draw. A timeout alone is not an exploit, a pathology, or proof that a policy is weak. The study does not require every match to end in a win.

All new rates and flags are descriptive for this fixed schedule. A seat swap is paired, deterministic seed labels can duplicate trajectories, and maps are selected fixed scenarios. No confidence intervals, p-values, inferential claims, or post-result thresholds are planned. In particular, do not sum the 84 FOR-22 and 330 FOR-23 timeout rows as independent evidence or call their differing raw rates an increase.

Final interpretation must keep four possibilities distinct:

1. **Rule-level pathology:** evidence that a rule-level timeout exploit exists; timeout frequency by itself does not qualify.
2. **Policy-level failure:** current baseline bots appear to stall in these scenarios, without a universal strategy claim.
3. **Nonpathological stalemate:** both sides can sustain defense and a one-sided exit loses for the acting colony.
4. **Insufficient evidence:** missing metrics, insufficient distinct trajectories, contradictory evidence, or an inconclusive probe.

The final design recommendation answers whether a separately preregistered mechanical balance investigation is needed before Gate 2A, or whether bot optimization under frozen rules is a reasonable next experiment. Any further experiment is a recommendation for a possible FOR-25 only; it is not part of this implementation. Gate 0 and Gate 1 remain unpassed unless decided separately.

## 2. Known evidence frozen before this study

FOR-20 through FOR-23 results were available when this protocol was written. They may guide the fixed matchup choice below, but are historical context, not observations in this schedule.

* **FOR-20 economic pilot:** 144 rows across 48 conditions and 3 seed labels per condition collapsed to one unique trajectory per condition (96 duplicate rows). All runs stopped at pilot horizon `t=1000`, not at a game terminal. Four of twelve `eggs_per_lay=1` active conditions met the preregistered viability screen, and every active condition had food-rejection concerns. Worker-count plateaus and feeding/reproduction trade-offs were observed, but the study had no opponent or combat and cannot explain FOR-19 timeouts. Source/protocol evidence is recorded in `docs/economy_pilot_findings.md` and `docs/economy_pilot_protocol.md`; protocol SHA-256 `57f33e613599542d89f8a5a6ea8c5d0a33f29022564b056fb2c0a93bd87d93d0`, protocol-only commit `e1573f8cafc2e9e7d093a4c5ae573760ded954fd`, matrix source commit `9a3e9767bcad2ce38a775cd807f18f55c97bea48`.
* **FOR-21 causal probes:** 48 baseline episodes, 80 eligible checkpoint states, and 240 scheduled axis rows; each axis was inconclusive under the predeclared unique-trajectory minimum. The home-only forage contrast was descriptive and heterogeneous; a one-shot action did not establish strategy dominance. Strict replay used 623,582 calls. See `docs/gate1_causal_findings.md` and the frozen `docs/gate1_causal_protocol.md` (SHA-256 `a87cf2f399375c7841ae10f0fb34ee14b2f75c62f0b70400ad04f40684a39c22`; source commit `e5309ee0e70d2f64e7462aebf994338854fefe30`).
* **FOR-22 cross-map context:** 720/720 terminal matches used 401,898 calls. There were 84 engine timeout draws in seven matchup×map groups; six groups were Turtle–Balanced or Turtle–Raider on R0, and 16 groups met the repetitive-order flag. The 720 scheduled rows collapsed to 90 distinct paired-trajectory blocks; deterministic repeats and mirrored seats are not independent. See `docs/gate1_context_findings.md`; source commit `4b1bf18dfdfb8585a5a5045c4b11d742b7d6cf9f`, source fingerprint `5410221b382a3840c269d8e13177164a9a3cbba49904b8ebff5f247191bcc6b4`, protocol SHA-256 `b0eb1685a2d0e024e915f42d7833e6f27fc9be4add4c26ceac921b77f07edff3`.
* **FOR-23 feedback ablation:** 1,728/1,728 matches used 1,395,430 calls; 330 ended by engine timeout. All four research labels were INCONCLUSIVE because support spanned only one opponent family. Its full rerun followed a documented protocol deviation and is exploratory. See `docs/gate1_feedback_findings.md`; source commit `8b9117d1b3d6229dafa3f7ad7e44f39f00e00044`, source fingerprint `5843c847756e556755bb753278e0f1d58e2e15afaad9e9139106cde9a91f0900`, protocol SHA-256 `98b166ad5fd31451990f7aaf49a7fc23c34d5ba10a4193a37d5d992d244eca4b`.

The exact archived FOR-22 inputs read before registration are:

| Artifact | SHA-256 |
| --- | --- |
| `results/gate1_context/manifest.json` | `94457151fbf59b07e189ef52e68a16885a34cd177ea7e2ec55f695e7276d529d` |
| `results/gate1_context/matches.jsonl` | `cb895bfe85d2aa641c30a05ee300f35e63e9b7b90e580e5f8f1a59c20859b7b4` |
| `results/gate1_context/summary.json` | `4c95300506662a03e79beab41e68eceb30c848f95d5b9c9f4fa2400660559944` |

The exact archived FOR-23 inputs read before registration are:

| Artifact | SHA-256 |
| --- | --- |
| `results/gate1_feedback/manifest.json` | `60b76d98b616a395401a12461f4ffd4953d8c13437ee5fcf7af5f5edeac3aef7` |
| `results/gate1_feedback/matches.jsonl` | `84243c49ef0b0ac0d199a6f1ea0a96dccfe124a426986615e9eeeb7406c5c8ce` |
| `results/gate1_feedback/summary.json` | `60fc4b9a269867e6196e9efa230c27f90fc5a7de7056f610ce6c4b091fabe1f3` |

The preliminary metadata taxonomy is: matchup/map; engine terminal reason, winner and score; duration and timeout; orders and passive standing orders; lay attempts/rejections; harvest/upkeep/starvation; food, army and queen HP; policy stream/seat identity; and requested metrics absent from archival files. Missing archival values are `not_measured`. A hash does not permit reconstructing omitted events or state. The FOR-22 and FOR-23 artifacts above are used only to choose the fixed schedule and to describe known history; they are never added to the new independent sample.

## 3. Frozen engine, configurations, policies and hashes

Use the existing `gate1_context._build_configs()` configurations and require exact canonical `config_hash` equality with these published FOR-22 values:

| Map | Config SHA-256 |
| --- | --- |
| `R0-T0` | `3f6aaa36a0d3d15888fa42f142df1151376794e62b553ab8e5f7659c9794145d` |
| `R0-T1` | `78e9c6979a2fb3973ae9a275f79ad5b62cb4b6ed3893a8f8469e4cc599e5fb34` |
| `R0-T2` | `b9e43b2aec34f303f371a9f9e5a2bcd8b5c5d894e608a3ca9e71386339f632f1` |
| `R1-T0` | `407bb3907099533adfe10b7c300a480b45b5d604dc019693b1ac99f47502303f` |
| `R1-T1` | `3bf8dabf6011ed88275c3fcfec3875c4da46ef849e7da7b0ac17bb2cd0e9e605` |
| `R1-T2` | `1b6f1f999246711919fb4dee9238e5fed31c5baac0dca779ea76c47afe167ceb` |

Use the unchanged FOR-19 `random:v1`, `turtle:v1`, `rush:v1`, `balanced:v1`, and `raider:v1` policies, the existing `ObservationV2`, `ActionV2`, mask, shared decoder/validator, `K=30`, policy-identity PCG32 streams, and `T_max=3000`. The baseline Python source at the preregistration base has recursive `src/formicmind/**/*.py` fingerprint SHA-256 `5843c847756e556755bb753278e0f1d58e2e15afaad9e9139106cde9a91f0900` under `gate1_context._source_fingerprint()`'s sorted-path/canonical-manifest algorithm. The final manifest must bind both base and final harness source identities; strict verification is valid only for the exact recorded final fingerprint.

Frozen policy parameter digests:

| Policy | Policy ID | Parameter SHA-256 | Stream IDs (replicate 0/1) |
| --- | --- | --- | --- |
| `random` | `random-v1` | `254037fbbe5f089fc1f909405ebcc3567573682553d3715d9bab8cf292b07f2c` | `policy:random-v1:0`, `policy:random-v1:1` |
| `turtle` | `turtle-v1` | `6c9fa1c676a553294c26ffb4f3372953f93ecec73b5d8669f99a451a05e0ee4e` | `policy:turtle-v1:0`, `policy:turtle-v1:1` |
| `rush` | `rush-v1` | `cfea106a71257d8ebe79cfa6ac090b1927fd3010ca905516e2e41512460ea496` | `policy:rush-v1:0`, `policy:rush-v1:1` |
| `balanced` | `balanced-v1` | `03c6760da2b2fc42108133c8b81a91b70e3c086327fbabedf9d3f30eb7dcdd0f` | `policy:balanced-v1:0`, `policy:balanced-v1:1` |
| `raider` | `raider-v1` | `6d4a23b0071c6ee308b8149850debe488e83d79c7fda50f19ee5053310c39b19` | `policy:raider-v1:0`, `policy:raider-v1:1` |

No manual parameter changes, policy selection, tuning, reward changes, or policy/action fallback is permitted. Keep each policy stream attached to policy identity when seats swap. The two streams are distinct where both replicates are needed; do not share a stream across simultaneous seats.

## 4. Fixed baseline tournament

Run the four unordered, two-sided matchups below on all six maps:

| Pair | Purpose fixed before run |
| --- | --- |
| `turtle–balanced` | Historical timeout-prone defensive comparison |
| `turtle–raider` | Historical timeout-prone defensive comparison |
| `rush–balanced` | Aggressive comparison |
| `random–turtle` | Stochastic comparison |

Use exactly the six open seeds `20270201`, `20270202`, `20270203`, `20270204`, `20270205`, `20270206`, whose canonical integer-array SHA-256 is `ed35d5d2f19af9a36ae72dd6afc33709fd5d43240e626679144b6ee8f3c5b18a`. For every map×pair×seed, schedule both seat assignments: first listed policy on A / first listed policy on B. Assignment descriptor array canonical SHA-256: `edbd1306cfd9beea32c7703bb5e77cbb6a89277ef33d6bf999dddc46d6c93252`. Keep all 288 stable schedule rows even when trajectories repeat.

The complete schedule is `6 maps × 4 matchups × 6 seeds × 2 seats = 288` real terminal episodes; maximum baseline budget is `288 × 3000 = 864,000 step_tick calls`. Run `reset(config, seed)` followed by the real `step_tick` until engine terminal. Never fabricate a timeout or outcome at a harness boundary. An engine timeout is a draw only if `EpisodeOutcome` reports that terminal. A nonterminal cutoff, technical failure, or missing terminal is incomplete/censored and invalidates affected analysis. Do not replace seeds, omit failed rows, alter the schedule, or continue after a stop-triggering integrity failure.

## 5. Online measurements and taxonomy

For every match capture after completed ticks `0, 100, 300, 600, 900, 1500, 2100, 2700`, and at terminal. Tick zero is the reset state before the first tick. A checkpoint that cannot be reached because the episode already ended is `not_measured`, with the terminal tick/reason. Do not substitute zero or another horizon.

At each available checkpoint and terminal, record policy-identity-relative live values for both colonies: total and stationed/in-transit army and worker counts, brood by caste, queen alive/HP, colony food, per-node food, soldier location (home/center/enemy home/other) and transit destination. Online event counters per policy include harvest, upkeep due/paid/missing, lay attempts/successes/food rejections, starvation deaths, combat events and damage, queen damage, order count, retargets, and whether raid was ever ordered. Track decision counts, `lay_none`, home-only forage, garrison/defend/raid choices, distinct army choices and distinct forage vectors. For last-600 completed ticks, retain equivalent rolling event counters and queen damage; for last-900 completed ticks retain whether `army_choice=6` occurred. Update counts/digests online; do not retain full per-tick event lists or a million-tick replay in memory. Use existing state/events, and do not add instrumentation to physics or engine behavior.

Record terminal reason/winner/score, terminal tick, duration, completion/censoring state, raw engine timeout count, terminal colony metrics, and actual `step_tick` calls. Keep unknowns as `not_measured`. Event and state fields absent in archival data remain so in the retrospective appendix.

### Predeclared descriptive flags

Each `matchup × map` has 12 raw scheduled episodes. Publish numerator, denominator, raw descriptive fraction and unique policy-relative trajectory-block count for every flag. Flags are review prompts, not pass/fail gates, balance claims, or independent sample counts.

* `timeout_concentration`: at least `3/12` true engine timeout outcomes.
* `repetitive_orders`: at least `3/12` episodes with 10 or more decisions and, for either policy, one army choice on at least 90% of decisions, or one forage vector on at least 90%, or `lay_none` on at least 75%.
* `late_zero_queen_damage`: at least `3/12` timeout episodes where neither queen received damage in the final 600 completed ticks. Report the final-600 combat-event count separately.
* `passive_raid_avoidance`: at least `3/12` timeout episodes where at least one policy-identity side never chose `army_choice=6` in the final 900 completed ticks.
* `economic_stress`: at least `3/12` timeout episodes where either side had starvation deaths or food-rejected lays in the final 600 completed ticks. This is not itself a stall classification.
* `food_stockpile`: at least `3/12` timeout episodes where both queens live, both sides have at least 300 colony food, and both have at least one adult soldier at terminal. Do not alter these thresholds.

All flag boundaries are inclusive at the stated threshold. A group with fewer than 12 valid scheduled rows remains incomplete; do not change its denominator or call the flag a negative finding.

## 6. Trajectory normalization and independence

Keep every raw schedule row, world-specific state/checkpoint hash, action/order trace digest, and event digest for strict integrity. In addition compute a policy-identity-relative trace: normalize sides, mirrored node IDs, and persistent entity IDs into the first-listed policy's local frame; include canonical per-tick state and normalized events, policy decisions, and the consumed pre/post RNG state for `random:v1`. Do not put the root seed, seat label, row ID, or unused random stream state in the equivalence digest. Hash the canonical ordered trajectory. A policy-pair block is the composite of the two seat-assignment traces for that same map, matchup and seed. Repeated deterministic traces collapse, but genuinely different random RNG trajectories remain distinct.

First pair the two seat assignments within a `(map, matchup, seed)` block. Then group identical composite policy-relative paired trajectory digests within the same map and matchup. Report exact scheduled rows, seed blocks, mirror pairs, distinct per-match traces, distinct paired-trajectory blocks, duplicates, and distinct random-policy trajectories. Do not treat A/B seat swaps, six seed labels, or duplicated deterministic paths as independent N. Do not collapse across maps or matchups. Raw schedule percentages must be labeled descriptive, and outcome summaries must state whether they use raw rows or unique blocks. No CI/p-value is allowed.

## 7. Fixed one-shot counterfactual probes

Probe only baseline schedule rows for maps `R0-T0` and `R1-T0`, and pairs `turtle–balanced` and `turtle–raider`, using all six seeds and both seat assignments: exactly `2 × 2 × 6 × 2 = 48` baseline episodes. Each baseline row receives the full predeclared descriptor product:

* checkpoint decision ticks `300` and `900`;
* target role: each of the two policy identities in that matchup;
* one alternative: `raid_once` or `home_forage_once`.

This is exactly `48 × 2 × 2 × 2 = 384` planned probe rows, regardless of eligibility. Probe row ordering is baseline schedule order, then checkpoint ascending, target policy order as listed in the matchup, then intervention order (`raid_once`, `home_forage_once`). No result-dependent selection or replacement is allowed.

For every scheduled decision tick, save the full baseline checkpoint only in memory: `SimState`, `RunState`, both policy windows, and all RNG stream states. Before treatment, compare baseline and probe checkpoint `state_hash`, `checkpoint_hash`, policy RNG states, first-decision input/mask/action hashes, and time fields. A control continuation from the checkpoint must reproduce the baseline suffix's canonical outcome and trace exactly; control verification does not create another analysis row. If any control comparison fails, stop and mark the attempt invalid.

Run a probe from a deep copy of the actual checkpoint through real `step_tick` calls with unchanged controllers. At the single selected policy decision, first consume the original policy decision and RNG draws, then make exactly one legal replacement action; at every later decision restore the original policy and RNG semantics. Do not patch the engine, add reward, inspect hidden opponent state, or use future information. Only the chosen action field may differ from the normal baseline action.

| Intervention | Frozen replacement rule | Eligibility |
| --- | --- | --- |
| `raid_once` | Set `army_choice=6`; copy every other field from that policy's normal action. | Current mask allows `army_choice=6`, candidate differs from normal action, and the full `ActionV2` passes the unchanged shared validator. |
| `home_forage_once` | Set `forage_share_q=(65535,0,0,0,0,0)`; copy every other field from the normal action. | Candidate differs from normal action and the full `ActionV2` passes the unchanged shared validator. |

If the checkpoint is already terminal/unreached, baseline action is unavailable, candidate equals baseline, current legality/validation fails, or any required precondition fails, emit the planned row as `INELIGIBLE` or `NO_EFFECT` with a precise reason and zero continuation calls. Never substitute another action, side, checkpoint or seed. `NO_EFFECT` is reserved for a legal one-shot action that is identical to the baseline action; invalid/masked candidates are `INELIGIBLE`. Retain every planned row.

For each eligible probe, record intervention decision tick, changed action axis/value, first changed order tick, first **physical** divergence tick, trace hashes, terminal outcome, target terminal score delta versus baseline, and the predeclared checkpoint/terminal metrics. Use the FOR-21 physical projection (physical state and non-order events); an order-only change is not physical divergence. Physical traces may later reconverge, so also report first divergence and terminal/intermediate values. Do not infer an effect solely from issued order or endpoint hash.

The planned full-run ceiling is `864,000` baseline calls plus `1,152,000` probe continuation calls (`384 × 3000`) = **2,016,000 `step_tick` calls**. Charge actual executed calls separately to baseline and eligible probe phases. Ineligible rows cost zero probe continuation calls. Strict verifier replay work is reported in a separate verification counter and does not enlarge the study's primary call budget. Abort immediately on a primary run budget excess. An engine timeout is a real terminal draw, not a harness cutoff.

### Predeclared probe labels

Labels are descriptive. Baseline score is from its real `EpisodeOutcome`; target scores are `0`, `0.5`, or `1` only.

* `TIMEOUT_ESCAPE`: baseline ended in an engine timeout and the valid probe ends by queen death or economic collapse.
* `BENEFICIAL_TIMEOUT_ESCAPE`: `TIMEOUT_ESCAPE` and target score rises from `0.5` to `1`.
* `HARMFUL_TIMEOUT_ESCAPE`: `TIMEOUT_ESCAPE` and target score falls from `0.5` to `0`.
* `TIMEOUT_PERSISTED`: both baseline and probe end in engine timeout draws.
* `OTHER_OUTCOME_CHANGE`: a valid probe changes outcome when baseline was not a timeout, or a valid terminal change does not meet the escape definitions.
* `NO_OUTCOME_CHANGE`: valid terminal outcome/score unchanged; retain intermediate effects separately.
* `INELIGIBLE`: a required checkpoint/action/legal precondition is absent or fails.
* `NO_EFFECT`: legal alternative exactly equals the normal baseline action; no continuation is run.
* `INCOMPLETE`: technical failure or nonterminal probe cutoff; never score as a draw.

Report scheduled, eligible and unique eligible contexts, and counts by target role, intervention, map/resource family and checkpoint. Collapse duplicated probe contexts using their baseline policy-relative trajectory/checkpoint digest plus role/checkpoint/intervention; keep all scheduled probe rows. Do not average duplicated rows as independent evidence. Do not search for a better checkpoint/dose or calculate CI/p-values.

## 8. Artifacts, provenance and strict verification

Required deliverables:

* this frozen protocol, committed alone before implementation;
* `src/formicmind/gate1_stalling.py` with `python -m formicmind.gate1_stalling run --out <NEW_DIR>` and `verify --input <DIR>`;
* `results/gate1_stalling/manifest.json`, `baselines.jsonl` (288 rows), `probes.jsonl` (384 rows), and `summary.json`;
* `docs/gate1_stalling_findings.md` with exact sources, findings, 2–3 concrete event/state timelines where available, recommendations and limitations.

The manifest records protocol ID/SHA-256, base SHA, final source commit, recursive `src/formicmind/**/*.py` source fingerprint, six config hashes, policy IDs/parameter/stream hashes, seed and seat schedule hashes, versions, run commands, package/Python versions, exact planned counts and budget. Rows bind to protocol/source/config/policy/seed/schedule IDs and retain raw integrity digests. Hash every emitted artifact in the manifest; omit wall-clock values from canonical row/summary hashes.

`verify --input` strictly regenerates all 288 baselines and all 384 planned descriptors, fully replaying every eligible probe from its saved checkpoint with the same engine and decoder. It checks each expected schedule row, config, policy, seed, action legality, checkpoint/control equality, first order/physical divergence, event/state/terminal hashes, phase counters, trajectory grouping, flags, labels, summary and artifact digests. It must reject altered/missing rows, false terminal outcomes, protocol/config/policy/source drift, incomplete or extra schedule rows, and a modified summary. Re-aggregating saved JSON without replay is not verification. Verification failure is explicit; do not silently relabel or repair results. Documentation-only changes after the frozen run do not alter source fingerprint.

On the first technical failure, preserve an invalid attempt with its failure point and counters, then stop. Do not change fixed inputs or run past the budget. A technical repair may be followed by a full restart only under the same still-unviewed schedule and unchanged protocol; preserve the failed attempt separately and label any deviation honestly. Never overwrite a nonempty output directory.

## 9. Required tests and external checks

Tests must cover exact `288` baseline rows and `384` planned probe descriptors; exact six config and policy/seed hashes; paired seat and duplicate-trajectory collapse while preserving distinct random paths; missing horizon/terminal metadata; threshold boundaries at exactly 3/12, 10 decisions, 90% army/forage repetition and 75% lay-none; a real `T_max=3000` timeout rather than fabricated timeout; one action/one axis/one decision only; mask-illegal raid and identical-action handling; checkpoint/state/RNG equality before alteration; order-only versus physical divergence; early terminal metrics; strict tamper/missing-row rejection; and deterministic replay.

Required validation on the exact PR head: strict `verify`, complete Python pytest suite, frontend production build and Vitest suite, and Windows CI. Windows CI success is not Gate 0 performance p10 measurement. Report the exact PR head and each check result; do not merge.

## 10. Explicit exclusions

Do not change `default_config`, map layout, resource/combat balance, tick or physics, combat, terminal scoring, timeout rules, policies or policy parameters, RNG, observations/actions, masks, decoder/validator, replay format, FOR-20/21/22/23 frozen protocols or results, or existing golden fixtures. Do not add anti-draw penalties, starvation, forced aggression, reward shaping, army-strength initiative/score, an artificial “strong bot,” learning, or post-result tuning. Code changes are limited to the experimental harness, its tests, and new FOR-24 documents/results.
