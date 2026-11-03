# Gate 1D stalling audit: findings and design recommendation

## Decision

The audit does not show a rule-level timeout exploit, and it does not justify changing combat, economy, terminal scoring, timeout handling, or bot balance before a separate decision. It does show repeatable stalling patterns for the current defensive baseline policies in the fixed scenarios. The counterfactual evidence is too narrow to establish why those games stall: the 12 scheduled seed/seat rows in each deterministic defensive cell collapse to one paired trajectory, and only one unique checkpoint context was available per intervention in an eligible cell.

**Recommendation:** keep the rules frozen and make bot optimization under the existing rules the next experiment (a possible FOR-25). Require broader opponent families and genuinely distinct trajectory/checkpoint contexts before using results to justify a mechanical balance investigation. FOR-24 does not pass Gate 1 or authorize Gate 2A; those remain separate decisions.

## Preregistered run and provenance

The frozen protocol was committed by itself before implementation in commit `5180ba8` (protocol SHA-256 `0d54603c90526f7d51705f5fcf7c70290d854324be5eb73297932810ec1f3b5d`). The final corrected study used source commit `c75ce28185431d7f6ff7e88fae91af62eb35044c`, recursive source fingerprint `5fd459d1881a2316ef10f120b6d913706c77591ab22d198d9f7f1b972ef195d8`, and base `9b09e7e7e7229868a7c861aeafebbc69bbb322a3`.

The fixed schedule completed all 288 terminal baselines and all 384 planned probe descriptors. Baselines used 335,968 `step_tick` calls; eligible probe continuations used 403,896; total primary use was 739,864 of the 2,016,000-call ceiling. No study row was cut off or replaced. There were 192 eligible probes and 192 ineligible probes; the latter were retained with zero continuation calls.

Strict verification passed: all 288 baselines and 384 probe descriptors were regenerated, and all 192 eligible probes were replayed. Verification used 739,864 baseline/probe replay calls plus 115,200 control-suffix calls (855,064 verification-only calls total, reported separately from the primary study budget). Raw rows, hashes, terminal outcomes, trajectory grouping, flags, and summary matched exactly.

Final artifacts:

| Artifact | SHA-256 |
| --- | --- |
| `results/gate1_stalling/manifest.json` | `be03ac6e02cbf911ddac1a1a59e82b7676a41c9603a02533d352f09dd66c5521` |
| `results/gate1_stalling/baselines.jsonl` | `ebd37e04504101c24e7744a1a461e18c38c67c2873f2b9b255270e657a3c4ba6` |
| `results/gate1_stalling/probes.jsonl` | `a59f20c12cdbcb8ce09d7923d346c4eef4bcaf15d4e666794601278d235294c2` |
| `results/gate1_stalling/summary.json` | `47c0eb534a6e193f98ba5aa90fbb985b3f19f864ff3150da265b29b58f545db4` |

### Analysis correction

The first complete output was preserved in the local `results/gate1_stalling_attempt1/` archive. Its probe deduplication key accidentally included a raw control-suffix digest, so deterministic mirror/seed duplicates were counted as distinct contexts even though their preregistered policy-relative checkpoints were identical. After inspecting the output, the key was corrected to use the normalized checkpoint plus map, matchup, target policy, checkpoint tick, and intervention, as required by the frozen independence rule. The full fixed schedule was then rerun and these final artifacts use the corrected grouping. The protocol file, schedule, seeds, engine inputs, treatments, thresholds, and raw outcome labels were not changed. The initial output must not be used as final evidence.

## Fixed-schedule results

The 288 raw baseline rows ended in 86 engine timeout draws, 134 queen deaths, and 68 economic collapses. The timeout fraction `86/288` is descriptive for this schedule; it is not an estimate of a general game-wide rate.

The 86 timeouts were concentrated in seven matchup-by-map cells: Turtle–Balanced on R0-T0, R0-T1, R0-T2, and R1-T1; Turtle–Raider on R0-T0, R0-T1, and R0-T2. Each cell had 12/12 timeout rows, but each had only one distinct paired-trajectory block. Two additional raw timeout rows occurred in Random–Turtle on R1-T2, a cell with six distinct paired blocks. No `food_stockpile` flag met its threshold.

The timeout-prone cells also met the preregistered review flags for repetitive orders, zero queen damage in the final 600 ticks, passive raid avoidance in the final 900 ticks, and economic stress. These are diagnostic flags, not proof of pathology. Repetitive orders also flagged Turtle–Balanced on R1-T0, where all 12 rows ended by queen death before tick 300.

Across all cells, 108 per-match policy-relative trajectories formed 54 distinct paired blocks. Thirty distinct random-policy stream trajectories were retained. Eighteen of 24 matchup-by-map cells had one distinct paired block; six cells had six. Seat mirrors, seed labels, and repeated deterministic paths therefore do not provide 288 independent observations. No confidence intervals or p-values are reported.

### One-shot probes

All eligible probes came from R0-T0. On R1-T0, every Turtle–Balanced baseline ended by queen death at tick 297 and every Turtle–Raider baseline at tick 107, so neither preregistered checkpoint (300 or 900) existed. All 192 R1-T0 probe descriptors are correctly marked `INELIGIBLE`.

The 192 eligible raw rows reduced to 16 unique checkpoint/intervention contexts: 14 `TIMEOUT_PERSISTED` and two `HARMFUL_TIMEOUT_ESCAPE`. Each was repeated across seed labels and seat mirrors; there was one unique context per treatment within each target/checkpoint group. No beneficial escape occurred. The two harmful unique contexts are the same home-only-forage intervention at checkpoints 300 and 900 on the same deterministic Turtle–Raider / Raider-target trajectory. The raw 24 harmful rows are repeated executions of those two contexts, not 24 independent outcomes.

`raid_once` changed a legal army choice from 5 to 6 at the one target decision. In the eligible Turtle–Raider / Raider-target context it caused immediate physical divergence, but the game still reached the ordinary timeout draw and the target score stayed at 0.5. The intervention therefore did not demonstrate a way out of the timeout.

`home_forage_once` replaced the target's forage vector with `[65535, 0, 0, 0, 0, 0]` for exactly one decision. In the two unique Raider-target contexts, the continuation ended in economic collapse 30 ticks after the intervention; the target score changed from 0.5 to 0. This is a harmful one-shot result in one state, not evidence that home-only forage is generally harmful.

## Representative timelines

1. **R0-T0 Turtle–Balanced, seed 20270201, first_A.** At reset both colonies had 240 queen HP, four soldiers, 12 workers, and 60 food. At tick 300, Balanced had zero food, four soldiers, and five workers; Turtle had 205 food, no soldiers, and 32 workers. Those colony counts remained unchanged through ticks 900, 1500, 2100, and 2700. The engine timed out at terminal tick 2999 with both queens alive at 240 HP. In the final 600 ticks there were zero combat events and zero queen damage; Turtle had 41 starvation deaths and 40 rejected lays. This is a real timeout draw with economic stress and a large standing workforce, not by itself evidence of a rule exploit.

2. **R0-T0 Turtle–Raider, seed 20270201, Raider target, tick 300.** The baseline army choice was 5; `raid_once` set it to 6 and diverged physically at tick 300. The continuation still timed out at tick 2999, with both policy scores at 0.5. Its 12 raw seed/seat rows collapse to one unique checkpoint context.

3. **Same matchup and checkpoint, `home_forage_once`.** The target's normal forage vector `[0, 27628, 0, 37907, 0, 0]` changed to `[65535, 0, 0, 0, 0, 0]` for one decision. The continuation ended by economic collapse at terminal tick 329 (state tick 330); the target scored 0 rather than the baseline 0.5. Its 12 raw rows also collapse to one unique checkpoint context.

## Interpretation alongside FOR-20 through FOR-23

| Study | What it established | Limit for FOR-24 interpretation |
| --- | --- | --- |
| [FOR-20 economic pilot](economy_pilot_findings.md) | 144 rows collapsed to 48 conditions; four of 12 active conditions met the viability screen and all active conditions had food-rejection concerns. | It stopped at tick 1000 without an opponent or game terminal; it cannot explain these timeouts. |
| [FOR-21 causal probes](gate1_causal_findings.md) | 48 baselines, 80 eligible checkpoints, and 240 axis rows; labels remained inconclusive, with 623,582 replay calls. | Home-only forage evidence was descriptive and heterogeneous, not a general strategy effect. |
| [FOR-22 cross-map context](gate1_context_findings.md) | 84/720 timeout draws across seven cells and 16 repetitive-order groups; rows collapsed to 90 paired blocks. | Its schedules and units differ. The 84 historic rows are context, not observations to pool with FOR-24. |
| [FOR-23 feedback ablation](gate1_feedback_findings.md) | 330/1,728 timeout draws; all four labels were inconclusive, and the run was exploratory after a documented protocol deviation. | Its raw timeout fraction cannot be compared or summed with this fixed schedule. |

The same seven timeout-prone cells appear in FOR-22 and this study, which makes them useful targets for a new policy experiment. The repeated FOR-24 rows within those cells are still one paired trajectory per cell; recurrence across related studies does not make them independent or demonstrate a universal rules problem.

## Design recommendation and limits

The evidence fits **policy-level stalling in specific current baseline matchups** better than a demonstrated rule-level pathology. It does not establish a broad nonpathological-stalemate mechanism either: timeout matches include substantial economic stress, and the single-state one-shot probes do not isolate a general cause. For causal or universal claims, the correct category remains **insufficient evidence**.

For a possible FOR-25, keep rules, economy, scoring, and timeout handling frozen; test bot decision improvements against multiple opponent families and prioritize genuinely distinct baseline/checkpoint trajectories. Treat raid initiation, food allocation, worker/soldier growth, and sustained queen pressure as separate hypotheses with their own predeclared measures. Do not tune against the duplicated FOR-24 rows or present fewer draws as a design goal by itself. Only if broader independent policy runs expose a reproducible rule-level exploit should a separate mechanical balance investigation be considered before Gate 2A.

The study is a deterministic fixed-scenario audit, not a statistical sample of all maps or players. It does not measure Windows performance, prove strategy dominance, or decide a Gate 1/Gate 2A status.
