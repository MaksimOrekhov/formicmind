# Gate 1A causal action probes — findings

**Status:** completed open diagnostic; **Gate 1 is not passed**.  
**Protocol:** [`gate1_causal_protocol.md`](gate1_causal_protocol.md), first committed as `a16a733`, deterministic-hash clarification committed as `2dcbc8f`.  
**Engine/source commit:** `836eaa2266333ed98c79546c13a07b5eebe3189b` (`source_fingerprint_sha256` is in the manifest).  
**Starting main:** `13d80fd01898476eb5050cab7f232520886e7c42`.  
**Config:** default config, `config_sha256=3f6aaa36a0d3d15888fa42f142df1151376794e62b553ab8e5f7659c9794145d`. No gameplay, balance, physics, baseline policy, action, observation, decoder, or Replay v1 code was changed.

## Study size and baseline outcomes

The fixed schedule ran 48 complete baseline matches: three FOR-19 matchups, eight open seeds per matchup, and both seat assignments. Checkpoints were start-of-policy ticks `30`, `300`, and `900`. Earlier terminal matches have no later checkpoint; the schedule was not extended. The 80 reached checkpoint states generated 240 per-axis rows.

| Matchup (target) | Baseline terminal outcomes | Unique full trajectories |
| --- | --- | ---: |
| balanced–rush (`balanced`) | 16/16 target wins by queen death | 2 |
| turtle–raider (`turtle`) | 16/16 timeout draws | 2 |
| random–balanced (`random`) | 10 queen-death losses, 6 economic-collapse losses | 16 |

The eight seeds in each deterministic matchup repeat one trajectory per seat assignment. They remain in raw files but count as two unique trajectories, not sixteen independent observations. The random–balanced matchup produced 16 unique trajectories. Across the primary analysis, the bootstrap therefore has 20 unique trajectory clusters for lay and forage and 18 for army; all axes meet the preregistered minimums, though the deterministic matchups sit exactly at the minimum of two clusters each.

## Primary terminal-score results

Primary score is the target policy's true terminal reward (`0`, `0.5`, `1`). The table reports equal-matchup/equal-unique-trajectory paired deltas; MDE is `0.10`. Intervals are the preregistered 10,000-replicate, matchup-stratified cluster bootstrap.

| Intervention | Eligible pairs | Distinct checkpoint states / unique trajectories | Order changed / physical effect / empty | Mean score delta (95% interval) | Preregistered label |
| --- | ---: | ---: | ---: | ---: | --- |
| `lay_order` → worker/none | 80 | 24 / 20 | 80 / 80 / 0 | `0.000` (`0.000`, `0.000`) | Practically small |
| `forage_share_q` → home only | 80 | 24 / 20 | 80 / 80 / 0 | `−0.389` (`−0.389`, `−0.389`) | Harmful signal |
| `army_choice` → raid enemy home | 46 | 18 / 18 | 46 / 44 / 2 | `0.000` (`0.000`, `0.000`) | Practically small |

The forage result is heterogeneous by matchup and checkpoint:

- In balanced–rush, the target `balanced` control wins, while the home-only forage treatment loses, in all 16 scheduled rows at tick 30. Those rows reduce to two unique mirrored trajectories.
- In turtle–raider, the tick-30 treatment changes the target from timeout draw to queen-death loss in 16 rows (two unique trajectories); the tick-300 and tick-900 paired terminal scores are unchanged.
- In random–balanced, target `random` loses under both arms in all 16 unique trajectories, so its terminal reward cannot fall further. Physical and starvation effects still occur.

At the `+150` horizon for the balanced–rush forage probe, home-only treatment has, versus its control, 197 less target-colony food, 11 fewer workers, and 13 more starvation deaths in each of the two unique seat trajectories. This is a real downstream effect, not only a changed order log. Lay and army interventions also changed downstream state in most eligible pairs, but their paired terminal reward deltas stayed at zero in this schedule. The two eligible army pairs with no measured downstream state or terminal change are reported as empty; no axis met the preregistered dominant/free-action flag.

The bootstrap interval for each axis is degenerate because unique trajectory clusters within a matchup have identical mean deltas in this fixed schedule. It describes this diagnostic schedule, not generalization to other maps, seeds, or opponents. It is not a standalone significance claim. No terminal arm was censored or assigned a synthetic draw. Intermediate horizons can be censored after an early terminal: paired `+30` states are available for 78/80 lay and forage pairs and 44/46 army pairs; paired `+150` states are available for 48/80 lay and forage pairs and 16/46 army pairs. Censored values remain explicitly marked in the raw rows.

Army eligibility was absent in 34 scheduled rows because the target had no adult soldiers at the checkpoint. Lay and forage had no ineligible rows. All eligible actions passed the common validator and changed only their declared axis. Per-row actions, event counts, resources, populations, positions, intermediate hashes, tick-trace digests, terminal reasons, and final hashes are in `probes.jsonl`.

## Integrity and reproducibility

- `run` executed 623,582 `step_tick` calls, below the preregistered 3,000,000-call ceiling.
- `verify --input results/gate1_causal` fully re-executed all 48 baselines and every eligible paired arm. It matched all 48 baseline rows, all 240 probe rows, action and eligibility records, tick traces, endpoint/terminal hashes, and the recomputed summary.
- Exact commands (choose a new, not-yet-existing `--out` directory for a fresh run):

  ```sh
  PYTHONPATH=src python3.12 -m formicmind.gate1_causal run --out results/gate1_causal-reproduction
  PYTHONPATH=src python3.12 -m formicmind.gate1_causal verify --input results/gate1_causal
  ```

- `manifest_sha256` (canonical manifest bytes, without trailing newline): `bcfbdc28c18de3cacf9012bb652cd73f204f4d51ac8568fecf78fd42babcbbfd`.
- File SHA-256 values:

  | Artifact | SHA-256 |
  | --- | --- |
  | `manifest.json` | `3b5c78aef3fa382b0cb1d3e79e5e873ddf97a32a3d74f72430df60aabc228d7a` |
  | `baselines.jsonl` | `a1a9ce733161975a80494d5eb6566a700a2bfd2fe0f63b6a8cdbaa1e7d113816` |
  | `probes.jsonl` | `de2ba6fb274c6e26ef3cfd7a8205c6844c3684722beed3446f22f9623a4fafa5` |
  | `summary.json` | `85495155cbed902a67b21d7020b416a31b66acbefb9e2f088d4a2d513f2d1d3e` |

Three early harness invocations were discarded before publication: one exposed a missing trace-count accessor; one completed calculations but rejected a nullable eligibility field during canonical output; the retry then found the incomplete output directory left by that failure. The schedule and protocol were not changed. The incomplete directory was moved outside the repository, the serializer fix was committed as `836eaa2`, and the full schedule was rerun and independently verified at that source fingerprint.

## Checks and limits

- Targeted FOR-21 tests: `8 passed`.
- Full Python suite: `838 passed`.
- Vue production build: passed.
- Vitest: `18 passed`.
- Windows GitHub Actions: pending PR CI; local macOS results do not substitute for it.
- Gate 0 single-core Windows performance remains `NOT MEASURED`. This work tests only current default-config matches on one map; cross-map context maps, open-loop versus closed-loop comparison, and anti-stalling remain separate work.

The measured forage harm is a result about the fixed home-only alternative in these paired contexts. It is not authorization to change forage policy, balance, or physics. No broad claim about strategy strength or game balance follows from this open diagnostic.
