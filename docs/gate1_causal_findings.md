# Gate 1A causal action probes — findings

**Status:** completed open diagnostic; **Gate 1 is not passed**. The review correction makes every axis inconclusive under the preregistered minimum-count rules.

**Frozen protocol:** [`gate1_causal_protocol.md`](gate1_causal_protocol.md), first committed as `a16a733`; deterministic hash/interval clarification committed as `2dcbc8f`. Its SHA-256 remains `a87cf2f399375c7841ae10f0fb34ee14b2f75c62f0b70400ad04f40684a39c22`.

**Engine/source commit:** `e5309ee0e70d2f64e7462aebf994338854fefe30` (`source_fingerprint_sha256` is in the manifest).

**Starting main:** `13d80fd01898476eb5050cab7f232520886e7c42`.
**Config:** default config, `config_sha256=3f6aaa36a0d3d15888fa42f142df1151376794e62b553ab8e5f7659c9794145d`. No gameplay, balance, physics, baseline policy, action, observation, decoder, or Replay v1 code was changed.

## Post-review reporting correction (2026-11-03)

The initial report used world/seat-specific `trajectory_sha256` values directly as bootstrap clusters. That split a trajectory and its A/B seat mirror into two clusters, contrary to the frozen protocol. The original per-seat rows and `trajectory_sha256` fields remain intact. The updated artifacts add a verified `mirror_trajectory_sha256`: a target-relative canonical trace of every tick's complete state and normalized events, with random-policy RNG snapshots keyed to target/opponent role. Side-local persistent entity ordinals remove only world-side-specific IDs. Strict replay reproduces the mirror digests. The grouping uses this digest plus the mirror-normalized checkpoint state, so a seat swap is counted once while a different random seed remains a separate trajectory.

This is a reporting/aggregation correction only. The protocol file, eight open seeds (`20261201`–`20261208`), both seat assignments, matchups, checkpoint ticks, interventions, policies, and game rules are unchanged. The original report's cluster counts `2 / 2 / 16`, degenerate bootstrap intervals, and labels `PRACTICALLY_SMALL` / `HARMFUL_SIGNAL` are superseded. Recalculation yields `1 / 1 / 8` mirror-equivalence classes by matchup. The existing bootstrap minima are retained; since no matchup has two independent clusters, no interval is estimable and every axis is `INCONCLUSIVE`. The exact descriptive forage effect remains `−3500/9` millipoints (score delta `−7/18`, rounded `−0.389`). It is not an inferential harmful-signal classification.

The initial `empty_action` flag was inferred from endpoint snapshots. The updated harness hashes each post-tick physical state (resource stock, colony resources/population, unit HP and positions/transit, brood) plus non-order events through the full episode; it excludes standing-order and retarget metadata. `physical_effect` now means this full-episode trace differs. A changed order is called `empty_action` only when both arms terminate and the full physical traces match. Two `random-balanced|20261203|…|30|army_choice` seat rows had matching sampled endpoint metrics but different intermediate physical traces; they are now physical effects. The army axis has 46 full-trace effects and zero empty-action pairs (44 differed at sampled metric horizons).

## Study size and baseline outcomes

The fixed schedule ran 48 complete baseline matches: three FOR-19 matchups, eight open seeds per matchup, and both seat assignments. Checkpoints were start-of-policy ticks `30`, `300`, and `900`. Earlier terminal matches have no later checkpoint; the schedule was not extended. The 80 reached checkpoint states generated 240 per-axis rows. The recalculation used the same 623,582 `step_tick` calls.

| Matchup (target) | Baseline terminal outcomes | Raw trajectory digests | Independent mirror clusters |
| --- | --- | ---: | ---: |
| balanced–rush (`balanced`) | 16/16 target wins by queen death | 2 | 1 |
| turtle–raider (`turtle`) | 16/16 timeout draws | 2 | 1 |
| random–balanced (`random`) | 10 queen-death losses, 6 economic-collapse losses | 16 | 8 |

All 48 raw per-seat baseline rows preserve their original world-specific hashes and metrics. I compared them with the prior published rows: all 48 trajectory hashes and original non-provenance fields match exactly. The 240 probe rows likewise retain their original actions, eligibility, per-arm events, per-tick hashes, horizons, outcomes, and endpoint values; only reviewed classification fields and provenance changed, with mirror/physical trace fields added.

## Primary terminal-score results

Primary score is the target policy's terminal reward (`0`, `0.5`, `1`). The descriptive estimate is the preregistered equal-matchup/equal-independent-trajectory paired delta; MDE is `0.10`. The original 10-cluster / 12-state / ≥2-clusters-per-matchup bootstrap minima are unchanged. All intervals below are `not estimable`, so no bootstrap was run and no axis receives an inferential class.

| Intervention | Eligible pairs | Distinct mirror checkpoint states / trajectory clusters | Changed order / full physical effect / sampled-horizon effect / empty | Mean score delta | 95% interval | Classification |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `lay_order` → worker/none | 80 | 12 / 10 (`1 / 8 / 1`) | 80 / 80 / 80 / 0 | `0.000` | not estimable | `INCONCLUSIVE` |
| `forage_share_q` → home only | 80 | 12 / 10 (`1 / 8 / 1`) | 80 / 80 / 80 / 0 | `−7/18` (`−0.389`) | not estimable | `INCONCLUSIVE` |
| `army_choice` → raid enemy home | 46 | 9 / 9 (`1 / 7 / 1`) | 46 / 46 / 44 / 0 | `0.000` | not estimable | `INCONCLUSIVE` |

Per-matchup trajectory counts are ordered balanced–rush / random–balanced / turtle–raider. The lay and forage axes reach 10 total mirror clusters and 12 checkpoint states, but each deterministic matchup has only one cluster, below the required two. Army also misses the 10-cluster and 12-state minima. Exact eligible-row deltas remain available in `summary.json` and `probes.jsonl`.

The descriptive forage effect remains heterogeneous by matchup and checkpoint:

- In balanced–rush, the target `balanced` control wins while the home-only treatment loses at tick 30 in all 16 seat rows, which form one mirror class. The `+150` treatment delta is `−197` target food, `−11` workers, and `+13` starvation deaths per mirror trajectory.
- In turtle–raider, the tick-30 treatment changes the target from timeout draw to queen-death loss in all 16 rows, one mirror class; tick-300 and tick-900 terminal scores are unchanged.
- In random–balanced, target `random` loses under both arms in all 16 rows, eight mirror classes, so terminal reward cannot fall further. Physical effects still occur.

There are no empty-action pairs after full-trace classification. All 46 eligible army interventions changed physical trace; for two mirrored rows in `random-balanced|20261203|…|30|army_choice`, sampled endpoint metrics alone had missed an intermediate difference. Those rows now count as a physical effect, without changing their terminal score deltas. No terminal arm was censored or assigned a synthetic draw. Intermediate horizons can be censored after an early terminal: paired `+30` states are available for 78/80 lay and forage pairs and 44/46 army pairs; paired `+150` states are available for 48/80 lay and forage pairs and 16/46 army pairs. Censored values remain explicitly marked in raw rows.

Army eligibility was absent in 34 scheduled rows because the target had no adult soldiers at the checkpoint. Lay and forage had no ineligible rows. All eligible actions passed the common validator and changed only their declared axis.

## Integrity and reproducibility

- `run` executed 623,582 `step_tick` calls, below the preregistered 3,000,000-call ceiling.
- Strict `verify` fully re-executed all 48 baselines and every eligible paired arm, reproducing raw and mirror trajectory digests, full physical traces, endpoint/terminal hashes, and the recomputed summary. The four verified artifacts were copied byte-for-byte into `results/gate1_causal`.
- Exact commands (choose a new, not-yet-existing `--out` directory for a fresh run):

  ```sh
  PYTHONPATH=src python3.12 -m formicmind.gate1_causal run --out results/gate1_causal-reproduction
  PYTHONPATH=src python3.12 -m formicmind.gate1_causal verify --input results/gate1_causal
  ```

- `manifest_sha256` (canonical manifest bytes, without trailing newline): `09a66fcdd746e9110f47f5f6defb1bb885656fed8ca8255e88f40ca3821cecbd`.
- File SHA-256 values:

  | Artifact | SHA-256 |
  | --- | --- |
  | `manifest.json` | `c6f71af79fbd2f2294dd716f4341184f0581d9c1365e230da09b193ba128db08` |
  | `baselines.jsonl` | `4887fc6b0259f549bbabe2de6ce04182f563bd6bf993b678a35a7730b905a421` |
  | `probes.jsonl` | `1d44a6768b09754085a750ae8b4725b065d1c84ddc8c2c0e5ef35188b49fb859` |
  | `summary.json` | `34b10c0ab5d2e2f2c8093d38416795ea69706c727523d4e8ae61058fecfec612` |

## Checks and limits

- Targeted FOR-21 tests: `12 passed`.
- Full Python suite: `842 passed`.
- Vue production build: passed; Vitest: `18 passed`.
- Windows GitHub Actions: [run 41616200325](https://github.com/MaksimOrekhov/formicmind/actions/runs/41616200325) passed on corrected head `572b1d748da314b4b074c48917a53673bb70658f` (Python 3.12, Node 22).
- Gate 0 single-core Windows performance remains `NOT MEASURED`. This work tests only current default-config matches on one map; cross-map context maps, open-loop versus closed-loop comparison, and anti-stalling remain separate work.

The measured forage harm is descriptive evidence about the fixed home-only alternative in these paired contexts. The corrected independent sample is below the preregistered inference minimum; no broad claim about strategy strength or game balance follows from this diagnostic. It does not authorize changes to forage policy, balance, or physics.
