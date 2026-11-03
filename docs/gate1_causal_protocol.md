# Gate 1A: preregistered causal action probes

**Protocol ID:** `formicmind.gate1-causal/v1`  
**Registered:** 2026-11-03, before FOR-21 probe runs  
**Scope:** open engineering diagnosis on the current L0 engine. This is not a Gate 0 performance measurement, a Gate 1 pass, a balance test, or a claim that any baseline is strong.

This protocol follows FOR-21 and the accepted evaluation constraints in FOR-4. It uses the existing FOR-19 policies and match loop, with FOR-20's diagnostic results as prior evidence only. FOR-19's public seeds (`20261103..20261107`) and FOR-20's public seeds (`20261111..20261113`) are excluded. No sealed seed is used. The target is one legal action intervention at one pre-policy checkpoint in a real match; both branches then use the unchanged baseline policies and the unchanged `step_tick`.

## 1. Immutable setup and schedule

- Engine: current `default_config()` and its exact canonical `config_hash`; `TickMode.FULL`; `reset → step_tick` only. No alternate engine, patched balance, modified topology, or modified rules.
- Policies: exact existing descriptors `balanced:v1`, `rush:v1`, `turtle:v1`, `raider:v1`, and `random:v1` from `baseline_spec`. Preserve their code, parameters, observation/action interface, decoder, and policy stream IDs. No tuning or winner selection.
- Matchups and target policy (first name is the intervention target): `balanced–rush` (target `balanced`), `turtle–raider` (target `turtle`), `random–balanced` (target `random`).
- Open seed schedule: integers `20261201..20261208` inclusive. For each matchup and seed, run both seat assignments. Keep the target policy fixed while swapping its seat. This is 48 baseline matches; each scheduled row remains in the raw report even if its trajectory duplicates another row.
- Capture checkpoints only at the start of policy ticks `30`, `300`, and `900`, before that tick's `PRE_POLICY` update. If the baseline episode already terminated, record that checkpoint as `preterminal`; do not substitute a later tick or a different seed.
- Each eligible checkpoint is tested on the target policy's seat for the three predeclared axes below. Budget ceiling: 3,000,000 calls to `step_tick` across the 48 baseline matches and both arms of every possible probe pair. The fixed schedule is not shortened or extended based on outcomes. If the ceiling is reached, unrun rows are `not_run_budget`, the study is incomplete, and affected conclusions are inconclusive.
- No exploratory pilot, seed replacement, alternate checkpoint, or candidate action is added after viewing FOR-21 outcomes. Any follow-up requires a new protocol/version.

## 2. Checkpoint, arms, and action generation

At each checkpoint, deep-clone both `SimState` and `RunState` before the policy phase. Record `state_hash` and `checkpoint_hash`; control and treatment must start from clones with both hashes exactly equal. The stored checkpoint includes both colonies, intel, pending transit, policy-window statistics, every RNG stream, and run outcome. The actual `step_tick` calls perform the real `begin_policy_phase`, observation construction, mask construction, validation, and action application.

Build the target side's control decision from that checkpoint's own `ObservationV2`, `ActionMaskV2`, and policy RNG stream using the existing baseline policy and decoder. Revalidate the action with `validate_action`. A one-decision adapter is used in both arms: it executes the ordinary baseline decision at the checkpoint (consuming exactly the same decision RNG draws) and returns either that control action or the predeclared, revalidated candidate. This keeps the two RNG streams aligned through the intervention. The opponent always uses its ordinary baseline controller. From the next policy decision onward, both sides in both arms use the unchanged baseline controllers.

Candidate selection reads only the target side's observation, mask, own legal control action, and own-colony eligibility facts. It never reads the hidden opposing colony or future rollouts. It is fixed by the rules below, not selected for expected outcome. If a candidate is unavailable, illegal, or identical, record `ineligible` with a reason; do not substitute another candidate.

| Axis | Single candidate rule | Eligibility and one-axis check |
| --- | --- | --- |
| `lay_order` | Set to `worker` if the control order is `none` or `soldier`; otherwise set to `none`. | The target queen is alive, the candidate is legal under the checkpoint mask, and the action differs only in `lay_order`. Lack of food or a due lay is recorded as downstream context, not used to select the candidate. |
| `forage_share_q` | Replace the six shares with `(65535, 0, 0, 0, 0, 0)` (home only). | The target has at least one adult worker, the control shares are not already home-only, and the candidate passes the shared validator. Only `forage_share_q` may differ. |
| `army_choice` | Set `army_choice=6` (raid enemy home), retaining the control `avoid_center` value. | The target has at least one soldier (stationed or in transit), choice 6 is reachable and legal in the checkpoint mask, and that same `avoid_center` value is legal for choice 6. If satisfying the mask would require changing `avoid_center`, record `ineligible`; only `army_choice` may differ. |

For every probe, compare the complete serialized `ActionV2` values. The declared axis must be the only changed field. Both arm actions must be accepted by `validate_action` against the exact target observation and mask. A changed `OrderEvent` confirms only that the command was issued; it is not evidence of a world effect.

## 3. Outcomes and observation horizons

The primary endpoint is the target side's true terminal score (`1`, `0.5`, or `0`) from the engine's `EpisodeOutcome`. Run each arm through the real terminal rule. Queen death, economic collapse, and the engine's actual `T_max` timeout are reported separately. A harness stop before an engine terminal is `unknown/censored`, never a draw or a fabricated score. Terminal results before an intermediate horizon are retained as early terminals; the unavailable later horizon is marked censored.

At each probe, measure the first states after **30** and **150 completed ticks** from the intervention tick, when reachable, plus the terminal state. Record actual endpoint tick and completed-tick count. At each endpoint capture:

- target and opponent food, adult workers/soldiers, brood, queen HP, target-side starvation debt and upkeep remainder;
- food in every resource node, expressed in the target side's local node frame;
- adults by caste and local node, including in-transit counts/destinations and soldiers at home/center/enemy home;
- events since intervention: orders, retargets, departures/arrivals, harvest, upkeep, lay/hatch, starvation/death, combat/wounds/queen damage, terminal; keep event counts and a deterministic event digest, not full per-tick replay logs;
- `state_hash` and `checkpoint_hash` at both intermediate horizons and terminal (or last available state), terminal reason/tick/completion flag, and cutoff/censoring status.

For each pair, report treatment-minus-control deltas for the primary score and every available endpoint metric. Report order-event changes separately from physical-state changes. A physical effect is a difference in at least one predeclared game-state measure above or terminal outcome; event-only differences do not qualify. An empty action is one with a validated one-axis action and changed order event but no physical-state or terminal difference through the terminal state. If the full arm is censored, emptiness is unresolved.

## 4. MDE, unique units, and uncertainty

- The minimally useful effect (MDE) for the primary endpoint is an absolute paired score difference of **0.10** for the target side.
- The paired comparison is within one checkpoint state and one axis. The independent resampling unit is a **unique full baseline trajectory**, not a seed string, row, checkpoint, or seat-swap label. Give each baseline run a trajectory digest over its per-tick state hashes and canonical event stream; include the random policy stream state when `random:v1` is present. Repeated seed rows with the same digest stay visible but count once in unique-trajectory summaries and uncertainty calculations. Checkpoints from one trajectory are clustered together.
- Also report the number of distinct pre-intervention `state_hash` values per axis. Mirrored seats and repeated trajectories are never counted as independent evidence merely because their row IDs differ.
- Summarize each unique trajectory's eligible checkpoint deltas first. Estimate an axis mean by giving each matchup equal weight, then each unique trajectory within that matchup equal weight. Do not select matchups or checkpoints based on observed effects.
- Show a two-sided 95% percentile cluster-bootstrap interval only when that axis has at least **10 unique eligible trajectory digests**, at least **12 distinct eligible pre-intervention state hashes**, and at least two unique trajectories in each of the three matchups. For each axis, seed a PCG32 bootstrap stream with the little-endian u64 from the first 8 bytes of `SHA256(UTF8("formicmind.gate1-causal/v1|bootstrap|" + axis))`. For each of 10,000 replicates, draw with replacement within each matchup exactly as many trajectory clusters as that matchup has; then compute the same equal-matchup/equal-trajectory mean. Sort replicate means and take nearest-rank 2.5th and 97.5th percentiles. Record seed, replicate count, and interval endpoints in the manifest. Otherwise show exact paired values/ranges and label inferential uncertainty `not estimable`; do not substitute an episode-level interval or p-value.
- If any of those minimum counts is not met, the primary conclusion for that axis is `INCONCLUSIVE`. With an interval, label a favorable signal only if its lower bound is greater than `+0.10`, a harmful signal only if its upper bound is less than `-0.10`, and a practically small effect only if the entire interval lies within `[-0.10,+0.10]`. All other cases are `INCONCLUSIVE`. These are diagnostic labels, not a Gate 1 decision.
- Intermediate outcomes are descriptive paired deltas only; no multiple-outcome significance claims. Report missing/censored values as missing with their reason, never as zero.

## 5. Empty, dominant, and free-action flags

Report, by axis and matchup, eligible/ineligible counts, one-axis validation count, changed-order count, physical-effect count, empty-action count, and unresolved censored count. Flag a **possible dominant/free action for review** only if all minimum unique-count conditions above hold, the primary interval is wholly beyond the `+0.10` MDE, every unique-trajectory primary delta is nonnegative, and no predeclared terminal, queen-HP, adult-survival, starvation, or target-food endpoint shows a treatment loss. The flag is a follow-up risk, not proof of free value or authorization to alter the game. Do not claim dominance from a logit, order event, single match, or a favorable intermediate metric alone.

## 6. Integrity, verification, and stop conditions

Every row binds to protocol SHA-256, source commit, SHA-256 fingerprint of all Python implementation files recursively under `src/formicmind/`, config hash/canonical config, rules/balance/observation/action/decoder/RNG/serialization versions, policy IDs/parameter digests/stream IDs, seed, matchup, seat assignment, target, checkpoint tick, eligibility reason, and both arm hashes. JSONL rows have a stable order and canonical JSON encoding; aggregates exclude wall-clock metadata.

The unique baseline trajectory digest is SHA-256 over canonical JSON containing, in tick order, each tick's start `state_hash`, that tick's ordered `Event.to_canonical()` list, and end `state_hash`; at each policy tick also include the pre/post canonical state of `policy:random-v1:0` if that stream is registered. Do not include root seed or wall-clock data. This groups repeated deterministic trajectories but keeps distinct random-policy streams separate.

`verify --input` must check schema, protocol/config/policy/source provenance, recompute each prescribed baseline and control/treatment continuation through `step_tick`, and compare action, eligibility, horizon/terminal metrics, and hashes to the saved rows before recomputing the aggregates. Re-aggregating saved values alone is not verification. At intervention, both adapters must recompute the same control action as the checkpoint preflight; disagreement is a verification failure. Cross-commit replay is supported only when the protocol SHA, config and policy descriptors, and recursive `src/formicmind/` Python source fingerprint still match; a changed implementation fingerprint is incompatible and must fail strict verification. Documentation/results-only commits do not invalidate verification.

Stop the run and mark it invalid/incomplete if any pre-intervention arm hashes differ, the control replay is nondeterministic, an action is invalid or changes an undeclared axis, a checkpoint is not at a scheduled pre-policy tick, source/config/policy provenance changes, the `step_tick` budget is exceeded, or a required hash/event digest disagrees on deterministic replay. Do not repair such a run by editing outputs or changing the schedule. The run may be fixed and repeated only under the same still-unviewed schedule and protocol, with the invalid attempt documented.

## 7. Interpretation and exclusions

Negative, zero, empty, or inconclusive outcomes are valid results. This is an open diagnostic sample and cannot establish general strategy value, balance, map/context robustness, or Gate 1 completion. FOR-21 covers only this single-state causal-intervention slice. Cross-map context maps, open-loop versus closed-loop comparison, anti-stalling, parameter/model optimization, and sealed evaluation remain separate work. Gate 0 single-core Windows performance remains `NOT MEASURED` unless separately measured on the required target machine. No gameplay source, physical rules, balance, existing baseline bot, action/observation/replay contract, or golden result may be changed for this study.
