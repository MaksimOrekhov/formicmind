# L0 strategic baselines (FOR-19)

These five manual policies are fixed design profiles for repeatable diagnostics. They are not trained, tuned on the diagnostic seeds, validated as strong agents, or evidence that the candidate balance is good. All use `PolicyController → decide → decode_action_logits → validate_action` and the existing `step_tick` loop.

Each policy sees only the local `ObservationV2`, its `ActionMaskV2`, and its own PCG32 stream. Public topology is used for route legality. Food and enemy estimates are discounted by `known/current/age`; unknown food contributes only a prior from public capacity and regeneration. Policy names are local-frame based and require no side A/B special cases.

| Selector | Stable policy ID | Fixed profile and parameters | Known limitation |
| --- | --- | --- | --- |
| `random` | `random:v1` | PCG32 samples lay/army logits and a 1–3 node forage support. | “Random” means repeatable stochastic logits through the shared decoder; the decoded action distribution is not mathematically uniform. |
| `turtle` | `turtle:v1` | Defends a known threat at strength 0.5; prefers workers unless a visible threat justifies soldiers; stops laying below 55 food and lays soldiers only at 80+ food. Normally forages one target, two below 80 food; prefers avoiding the center when the mask permits. | Has no dedicated wall/build action; defense is an army destination choice. |
| `rush` | `rush:v1` | Builds soldiers until at least 4 or 42% of workers, then raids when no known defense target is urgent. Defends known threats at strength 1.0; stops laying below 35 food or starvation-debt signal 0.12. Forage support 1; prefers the direct route. | Cannot command a launch tick independently of the existing policy interval or change combat rules. |
| `balanced` | `balanced:v1` | Targets at least 6 soldiers and 28% of workers before raiding; defense orders prioritize local home nodes 0–2. A ready army raids without needing queen intel. It lays toward raid readiness when food is at least 45, unless a nearby threat calls for soldiers sooner. Food below 65 or debt signal above 0.08 broadens forage support to 3; severe debt above 0.15 or food below 30 stops laying. | Readiness and defense ranges are fixed heuristics, not an optimized economy or a “strong baseline” claim. |
| `raider` | `raider:v1` | Targets at least 5 soldiers and 40% of workers. Before readiness it intercepts currently visible enemy soldiers; once ready it raids unless current enemy soldiers are at home or local node 1. Enemy workers and sightings farther forward do not block a raid. It does not need queen intel; forage support is 2. | ActionV2 has no waypoint or squad-micro command; movement pressure uses the nearest legal army destination behavior. |

The manifests record each full ID, stream ID, replicate, complete fixed parameter object, and SHA-256 parameter digest. The role stream uses a validated slug (`policy:random-v1:0`) because the shared RNG stream grammar intentionally excludes `:` inside its policy-id segment; the manifest retains the canonical display ID (`random:v1`). If identical policy versions occupy both seats, they receive replicates 0 and 1. A paired seat swap moves the same controller descriptor and stream to the other side.

## Run and evaluate

```powershell
sim run --seed 20261103 --a balanced --b rush
sim eval --episodes 20 --seed-start 20261103 --a balanced --b rush --paired --json-out results/balanced-rush.json
```

Scripted smoke fixtures remain available only with `--a scripted --b scripted --scenario <fixture>`. Mixing a scripted seat with a policy or combining a policy with `--scenario` is an explicit usage error. `--manifest-out` writes a strict reproducible manifest; replay it with `sim eval --manifest PATH`. Legacy scripted manifests keep their existing descriptors and validation shape.

## Replay a single match

Batch `sim eval` does not write replay files or retain tick events. For a match intended for inspection, use the separate single-episode path:

```powershell
sim run --seed 20261103 --a balanced --b rush --replay-out results/balanced-rush.jsonl
sim replay --verify results/balanced-rush.jsonl
```

The file is Replay v1 JSONL written by the existing `record_episode`/`ReplayWriter`; no wire-format change is made. To inspect it in the desktop viewer, start `viewer` with `npm run dev`, open the shown local page, and import the JSONL through its local replay import control. The viewer is read-only and does not replace Python replay verification.

## Open diagnostic schedule

Before inspecting outcomes, the diagnostic seed schedule is fixed at seeds `20261103` through `20261107`, with both A/B seat assignments for each seed:

```powershell
sim eval --episodes 5 --seed-start 20261103 --a balanced --b rush --paired --json-out results/diag-balanced-rush.json
sim eval --episodes 5 --seed-start 20261103 --a turtle --b raider --paired --json-out results/diag-turtle-raider.json
sim eval --episodes 5 --seed-start 20261103 --a random --b balanced --paired --json-out results/diag-random-balanced.json
```

These are open engineering diagnostics for checking determinism, legal behavior, response to visible information, and seat assignment. They are not sealed test seeds, a tuning set, a statistical comparison, or a balance/strength result. The refreshed runs produced balanced–rush 5–5 queen-death wins, turtle–raider 10 timeout draws, and random–balanced 5–5 wins; do not interpret these counts as strength evidence. Balanced–rush and turtle–raider each had one unique `state_hash`/outcome/length trajectory per seat assignment across the five roots (two unique trajectories overall); their repeated rows verify deterministic replay and mirrored seats, not ten independent situations. Random–balanced had ten distinct trajectories. Root seeds may still change checkpoint hashes even when the game trajectory repeats.

The paired command outputs, manifests, aggregates and every result row with final hashes are preserved in [`results/for19-diagnostics.json`](../results/for19-diagnostics.json) for review and exact replay.

FOR-18 Gate 0 single-core Windows performance remains **NOT MEASURED**. Functional tests and these match diagnostics do not satisfy the 10k warmup + 5×100k tick hardware benchmark.
