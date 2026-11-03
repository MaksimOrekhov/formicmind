# FOR-24 protocol deviation and exploratory classification

**Classification:** reproducible exploratory diagnosis with a documented protocol deviation. The corrected fixed-schedule artifacts are strictly replayable, but this is not a clean confirmatory result: the complete first output was inspected before the deduplication correction.

## Timeline and provenance

| Stage | Commit / record | What is known |
| --- | --- | --- |
| Frozen protocol | `5180ba8de9249e2c4bdbbbcaf1f6bbd6a2355169` | Committed alone before implementation; SHA-256 `0d54603c90526f7d51705f5fcf7c70290d854324be5eb73297932810ec1f3b5d`. |
| First implementation and run | `07e0e135cd83493bd46de481de3cefe2a11e44be` | A complete first output was produced and inspected. The fixed schedule called for 288 baselines and 384 probe descriptors (192 eligible at the corrected run). The original output's row counts and artifact hashes cannot be independently checked from this checkout. |
| Deduplication correction | `c75ce28185431d7f6ff7e88fae91af62eb35044c` | Removed the raw `control_suffix_sha256` from the probe-context key and keyed contexts by the frozen normalized checkpoint, map, matchup, target policy, checkpoint tick, and intervention. The fixed schedule was rerun with this correction. |
| Corrected result documentation | `4177e915c4de9c4318b0f2e2a846a87c8d98cada` | Publishes the corrected raw rows, summary, manifest, and findings. The corrected artifacts are independently replayed by the strict verifier. |

## Deviation and its effect

The initial probe deduplication key included `control_suffix_sha256`, a digest of the baseline suffix after a checkpoint. That field is not part of the preregistered context identity. Deterministic mirror/seed repeats with the same normalized checkpoint could therefore be counted as separate contexts. The issue was identified after the complete first output had been viewed. The correction followed that inspection, so the protocol's restart condition for a still-unviewed schedule no longer applied. This is the protocol deviation; it is not hidden by the corrected rerun.

The code change was limited to the derived unique-context digest. It did not change the schedule, seed/map/config inputs, baseline policies, intervention actions, engine, state transitions, event collection, or outcome classification. Consequently the per-match physical trajectories, terminal reasons, and scores were not changed by the correction; the context hashes and aggregates that depend on grouping were recalculated. The first attempt is unavailable for a row-by-row comparison here, so this conclusion follows from the scoped code diff rather than an independent artifact comparison.

The first attempt was reported as preserved under `results/gate1_stalling_attempt1/`, but that directory and its manifest are absent from this checkout and are not part of the PR. The original attempt's artifact SHA-256 values and source fingerprint therefore cannot be published or checked. Its Git source commit above is available; the old output is not publicly reproducible from this repository. Do not treat the corrected strict replay as a replay of the missing first attempt.

## Interpretation

Use the final corrected artifacts as a **reproducible exploratory** fixed-scenario diagnosis. Their 288 baseline rows, 384 planned probes, corrected unique-context grouping, and associated descriptive findings are preserved and strict-verifiable. Because the first complete output was inspected before correcting the grouping, the study must not be described as cleanly confirmatory or as an unqualified preregistered result. The frozen protocol itself, its schedule, thresholds, policies, and game rules remain unchanged.
