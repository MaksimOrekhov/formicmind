# L0 evaluation and benchmark CLI

The `sim` commands below run the existing Python L0 simulator through its single `step_tick` implementation. They
use the existing scripted smoke fixtures, or an empty scripted controller for a technical workload. They add no
game rules or strategic bots. Replay v1 recording and verification remain available through `sim replay --verify`.

Install the development dependencies from the repository root, then run the CLI from PowerShell:

```powershell
python -m pip install -e ".[dev]"
sim run --seed 20261103 --a scripted --b scripted --scenario raid_win --json-out out/run.json
sim eval --episodes 20 --seed-start 20261103 --a scripted --b scripted --scenario raid_win --paired `
  --json-out out/eval.json --manifest-out out/eval.manifest.json
sim benchmark --runs 5 --ticks-per-run 100000 --warmup-ticks 10000 --seed-start 20261103 `
  --a scripted --b scripted --scenario timeout_draw --pin-core 0 `
  --json-out out/benchmark.json --manifest-out out/benchmark.manifest.json
```

Use a logical CPU index present in the process's allowed mask for `--pin-core`. If it is omitted, the benchmark
tries to pin to the first allowed core. Affinity is best-effort and is restored after warmup, samples, and optional
diagnostics, including when an exception escapes. The JSON records whether pinning and restoration succeeded.
`sim benchmark` returns a successful run even when affinity is unsupported; such a run is marked `NOT MEASURED`
for Windows single-core comparison. CI checks correctness and does not count as a performance measurement.

## Commands and result semantics

- `sim run` executes one episode and reports outcome, terminal tick, completed state tick, ticks executed, score,
  final population/resources, and config/state/checkpoint hashes. `--max-ticks` is a harness cutoff; a truncated
  episode has no winner, reason, or score.
- `sim eval` assigns seed `seed_start + i` to episode `i`. With `--paired`, each seed is run twice with the
  existing scripted A/B controllers swapped between seats. JSON contains per-episode rows and aggregates. Only
  terminal episodes contribute wins/draws/reasons and finished scores; cutoffs are counted separately.
- `sim benchmark` resets and runs exact calls to `step_tick`, counting an incomplete mid-tick terminal as one
  executed call. When a sample budget ends before a game ends, that is recorded as a benchmark boundary cutoff,
  not as a game result. The benchmark releases each tick result immediately, keeps no replay or whole-match event
  list, and omits state/checkpoint hashing from timed work. It still includes reset, simulation, and compact
  episode-boundary bookkeeping in wall time.

Benchmark seeds use `seed_start + episode_index` and the same schedule is replayed for warmup and every sample.
The recommended baseline is 10,000 warmup ticks followed by five 100,000-tick samples. Throughput is measured
with `perf_counter_ns`. p10 uses nearest rank (`ceil(0.10 * n)`, the minimum of five samples); p50 is the median
per-sample ticks/second. Raw elapsed nanoseconds, tick counts, outcomes, and rates are in the JSON result.
`--phase-timing-ticks N` runs a separate per-phase diagnostic after the main samples. Its callbacks and clock reads
add overhead; the diagnostic is labeled and excluded from throughput.

The 5,000 ticks/second figure is a candidate single-core comparison threshold, not a Gate 0 pass on its own. To
report a Windows single-core comparison, the result must identify Windows and a processor and confirm that
single-core affinity was applied and restored. Results without those fields are marked `NOT MEASURED`; do not
compare macOS/Linux CI or a short smoke run to the Windows target.

## Reproducible manifests

Each result JSON embeds a manifest and its canonical SHA-256. `--manifest-out` writes that manifest separately.
It records the command parameters and seed schedule, default config hash and timing limits, rules/balance/RNG/
serialization/Replay versions, scripted controller identifiers and script hashes, and current Git commit/dirty
state. Replay it with the same command, for example:

```powershell
sim eval --manifest out/eval.manifest.json --json-out out/eval-repeat.json
```

Manifest replay rejects mismatched simulator contracts, controller definitions, Git state, or execution options.
The output result contains the same deterministic episode rows and manifest digest; benchmark wall-clock samples
are expected to differ. Keep the manifest and result JSON together when archiving an experiment.

The supported example scenarios are `raid_win`, `mutual_raid_draw`, `collapse`, and `timeout_draw`, defined by
the existing functional fixtures. They are correctness workloads, not strategy or balance evidence. The `technical`
scenario uses empty scripted actions and is useful for controlled harness checks.
