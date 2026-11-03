# FOR-20 economic feasibility pilot findings

## Run identity and validity

* Protocol first commit: `e1573f8cafc2e9e7d093a4c5ae573760ded954fd`; protocol SHA-256: `57f33e613599542d89f8a5a6ea8c5d0a33f29022564b056fb2c0a93bd87d93d0`.
* Harness commit used for the matrix: `9a3e9767bcad2ce38a775cd807f18f55c97bea48`, based on `origin/main` `b7ab8d743b961e6d1a18b9e63951acdcb78537f3`.
* Commands:

  ```bash
  PYTHONPATH=src /opt/homebrew/bin/python3.12 -m formicmind.economy_pilot run --out results/economy_pilot
  PYTHONPATH=src /opt/homebrew/bin/python3.12 -m formicmind.economy_pilot verify --input results/economy_pilot
  ```

The locked matrix contains 144 run rows, 48 conditions, and 3 open seeds per condition. The initial run and verification replay matched for all rows, events, checkpoint state hashes, and summaries. Each condition had one unique state trajectory; its other two seed rows duplicated it (96 duplicate rows total), so the three seeds are determinism checks, not independent samples. All runs reached the pilot horizon at `t=1000`; none reached a game terminal. All 100-tick checkpoints were present. Default topology was retained; the fallback was not used. Every tick passed full-mode invariant checks, mirror symmetry, and isolation checks; there were no combat, queen-damage, inter-colony contact, or data-quality failures.

The horizon is a pilot cutoff, not a timeout or match result. Both sides were symmetric; the metrics below show one colony per unique trajectory.

## Existing candidate (`eggs_per_lay=1`)

The predeclared engineering screen found 4 of 12 lay-active conditions viable candidates: 2 of 4 worker-first and 2 of 4 soldier-focused conditions. Both viable worker-first conditions used a single local source (one default, one scarcity); both viable soldier-focused conditions used the default resource profile (single and split forage). Alternating had no viable candidate. The no-lay control retained at least one adult worker through the horizon in all 4 conditions; it did not reproduce and is classified as a survival control only.

| Lay/forage/profile | Adults W/S at 1000 | Harvest | Egg spend / eggs laid | Food-rejected / eligible lay ticks | Hatched W/S | Starvation deaths; debt at 1000 | Viability screen |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| worker-first / single / default | 16/3 | 7,984 | 40 / 4 | 960/964 | 4/0 | 1; 0 | candidate, with rejection red flag |
| worker-first / single / scarcity | 12/0 | 4,026 | 20 / 2 | 980/982 | 2/0 | 6; 0 | candidate, with rejection red flag |
| worker-first / split / default | 42/0 | 15,820 | 370 / 37 | 630/667 | 37/0 | 11; 3 | no: late starvation/debt |
| worker-first / split / scarcity | 18/2 | 8,042 | 60 / 6 | 940/946 | 6/0 | 2; 1 | no: late starvation/debt |
| soldier-focused / single / default | 12/5 | 7,984 | 40 / 2 | 980/982 | 0/2 | 1; 0 | candidate, with rejection red flag |
| soldier-focused / split / default | 12/10 | 11,970 | 200 / 10 | 900/910 | 0/10 | 4; 0 | candidate, with rejection red flag |
| soldier-focused / single / scarcity | 12/0 | 4,026 | 40 / 2 | 980/982 | 0/2 | 6; 3 | no: late starvation/debt |
| soldier-focused / split / scarcity | 12/5 | 8,042 | 80 / 4 | 960/964 | 0/4 | 3; 3 | no: late starvation/debt |

The screen is deliberately not an approval: every active condition above triggered the preregistered food-rejection flag. An eligible food-rejected lay does not spend the cooldown in the existing engine, so repeated underfunded attempts emit repeated `LayEvent(laid=false)` events. Across the matrix, 36 of 48 unique conditions met the ≥25% refusal red flag; 24 of 48 met the late-starvation/debt red flag. No worker-reduction or total-local-food-exhaustion flag occurred. The counts in `summary.json` are side-level rows (mirrored sides and duplicate seeds included), not independent observations.

For the no-lay controls at `eggs_per_lay=1`, adult workers were 12 at `t=1000` in all four resource/forage combinations, with zero egg spend or hatch. Under default split forage, the warehouse held 4,030 food and local source stocks held 140 more. Under scarcity single-source forage, four soldiers starved, leaving 12 workers and 64 accessible food; this remains a survival-only control, not an active-regime viability result.

## Effect of batch size

Batch size effects were not monotone. For worker-first, single-source/default, the final adult population and total successful worker eggs were the same for `eggs_per_lay=1/2/4` (16 workers, 3 soldiers, 4 worker hatchlings, 40 egg food). Successful batch count fell from 4 to 2 to 1, while food-rejected eligible lay ticks rose from `960/964` to `980/982` to `990/991`.

With worker-first split-source/default, successful worker eggs fell `37 → 34 → 24`, end workers `42 → 42 → 36`, and starvation deaths `11 → 8 → 1` as the batch increased from 1 to 4. Debt at `t=1000` was `3 → 3 → 0`, but each condition still failed the viability screen because of starvation late in the horizon. With worker-first single-source/scarcity, the same change produced `2 → 4 → 4` worker hatchlings, `6 → 8 → 8` starvation deaths, and end debt `0 → 4 → 3`; only the one-egg condition met the screen. The split-source/scarcity conditions ended at 18/2, 20/1, and 20/1 workers/soldiers, with debt `1`, `3`, and `2` respectively.

Soldier-focused/default split forage hatched `10`, `8`, and `8` soldiers for batch sizes 1, 2, and 4. At batch size 4 with single-source forage, the 80-food batch cost was never paid; this condition had no hatch. No-lay controls were unchanged across batch sizes, as expected. The full values for every condition and checkpoint are in the raw files.

Split forage generally increased harvest, but also supported faster worker growth and greater upkeep pressure. For example, worker-first/default harvest at `t=1000` was 15,820–15,880 under split forage versus 7,984 under single-source forage; the split-source worker population reached 36–42 and incurred starvation, while single-source ended at 16 workers. Scarcity reduced harvest roughly by half and produced low local stocks and more late starvation in active reproduction conditions.

## FOR-19 diagnostic comparison and recommendation

FOR-19 recorded `turtle–raider` as 10 timeout draws at `T_max=3000`, with two unique seat-assignment trajectories. This pilot found worker-count plateaus from `t=300` to `t=600` in the isolated scenarios, and material resource/upkeep differences by forage and lay regime. It stopped at `t=1000`, had no opponent, and produced no game terminal. It therefore cannot identify the cause of the FOR-19 timeouts; stagnation remains one hypothesis among several.

The isolated results show a measurable feeding-versus-reproduction trade-off and different worker/soldier outcomes. Defense and combat were intentionally absent, so the pilot cannot assess the full reproduction/feeding/defense choice or declare Gate 1 ready. Recommendation: use these risks to design the preregistered Gate 1 diagnostic probes while keeping the candidate rules frozen. This pilot does not justify a balance or physics change, a population cap, or a strength claim. No candidate balance edits are proposed.

## Artifacts

* `docs/economy_pilot_protocol.md` — frozen protocol.
* `results/economy_pilot/manifest.json` — source/protocol identity, full configs, hashes, and fixed matrix.
* `results/economy_pilot/runs.json` — 144 raw run rows, per-side 100-tick checkpoints, cumulative metrics, and trajectory/event/state/checkpoint hashes.
* `results/economy_pilot/summary.json` — 48 condition summaries with unique-trajectory grouping and preregistered red flags.
