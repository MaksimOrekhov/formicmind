# Gate 1B cross-map context study — findings

**Status:** complete open diagnostic. Gate 0 and Gate 1 are not passed.  
**Study label:** DESCRIPTIVE_CANDIDATES_PRESENT  
**Frozen protocol:** first protocol-only commit f188558; protocol SHA-256 b0eb1685a2d0e024e915f42d7833e6f27fc9be4add4c26ceac921b77f07edff3.  
**Harness/source commit:** 4b1bf18dfdfb8585a5a5045c4b11d742b7d6cf9f, based on origin/main 778e518eb432af4adcc56e59664146fcce8e0e09.  
**Python source fingerprint:** 5410221b382a3840c269d8e13177164a9a3cbba49904b8ebff5f247191bcc6b4.

No game rules, balance, topology defaults, physics, terminal conditions, ObservationV2, ActionV2, decoder, or FOR-19 policy code changed. R1 is a preregistered diagnostic resource scenario, not an approved economy.

## Schedule and integrity

The complete fixed schedule ran: 6 maps × 10 policy pairs × 6 open seeds × 2 seats = **720/720 terminal matches**. There were no technical failures or harness cutoffs. The engine recorded terminal outcomes for every match. The run used **401,898 step_tick calls** of the 2,160,000-call ceiling.

Seat swaps were averaged inside each pair block. The 180 scheduled cross-map seed blocks reduced to **90 distinct paired-trajectory blocks** after collapsing mirror seats and repeated deterministic trajectories. Each of the four matchups containing random:v1 had six unique paired blocks per terrain profile. Each of the six all-fixed-policy matchups had one unique paired block per profile; changing the seed did not create six independent runs. The raw file still preserves all 720 scheduled rows. The trace accounting found 63 distinct random-policy RNG-stream trajectories across its scheduled appearances.

Strict verify re-executed the full 720-match matrix and reproduced all row hashes and aggregates:

* matches.jsonl SHA-256: cb895bfe85d2aa641c30a05ee300f35e63e9b7b90e580e5f8f1a59c20859b7b4
* summary.json SHA-256: 4c95300506662a03e79beab41e68eceb30c848f95d5b9c9f4fa2400660559944
* Verify result: 720 matches, 401,898 step_tick calls, status verified.

## Primary context results

The equally weighted R1-minus-R0 match-score contrast crossed the 0.15 MDE and had the same direction on at least two terrain profiles for one pair: random-raider, Δ = **1/6** (0.167), with profile deltas T0 = 1/6, T1 = 1/3, T2 = 0. This is a CONTEXT_SENSITIVITY_CANDIDATE for this fixed policy pair and these fixed maps.

No evaluable pair crossed the preregistered pairwise rank-reversal margins. The six all-fixed-policy pairs are INCONCLUSIVE for both labels because each terrain profile had only one distinct paired trajectory, below the predeclared two-block minimum. Their descriptive point estimates remain visible below.

| Matchup | Distinct blocks by T0/T1/T2 | Equal-weight context Δ | Context label | Pairwise rank-reversal label |
| --- | ---: | ---: | --- | --- |
| random-turtle | 6 / 6 / 6 | 1/18 | NO_CONTEXT_SIGNAL_IN_SCHEDULE | NO_CONTEXT_SIGNAL_IN_SCHEDULE |
| random-rush | 6 / 6 / 6 | 1/9 | NO_CONTEXT_SIGNAL_IN_SCHEDULE | NO_CONTEXT_SIGNAL_IN_SCHEDULE |
| random-balanced | 6 / 6 / 6 | 0 | NO_CONTEXT_SIGNAL_IN_SCHEDULE | NO_CONTEXT_SIGNAL_IN_SCHEDULE |
| random-raider | 6 / 6 / 6 | 1/6 | CONTEXT_SENSITIVITY_CANDIDATE | NO_CONTEXT_SIGNAL_IN_SCHEDULE |
| turtle-rush | 1 / 1 / 1 | -1 | INCONCLUSIVE | INCONCLUSIVE |
| turtle-balanced | 1 / 1 / 1 | -1/3 | INCONCLUSIVE | INCONCLUSIVE |
| turtle-raider | 1 / 1 / 1 | -1/2 | INCONCLUSIVE | INCONCLUSIVE |
| rush-balanced | 1 / 1 / 1 | 0 | INCONCLUSIVE | INCONCLUSIVE |
| rush-raider | 1 / 1 / 1 | -1/3 | INCONCLUSIVE | INCONCLUSIVE |
| balanced-raider | 1 / 1 / 1 | 0 | INCONCLUSIVE | INCONCLUSIVE |

The score matrix below gives the first-listed policy's mean terminal score (1 win, 1/2 engine draw, 0 loss), with unique paired blocks equally weighted. Columns abbreviate the matchup names: r=random, t=turtle, ru=rush, b=balanced, ra=raider.

| Map | r-t | r-ru | r-b | r-ra | t-ru | t-b | t-ra | ru-b | ru-ra | b-ra |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R0-T0 | 1/3 | 1/6 | 1/6 | 1/3 | 1 | 1/2 | 1/2 | 0 | 1 | 0 |
| R0-T1 | 1/2 | 1/6 | 1/3 | 1/6 | 1 | 1/2 | 1/2 | 0 | 1 | 0 |
| R0-T2 | 1/3 | 1/3 | 0 | 1/6 | 1 | 1/2 | 1/2 | 0 | 1 | 0 |
| R1-T0 | 1/2 | 1/3 | 1/6 | 1/2 | 0 | 0 | 0 | 0 | 1 | 0 |
| R1-T1 | 1/2 | 1/3 | 1/6 | 1/2 | 0 | 1/2 | 0 | 0 | 0 | 0 |
| R1-T2 | 1/3 | 1/3 | 1/6 | 1/6 | 0 | 0 | 0 | 0 | 1 | 0 |

With equal weight for each of the four opponents and each terrain profile, the descriptive policy ordering changes from R0 to R1:

| Family | Rank 1 | Rank 2 | Rank 3 | Rank 4 | Rank 5 |
| --- | --- | --- | --- | --- | --- |
| R0 | turtle (47/72) | balanced (7/12) | raider (41/72) | rush (4/9) | random (1/4) |
| R1 | raider (53/72) | balanced (2/3) | rush (7/12) | random (1/3) | turtle (13/72) |

This is an aggregate ranking rearrangement on the six fixed maps. It is not a preregistered pairwise reversal candidate: no evaluable P-Q pair had a family margin of at least +0.10 and the opposite-family margin at most -0.10. The point estimates for the all-fixed-policy pairs remain inconclusive under the independent-trajectory rule.

## Terminal and anti-degeneracy diagnostics

There were **84 engine timeout draws** (seven matchup × map groups at 12/12 matches each); the remaining 636 matches ended by queen death or economic collapse. Six timeout groups were Turtle-Balanced or Turtle-Raider on R0 maps: all 12 matches timed out at each of T0, T1, and T2. This reproduces the qualitative FOR-19 Turtle-Raider timeout pattern on R0. On R1, Turtle-Raider had 12/12 queen-death outcomes at each route profile. These are deterministic single-trajectory descriptions for the fixed policies; the resource-family difference does not meet the preregistered independent-block minimum for a label.

Predeclared review flags, using 12 scheduled matches per matchup × map:

* Timeout concentration: 7 groups.
* Extended zero-combat: 0 groups.
* Food stockpile with live armies at timeout: 0 groups.
* Repetitive policy orders: 16 groups.

These flags identify review targets, not balance defects or dominant actions. No win-rate cap was used.

## Limits and interpretation

The maps are a handpicked fixed set, not a random sample from a map population; no 95% confidence intervals or p-values are reported. Half of the 180 scheduled paired seed blocks duplicate another paired trajectory, and all-fixed-policy matchups reduce to one unique block per terrain. The sensitivity candidate applies only to the fixed random:v1 versus raider:v1 comparison. The aggregate rank change is descriptive and does not establish a general strategy ranking.

R1 does not approve a new resource balance. The study does not test open-loop versus closed-loop feedback, anti-stalling beyond the listed probes, bot quality, or sealed generalization. Gate 0 Windows single-core performance remains NOT MEASURED. No Gate 0 or Gate 1 pass is claimed.

## Reproduction and checks

Use Python 3.12. For a fresh run, choose an output path that does not already exist:

    PYTHONPATH=src python -m formicmind.gate1_context run --out results/gate1_context-reproduction
    PYTHONPATH=src python -m formicmind.gate1_context verify --input results/gate1_context

Local checks on the harness revision: full Python suite **851 passed**; viewer production build passed; Vitest **18 passed**. Windows GitHub Actions is reported on the FOR-22 PR.

Raw match rows and machine-readable aggregates are in results/gate1_context/; the frozen design is docs/gate1_context_protocol.md.
