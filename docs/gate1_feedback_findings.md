# Gate 1C feedback ablation findings

**Issue:** FOR-23  
**Protocol:** `formicmind.gate1-feedback/v1`  
**Base:** `af4d6ff5364e29add002aab504d2609b3d797fb8`  
**Protocol SHA-256:** `98b166ad5fd31451990f7aaf49a7fc23c34d5ba10a4193a37d5d992d244eca4b`  
**Source commit:** `f20ed04abc931582d916a49b8b3c0d6c160c8718`  
**Source fingerprint:** `7b4e284be7430005ffc20c2d0e26941b000387725a924e63bb626a50c6904bd0`

## Execution

The complete factorial schedule was run: 6 maps × 4 targets × 3 opponents × 6 open seeds × 2 seat assignments × 2 modes = **1,728 matches** (432 matched seed blocks). This follows the user's clarification that the listed factors take priority over Linear's incorrect 864-match product. The call ceiling was correspondingly 5,184,000 `step_tick` calls.

All 1,728 matches completed: 1,052 ended by queen death, 346 by economic collapse, and 330 by engine timeout draw; none were censored. The run used 1,395,430 `step_tick` calls. All 864 per-seat time-zero matched comparisons passed. The strict verifier replayed the full schedule and confirmed the JSONL rows and summary byte-for-byte.

The frozen arm used the unchanged baseline and zero parameter tuning, just like the closed-loop arm. Its policy input kept the initial 221 non-time observation floats and initial mask, refreshed only the two time floats, and used the current mask only in the shared decoder and validator. Across the retained unique trajectories there were 8,166 frozen target decisions, with zero changes in live mask and zero mask-only legality changes.

## Results

The estimate is the equal-weight mean of the 18 map × opponent cells after collapsing seats and deduplicating the four policy-relative matched traces. Values are target-relative score points on the 0–1 scale. They are descriptive estimates for this fixed schedule, not confidence bounds or significance tests.

| Target | Closed − frozen | Positive map means | Unique paired blocks | Cells with ≥2 blocks | Supported opponent families | Label |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| balanced | 11/27 (+0.4074) | 6/6 | 48/108 | 6/18 | random | INCONCLUSIVE |
| raider | 5/27 (+0.1852) | 5/6 | 48/108 | 6/18 | random | INCONCLUSIVE |
| turtle | 5/24 (+0.2083) | 5/6 | 48/108 | 6/18 | random | INCONCLUSIVE |
| rush | 1/54 (+0.0185) | 2/6 | 48/108 | 6/18 | random | INCONCLUSIVE |

The candidate labels are withheld for every target because supported cells cover only one opponent family. Each target has six qualifying cells, one per map against `random`; the deterministic `balanced` and `raider` opponent cells each collapse to one unique joint trajectory across their six seed labels. This fails the preregistered requirement to span at least two opponent families. The first three estimates exceed the 0.10 MDE, but that alone does not satisfy the coverage rule. The exact cell, map, opponent, and trajectory results are in `results/gate1_feedback/summary.json`.

Across retained unique matched seat pairs, live observations differed on 590/782 shared decisions for balanced, 1,122/1,326 for raider, 1,664/1,856 for turtle, and 114/304 for rush. Logits and validated actions differed on 686, 1,214, 1,760, and 180 shared decisions respectively. These are paired trace diagnostics, not additional independent samples.

## Limitations and protocol deviation

The six maps are fixed hand-picked scenarios, and the six open seeds do not make those maps a random sample. No confidence intervals or p-values were computed. The target policies are unchanged hand-written baselines; this does not compare against a tuned open-loop plan, demonstrate that learning helps, or pass Gate 1. The observed zero live-mask changes also means the schedule provides no empirical exercise of mask-only legality adaptation.

The first full simulation attempt reached all 432 matched blocks but failed while serializing the manifest because canonical JSON does not accept floating-point metadata. That invalid attempt is preserved in `results/gate1_feedback_attempt1_failure.json`; it emitted no result artifacts and was not used for analysis. After fixing metadata serialization, the entire unchanged schedule was restarted from its first seed and the resulting artifacts passed strict replay. Because the protocol says a technical error invalidates and stops a study, this recovery is a **protocol deviation**. Treat the verified second execution as reproducible exploratory diagnostics, not as a clean preregistered confirmatory result. No seeds, maps, policies, parameters, schedule factors, or decision rules were changed.

## Artifacts and checks

- `results/gate1_feedback/matches.jsonl` SHA-256: `5827685c0a2efcc06c97e9771d0ee44868989fd24259d5272ce90c419baea574`
- `results/gate1_feedback/summary.json` SHA-256: `d78eb85fe1532b25b59f5beeb64ec1a198d66da811647822d3f134cf45fc75b1`
- Strict replay: `verified`, 1,728 matches, 1,395,430 calls.
- Python suite: 863 passed.
- Viewer production build: passed.
- Viewer Vitest: 18 passed.
- Windows CI: pending PR run.

Commands:

```text
PYTHONPATH=src python -m formicmind.gate1_feedback run --out results/gate1_feedback
PYTHONPATH=src python -m formicmind.gate1_feedback verify --input results/gate1_feedback
python -m pytest -q -ra
cd viewer && npm run build && npm run test
```
