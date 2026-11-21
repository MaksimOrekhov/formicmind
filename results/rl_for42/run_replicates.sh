#!/bin/bash
# FOR-42 protocol section 7: P1b, P1c (replicates, seeds 43/44) and P2 (shaping ablation, coef 0.05, seed 42).
cd /e/FormicMind
PY=.venv/Scripts/python
run() { # name seed extra...
  name=$1; seed=$2; shift 2
  rm -rf results/rl_for42/$name
  $PY -m formicmind.rl train --out results/rl_for42/$name --set name=$name --set run_seed=$seed --set device=cuda --set workers=10 "$@" > results/rl_for42/$name.log 2>&1
  $PY -m formicmind.rl eval --run results/rl_for42/$name --suite full --workers 10 --vs-snapshots 8 16 --output results/rl_for42/$name/eval/final_full.json > results/rl_for42/$name.eval.log 2>&1
}
run p1b_seed43 43
run p1c_seed44 44
run p2_shaping_seed42 42 --set shaping_coef=0.05
echo ALL_DONE > results/rl_for42/replicates.done
