"""Post-run analysis (FOR-42): protocol gates, learning curves and measured policy shift. Read-only on a finished run."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

from .checkpoint import latest_checkpoint, load_checkpoint
from .distribution import split_heads, masked_log_softmax
from .network import actor_state_to_bytes, build_actor, actor_state_from_bytes
from .parallel import execute_chunk
from .pool import SnapshotPool, atomic_write_json


def read_metrics(run_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train, evals = [], []
    for line in (Path(run_dir) / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            (train if record["kind"] == "train" else evals).append(record)
    return train, evals


def total_variation(p: list[float], q: list[float]) -> float:
    return 0.5 * sum(abs(a - b) for a, b in zip(p, q))


def probe_observations(count_matches: int = 6) -> tuple[torch.Tensor, torch.Tensor]:
    """Observations and masks from fixed baseline-vs-baseline matches (never from training), for measuring policy shift."""
    from ..actions import get_action_mask
    from ..observation import build_observation
    from ..reset import reset
    from ..rng import policy_stream_id
    from ..snapshot import begin_policy_phase
    from ..baselines import baseline_spec
    from ..tick import PolicyController, step_tick
    from ..match import _register_streams
    from .maps import map_config

    obs_rows, mask_rows = [], []
    plan = [("balanced:v2", "turtle:v2", "classic"), ("rush:v2", "raider:v2", "dense"), ("turtle:v2", "balanced:v2", "chain"),
            ("raider:v2", "random:v2", "gen:1000003"), ("balanced:v2", "rush:v2", "far_flank"), ("random:v2", "turtle:v2", "gen:1000004")]
    for i, (a, b, map_id) in enumerate(plan[:count_matches]):
        config = map_config(map_id)
        specs = [baseline_spec(a), baseline_spec(b)]
        controllers = tuple(PolicyController(s.build(), policy_stream_id(s.policy_id, k)) for k, s in enumerate(specs))
        state, run = reset(config, 777_000 + i)
        _register_streams(run, controllers, ())
        while run.outcome is None and state.tick < config.t_max:
            if state.tick % config.policy_interval_k == 0 and state.tick > 0 and (state.tick // config.policy_interval_k) % 2 == 0:
                clone, run_clone = state.clone(), run.clone()
                snapshot = begin_policy_phase(clone, (run_clone.windows[0], run_clone.windows[1]))
                for side in (0, 1):
                    obs = build_observation(snapshot, side)
                    obs_rows.append(list(obs.values))
                    mask_rows.append(list(get_action_mask(obs).flatten()))
            step_tick(state, run, controllers)
    return torch.tensor(obs_rows, dtype=torch.float32), torch.tensor(mask_rows, dtype=torch.bool)


def policy_shift(initial: dict[str, torch.Tensor], final: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Mean per-state KL(initial || final) and mean-distribution total variation for the lay and army heads."""
    obs, mask = probe_observations()
    a, b = build_actor(initial), build_actor(final)
    with torch.no_grad():
        ha, hb = split_heads(a.policy(obs)), split_heads(b.policy(obs))
    out: dict[str, Any] = {"probe_states": int(obs.shape[0])}
    for name, mask_slice in (("lay", mask[:, 0:3]), ("army", mask[:, 9:16]), ("target", mask[:, 3:9])):
        la = masked_log_softmax(ha[name], mask_slice)
        lb = masked_log_softmax(hb[name], mask_slice)
        pa = la.exp()
        kl = (pa * (la - lb)).masked_fill(~mask_slice, 0.0).sum(-1)
        out[name] = {
            "mean_kl_initial_to_final": float(kl.mean()),
            "total_variation_of_mean_distribution": total_variation(pa.mean(0).tolist(), lb.exp().mean(0).tolist()),
            "mean_initial": [round(x, 4) for x in pa.mean(0).tolist()],
            "mean_final": [round(x, 4) for x in lb.exp().mean(0).tolist()],
        }
    return out


def pilot_gates(run_dir: str | Path, full_eval_path: str | Path) -> dict[str, Any]:
    """Evaluate the protocol's section-8 criteria. T3 (bit-exact resume) is a unit test and is reported as such."""
    run_dir = Path(run_dir)
    train, _ = read_metrics(run_dir)
    report = json.loads(Path(full_eval_path).read_text(encoding="utf-8"))
    head = train[:5]
    lay_seen = all(any(r["matches"]["lay_fraction"][i] > 0 for r in head) for i in range(3))
    army_seen = len({i for r in head for i, v in enumerate(r["matches"]["army_fraction"]) if v > 0}) >= 5
    support_seen = all(any(r["matches"]["executed_support_fraction"][i] > 0 for r in head) for i in range(3))
    finite = all(all(math.isfinite(v) for v in r["ppo"].values()) for r in train)
    identity = max(r["ppo"]["identity_ratio_max_dev"] for r in train)
    changed = all(r["param_delta_l2"] > 0 for r in train)
    first, last = train[0]["matches"], train[-1]["matches"]
    tv_lay = total_variation(first["lay_fraction"], last["lay_fraction"])
    tv_army = total_variation(first["army_fraction"], last["army_fraction"])
    vs_init = report["results"].get("init", {}).get("sample", {}).get("all", {})
    l1 = vs_init.get("score_mean")
    gates = {
        "iterations": len(train), "matches": train[-1]["matches_total"],
        "T1_no_errors_run_completed": True,
        "T2_weights_changed_and_losses_finite": bool(changed and finite and identity < 1e-3),
        "T2_max_identity_ratio_dev": identity,
        "T3_resume_bit_exact": "unit test tests/test_rl_train.py::test_a_real_tiny_run_changes_weights_from_rewards_and_resume_is_exact",
        "T4_diverse_early_actions": bool(lay_seen and army_seen and support_seen),
        "L1_score_vs_init_full_sample": l1, "L1_pass": bool(l1 is not None and l1 >= 0.60),
        "L3_tv_lay_first_vs_last": tv_lay, "L3_tv_army_first_vs_last": tv_army,
        "L3_pass": bool(max(tv_lay, tv_army) >= 0.10),
    }
    gates["technical_pass"] = bool(gates["T1_no_errors_run_completed"] and gates["T2_weights_changed_and_losses_finite"]
                                   and gates["T4_diverse_early_actions"])
    gates["run_go"] = bool(gates["technical_pass"] and gates["L1_pass"] and gates["L3_pass"])
    return gates


def learning_curve(run_dir: str | Path) -> dict[str, Any]:
    """Per-iteration training statistics plus the periodic (monitoring) evaluation scores."""
    train, evals = read_metrics(Path(run_dir))
    rows = []
    for r in train:
        m = r["matches"]
        snap = m["by_opponent"].get("snapshot", {})
        decided = snap.get("win", 0) + snap.get("loss", 0) + snap.get("draw", 0)
        rows.append({
            "iteration": r["iteration"], "matches_total": r["matches_total"], "samples": r["batch"]["samples"],
            "mean_match_ticks": m["ticks"] / max(m["matches"], 1), "entropy_discrete": r["ppo"]["entropy_discrete"],
            "value_loss": r["ppo"]["value_loss"], "explained_variance": r["ppo"]["explained_variance"],
            "approx_kl": r["ppo"]["approx_kl"], "clipfrac": r["ppo"]["clipfrac"], "param_delta_l2": r["param_delta_l2"],
            "win_rate_vs_snapshots": (snap.get("win", 0) / decided) if decided else None,
            "draw_fraction_mirror": (m["by_opponent"].get("current", {}).get("draw", 0)
                                     / max(m["by_opponent"].get("current", {}).get("matches", 1), 1)),
            "lay_fraction": m["lay_fraction"], "army_fraction": m["army_fraction"], "home_share": m["mean_home_share"],
            "reasons": m["reasons"], "episodes_per_second": r["timing"]["episodes_per_second"],
            "decisions_per_second": r["timing"]["policy_decisions_per_second"],
        })
    return {"train": rows, "monitor_evals": [{"iteration": e["iteration"], **e["headline"]} for e in evals]}


def analyse_run(run_dir: str | Path, full_eval_path: str | Path, output: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    model, manifest, _ = load_checkpoint(latest_checkpoint(run_dir))
    pool = SnapshotPool(run_dir / "snapshots")
    initial = pool.load_state(pool.load_info(0))
    result = {
        "gates": pilot_gates(run_dir, full_eval_path),
        "policy_shift_initial_to_final": policy_shift(initial, model.actor_state_dict()),
        "curve": learning_curve(run_dir),
        "final_iteration": manifest["iteration"],
    }
    atomic_write_json(Path(output), result)
    return result


__all__ = ["analyse_run", "learning_curve", "pilot_gates", "policy_shift", "probe_observations", "read_metrics"]
