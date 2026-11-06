"""FOR-43: the reward / discount / GAE contract of Ecosystem v1 (docs/ecosystem_v1_contracts.md section 9) is
independent of how often the policy decides. Reference arithmetic only; no engine, no trainer, no torch.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ecosystem_reward_ref.py"
H = 6000


@pytest.fixture(scope="module")
def ref():
    spec = importlib.util.spec_from_file_location("ecosystem_reward_ref", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _varying(ref, n=H):
    # a deterministic, non-constant viability trace in 0..Q (a slow wave plus a step, never random)
    return [min(ref.Q, int(ref.Q * (0.5 + 0.4 * math.sin(t / 211.0)) * (0.5 if 2500 <= t < 3100 else 1.0))) for t in range(n)]


def _partitions(ref):
    return {
        "200x30": ref.partition(H),
        "600x10": ref.partition(H, base=10),
        "mixed": ref.partition(H, alert_windows=[(900, 1200), (4000, 4090), (5800, 5990)]),
        "irregular": [0, 7, 30, 31, 100, 1500, 2999, 3000, 5500],
    }


def _disc_total(transitions):
    total, scale = 0.0, 1.0
    for t in transitions:
        total += scale * t.reward_disc
        scale *= t.discount
    return total


def test_undiscounted_reward_does_not_depend_on_how_often_the_policy_decides(ref):
    constant = [int(0.6 * ref.Q)] * H
    totals = {name: sum(t.reward for t in ref.build_transitions(constant, ticks, H)) for name, ticks in _partitions(ref).items()}
    for name, value in totals.items():
        assert value == pytest.approx(int(0.6 * ref.Q) / ref.Q, abs=1e-12), name
    wave = _varying(ref)
    expected = sum(wave) / (ref.Q * H)
    for name, ticks in _partitions(ref).items():
        assert sum(t.reward for t in ref.build_transitions(wave, ticks, H)) == pytest.approx(expected, abs=1e-12), name


def test_an_alert_adds_no_reward(ref):
    constant = [ref.Q // 2] * H
    calm = ref.build_transitions(constant, ref.partition(H), H)
    fighting = ref.build_transitions(constant, ref.partition(H, alert_windows=[(0, H)]), H)
    assert len(fighting) > 2 * len(calm), "permanent combat triples the number of decisions"
    assert sum(t.reward for t in fighting) == pytest.approx(sum(t.reward for t in calm), abs=1e-12)
    assert sum(t.delta for t in fighting) == sum(t.delta for t in calm) == H


def test_per_interval_reward_is_proportional_to_elapsed_ticks(ref):
    constant = [ref.Q // 2] * H
    for t in ref.build_transitions(constant, _partitions(ref)["irregular"], H):
        assert t.reward == pytest.approx(0.5 * t.delta / H, abs=1e-12)


def test_discount_is_time_aware_and_composes(ref):
    assert ref.interval_discount(30) == pytest.approx(0.997)
    assert ref.interval_discount(10) ** 3 == pytest.approx(ref.interval_discount(30), rel=1e-12)
    assert ref.interval_discount(7) * ref.interval_discount(23) == pytest.approx(ref.interval_discount(30), rel=1e-12)
    # the horizon of the discount is the same in world ticks whatever the decision frequency
    assert ref.interval_discount(10) > ref.interval_discount(30)


def test_discounted_return_matches_the_tick_level_definition_for_every_split(ref):
    for trace in ([int(0.6 * ref.Q)] * H, _varying(ref)):
        want = ref.discounted_return_ticks(trace, H)
        for name, ticks in _partitions(ref).items():
            got = _disc_total(ref.build_transitions(trace, ticks, H))
            assert got == pytest.approx(want, rel=1e-12, abs=1e-12), name


def test_death_penalty_is_paid_exactly_once_even_with_alerts(ref):
    death = 1234
    trace = [int(0.6 * ref.Q)] * H
    calm = ref.build_transitions(trace, ref.partition(death), H, death_tick=death)
    alerts = ref.build_transitions(trace, ref.partition(death, alert_windows=[(1100, 1234)]), H, death_tick=death)
    for transitions in (calm, alerts):
        assert [t.terminated for t in transitions].count(True) == 1 and transitions[-1].terminated
        assert not any(t.truncated for t in transitions)
        assert sum(t.reward for t in transitions) == pytest.approx(int(0.6 * ref.Q) / ref.Q * death / H - 1.0, abs=1e-12)
    assert len(alerts) > len(calm)
    # tick-level agreement including the discounted penalty
    want = ref.discounted_return_ticks(trace, H, death_tick=death)
    for transitions in (calm, alerts):
        assert _disc_total(transitions) == pytest.approx(want, rel=1e-12, abs=1e-12)


def test_a_decision_at_or_after_the_end_of_the_episode_is_rejected(ref):
    trace = [ref.Q] * H
    with pytest.raises(ValueError):
        ref.build_transitions(trace, [0, 30, 1234], H, death_tick=1234)
    with pytest.raises(ValueError):
        ref.build_transitions(trace, [30, 60], H)
    with pytest.raises(ValueError):
        ref.build_transitions(trace[:100], [0], H)


def test_gae_with_lambda_one_equals_the_monte_carlo_advantage_for_variable_intervals(ref):
    trace = _varying(ref)
    ticks = _partitions(ref)["mixed"]
    transitions = ref.build_transitions(trace, ticks, H)
    values = [0.3 + 0.1 * math.cos(i / 13.0) for i in range(len(transitions))]  # an arbitrary (wrong) critic
    last = 0.37
    adv, targets = ref.gae(transitions, values, last, lambda30=1.0)
    g = ref.gamma_tick()
    for i in (0, 1, 10, len(transitions) // 2, len(transitions) - 1):
        start = transitions[i].start
        mc = sum(trace[tau] * g ** (tau - start) for tau in range(start, H)) / (ref.Q * H) + g ** (H - start) * last
        assert targets[i] == pytest.approx(mc, abs=1e-9), i
        assert adv[i] == pytest.approx(mc - values[i], abs=1e-9), i


def test_gae_is_zero_for_the_exact_value_function_whatever_the_split(ref):
    value = 0.6
    trace = [int(value * ref.Q)] * H
    g = ref.gamma_tick()

    def exact(t):
        return sum(trace[tau] * g ** (tau - t) for tau in range(t, H)) / (ref.Q * H)

    for name, ticks in _partitions(ref).items():
        transitions = ref.build_transitions(trace, ticks, H)
        values = [exact(t.start) for t in transitions]
        adv, _ = ref.gae(transitions, values, 0.0, lambda30=0.95)
        assert max(abs(a) for a in adv) < 1e-12, name


def test_bootstrap_is_the_learned_critic_only_and_only_after_truncation(ref):
    trace = [ref.Q // 2] * H
    values_for = lambda tr: [0.2] * len(tr)  # noqa: E731
    truncated = ref.build_transitions(trace, ref.partition(H), H)
    a1, _ = ref.gae(truncated, values_for(truncated), critic_last_value=0.1)
    a2, _ = ref.gae(truncated, values_for(truncated), critic_last_value=0.9)
    assert a2[-1] - a1[-1] == pytest.approx(truncated[-1].discount * 0.8, rel=1e-12)
    assert truncated[-1].truncated and not truncated[-1].terminated

    death = 2001
    terminated = ref.build_transitions(trace, ref.partition(death), H, death_tick=death)
    b1, _ = ref.gae(terminated, values_for(terminated), critic_last_value=0.1)
    b2, _ = ref.gae(terminated, values_for(terminated), critic_last_value=0.9)
    assert b1 == b2, "a terminal transition must not bootstrap, whatever value is supplied"
    # the viability score is never an input of GAE: the API only takes the critic's value
    import inspect
    assert "critic_last_value" in inspect.signature(ref.gae).parameters
    assert not any("viab" in name for name in inspect.signature(ref.gae).parameters)


def test_the_summary_script_runs(ref, capsys):
    assert ref.main() == 0
    assert "tick-level discounted reference" in capsys.readouterr().out


def test_a_per_decision_reward_would_pay_for_staying_in_combat_and_the_contract_does_not(ref):
    """Regression for the design flaw found in review: ``r = v / N_dec`` per decision rewards more decisions."""
    v = ref.Q // 2
    trace = [v] * H
    n_dec = H // ref.K

    def naive(ticks):
        return sum(trace[t] / ref.Q / n_dec for t in ticks)

    calm, fighting = ref.partition(H), ref.partition(H, alert_windows=[(0, H)])
    assert naive(fighting) > 2.5 * naive(calm)  # the exploit: constant combat triples the reward
    fixed = [sum(t.reward for t in ref.build_transitions(trace, ticks, H)) for ticks in (calm, fighting)]
    assert fixed[0] == pytest.approx(fixed[1], abs=1e-12)
