"""ActionMaskV2 (30 bool), ActionV2 and the fail-fast validator."""

import dataclasses

import pytest

from formicmind.actions import (
    ActionMaskV2,
    ActionV2,
    InvalidAction,
    MaskError,
    ValidatedActionV2,
    flatten_mask_v2,
    get_action_mask,
    mask_from_flat,
    validate_action,
)
from formicmind.observation import ObservationV2, build_observation, f32
from formicmind.snapshot import begin_policy_phase

from helpers import busy_state, fresh_state, mirror_state

T, F = True, False

# FOR-2 v0.2 section 3.4, "Golden mask G0": lay all legal, all six forage targets, all seven
# army choices; avoid_center forbidden for garrison_home and defend(C), allowed otherwise.
G0_MASK = (
    T, T, T,
    T, T, T, T, T, T,
    T, T, T, T, T, T, T,
    T, F,   # garrison_home
    T, T,   # defend(N1)
    T, T,   # defend(N2)
    T, F,   # defend(C)
    T, T,   # defend(N4)
    T, T,   # defend(N5)
    T, T,   # raid(enemy_home)
)


def g0_obs(side=0):
    return build_observation(begin_policy_phase(fresh_state(0)), side)


def with_values(obs, edit):
    values = list(obs.values)
    edit(values)
    return ObservationV2(tuple(f32(v) for v in values))


def edge_index(i, j):
    index = 0
    for a in range(7):
        for b in range(a + 1, 7):
            if (a, b) == (i, j):
                return 47 + index
            index += 1
    raise AssertionError


def good_action(**overrides):
    base = dict(lay_order=1, forage_share_q=(0, 65535, 0, 0, 0, 0), army_choice=2, avoid_center=False)
    base.update(overrides)
    return ActionV2(**base)


def test_g0_mask_is_exactly_30_bool_in_the_documented_order():
    mask = get_action_mask(g0_obs())
    flat = flatten_mask_v2(mask)
    assert len(flat) == 30 and all(type(b) is bool for b in flat)
    assert flat == G0_MASK
    assert mask_from_flat(flat) == mask


def test_mask_is_the_same_for_both_seats_and_for_mirrored_states():
    assert get_action_mask(g0_obs(0)) == get_action_mask(g0_obs(1))
    state = busy_state()
    mirrored = mirror_state(state)
    a = get_action_mask(build_observation(begin_policy_phase(state), 0))
    b = get_action_mask(build_observation(begin_policy_phase(mirrored), 1))
    assert a == b


def test_mask_is_a_pure_function_of_the_observation():
    obs = g0_obs()
    assert get_action_mask(obs) == get_action_mask(obs)


def test_dead_queen_forbids_producing_orders_only():
    def kill(values):
        values[12 + 5] = 0.0
    mask = get_action_mask(with_values(g0_obs(), kill))
    assert mask.lay == (T, F, F)


def test_unreachable_targets_are_masked_from_the_public_edges():
    def cut_enemy_home(values):
        for i in range(6):
            values[edge_index(i, 6)] = 0.0
    mask = get_action_mask(with_values(g0_obs(), cut_enemy_home))
    assert mask.army_choice == (T, T, T, T, T, T, F)
    assert mask.army_avoid_center[6] == (F, F)
    assert mask.army_avoid_center[1] == (T, T)


def test_avoid_center_is_forbidden_when_no_route_without_c_exists():
    def cut_a_n2(values):
        values[edge_index(0, 2)] = 0.0          # A has only A-N1-C left
    mask = get_action_mask(with_values(g0_obs(), cut_a_n2))
    # N1 is still reachable directly; everything else needs C; garrison and defend(C) never allow it
    assert mask.army_avoid_center == ((T, F), (T, T), (T, F), (T, F), (T, F), (T, F), (T, F))
    assert mask.army_choice == (T,) * 7


def test_mask_type_guards():
    good = get_action_mask(g0_obs())
    for kwargs in (
        dict(lay=(T, T)),
        dict(forage_target=(T,) * 5),
        dict(army_choice=(T,) * 8),
        dict(army_avoid_center=((T, T),) * 6),
        dict(army_avoid_center=((T, T, T),) + good.army_avoid_center[1:]),
        dict(lay=(1, 1, 1)),
    ):
        with pytest.raises(MaskError):
            dataclasses.replace(good, **kwargs)
    with pytest.raises(MaskError):
        mask_from_flat(G0_MASK[:-1])


def test_validator_accepts_a_good_action_and_returns_a_validated_wrapper():
    obs = g0_obs()
    mask = get_action_mask(obs)
    result = validate_action(obs, mask, good_action())
    assert isinstance(result, ValidatedActionV2) and result.action == good_action()
    for q in [(65535, 0, 0, 0, 0, 0), (21845, 21845, 21845, 0, 0, 0), (1, 2, 65532, 0, 0, 0)]:
        validate_action(obs, mask, good_action(forage_share_q=q))
    validate_action(obs, mask, good_action(army_choice=6, avoid_center=True))


@pytest.mark.parametrize(
    "bad",
    [
        dict(lay_order=3), dict(lay_order=-1), dict(lay_order=True), dict(lay_order=1.0),
        dict(army_choice=7), dict(army_choice=-1), dict(army_choice=False),
        dict(avoid_center=1), dict(avoid_center=None),
        dict(forage_share_q=(65535, 0, 0, 0, 0)),
        dict(forage_share_q=[0, 65535, 0, 0, 0, 0]),
        dict(forage_share_q=(0, 65535.0, 0, 0, 0, 0)),
        dict(forage_share_q=(0, 65536, -1, 0, 0, 0)),
        dict(forage_share_q=(0, 65534, 0, 0, 0, 0)),
        dict(forage_share_q=(0, 65536, 0, 0, 0, 0)),
        dict(forage_share_q=(16384, 16384, 16384, 16383, 0, 0)),   # support 4
        dict(forage_share_q=(0, 0, 0, 0, 0, 0)),
        dict(army_choice=0, avoid_center=True),                      # garrison_home + avoid
        dict(army_choice=3, avoid_center=True),                      # defend(C) + avoid
    ],
)
def test_validator_rejects_invalid_actions(bad):
    obs = g0_obs()
    with pytest.raises(InvalidAction):
        validate_action(obs, get_action_mask(obs), good_action(**bad))


def test_validator_enforces_the_mask():
    obs = g0_obs()
    base = get_action_mask(obs)
    no_worker = dataclasses.replace(base, lay=(T, F, T))
    with pytest.raises(InvalidAction):
        validate_action(obs, no_worker, good_action(lay_order=1))
    validate_action(obs, no_worker, good_action(lay_order=2))
    no_n1 = dataclasses.replace(base, forage_target=(T, F, T, T, T, T))
    with pytest.raises(InvalidAction):
        validate_action(obs, no_n1, good_action(forage_share_q=(0, 65535, 0, 0, 0, 0)))
    no_raid = dataclasses.replace(base, army_choice=(T, T, T, T, T, T, F))
    with pytest.raises(InvalidAction):
        validate_action(obs, no_raid, good_action(army_choice=6))


def test_validator_rejects_non_action_input_without_side_effects():
    obs = g0_obs()
    mask = get_action_mask(obs)
    with pytest.raises(InvalidAction):
        validate_action(obs, mask, (1, (0, 65535, 0, 0, 0, 0), 2, False))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        validate_action("obs", mask, good_action())  # type: ignore[arg-type]
