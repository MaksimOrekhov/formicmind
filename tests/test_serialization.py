import hashlib
import json

import pytest

from formicmind import (
    canonical_loads,
    checkpoint_from_bytes,
    checkpoint_hash,
    checkpoint_to_bytes,
    config_from_bytes,
    config_hash,
    config_to_bytes,
    default_config,
    reset,
    state_from_bytes,
    state_hash,
    state_to_bytes,
)
from formicmind.canonical import CanonicalError, canonical_dumps
from formicmind.model import BroodRecord
from formicmind.rng import policy_stream_id
from formicmind.serialization import state_to_canonical

from conftest import OTHER_TEST_SEED, TEST_SEED

# Regression hashes of the initial state under the default config. They pin the
# canonical byte layout; they are produced by this implementation and become
# authoritative only after review of the foundation PR. Any change to the schema,
# defaults or serialization must update them consciously.
GOLDEN_CONFIG_HASH = "c93de10ef90cc2b77ad11f635dac5cf2c5d9a8984524b0898caaaa26f63f6274"
GOLDEN_INITIAL_STATE_HASH = "38852d486c85adce5b97c0bdcf9c036621cdc62a82ce69c0335c5f93539dde07"
GOLDEN_INITIAL_STATE_BYTES = 5234


def test_golden_hashes(config):
    state, _ = reset(config, TEST_SEED)
    assert config_hash(config) == GOLDEN_CONFIG_HASH
    assert state_hash(state) == GOLDEN_INITIAL_STATE_HASH
    assert len(state_to_bytes(state)) == GOLDEN_INITIAL_STATE_BYTES


def test_hash_is_domain_separated_sha256_of_canonical_bytes(config):
    state, _ = reset(config, TEST_SEED)
    assert state_hash(state) == hashlib.sha256(
        b"formicmind/state/v1\x00" + state_to_bytes(state)
    ).hexdigest()
    assert config_hash(config) == hashlib.sha256(
        b"formicmind/config/v1\x00" + config_to_bytes(config)
    ).hexdigest()


def test_state_round_trip_bytes_and_hash(config):
    state, _ = reset(config, TEST_SEED)
    data = state_to_bytes(state)
    restored = state_from_bytes(data, config)
    assert state_to_bytes(restored) == data
    assert state_hash(restored) == state_hash(state)
    assert restored.next_unit_id == state.next_unit_id == 33
    assert restored == state


def test_config_round_trip(config):
    data = config_to_bytes(config)
    restored = config_from_bytes(data)
    assert restored == config
    assert config_to_bytes(restored) == data


def test_round_trip_of_non_initial_state(config):
    state, _ = reset(config, TEST_SEED)
    c = state.colonies[1]
    c.units[2].hp = 7
    c.units[3].node, c.units[3].from_node, c.units[3].to_node = -1, 6, 4
    c.units[3].arrival_tick = 5
    c.brood.append(BroodRecord(state.allocate_unit_id(), 1, 0, 30))
    c.food, c.upkeep_remainder10, c.starvation_debt = 123, 7, 3
    c.lay_order, c.forage_q, c.army_choice, c.avoid_center = 2, [10000, 20000, 35535, 0, 0, 0], 6, True
    c.intel.nodes[0].seen_tick, c.intel.nodes[0].food_last = 0, 9
    c.intel.enemy_queen_hp_last, c.intel.enemy_queen_seen_tick = 100, 0
    state.node_food[3] = 77
    data = state_to_bytes(state)
    restored = state_from_bytes(data, config)
    assert restored == state and state_hash(restored) == state_hash(state)


def test_any_state_change_changes_hash(config):
    state, _ = reset(config, TEST_SEED)
    base = state_hash(state)
    mutations = [
        lambda s: setattr(s, "tick", 1),
        lambda s: s.node_food.__setitem__(1, 59),
        lambda s: setattr(s.colonies[0], "food", 61),
        lambda s: setattr(s.colonies[1].units[0], "hp", 19),
        lambda s: setattr(s.colonies[0], "lay_order", 1),
        lambda s: setattr(s.colonies[1].intel.nodes[2], "seen_tick", 0),
        lambda s: setattr(s, "next_unit_id", 34),
    ]
    hashes = {base}
    for mutate in mutations:
        copy = state.clone()
        mutate(copy)
        hashes.add(state_hash(copy))
    assert len(hashes) == len(mutations) + 1


def test_checkpoint_round_trip_restores_rng_and_next_id(config):
    ids = [policy_stream_id("alpha", 0), policy_stream_id("beta", 1)]
    state, run = reset(config, TEST_SEED, ids)
    state.allocate_unit_id()
    for _ in range(13):
        run.stream("world").next_u32()
        run.stream(ids[0]).below(7)
    data = checkpoint_to_bytes(state, run)
    state2, run2 = checkpoint_from_bytes(data, config)
    assert checkpoint_hash(state2, run2) == checkpoint_hash(state, run)
    assert state_hash(state2) == state_hash(state) and state2.next_unit_id == 34
    assert run2.root_seed == TEST_SEED
    for sid in ["world", *ids]:
        assert [run2.stream(sid).next_u32() for _ in range(20)] == [
            run.stream(sid).next_u32() for _ in range(20)]


def test_state_hash_and_checkpoint_hash_are_distinct_and_rng_sensitive(config):
    state, run = reset(config, TEST_SEED)
    h_state, h_ckpt = state_hash(state), checkpoint_hash(state, run)
    assert h_state != h_ckpt
    run.stream("world").next_u32()
    assert state_hash(state) == h_state          # RNG is not part of the state hash
    assert checkpoint_hash(state, run) != h_ckpt  # but is part of the checkpoint hash
    other_state, other_run = reset(config, OTHER_TEST_SEED)
    assert state_hash(other_state) == h_state
    assert checkpoint_hash(other_state, other_run) != checkpoint_hash(*reset(config, TEST_SEED))


def test_u64_values_are_strings_in_serialized_run_state(config):
    _, run = reset(config, 2**64 - 1)
    text = checkpoint_to_bytes(reset(config, 2**64 - 1)[0], run).decode("ascii")
    doc = json.loads(text)["run"]
    assert doc["root_seed"] == "18446744073709551615"
    assert all(type(v) is str for s in doc["streams"].values() for v in s.values())


def _walk(value):
    yield value
    if isinstance(value, dict):
        for v in value.values():
            yield from _walk(v)
    elif isinstance(value, list):
        for v in value:
            yield from _walk(v)


def test_serialized_form_has_only_stable_plain_types(config):
    state, run = reset(config, TEST_SEED, [policy_stream_id("alpha", 0)])
    for data in (config_to_bytes(config), state_to_bytes(state), checkpoint_to_bytes(state, run)):
        text = data.decode("ascii")
        doc = canonical_loads(data)
        assert all(type(v) in (int, str, bool, list, dict) for v in _walk(doc))
        assert "NaN" not in text and "null" not in text and ".0" not in text
        lowered = text.lower()
        for unstable in ("uuid", "timestamp", "datetime", "hostname", "pid", "path", "wall"):
            assert unstable not in lowered


def test_state_schema_is_the_expected_key_set(config):
    state, _ = reset(config, TEST_SEED)
    doc = state_to_canonical(state)
    assert set(doc) == {"schema", "serialization_version", "rules_version", "config_sha256",
                        "tick", "next_unit_id", "node_food", "colonies"}
    assert set(doc["colonies"][0]) == {
        "side", "food", "queen_hp", "next_lay_tick", "upkeep_remainder10", "starvation_debt",
        "lay_order", "forage_q", "army_choice", "avoid_center", "units", "brood", "intel"}
    assert set(doc["colonies"][0]["units"][0]) == {
        "unit_id", "caste", "hp", "node", "from_node", "to_node", "arrival_tick", "target_node"}
    assert set(doc["colonies"][0]["intel"]) == {"nodes", "enemy_queen_hp_last", "enemy_queen_seen_tick"}


def test_object_construction_order_does_not_affect_bytes(config):
    state, _ = reset(config, TEST_SEED)
    doc = state_to_canonical(state)

    def reverse_keys(v):
        if isinstance(v, dict):
            return {k: reverse_keys(v[k]) for k in reversed(list(v))}
        if isinstance(v, list):
            return [reverse_keys(x) for x in v]
        return v

    assert canonical_dumps(reverse_keys(doc)) == state_to_bytes(state)


def _mutated(config, edit):
    state, _ = reset(config, TEST_SEED)
    doc = state_to_canonical(state)
    edit(doc)
    return canonical_dumps(doc)


@pytest.mark.parametrize(
    "edit",
    [
        lambda d: d.__setitem__("extra", 1),
        lambda d: d.pop("tick"),
        lambda d: d["colonies"][0].__setitem__("food", 60.0),
        lambda d: d["colonies"][0].__setitem__("avoid_center", 0),
        lambda d: d["colonies"][0]["units"].reverse(),             # unsorted units
        lambda d: d["colonies"][0]["units"][0].__setitem__("hp", 99),
        lambda d: d["colonies"].reverse(),                          # side != index
        lambda d: d.__setitem__("schema", "formicmind.sim_state/2"),
        lambda d: d.__setitem__("serialization_version", "v0"),
        lambda d: d.__setitem__("config_sha256", "0" * 64),
        lambda d: d.__setitem__("rules_version", "l0-proto-other"),
        lambda d: d["node_food"].append(0),
    ],
)
def test_loader_rejects_corrupted_state(config, edit):
    with pytest.raises((CanonicalError, ValueError)):
        state_from_bytes(_mutated(config, edit), config)


def test_loader_rejects_non_canonical_bytes(config):
    state, _ = reset(config, TEST_SEED)
    pretty = json.dumps(json.loads(state_to_bytes(state)), indent=1).encode("ascii")
    with pytest.raises(CanonicalError):
        state_from_bytes(pretty, config)


def test_loader_rejects_state_from_a_different_config(config):
    import dataclasses

    state, _ = reset(config, TEST_SEED)
    other = dataclasses.replace(config, t_max=2999)
    with pytest.raises(CanonicalError):
        state_from_bytes(state_to_bytes(state), other)


def test_checkpoint_loader_rejects_bad_run_state(config):
    state, run = reset(config, TEST_SEED)
    doc = json.loads(checkpoint_to_bytes(state, run))
    doc["run"]["streams"]["world"]["inc"] = "2"
    with pytest.raises(CanonicalError):
        checkpoint_from_bytes(canonical_dumps(doc), config)
    doc = json.loads(checkpoint_to_bytes(state, run))
    doc["run"]["rng_version"] = "other"
    with pytest.raises(CanonicalError):
        checkpoint_from_bytes(canonical_dumps(doc), config)
    doc = json.loads(checkpoint_to_bytes(state, run))
    doc["run"]["streams"]["bad id"] = doc["run"]["streams"]["world"]
    with pytest.raises(CanonicalError):
        checkpoint_from_bytes(canonical_dumps(doc), config)


def test_default_config_instances_hash_equal():
    assert config_hash(default_config()) == config_hash(default_config())
