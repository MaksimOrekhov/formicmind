"""The policy contract ``PolicyE1`` and the two policies that go through it: ``random_valid`` and an observation-only fixture adapter.

A policy sees **only** ``(ObservationE1, ActionMasksE1)`` and returns either latents (a flat vector for ``decoder.decode``, the interface of a
learned network) or a complete ``ActionE1`` (a scripted policy). It is constructed with public rules (``ConfigE1``) at most; it never receives the
world, the map, another colony or its own colony id. One policy instance belongs to one colony, with its own random stream; the instances share
nothing, so decisions are independent by construction.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..canonical import CanonicalError
from ..rng import Pcg32, new_stream, policy_stream_id
from .action import ActionE1, ActionMasksE1, SquadActionE1, NO_SQUAD
from .config import ConfigE1
from .decoder import LATENT_SIZE
from .observation import ObservationE1

POLICY_KINDS = ("random_valid", "heuristic")
HEURISTIC_VARIANTS = ("peaceful", "explorer", "raider", "aggressor")


@runtime_checkable
class PolicyE1(Protocol):
    """``act`` returns ``LATENT_SIZE`` finite floats (decoded against the masks) or an ``ActionE1`` (validated against the masks)."""

    name: str

    def act(self, obs: ObservationE1, masks: ActionMasksE1) -> ActionE1 | Sequence[float]: ...

    def snapshot(self) -> dict: ...

    def restore(self, data: dict) -> None: ...


class RandomValidPolicy:
    """Uniform random latents from the colony's own PCG32 stream: after the decoder every action is valid, none is meaningful."""

    name = "random_valid"

    def __init__(self, rng: Pcg32) -> None:
        self.rng = rng

    def act(self, obs: ObservationE1, masks: ActionMasksE1) -> tuple[float, ...]:
        return tuple((self.rng.next_u32() / 4294967296.0 - 0.5) * 8.0 for _ in range(LATENT_SIZE))

    def snapshot(self) -> dict:
        return {"rng": self.rng.to_canonical()}

    def restore(self, data: dict) -> None:
        self.rng = Pcg32.from_canonical(data["rng"])


def _split_sixteen(weights: list[float]) -> tuple[int, ...]:
    """Positive weights -> shares of sixteenths, each at least one, summing to 16 (largest remainder, ties to the lower index)."""
    if not weights:
        return ()
    spare = 16 - len(weights)
    total = sum(weights)
    exact = [w / total * spare for w in weights]
    shares = [1 + int(x) for x in exact]
    for i in sorted(range(len(weights)), key=lambda i: (-(exact[i] - int(exact[i])), i))[:16 - sum(shares)]:
        shares[i] += 1
    return tuple(shares)


def _bucket_upper(bucket: int) -> int:
    return 0 if bucket == 0 else (127 if bucket >= 7 else (1 << bucket) - 1)


class ObservationHeuristicPolicy:
    """The diagnostic fixtures re-expressed over ``ObservationE1`` only (no world oracle): fair economy, optional scouting, optional raiding.

    * ``peaceful``: lay when the warehouse covers the next upkeep plus a reserve; forage the nearest known patches.
    * ``explorer``: peaceful, plus up to two scouts walking to the farthest reachable unexplored node within 30 ticks.
    * ``raider``: explorer, plus soldiers, and an attack on the weakest neighbour nest it has found when its idle army outnumbers what it last saw.
    * ``aggressor``: explorer, plus soldiers, and an attack on the first legal target as soon as two soldiers are idle: a deliberately reckless
      exerciser of the attack path through ActionE1 (so that fights appear in a replay), not a strategy.
    It is a check of the interface, not a teacher; it does not try to be good at war.
    """

    def __init__(self, config: ConfigE1, variant: str = "peaceful") -> None:
        if variant not in HEURISTIC_VARIANTS:
            raise ValueError(f"variant must be one of {HEURISTIC_VARIANTS}")
        self.economy, self.variant = config.economy, variant
        self.name = f"heuristic:{variant}"

    def act(self, obs: ObservationE1, masks: ActionMasksE1) -> ActionE1:
        f, e, nodes = obs.facts, self.economy, obs.node_ints
        reserve = (f.burn_tenths * 25 + 19) // 20
        egg = 3 if f.food - f.burn_tenths // 10 >= reserve + e.worker_egg_cost else 0
        soldiers_wanted = f.soldiers + f.brood_soldiers < 6 and f.food >= 60 and f.workers >= 10
        share = 2 if self.variant in ("raider", "aggressor") and soldiers_wanted else 0

        # the three patches with the best expected yield per trip: what the colony last saw there, discounted by the known walking time
        rated = sorted(((-(obs.node_features[slot][9] * obs.node_features[slot][10] * 200 + 8) / (2 * nodes[slot][3] + 8), slot)
                        for slot, ok in enumerate(masks.forage_target) if ok and nodes[slot][3] < 16))[:3]
        targets = tuple(slot for _, slot in rated)
        shares = _split_sixteen([-score for score, _ in rated])
        frac = 8 if targets and f.food < e.storage_cap and masks.forage_frac[8] else 0
        if not frac:
            targets, shares = (), ()

        scouts, scout_targets = 0, ()
        if self.variant != "peaceful" and f.workers >= 12 and f.food >= 30 and f.scouts_out < 2 and masks.scout_n[1]:
            frontier = sorted(((-nodes[slot][3], slot) for slot, ok in enumerate(masks.scout_target)
                               if ok and nodes[slot][1] == 0 and nodes[slot][3] <= 30))
            if frontier:
                scouts, scout_targets = 1, (frontier[0][1],)

        squads = [NO_SQUAD, NO_SQUAD, NO_SQUAD]
        attack_out = any(round(v * 3) == 2 for v in obs.global_features[23:26])
        if self.variant == "raider" and not attack_out and f.food >= 80 and masks.squad_mission[0][2]:
            options = []
            for node_slot, colony_slot in masks.attack_pairs:
                if obs.foreign_nest_slots[colony_slot] != node_slot or obs.foreign_features[colony_slot][6] == 1.0:
                    continue  # only a found nest whose queen is not known to be dead
                upper = _bucket_upper(round(obs.foreign_features[colony_slot][4] * 7))
                options.append((upper, nodes[node_slot][3], colony_slot, node_slot))
            if options:
                upper, _, colony_slot, node_slot = min(options)
                if f.idle_soldiers_home >= max(4, upper + 2):
                    squads[0] = SquadActionE1("attack", 4, node_slot, colony_slot)
        if self.variant == "aggressor" and masks.attack_pairs and f.idle_soldiers_home >= 2 and f.food >= 20 and masks.squad_mission[0][2]:
            node_slot, colony_slot = masks.attack_pairs[0]
            squads[0] = SquadActionE1("attack", 4, node_slot, colony_slot)
        return ActionE1(egg, share, frac, scouts, False, targets, shares, scout_targets, tuple(squads))  # type: ignore[arg-type]

    def snapshot(self) -> dict:
        return {}

    def restore(self, data: dict) -> None:
        return None


# ------------------------------------------------------------------------------------------------ run specification


@dataclass(frozen=True, slots=True)
class PolicySpec:
    kind: str           # "random_valid" | "heuristic"
    variant: str = ""   # heuristic: peaceful | explorer | raider


@dataclass(frozen=True, slots=True)
class PolicyRunSpec:
    """Who decides for each colony and the root seed of their private random streams (stream ``policy:<kind>:<colony>``)."""

    policy_seed: int
    policies: tuple[PolicySpec, ...]


POLICY_RUN_SCHEMA = "formicmind.eco.policy-run/1"


def uniform_spec(colonies: int, kind: str, policy_seed: int, variant: str = "") -> PolicyRunSpec:
    return PolicyRunSpec(policy_seed, tuple(PolicySpec(kind, variant or ("peaceful" if kind == "heuristic" else "")) for _ in range(colonies)))


def stream_id_of(spec: PolicySpec, colony: int) -> str:
    return policy_stream_id(spec.kind, colony)


def build_policies(spec: PolicyRunSpec, config: ConfigE1) -> list[PolicyE1]:
    """One independent policy per colony; each random one draws from its own stream derived from the policy seed and the colony index."""
    if len(spec.policies) != config.n_colonies:
        raise ValueError("one policy per colony is required")
    out: list[PolicyE1] = []
    for colony, item in enumerate(spec.policies):
        if item.kind == "random_valid":
            out.append(RandomValidPolicy(new_stream(spec.policy_seed, stream_id_of(item, colony))))
        elif item.kind == "heuristic":
            out.append(ObservationHeuristicPolicy(config, item.variant or "peaceful"))
        else:
            raise ValueError(f"unknown policy kind {item.kind!r}; expected one of {POLICY_KINDS}")
    return out


def spec_to_canonical(spec: PolicyRunSpec) -> dict:
    return {
        "schema": POLICY_RUN_SCHEMA, "policy_seed": str(spec.policy_seed),
        "policies": [{"colony": i, "kind": p.kind, "variant": p.variant, "stream": stream_id_of(p, i)} for i, p in enumerate(spec.policies)],
    }


def spec_from_canonical(data: object) -> PolicyRunSpec:
    if (type(data) is not dict or set(data) != {"schema", "policy_seed", "policies"} or data["schema"] != POLICY_RUN_SCHEMA
            or type(data["policy_seed"]) is not str or not data["policy_seed"].isdigit() or type(data["policies"]) is not list):
        raise CanonicalError("not a formicmind.eco.policy-run/1 specification")
    items = []
    for index, row in enumerate(data["policies"]):
        if (type(row) is not dict or set(row) != {"colony", "kind", "variant", "stream"} or row["colony"] != index
                or row["kind"] not in POLICY_KINDS or type(row["variant"]) is not str):
            raise CanonicalError(f"policy {index}: bad record")
        item = PolicySpec(row["kind"], row["variant"])
        if row["stream"] != stream_id_of(item, index):
            raise CanonicalError(f"policy {index}: the stream id does not follow from the kind and the colony")
        items.append(item)
    return PolicyRunSpec(int(data["policy_seed"]), tuple(items))
