"""Curriculum action masks (FOR-50): a static, named restriction of the action space that is applied as a pure intersection of masks.

A profile is a *training condition*, not a rule of the world and not a policy: ``apply_curriculum_masks(obs, base, profile)`` takes the masks that
``action_masks(obs)`` derived from the observation and returns the same masks with a few options switched off. The result is the one *effective* mask
of the decision: the network samples from it, ``log_prob`` is taken under it, the decoder decodes against it, the validator checks the action
against it and only then are the orders applied. Nothing is masked after the fact and nothing is overridden after ``log_prob``.

* ``none`` (the default): the very same ``ActionMasksE1`` object comes back, so every run, fixture and replay without a profile is byte-for-byte what
  it was before this module existed.
* ``nest-v1`` (stage 1 "Nest", ``abundance`` scenario): ``soldier_share`` is 0 (every new egg is a worker; the two starting soldiers stay and age
  as usual), ``scout_n`` is 0 (no scouts) and all three squads are ``none`` (no guard, attack or withdraw). Egg level, forage fraction, forage
  targets and shares, and ``workers_flee`` stay entirely the policy's choice, as do all physical rules.

This module imports only ``action`` (itself observation-only): it can read nothing but the observation and the static profile.
"""

from __future__ import annotations

from dataclasses import replace

from .action import ActionMasksE1, SQUADS
from .observation import ObservationE1

CURRICULUM_NONE = "none"
CURRICULUM_NEST_V1 = "nest-v1"
CURRICULUM_PROFILES = (CURRICULUM_NONE, CURRICULUM_NEST_V1)
CURRICULUM_VERSION = "curriculum/1"
# the squad mission order of ``action.SQUAD_MISSIONS``: only "none" stays legal under the nest profile
_ONLY_NONE = (True, False, False, False)


class CurriculumError(ValueError):
    """An unknown curriculum profile name."""


def validate_profile(profile: object) -> str:
    """Return ``profile`` if it names a known curriculum profile, else raise ``CurriculumError``."""
    if type(profile) is not str or profile not in CURRICULUM_PROFILES:
        raise CurriculumError(f"curriculum profile must be one of {CURRICULUM_PROFILES}, got {profile!r}")
    return profile


def apply_curriculum_masks(obs: ObservationE1, base: ActionMasksE1, profile: str = CURRICULUM_NONE) -> ActionMasksE1:
    """The effective masks of ``profile`` for one decision: ``base`` intersected with the profile's static restrictions.

    Pure: depends on ``base`` and the profile only (``obs`` is accepted so that a future profile may use local facts, never anything else).
    ``none`` returns ``base`` itself. A restriction never switches an option *on*: every flag of the result is ``base`` AND the profile's rule.
    """
    validate_profile(profile)
    if profile == CURRICULUM_NONE:
        return base
    only_zero_soldiers = tuple(i == 0 and flag for i, flag in enumerate(base.soldier_share))
    no_scouts = tuple(i == 0 and flag for i, flag in enumerate(base.scout_n))
    no_missions = tuple(tuple(flag and allowed for flag, allowed in zip(row, _ONLY_NONE)) for row in base.squad_mission)
    if len(no_missions) != SQUADS:
        raise CurriculumError("the masks do not describe three squads")
    return replace(base, soldier_share=only_zero_soldiers, scout_n=no_scouts, squad_mission=no_missions)


def describe_profile(profile: str) -> str:
    """One line for logs, reports and the viewer: what the profile restricts."""
    validate_profile(profile)
    if profile == CURRICULUM_NONE:
        return "none (full action space)"
    return "nest-v1 (diagnostic: new eggs are workers only, no scouts, no squad orders; egg level, foraging and the flee flag are free)"
