"""Opt-in profiles of the information supplied at world initialization."""

DEFAULT_KNOWLEDGE = "default"
UNKNOWN_FOOD_V1 = "unknown-food-v1"
INITIAL_KNOWLEDGE_PROFILES = (DEFAULT_KNOWLEDGE, UNKNOWN_FOOD_V1)


def validate_initial_knowledge(profile: str) -> str:
    if profile not in INITIAL_KNOWLEDGE_PROFILES:
        raise ValueError(f"unknown initial_knowledge {profile!r}; expected one of {INITIAL_KNOWLEDGE_PROFILES}")
    return profile
