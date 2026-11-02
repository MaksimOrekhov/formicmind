"""Shared fixtures.

Seeds below are arbitrary engineering constants for development tests. They are
not taken from, and must not be replaced by, any FOR-4 sealed seed or template
placeholder.
"""

import pytest

from formicmind import default_config, reset

TEST_SEED = 20261102
OTHER_TEST_SEED = 424242


@pytest.fixture
def config():
    return default_config()


@pytest.fixture
def fresh(config):
    state, run = reset(config, TEST_SEED, ["policy:alpha:0", "policy:beta:0"])
    return state, run
