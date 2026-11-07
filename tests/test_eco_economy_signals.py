"""FOR-48 review fix: the observation's income / burn / runway follow contracts 9.1 and use the colony's own accounting only.

income = food actually delivered into the own nest (harvested food still on the workers is not spendable); burn = upkeep *due* (it stays
positive while the colony starves); runway = food / (burn_ema - income_ema), infinite when the income covers the burn.
"""

from __future__ import annotations

from dataclasses import replace

from test_eco_policy_helpers import Run
from formicmind.eco.engine import step_world_tick
from formicmind.eco.model import RollingE1, WindowStatsE1
from formicmind.eco.observation import Q, build_observation, net_runway_feature, next_rolling

RUNWAY_INDEX, INCOME_INDEX = 9, 8


def _rolling(income: float, burn: float, pop: int = 20) -> RollingE1:
    return RollingE1(int(income * Q), int(burn * Q), pop * Q)


def test_harvested_food_that_was_not_delivered_is_no_income():
    start = _rolling(income=0.0, burn=5.0)
    harvested_only = next_rolling(start, WindowStatsE1(harvested=300, delivered=0, upkeep_due=150, upkeep_paid=150), 20, 2, 30)
    assert harvested_only.income_ema_q == 0 and harvested_only.burn_ema_q == start.burn_ema_q
    delivered = next_rolling(start, WindowStatsE1(harvested=0, delivered=90, upkeep_due=150, upkeep_paid=150), 20, 2, 30)
    assert delivered.income_ema_q == (3 * 0 + 90 * Q // 30) // 4 > 0


def test_burn_is_the_upkeep_due_even_when_it_could_not_be_paid():
    start = _rolling(income=0.0, burn=5.0)
    starving = next_rolling(start, WindowStatsE1(upkeep_due=150, upkeep_paid=0), 20, 2, 30)
    assert starving.burn_ema_q == start.burn_ema_q                      # still owed: 5.0 per tick, not decaying towards zero
    assert next_rolling(start, WindowStatsE1(upkeep_due=150, upkeep_paid=30), 20, 2, 30).burn_ema_q == start.burn_ema_q


def test_runway_is_food_over_the_net_burn_and_infinite_when_income_covers_it():
    assert net_runway_feature(80, _rolling(0.0, 5.0)) == 16 / 40                    # 80 / 5 ticks
    assert net_runway_feature(80, _rolling(3.0, 5.0)) == 1.0                         # 80 / 2 = 40 ticks: saturated
    assert net_runway_feature(10, _rolling(3.0, 5.0)) == 5 / 40                      # a steady income stretches 2 ticks into 5
    assert net_runway_feature(1, _rolling(6.0, 5.0)) == 1.0                          # income >= burn with almost no stored food
    assert net_runway_feature(1, _rolling(5.0, 5.0)) == 1.0
    assert net_runway_feature(0, _rolling(0.0, 5.0)) == 0.0                          # empty store, nothing coming in
    assert net_runway_feature(0, _rolling(5.0, 5.0)) == 1.0


def test_the_observation_carries_the_net_runway_and_the_delivered_income_of_its_own_colony():
    run = Run(seed=50012)
    colony = run.state.colonies[0]
    own = [c for c in run.state.cohorts if c.colony_id == 0]
    for food, income, burn in ((12, 2.5, 6.0), (12, 6.0, 6.0), (300, 0.0, 6.0)):
        poor = replace(colony, food=food)
        obs = build_observation(poor, own, 0, run.config, decision_index=0, ticks_since_last_decision=0, rolling=_rolling(income, burn))
        net = burn - income
        assert obs.global_features[RUNWAY_INDEX] == (1.0 if net <= 0 else min(1.0, food / net / 40)) or abs(
            obs.global_features[RUNWAY_INDEX] - (1.0 if net <= 0 else min(1.0, food / net / 40))) < 1e-6
        assert abs(obs.global_features[INCOME_INDEX] - min(1.0, income / 40)) < 1e-6
    # same stored food and same upkeep: only the income differs, and so does the runway
    low = build_observation(replace(colony, food=12), own, 0, run.config, decision_index=0, ticks_since_last_decision=0,
                            rolling=_rolling(0.0, 6.0)).global_features[RUNWAY_INDEX]
    high = build_observation(replace(colony, food=12), own, 0, run.config, decision_index=0, ticks_since_last_decision=0,
                             rolling=_rolling(4.0, 6.0)).global_features[RUNWAY_INDEX]
    assert high > low


def test_the_engine_books_delivery_and_upkeep_due_in_the_colonys_own_window():
    """Real window (ticks 151..179 of seed 50012, after the decision at tick 150): window.delivered is the colony's own share of the nest
    deliveries and is smaller than window.harvested where food was still on its way home; the food ledger keeps its old counters."""
    run = Run(seed=50012)
    delivered = [0] * 8
    while run.state.tick < 179:
        record = step_world_tick(run.world.state, run.world.run, run.config, run.world.world_map, run.controller, routes=run.routes,
                                 validate_input=run.world.state.tick == 0)
        if record.tick > 150:
            for cid, amount in record.delivered_by_colony.items():
                delivered[cid] += amount
    windows = [c.window for c in run.state.colonies]
    assert [w.delivered for w in windows] == delivered
    assert all(w.upkeep_due >= w.upkeep_paid > 0 for w in windows)
    assert windows[1].harvested == 176 and windows[1].delivered == 160     # harvested, but 16 units of it are still cargo on the way home
    assert all(w.harvested >= w.delivered for w in windows) and sum(w.delivered for w in windows) > 0


def test_a_starving_colony_keeps_a_positive_burn_in_its_window():
    run = Run(seed=50012)
    colony = run.state.colonies[0]
    for _ in range(8):                                                    # the store is empty at every upkeep: all of it is missing
        colony.food = 0
        step_world_tick(run.world.state, run.world.run, run.config, run.world.world_map, run.controller, routes=run.routes,
                        validate_input=run.world.state.tick == 0)
    assert colony.window.upkeep_paid == 0 and colony.window.upkeep_due > 0
    closed = next_rolling(colony.rolling, colony.window, 20, 2, 8)
    owed = (3 * colony.rolling.burn_ema_q + colony.window.upkeep_due * Q // 8) // 4
    assert closed.burn_ema_q == owed > 3 * colony.rolling.burn_ema_q // 4  # the old upkeep_paid signal would have been exactly 3/4 of the old EMA
