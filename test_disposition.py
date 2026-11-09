import math

import pytest

from disposition import (
    Assumptions,
    break_even_sell_through,
    decide,
    option_values,
    price_from_ceilings,
    rank_by_headroom,
)

BASE = dict(
    resale_price=55.0, prep_cost=8.0, shipping_cost=8.0, sell_through=0.6,
    holding_cost=4.0, liquidation_value=18.0, liquidation_cost=2.0,
    donation_value=0.0, donation_cost=1.0,
)


def make(**overrides) -> Assumptions:
    return Assumptions(**{**BASE, **overrides})


def test_option_values():
    v = option_values(make())
    # -8 + 0.6 * 47 + 0.4 * (16 - 4)
    assert v["resell"] == pytest.approx(25.0)
    assert v["liquidate"] == pytest.approx(16.0)
    assert v["donate"] == pytest.approx(-1.0)


def test_break_even_formula():
    # (8 + 4) / (55 - 8 - 16 + 4)
    assert break_even_sell_through(make()) == pytest.approx(12 / 35)


def test_resale_equals_alternative_at_break_even():
    q = break_even_sell_through(make())
    v = option_values(make(sell_through=q))
    assert v["resell"] == pytest.approx(v["liquidate"])


@pytest.mark.parametrize("delta, expected", [(-0.01, "liquidate"), (0.01, "resell")])
def test_choice_flips_at_break_even(delta, expected):
    q = break_even_sell_through(make())
    assert decide(make(sell_through=q + delta))["choice"] == expected


def test_donation_wins_when_it_is_the_best_exit():
    a = make(sell_through=0.0, liquidation_value=0.0, donation_value=10.0)
    assert decide(a)["choice"] == "donate"


def test_no_break_even_when_a_sale_adds_nothing():
    # Selling nets 10 - 8 = 2, below the unsold path of 16 - 4 = 12.
    assert break_even_sell_through(make(resale_price=10.0)) is None


def test_break_even_above_one_means_resale_cannot_win():
    a = make(resale_price=30.0, prep_cost=20.0)
    q = break_even_sell_through(a)
    assert q > 1
    assert decide(make(resale_price=30.0, prep_cost=20.0, sell_through=1.0))["choice"] == "liquidate"


def test_decide_reports_margin():
    d = decide(make())
    assert d["choice"] == "resell"
    assert d["margin"] == pytest.approx(9.0)


@pytest.mark.parametrize("field, value", [
    ("sell_through", 1.2), ("sell_through", -0.1), ("prep_cost", -1.0),
    ("resale_price", math.nan), ("holding_cost", math.inf), ("shipping_cost", "8"),
])
def test_invalid_assumptions_rejected(field, value):
    with pytest.raises(ValueError):
        make(**{field: value})


def test_price_from_ceilings():
    assert price_from_ceilings([60.0, 50.0, 58.0], min_buyers=1) == 60.0
    assert price_from_ceilings([60.0, 50.0, 58.0], min_buyers=2) == 58.0
    assert price_from_ceilings([60.0], min_buyers=2) is None
    with pytest.raises(ValueError):
        price_from_ceilings([60.0], min_buyers=0)


def test_rank_by_headroom():
    buyers = [
        {"buyer_id": "B", "max_wtp_cad": 58.0},
        {"buyer_id": "A", "max_wtp_cad": 56.0},
        {"buyer_id": "C", "max_wtp_cad": 58.0},
    ]
    assert [b["buyer_id"] for b in rank_by_headroom(buyers, 55.0)] == ["B", "C", "A"]
