"""
Disposition economics for one returned item: resell, liquidate or donate.

Resale is uncertain: preparation is paid up front, and an item that does not
sell within the listing window still has to leave through liquidation or
donation, after a holding cost. With

    p     resale price            s   shipping (paid only on a sale)
    c     preparation cost        h   holding cost of an unsold item
    q     sell-through rate       A   best of liquidation and donation (net)

the expected value of trying to resell is

    E[resell] = -c + q (p - s) + (1 - q) (A - h)

and resale beats the best alternative when q exceeds the break-even rate

    q* = (c + h) / (p - s - A + h).

All inputs are per-unit CAD figures supplied by the caller; this module holds
no data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class Assumptions:
    resale_price: float
    prep_cost: float
    shipping_cost: float
    sell_through: float
    holding_cost: float
    liquidation_value: float
    liquidation_cost: float
    donation_value: float
    donation_cost: float

    def __post_init__(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{f.name} must be a finite number, got {value!r}")
            if value < 0:
                raise ValueError(f"{f.name} must be non-negative, got {value}")
        if self.sell_through > 1:
            raise ValueError(f"sell_through must be in [0, 1], got {self.sell_through}")


def option_values(a: Assumptions) -> dict[str, float]:
    """Expected net value of each disposition option, in CAD."""
    liquidate = a.liquidation_value - a.liquidation_cost
    donate = a.donation_value - a.donation_cost
    unsold = max(liquidate, donate) - a.holding_cost
    resell = (
        -a.prep_cost
        + a.sell_through * (a.resale_price - a.shipping_cost)
        + (1 - a.sell_through) * unsold
    )
    return {"resell": resell, "liquidate": liquidate, "donate": donate}


def break_even_sell_through(a: Assumptions) -> float | None:
    """Sell-through rate at which resale equals the best alternative.

    None when a sale is worth no more than the unsold path, so no rate makes
    resale win. A value above 1 means resale cannot win at this price.
    """
    alternative = max(a.liquidation_value - a.liquidation_cost,
                      a.donation_value - a.donation_cost)
    gain = a.resale_price - a.shipping_cost - alternative + a.holding_cost
    if gain <= 0:
        return None
    return (a.prep_cost + a.holding_cost) / gain


def decide(a: Assumptions) -> dict:
    """Best option, its margin over the runner-up, and the break-even rate."""
    values = option_values(a)
    ordered = sorted(values.items(), key=lambda kv: kv[1], reverse=True)
    (choice, best), (_, runner_up) = ordered[0], ordered[1]
    return {
        "choice": choice,
        "values": {k: round(v, 2) for k, v in values.items()},
        "margin": round(best - runner_up, 2),
        "break_even_sell_through": break_even_sell_through(a),
    }


def price_from_ceilings(ceilings: list[float], min_buyers: int = 2) -> float | None:
    """Highest price that at least *min_buyers* stated ceilings still clear.

    Requiring more than one buyer leaves a fallback if the first declines.
    None when fewer than *min_buyers* ceilings are given.
    """
    if min_buyers < 1:
        raise ValueError("min_buyers must be at least 1")
    ranked = sorted(ceilings, reverse=True)
    if len(ranked) < min_buyers:
        return None
    return ranked[min_buyers - 1]


def rank_by_headroom(buyers: list[dict], price: float) -> list[dict]:
    """Order buyers by how far their ceiling sits above the price.

    At a fixed price every buyer yields the same recovery, so the order
    decides who is asked first. A buyer whose ceiling clears the price by
    more is the less price-sensitive one; buyer_id breaks ties.
    """
    return sorted(buyers, key=lambda b: (-(b["max_wtp_cad"] - price), b["buyer_id"]))
