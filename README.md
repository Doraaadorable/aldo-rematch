# ReMatch: deciding what happens to a returned shoe

A returned pair that passes inspection can be resold to a waiting buyer, sent to a liquidator or donated. ReMatch is an agent that works through that decision: it looks up the return, checks inspection status, finds opted-in buyers whose size, grade and price ceiling fit, creates the offer and notifies the waitlist.

Built at the [Retail Gen AI Hackathon](https://www.mcgill.ca/bensadoun-school/featured-events/retail-gen-ai-hackathon) (McGill Bensadoun School of Retail Management with ALDO Group, AWS and OODARIS, October 2026), where our team of six placed **2nd of 10 finalists**. I was the team's modeller and wrote the code and the Streamlit dashboard. This repository contains the synthetic-data version; the version built on competition data is kept private.

## How decisions are made

The language model (Amazon Nova 2 Lite on Bedrock, via Strands Agents) chooses which tool to call next. Every number and every gate is computed in Python:

| Step | Rule |
|---|---|
| Inspection gate | Only a `passed` inspection proceeds; offers also require eligibility approval recorded in the inspection record, which the model cannot set |
| Buyer screen | Opted in, SKU and grade accepted, stated ceiling ≥ price |
| Price | Fixed by the inspection record; the model cannot supply one |
| Buyer order | At a fixed price every qualifying buyer yields the same recovery, so buyers are ordered by headroom (ceiling − price): the least price-sensitive buyer is asked first |
| Notifications | Up to five waitlist buyers, by the same order; dry run unless `NOTIFY_LIVE=1` and confirmed in the UI |
| Turn limit | 10 agent turns |

## Resell, liquidate or donate (`disposition.py`)

Resale is not free money: cleaning and handling are paid up front, and an item that does not sell within the listing window still has to be liquidated or donated after a holding cost. With price *p*, shipping *s* (paid only on a sale), preparation *c*, holding cost *h*, sell-through rate *q* and *A* the better of liquidation and donation (net of their costs):

```
E[resell] = −c + q(p − s) + (1 − q)(A − h)
q*        = (c + h) / (p − s − A + h)
```

Resale beats the alternatives only when the sell-through rate exceeds *q\**. With the demo assumptions (price 55, shipping 8, preparation 8, holding 4, liquidation nets 16) the break-even is 34%: resale is the better choice at 60% sell-through and liquidation at 30%.

`price_from_ceilings` gives the highest price that at least *k* buyers' stated ceilings clear; with *k* = 2 a second buyer remains if the first declines. All inputs are adjustable in the dashboard.

## Run it

```bash
pip install -r requirements-dev.txt
python -m pytest -q
streamlit run app.py
```

The tests and the dashboard need no AWS credentials; the agent runs only when you press **Run agent**, which needs Bedrock access in `us-east-1`. `python agent_strands.py` runs the agent from the command line.

## Tests

79 offline tests, run in CI along with a render check of the dashboard:

| File | Covers |
|---|---|
| `test_offline.py` | Inspection gates, buyer screening and order, offer creation and idempotency, state resets |
| `test_notifications.py` | Waitlist consent, recipient cap and order, dry-run vs live sending, SNS subscription checks (mocked) |
| `test_disposition.py` | Option values, break-even rate, decision flips at the break-even, pricing from ceilings, input validation |

## Files

| Path | Contents |
|---|---|
| `agent_strands.py` | Tools, synthetic fixtures, agent construction, CLI |
| `disposition.py` | Resell / liquidate / donate economics |
| `notifications.py` | Waitlist store and SNS notification service |
| `app.py` | Streamlit dashboard |
