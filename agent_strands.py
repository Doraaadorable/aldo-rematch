"""
agent_strands.py  –  Strands SDK migration + Return-to-Resale Matching
=======================================================================
Uses strands-agents >= 1.0 APIs.  Key references:

  • @tool decorator         strands.tools.decorator
  • Agent class             strands.agent.agent
  • BedrockModel            strands.models.bedrock
  • Limits (turn cap)       strands.types.agent.Limits
  • BotocoreConfig          botocore.config.Config  (passed as boto_client_config)

Synthetic demo scenarios
------------------------
DEMO-001  Original case.  condition="not inspected"; INSPECT-DEMO-001
          has status="pending".  Verifies that a pending inspection blocks
          resale matching and offer creation.

DEMO-003  Pre-approved case.  condition="inspected – grade B";
          INSPECT-DEMO-003 has status="passed" and
          demo_eligibility_approved=True.  Exercises the full
          resale-offer creation path.

Design constraints
------------------
• All returns, buyers, prices and costs are synthetic fixtures.
• Net recovery = price − shipping − cleaning − handling.  At a fixed price it
  is the same for every qualifying buyer, so buyers are ordered by headroom
  (max_wtp − price): who is asked first, not who pays more.
• create_resale_offer enforces the fixture's proposed_sale_price_cad;
  the model cannot supply a price.
• Only "passed" inspections may proceed to resale; any other status is
  rejected.
• demo_eligibility_approved must be True in the fixture before an offer can
  be created.  The model cannot set it.
• Tools are plain functions callable without Bedrock; the tests need no AWS
  credentials.  build_agent() is the only place that touches Bedrock.
• State lives in <script-dir>/demo_state/ regardless of cwd.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from disposition import rank_by_headroom

# ---------------------------------------------------------------------------
# State directory – anchored to this script's location, not cwd
# ---------------------------------------------------------------------------

_STATE_DIR: Path = Path(__file__).resolve().parent / "demo_state"


# ---------------------------------------------------------------------------
# Synthetic demo fixtures
# ---------------------------------------------------------------------------

# Return records.
# `condition` reflects current known physical state (display only).
# The authoritative resale gate is the inspection record status.
RETURNS: dict[str, dict] = {
    "DEMO-001": {
        "return_order_number": "DEMO-001",
        "product": "Demo Sneakers",
        "sku": "SHOE-WHT-9",
        "reason": "Too small",
        "refund_amount": 80.00,
        "currency": "CAD",
        "final_sale": "unknown",
        "condition": "not inspected",          # INSPECT-DEMO-001 is pending
    },
    "DEMO-003": {
        "return_order_number": "DEMO-003",
        "product": "Demo Chelsea Boots",
        "sku": "BOOT-TAN-7",
        "reason": "Ordered wrong size",
        "refund_amount": 135.00,
        "currency": "CAD",
        "final_sale": "no",
        "condition": "inspected – grade B",    # INSPECT-DEMO-003 passed
    },
}

# Inspection records.
# Statuses: "pending" | "passed" | "failed"
# Only "passed" may proceed to resale matching and offer creation.
# All monetary values are SYNTHETIC ASSUMPTIONS.
INSPECTION_RESULTS: dict[str, dict] = {
    "INSPECT-DEMO-001": {
        "inspection_id": "INSPECT-DEMO-001",
        "return_order_number": "DEMO-001",
        "status": "pending",                        # SYNTHETIC – blocks resale
        "grade": None,
        "notes": "Awaiting physical review.",
        "proposed_sale_price_cad": None,
        "assumed_costs_cad": None,
        "demo_eligibility_approved": False,
    },
    "INSPECT-DEMO-003": {
        "inspection_id": "INSPECT-DEMO-003",
        "return_order_number": "DEMO-003",
        "status": "passed",                         # SYNTHETIC
        "grade": "B",                               # SYNTHETIC grade
        "notes": "Light wear on toe box; otherwise clean and resaleable.",
        "proposed_sale_price_cad": 55.00,           # SYNTHETIC ASSUMPTION
        "assumed_costs_cad": {
            "shipping": 8.00,                       # SYNTHETIC ASSUMPTION
            "cleaning": 5.00,                       # SYNTHETIC ASSUMPTION
            "handling": 3.00,                       # SYNTHETIC ASSUMPTION
        },
        # Pre-approved in fixture data only.
        # The agent must never set or invent this value.
        "demo_eligibility_approved": True,
    },
}

# Opted-in buyers.
# max_wtp_cad: self-reported price ceiling – used only as a feasibility screen.
# It is NOT evidence of higher net recovery or stronger purchase intent.
# All data are SYNTHETIC ASSUMPTIONS.
BUYERS: list[dict] = [
    {
        "buyer_id": "BUYER-A",
        "name": "Alice Demo",
        "opted_in": True,
        "accepted_grades": ["A", "B"],
        "sku_preferences": ["SHOE-WHT-9", "BOOT-TAN-7"],
        "max_wtp_cad": 60.00,                       # SYNTHETIC CEILING
    },
    {
        "buyer_id": "BUYER-B",
        "name": "Bob Demo",
        "opted_in": True,
        "accepted_grades": ["A"],                   # grade B not accepted
        "sku_preferences": ["SHOE-WHT-9", "BOOT-TAN-7"],
        "max_wtp_cad": 65.00,                       # SYNTHETIC CEILING
    },
    {
        "buyer_id": "BUYER-C",
        "name": "Carol Demo",
        "opted_in": True,
        "accepted_grades": ["A", "B", "C"],
        "sku_preferences": ["BOOT-TAN-7"],
        "max_wtp_cad": 50.00,                       # SYNTHETIC CEILING – below $55
    },
    {
        "buyer_id": "BUYER-D",
        "name": "Dave Demo",
        "opted_in": False,                          # not opted-in
        "accepted_grades": ["A", "B"],
        "sku_preferences": ["BOOT-TAN-7"],
        "max_wtp_cad": 70.00,
    },
    {
        "buyer_id": "BUYER-E",
        "name": "Eve Demo",
        "opted_in": True,
        "accepted_grades": ["A", "B"],
        "sku_preferences": ["BOOT-TAN-7"],
        "max_wtp_cad": 58.00,                       # SYNTHETIC CEILING – above $55
    },
]


# ---------------------------------------------------------------------------
# Internal validation helpers (not tools – not exposed to the model)
# ---------------------------------------------------------------------------

def _validate_price(price: object, label: str = "price") -> str | None:
    """Return an error string if *price* is not a finite positive number."""
    if not isinstance(price, (int, float)):
        return f"{label} must be a number, got {type(price).__name__}"
    if not math.isfinite(price):
        return f"{label} must be finite, got {price}"
    if price <= 0:
        return f"{label} must be positive, got {price}"
    return None


def _validate_costs(costs: dict) -> str | None:
    """Return an error string if any cost value is not a finite non-negative number."""
    for key, value in costs.items():
        if not isinstance(value, (int, float)):
            return f"cost '{key}' must be a number, got {type(value).__name__}"
        if not math.isfinite(value):
            return f"cost '{key}' must be finite, got {value}"
        if value < 0:
            return f"cost '{key}' must be non-negative, got {value}"
    return None


# ---------------------------------------------------------------------------
# Tools – importable and callable without Bedrock / AWS credentials
# ---------------------------------------------------------------------------

from strands import tool  # noqa: E402


# ── Phase 1: original tools (preserved unchanged) ───────────────────────────

@tool
def get_return_case(return_order_number: str) -> dict:
    """
    Retrieve a return case by its return order number.

    Args:
        return_order_number: The unique return identifier (e.g. DEMO-001).

    Returns:
        Dict with ``found`` (bool). When found, includes the full ``case``
        record; otherwise a human-readable ``message``.
    """
    print(f"\n[TOOL] Querying return: {return_order_number}")
    case = RETURNS.get(return_order_number)
    if case is None:
        return {"found": False, "message": "Return not found"}
    return {"found": True, "case": case}


@tool
def create_inspection_request(return_order_number: str) -> dict:
    """
    Create a simulated inspection request for an existing return.

    Idempotent: a second call returns the existing record with created=False.

    Important: creating a request does NOT complete the inspection, approve
    return eligibility, or issue a refund.

    Args:
        return_order_number: The return order number to inspect.

    Returns:
        The inspection record dict. ``created`` is True for a new record,
        False when reused. Returns an ``error`` key if the return is not found.
    """
    if return_order_number not in RETURNS:
        return {"error": "Return not found", "return_order_number": return_order_number}

    _STATE_DIR.mkdir(exist_ok=True)
    path = _STATE_DIR / f"inspection_{return_order_number}.json"

    if path.exists():
        record = json.loads(path.read_text())
        record["created"] = False
        print(f"\n[TOOL] Reused existing inspection: {record['inspection_id']}")
        return record

    record = {
        "inspection_id": f"INSPECT-{return_order_number}",
        "return_order_number": return_order_number,
        "status": "pending",
        "simulation": True,
        "created": True,
    }
    path.write_text(json.dumps(record, indent=2))
    print(f"\n[TOOL] Created inspection: {record['inspection_id']}")
    return record


# ── Phase 2: Return-to-Resale tools ─────────────────────────────────────────

@tool
def get_inspection_result(inspection_id: str) -> dict:
    """
    Retrieve the inspection result for a given inspection ID.

    If status is "pending" the physical inspection is not yet complete.
    The agent MUST NOT proceed with resale matching or offer creation when
    status is not "passed". Report the status and stop.
    Do not invent or assume an inspection outcome.

    All grades, prices, and costs in the result are SYNTHETIC ASSUMPTIONS.

    Args:
        inspection_id: The inspection ID (e.g. INSPECT-DEMO-003).

    Returns:
        The inspection result dict including status, grade, proposed sale
        price (CAD, SYNTHETIC), and assumed cost breakdown (SYNTHETIC).
        Returns an ``error`` key if the inspection_id is not found.
    """
    print(f"\n[TOOL] Fetching inspection result: {inspection_id}")
    result = INSPECTION_RESULTS.get(inspection_id)
    if result is None:
        return {"error": "Inspection not found", "inspection_id": inspection_id}
    return dict(result)   # shallow copy – callers must not mutate the fixture


@tool
def get_eligible_buyers(sku: str, grade: str) -> dict:
    """
    Return opted-in buyers compatible with the given SKU and inspection grade.

    A buyer's max_wtp_cad is a self-reported price ceiling used only to
    screen whether they can afford the proposed price. It is NOT evidence
    of higher net recovery or stronger purchase intent.

    All buyer data are SYNTHETIC ASSUMPTIONS.

    Args:
        sku:   Item SKU from the return record (e.g. BOOT-TAN-7).
        grade: Inspection grade of the item (e.g. "B").

    Returns:
        Dict with:
          ``eligible``  – buyers that pass opt-in, SKU, and grade checks
          ``excluded``  – buyers that failed, each with a ``reasons`` list
    """
    print(f"\n[TOOL] Finding eligible buyers for SKU={sku!r} grade={grade!r}")
    eligible = []
    excluded = []

    for buyer in BUYERS:
        reasons = []
        if not buyer["opted_in"]:
            reasons.append("not opted-in")
        if sku not in buyer["sku_preferences"]:
            reasons.append(
                f"SKU {sku!r} not in preferences {buyer['sku_preferences']}"
            )
        if grade not in buyer["accepted_grades"]:
            reasons.append(
                f"grade {grade!r} not in accepted grades {buyer['accepted_grades']}"
            )

        if reasons:
            excluded.append({"buyer_id": buyer["buyer_id"], "reasons": reasons})
        else:
            eligible.append(buyer)

    return {"eligible": eligible, "excluded": excluded}


@tool
def calculate_resale_matches(inspection_id: str) -> dict:
    """
    Screen buyers for an inspected item and order those who qualify.

    net_recovery = proposed_sale_price − shipping − cleaning − handling

    A buyer qualifies when opted in, the SKU and grade match, the stated
    ceiling max_wtp_cad is at least the proposed price, and net recovery is
    positive.  Qualifying buyers are ordered by headroom (max_wtp − price),
    largest first, buyer_id breaking ties.  Only "passed" inspections proceed.

    Args:
        inspection_id: The inspection whose price and costs are used.

    Returns:
        Dict with best_match, ranked_eligible, excluded, net_recovery_cad,
        proposed_sale_price_cad, cost_breakdown and notes.
    """
    print(f"\n[TOOL] Calculating resale matches for {inspection_id}")

    insp = INSPECTION_RESULTS.get(inspection_id)
    if insp is None:
        return {"error": "Inspection not found", "inspection_id": inspection_id}

    # Reject every status except "passed"
    if insp["status"] != "passed":
        return {
            "error": (
                f"Inspection status is '{insp['status']}' – resale matching "
                f"requires status 'passed'."
            ),
            "inspection_id": inspection_id,
            "status": insp["status"],
        }

    proposed_price = insp["proposed_sale_price_cad"]
    costs = insp["assumed_costs_cad"]

    # Validate fixture data (guards against accidental misconfiguration)
    price_err = _validate_price(proposed_price, "proposed_sale_price_cad")
    if price_err:
        return {"error": f"Invalid fixture data: {price_err}"}
    cost_err = _validate_costs(costs)
    if cost_err:
        return {"error": f"Invalid fixture data: {cost_err}"}

    total_cost = costs["shipping"] + costs["cleaning"] + costs["handling"]
    net_recovery = round(proposed_price - total_cost, 2)

    sku = RETURNS[insp["return_order_number"]]["sku"]
    grade = insp["grade"]

    buyer_result = get_eligible_buyers(sku, grade)
    excluded: list[dict] = list(buyer_result["excluded"])
    qualified: list[dict] = []

    for buyer in buyer_result["eligible"]:
        if buyer["max_wtp_cad"] < proposed_price:
            excluded.append({
                "buyer_id": buyer["buyer_id"],
                "reasons": [
                    f"max_wtp {buyer['max_wtp_cad']:.2f} CAD < "
                    f"proposed price {proposed_price:.2f} CAD (SYNTHETIC)"
                ],
            })
        elif net_recovery <= 0:
            excluded.append({
                "buyer_id": buyer["buyer_id"],
                "reasons": [
                    f"net recovery {net_recovery:.2f} CAD is not positive – "
                    f"assumed costs ({total_cost:.2f} CAD) meet or exceed "
                    f"proposed price ({proposed_price:.2f} CAD) (SYNTHETIC)"
                ],
            })
        else:
            qualified.append(buyer)

    ranked = rank_by_headroom(qualified, proposed_price)
    best = ranked[0] if ranked else None

    notes_parts = [
        f"Proposed price: {proposed_price:.2f} CAD.",
        f"Costs: {total_cost:.2f} CAD (shipping {costs['shipping']:.2f}, "
        f"cleaning {costs['cleaning']:.2f}, handling {costs['handling']:.2f}).",
        f"Net recovery: {net_recovery:.2f} CAD.",
        f"Qualifying buyers: {len(ranked)}. Excluded: {len(excluded)}.",
    ]
    if best:
        headroom = best["max_wtp_cad"] - proposed_price
        notes_parts.append(
            f"Best match: {best['buyer_id']} (ceiling {headroom:.2f} CAD above price)."
        )
    else:
        notes_parts.append("No qualifying buyer found.")

    return {
        "best_match": best,
        "ranked_eligible": ranked,
        "excluded": excluded,
        "net_recovery_cad": net_recovery,
        "proposed_sale_price_cad": proposed_price,
        "cost_breakdown": costs,
        "notes": "  ".join(notes_parts),
    }


@tool
def create_resale_offer(inspection_id: str, buyer_id: str) -> dict:
    """
    Create a persistent simulated resale offer.

    The sale price is taken directly from the inspection fixture and cannot
    be overridden by the model.

    Prerequisites enforced (the model must not attempt to bypass them):
      1. Inspection status must be "passed" – any other status is rejected.
      2. demo_eligibility_approved must be True in the inspection fixture.
         The model must never invent or assume this approval.
      3. buyer_id must appear in the ranked_eligible list from
         calculate_resale_matches for this inspection.
      4. An offer for the same inspection_id + buyer_id must not already
         exist (duplicate prevention).

    This creates a SIMULATED OFFER only.  It does NOT:
      • complete a sale or process a payment
      • approve return eligibility
      • issue a refund

    All amounts are SYNTHETIC ASSUMPTIONS.

    Args:
        inspection_id: The inspection ID for the item being offered.
        buyer_id:      The buyer to receive the offer.

    Returns:
        The offer record with ``created`` (True = new, False = duplicate).
        Returns an ``error`` key if any prerequisite is not met.
    """
    print(f"\n[TOOL] Creating resale offer: {inspection_id} → {buyer_id}")

    # Prerequisite 1: inspection must exist and be passed
    insp = INSPECTION_RESULTS.get(inspection_id)
    if insp is None:
        return {"error": "Inspection not found", "inspection_id": inspection_id}
    if insp["status"] != "passed":
        return {
            "error": (
                f"Inspection status is '{insp['status']}' – a resale offer "
                f"requires status 'passed'."
            ),
            "inspection_id": inspection_id,
            "status": insp["status"],
        }

    # Prerequisite 2: explicit demo eligibility approval (fixture data only)
    if not insp.get("demo_eligibility_approved", False):
        return {
            "error": (
                "demo_eligibility_approved is False for this inspection.  "
                "Approval is set in fixture data only; the agent must not "
                "invent or assume it."
            ),
            "inspection_id": inspection_id,
            "demo_eligibility_approved": False,
        }

    # Enforce the configured sale price – model cannot supply its own
    proposed_price = insp["proposed_sale_price_cad"]
    price_err = _validate_price(proposed_price, "proposed_sale_price_cad")
    if price_err:
        return {"error": f"Invalid fixture data: {price_err}"}

    # Prerequisite 3: buyer must be in the eligible match list
    match_result = calculate_resale_matches(inspection_id)
    if "error" in match_result:
        return match_result

    ranked_ids = [b["buyer_id"] for b in match_result["ranked_eligible"]]
    if buyer_id not in ranked_ids:
        exclusion_reasons = [
            e["reasons"]
            for e in match_result["excluded"]
            if e["buyer_id"] == buyer_id
        ]
        return {
            "error": f"Buyer {buyer_id!r} is not in the eligible match list.",
            "inspection_id": inspection_id,
            "buyer_id": buyer_id,
            "exclusion_reasons": exclusion_reasons or ["Not found in buyer pool"],
        }

    # Prerequisite 4: duplicate prevention
    _STATE_DIR.mkdir(exist_ok=True)
    offer_file = _STATE_DIR / f"offer_{inspection_id}_{buyer_id}.json"

    if offer_file.exists():
        existing = json.loads(offer_file.read_text())
        existing["created"] = False
        print(f"\n[TOOL] Reused existing offer: {existing['offer_id']}")
        return existing

    # Create and persist the offer
    offer = {
        "offer_id": f"OFFER-{inspection_id}-{buyer_id}",
        "inspection_id": inspection_id,
        "buyer_id": buyer_id,
        "proposed_sale_price_cad": proposed_price,
        "net_recovery_cad": match_result["net_recovery_cad"],
        "status": "simulated_offer",
        "simulation": True,
        "created": True,
        "disclaimer": (
            "SIMULATED OFFER ONLY – not a completed sale, payment, or return "
            "eligibility approval.  All amounts are SYNTHETIC ASSUMPTIONS.  "
            "Net recovery is an estimated cost-recovery figure, not realized "
            "revenue or profit."
        ),
    }
    offer_file.write_text(json.dumps(offer, indent=2))
    print(f"\n[TOOL] Created resale offer: {offer['offer_id']}")
    return offer


# ---------------------------------------------------------------------------
# Agent factory – lazy, never touches Bedrock at import time
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a retail returns assistant operating on synthetic demo data. "
    "Always call the available tools to retrieve facts before drawing conclusions. "

    # Hard safety rules
    "You must NEVER approve return eligibility or issue a refund. "
    "You must NEVER invent, assume, or imply an inspection result or grade. "
    "You must NEVER invent, assume, or imply demo_eligibility_approved=True. "
    "A non-passed inspection (pending, failed, or other) MUST block resale "
    "matching and offer creation – report the status and stop. "

    # Pricing rule
    "The sale price is fixed in the inspection fixture; never supply or "
    "invent a different price when calling create_resale_offer. "

    # Phase 1 rules
    "Creating an inspection request does NOT complete the inspection, "
    "approve a return, or trigger a refund. "
    "Report created=True (new) or created=False (reused) for inspection requests. "

    # Phase 2 rules
    "A buyer's max_wtp is a price-feasibility ceiling only – it is NOT "
    "evidence of higher net recovery or stronger purchase intent. "
    "Net recovery is an estimated cost-recovery figure, not realized revenue or profit. "
    "All prices, costs, and net-recovery figures from tools are SYNTHETIC ASSUMPTIONS – "
    "label them as such when reporting. "
    "Always explain which buyers were excluded and why. "
    "A resale offer is SIMULATED ONLY – not a completed sale or payment. "

    # Phase 3 rules
    "When calling notify_waitlist_buyers, do NOT supply email addresses, "
    "topic ARNs, or a hand-picked recipient list – the tool computes eligibility itself. "
    "Notifications run in dry_run mode from the agent; live sends require explicit "
    "UI confirmation and cannot be activated by the model. "
    "An SNS MessageId means the message was accepted for delivery by SNS – "
    "it does not mean received, opened, or purchased. "
    "Never describe notifications as completed purchases or realized recovery. "

    "Answer briefly in English."
)

# Maximum agent-loop iterations per invocation.
# Passed via Limits(turns=...) – the documented SDK-supported loop cap.
# One turn = one model call + any tool executions that follow it.
_MAX_TURNS: int = 10


def build_agent():
    """
    Construct and return a Strands Agent backed by Amazon Bedrock (Nova Lite).

    Not called at import time – importing this module never touches Bedrock.
    Requires valid AWS credentials at call time.
    """
    from botocore.config import Config as BotocoreConfig

    from strands import Agent
    from strands.models.bedrock import BedrockModel

    bedrock_client_cfg = BotocoreConfig(
        retries={"mode": "standard", "total_max_attempts": 4},
        connect_timeout=10,
        read_timeout=60,
    )

    model = BedrockModel(
        model_id="global.amazon.nova-2-lite-v1:0",
        region_name="us-east-1",
        boto_client_config=bedrock_client_cfg,
        max_tokens=1000,
    )

    return Agent(
        model=model,
        system_prompt=_SYSTEM_PROMPT,
        tools=[
            get_return_case,
            create_inspection_request,
            get_inspection_result,
            get_eligible_buyers,
            calculate_resale_matches,
            create_resale_offer,
            notify_waitlist_buyers,
        ],
    )


# ---------------------------------------------------------------------------
# Demo prompts
# ---------------------------------------------------------------------------

_PROMPT_SUCCESS = (
    "Demo scenario – successful resale offer and waitlist notification (DEMO-003).\n\n"
    "1. Look up return DEMO-003.\n"
    "2. Retrieve inspection result INSPECT-DEMO-003.\n"
    "3. Calculate resale matches.\n"
    "4. Report the best match, the estimated net recovery, and explain "
    "every excluded buyer.  Label all figures as SYNTHETIC ASSUMPTIONS.\n"
    "5. Create a resale offer for the best-matched buyer.\n"
    "6. Report whether the offer was newly created or reused.\n"
    "7. Notify waitlist buyers for INSPECT-DEMO-003 using the offer ID from step 5.\n"
    "8. Report how many notifications were previewed and for which buyers.\n"
    "Do not approve a refund.  "
    "Do not describe net recovery as profit or revenue.  "
    "Do not describe notifications as purchases."
)

_PROMPT_PENDING = (
    "Demo scenario – pending inspection block (DEMO-001).\n\n"
    "1. Look up return DEMO-001.\n"
    "2. Retrieve inspection result INSPECT-DEMO-001.\n"
    "3. Report whether resale matching can proceed and explain why.\n"
    "Do not invent an inspection outcome or attempt to create an offer."
)


# ── Phase 3: Waitlist notification tool ─────────────────────────────────────

@tool
def notify_waitlist_buyers(inspection_id: str, offer_id: str) -> dict:
    """
    Identify eligible waitlist users and send (or preview) availability notifications.

    This tool recomputes recipient eligibility itself using the existing
    matching logic.  The model must NOT supply email addresses, topic ARNs,
    or a hand-picked list of recipients.

    Prerequisites enforced:
      1. Inspection status must be "passed".
      2. demo_eligibility_approved must be True in the inspection fixture.

    Recipient eligibility requires:
      - active waitlist membership
      - explicit notification consent (wants_notification=True)
      - SKU compatibility
      - accepted condition grade
      - max_wtp_cad >= proposed_sale_price_cad

    Up to five recipients are selected, ordered by headroom (max_wtp − price).
    If fewer than five qualify, only those are notified.

    Mode is dry_run by default.  Live sends require NOTIFY_LIVE=1 in the
    environment AND explicit UI confirmation – the model cannot activate live mode.

    All amounts are SYNTHETIC ASSUMPTIONS.
    An SNS MessageId means accepted for delivery, not received or purchased.

    Args:
        inspection_id: The inspection ID (e.g. INSPECT-DEMO-003).
        offer_id:      The associated resale offer ID for reference in notifications.

    Returns:
        Dict with ``results`` (per-recipient), ``selection`` summary,
        ``dry_run`` flag, ``mode`` label, and ``disclaimer``.
    """
    from notifications import notify_matched_buyers, waitlist_store, sns_service

    insp = INSPECTION_RESULTS.get(inspection_id)
    if insp is None:
        return {"error": "Inspection not found", "inspection_id": inspection_id}

    ret = RETURNS.get(insp.get("return_order_number", ""), {})
    sku          = ret.get("sku", "")
    product_name = ret.get("product", "unknown product")

    entries = waitlist_store.active_consented()

    print(f"\n[TOOL] Notifying waitlist buyers for {inspection_id} "
          f"(offer: {offer_id}, {len(entries)} active+consented waitlist entries)")

    return notify_matched_buyers(
        inspection_id=inspection_id,
        offer_id=offer_id,
        product_name=product_name,
        sku=sku,
        inspection_result=insp,
        waitlist_entries=entries,
        dry_run=True,          # always dry_run from agent; live requires UI confirmation
        sns_service=sns_service,
    )


# ---------------------------------------------------------------------------
# Demo reset helper (no Bedrock call)
# ---------------------------------------------------------------------------

def list_offer_files(inspection_id: str) -> list[Path]:
    """
    Return all simulated offer files in _STATE_DIR for *inspection_id*.

    Only files matching  offer_<inspection_id>_*.json  are returned.
    Inspection request files and offer files for other inspections are
    never touched.
    """
    if not _STATE_DIR.exists():
        return []
    return sorted(_STATE_DIR.glob(f"offer_{inspection_id}_*.json"))


def reset_offers_for_inspection(inspection_id: str) -> dict:
    """
    Delete simulated offer file(s) for *inspection_id* only.

    Inspection request files and offer files belonging to other inspections
    are preserved.  Does NOT call Bedrock.

    Returns:
        Dict with:
          ``deleted``  – list of filenames that were removed
          ``preserved``– list of filenames in _STATE_DIR that were kept
    """
    files_to_delete = list_offer_files(inspection_id)
    deleted = []
    for f in files_to_delete:
        f.unlink()
        deleted.append(f.name)
        print(f"\n[RESET] Deleted offer file: {f.name}")

    preserved = sorted(p.name for p in _STATE_DIR.iterdir()) if _STATE_DIR.exists() else []
    return {"deleted": deleted, "preserved": preserved}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_agent(prompt: str = _PROMPT_SUCCESS) -> None:
    """
    Run the agent with *prompt*.

    Bounded to _MAX_TURNS iterations via Limits(turns=…). Requires AWS credentials.
    """
    from strands.types.agent import Limits

    agent = build_agent()
    print(f"\n[USER] {prompt}\n")
    agent(prompt, limits=Limits(turns=_MAX_TURNS))


if __name__ == "__main__":
    import sys
    scenario = sys.argv[1] if len(sys.argv) > 1 else "success"
    if scenario == "pending":
        run_agent(_PROMPT_PENDING)
    else:
        run_agent(_PROMPT_SUCCESS)
