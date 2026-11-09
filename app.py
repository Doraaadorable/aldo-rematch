"""
app.py  –  Streamlit demo: Return-to-Resale Matching + Waitlist Notifications
==============================================================================
Runs against agent_strands.py (tools) and notifications.py (waitlist / SNS).
The CLI entry point (python agent_strands.py) is preserved unchanged.

Launch (localhost only):
    streamlit run app.py --server.address localhost --server.port 8501

Environment variables (optional):
    NOTIFY_LIVE=1                    – enable live SNS publishes (default: dry-run)
    BUYER_SNS_TOPIC_BUYER_A=arn:…    – SNS topic ARN for BUYER-A
    BUYER_SNS_TOPIC_BUYER_E=arn:…    – SNS topic ARN for BUYER-E

Design rules:
- The agent is NEVER invoked automatically on page load or rerun.
- Live SNS publication requires NOTIFY_LIVE=1 AND explicit UI confirmation.
- All displayed facts come from tool results or persisted state files.
- No success messages are hard-coded.
- All synthetic labels are shown wherever prices/costs appear.
- Model errors are shown explicitly; incomplete runs are not presented as successful.
- Notification reset is separate from offer reset.
- Existing offer/inspection reset behaviour is unchanged.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

import agent_strands as mod
import disposition as dsp
from notifications import (
    WaitlistStore,
    SnsNotificationService,
    select_notification_recipients,
    build_notification_payload,
    notify_matched_buyers,
    _topic_env_var,
    _NOTIFY_LIVE_ENV,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Return-to-Resale Matching Demo",
    page_icon="♻️",
    layout="wide",
)

_SYNTHETIC_BADGE = "All returns, buyers, prices and costs on this page are synthetic fixtures."

st.title("♻️ Return-to-Resale Matching — Demo")
st.caption(_SYNTHETIC_BADGE)

# Shared state dir and store
_state_dir = mod._STATE_DIR
_wl_store   = WaitlistStore()
_sns_svc    = SnsNotificationService()

# ─────────────────────────────────────────────────────────────────────────────
# Section 1 – Scenario selector
# ─────────────────────────────────────────────────────────────────────────────

SCENARIOS: dict[str, str] = {
    "✅  DEMO-003 – Inspection passed — eligibility pre-approved (fixture)": "DEMO-003",
    "⏳  DEMO-001 – Pending inspection (resale blocked)": "DEMO-001",
}

st.subheader("1. Select scenario")
scenario_label = st.radio("Scenario", list(SCENARIOS.keys()), label_visibility="collapsed")
return_id     = SCENARIOS[scenario_label]
inspection_id = f"INSPECT-{return_id}"

ret    = mod.RETURNS.get(return_id, {})
insp   = mod.INSPECTION_RESULTS.get(inspection_id, {})
status = insp.get("status", "–")
sku    = ret.get("sku", "")

status_badge = {
    "passed": "🟢 passed", "pending": "🟡 pending", "failed": "🔴 failed",
}.get(status, f"⚪ {status}")

# ─────────────────────────────────────────────────────────────────────────────
# Section 2 – Return & inspection details
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("2. Return & inspection details")
price = insp.get("proposed_sale_price_cad")
costs = insp.get("assumed_costs_cad") or {}

col_ret, col_insp = st.columns(2)
with col_ret:
    st.markdown("**Return record** *(fixture)*")
    st.table({
        "Field": ["Return #","Product","SKU","Reason","Refund amount","Final sale","Condition"],
        "Value": [
            ret.get("return_order_number","–"), ret.get("product","–"),
            ret.get("sku","–"), ret.get("reason","–"),
            f"{ret.get('refund_amount','–')} {ret.get('currency','')}",
            ret.get("final_sale","–"), ret.get("condition","–"),
        ],
    })

with col_insp:
    st.markdown("**Inspection record** *(fixture)*")
    st.table({
        "Field": [
            "Inspection ID","Status","Grade","Notes",
            "Proposed price","Shipping",
            "Cleaning","Handling","Eligibility approved",
        ],
        "Value": [
            insp.get("inspection_id","–"), status_badge,
            insp.get("grade") or "–", insp.get("notes","–"),
            f"CAD {price:.2f}" if isinstance(price,(int,float)) else "–",
            f"CAD {costs['shipping']:.2f}"  if costs else "–",
            f"CAD {costs['cleaning']:.2f}"  if costs else "–",
            f"CAD {costs['handling']:.2f}"  if costs else "–",
            ("✅ Yes (set in fixture data — agent cannot grant this)"
             if insp.get("demo_eligibility_approved") else "❌ No"),
        ],
    })

# ─────────────────────────────────────────────────────────────────────────────
# Section 3 – Buyer qualification (deterministic Python, no Bedrock)
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("3. Buyer qualification *(deterministic — no model call)*")

if status == "passed":
    match_result = mod.calculate_resale_matches(inspection_id)
    if "error" in match_result:
        st.error(f"Match calculation error: {match_result['error']}")
    else:
        proposed_price = match_result["proposed_sale_price_cad"]
        net_rec        = match_result["net_recovery_cad"]
        costs_used     = match_result["cost_breakdown"]

        st.markdown("**Price and costs**")
        bc1, bc2, bc3, bc4, bc5 = st.columns(5)
        bc1.metric("Proposed price",  f"CAD {proposed_price:.2f}")
        bc2.metric("Shipping",        f"CAD {costs_used['shipping']:.2f}")
        bc3.metric("Cleaning",        f"CAD {costs_used['cleaning']:.2f}")
        bc4.metric("Handling",        f"CAD {costs_used['handling']:.2f}")
        bc5.metric("Net recovery",    f"CAD {net_rec:.2f}",
                   help="Proposed price − shipping − cleaning − handling")

        ranked   = match_result["ranked_eligible"]
        excluded = match_result["excluded"]
        best     = match_result["best_match"]

        st.markdown("**Qualifying buyers**")
        st.caption(
            "A buyer qualifies on SKU preference, accepted grade, opt-in, and a "
            "ceiling (max WTP) at or above the price.  Every qualifying buyer yields "
            "the same net recovery, so they are ordered by headroom (ceiling − price): "
            "the least price-sensitive buyer is asked first."
        )
        if ranked:
            st.dataframe(pd.DataFrame([{
                "Rank": i+1, "Buyer ID": b["buyer_id"], "Name": b["name"],
                "Max WTP": f"CAD {b['max_wtp_cad']:.2f}",
                "Headroom": f"CAD {b['max_wtp_cad'] - proposed_price:.2f}",
                "Selected": ("✅ Best match"
                             if best and b["buyer_id"]==best["buyer_id"] else ""),
            } for i, b in enumerate(ranked)]), hide_index=True, width="stretch")
        else:
            st.warning("No qualifying buyers for this inspection.")

        st.markdown("**Excluded buyers**")
        if excluded:
            st.dataframe(pd.DataFrame([{
                "Buyer ID": e["buyer_id"], "Reasons": "; ".join(e["reasons"])
            } for e in excluded]), hide_index=True, width="stretch")
        else:
            st.info("No buyers were excluded.")
else:
    st.info(
        f"Inspection status is **{status}** – resale matching requires status **passed**.  "
        "The buyer qualification table is not shown."
    )

# ─────────────────────────────────────────────────────────────────────────────
# Section 3b – Disposition decision
# ─────────────────────────────────────────────────────────────────────────────

if status == "passed" and "error" not in match_result:
    st.subheader("3b. Resell, liquidate or donate?")
    st.caption(
        "Expected value per unit under the assumptions below.  Resale pays "
        "cleaning and handling up front; an item that does not sell within the "
        "window pays a holding cost and leaves through the better of liquidation "
        "or donation."
    )

    ceilings = [b["max_wtp_cad"] for b in mod.get_eligible_buyers(sku, insp["grade"])["eligible"]]
    dc1, dc2 = st.columns(2)
    with dc1:
        min_buyers = st.number_input("Buyers whose ceiling must clear the price", 1, 5, 2)
        suggested = dsp.price_from_ceilings(ceilings, int(min_buyers))
        st.caption(
            f"Ceilings of matching buyers: {', '.join(f'{c:.0f}' for c in sorted(ceilings, reverse=True))} CAD.  "
            + (f"Highest price at least {int(min_buyers)} of them clear: **CAD {suggested:.2f}**."
               if suggested is not None else "Not enough matching buyers for that requirement.")
        )
        d_price = st.number_input("Resale price (CAD)", 0.0, 500.0, float(proposed_price), 1.0)
        d_q = st.slider("Sell-through rate within the window", 0.0, 1.0, 0.6, 0.05)
        d_hold = st.number_input("Holding cost if unsold (CAD)", 0.0, 100.0, 4.0, 0.5)
    with dc2:
        d_liq = st.number_input("Liquidation value (CAD)", 0.0, 500.0, 18.0, 1.0)
        d_liq_cost = st.number_input("Liquidation cost (CAD)", 0.0, 100.0, 2.0, 0.5)
        d_don = st.number_input("Donation value, e.g. tax or avoided disposal (CAD)", 0.0, 500.0, 0.0, 1.0)
        d_don_cost = st.number_input("Donation cost (CAD)", 0.0, 100.0, 1.0, 0.5)

    assumptions = dsp.Assumptions(
        resale_price=d_price,
        prep_cost=costs_used["cleaning"] + costs_used["handling"],
        shipping_cost=costs_used["shipping"],
        sell_through=d_q,
        holding_cost=d_hold,
        liquidation_value=d_liq,
        liquidation_cost=d_liq_cost,
        donation_value=d_don,
        donation_cost=d_don_cost,
    )
    decision = dsp.decide(assumptions)
    labels = {"resell": "Resell", "liquidate": "Liquidate", "donate": "Donate"}
    vc = st.columns(4)
    for col, key in zip(vc, ("resell", "liquidate", "donate")):
        col.metric(labels[key], f"CAD {decision['values'][key]:.2f}",
                   delta="best" if decision["choice"] == key else None, delta_color="off")
    q_star = decision["break_even_sell_through"]
    vc[3].metric("Break-even sell-through",
                 "never" if q_star is None or q_star > 1 else f"{q_star:.0%}")
    st.markdown(
        f"**{labels[decision['choice']]}**, ahead of the next option by CAD {decision['margin']:.2f}.  "
        + ("Resale cannot beat the alternatives at this price."
           if q_star is None or q_star > 1 else
           f"Resale wins whenever more than {q_star:.0%} of such items sell within the window.")
    )

# ─────────────────────────────────────────────────────────────────────────────
# Section 4 – Persisted offer state
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("4. Persisted offer state")
offer_pattern = f"offer_{inspection_id}_*.json"
offer_files   = sorted(_state_dir.glob(offer_pattern)) if _state_dir.exists() else []


def _render_offer_card(offer_data: dict, *, source_label: str) -> None:
    oc1, oc2, oc3, oc4 = st.columns(4)
    oc1.metric("Offer ID",                       offer_data.get("offer_id","–"))
    oc2.metric("Buyer",                          offer_data.get("buyer_id","–"))
    oc3.metric("Price",              f"CAD {offer_data.get('proposed_sale_price_cad',0):.2f}")
    oc4.metric("Net recovery",  f"CAD {offer_data.get('net_recovery_cad',0):.2f}")
    st.caption(
        f"{source_label}  |  Status: `{offer_data.get('status','–')}` | "
        f"File: `{offer_data.get('offer_id','unknown')}`  —  "
        + offer_data.get("disclaimer","")
    )


if offer_files:
    st.markdown("**Existing simulated offer(s) on disk** *(read from demo_state/ before agent run)*")
    for f in offer_files:
        _render_offer_card(json.loads(f.read_text()), source_label="📁 Already on disk")
else:
    if status != "passed":
        st.info(
            "No offer exists for this inspection.  "
            "Offer creation is blocked until the inspection passes and "
            "return eligibility is approved."
        )
    else:
        st.info("No persisted offer found for this inspection. Run the agent to create one.")

# ─────────────────────────────────────────────────────────────────────────────
# Section 5 – Waitlist: "Notify me when available"
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("5. Waitlist — Notify me when available *(synthetic buyers only)*")
st.caption(
    "This form uses synthetic buyer identities only. "
    "Do not enter real personal information. "
    "Notification opt-in is separate from acceptance of an item's condition grade."
)

_KNOWN_SKUS = sorted({r["sku"] for r in mod.RETURNS.values()})
_ALL_GRADES  = ["A","B","C"]

with st.form("waitlist_form", clear_on_submit=True):
    wl_col1, wl_col2 = st.columns(2)
    with wl_col1:
        wl_buyer_id   = st.text_input("Buyer ID (synthetic)", placeholder="e.g. BUYER-F")
        wl_sku        = st.selectbox("SKU preference", _KNOWN_SKUS)
        wl_grades     = st.multiselect("Accepted condition grades", _ALL_GRADES, default=["A","B"])
    with wl_col2:
        wl_wtp        = st.number_input(
            "Max willingness-to-pay CAD",
            min_value=0.01, value=60.00, step=1.00,
        )
        wl_notify     = st.checkbox(
            "I opt in to receive availability notifications for this SKU",
            value=True,
        )
        st.caption(
            "Opting in to notifications is separate from accepting an item's condition. "
            "You may still reject an item after seeing the details."
        )
    submitted = st.form_submit_button("Join / update waitlist")

if submitted:
    if not wl_buyer_id.strip():
        st.error("Buyer ID is required.")
    elif not wl_grades:
        st.error("Select at least one accepted grade.")
    else:
        entry = _wl_store.upsert(
            buyer_id=wl_buyer_id.strip(),
            sku=wl_sku,
            accepted_grades=wl_grades,
            max_wtp_cad=float(wl_wtp),
            wants_notification=wl_notify,
        )
        verb = "Updated" if entry["is_update"] else "Added"
        st.success(
            f"{verb} waitlist entry for **{entry['buyer_id']}** / SKU **{entry['sku']}**.  "
            f"Notification consent: {'✅ yes' if entry['wants_notification'] else '❌ no'}"
        )

# Show current waitlist
all_entries = _wl_store.all_entries()
if all_entries:
    st.markdown("**Current waitlist entries** *(persisted locally)*")
    st.dataframe(pd.DataFrame([{
        "Buyer ID":       e["buyer_id"],
        "SKU":            e["sku"],
        "Accepted grades": ", ".join(e.get("accepted_grades",[])),
        "Max WTP": f"CAD {e.get('max_wtp_cad',0):.2f}",
        "Notification opt-in": "✅" if e.get("wants_notification") else "❌",
        "Active":         "✅" if e.get("active") else "❌",
        "Joined":         e.get("joined_at","–")[:19],
    } for e in all_entries]), hide_index=True, width="stretch")
else:
    st.info("No waitlist entries yet. Use the form above to add one.")

# ─────────────────────────────────────────────────────────────────────────────
# Section 6 – Notification preview / send (no Bedrock)
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("6. Notification preview *(deterministic — no model call)*")

live_enabled = os.environ.get(_NOTIFY_LIVE_ENV, "0").strip() == "1"

if status != "passed":
    st.info(
        "Notifications require a passed inspection with eligibility approved.  "
        "Not available for the pending scenario."
    )
elif not insp.get("demo_eligibility_approved"):
    st.warning("demo_eligibility_approved is False – notifications blocked.")
else:
    # Select recipients deterministically
    active_consented = _wl_store.active_consented()
    selection = select_notification_recipients(
        inspection_result=insp,
        sku=sku,
        proposed_price_cad=price,
        waitlist_entries=active_consented,
    )

    st.markdown("**Eligible waitlist recipients** (up to 5, ordered by headroom)")
    st.caption(
        "Eligibility requires active waitlist membership, notification consent, "
        "SKU match, accepted grade, and max WTP ≥ proposed price."
    )

    if selection["selected"]:
        sel_rows = []
        for i, e in enumerate(selection["selected"]):
            sel_rows.append({
                "Rank": i+1, "Buyer ID": e["buyer_id"],
                "SKU": e["sku"],
                "Max WTP": f"CAD {e.get('max_wtp_cad',0):.2f}",
                "Notification consent": "✅",
            })
        st.dataframe(pd.DataFrame(sel_rows), hide_index=True, width="stretch")
        if selection["capped"]:
            st.caption(
                f"ℹ️ {selection['total_eligible']} buyers qualified; "
                "only the first 5 by headroom are shown."
            )
    else:
        st.info("No waitlist users qualify for notification with the current entries.")

    if selection["excluded"]:
        with st.expander("Excluded from notification"):
            st.dataframe(pd.DataFrame([{
                "Buyer ID": e["buyer_id"], "Reasons": "; ".join(e["reasons"])
            } for e in selection["excluded"]]), hide_index=True, width="stretch")

    # Notification preview
    if selection["selected"] and offer_files:
        offer_id = json.loads(offer_files[0].read_text()).get("offer_id","")
    else:
        offer_id = None

    if selection["selected"]:
        st.markdown("**Notification message preview** *(dry-run – no AWS call)*")
        for entry in selection["selected"]:
            payload = build_notification_payload(
                entry=entry,
                inspection_result=insp,
                sku=sku,
                product_name=ret.get("product","–"),
                offer_id=offer_id,
            )
            with st.expander(f"Preview for {entry['buyer_id']}"):
                st.text(f"Subject: {payload['subject']}")
                st.text_area("Message body", payload["body"], height=200,
                             key=f"preview_{entry['buyer_id']}", disabled=True)

        # Topic ARN configuration status
        st.markdown("**SNS topic configuration** *(per-buyer env vars)*")
        arn_rows = []
        for entry in selection["selected"]:
            env_var = _topic_env_var(entry["buyer_id"])
            arn     = os.environ.get(env_var)
            arn_rows.append({
                "Buyer ID": entry["buyer_id"],
                "Env var":  env_var,
                "Configured": "✅ Yes" if arn else "❌ Not set",
                "ARN preview": (arn[:40]+"…" if arn and len(arn)>40 else arn or "–"),
            })
        st.dataframe(pd.DataFrame(arn_rows), hide_index=True, width="stretch")
        st.caption(
            "Each buyer must have a dedicated SNS Standard topic with only "
            "team-controlled, confirmed test-email subscriptions.  "
            "See README for setup instructions."
        )

        # Dry-run send (always safe)
        n_col1, n_col2 = st.columns(2)
        with n_col1:
            dry_run_clicked = st.button("📋  Run dry-run preview", key="dry_run_btn")
        with n_col2:
            if live_enabled:
                live_confirm = st.checkbox(
                    "I confirm: publish live SNS messages to the configured topics",
                    key="live_confirm",
                )
                live_clicked = st.button(
                    "📡  Send live notifications",
                    disabled=not live_confirm,
                    type="primary",
                    key="live_btn",
                )
            else:
                st.info(
                    f"Live sends disabled. Set `{_NOTIFY_LIVE_ENV}=1` to enable."
                )
                live_clicked = False

        def _run_notifications(dry: bool) -> None:
            result = notify_matched_buyers(
                inspection_id=inspection_id,
                offer_id=offer_id,
                product_name=ret.get("product","–"),
                sku=sku,
                inspection_result=insp,
                waitlist_entries=active_consented,
                dry_run=dry,
                sns_service=_sns_svc,
            )
            if "error" in result:
                st.error(result["error"])
                return

            mode_label = result.get("mode","")
            st.markdown(f"**Notification results** – `{mode_label}`")

            status_icon = {
                "previewed":      "📋",
                "accepted_by_sns":"✅",
                "skipped":        "⏭️",
                "failed":         "❌",
                "unknown":        "❓",
            }
            for r in result["results"]:
                s     = r.get("status","unknown")
                icon  = status_icon.get(s,"❓")
                mid   = r.get("message_id")
                buyer = r.get("buyer_id","–")

                if s == "accepted_by_sns":
                    st.success(f"{icon} **{buyer}** – accepted by SNS | MessageId: `{mid}`")
                    st.caption(
                        "MessageId confirms SNS accepted the message for delivery. "
                        "It does not mean the message was received, opened, or purchased."
                    )
                elif s == "previewed":
                    st.info(f"{icon} **{buyer}** – previewed (dry-run, no AWS call made)")
                elif s == "skipped":
                    st.warning(f"{icon} **{buyer}** – skipped: {r.get('reason','already sent')}")
                elif s == "unknown":
                    st.warning(
                        f"{icon} **{buyer}** – outcome unknown (possible timeout). "
                        "Review manually before retrying."
                    )
                elif s == "failed":
                    st.error(f"{icon} **{buyer}** – failed: {r.get('reason','–')}")

            st.caption(result.get("disclaimer",""))

        if dry_run_clicked:
            _run_notifications(dry=True)
        if live_clicked and live_enabled:
            _run_notifications(dry=False)

# ─────────────────────────────────────────────────────────────────────────────
# Section 7 – Demo resets
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("7. Demo resets *(no Bedrock call)*")
reset_col1, reset_col2, reset_col3 = st.columns(3)

# Offer reset
with reset_col1:
    st.markdown("**Reset simulated offer**")
    files_to_reset = mod.list_offer_files(inspection_id)
    if not files_to_reset:
        st.info(f"No offer files for `{inspection_id}`.")
    else:
        names = [f.name for f in files_to_reset]
        st.warning("Will delete:\n" + "\n".join(f"- `{n}`" for n in names))
        offer_confirmed = st.checkbox(
            f"Confirm offer reset for `{inspection_id}`",
            key="offer_reset_confirm",
        )
        if st.button(
            "🗑  Delete offer",
            key="offer_reset_btn",
            disabled=not offer_confirmed,
        ):
            res = mod.reset_offers_for_inspection(inspection_id)
            if res["deleted"]:
                st.success("Deleted: " + ", ".join(f"`{n}`" for n in res["deleted"]))
            else:
                st.error("No files were deleted.")
            st.rerun()

# Dry-run notification reset
with reset_col2:
    st.markdown("**Reset dry-run notification records**")
    dry_notif_files = _wl_store.list_notification_files(inspection_id, dry_run=True)
    if not dry_notif_files:
        st.info("No dry-run notification records.")
    else:
        dry_names = [f.name for f in dry_notif_files]
        st.warning("Will delete:\n" + "\n".join(f"- `{n}`" for n in dry_names))
        dry_confirmed = st.checkbox(
            "Confirm dry-run notification reset",
            key="dry_notif_confirm",
        )
        if st.button(
            "🗑  Reset dry-run notifications",
            key="dry_notif_btn",
            disabled=not dry_confirmed,
        ):
            res = _wl_store.reset_notifications(inspection_id, dry_run=True)
            if res["deleted"]:
                st.success("Deleted: " + ", ".join(f"`{n}`" for n in res["deleted"]))
            else:
                st.error("No files were deleted.")
            st.rerun()

# Live notification reset
with reset_col3:
    st.markdown("**Reset live notification records**")
    live_notif_files = _wl_store.list_notification_files(inspection_id, dry_run=False)
    if not live_notif_files:
        st.info("No live notification records.")
    else:
        live_names = [f.name for f in live_notif_files]
        st.warning("Will delete:\n" + "\n".join(f"- `{n}`" for n in live_names))
        live_confirmed = st.checkbox(
            "Confirm live notification reset",
            key="live_notif_confirm",
        )
        if st.button(
            "🗑  Reset live notifications",
            key="live_notif_btn",
            disabled=not live_confirmed,
        ):
            res = _wl_store.reset_notifications(inspection_id, dry_run=False)
            if res["deleted"]:
                st.success("Deleted: " + ", ".join(f"`{n}`" for n in res["deleted"]))
            else:
                st.error("No files were deleted.")
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# Section 8 – Run agent
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("8. Run agent")
st.caption(
    "The agent is never invoked automatically. Click the button to run it. "
    "Requires valid AWS credentials. Agent notifications always run in dry-run mode; "
    "live sends require explicit UI confirmation above."
)

agent_prompt = mod._PROMPT_SUCCESS if return_id == "DEMO-003" else mod._PROMPT_PENDING
with st.expander("Prompt sent to agent", expanded=False):
    st.code(agent_prompt, language="text")

_pre_run_offer_names: set[str] = (
    {f.name for f in _state_dir.glob(offer_pattern)} if _state_dir.exists() else set()
)

run_clicked = st.button("▶  Run agent", type="primary")

if run_clicked:
    events: list[dict[str, Any]] = []
    text_buffer: list[str] = []
    lock = threading.Lock()

    def _capture_handler(**kwargs: Any) -> None:
        with lock:
            data     = kwargs.get("data","")
            complete = kwargs.get("complete", False)
            if data:
                text_buffer.append(data)
            if complete and text_buffer:
                events.append({"type":"text","content":"".join(text_buffer)})
                text_buffer.clear()
            tool_use = (
                kwargs.get("event",{})
                .get("contentBlockStart",{})
                .get("start",{})
                .get("toolUse")
            )
            if tool_use:
                events.append({"type":"tool_call","name":tool_use.get("name","unknown")})
            result_obj = kwargs.get("result")
            if result_obj is not None:
                try:
                    for block in (getattr(result_obj,"content",None) or []):
                        if isinstance(block,dict) and "json" in block:
                            events.append({"type":"tool_result","data":block["json"]})
                        elif hasattr(block,"json"):
                            events.append({"type":"tool_result","data":block.json})
                except Exception:
                    pass

    with st.spinner("Agent running… (this calls Bedrock)"):
        run_error = None; agent_output = ""; stop_reason = ""
        try:
            from strands.types.agent import Limits
            agent = mod.build_agent()
            agent.callback_handler = _capture_handler
            result_obj  = agent(agent_prompt, limits=Limits(turns=mod._MAX_TURNS))
            agent_output = str(result_obj)
            stop_reason  = str(result_obj.stop_reason)
        except Exception as exc:
            run_error = str(exc)

    if text_buffer:
        events.append({"type":"text","content":"".join(text_buffer)})

    if run_error:
        st.error(f"**Agent error — run did not complete successfully.**\n\n{run_error}")
        st.caption(
            "Common causes: AWS credentials not configured, Bedrock model not "
            "enabled in your region, or network connectivity issue."
        )
        st.stop()

    hit_limit = stop_reason == "limit_turns"
    if hit_limit:
        st.warning(
            f"⚠️ Agent reached the turn limit ({mod._MAX_TURNS} turns). "
            "Response may be incomplete."
        )

    tool_events = [e for e in events if e["type"] in ("tool_call","tool_result")]
    if tool_events:
        st.markdown("**Tool calls & results**")
        tidx = 0
        for ev in tool_events:
            if ev["type"] == "tool_call":
                tidx += 1
                st.markdown(f"🔧 **Tool #{tidx}:** `{ev['name']}`")
            elif ev["type"] == "tool_result":
                with st.expander(f"Tool #{tidx} result", expanded=False):
                    st.json(ev["data"])

    st.markdown("**Agent response**")
    text_events = [e for e in events if e["type"]=="text"]
    combined = "\n".join(e["content"] for e in text_events).strip()
    display  = agent_output.strip() or combined
    if display:
        st.markdown(display)
    elif hit_limit:
        st.warning("No text response captured before the turn limit.")
    else:
        st.info("No text response captured.")

    st.caption(f"Stop reason: `{stop_reason}`")

    st.markdown("**Offer state after run** *(re-read from disk)*")
    for f in sorted(_state_dir.glob(offer_pattern)) if _state_dir.exists() else []:
        offer_data = json.loads(f.read_text())
        was_pre = f.name in _pre_run_offer_names
        if was_pre:
            st.info("📁 Already existed before this run (offer reused)")
        else:
            st.success("🆕 Newly created offer file — written during this run")
        _render_offer_card(offer_data, source_label=(
            "📁 Already existed before run" if was_pre
            else "🆕 Newly created this run"
        ))

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "ReMatch — ALDO Group challenge prototype.  An SNS MessageId confirms the "
    "message was accepted for delivery, not that it was read or led to a purchase."
)
