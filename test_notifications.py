"""
test_notifications.py  –  Offline tests for notifications.py
=============================================================
All tests run without AWS credentials or any network call.
SNS is mocked; no real publishes occur.

Tests cover:
  - waitlist opt-in and active membership requirements
  - up-to-five recipient cap and fewer-than-five behaviour
  - pending inspection and missing approval blocking notify_matched_buyers
  - dry-run makes no SNS call
  - correct buyer-to-topic routing (env var lookup)
  - duplicate-send prevention (skipped status)
  - partial failure and unknown (timeout) outcomes
  - notification reset does not touch offer or inspection files

Run with:
    python -m pytest test_notifications.py -v
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import notifications as notif
from notifications import (
    WaitlistStore,
    SnsNotificationService,
    select_notification_recipients,
    build_notification_payload,
    notify_matched_buyers,
    _topic_env_var,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolated_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect notifications._STATE_DIR to a fresh tmp dir for every test."""
    monkeypatch.setattr(notif, "_STATE_DIR", tmp_path)
    yield tmp_path


@pytest.fixture()
def store(tmp_path: Path) -> WaitlistStore:
    return WaitlistStore()


# Minimal passed inspection fixture reused across tests
_PASSED_INSP = {
    "inspection_id":            "INSPECT-DEMO-003",
    "return_order_number":      "DEMO-003",
    "status":                   "passed",
    "grade":                    "B",
    "proposed_sale_price_cad":  55.00,
    "assumed_costs_cad":        {"shipping": 8.0, "cleaning": 5.0, "handling": 3.0},
    "demo_eligibility_approved": True,
}
_PENDING_INSP = {
    "inspection_id":            "INSPECT-DEMO-001",
    "return_order_number":      "DEMO-001",
    "status":                   "pending",
    "grade":                    None,
    "proposed_sale_price_cad":  None,
    "assumed_costs_cad":        None,
    "demo_eligibility_approved": False,
}
_SKU = "BOOT-TAN-7"


def _make_entry(
    buyer_id: str,
    *,
    sku: str = _SKU,
    grades: list[str] | None = None,
    wtp: float = 60.0,
    opted_in: bool = True,
    active: bool = True,
    wants_notification: bool = True,
) -> dict:
    return {
        "buyer_id":           buyer_id,
        "sku":                sku,
        "accepted_grades":    grades or ["A","B"],
        "max_wtp_cad":        wtp,
        "wants_notification": wants_notification,
        "active":             active,
        "joined_at":          "2026-01-01T00:00:00+00:00",
    }


# ---------------------------------------------------------------------------
# WaitlistStore tests
# ---------------------------------------------------------------------------

def test_upsert_creates_new_entry(store: WaitlistStore):
    entry = store.upsert("BUYER-X","SHOE-WHT-9",["A","B"],55.0,True)
    assert entry["is_update"] is False
    assert entry["buyer_id"] == "BUYER-X"
    assert entry["wants_notification"] is True


def test_upsert_updates_existing_entry(store: WaitlistStore):
    store.upsert("BUYER-X","SHOE-WHT-9",["A"],50.0,True)
    entry = store.upsert("BUYER-X","SHOE-WHT-9",["A","B"],60.0,False)
    assert entry["is_update"] is True
    assert entry["max_wtp_cad"] == 60.0
    assert entry["wants_notification"] is False
    # Only one file on disk (upsert overwrites, not appends)
    assert len(store.all_entries()) == 1


def test_active_consented_filters_correctly(store: WaitlistStore):
    store.upsert("BUYER-A","SKU-1",["B"],55.0,wants_notification=True)
    store.upsert("BUYER-B","SKU-1",["B"],55.0,wants_notification=False)  # no consent
    results = store.active_consented()
    ids = [e["buyer_id"] for e in results]
    assert "BUYER-A" in ids
    assert "BUYER-B" not in ids


# ---------------------------------------------------------------------------
# Opt-in and active membership requirements
# ---------------------------------------------------------------------------

def test_not_opted_in_excluded_from_recipients():
    """wants_notification=False excludes the entry from notification recipients."""
    entries = [_make_entry("BUYER-X", wants_notification=False)]
    result  = select_notification_recipients(_PASSED_INSP, _SKU, 55.0, entries)
    assert len(result["selected"]) == 0
    assert any("consent" in r for e in result["excluded"] for r in e["reasons"])


def test_inactive_excluded_from_recipients():
    """active=False excludes the entry."""
    entries = [_make_entry("BUYER-X", active=False)]
    result  = select_notification_recipients(_PASSED_INSP, _SKU, 55.0, entries)
    assert len(result["selected"]) == 0
    assert any("active" in r for e in result["excluded"] for r in e["reasons"])


def test_sku_mismatch_excluded():
    """Wrong SKU excludes the entry."""
    entries = [_make_entry("BUYER-X", sku="WRONG-SKU")]
    result  = select_notification_recipients(_PASSED_INSP, _SKU, 55.0, entries)
    assert len(result["selected"]) == 0
    assert any("SKU" in r for e in result["excluded"] for r in e["reasons"])


def test_grade_mismatch_excluded():
    """Grade not in accepted_grades excludes the entry."""
    entries = [_make_entry("BUYER-X", grades=["A"])]   # item is grade B
    result  = select_notification_recipients(_PASSED_INSP, _SKU, 55.0, entries)
    assert len(result["selected"]) == 0
    assert any("grade" in r for e in result["excluded"] for r in e["reasons"])


def test_wtp_below_price_excluded():
    """max_wtp < proposed_price excludes the entry."""
    entries = [_make_entry("BUYER-X", wtp=40.0)]       # price is 55.00
    result  = select_notification_recipients(_PASSED_INSP, _SKU, 55.0, entries)
    assert len(result["selected"]) == 0
    assert any("max_wtp" in r for e in result["excluded"] for r in e["reasons"])


# ---------------------------------------------------------------------------
# Up-to-five cap and fewer-than-five behaviour
# ---------------------------------------------------------------------------

def test_six_eligible_capped_at_five():
    """Six qualifying entries → only five selected, capped=True."""
    entries = [_make_entry(f"BUYER-{c}") for c in "ABCDEF"]
    result  = select_notification_recipients(_PASSED_INSP, _SKU, 55.0, entries)
    assert len(result["selected"]) == 5
    assert result["capped"] is True
    assert result["total_eligible"] == 6


def test_two_eligible_selects_two():
    """Two qualifying entries → exactly two selected, capped=False."""
    entries = [_make_entry("BUYER-A"), _make_entry("BUYER-B")]
    result  = select_notification_recipients(_PASSED_INSP, _SKU, 55.0, entries)
    assert len(result["selected"]) == 2
    assert result["capped"] is False


def test_zero_eligible_selects_none():
    """No qualifying entries → empty selected list."""
    result = select_notification_recipients(_PASSED_INSP, _SKU, 55.0, [])
    assert result["selected"] == []
    assert result["capped"] is False


def test_ranking_is_alphabetical_buyer_id():
    """Recipients are ranked alphabetically by buyer_id."""
    entries = [_make_entry("BUYER-C"), _make_entry("BUYER-A"), _make_entry("BUYER-B")]
    result  = select_notification_recipients(_PASSED_INSP, _SKU, 55.0, entries)
    ids     = [e["buyer_id"] for e in result["selected"]]
    assert ids == sorted(ids)


# ---------------------------------------------------------------------------
# Blocking: pending inspection and missing approval
# ---------------------------------------------------------------------------

def test_pending_inspection_blocks_notify():
    entries = [_make_entry("BUYER-A")]
    result  = notify_matched_buyers(
        inspection_id="INSPECT-DEMO-001",
        offer_id=None,
        product_name="Demo Item",
        sku=_SKU,
        inspection_result=_PENDING_INSP,
        waitlist_entries=entries,
    )
    assert "error" in result
    assert "passed" in result["error"]


def test_missing_approval_blocks_notify():
    insp_no_approval = {**_PASSED_INSP, "demo_eligibility_approved": False}
    entries = [_make_entry("BUYER-A")]
    result  = notify_matched_buyers(
        inspection_id="INSPECT-DEMO-003",
        offer_id=None,
        product_name="Demo Item",
        sku=_SKU,
        inspection_result=insp_no_approval,
        waitlist_entries=entries,
    )
    assert "error" in result
    assert "eligibility" in result["error"].lower()


# ---------------------------------------------------------------------------
# Dry-run: no SNS call
# ---------------------------------------------------------------------------

def test_dry_run_makes_no_sns_call():
    """In dry_run mode, the SNS client's publish method must never be called."""
    mock_client = MagicMock()
    svc         = SnsNotificationService()
    svc._sns_client = mock_client   # inject mock before any call

    entries = [_make_entry("BUYER-A")]
    result  = notify_matched_buyers(
        inspection_id="INSPECT-DEMO-003",
        offer_id="OFFER-001",
        product_name="Demo Item",
        sku=_SKU,
        inspection_result=_PASSED_INSP,
        waitlist_entries=entries,
        dry_run=True,
        sns_service=svc,
    )
    mock_client.publish.assert_not_called()
    assert result["dry_run"] is True
    assert result["results"][0]["status"] == "previewed"


# ---------------------------------------------------------------------------
# Correct buyer-to-topic routing
# ---------------------------------------------------------------------------

def test_topic_env_var_name():
    """_topic_env_var converts BUYER-A to BUYER_SNS_TOPIC_BUYER_A."""
    assert _topic_env_var("BUYER-A") == "BUYER_SNS_TOPIC_BUYER_A"
    assert _topic_env_var("BUYER-E") == "BUYER_SNS_TOPIC_BUYER_E"


def test_live_send_uses_correct_topic_arn(monkeypatch: pytest.MonkeyPatch):
    """Live send publishes to the topic ARN configured for the buyer."""
    arn = "arn:aws:sns:us-east-1:123456789012:demo-buyer-a"
    monkeypatch.setenv("BUYER_SNS_TOPIC_BUYER_A", arn)
    monkeypatch.setenv("NOTIFY_LIVE", "1")

    mock_client = MagicMock()
    mock_client.publish.return_value = {"MessageId": "msg-abc-123"}
    # One confirmed subscription
    mock_client.list_subscriptions_by_topic.return_value = {"Subscriptions": [
        {"SubscriptionArn": "arn:aws:sns:us-east-1:123456789012:demo-buyer-a:sub-001",
         "Protocol": "email"}
    ]}

    svc = SnsNotificationService()
    svc._sns_client = mock_client

    payload = build_notification_payload(
        entry=_make_entry("BUYER-A"),
        inspection_result=_PASSED_INSP,
        sku=_SKU,
        product_name="Demo Chelsea Boots",
        offer_id="OFFER-001",
    )
    result = svc.send(
        payload=payload,
        inspection_id="INSPECT-DEMO-003",
        offer_version="OFFER-001",
        dry_run=False,
    )
    mock_client.publish.assert_called_once()
    call_kwargs = mock_client.publish.call_args[1]
    assert call_kwargs["TopicArn"] == arn
    assert result["status"] == "accepted_by_sns"
    assert result["message_id"] == "msg-abc-123"


def test_live_send_fails_if_no_topic_arn_configured(monkeypatch: pytest.MonkeyPatch):
    """Live send returns status='failed' when no ARN env var is set for the buyer."""
    monkeypatch.delenv("BUYER_SNS_TOPIC_BUYER_A", raising=False)
    monkeypatch.setenv("NOTIFY_LIVE", "1")

    svc = SnsNotificationService()
    payload = build_notification_payload(
        entry=_make_entry("BUYER-A"),
        inspection_result=_PASSED_INSP,
        sku=_SKU,
        product_name="Demo Item",
        offer_id=None,
    )
    result = svc.send(
        payload=payload,
        inspection_id="INSPECT-DEMO-003",
        offer_version="v1",
        dry_run=False,
    )
    assert result["status"] == "failed"
    assert "BUYER_SNS_TOPIC_BUYER_A" in result["reason"]


def test_live_send_skipped_when_notify_live_not_set(monkeypatch: pytest.MonkeyPatch):
    """Without NOTIFY_LIVE=1, live send returns status='skipped'."""
    monkeypatch.delenv("NOTIFY_LIVE", raising=False)

    svc = SnsNotificationService()
    payload = build_notification_payload(
        entry=_make_entry("BUYER-A"),
        inspection_result=_PASSED_INSP,
        sku=_SKU,
        product_name="Demo Item",
        offer_id=None,
    )
    result = svc.send(
        payload=payload,
        inspection_id="INSPECT-DEMO-003",
        offer_version="v1",
        dry_run=False,
    )
    assert result["status"] == "skipped"
    assert "NOTIFY_LIVE" in result["reason"]


# ---------------------------------------------------------------------------
# Duplicate-send prevention
# ---------------------------------------------------------------------------

def test_duplicate_dry_run_returns_skipped():
    """Second dry-run for the same (inspection, buyer, version) returns skipped."""
    svc     = SnsNotificationService()
    payload = build_notification_payload(
        entry=_make_entry("BUYER-A"),
        inspection_result=_PASSED_INSP,
        sku=_SKU,
        product_name="Demo Item",
        offer_id="OFFER-001",
    )
    first  = svc.send(payload, "INSPECT-DEMO-003", "OFFER-001", dry_run=True)
    second = svc.send(payload, "INSPECT-DEMO-003", "OFFER-001", dry_run=True)
    assert first["status"]  == "previewed"
    assert second["status"] == "skipped"
    assert "already recorded" in second["reason"]


def test_dry_and_live_records_are_separate(monkeypatch: pytest.MonkeyPatch):
    """Dry-run and live records are written to separate directories; no cross-skip."""
    arn = "arn:aws:sns:us-east-1:123456789012:demo-buyer-a"
    monkeypatch.setenv("BUYER_SNS_TOPIC_BUYER_A", arn)
    monkeypatch.setenv("NOTIFY_LIVE", "1")

    mock_client = MagicMock()
    mock_client.publish.return_value = {"MessageId": "msg-xyz"}
    mock_client.list_subscriptions_by_topic.return_value = {"Subscriptions": [
        {"SubscriptionArn": "arn:aws:sns:us-east-1:123456789012:demo-buyer-a:sub-001",
         "Protocol": "email"}
    ]}

    svc = SnsNotificationService()
    svc._sns_client = mock_client

    payload = build_notification_payload(
        entry=_make_entry("BUYER-A"),
        inspection_result=_PASSED_INSP,
        sku=_SKU,
        product_name="Demo Item",
        offer_id="OFFER-001",
    )
    dry_result  = svc.send(payload, "INSPECT-DEMO-003", "OFFER-001", dry_run=True)
    live_result = svc.send(payload, "INSPECT-DEMO-003", "OFFER-001", dry_run=False)

    assert dry_result["status"]  == "previewed"
    assert live_result["status"] == "accepted_by_sns"


# ---------------------------------------------------------------------------
# Partial failure and unknown (timeout) outcomes
# ---------------------------------------------------------------------------

def test_sns_exception_returns_failed(monkeypatch: pytest.MonkeyPatch):
    """A non-timeout SNS exception is recorded as status='failed'."""
    arn = "arn:aws:sns:us-east-1:123456789012:demo-buyer-a"
    monkeypatch.setenv("BUYER_SNS_TOPIC_BUYER_A", arn)
    monkeypatch.setenv("NOTIFY_LIVE", "1")

    mock_client = MagicMock()
    mock_client.list_subscriptions_by_topic.return_value = {"Subscriptions": [
        {"SubscriptionArn": "arn:aws:sns:us-east-1:123456789012:demo-buyer-a:sub-001",
         "Protocol": "email"}
    ]}
    mock_client.publish.side_effect = Exception("AccessDeniedException")

    svc = SnsNotificationService()
    svc._sns_client = mock_client

    payload = build_notification_payload(
        entry=_make_entry("BUYER-A"),
        inspection_result=_PASSED_INSP,
        sku=_SKU,
        product_name="Demo Item",
        offer_id=None,
    )
    result = svc.send(payload, "INSPECT-DEMO-003", "v1", dry_run=False)
    assert result["status"] == "failed"
    assert "AccessDeniedException" in result["reason"]


def test_timeout_returns_unknown(monkeypatch: pytest.MonkeyPatch):
    """A timeout exception is recorded as status='unknown' (not auto-retried)."""
    arn = "arn:aws:sns:us-east-1:123456789012:demo-buyer-a"
    monkeypatch.setenv("BUYER_SNS_TOPIC_BUYER_A", arn)
    monkeypatch.setenv("NOTIFY_LIVE", "1")

    mock_client = MagicMock()
    mock_client.list_subscriptions_by_topic.return_value = {"Subscriptions": [
        {"SubscriptionArn": "arn:aws:sns:us-east-1:123456789012:demo-buyer-a:sub-001",
         "Protocol": "email"}
    ]}
    mock_client.publish.side_effect = Exception("Read timeout on endpoint")

    svc = SnsNotificationService()
    svc._sns_client = mock_client

    payload = build_notification_payload(
        entry=_make_entry("BUYER-A"),
        inspection_result=_PASSED_INSP,
        sku=_SKU,
        product_name="Demo Item",
        offer_id=None,
    )
    result = svc.send(payload, "INSPECT-DEMO-003", "v1-timeout", dry_run=False)
    assert result["status"] == "unknown", f"Expected unknown, got {result['status']}"


def test_partial_failure_second_buyer_still_notified(monkeypatch: pytest.MonkeyPatch):
    """If the first recipient fails, the second is still processed."""
    arn_a = "arn:aws:sns:us-east-1:123456789012:demo-buyer-a"
    arn_e = "arn:aws:sns:us-east-1:123456789012:demo-buyer-e"
    monkeypatch.setenv("BUYER_SNS_TOPIC_BUYER_A", arn_a)
    monkeypatch.setenv("BUYER_SNS_TOPIC_BUYER_E", arn_e)
    monkeypatch.setenv("NOTIFY_LIVE", "1")

    call_count = {"n": 0}
    def _publish(**kwargs):
        call_count["n"] += 1
        if kwargs.get("TopicArn") == arn_a:
            raise Exception("AccessDenied for BUYER-A")
        return {"MessageId": f"msg-{call_count['n']}"}

    mock_client = MagicMock()
    mock_client.publish.side_effect = _publish
    # Both topics have confirmed subscriptions
    mock_client.list_subscriptions_by_topic.return_value = {"Subscriptions": [
        {"SubscriptionArn": "arn:aws:sns:us-east-1:123456789012:sub-001",
         "Protocol": "email"}
    ]}

    svc = SnsNotificationService()
    svc._sns_client = mock_client

    entries = [
        _make_entry("BUYER-A"),
        _make_entry("BUYER-E"),
    ]
    result = notify_matched_buyers(
        inspection_id="INSPECT-DEMO-003",
        offer_id="OFFER-001",
        product_name="Demo Item",
        sku=_SKU,
        inspection_result=_PASSED_INSP,
        waitlist_entries=entries,
        dry_run=False,
        sns_service=svc,
    )
    statuses = {r["buyer_id"]: r["status"] for r in result["results"]}
    assert statuses["BUYER-A"] == "failed"
    assert statuses["BUYER-E"] == "accepted_by_sns"


# ---------------------------------------------------------------------------
# Subscription verification tests
# ---------------------------------------------------------------------------

def _confirmed_sub_response(arn_prefix: str) -> dict:
    """Helper: mock list_subscriptions_by_topic response with one confirmed sub."""
    return {"Subscriptions": [
        {"SubscriptionArn": f"{arn_prefix}:sub-confirmed-001", "Protocol": "email"}
    ]}


def test_verify_subscription_confirmed_returns_ok_for_confirmed_sub():
    """verify_subscription_confirmed returns ok=True when a real ARN is present."""
    arn = "arn:aws:sns:us-east-1:123456789012:demo-buyer-a"
    mock_client = MagicMock()
    mock_client.list_subscriptions_by_topic.return_value = _confirmed_sub_response(arn)

    svc = SnsNotificationService()
    svc._sns_client = mock_client

    result = svc.verify_subscription_confirmed(arn)
    assert result["ok"] is True
    assert result["reason"] is None
    mock_client.list_subscriptions_by_topic.assert_called_once_with(TopicArn=arn)


def test_verify_subscription_pending_returns_not_ok():
    """verify_subscription_confirmed returns ok=False when sub is PendingConfirmation."""
    arn = "arn:aws:sns:us-east-1:123456789012:demo-buyer-a"
    mock_client = MagicMock()
    mock_client.list_subscriptions_by_topic.return_value = {"Subscriptions": [
        {"SubscriptionArn": "PendingConfirmation", "Protocol": "email"}
    ]}

    svc = SnsNotificationService()
    svc._sns_client = mock_client

    result = svc.verify_subscription_confirmed(arn)
    assert result["ok"] is False
    assert "PendingConfirmation" in result["reason"] or "pending" in result["reason"].lower()


def test_verify_subscription_absent_returns_not_ok():
    """verify_subscription_confirmed returns ok=False when there are no subscriptions."""
    arn = "arn:aws:sns:us-east-1:123456789012:demo-buyer-a"
    mock_client = MagicMock()
    mock_client.list_subscriptions_by_topic.return_value = {"Subscriptions": []}

    svc = SnsNotificationService()
    svc._sns_client = mock_client

    result = svc.verify_subscription_confirmed(arn)
    assert result["ok"] is False
    assert "no subscriptions" in result["reason"].lower()


def test_verify_subscription_api_error_returns_not_ok():
    """verify_subscription_confirmed returns ok=False when the API call fails."""
    arn = "arn:aws:sns:us-east-1:123456789012:demo-buyer-a"
    mock_client = MagicMock()
    mock_client.list_subscriptions_by_topic.side_effect = Exception("AccessDenied")

    svc = SnsNotificationService()
    svc._sns_client = mock_client

    result = svc.verify_subscription_confirmed(arn)
    assert result["ok"] is False
    assert "AccessDenied" in result["reason"]


def test_pending_subscription_skips_send_not_publishes(monkeypatch: pytest.MonkeyPatch):
    """
    When subscription is PendingConfirmation, send() returns status='skipped'
    and publish() is never called.  SNS accepting a publish call on a topic with
    no confirmed subscriber is not treated as a successful notification.
    """
    arn = "arn:aws:sns:us-east-1:123456789012:demo-buyer-a"
    monkeypatch.setenv("BUYER_SNS_TOPIC_BUYER_A", arn)
    monkeypatch.setenv("NOTIFY_LIVE", "1")

    mock_client = MagicMock()
    mock_client.list_subscriptions_by_topic.return_value = {"Subscriptions": [
        {"SubscriptionArn": "PendingConfirmation", "Protocol": "email"}
    ]}
    # publish would succeed if called, but must NOT be called
    mock_client.publish.return_value = {"MessageId": "should-not-appear"}

    svc = SnsNotificationService()
    svc._sns_client = mock_client

    payload = build_notification_payload(
        entry=_make_entry("BUYER-A"),
        inspection_result=_PASSED_INSP,
        sku=_SKU,
        product_name="Demo Item",
        offer_id="OFFER-001",
    )
    result = svc.send(payload, "INSPECT-DEMO-003", "OFFER-001", dry_run=False)

    mock_client.publish.assert_not_called()
    assert result["status"] == "skipped"
    assert "subscription" in result["reason"].lower() or "pending" in result["reason"].lower()


def test_absent_subscription_skips_send_not_publishes(monkeypatch: pytest.MonkeyPatch):
    """When no subscription exists on the topic, send() is skipped, not published."""
    arn = "arn:aws:sns:us-east-1:123456789012:demo-buyer-a"
    monkeypatch.setenv("BUYER_SNS_TOPIC_BUYER_A", arn)
    monkeypatch.setenv("NOTIFY_LIVE", "1")

    mock_client = MagicMock()
    mock_client.list_subscriptions_by_topic.return_value = {"Subscriptions": []}
    mock_client.publish.return_value = {"MessageId": "should-not-appear"}

    svc = SnsNotificationService()
    svc._sns_client = mock_client

    payload = build_notification_payload(
        entry=_make_entry("BUYER-A"),
        inspection_result=_PASSED_INSP,
        sku=_SKU,
        product_name="Demo Item",
        offer_id="OFFER-001",
    )
    result = svc.send(payload, "INSPECT-DEMO-003", "OFFER-001-absent", dry_run=False)

    mock_client.publish.assert_not_called()
    assert result["status"] == "skipped"


def test_subscription_check_api_failure_skips_send(monkeypatch: pytest.MonkeyPatch):
    """When list_subscriptions_by_topic raises, send() skips rather than publishes."""
    arn = "arn:aws:sns:us-east-1:123456789012:demo-buyer-a"
    monkeypatch.setenv("BUYER_SNS_TOPIC_BUYER_A", arn)
    monkeypatch.setenv("NOTIFY_LIVE", "1")

    mock_client = MagicMock()
    mock_client.list_subscriptions_by_topic.side_effect = Exception(
        "AuthorizationError: not authorised to list subscriptions"
    )
    mock_client.publish.return_value = {"MessageId": "should-not-appear"}

    svc = SnsNotificationService()
    svc._sns_client = mock_client

    payload = build_notification_payload(
        entry=_make_entry("BUYER-A"),
        inspection_result=_PASSED_INSP,
        sku=_SKU,
        product_name="Demo Item",
        offer_id="OFFER-001",
    )
    result = svc.send(payload, "INSPECT-DEMO-003", "OFFER-001-apicheck", dry_run=False)

    mock_client.publish.assert_not_called()
    assert result["status"] == "skipped"
    assert "AuthorizationError" in result["reason"]


# ---------------------------------------------------------------------------
# Notification reset does not touch offer or inspection files
# ---------------------------------------------------------------------------

def test_notification_reset_preserves_offer_and_inspection_files(tmp_path: Path):
    """
    reset_notifications deletes only notification records for the given
    inspection; offer and inspection request files are untouched.
    """
    store = WaitlistStore()

    # Plant an offer file and an inspection request file (different naming)
    offer_file = tmp_path / "offer_INSPECT-DEMO-003_BUYER-A.json"
    offer_file.write_text('{"offer_id":"OFFER-TEST"}')
    insp_file  = tmp_path / "inspection_DEMO-003.json"
    insp_file.write_text('{"inspection_id":"INSPECT-DEMO-003"}')

    # Write a dry-run notification record
    svc     = SnsNotificationService()
    payload = build_notification_payload(
        entry=_make_entry("BUYER-A"),
        inspection_result=_PASSED_INSP,
        sku=_SKU,
        product_name="Demo Item",
        offer_id="OFFER-TEST",
    )
    svc.send(payload, "INSPECT-DEMO-003", "OFFER-TEST", dry_run=True)

    # Confirm the notif record was written
    notif_files_before = store.list_notification_files("INSPECT-DEMO-003", dry_run=True)
    assert len(notif_files_before) == 1

    # Reset notifications only
    result = store.reset_notifications("INSPECT-DEMO-003", dry_run=True)
    assert len(result["deleted"]) == 1

    # Offer and inspection files must still exist, unchanged
    assert offer_file.exists(),  "Offer file was incorrectly deleted"
    assert insp_file.exists(),   "Inspection request file was incorrectly deleted"
    assert offer_file.read_text() == '{"offer_id":"OFFER-TEST"}'
    assert insp_file.read_text()  == '{"inspection_id":"INSPECT-DEMO-003"}'

    # Notification record is gone
    assert store.list_notification_files("INSPECT-DEMO-003", dry_run=True) == []


# ---------------------------------------------------------------------------
# Plain runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
