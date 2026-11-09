"""
notifications.py  –  Waitlist management and SNS notification service
=======================================================================
Provides:

  WaitlistStore            – persist and retrieve waitlist entries (JSON files)
  select_notification_recipients – reuse matching logic; cap at 5; rank by buyer_id
  SnsNotificationService   – dry_run (no AWS call) and live (boto3 SNS) modes
  build_notification_payload – construct message content (no PII, no ARNs)
  notify_matched_buyers    – orchestrate selection + send; called by the agent tool

Design constraints
------------------
• Synthetic data only – no real buyers, email addresses, or purchase records.
• SNS client is created lazily; imports never require AWS credentials.
• dry_run is the default; live mode requires both the NOTIFY_LIVE env var and
  explicit UI confirmation.
• Topic ARNs are loaded from BUYER_SNS_TOPIC_<BUYER_ID> environment variables
  (e.g. BUYER_SNS_TOPIC_BUYER_A). No ARNs are hardcoded in source files.
• Each topic must be dedicated to that demo buyer with only team-controlled,
  confirmed test-email subscriptions.  Do not publish to shared broadcast topics.
• Notification opt-in (wants_notification) is kept separate from acceptance of
  an item's condition grade.
• Notification records are keyed by (inspection_id, buyer_id, offer_version) to
  prevent duplicate sends.  dry-run records are stored separately from live records.
• SNS MessageId means "accepted for delivery", not received, opened, or purchased.
• An ambiguous timeout is recorded as status="unknown", not retried automatically.
• Notification reset is separate from the existing offer-reset operation.

Status values: "previewed" | "accepted_by_sns" | "skipped" | "failed" | "unknown"
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# State directory (shared with agent_strands._STATE_DIR – same physical dir)
# ---------------------------------------------------------------------------

_STATE_DIR: Path = Path(__file__).resolve().parent / "demo_state"

# Sub-directories within demo_state/
_WAITLIST_DIR_NAME   = "waitlist"
_NOTIF_DRY_DIR_NAME  = "notifications_dry"
_NOTIF_LIVE_DIR_NAME = "notifications_live"


def _waitlist_dir() -> Path:
    d = _STATE_DIR / _WAITLIST_DIR_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def _notif_dir(dry_run: bool) -> Path:
    name = _NOTIF_DRY_DIR_NAME if dry_run else _NOTIF_LIVE_DIR_NAME
    d = _STATE_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Waitlist store
# ---------------------------------------------------------------------------

class WaitlistStore:
    """
    Persist and retrieve waitlist entries as JSON files.

    Each entry is keyed by (buyer_id, sku) to avoid duplicates.
    A re-submission by the same buyer for the same SKU updates the entry.
    """

    def upsert(
        self,
        buyer_id: str,
        sku: str,
        accepted_grades: list[str],
        max_wtp_cad: float,
        wants_notification: bool,
        joined_at: str | None = None,
    ) -> dict:
        """
        Insert or update a waitlist entry for (buyer_id, sku).

        ``wants_notification`` records explicit notification consent.
        It is distinct from accepted_grades (condition acceptance).

        Args:
            buyer_id:           Unique buyer identifier.
            sku:                SKU the buyer is interested in.
            accepted_grades:    Condition grades the buyer accepts (e.g. ["A","B"]).
            max_wtp_cad:        Max willingness-to-pay ceiling in CAD (SYNTHETIC).
            wants_notification: Explicit consent to receive availability notifications.
            joined_at:          ISO-8601 timestamp; defaults to now (UTC).

        Returns:
            The upserted entry dict with ``is_update`` flag.
        """
        ts = joined_at or datetime.now(timezone.utc).isoformat()
        path = _waitlist_dir() / f"waitlist_{buyer_id}_{sku}.json"
        is_update = path.exists()
        entry = {
            "buyer_id":            buyer_id,
            "sku":                 sku,
            "accepted_grades":     accepted_grades,
            "max_wtp_cad":         max_wtp_cad,         # SYNTHETIC CEILING
            "wants_notification":  wants_notification,
            "joined_at":           ts,
            "active":              True,
            "is_update":           is_update,
        }
        path.write_text(json.dumps(entry, indent=2))
        return entry

    def get(self, buyer_id: str, sku: str) -> dict | None:
        """Return the waitlist entry for (buyer_id, sku), or None."""
        path = _waitlist_dir() / f"waitlist_{buyer_id}_{sku}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def all_entries(self) -> list[dict]:
        """Return all waitlist entries."""
        d = _waitlist_dir()
        return [json.loads(f.read_text()) for f in sorted(d.glob("waitlist_*.json"))]

    def active_consented(self) -> list[dict]:
        """Return entries where active=True and wants_notification=True."""
        return [
            e for e in self.all_entries()
            if e.get("active") and e.get("wants_notification")
        ]

    def delete(self, buyer_id: str, sku: str) -> bool:
        """Delete a waitlist entry. Returns True if deleted, False if not found."""
        path = _waitlist_dir() / f"waitlist_{buyer_id}_{sku}.json"
        if path.exists():
            path.unlink()
            return True
        return False

    def list_notification_files(
        self, inspection_id: str, *, dry_run: bool
    ) -> list[Path]:
        """Return notification record files for *inspection_id*."""
        d = _notif_dir(dry_run)
        return sorted(d.glob(f"notif_{inspection_id}_*.json"))

    def reset_notifications(
        self, inspection_id: str, *, dry_run: bool
    ) -> dict:
        """
        Delete notification records for *inspection_id*.
        Does not touch waitlist entries, offer files, or inspection files.
        """
        files = self.list_notification_files(inspection_id, dry_run=dry_run)
        deleted = []
        for f in files:
            f.unlink()
            deleted.append(f.name)
        preserved = [
            p.name
            for p in _notif_dir(dry_run).iterdir()
        ] if _notif_dir(dry_run).exists() else []
        return {"deleted": deleted, "preserved": preserved}


# Default shared instance used by tools and UI
_store = WaitlistStore()


# ---------------------------------------------------------------------------
# Recipient selection
# ---------------------------------------------------------------------------

def select_notification_recipients(
    inspection_result: dict,
    sku: str,
    proposed_price_cad: float,
    waitlist_entries: list[dict],
    max_recipients: int = 5,
) -> dict:
    """
    Select up to *max_recipients* waitlist users to notify.

    Eligibility requires ALL of:
      1. active=True in waitlist entry
      2. wants_notification=True (explicit notification consent)
      3. SKU matches
      4. Inspection grade is in accepted_grades
      5. max_wtp_cad >= proposed_price_cad  (price ceiling check)

    Ranking uses buyer_id alphabetically as the sole deterministic tiebreak.
    All qualifying buyers have equal standing at the same proposed price.
    This is NOT a prediction of purchase likelihood.

    If fewer than max_recipients qualify, only those are selected.
    Recipients are never invented.

    Args:
        inspection_result:  Dict from INSPECTION_RESULTS (must have status="passed").
        sku:                Item SKU.
        proposed_price_cad: Proposed resale price in CAD (SYNTHETIC).
        waitlist_entries:   List of waitlist entry dicts to filter.
        max_recipients:     Hard cap (default 5).

    Returns:
        Dict with:
          ``selected``   – list of entry dicts (up to max_recipients)
          ``excluded``   – list of dicts with buyer_id and reasons
          ``total_eligible`` – count before capping
          ``capped``     – True if the cap was applied
    """
    grade = inspection_result.get("grade")

    selected = []
    excluded = []

    for entry in waitlist_entries:
        reasons = []
        if not entry.get("active"):
            reasons.append("not active on waitlist")
        if not entry.get("wants_notification"):
            reasons.append("notification consent not given")
        if entry.get("sku") != sku:
            reasons.append(
                f"SKU {entry.get('sku')!r} does not match item SKU {sku!r}"
            )
        if grade not in (entry.get("accepted_grades") or []):
            reasons.append(
                f"grade {grade!r} not in accepted_grades "
                f"{entry.get('accepted_grades')}"
            )
        if (entry.get("max_wtp_cad") or 0) < proposed_price_cad:
            reasons.append(
                f"max_wtp {entry.get('max_wtp_cad'):.2f} CAD < "
                f"proposed price {proposed_price_cad:.2f} CAD (SYNTHETIC)"
            )
        if reasons:
            excluded.append({"buyer_id": entry["buyer_id"], "reasons": reasons})
        else:
            selected.append(entry)

    # Deterministic ranking: buyer_id alphabetical
    selected.sort(key=lambda e: e["buyer_id"])
    total_eligible = len(selected)
    capped = total_eligible > max_recipients
    selected = selected[:max_recipients]

    return {
        "selected":       selected,
        "excluded":       excluded,
        "total_eligible": total_eligible,
        "capped":         capped,
    }


# ---------------------------------------------------------------------------
# Notification payload builder
# ---------------------------------------------------------------------------

def build_notification_payload(
    entry: dict,
    inspection_result: dict,
    sku: str,
    product_name: str,
    offer_id: str | None,
) -> dict:
    """
    Build the notification message content for one recipient.

    Does NOT include:
    - Other buyers' details
    - Internal net-recovery estimates
    - Claims that a purchase has occurred
    - Topic ARNs or checkout URLs

    All prices are labelled SYNTHETIC.

    Args:
        entry:             Waitlist entry for the recipient.
        inspection_result: Passed inspection dict.
        sku:               Item SKU.
        product_name:      Human-readable product name.
        offer_id:          Optional associated offer ID.

    Returns:
        Dict containing message subject and body (no ARNs or credentials).
    """
    price  = inspection_result.get("proposed_sale_price_cad")
    grade  = inspection_result.get("grade")
    notes  = inspection_result.get("notes", "")
    notification_id = str(uuid.uuid4())

    subject = f"[SYNTHETIC DEMO] {product_name} ({sku}) is available for resale"
    body = (
        f"Hi {entry['buyer_id']},\n\n"
        f"A return you may be interested in has passed inspection and is available "
        f"for resale.\n\n"
        f"Item: {product_name}\n"
        f"SKU: {sku}\n"
        f"Condition grade: {grade}  (your accepted grades: "
        f"{entry.get('accepted_grades')})\n"
        f"Proposed price: CAD {price:.2f}  *** SYNTHETIC ASSUMPTION – demo only ***\n"
        f"Inspection notes: {notes}\n"
        f"Offer reference: {offer_id or 'N/A'}\n"
        f"Notification ID: {notification_id}\n\n"
        "To view or accept this item, contact the demo team directly.\n"
        "This message was generated by a hackathon prototype using synthetic data.\n"
        "It does not confirm a purchase, payment, or product reservation.\n"
        "SNS MessageId, if present, indicates the message was accepted for delivery "
        "by SNS — it does not mean the message was received, opened, or acted upon."
    )
    return {
        "notification_id": notification_id,
        "buyer_id":        entry["buyer_id"],
        "sku":             sku,
        "product_name":    product_name,
        "grade":           grade,
        "proposed_price_cad": price,
        "offer_id":        offer_id,
        "subject":         subject,
        "body":            body,
    }


# ---------------------------------------------------------------------------
# SNS notification service
# ---------------------------------------------------------------------------

# Environment variable that enables live sends.
# Must be set to exactly "1" to activate live mode.
_NOTIFY_LIVE_ENV = "NOTIFY_LIVE"

# Topic ARN environment variable prefix.
# For BUYER-A: BUYER_SNS_TOPIC_BUYER_A
# Underscores replace hyphens in the buyer_id.
_TOPIC_ENV_PREFIX = "BUYER_SNS_TOPIC_"


def _topic_env_var(buyer_id: str) -> str:
    """Return the environment variable name for *buyer_id*'s SNS topic ARN."""
    return _TOPIC_ENV_PREFIX + buyer_id.upper().replace("-", "_")


def _get_topic_arn(buyer_id: str) -> str | None:
    """Return the SNS topic ARN for *buyer_id* from environment, or None."""
    return os.environ.get(_topic_env_var(buyer_id))


class SnsNotificationService:
    """
    Send or preview availability notifications via Amazon SNS.

    Modes
    -----
    dry_run=True  (default): Build payloads, record previewed records, make NO
                  AWS/SNS call.  Safe to run without credentials.
    dry_run=False:           Publish via boto3 SNS Publish API using the
                  existing AWS credential chain.  Requires NOTIFY_LIVE=1 env
                  var AND a topic ARN configured for the buyer.

    Before publishing, verify_subscription_confirmed() calls
    sns:ListSubscriptionsByTopic to confirm at least one subscription on the
    topic is in "Subscribed" state (SubscriptionArn is a real ARN, not
    "PendingConfirmation" or "Deleted").  If the check fails or finds no
    confirmed subscription, the send is skipped with an explicit reason.

    SNS accepting a message (returning a MessageId) on a topic with no
    confirmed subscription is NOT treated as a successful notification.
    accepted_by_sns means SNS accepted the publish call; it does not mean
    any recipient received or opened the message.

    The SNS client is created lazily on first live send; imports do not require
    credentials.
    """

    def __init__(self, region_name: str = "us-east-1"):
        self._region = region_name
        self._sns_client: Any = None   # created lazily

    def _client(self) -> Any:
        if self._sns_client is None:
            import boto3
            self._sns_client = boto3.client("sns", region_name=self._region)
        return self._sns_client

    def _is_live_enabled(self) -> bool:
        return os.environ.get(_NOTIFY_LIVE_ENV, "0").strip() == "1"

    def verify_subscription_confirmed(self, topic_arn: str) -> dict:
        """
        Check that at least one subscription on *topic_arn* is confirmed.

        Calls sns:ListSubscriptionsByTopic and inspects each subscription's
        SubscriptionArn field:
          - A real ARN (starts with "arn:") means the subscription is confirmed.
          - "PendingConfirmation" means the email link has not been clicked.
          - "Deleted" means the subscription was removed.

        Returns:
            Dict with:
              ``ok``      – True only if at least one confirmed subscription found
              ``reason``  – human-readable explanation when ok=False
              ``subscriptions`` – list of raw subscription dicts from AWS
        """
        try:
            response = self._client().list_subscriptions_by_topic(
                TopicArn=topic_arn
            )
        except Exception as exc:
            return {
                "ok":     False,
                "reason": f"Could not verify subscriptions: {exc}",
                "subscriptions": [],
            }

        subscriptions = response.get("Subscriptions", [])
        confirmed = [
            s for s in subscriptions
            if s.get("SubscriptionArn", "").startswith("arn:")
        ]
        pending = [
            s for s in subscriptions
            if s.get("SubscriptionArn") == "PendingConfirmation"
        ]

        if confirmed:
            return {"ok": True, "reason": None, "subscriptions": subscriptions}

        if pending:
            return {
                "ok":     False,
                "reason": (
                    f"Topic has {len(pending)} subscription(s) with "
                    "SubscriptionArn='PendingConfirmation'. "
                    "The subscriber must click the confirmation email link before "
                    "messages can be delivered. Skip this recipient until confirmed."
                ),
                "subscriptions": subscriptions,
            }

        if subscriptions:
            return {
                "ok":     False,
                "reason": (
                    f"Topic has {len(subscriptions)} subscription(s) but none are "
                    "confirmed (all are Deleted or in an unknown state). "
                    "Check the topic's subscription list in the AWS console."
                ),
                "subscriptions": subscriptions,
            }

        return {
            "ok":     False,
            "reason": (
                "Topic has no subscriptions. "
                "Add and confirm a test-email subscription before sending."
            ),
            "subscriptions": [],
        }

    def send(
        self,
        payload: dict,
        inspection_id: str,
        offer_version: str,
        dry_run: bool = True,
    ) -> dict:
        """
        Send (or preview) one notification for *payload["buyer_id"]*.

        Duplicate prevention: if a record for
        (inspection_id, buyer_id, offer_version) already exists in the target
        record directory, skip and return status="skipped".

        For live sends, verify_subscription_confirmed() is called before
        publish().  If no confirmed subscription is found, the send is skipped
        with status="skipped" and the verification reason.  SNS accepting a
        Publish call on a topic with no confirmed subscriber is not treated
        as a successful notification.

        Args:
            payload:        From build_notification_payload.
            inspection_id:  For deduplication key and record filename.
            offer_version:  Version string (e.g. offer_id or "v1").
            dry_run:        True = preview only; False = publish via SNS.

        Returns:
            Result dict with status, notification_id, and optional message_id.
        """
        buyer_id        = payload["buyer_id"]
        notification_id = payload["notification_id"]
        record_key      = f"notif_{inspection_id}_{buyer_id}_{offer_version}.json"
        record_path     = _notif_dir(dry_run) / record_key

        # Duplicate prevention
        if record_path.exists():
            existing = json.loads(record_path.read_text())
            return {
                "status":          "skipped",
                "reason":          "notification already recorded",
                "notification_id": existing.get("notification_id"),
                "buyer_id":        buyer_id,
                "dry_run":         dry_run,
                "existing_record": record_key,
            }

        base_record = {
            "notification_id": notification_id,
            "inspection_id":   inspection_id,
            "buyer_id":        buyer_id,
            "offer_version":   offer_version,
            "dry_run":         dry_run,
            "subject":         payload["subject"],
            "sku":             payload["sku"],
            "grade":           payload["grade"],
            "proposed_price_cad": payload["proposed_price_cad"],
            "sent_at":         datetime.now(timezone.utc).isoformat(),
        }

        if dry_run:
            record = {**base_record, "status": "previewed", "message_id": None}
            record_path.write_text(json.dumps(record, indent=2))
            return {**record, "dry_run": True}

        # ── Live send ────────────────────────────────────────────────────────
        if not self._is_live_enabled():
            return {
                "status":  "skipped",
                "reason":  (
                    f"Live mode is disabled. Set {_NOTIFY_LIVE_ENV}=1 to enable."
                ),
                "buyer_id":  buyer_id,
                "dry_run":   False,
            }

        topic_arn = _get_topic_arn(buyer_id)
        if not topic_arn:
            env_var = _topic_env_var(buyer_id)
            record = {
                **base_record,
                "status": "failed",
                "reason": (
                    f"No SNS topic ARN configured for {buyer_id}. "
                    f"Set environment variable {env_var}=arn:aws:sns:…"
                ),
                "message_id": None,
            }
            record_path.write_text(json.dumps(record, indent=2))
            return {**record, "dry_run": False}

        # ── Subscription check before publish ────────────────────────────────
        sub_check = self.verify_subscription_confirmed(topic_arn)
        if not sub_check["ok"]:
            record = {
                **base_record,
                "status": "skipped",
                "reason": (
                    "Subscription check failed — not publishing. "
                    + sub_check["reason"]
                ),
                "message_id": None,
            }
            record_path.write_text(json.dumps(record, indent=2))
            return {**record, "dry_run": False}

        # ── Publish ──────────────────────────────────────────────────────────
        try:
            response = self._client().publish(
                TopicArn=topic_arn,
                Subject=payload["subject"],
                Message=payload["body"],
            )
            message_id = response.get("MessageId", "unknown")
            record = {
                **base_record,
                "status":     "accepted_by_sns",
                "message_id": message_id,
                "topic_arn":  topic_arn,
            }
            record_path.write_text(json.dumps(record, indent=2))
            return {**record, "dry_run": False}

        except Exception as exc:
            err_str = str(exc)
            # Classify timeout as "unknown" to avoid automatic retry
            is_timeout = any(
                kw in err_str.lower()
                for kw in ("timeout", "timed out", "read timeout")
            )
            record = {
                **base_record,
                "status":     "unknown" if is_timeout else "failed",
                "reason":     err_str,
                "message_id": None,
            }
            record_path.write_text(json.dumps(record, indent=2))
            return {**record, "dry_run": False}


# ---------------------------------------------------------------------------
# High-level orchestration (called by the agent tool and UI)
# ---------------------------------------------------------------------------

def notify_matched_buyers(
    inspection_id: str,
    offer_id: str | None,
    product_name: str,
    sku: str,
    inspection_result: dict,
    waitlist_entries: list[dict],
    dry_run: bool = True,
    sns_service: SnsNotificationService | None = None,
    max_recipients: int = 5,
) -> dict:
    """
    Select eligible waitlist users and send (or preview) notifications.

    Prerequisites (returns error dict if not met):
      1. inspection_result status must be "passed"
      2. inspection_result demo_eligibility_approved must be True

    Args:
        inspection_id:      For deduplication records.
        offer_id:           Associated resale offer ID (or None).
        product_name:       Human-readable product name.
        sku:                Item SKU.
        inspection_result:  Full inspection record dict.
        waitlist_entries:   All entries to filter (from WaitlistStore).
        dry_run:            True = preview only; False = live SNS publish.
        sns_service:        SnsNotificationService instance; created if None.
        max_recipients:     Hard cap on recipients (default 5).

    Returns:
        Dict with:
          ``results``         – per-recipient result dicts
          ``selection``       – output of select_notification_recipients
          ``dry_run``         – bool
          ``disclaimer``      – mandatory label
    """
    # Prerequisite 1
    if inspection_result.get("status") != "passed":
        return {
            "error": (
                f"Inspection status is '{inspection_result.get('status')}' – "
                "notifications require status 'passed'."
            ),
            "inspection_id": inspection_id,
        }

    # Prerequisite 2
    if not inspection_result.get("demo_eligibility_approved"):
        return {
            "error": (
                "demo_eligibility_approved is False – "
                "notifications require explicit eligibility approval."
            ),
            "inspection_id": inspection_id,
        }

    proposed_price = inspection_result.get("proposed_sale_price_cad")

    selection = select_notification_recipients(
        inspection_result=inspection_result,
        sku=sku,
        proposed_price_cad=proposed_price,
        waitlist_entries=waitlist_entries,
        max_recipients=max_recipients,
    )

    if sns_service is None:
        sns_service = SnsNotificationService()

    # offer_version: use offer_id if available, else inspection_id + "v1"
    offer_version = offer_id or f"{inspection_id}-v1"

    results = []
    for entry in selection["selected"]:
        payload = build_notification_payload(
            entry=entry,
            inspection_result=inspection_result,
            sku=sku,
            product_name=product_name,
            offer_id=offer_id,
        )
        result = sns_service.send(
            payload=payload,
            inspection_id=inspection_id,
            offer_version=offer_version,
            dry_run=dry_run,
        )
        results.append(result)

    mode_label = "DRY RUN – no AWS call made" if dry_run else "LIVE – published to SNS"
    return {
        "results":   results,
        "selection": selection,
        "dry_run":   dry_run,
        "mode":      mode_label,
        "disclaimer": (
            "All item details and prices are SYNTHETIC ASSUMPTIONS for demo purposes. "
            "An SNS MessageId indicates the message was accepted for delivery by SNS; "
            "it does not mean received, opened, or purchased. "
            "This is not a marketing or sales system."
        ),
    }


# ---------------------------------------------------------------------------
# Default shared instances
# ---------------------------------------------------------------------------

waitlist_store  = _store
sns_service     = SnsNotificationService()
