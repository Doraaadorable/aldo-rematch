"""
test_offline.py  –  Offline tests for agent_strands.py
=======================================================
Currently 22 tool/fixture tests + 6 reset tests = 28 tests total.
All tests run without AWS credentials or any network call.
A temporary directory is used as the state directory so existing
demo_state records are never touched.

Run with:
    python -m pytest test_offline.py -v --import-mode=importlib
or directly:
    python test_offline.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import agent_strands as mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolated_state_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect _STATE_DIR to a fresh tmp directory for every test."""
    monkeypatch.setattr(mod, "_STATE_DIR", tmp_path)
    yield tmp_path


# ---------------------------------------------------------------------------
# Import smoke tests
# ---------------------------------------------------------------------------

def test_imports_agent_and_tool():
    """from strands import Agent, tool must succeed."""
    from strands import Agent, tool  # noqa: F401


def test_imports_bedrock_model():
    """from strands.models.bedrock import BedrockModel must succeed."""
    from strands.models.bedrock import BedrockModel  # noqa: F401


def test_imports_limits():
    """from strands.types.agent import Limits must succeed."""
    from strands.types.agent import Limits  # noqa: F401


# ---------------------------------------------------------------------------
# Phase-1: return lookup and inspection-request tests
# ---------------------------------------------------------------------------

def test_known_return_lookup_succeeds():
    """DEMO-001 lookup returns found=True with the current fixture product name."""
    result = mod.get_return_case("DEMO-001")
    assert result["found"] is True, f"Expected found=True, got: {result}"
    assert result["case"]["return_order_number"] == "DEMO-001"
    assert result["case"]["product"] == "Demo Sneakers"   # current fixture value


def test_unknown_return_lookup_not_found():
    """A fabricated ID returns found=False."""
    result = mod.get_return_case("FAKE-999")
    assert result["found"] is False, f"Expected found=False, got: {result}"


def test_unknown_return_creates_no_inspection_file(tmp_path: Path):
    """Inspection request for a non-existent return writes no file."""
    result = mod.create_inspection_request("FAKE-999")
    assert "error" in result, f"Expected error key, got: {result}"
    assert list(tmp_path.iterdir()) == [], "Expected no files in state dir"


def test_first_inspection_created_true(tmp_path: Path):
    """First create_inspection_request for DEMO-001 returns created=True."""
    result = mod.create_inspection_request("DEMO-001")
    assert "error" not in result, f"Unexpected error: {result}"
    assert result["created"] is True
    assert result["inspection_id"] == "INSPECT-DEMO-001"
    assert result["return_order_number"] == "DEMO-001"
    assert result["status"] == "pending"
    assert (tmp_path / "inspection_DEMO-001.json").exists()


def test_second_inspection_created_false_same_id(tmp_path: Path):
    """Second create_inspection_request for DEMO-001 reuses the record."""
    first = mod.create_inspection_request("DEMO-001")
    second = mod.create_inspection_request("DEMO-001")
    assert second["created"] is False
    assert second["inspection_id"] == first["inspection_id"]
    assert len(list(tmp_path.iterdir())) == 1, "Expected exactly one state file"


def test_demo_state_files_contain_valid_json():
    """
    Any JSON files already in demo_state/ can be parsed without error.
    (This checks structural integrity, not that the tests left them unchanged –
    all test writes go to isolated tmp_path directories, not demo_state/.)
    """
    real_state = Path(__file__).resolve().parent / "demo_state"
    if not real_state.exists():
        return
    for f in real_state.glob("*.json"):
        parsed = json.loads(f.read_text())
        assert parsed, f"{f.name} parsed to a falsy value"


# ---------------------------------------------------------------------------
# Phase-2: buyer eligibility and net-recovery tests
# ---------------------------------------------------------------------------

def test_demo003_buyer_a_and_e_qualify():
    """BUYER-A and BUYER-E both clear opt-in, SKU, grade, and WTP checks."""
    result = mod.calculate_resale_matches("INSPECT-DEMO-003")
    assert "error" not in result, f"Unexpected error: {result}"
    ranked_ids = [b["buyer_id"] for b in result["ranked_eligible"]]
    assert "BUYER-A" in ranked_ids, f"BUYER-A missing from ranked: {ranked_ids}"
    assert "BUYER-E" in ranked_ids, f"BUYER-E missing from ranked: {ranked_ids}"


def test_demo003_buyer_b_excluded_grade():
    """BUYER-B is excluded because accepted_grades=['A'] and the item is grade B."""
    result = mod.calculate_resale_matches("INSPECT-DEMO-003")
    excluded_ids = {e["buyer_id"]: e["reasons"] for e in result["excluded"]}
    assert "BUYER-B" in excluded_ids, f"BUYER-B not in excluded: {excluded_ids}"
    reasons = " ".join(excluded_ids["BUYER-B"])
    assert "grade" in reasons.lower(), f"Expected grade reason, got: {reasons}"


def test_demo003_buyer_c_excluded_wtp():
    """BUYER-C is excluded because max_wtp=50.00 < proposed price 55.00 CAD."""
    result = mod.calculate_resale_matches("INSPECT-DEMO-003")
    excluded_ids = {e["buyer_id"]: e["reasons"] for e in result["excluded"]}
    assert "BUYER-C" in excluded_ids, f"BUYER-C not in excluded: {excluded_ids}"
    reasons = " ".join(excluded_ids["BUYER-C"])
    assert "max_wtp" in reasons.lower() or "50" in reasons, (
        f"Expected WTP reason, got: {reasons}"
    )


def test_demo003_buyer_d_excluded_not_opted_in():
    """BUYER-D is excluded because opted_in=False."""
    result = mod.calculate_resale_matches("INSPECT-DEMO-003")
    excluded_ids = {e["buyer_id"]: e["reasons"] for e in result["excluded"]}
    assert "BUYER-D" in excluded_ids, f"BUYER-D not in excluded: {excluded_ids}"
    reasons = " ".join(excluded_ids["BUYER-D"])
    assert "opted" in reasons.lower(), f"Expected opted-in reason, got: {reasons}"


def test_demo003_net_recovery_is_39():
    """Net recovery = 55.00 - 8.00 - 5.00 - 3.00 = 39.00 CAD (SYNTHETIC)."""
    result = mod.calculate_resale_matches("INSPECT-DEMO-003")
    assert "error" not in result, f"Unexpected error: {result}"
    assert result["net_recovery_cad"] == 39.00, (
        f"Expected 39.00, got {result['net_recovery_cad']}"
    )


def test_demo003_best_match_is_buyer_a():
    """With tied net recovery, buyer_id tiebreak makes BUYER-A the best match."""
    result = mod.calculate_resale_matches("INSPECT-DEMO-003")
    assert result["best_match"] is not None
    assert result["best_match"]["buyer_id"] == "BUYER-A", (
        f"Expected BUYER-A (tiebreak), got {result['best_match']['buyer_id']}"
    )


# ---------------------------------------------------------------------------
# Phase-2: non-passed inspection blocking tests
# ---------------------------------------------------------------------------

def test_pending_inspection_blocks_resale_matching():
    """calculate_resale_matches returns an error when status='pending'."""
    result = mod.calculate_resale_matches("INSPECT-DEMO-001")
    assert "error" in result, f"Expected error for pending inspection: {result}"
    assert result.get("status") == "pending"


def test_failed_inspection_blocks_resale_matching(monkeypatch: pytest.MonkeyPatch):
    """calculate_resale_matches returns an error when status='failed'."""
    patched = dict(mod.INSPECTION_RESULTS["INSPECT-DEMO-001"])
    patched["status"] = "failed"
    monkeypatch.setitem(mod.INSPECTION_RESULTS, "INSPECT-DEMO-001", patched)
    result = mod.calculate_resale_matches("INSPECT-DEMO-001")
    assert "error" in result, f"Expected error for failed inspection: {result}"
    assert "passed" in result["error"]


def test_unknown_status_blocks_resale_matching(monkeypatch: pytest.MonkeyPatch):
    """calculate_resale_matches returns an error for an unrecognised status."""
    patched = dict(mod.INSPECTION_RESULTS["INSPECT-DEMO-001"])
    patched["status"] = "unknown_status"
    monkeypatch.setitem(mod.INSPECTION_RESULTS, "INSPECT-DEMO-001", patched)
    result = mod.calculate_resale_matches("INSPECT-DEMO-001")
    assert "error" in result, f"Expected error for unknown status: {result}"
    assert "passed" in result["error"]


def test_pending_inspection_blocks_offer_creation():
    """create_resale_offer returns an error when inspection is still pending."""
    result = mod.create_resale_offer("INSPECT-DEMO-001", "BUYER-A")
    assert "error" in result, f"Expected error for pending inspection: {result}"
    assert result.get("status") == "pending"


# ---------------------------------------------------------------------------
# Phase-2: eligibility-approval gate
# ---------------------------------------------------------------------------

def test_missing_approval_blocks_offer_creation(monkeypatch: pytest.MonkeyPatch):
    """create_resale_offer is blocked when demo_eligibility_approved=False."""
    patched = dict(mod.INSPECTION_RESULTS["INSPECT-DEMO-003"])
    patched["demo_eligibility_approved"] = False
    monkeypatch.setitem(mod.INSPECTION_RESULTS, "INSPECT-DEMO-003", patched)
    result = mod.create_resale_offer("INSPECT-DEMO-003", "BUYER-A")
    assert "error" in result, f"Expected error when approval=False: {result}"
    assert result.get("demo_eligibility_approved") is False


# ---------------------------------------------------------------------------
# Phase-2: offer creation and duplicate prevention
# ---------------------------------------------------------------------------

def test_offer_creation_succeeds_for_approved_case(tmp_path: Path):
    """create_resale_offer returns created=True for DEMO-003 + BUYER-A."""
    result = mod.create_resale_offer("INSPECT-DEMO-003", "BUYER-A")
    assert "error" not in result, f"Unexpected error: {result}"
    assert result["created"] is True
    assert result["offer_id"] == "OFFER-INSPECT-DEMO-003-BUYER-A"
    assert result["inspection_id"] == "INSPECT-DEMO-003"
    assert result["buyer_id"] == "BUYER-A"
    assert result["proposed_sale_price_cad"] == 55.00
    assert result["net_recovery_cad"] == 39.00
    # Offer file must exist on disk
    offer_file = tmp_path / "offer_INSPECT-DEMO-003_BUYER-A.json"
    assert offer_file.exists(), "Offer file was not written to state dir"


def test_duplicate_offer_returns_same_id_and_created_false(tmp_path: Path):
    """Second call for the same inspection+buyer returns created=False, same offer_id."""
    first = mod.create_resale_offer("INSPECT-DEMO-003", "BUYER-A")
    second = mod.create_resale_offer("INSPECT-DEMO-003", "BUYER-A")
    assert second["created"] is False
    assert second["offer_id"] == first["offer_id"]
    # Still only one file
    offer_files = list(tmp_path.glob("offer_*.json"))
    assert len(offer_files) == 1, f"Expected 1 offer file, found: {offer_files}"


# ---------------------------------------------------------------------------
# Demo reset helper tests
# ---------------------------------------------------------------------------

def test_list_offer_files_empty_when_none_exist(tmp_path: Path):
    """list_offer_files returns an empty list when no offer files are present."""
    files = mod.list_offer_files("INSPECT-DEMO-003")
    assert files == [], f"Expected empty list, got: {files}"


def test_list_offer_files_returns_only_matching_inspection(tmp_path: Path):
    """list_offer_files returns only files for the specified inspection."""
    # Create offers for two different inspections
    mod.create_resale_offer("INSPECT-DEMO-003", "BUYER-A")
    # Simulate an unrelated file for a different inspection
    unrelated = tmp_path / "offer_INSPECT-OTHER_BUYER-Z.json"
    unrelated.write_text('{"offer_id": "OFFER-OTHER"}')

    files = mod.list_offer_files("INSPECT-DEMO-003")
    file_names = [f.name for f in files]
    assert any("INSPECT-DEMO-003" in n for n in file_names), (
        f"Expected DEMO-003 file in {file_names}"
    )
    assert not any("INSPECT-OTHER" in n for n in file_names), (
        f"Unrelated inspection file should not appear: {file_names}"
    )


def test_reset_deletes_offer_for_inspection(tmp_path: Path):
    """reset_offers_for_inspection removes the offer file and reports it deleted."""
    mod.create_resale_offer("INSPECT-DEMO-003", "BUYER-A")
    assert len(mod.list_offer_files("INSPECT-DEMO-003")) == 1

    result = mod.reset_offers_for_inspection("INSPECT-DEMO-003")
    assert len(result["deleted"]) == 1
    assert "offer_INSPECT-DEMO-003_BUYER-A.json" in result["deleted"]
    assert mod.list_offer_files("INSPECT-DEMO-003") == []


def test_reset_preserves_unrelated_files(tmp_path: Path):
    """reset_offers_for_inspection does not delete files for other inspections."""
    # Create offer for DEMO-003
    mod.create_resale_offer("INSPECT-DEMO-003", "BUYER-A")
    # Plant an unrelated file (simulates another inspection's offer)
    unrelated = tmp_path / "offer_INSPECT-OTHER_BUYER-Z.json"
    unrelated.write_text('{"offer_id": "OFFER-OTHER"}')

    mod.reset_offers_for_inspection("INSPECT-DEMO-003")

    # Unrelated file must still exist
    assert unrelated.exists(), "Unrelated offer file was incorrectly deleted"


def test_reset_on_empty_state_dir_is_harmless(tmp_path: Path):
    """reset_offers_for_inspection on a missing state dir returns empty deleted list."""
    # _STATE_DIR is tmp_path (empty), no files created
    result = mod.reset_offers_for_inspection("INSPECT-DEMO-003")
    assert result["deleted"] == []


def test_reset_offer_preserves_inspection_request_file(tmp_path: Path):
    """
    Reset deletes the offer file but leaves the inspection request file intact
    with identical contents.

    Scenario:
      1. create_inspection_request writes  inspection_DEMO-003.json
      2. create_resale_offer writes        offer_INSPECT-DEMO-003_BUYER-A.json
      3. reset_offers_for_inspection deletes the offer file only
      4. inspection_DEMO-003.json still exists and its content is unchanged
    """
    # Step 1 – create the inspection request file
    insp_result = mod.create_inspection_request("DEMO-003")
    assert "error" not in insp_result, f"Unexpected error: {insp_result}"
    insp_file = tmp_path / "inspection_DEMO-003.json"
    assert insp_file.exists(), "Inspection request file was not created"
    insp_contents_before = insp_file.read_text()

    # Step 2 – create the offer file
    offer_result = mod.create_resale_offer("INSPECT-DEMO-003", "BUYER-A")
    assert "error" not in offer_result, f"Unexpected error: {offer_result}"
    offer_file = tmp_path / "offer_INSPECT-DEMO-003_BUYER-A.json"
    assert offer_file.exists(), "Offer file was not created"

    # Step 3 – reset (offer only)
    reset_result = mod.reset_offers_for_inspection("INSPECT-DEMO-003")
    assert "offer_INSPECT-DEMO-003_BUYER-A.json" in reset_result["deleted"], (
        f"Offer file not in deleted list: {reset_result['deleted']}"
    )

    # Step 4 – offer gone, inspection record intact and unchanged
    assert not offer_file.exists(), "Offer file should have been deleted"
    assert insp_file.exists(), "Inspection request file was incorrectly deleted"
    insp_contents_after = insp_file.read_text()
    assert insp_contents_after == insp_contents_before, (
        "Inspection request file contents changed after reset:\n"
        f"  before: {insp_contents_before}\n"
        f"  after:  {insp_contents_after}"
    )


# ---------------------------------------------------------------------------
# Plain runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
