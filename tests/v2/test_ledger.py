"""Unit and boundary contract tests for DurableSQLiteLedger (Package B)."""
import concurrent.futures
import json
from pathlib import Path
import sqlite3
import pytest

from scripts.v2_pilot import DurableSQLiteLedger, PREV_BUILD_DIR, DIRECTED_DIR


def test_ledger_historical_reconciliation(tmp_path):
    """Historical incomplete usage stays unverified and cannot authorize dispatch."""
    db_path = tmp_path / "test_ledger.db"
    ledger = DurableSQLiteLedger(db_path, max_generations_sublimit=12, max_embeddings_sublimit=6)

    ledger.seed_historical_runs(PREV_BUILD_DIR, DIRECTED_DIR)
    summary = ledger.get_summary()
    committed = summary["committed_usage"]

    assert summary["reservations_count"] == 38
    assert committed["total_generations"] == 28
    assert committed["total_embeddings"] == 10
    assert summary["accounting_status"] == "unverified"
    assert summary["remaining_cost_usd"] is None
    assert summary["live_enabled"] is False
    assert committed["input_tokens"] > 70000
    assert committed["estimated_cost_usd"] < 0.10

    # Verify that historical claim is recorded
    with ledger._connect() as conn:
        c8 = conn.execute("SELECT * FROM reservations WHERE claim_id='claim_8_1793284943566'").fetchone()
        assert c8 is not None
        assert c8["claim_id"] == "claim_8_1793284943566"

        c11 = conn.execute("SELECT * FROM reservations WHERE claim_id='claim_11_1793284995528'").fetchone()
        assert c11 is not None
        assert c11["status"] == "HISTORICAL_UNVERIFIED"
        assert c11["actual_input"] is None and c11["actual_output"] is None

        c17 = conn.execute("SELECT * FROM reservations WHERE claim_id='claim_17_1793285088431'").fetchone()
        assert c17 is not None
        assert c17["status"] == "HISTORICAL_UNVERIFIED"
        assert c17["actual_input"] is None and c17["actual_output"] is None

        c33 = conn.execute("SELECT * FROM reservations WHERE claim_id='claim_33_1793285382893'").fetchone()
        assert c33 is not None
        assert c33["status"] == "HISTORICAL_UNVERIFIED"
        assert c33["actual_input"] is None and c33["actual_output"] is None

        # Historical embedding estimates must never be presented as measured usage.
        c2 = conn.execute("SELECT * FROM reservations WHERE claim_id='claim_2_1793284865187'").fetchone()
        assert c2 is not None
        assert c2["status"] == "HISTORICAL_UNVERIFIED"
        assert c2["actual_input"] is None and c2["actual_output"] is None

    assert summary["reserved_cost_upper_usd"] is None
    assert summary["measured_usage"]["embeddings"] == 0
    assert summary["measured_usage"]["embedding_tokens"] == 0
    assert summary["estimated_usage"]["estimated_embedding_tokens"] > 0
    with pytest.raises(RuntimeError, match="unverified"):
        ledger.reserve("generation", 1, 1, "B", "synthetic_no_dispatch", "mock")
    assert ledger.get_summary()["known_requests"] == 38


def test_ledger_reseed_idempotency(tmp_path):
    """Calling seed_historical_runs multiple times does not duplicate entries or costs."""
    db_path = tmp_path / "test_idempotent.db"
    ledger = DurableSQLiteLedger(db_path, max_generations_sublimit=12, max_embeddings_sublimit=6)

    ledger.seed_historical_runs(PREV_BUILD_DIR, DIRECTED_DIR)
    s1 = ledger.get_summary()

    # Re-seed
    ledger.seed_historical_runs(PREV_BUILD_DIR, DIRECTED_DIR)
    s2 = ledger.get_summary()

    assert s1["reservations_count"] == s2["reservations_count"]
    assert s1["committed_usage"] == s2["committed_usage"]


def test_ledger_limits_enforced_at_boundary(tmp_path):
    """Verifies that all limits (gen, emb, input, output, emb_tokens, cost) trigger fail-closed rejection."""
    db_path = tmp_path / "test_limits.db"
    # Create ledger with tight limits for unit testing
    ledger = DurableSQLiteLedger(db_path, max_generations_sublimit=2, max_embeddings_sublimit=2)

    with ledger._connect() as conn:
        conn.execute("""
            UPDATE limits SET max_gen=2, max_emb=2, max_input_tokens=5000, max_output_tokens=3000,
                              max_emb_tokens=200, max_cost_usd=0.01 WHERE id=1
        """)
        conn.commit()

    # Generation 1: OK
    c1 = ledger.reserve("generation", est_input=1000, max_output=1000, arm="B", case_id="c1", model="m")
    ledger.reconcile(c1, actual_input=1000, actual_output=1000, duration_s=1.0, success=True)

    # Generation 2: OK (reaches max_gen=2 and max_directed=2)
    c2 = ledger.reserve("generation", est_input=1000, max_output=1000, arm="B", case_id="c2", model="m")
    ledger.reconcile(c2, actual_input=1000, actual_output=1000, duration_s=1.0, success=True)

    # Generation 3: Must be rejected by both sublimit and global limit
    with pytest.raises(RuntimeError, match="Budget limit exceeded"):
        ledger.reserve("generation", est_input=1000, max_output=1000, arm="B", case_id="c3", model="m")

    # Embedding 1: OK
    e1 = ledger.reserve("embedding", est_input=100, max_output=0, arm="B", case_id="e1", model="m-emb")
    ledger.reconcile(e1, actual_input=100, actual_output=0, duration_s=0.2, success=True)

    # Embedding tokens are isolated from generation's input/output caps.
    usage = ledger.get_summary()["committed_usage"]
    assert usage["input_tokens"] == 2000
    assert usage["output_tokens"] == 2000
    assert usage["embedding_tokens"] == 100

    # Embedding 2: Input exceeds max_emb_tokens=200 (100 + 150 = 250 > 200)
    with pytest.raises(RuntimeError, match="Budget limit exceeded"):
        ledger.reserve("embedding", est_input=150, max_output=0, arm="B", case_id="e2", model="m-emb")


def test_ledger_concurrency_atomic_reservation(tmp_path):
    """Verifies atomic multi-threaded competition for the final reservation slot."""
    db_path = tmp_path / "test_concurrency.db"
    # Sublimit allows exactly 1 reservation
    ledger = DurableSQLiteLedger(db_path, max_generations_sublimit=1, max_embeddings_sublimit=1)

    successes = []
    rejections = []

    def try_reserve(idx):
        try:
            res = ledger.reserve("generation", est_input=1000, max_output=500, arm="B", case_id=f"case_{idx}", model="m")
            successes.append(res["claim_id"])
        except RuntimeError:
            rejections.append(idx)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(try_reserve, i) for i in range(8)]
        concurrent.futures.wait(futures)

    # Exactly 1 reservation succeeded, exactly 7 rejected
    assert len(successes) == 1
    assert len(rejections) == 7


def test_ledger_failure_retains_conservative_upper_bound(tmp_path):
    """A synthetic certified reserve survives timeout without fabricated actual usage."""
    db_path = tmp_path / "test_retention.db"
    ledger = DurableSQLiteLedger(db_path, max_generations_sublimit=5, max_embeddings_sublimit=5)

    claim = ledger.reserve("generation", est_input=2500, max_output=2048, arm="B", case_id="c_err", model="m")

    reserved = ledger.get_summary()
    # Simulate timeout / 504 error before response received
    ledger.reconcile(claim, actual_input=0, actual_output=0, duration_s=30.0, success=False, error_msg="ServerError: 504 DEADLINE_EXCEEDED")

    with ledger._connect() as conn:
        row = conn.execute("SELECT * FROM reservations WHERE claim_id=?", (claim["claim_id"],)).fetchone()
        assert row["status"] == "FAILED_BOUNDED"
        assert row["actual_input"] is None
        assert row["actual_output"] is None
        assert row["est_input"] == 2500
        assert row["est_output"] == 2048
        assert row["cost_usd"] > 0.005
        assert "504" in row["error_msg"]

    retained = ledger.get_summary()
    assert retained["reserved_cost_upper_usd"] == reserved["reserved_cost_upper_usd"]
    assert retained["remaining_cost_usd"] == reserved["remaining_cost_usd"]
    assert retained["measured_usage"]["generations"] == 0
    assert retained["measured_usage"]["input_tokens"] == 0
    assert retained["estimated_usage"]["estimated_input_tokens"] == 2500
    assert retained["estimated_usage"]["estimated_output_tokens"] == 2048


def test_ledger_fail_closed_on_corrupted_db(tmp_path):
    """If database file is corrupted, ledger operations fail closed with an error."""
    db_path = tmp_path / "test_corrupt.db"
    ledger = DurableSQLiteLedger(db_path)
    # Corrupt the file and remove wal sidecars to force DatabaseError
    db_path.write_bytes(b"CORRUPTED_GARBAGE_HEADER_1234567890" * 10)
    for ext in ("-wal", "-shm"):
        f = db_path.with_name(db_path.name + ext)
        if f.exists():
            f.unlink()

    with pytest.raises((sqlite3.DatabaseError, RuntimeError)):
        ledger.reserve("generation", est_input=1000, max_output=500, arm="B", case_id="c1", model="m")
