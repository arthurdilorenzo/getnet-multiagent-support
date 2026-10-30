"""Counterproofs for Package A: Component Binding, Bootstrap, and Store Isolation."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import pytest

from app.memory.store import StateStore
from app.hitl.queue import HITLQueueManager
from app.v2.adapters import StateAdapter, ToolGateway, IsolatedToolRegistry
from app.security.auth import SecurityContext


def test_tool_gateway_persists_strictly_to_injected_store(tmp_path):
    """Counterproof 1: Tool operations must record in injected store, never in global/operational."""
    # Control DB representing an operational store that should NEVER be touched
    control_db = tmp_path / "operational_control.db"
    isolated_b_db = tmp_path / "arm_b_isolated.db"

    # Customers fixture with test user
    customers_file = tmp_path / "customers.json"
    customers_file.write_text(
        json.dumps({
            "customers": {
                "test_user_01": {
                    "user_id": "test_user_01",
                    "nome": "Padaria Isolada",
                    "terminals": [{"terminal_id": "POS-1000", "model": "Smart"}],
                }
            }
        }),
        encoding="utf-8",
    )

    store_b = StateStore(isolated_b_db)
    queue_b = HITLQueueManager(store=store_b)
    gateway_b = ToolGateway(customers_file, store=store_b, queue=queue_b)

    # Initialize a session in store_b
    store_b.ensure_session("sess_iso_1", "test_user_01")

    sec_ctx = SecurityContext(
        authenticated_user_id="test_user_01",
        role="merchant",
        scopes=["customer:read_own", "chat:write"],
    )

    # Execute tool
    res = gateway_b.execute(
        "get_customer_profile",
        {},
        sec_ctx,
        session_id="sess_iso_1",
    )
    assert res.success is True
    assert res.data["user_id"] == "test_user_01"

    # Verify isolated_b_db has the tool_operations entry
    with store_b._connect() as conn:
        row = conn.execute("SELECT * FROM tool_operations WHERE tool_name='get_customer_profile'").fetchone()
        assert row is not None
        assert row["user_id"] == "test_user_01"
        assert row["outcome_state"] == "completed"

    # Verify control_db was NEVER created or touched
    assert not control_db.exists()


def test_operational_sentinel_blocks_leakage(tmp_path):
    """Counterproof 2: A sentinel operational target fails if opened; v2 gateway bypasses it."""
    class PoisonedStore:
        def __init__(self):
            self._lock = self
        def __enter__(self):
            raise AssertionError("LEAK DETECTED: Operational store was accessed!")
        def __exit__(self, *args):
            pass
        def _connect(self):
            raise AssertionError("LEAK DETECTED: Operational store _connect was called!")

    import app.memory.store
    import app.tools.registry

    orig_memory_store = app.memory.store.state_store
    orig_registry_store = app.tools.registry.state_store

    try:
        # Install poisoned sentinel as the global state_store
        poison = PoisonedStore()
        app.memory.store.state_store = poison
        app.tools.registry.state_store = poison

        # Create isolated environment
        iso_db = tmp_path / "valid_iso.db"
        customers_file = tmp_path / "customers.json"
        customers_file.write_text(json.dumps({"customers": {"u1": {"user_id": "u1"}}}), encoding="utf-8")

        store = StateStore(iso_db)
        queue = HITLQueueManager(store=store)
        gateway = ToolGateway(customers_file, store=store, queue=queue)

        store.ensure_session("s1", "u1")
        sec_ctx = SecurityContext(authenticated_user_id="u1", role="merchant", scopes=["customer:read_own"])

        # Tool execution must succeed without touching the poisoned sentinel
        res = gateway.execute("get_customer_profile", {}, sec_ctx, session_id="s1")
        assert res.success is True

        with store._connect() as conn:
            cnt = conn.execute("SELECT count(*) FROM tool_operations").fetchone()[0]
            assert cnt == 1
    finally:
        app.memory.store.state_store = orig_memory_store
        app.tools.registry.state_store = orig_registry_store


def test_prepare_and_execution_subprocess_isolation(tmp_path):
    """Counterproof 3: Arm A and Arm B prepared and executed in separate subprocesses have zero cross-leak."""
    fixtures_dir = tmp_path / "fixtures"

    # Python script that prepares arms and executes an isolated tool action in B
    script = f"""
import os, sys, json
from pathlib import Path

repo_root = Path('{Path(__file__).resolve().parents[2]}')
sys.path.insert(0, str(repo_root))

from tests.v2.fixture_factory import prepare_arms, get_pilot_cases
from app.memory.store import StateStore
from app.hitl.queue import HITLQueueManager
from app.v2.adapters import StateAdapter, ToolGateway
from app.security.auth import SecurityContext

cases = get_pilot_cases()[:2]
arms = prepare_arms(Path('{fixtures_dir}'), cases)

# In Arm B, record a specific tool operation
store_b = StateStore(Path(arms['B']['sqlite']))
queue_b = HITLQueueManager(store=store_b)
gateway_b = ToolGateway(Path(arms['B']['customers']), store=store_b, queue=queue_b)

sec_b = SecurityContext(authenticated_user_id='lojista_moveis_01', role='merchant', scopes=['customer:read_own'])
res = gateway_b.execute('get_customer_profile', {{}}, sec_b, session_id='pilot_blind_01')
assert res.success is True

# Check that Arm A has 0 tool_operations
store_a = StateStore(Path(arms['A']['sqlite']))
with store_a._connect() as conn:
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    if 'tool_operations' in tables:
        cnt_a = conn.execute('SELECT count(*) FROM tool_operations').fetchone()[0]
        assert cnt_a == 0, f'Expected 0 in Arm A, got {{cnt_a}}'

with store_b._connect() as conn:
    cnt_b = conn.execute('SELECT count(*) FROM tool_operations').fetchone()[0]
    assert cnt_b == 1, f'Expected 1 in Arm B, got {{cnt_b}}'

    sys.stdout.write('SUBPROCESS_ISOLATION_OK\\n')
"""
    cmd = [sys.executable, "-c", script]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert res.returncode == 0, f"Subprocess failed:\nstdout:\n{res.stdout}\nstderr:\n{res.stderr}"
    assert "SUBPROCESS_ISOLATION_OK" in res.stdout


def test_repeated_preparation_preserves_existing_state(tmp_path):
    """Counterproof 4: Repeated prepare does not duplicate history, wipe state, or overwrite mutated cache."""
    from tests.v2.fixture_factory import prepare_arms, get_pilot_cases
    cases = get_pilot_cases()[:2]

    # First prepare
    arms = prepare_arms(tmp_path, cases)
    db_b = Path(arms["B"]["sqlite"])
    cache_b = Path(arms["B"]["embedding_cache"])

    # Simulate an interaction that mutates Arm B
    store_b = StateStore(db_b)
    state_b = StateAdapter(store_b)
    store_b.append_message("pilot_blind_01", "lojista_moveis_01", "assistant", "Mensagem simulada de resposta.")
    state_b.save("pilot_blind_01", "lojista_moveis_01", {"custom_fact": "valor_preservado", "entities": {"terminal_id": "POS-CUSTOM"}})

    # Mutate cache
    cache_data = json.loads(cache_b.read_text(encoding="utf-8"))
    cache_data["vectors"]["mutated_key"] = [0.1] * 384
    cache_b.write_text(json.dumps(cache_data), encoding="utf-8")

    # Count messages before second prepare
    with store_b._connect() as conn:
        msg_count_before = conn.execute("SELECT count(*) FROM messages WHERE session_id='pilot_blind_01'").fetchone()[0]

    # Second prepare on the SAME directory
    prepare_arms(tmp_path, cases)

    # Verify message count is unchanged (no duplication of initial messages)
    with store_b._connect() as conn:
        msg_count_after = conn.execute("SELECT count(*) FROM messages WHERE session_id='pilot_blind_01'").fetchone()[0]
    assert msg_count_before == msg_count_after

    # Verify session facts were not wiped or overwritten
    ctx = state_b.context("pilot_blind_01", "lojista_moveis_01")
    assert ctx["v2"].get("custom_fact") == "valor_preservado"
    assert ctx["v2"].get("entities", {}).get("terminal_id") == "POS-CUSTOM"

    # Verify cache was not overwritten
    reloaded_cache = json.loads(cache_b.read_text(encoding="utf-8"))
    assert "mutated_key" in reloaded_cache["vectors"]
