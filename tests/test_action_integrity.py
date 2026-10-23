"""Operational actions must match the authorized target and report real outcomes."""

import pytest
from app.agents.escalation_agent import EscalationAgent
from app.memory.store import StateStore
from app.schemas import ToolExecutionResult
from app.security.auth import SecurityContext
from app.tools import customer_tools
from app.tools.registry import tool_registry


@pytest.mark.parametrize("user_id,terminal_id", [
    ("cliente1988", "GET-CLASSICA-33901"),
    ("cliente1988", "GET-NONEXISTENT-999"),
    ("cliente_inexistente", "GET-SMART-88219"),
])
def test_remote_action_rejects_foreign_or_unknown_terminal_without_writing(tmp_path, monkeypatch, user_id, terminal_id):
    store = StateStore(db_path=tmp_path / "action_test.db")
    monkeypatch.setattr(customer_tools, "state_store", store)
    result = tool_registry.execute(
        "execute_terminal_remote_action",
        {"terminal_id": terminal_id, "action_type": "remote_reboot", "idempotency_key": "wrong-target"},
        security_context=SecurityContext(authenticated_user_id=user_id),
    )
    assert result.success is False
    assert result.error_code == "CROSS_TENANT_ACCESS_DENIED"
    with store._connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM terminal_actions").fetchone()[0] == 0


def test_failed_ticket_creation_abstains_without_claiming_persisted_protocol(monkeypatch):
    monkeypatch.setattr(tool_registry, "execute", lambda **kwargs: ToolExecutionResult(
        tool_name="create_escalation_ticket", success=False, error_code="TOOL_TIMEOUT", retryable=True,
        error_message="Storage unavailable",
    ))
    result = EscalationAgent().run(
        message="Quero falar com atendente humano",
        security_context=SecurityContext(authenticated_user_id="cliente1988"),
        session_id="sess_integrity",
    )
    assert result.abstained is True
    assert not result.structured_data.get("protocol_number")
    assert "GET-ESC-PENDING" not in result.response
    assert "Protocolo Persistido" not in result.response
