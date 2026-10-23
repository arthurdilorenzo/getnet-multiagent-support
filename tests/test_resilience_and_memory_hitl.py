"""Resilience, external API outage honesty, multi-turn SQLite memory, and end-to-end HITL lifecycle tests."""

from unittest.mock import patch
import pytest
from app.agents.orchestrator import MultiAgentOrchestrator
from app.schemas import ChatRequest
from app.observability.resilience import (
    breaker_registry,
    call_with_resilience,
    CircuitBreakerOpenError,
    ExternalServiceUnavailableError,
)
from app.memory.store import StateStore
from app.hitl.queue import HITLQueueManager


def test_weather_outage_never_fabricates_temperature():
    """CRITICAL GATE: When Open-Meteo is down, the system must abstain honestly and NEVER invent temperatures."""
    breaker_registry.reset_all()
    orch = MultiAgentOrchestrator()

    with patch("app.tools.search_tools.httpx.Client.get", side_effect=Exception("Connection timeout to Open-Meteo")):
        resp = orch.process(
            ChatRequest(
                message="Qual é a previsão do tempo em Porto Alegre para amanhã?",
                user_id="cliente1988",
                session_id="sess_weather_outage_test",
            )
        )

    reply = resp.response
    assert resp.abstained is True
    assert "indispon" in reply.lower() or "falha" in reply.lower() or "não foi possível" in reply.lower()
    # Must NEVER invent fake fallback temperatures or fake cache sources
    assert "24.5" not in reply
    assert "Serviço Meteorológico Regional (Cache)" not in reply
    breaker_registry.reset_all()


def test_currency_outage_never_fabricates_exchange_rate():
    """CRITICAL GATE: When AwesomeAPI and Frankfurter are down, the system must NEVER invent exchange rates."""
    breaker_registry.reset_all()
    orch = MultiAgentOrchestrator()

    with patch("app.tools.search_tools.httpx.Client.get", side_effect=Exception("DNS failure")):
        resp = orch.process(
            ChatRequest(
                message="Qual é a cotação do euro hoje?",
                user_id="cliente1988",
                session_id="sess_fx_outage_test",
            )
        )

    reply = resp.response
    assert resp.abstained is True
    assert "indispon" in reply.lower() or "falha" in reply.lower() or "não foi possível" in reply.lower()
    assert "6,18" not in reply
    assert "6.18" not in reply
    assert "Banco Central / Mercado Financeiro" not in reply
    breaker_registry.reset_all()


def test_circuit_breaker_opens_after_threshold_failures():
    """Verify CircuitBreaker transitions from CLOSED to OPEN after repeated failures."""
    breaker_registry.reset_all()
    cb = breaker_registry.get("weather_api")
    assert cb.state == "CLOSED"

    def _failing_fn():
        raise RuntimeError("Simulated upstream failure")

    for _ in range(cb.failure_threshold):
        with pytest.raises(ExternalServiceUnavailableError):
            call_with_resilience("weather_api", _failing_fn, max_retries=0)

    assert cb.state == "OPEN"
    with pytest.raises(CircuitBreakerOpenError):
        call_with_resilience("weather_api", _failing_fn, max_retries=0)
    breaker_registry.reset_all()


def test_multi_turn_sqlite_session_memory_persistence():
    """Verify session_id persists multi-turn conversation in SQLite and passes context across turns."""
    orch = MultiAgentOrchestrator()
    session_id = "sess_cliente1988_multiturn_01"

    # Turn 1: Merchant asks about account status
    res1 = orch.process(
        ChatRequest(
            message="Qual é o status do meu cadastro e meu plano de recebimento?",
            user_id="cliente1988",
            session_id=session_id,
        )
    )
    assert res1.session_id == session_id
    assert "CUSTOMER_SUPPORT_AGENT" in res1.agent_used

    # Turn 2: Follow-up on the same session_id
    res2 = orch.process(
        ChatRequest(
            message="E quando cai o dinheiro das minhas vendas de ontem?",
            user_id="cliente1988",
            session_id=session_id,
        )
    )
    assert res2.session_id == session_id
    assert "get_sales_settlement" in res2.tools_called
    assert "1,840.50" in res2.response or "1,415.80" in res2.response or "LOTE-STL-20261022-881" in res2.response


def test_dynamic_replanning_and_escalation_on_hardware_tamper():
    """Verify supervisor dynamically invokes EscalationAgent when CustomerSupportAgent detects hardware tamper (cliente3012)."""
    orch = MultiAgentOrchestrator()

    res = orch.process(
        ChatRequest(
            message="Minha maquininha caiu no chão e está com erro de tamper na tela, não passa nada!",
            user_id="cliente3012",
            session_id="sess_cliente3012_replan",
        )
    )
    # Must start in CUSTOMER_SUPPORT_AGENT, detect hardware fault / requires_replacement, and dynamically replan to HUMAN_ESCALATION_AGENT
    assert "CUSTOMER_SUPPORT_AGENT" in res.agent_used
    assert "HUMAN_ESCALATION_AGENT" in res.agent_used
    assert res.escalated is True
    assert res.ticket_id is not None and res.ticket_id.startswith("GET-ESC-")


def test_hitl_full_lifecycle_create_and_operator_reply(tmp_path):
    """Verify real HITL ticket persistence in SQLite, operator reply submission, and session history injection."""
    db_file = tmp_path / "test_hitl.db"
    store = StateStore(db_path=db_file)
    hitl = HITLQueueManager(store=store)
    session_id = "sess_cliente1988_hitl_99"

    # 1. Create ticket
    ticket = hitl.create_ticket(
        user_id="cliente1988",
        session_id=session_id,
        category="fee_renegotiation",
        reason="Contestação formal de taxas de antecipação",
        priority="high",
        diagnostic_snapshot={"plan": "Receba Já D+1"},
    )
    ticket_id = ticket.ticket_id
    assert ticket.status == "OPEN"

    # 2. Verify ticket appears in persistent SQLite queue
    open_tickets = hitl.list_tickets(status_filter="OPEN")
    assert any(t.ticket_id == ticket_id for t in open_tickets)

    # 3. Human operator claims and replies to ticket
    claimed = hitl.claim_ticket(ticket_id=ticket_id, operator_id="op_marcos_silva")
    assert claimed is not None and claimed.status == "CLAIMED"

    updated = hitl.reply_to_ticket(
        ticket_id=ticket_id,
        operator_id="op_marcos_silva",
        reply_message="Olá! Aprovamos a redução da sua taxa de antecipação para 1,49% a.m. no contrato vigente.",
        resolve_ticket=True,
    )
    assert updated is not None
    assert updated.status == "RESOLVED"
    assert updated.assigned_operator == "op_marcos_silva"

    # 4. Verify operator reply was injected into the customer's conversation session and status is active
    ctx = store.get_conversation_context(session_id=session_id, user_id="cliente1988")
    assert ctx["status"] == "active"
    messages = ctx["recent_messages"]
    assert any("op_marcos_silva" in m["content"] and "1,49%" in m["content"] for m in messages)
