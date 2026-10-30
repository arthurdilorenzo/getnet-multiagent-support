import pytest
import sqlite3
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

from app.hitl.queue import HITLQueueManager
import app.agents.orchestrator as orchestrator_module
from app.schemas import ChatRequest
from app.memory.store import StateStore
import app.memory.store as store_module


def test_operator_reply_records_guidance_without_claiming_agent_execution_or_plan_completion(tmp_path):
    """
    Detecta regressão onde o orquestrador alega falsamente ter retomado ou concluído
    um plano técnico após resposta do operador, sem executar ferramentas ou diagnóstico real.
    Garante que o mesmo StateStore temporário seja consumido pelo orquestrador e pelo queue.
    """
    db_file = tmp_path / "test_hitl.db"
    store = StateStore(db_path=db_file)
    queue = HITLQueueManager(store=store)

    session_id = "sess_hitl_test_1"
    user_id = "cliente1988"
    store.ensure_session(session_id, user_id)

    # 1. Cria ticket com snapshot de terminal
    ticket = queue.create_ticket(
        session_id=session_id,
        user_id=user_id,
        priority="critical",
        category="terminal_hardware",
        reason="Defeito no leitor de chip",
        diagnostic_snapshot={"terminal_id": "GET-CLASSICA-33901", "status": "Offline"},
    )
    assert ticket is not None

    # 2. Operador responde com diretriz humana única
    reply_msg = "Favor realizar teste de comunicação no menu do terminal antes de qualquer reinício [Protocolo Orion-99]."
    updated = queue.reply_to_ticket(
        ticket_id=ticket.ticket_id,
        operator_id="op_analyst_01",
        reply_message=reply_msg,
        resolve_ticket=True,
    )
    assert updated is not None
    assert updated.status == "RESOLVED"
    assert updated.assigned_operator == "op_analyst_01"
    assert updated.operator_response == reply_msg

    # Prova de persistência através de uma NOVA instância do StateStore sobre o mesmo banco
    store2 = StateStore(db_path=db_file)
    ctx2 = store2.get_conversation_context(session_id, user_id)
    latest_hitl = ctx2.get("session_facts", {}).get("latest_hitl_resumption", "")
    assert reply_msg in latest_hitl
    assert "op_analyst_01" in latest_hitl

    operator_msgs = [m for m in ctx2.get("recent_messages", []) if m["role"] == "human_operator"]
    assert any(reply_msg in m["content"] for m in operator_msgs)

    # agent_resumption_response NÃO pode declarar que o plano foi concluído pelo agente
    resumption = updated.agent_resumption_response
    assert resumption is not None
    assert "concluído pelo agente" not in resumption.lower()
    assert "plano concluído e sessão liberada" not in resumption.lower()
    assert "orientação do especialista registrada" in resumption.lower()

    # 3. No próximo turno, cliente envia "Obrigado"
    # Prova por identidade e caminho que o orquestrador consome a base isolada do teste
    with patch.object(orchestrator_module, "state_store", store), \
         patch.object(store_module, "state_store", store), \
         patch("app.security.action_policy.state_store", store), \
         patch("app.tools.customer_tools.state_store", store), \
         patch("app.tools.customer_tools.hitl_queue", queue):
        assert orchestrator_module.state_store is store
        assert orchestrator_module.state_store.db_path == db_file

        orchestrator = orchestrator_module.MultiAgentOrchestrator()
        req = ChatRequest(message="Obrigado", user_id=user_id, session_id=session_id)
        resp = orchestrator.process(req)

        # Asserção positiva: a orientação humana única DEVE aparecer na resposta ao usuário
        assert reply_msg in resp.response
        # Asserções negativas: nenhuma falsa conclusão
        assert "retomado e concluído pelo agente" not in resp.response.lower()
        assert "plano concluído e sessão liberada" not in resp.response.lower()
        # Nenhuma proposta de ação remota executada
        assert len(resp.action_proposals) == 0


def test_resolving_one_ticket_keeps_session_paused_if_another_is_open(tmp_path):
    """
    Detecta regressão onde a liberação de um ticket resolve indevidamente uma sessão
    que ainda possui outro chamado aberto.
    """
    db_file = tmp_path / "test_hitl_multi.db"
    store = StateStore(db_path=db_file)
    queue = HITLQueueManager(store=store)

    session_id = "sess_multi_tickets"
    user_id = "cliente1988"
    store.ensure_session(session_id, user_id)

    t1 = queue.create_ticket(
        session_id=session_id,
        user_id=user_id,
        priority="high",
        category="terminal",
        reason="Erro leitor",
    )
    t2 = queue.create_ticket(
        session_id=session_id,
        user_id=user_id,
        priority="critical",
        category="pix",
        reason="Chave bloqueada",
    )

    # Resolve t1; t2 continua OPEN
    updated_t1 = queue.reply_to_ticket(
        ticket_id=t1.ticket_id,
        operator_id="op_01",
        reply_message="Orientação leitor dada.",
        resolve_ticket=True,
    )
    assert updated_t1.status == "RESOLVED"

    # A sessão deve permanecer waiting_human porque t2 ainda está OPEN
    session_ctx = store.get_conversation_context(session_id, user_id)
    assert session_ctx["status"] == "waiting_human"


def test_sla_resolution_remains_overdue_for_unresolved_ticket_after_reply(tmp_path):
    """
    Detecta regressão onde responder a um ticket sem resolvê-lo mascara o estouro do SLA
    de resolução ou remove a sessão do estado de espera por humano.
    """
    db_file = tmp_path / "test_hitl_sla.db"
    store = StateStore(db_path=db_file)
    queue = HITLQueueManager(store=store)

    session_id = "sess_sla_test"
    user_id = "cliente1988"
    store.ensure_session(session_id, user_id)

    t = queue.create_ticket(
        session_id=session_id,
        user_id=user_id,
        priority="critical",  # 15 mins
        category="chargeback",
        reason="Contestação urgente",
    )

    # Simula passagem de 20 minutos no banco
    past_iso = (datetime.now(timezone.utc) - timedelta(minutes=20)).isoformat()
    with store._lock, store._connect() as conn:
        conn.execute(
            "UPDATE escalation_tickets SET created_at = ? WHERE ticket_id = ?",
            (past_iso, t.ticket_id),
        )

    # Operador responde com resolve_ticket=False (status RESPONDED)
    updated = queue.reply_to_ticket(
        ticket_id=t.ticket_id,
        operator_id="op_02",
        reply_message="Aguardando comprovante adicional do cliente.",
        resolve_ticket=False,
    )
    assert updated.status == "RESPONDED"
    # Como o SLA é de RESOLUÇÃO, ticket não resolvido após 15 min DEVE permanecer overdue=True
    assert updated.is_overdue is True
    # E a sessão continua waiting_human
    session_ctx = store.get_conversation_context(session_id, user_id)
    assert session_ctx["status"] == "waiting_human"


def test_cross_session_and_ownership_isolation(tmp_path):
    """
    Detecta violação de isolamento entre sessões/usuários ao consultar ou responder tickets.
    """
    db_file = tmp_path / "test_hitl_isolation.db"
    store = StateStore(db_path=db_file)
    queue = HITLQueueManager(store=store)

    store.ensure_session("sess_user_a", "user_a")
    store.ensure_session("sess_user_b", "user_b")

    t_a = queue.create_ticket(
        session_id="sess_user_a",
        user_id="user_a",
        priority="medium",
        category="pix",
        reason="Dúvida Pix",
    )

    assert store.has_pending_human("sess_user_a", "user_a") is True
    assert store.has_pending_human("sess_user_b", "user_b") is False

    # Listagem filtrada por user_id isola tickets de outro usuário
    tickets_a = queue.list_tickets(user_id="user_a")
    tickets_b = queue.list_tickets(user_id="user_b")
    assert len(tickets_a) == 1
    assert len(tickets_b) == 0


def test_waiting_human_session_blocks_remote_action_proposals(tmp_path):
    """
    Detecta tentativa de executar ou propor ações remotas mutáveis enquanto a sessão
    está sob escalonamento humano pendente.
    """
    db_file = tmp_path / "test_hitl_action_blocked.db"
    store = StateStore(db_path=db_file)
    queue = HITLQueueManager(store=store)

    session_id = "sess_blocked_action"
    user_id = "cliente1988"
    store.ensure_session(session_id, user_id)

    queue.create_ticket(
        session_id=session_id,
        user_id=user_id,
        priority="critical",
        category="terminal",
        reason="Terminal travado aguardando técnico",
    )

    with patch.object(orchestrator_module, "state_store", store), \
         patch.object(store_module, "state_store", store), \
         patch("app.security.action_policy.state_store", store), \
         patch("app.tools.customer_tools.state_store", store), \
         patch("app.tools.customer_tools.hitl_queue", queue):
        orchestrator = orchestrator_module.MultiAgentOrchestrator()
        req = ChatRequest(message="Reinicie o terminal agora", user_id=user_id, session_id=session_id)
        resp = orchestrator.process(req)

        # Sob espera de humano, nenhuma proposta de ação mutável é aprovada
        assert len(resp.action_proposals) == 0
