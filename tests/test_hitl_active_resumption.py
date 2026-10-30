"""
Contract tests for Active HITL Resumption (Alavanca B).
Verifies:
1. Active agent resumption executes real diagnostics and synthesizes a complete resolution.
2. The assistant response is persisted in SQLite session memory so the merchant receives it
   immediately without needing to send 'Obrigado' or any follow-up message.
3. Repeated replies / resumption calls are idempotent and do not duplicate execution.
4. Write actions (e.g. remote terminal reboot) remain strictly two-phase proposals requiring
   explicit merchant token confirmation, never dispatched autonomously.
5. Concurrent open tickets keep the session in 'waiting_human' state until all are resolved.
6. HTTP API integration via POST /api/hitl/tickets/{ticket_id}/reply (with resume_agent=True)
   and POST /api/hitl/tickets/{ticket_id}/resume.
"""

from tests.legacy_runtime_adapter import bind_legacy_runtime
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.hitl.queue import HITLQueueManager
from app.memory.store import StateStore
import app.memory.store as store_module
from app.agents.orchestrator import MultiAgentOrchestrator
import app.agents.orchestrator as orchestrator_module
from app.security.auth import AuthManager
import app.main as main_module


@pytest.fixture
def hitl_test_env(tmp_path, monkeypatch):
    db_file = tmp_path / "hitl_resumption_test.db"
    store = StateStore(db_file)
    queue = HITLQueueManager(store=store)
    orch = MultiAgentOrchestrator()

    config = Settings(_env_file=None, APP_ENV="development", JWT_SECRET="test-jwt-secret-with-sufficient-entropy-1234567890", AUTH_API_KEY="test-api-key")

    monkeypatch.setattr(store_module, "state_store", store)
    monkeypatch.setattr(orchestrator_module, "state_store", store)
    monkeypatch.setattr(main_module, "state_store", store)
    monkeypatch.setattr(main_module, "hitl_queue", queue)
    monkeypatch.setattr(main_module, "orchestrator", orch)
    bind_legacy_runtime(monkeypatch, main_module, orch, store=store, queue=queue)
    monkeypatch.setattr("app.security.action_policy.state_store", store)
    monkeypatch.setattr("app.tools.customer_tools.state_store", store)
    monkeypatch.setattr("app.tools.customer_tools.hitl_queue", queue)

    client = TestClient(main_module.app)
    return store, queue, orch, client, config


def test_active_resumption_executes_diagnostic_and_persists_assistant_turn(hitl_test_env):
    """
    Aceite por cenário real: operador fornece orientação, agente executa a consulta/diagnóstico
    necessário, trace comprova a ferramenta e o resultado, resposta é persistida e o cliente
    consegue recebê-la sem precisar mandar 'obrigado' para disparar o processo.
    """
    store, queue, orch, client, config = hitl_test_env
    user_id = "cliente1988"
    session_id = "sess_active_resumption_pos_01"
    store.ensure_session(session_id, user_id)

    # 1. Simula mensagem inicial do lojista com problema de conectividade
    store.append_message(
        session_id=session_id,
        user_id=user_id,
        role="user",
        content="Minha maquininha GET-SMART-88219 não está conectando na rede.",
    )

    # 2. Criação do chamado HITL
    ticket = queue.create_ticket(
        session_id=session_id,
        user_id=user_id,
        priority="high",
        category="terminal_hardware",
        reason="Falha de conexão no leitor POS",
        diagnostic_snapshot={"terminal_id": "GET-SMART-88219", "status": "Offline"},
    )
    assert ticket.status == "OPEN"
    ctx_before = store.get_conversation_context(session_id, user_id)
    assert ctx_before["status"] == "waiting_human"

    # 3. Operador responde ativando a retomada imediata (resume_agent=True)
    reply_text = "Verificamos liberação de rede no switch central. Favor validar conectividade e status operacional do terminal."
    updated = queue.reply_to_ticket(
        ticket_id=ticket.ticket_id,
        operator_id="op_analyst_telecom",
        reply_message=reply_text,
        resolve_ticket=True,
        resume_agent=True,
        orchestrator=orch,
    )

    assert updated is not None
    assert updated.status == "RESOLVED"

    # 4. Prova de que a resposta foi persistida para o lojista sem necessidade de 'obrigado'
    ctx_after = store.get_conversation_context(session_id, user_id)
    assert ctx_after["status"] == "active"
    recent_messages = ctx_after["recent_messages"]

    # Mensagens esperadas: user -> human_operator -> assistant
    assert len(recent_messages) >= 3
    operator_msg = [m for m in recent_messages if m["role"] == "human_operator"][-1]
    assert reply_text in operator_msg["content"]
    assert "op_analyst_telecom" in operator_msg["agent_used"]

    assistant_msg = [m for m in recent_messages if m["role"] == "assistant"][-1]
    assert "MULTI_AGENT_SUPERVISOR:HITL_RESUMPTION" in assistant_msg["agent_used"]
    assert reply_text in assistant_msg["content"]
    assert "GET-SMART-88219" in assistant_msg["content"]
    assert "diagnose_terminal_issue" in assistant_msg["tools_called"]

    # 5. Prova de que o ticket contém o turno retomado
    assert updated.resumed_turn is not None
    assert updated.resumed_turn.get("ticket_id") == ticket.ticket_id
    assert "diagnose_terminal_issue" in updated.resumed_turn.get("tools_called", [])


def test_active_resumption_idempotency_prevents_duplicate_execution(hitl_test_env):
    """
    Reply repetido ou retomadas sucessivas não duplicam a execução de ferramentas
    nem criam mensagens adicionais na sessão.
    """
    store, queue, orch, client, config = hitl_test_env
    user_id = "cliente1988"
    session_id = "sess_idempotent_hitl_01"
    store.ensure_session(session_id, user_id)

    ticket = queue.create_ticket(
        session_id=session_id,
        user_id=user_id,
        priority="medium",
        category="sales_settlement",
        reason="Dúvida sobre lote de liquidação",
        diagnostic_snapshot={"action": "sales_settlement"},
    )

    # Primeira execução
    res1 = orch.resume_session_from_hitl(
        session_id=session_id,
        user_id=user_id,
        ticket_id=ticket.ticket_id,
        operator_reply="Lote liberado pela tesouraria nesta manhã.",
        operator_id="op_finance_01",
        store=store,
    )
    assert res1 is not None
    assert "get_sales_settlement" in res1.tools_called

    ctx1 = store.get_conversation_context(session_id, user_id)
    msg_count_1 = len(ctx1["recent_messages"])

    # Segunda execução (repetida com a mesma chave de ticket)
    res2 = orch.resume_session_from_hitl(
        session_id=session_id,
        user_id=user_id,
        ticket_id=ticket.ticket_id,
        operator_reply="Lote liberado pela tesouraria nesta manhã.",
        operator_id="op_finance_01",
        store=store,
    )
    assert res2 is not None
    assert res2.response == res1.response
    assert "Cached" in res2.intent

    ctx2 = store.get_conversation_context(session_id, user_id)
    msg_count_2 = len(ctx2["recent_messages"])
    # Nenhuma mensagem duplicada foi gravada no banco
    assert msg_count_1 == msg_count_2


def test_active_resumption_write_action_generates_unconfirmed_proposal(hitl_test_env):
    """
    Propostas de escrita explicitamente solicitadas pelo operador humano (ex: reinício remoto do terminal)
    permanecem sob o modelo de proposta em duas fases (PENDING_CONFIRMATION) com hash
    criptográfico e exigem confirmação explícita do titular. Nunca são despachadas diretamente.
    """
    store, queue, orch, client, config = hitl_test_env
    user_id = "cliente1988"
    session_id = "sess_remote_action_hitl"
    store.ensure_session(session_id, user_id)

    ticket = queue.create_ticket(
        session_id=session_id,
        user_id=user_id,
        priority="critical",
        category="terminal_hardware",
        reason="Travamento de tela no POS",
        diagnostic_snapshot={"terminal_id": "GET-SMART-88219"},
    )

    # Operador responde resolvendo o chamado e instruindo reinício remoto com retomada ativa
    updated = queue.reply_to_ticket(
        ticket_id=ticket.ticket_id,
        operator_id="op_field_engineer",
        reply_message="Crie uma proposta de reinício do terminal GET-SMART-88219.",
        resolve_ticket=True,
        resume_agent=True,
        orchestrator=orch,
    )

    assert updated is not None
    assert updated.resumed_turn is not None
    res = updated.resumed_turn
    assert "nenhum comando foi aplicado" in res["response"]
    assert len(res["action_proposals"]) == 1
    proposal = res["action_proposals"][0]
    assert proposal["status"] == "PENDING_CONFIRMATION"
    assert proposal["action_type"] == "remote_reboot"
    assert proposal["terminal_id"] == "GET-SMART-88219"
    assert "payload_hash" in proposal
    assert len(proposal["payload_hash"]) == 64

    # A resposta ao usuário deixa explícito que a ação aguarda confirmação segura
    assert "Proposta ID" in res["response"]
    assert proposal["proposal_id"] in res["response"]


def test_active_resumption_keeps_session_paused_if_concurrent_ticket_is_open(hitl_test_env):
    """
    Se houver múltiplos chamados para a mesma sessão e apenas um for resolvido,
    o estado da sessão permanece 'waiting_human'.
    """
    store, queue, orch, client, config = hitl_test_env
    user_id = "cliente1988"
    session_id = "sess_concurrent_tickets"
    store.ensure_session(session_id, user_id)

    t1 = queue.create_ticket(session_id=session_id, user_id=user_id, priority="high", category="terminal", reason="Falha leitor")
    t2 = queue.create_ticket(session_id=session_id, user_id=user_id, priority="critical", category="pix", reason="Falha chave Pix")

    # Resolve t1 com retomada ativa; t2 continua OPEN
    updated_t1 = queue.reply_to_ticket(
        ticket_id=t1.ticket_id,
        operator_id="op_01",
        reply_message="Orientação de hardware prestada.",
        resolve_ticket=True,
        resume_agent=True,
        orchestrator=orch,
    )
    assert updated_t1.status == "RESOLVED"

    # A sessão continua waiting_human porque t2 ainda não foi resolvido
    ctx = store.get_conversation_context(session_id, user_id)
    assert ctx["status"] == "waiting_human"


def test_api_hitl_reply_with_resume_agent_flag(hitl_test_env):
    """
    Testa integração HTTP da rota POST /api/hitl/tickets/{ticket_id}/reply com resume_agent=True.
    """
    store, queue, orch, client, config = hitl_test_env
    user_id = "cliente1988"
    session_id = "sess_api_reply_resume"
    store.ensure_session(session_id, user_id)

    ticket = queue.create_ticket(
        session_id=session_id,
        user_id=user_id,
        priority="high",
        category="terminal",
        reason="Falha Wi-Fi",
        diagnostic_snapshot={"terminal_id": "GET-CLASSICA-10442"},
    )

    token = AuthManager.create_jwt_token("op_alice_admin", role="operator")
    headers = {"Authorization": f"Bearer {token}"}

    resp = client.post(
        f"/api/hitl/tickets/{ticket.ticket_id}/reply",
        headers=headers,
        json={
            "operator_id": "op_alice_admin",
            "reply_message": "Reset de roteador homologado. Validar POS.",
            "resolve_ticket": True,
            "resume_agent": True,
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "RESOLVED"
    assert data["assigned_operator"] == "op_alice_admin"
    assert data["resumed_turn"] is not None

    # Verifica persistência no SQLite
    ctx = store.get_conversation_context(session_id, user_id)
    assert ctx["status"] == "active"
    assert any("MULTI_AGENT_SUPERVISOR:HITL_RESUMPTION" in m["agent_used"] for m in ctx["recent_messages"])


def test_api_hitl_resume_dedicated_endpoint(hitl_test_env):
    """
    Testa rota dedicada POST /api/hitl/tickets/{ticket_id}/resume.
    """
    store, queue, orch, client, config = hitl_test_env
    user_id = "cliente1988"
    session_id = "sess_api_resume_endpoint"
    store.ensure_session(session_id, user_id)

    ticket = queue.create_ticket(
        session_id=session_id,
        user_id=user_id,
        priority="medium",
        category="sales_settlement",
        reason="Consulta saldo",
    )

    # 1. Tentar retomar ticket OPEN deve falhar com HTTP 400
    token = AuthManager.create_jwt_token("op_alice_admin", role="operator")
    headers = {"Authorization": f"Bearer {token}"}

    bad_resume = client.post(f"/api/hitl/tickets/{ticket.ticket_id}/resume", headers=headers)
    assert bad_resume.status_code == 400

    # 2. Operador resolve sem retomar
    queue.reply_to_ticket(
        ticket_id=ticket.ticket_id,
        operator_id="op_alice_admin",
        reply_message="Consulta autorizada.",
        resolve_ticket=True,
        resume_agent=False,
    )

    # 3. Chama endpoint dedicado de retomada
    good_resume = client.post(f"/api/hitl/tickets/{ticket.ticket_id}/resume", headers=headers)
    assert good_resume.status_code == 200
    res_data = good_resume.json()
    assert res_data["agent_used"] == "MULTI_AGENT_SUPERVISOR:HITL_RESUMPTION"
    # Sem data explícita, o snapshot histórico não resolve a consulta de saldo.
    assert res_data["answer_status"] == "partial"
    assert res_data["abstained"] is True
    assert "histórico" in res_data["response"].lower()
    assert "get_sales_settlement" in res_data["tools_called"]


def test_active_resumption_financial_settlement_uses_real_sales_data(hitl_test_env):
    """
    Regressão: retomada HITL de liquidação deve usar os campos reais de sales_data
    (net_payout_scheduled_today_brl, settlement_batch_id, payout_status),
    preservar o aviso temporal e a referência histórica, sem inventar R$ 0,00 ou N/A.
    """
    store, queue, orch, client, config = hitl_test_env
    user_id = "cliente1988"
    session_id = "sess_financial_resumption_01"
    store.ensure_session(session_id, user_id)

    ticket = queue.create_ticket(
        session_id=session_id,
        user_id=user_id,
        priority="medium",
        category="sales_settlement",
        reason="Consulta de vendas e liquidação",
        diagnostic_snapshot={"action": "sales_settlement"},
    )

    res = orch.resume_session_from_hitl(
        session_id=session_id,
        user_id=user_id,
        ticket_id=ticket.ticket_id,
        operator_reply="Consulte o valor líquido e o status do lote de vendas registrado.",
        operator_id="op_finance_01",
        store=store,
    )
    assert res is not None
    assert "get_sales_settlement" in res.tools_called
    # Real fixture values must be used, not fabricated defaults
    assert "R$ 0.00" not in res.response
    assert "R$ 0,00" not in res.response
    assert "`N/A`" not in res.response
    # Fixture net payout is 1415.80 and batch is LOTE-STL-20261022-881
    assert ("1,415.80" in res.response or "1.415,80" in res.response)
    assert "LOTE-STL-20261022-881" in res.response
    # Temporal notice and historical reference preserved
    assert ("histórico" in res.response.lower() or "snapshot" in res.response.lower() or "não confirma" in res.response.lower())
    assert "Programado para liquidação" in res.response or "D+1" in res.response


def test_active_resumption_unknown_terminal_informs_absence_without_telemetry(hitl_test_env):
    """
    Regressão: se o terminal não for encontrado (found=False), deve informar ausência
    e nunca inventar telemetria (Online, 4G, firmware v2.4.1 ou modelo Get Smart).
    """
    store, queue, orch, client, config = hitl_test_env
    user_id = "cliente1988"
    session_id = "sess_unknown_terminal_01"
    store.ensure_session(session_id, user_id)

    ticket = queue.create_ticket(
        session_id=session_id,
        user_id=user_id,
        priority="high",
        category="terminal_hardware",
        reason="Verifique firmware do terminal informado",
        diagnostic_snapshot={"terminal_id": "GET-SMART-777777"},
    )

    res = orch.resume_session_from_hitl(
        session_id=session_id,
        user_id=user_id,
        ticket_id=ticket.ticket_id,
        operator_reply="Valide o firmware e a conectividade do terminal informado.",
        operator_id="op_hardware_01",
        store=store,
    )
    assert res is not None
    assert "diagnose_terminal_issue" in res.tools_called
    # Informs absence
    assert ("não localizado" in res.response.lower() or "não localizamos" in res.response.lower() or "não encontrado" in res.response.lower())
    # Telemetry defaults must NOT be invented for missing terminal
    assert "v2.4.1" not in res.response
    assert "Status de Comunicação: `Online`" not in res.response
    assert "(4G)" not in res.response


def test_active_resumption_battery_zero_percent_preserved_and_absent_not_zero(hitl_test_env):
    """
    Regressão: bateria 0% legítima não pode ser descartada por 'battery_percent or battery',
    e campo de bateria ausente (None) não pode ser transformado em 0%.
    """
    store, queue, orch, client, config = hitl_test_env
    user_id = "cliente1988"
    session_id = "sess_battery_test_01"
    store.ensure_session(session_id, user_id)

    # Subcase A: battery_percent is legitimately 0
    with patch("app.tools.registry.tool_registry.execute") as mock_exec:
        from app.schemas import ToolExecutionResult
        mock_exec.return_value = ToolExecutionResult(
            tool_name="diagnose_terminal_issue",
            success=True,
            data={
                "found": True,
                "terminal": {
                    "terminal_id": "GET-SMART-88219",
                    "model": "Get Smart (Smart POS Android)",
                    "status": "Online",
                    "connection_type": "4G",
                    "firmware_version": "v4.18.2",
                    "battery_percent": 0,
                },
            },
        )
        t_zero = queue.create_ticket(
            session_id=session_id,
            user_id=user_id,
            priority="medium",
            category="terminal_hardware",
            reason="Checar bateria zero",
            diagnostic_snapshot={"terminal_id": "GET-SMART-88219"},
        )
        res_zero = orch.resume_session_from_hitl(
            session_id=session_id,
            user_id=user_id,
            ticket_id=t_zero.ticket_id,
            operator_reply="Verifique a bateria do terminal.",
            operator_id="op_tech_01",
            store=store,
        )
        # 0% must be preserved and not discarded
        assert "0%" in res_zero.response
        assert ("Bateria: 0%" in res_zero.response or "**Bateria:** 0%" in res_zero.response)

    # Subcase B: battery field is absent (None) -> must NOT turn into 0%
    session_id_b = "sess_battery_test_02"
    store.ensure_session(session_id_b, user_id)
    with patch("app.tools.registry.tool_registry.execute") as mock_exec:
        mock_exec.return_value = ToolExecutionResult(
            tool_name="diagnose_terminal_issue",
            success=True,
            data={
                "found": True,
                "terminal": {
                    "terminal_id": "GET-CLASSICA-10442",
                    "model": "Get Clássica (POS Balcão)",
                    "status": "Online",
                    "connection_type": "Wi-Fi",
                    "firmware_version": "v3.09.1",
                    "battery_percent": None,
                    "battery": None,
                },
            },
        )
        t_none = queue.create_ticket(
            session_id=session_id_b,
            user_id=user_id,
            priority="medium",
            category="terminal_hardware",
            reason="Checar terminal sem bateria",
            diagnostic_snapshot={"terminal_id": "GET-CLASSICA-10442"},
        )
        res_none = orch.resume_session_from_hitl(
            session_id=session_id_b,
            user_id=user_id,
            ticket_id=t_none.ticket_id,
            operator_reply="Verifique status do terminal.",
            operator_id="op_tech_02",
            store=store,
        )
        # Absent field must not become 0%
        assert "0%" not in res_none.response
        assert "Bateria" not in res_none.response



@pytest.mark.parametrize('code', ['TOOL_ERROR', 'ACTION_NOT_AUTHORIZED'])
def test_resumption_failure_never_claims_success(hitl_test_env, code):
    from app.schemas import ToolExecutionResult
    store, queue, orch, _, _ = hitl_test_env
    ticket = queue.create_ticket(session_id='failed', user_id='cliente1988', priority='high',
        category='terminal', reason='Consulte o status do terminal GET-SMART-88219',
        diagnostic_snapshot={'terminal_id': 'GET-SMART-88219'})
    with patch('app.tools.registry.tool_registry.execute', return_value=ToolExecutionResult(
            tool_name='diagnose_terminal_issue', success=False, error_code=code, error_message='Consulta recusada.')):
        with patch.object(orchestrator_module.metrics_collector, 'record_request') as metrics:
            result = orch.resume_session_from_hitl('failed', 'cliente1988', ticket.ticket_id,
                'Valide o status do terminal.', 'op', store=store)
    assert 'aplicada com sucesso' not in result.response
    assert result.answer_status != 'complete'
    assert result.confidence < 1
    assert result.abstained
    assert code in result.response
    assert metrics.call_args.kwargs['answer_status'] == result.answer_status


@pytest.mark.parametrize('guidance', ['Não reiniciar e não crie proposta', 'Prossiga assim',
    'Procedimento indicado: reiniciar o terminal GET-SMART-88219 para recarregar o kernel.'])
def test_resumption_guidance_does_not_invent_consent(hitl_test_env, guidance):
    store, queue, orch, _, _ = hitl_test_env
    ticket = queue.create_ticket(session_id='no_consent', user_id='cliente1988', priority='high',
        category='terminal', reason='Consulte o status do terminal GET-SMART-88219',
        diagnostic_snapshot={'terminal_id': 'GET-SMART-88219'})
    result = orch.resume_session_from_hitl('no_consent', 'cliente1988', ticket.ticket_id,
        guidance, 'op', store=store)
    assert result.action_proposals == []
    assert 'execute_terminal_remote_action' not in result.tools_called


def test_resume_endpoint_waits_for_other_ticket_before_any_tool(hitl_test_env):
    store, queue, orch, client, _ = hitl_test_env
    first = queue.create_ticket(session_id='blocked', user_id='cliente1988', priority='high',
        category='sales_settlement', reason='Consulte o saldo')
    queue.create_ticket(session_id='blocked', user_id='cliente1988', priority='high',
        category='terminal', reason='Falha de conexão')
    queue.reply_to_ticket(first.ticket_id, 'op', 'Consulte o saldo', resolve_ticket=True)
    token = AuthManager.create_jwt_token('op', role='operator')
    from app.schemas import ToolExecutionResult
    with patch('app.tools.registry.tool_registry.execute', return_value=ToolExecutionResult(
            tool_name='get_sales_settlement', success=False, error_code='TEST_REFUSAL')) as execute:
        response = client.post(f'/api/hitl/tickets/{first.ticket_id}/resume',
            headers={'Authorization': f'Bearer {token}'})
    assert execute.call_count == 0
    assert store.get_conversation_context('blocked', 'cliente1988')['status'] == 'waiting_human'
    assert response.status_code == 200
    assert response.json()['answer_status'] != 'complete'
    assert 'aguardando' in response.json()['response'].lower()
    assert 'liberada' not in response.json()['response']


def test_resumption_mixed_original_request_remains_partial(hitl_test_env):
    store, queue, orch, _, _ = hitl_test_env
    original = 'Consulte o status do terminal GET-SMART-88219; explique as regras de crediário na Noruega.'
    store.ensure_session('mixed', 'cliente1988')
    store.append_message(session_id='mixed', user_id='cliente1988', role='user', content=original)
    ticket = queue.create_ticket(session_id='mixed', user_id='cliente1988', priority='high',
        category='terminal', reason='Verificar terminal', diagnostic_snapshot={'terminal_id': 'GET-SMART-88219'})
    result = orch.resume_session_from_hitl('mixed', 'cliente1988', ticket.ticket_id,
        'Consulte o status do terminal.', 'op', store=store)
    assert result.answer_status == 'partial'
    assert result.abstained
    assert 'Noruega' in result.response
    assert 'aplicada com sucesso' not in result.response


def test_resumption_deterministic_even_with_provider_key(hitl_test_env, monkeypatch):
    store, queue, orch, _, _ = hitl_test_env
    ticket = queue.create_ticket(session_id='mode', user_id='cliente1988', priority='high',
        category='sales_settlement', reason='Consulte o saldo')
    monkeypatch.setattr(orchestrator_module.settings, 'GEMINI_API_KEY', 'not-a-real-key')
    result = orch.resume_session_from_hitl('mode', 'cliente1988', ticket.ticket_id,
        'Consulte o saldo', 'op', store=store)
    assert result.semantic_mode == 'limited_offline'


def test_reply_trigger_with_pending_ticket_executes_nothing(hitl_test_env):
    store, queue, orch, _, _ = hitl_test_env
    first = queue.create_ticket(session_id='reply_blocked', user_id='cliente1988', priority='high',
        category='terminal', reason='Consulte o status do terminal GET-SMART-88219')
    queue.create_ticket(session_id='reply_blocked', user_id='cliente1988', priority='high',
        category='sales_settlement', reason='Consulte o saldo')
    with patch('app.tools.registry.tool_registry.execute') as execute:
        result = queue.reply_to_ticket(first.ticket_id, 'op', 'Reinicie o terminal GET-SMART-88219.',
            resume_agent=True, orchestrator=orch)
    execute.assert_not_called()
    assert result.resumed_turn is None
    assert store.get_conversation_context('reply_blocked', 'cliente1988')['status'] == 'waiting_human'


def test_operator_guidance_cannot_expand_original_read_purpose_or_target(hitl_test_env):
    store, queue, orch, _, _ = hitl_test_env
    store.ensure_session('purpose', 'cliente1988')
    store.append_message(session_id='purpose', user_id='cliente1988', role='user',
        content='Consulte o status do terminal GET-SMART-88219.')
    ticket = queue.create_ticket(session_id='purpose', user_id='cliente1988', priority='high',
        category='terminal', reason='Verificar status')
    result = orch.resume_session_from_hitl('purpose', 'cliente1988', ticket.ticket_id,
        'Consulte minhas vendas; reinicie o terminal GET-CLASSICA-10442.', 'op', store=store)
    assert result.tools_called == ['diagnose_terminal_issue']
    assert result.action_proposals == []
    assert 'Volume Bruto' not in result.response
    assert result.answer_status == 'complete'


def test_explicit_proposal_preserves_security_context_and_holder_confirmation(hitl_test_env):
    from app.security.auth import SecurityContext
    store, queue, orch, _, _ = hitl_test_env
    store.ensure_session('scoped', 'cliente1988')
    store.append_message(session_id='scoped', user_id='cliente1988', role='user',
        content='Consulte o status do terminal GET-SMART-88219.')
    ticket = queue.create_ticket(session_id='scoped', user_id='cliente1988', priority='high',
        category='terminal', reason='Verificar status')
    queue.reply_to_ticket(ticket.ticket_id, 'op',
        'Crie uma proposta de reinício do terminal GET-SMART-88219.', resolve_ticket=True)
    context = SecurityContext(authenticated_user_id='cliente1988', role='merchant',
        scopes=['customer:read_own'])
    from app.tools.registry import tool_registry
    with patch.object(tool_registry, 'execute', wraps=tool_registry.execute) as execute:
        result = orch.resume_session_from_hitl('scoped', 'cliente1988', ticket.ticket_id,
            'Crie uma proposta de reinício do terminal GET-SMART-88219.', 'op',
            security_context=context, store=store)
    assert all(c.kwargs['security_context'] is context for c in execute.call_args_list)
    assert result.action_proposals == []
    assert result.answer_status != 'complete'
    assert result.abstained
    assert 'aplicada com sucesso' not in result.response


def test_original_veto_is_not_overridden_by_operator_proposal(hitl_test_env):
    store, queue, orch, _, _ = hitl_test_env
    store.ensure_session('veto', 'cliente1988')
    store.append_message(session_id='veto', user_id='cliente1988', role='user',
        content='Não reinicie o terminal GET-SMART-88219. Consulte o status do terminal GET-SMART-88219.')
    ticket = queue.create_ticket(session_id='veto', user_id='cliente1988', priority='high',
        category='terminal', reason='Verificar status')
    result = orch.resume_session_from_hitl('veto', 'cliente1988', ticket.ticket_id,
        'Crie uma proposta de reinício do terminal GET-SMART-88219.', 'op', store=store)
    assert result.action_proposals == []
    assert 'execute_terminal_remote_action' not in result.tools_called



def test_verified_proposal_is_partial_until_holder_confirmation(hitl_test_env):
    store, queue, orch, _, _ = hitl_test_env
    ticket = queue.create_ticket(session_id='proposal_pending', user_id='cliente1988', priority='high',
        category='terminal', reason='Verificar status',
        diagnostic_snapshot={'terminal_id': 'GET-SMART-88219'})
    queue.reply_to_ticket(ticket.ticket_id, 'op',
        'Crie uma proposta de reinício do terminal GET-SMART-88219.', resolve_ticket=True)
    result = orch.resume_session_from_hitl('proposal_pending', 'cliente1988', ticket.ticket_id,
        'Crie uma proposta de reinício do terminal GET-SMART-88219.', 'op', store=store)
    assert len(result.action_proposals) == 1
    assert result.action_proposals[0]['status'] == 'PENDING_CONFIRMATION'
    assert result.answer_status == 'partial'
    assert result.abstained
    assert 'nenhum comando foi aplicado' in result.response


def _stage3_ticket(queue, session_id):
    ticket = queue.create_ticket(session_id=session_id, user_id='cliente1988', priority='high',
        category='terminal', reason='Verificar status',
        diagnostic_snapshot={'terminal_id': 'GET-SMART-88219'})
    queue.reply_to_ticket(ticket.ticket_id, 'op', 'Consulte o status do terminal.', resolve_ticket=True)
    return ticket


def _stage3_diagnostic():
    from app.schemas import ToolExecutionResult
    return ToolExecutionResult(tool_name='diagnose_terminal_issue', success=True,
        data={'found': True, 'user_id': 'cliente1988', 'terminal': {
            'terminal_id': 'GET-SMART-88219', 'status': 'Online',
            'model': 'Get Smart', 'connection_type': '4G', 'signal_strength': 'Bom',
            'firmware_version': 'v4.18.2'}})


def test_durable_resumption_concurrent_stores_execute_once(hitl_test_env):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    store, queue, orch, _, _ = hitl_test_env
    ticket = _stage3_ticket(queue, 'concurrent_sqlite')
    second_store = StateStore(store.db_path)
    second_orch = MultiAgentOrchestrator()
    entered, release = Event(), Event()

    def slow_tool(**kwargs):
        entered.set()
        assert release.wait(timeout=5), 'finite tool barrier timed out'
        return _stage3_diagnostic()

    def resume(worker, database):
        return worker.resume_session_from_hitl(ticket.session_id, ticket.user_id,
            ticket.ticket_id, 'Consulte o status do terminal.', 'op', store=database)

    with patch('app.tools.registry.tool_registry.execute', side_effect=slow_tool) as execute:
        with ThreadPoolExecutor(max_workers=2) as workers:
            first = workers.submit(resume, orch, store)
            assert entered.wait(timeout=3)
            second = workers.submit(resume, second_orch, second_store)
            try:
                concurrent = second.result(timeout=3)
            finally:
                release.set()
            finished = first.result(timeout=3)
    assert execute.call_count == 1
    assert concurrent.answer_status != 'complete'
    assert concurrent.tools_called == []
    assert concurrent.abstained
    assert finished.tools_called == ['diagnose_terminal_issue']
    messages = second_store.get_conversation_context(ticket.session_id, ticket.user_id)['recent_messages']
    assert len([m for m in messages if m['agent_used'] == 'MULTI_AGENT_SUPERVISOR:HITL_RESUMPTION']) == 1


def test_durable_replay_after_restart_preserves_proposal_and_metadata(hitl_test_env):
    store, queue, orch, _, _ = hitl_test_env
    ticket = _stage3_ticket(queue, 'restart_proposal')
    guidance = 'Crie uma proposta de reinício do terminal GET-SMART-88219.'
    first = orch.resume_session_from_hitl(ticket.session_id, ticket.user_id, ticket.ticket_id,
        guidance, 'op', store=store)
    assert len(first.action_proposals) == 1
    restarted = StateStore(store.db_path)
    with patch('app.tools.registry.tool_registry.execute') as execute:
        replay = MultiAgentOrchestrator().resume_session_from_hitl(ticket.session_id, ticket.user_id,
            ticket.ticket_id, guidance, 'op', store=restarted)
    execute.assert_not_called()
    for field in ('response', 'action_proposals', 'sources', 'trace', 'trace_id', 'answer_status',
                  'abstained', 'confidence', 'ticket_id', 'tools_called', 'token_usage',
                  'evidence', 'semantic_mode', 'execution_time_ms'):
        assert getattr(replay, field) == getattr(first, field), field
    assert replay.answer_status == 'partial'
    proposal_id = replay.action_proposals[0]['proposal_id']
    assert f'/api/actions/{proposal_id}/confirm' in replay.response
    assert '/api/action-proposals/' not in replay.response


def test_finalization_failure_rolls_back_message_and_prevents_retry_after_restart(hitl_test_env):
    store, queue, orch, _, _ = hitl_test_env
    ticket = _stage3_ticket(queue, 'atomic_failure')
    append = store.append_message

    def fail_after_insert(*args, **kwargs):
        append(*args, **kwargs)
        raise RuntimeError('interruption before result persistence')

    with patch('app.tools.registry.tool_registry.execute', return_value=_stage3_diagnostic()) as execute:
        with patch.object(store, 'append_message', side_effect=fail_after_insert):
            with pytest.raises(RuntimeError, match='interruption'):
                orch.resume_session_from_hitl(ticket.session_id, ticket.user_id, ticket.ticket_id,
                    'Consulte o status do terminal.', 'op', store=store)
        restarted = StateStore(store.db_path)
        context = restarted.get_conversation_context(ticket.session_id, ticket.user_id)
        assert not [m for m in context['recent_messages'] if m['role'] == 'assistant']
        assert f'hitl_resumed_{ticket.ticket_id}' not in context['session_facts']
        retry = MultiAgentOrchestrator().resume_session_from_hitl(ticket.session_id, ticket.user_id,
            ticket.ticket_id, 'Consulte o status do terminal.', 'op', store=restarted)
    assert execute.call_count == 1
    assert retry.answer_status != 'complete'
    assert retry.abstained
    with restarted._connect() as conn:
        assert conn.execute('SELECT status FROM hitl_resumptions WHERE ticket_id = ?',
            (ticket.ticket_id,)).fetchone()['status'] == 'unknown'


@pytest.mark.parametrize('outcome', ['unknown', 'failed'])
def test_uncertain_tool_result_is_replayed_without_retry(hitl_test_env, outcome):
    from app.schemas import ToolExecutionResult
    store, queue, orch, _, _ = hitl_test_env
    ticket = _stage3_ticket(queue, 'unknown_outcome')
    unknown = ToolExecutionResult(tool_name='diagnose_terminal_issue', success=False,
        error_code='OUTCOME_UNKNOWN' if outcome == 'unknown' else 'TOOL_FAILED',
        error_message='Sem resultado verificado.', outcome_state=outcome)
    with patch('app.tools.registry.tool_registry.execute', return_value=unknown) as execute:
        first = orch.resume_session_from_hitl(ticket.session_id, ticket.user_id, ticket.ticket_id,
            'Consulte o status do terminal.', 'op', store=store)
        replay = MultiAgentOrchestrator().resume_session_from_hitl(ticket.session_id, ticket.user_id,
            ticket.ticket_id, 'Consulte o status do terminal.', 'op', store=StateStore(store.db_path))
    assert execute.call_count == 1
    assert replay.response == first.response
    assert replay.answer_status == first.answer_status != 'complete'
    assert replay.abstained
    with store._connect() as conn:
        assert conn.execute('SELECT status FROM hitl_resumptions WHERE ticket_id = ?',
            (ticket.ticket_id,)).fetchone()['status'] == outcome


@pytest.mark.parametrize('wrong_session,wrong_user', [('other_session', 'cliente1988'),
                                                       ('foreign_session', 'cliente2045')])
def test_resumption_validates_ticket_binding_before_tools(hitl_test_env, wrong_session, wrong_user):
    from app.security.auth import CrossTenantAccessError
    store, queue, orch, _, _ = hitl_test_env
    ticket = _stage3_ticket(queue, 'bound_ticket')
    store.ensure_session(wrong_session, wrong_user)
    with patch('app.tools.registry.tool_registry.execute', return_value=_stage3_diagnostic()) as execute:
        with pytest.raises((CrossTenantAccessError, ValueError)):
            orch.resume_session_from_hitl(wrong_session, wrong_user, ticket.ticket_id,
                'Consulte o status do terminal.', 'op', store=store)
    execute.assert_not_called()


def test_failure_before_tools_is_durable_and_not_published(hitl_test_env):
    store, queue, orch, _, _ = hitl_test_env
    ticket = _stage3_ticket(queue, 'before_tools_failure')
    with patch.object(store, 'get_conversation_context', side_effect=RuntimeError('memory unavailable')):
        with pytest.raises(RuntimeError, match='memory unavailable'):
            orch.resume_session_from_hitl(ticket.session_id, ticket.user_id, ticket.ticket_id,
                'Consulte o status do terminal.', 'op', store=store)
    restarted = StateStore(store.db_path)
    with patch('app.tools.registry.tool_registry.execute', return_value=_stage3_diagnostic()) as execute:
        response = MultiAgentOrchestrator().resume_session_from_hitl(ticket.session_id, ticket.user_id,
            ticket.ticket_id, 'Consulte o status do terminal.', 'op', store=restarted)
    execute.assert_not_called()
    assert response.answer_status != 'complete'
    assert response.abstained
    assert not [m for m in restarted.get_conversation_context(ticket.session_id, ticket.user_id)['recent_messages']
                if m['role'] == 'assistant']
    with restarted._connect() as conn:
        assert conn.execute('SELECT status FROM hitl_resumptions WHERE ticket_id = ?',
            (ticket.ticket_id,)).fetchone()['status'] == 'failed'



def test_running_reservation_survives_restart_without_automatic_takeover(hitl_test_env):
    store, queue, _, _, _ = hitl_test_env
    ticket = _stage3_ticket(queue, 'abandoned_running')
    claim = store.reserve_hitl_resumption(ticket.session_id, ticket.user_id, ticket.ticket_id)
    assert claim['acquired']
    restarted = StateStore(store.db_path)
    with patch('app.tools.registry.tool_registry.execute') as execute:
        response = MultiAgentOrchestrator().resume_session_from_hitl(ticket.session_id, ticket.user_id,
            ticket.ticket_id, 'Consulte o status do terminal.', 'op', store=restarted)
    execute.assert_not_called()
    assert response.abstained
    assert response.answer_status != 'complete'
    assert 'em andamento' in response.response
    with restarted._connect() as conn:
        row = conn.execute('SELECT status, response_json FROM hitl_resumptions WHERE ticket_id = ?',
            (ticket.ticket_id,)).fetchone()
        assert row['status'] == 'running'
        assert row['response_json'] is None
        assert conn.execute("SELECT count(*) FROM messages WHERE session_id = ? AND role = 'assistant'",
            (ticket.session_id,)).fetchone()[0] == 0


def test_legacy_resumption_is_imported_without_reexecution_or_data_loss(hitl_test_env):
    import json
    store, queue, _, _, _ = hitl_test_env
    ticket = _stage3_ticket(queue, 'legacy_import')
    legacy = {'response': 'Orientação antiga registrada; confirmação pendente.',
        'tools_called': ['execute_terminal_remote_action'], 'ticket_id': ticket.ticket_id,
        'trace_id': 'legacy-trace', 'action_proposals': [{'proposal_id': 'legacy-proposal',
            'status': 'PENDING_CONFIRMATION'}]}
    with store._connect() as conn:
        row = conn.execute('SELECT session_facts_json FROM sessions WHERE session_id = ?',
            (ticket.session_id,)).fetchone()
        facts = json.loads(row[0])
        facts[f'hitl_resumed_{ticket.ticket_id}'] = legacy
        facts['preserved_marker'] = {'value': 42}
        conn.execute('UPDATE sessions SET session_facts_json = ? WHERE session_id = ?',
            (json.dumps(facts), ticket.session_id))
    store.append_message(session_id=ticket.session_id, user_id=ticket.user_id, role='assistant',
        content=legacy['response'], agent_used='MULTI_AGENT_SUPERVISOR:HITL_RESUMPTION')
    restarted = StateStore(store.db_path)
    with patch('app.tools.registry.tool_registry.execute') as execute:
        response = MultiAgentOrchestrator().resume_session_from_hitl(ticket.session_id, ticket.user_id,
            ticket.ticket_id, 'Consulte o status do terminal.', 'op', store=restarted)
    execute.assert_not_called()
    assert response.response == legacy['response']
    assert response.action_proposals == legacy['action_proposals']
    assert response.trace_id == legacy['trace_id']
    assert response.answer_status == 'partial'
    context = restarted.get_conversation_context(ticket.session_id, ticket.user_id)
    assert context['session_facts']['preserved_marker'] == {'value': 42}
    assert context['session_facts'][f'hitl_resumed_{ticket.ticket_id}'] == legacy
    assert len([m for m in context['recent_messages'] if m['role'] == 'assistant']) == 1
