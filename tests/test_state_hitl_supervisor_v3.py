"""Regressions for persisted referents and atomic operator-owned HITL transitions."""
import concurrent.futures
import sqlite3

import pytest

from app.config import settings
from app.hitl.queue import HITLQueueManager
from app.memory.store import StateStore
from app.security.auth import CrossTenantAccessError


@pytest.fixture
def state(tmp_path):
    return StateStore(tmp_path / 'state.db')


def ticket(queue, key='first'):
    return queue.create_ticket('cliente1988', 'session-v3', 'support', 'Precisa de revisão', idempotency_key=key)


def test_resolved_ticket_cannot_be_claimed_or_replied(state):
    queue = HITLQueueManager(state)
    created = ticket(queue)
    queue.reply_to_ticket(created.ticket_id, 'operator-a', 'Resolvido com evidência.')
    before = state.get_conversation_context('session-v3', 'cliente1988')
    for operation in (
        lambda: queue.claim_ticket(created.ticket_id, 'operator-b'),
        lambda: queue.reply_to_ticket(created.ticket_id, 'operator-a', 'Reabertura silenciosa'),
    ):
        with pytest.raises(ValueError):
            operation()
    after = state.get_conversation_context('session-v3', 'cliente1988')
    assert after['recent_messages'] == before['recent_messages']
    assert queue.get_ticket(created.ticket_id).status == 'RESOLVED'


def test_another_operator_cannot_overwrite_claim_or_response(state):
    queue = HITLQueueManager(state)
    created = ticket(queue)
    queue.claim_ticket(created.ticket_id, 'operator-a')
    for operation in (
        lambda: queue.claim_ticket(created.ticket_id, 'operator-b'),
        lambda: queue.reply_to_ticket(created.ticket_id, 'operator-b', 'Resposta indevida'),
    ):
        with pytest.raises(ValueError):
            operation()
    assert queue.get_ticket(created.ticket_id).assigned_operator == 'operator-a'
    assert not state.get_conversation_context('session-v3', 'cliente1988')['recent_messages']


def test_resolving_one_of_two_tickets_keeps_session_paused(state):
    queue = HITLQueueManager(state)
    first, second = ticket(queue), ticket(queue, 'second')
    queue.reply_to_ticket(first.ticket_id, 'operator-a', 'Primeiro resolvido.')
    assert state.get_conversation_context('session-v3', 'cliente1988')['status'] == 'waiting_human'
    assert state.has_pending_human('session-v3', 'cliente1988') is True
    queue.reply_to_ticket(second.ticket_id, 'operator-a', 'Ainda em análise.', resolve_ticket=False)
    assert state.has_pending_human('session-v3', 'cliente1988') is True
    assert state.get_conversation_context('session-v3', 'cliente1988')['status'] == 'waiting_human'
    queue.reply_to_ticket(second.ticket_id, 'operator-a', 'Segundo resolvido.')
    assert state.has_pending_human('session-v3', 'cliente1988') is False
    assert state.get_conversation_context('session-v3', 'cliente1988')['status'] == 'active'
    with pytest.raises(CrossTenantAccessError):
        state.has_pending_human('session-v3', 'cliente2045')


def test_two_store_instances_cannot_race_operator_ownership(state):
    queue = HITLQueueManager(state)
    created = ticket(queue)
    def claim(operator):
        try:
            return HITLQueueManager(StateStore(state.db_path)).claim_ticket(created.ticket_id, operator).assigned_operator
        except ValueError:
            return 'rejected'
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(claim, ['operator-a', 'operator-b']))
    assert results.count('rejected') == 1
    assert queue.get_ticket(created.ticket_id).assigned_operator in {'operator-a', 'operator-b'}


def test_entities_and_topic_survive_window_compaction_and_restart(state, monkeypatch):
    monkeypatch.setattr(settings, 'MEMORY_WINDOW_MESSAGES', 2)
    state.append_message('memory-v3', 'cliente1988', 'user', 'Como funciona antecipação? Meu terminal é GET-SMART-0042, código 91.')
    for message in ('E a taxa dela?', 'E o prazo disso?', 'Obrigado', 'Mais uma dúvida'):
        state.append_message('memory-v3', 'cliente1988', 'user', message)
    ctx = StateStore(state.db_path).get_conversation_context('memory-v3', 'cliente1988', limit=2)
    assert ctx['entities']['terminal_id'] == 'GET-SMART-0042'
    assert ctx['entities']['decline_code'] == '91'
    assert ctx['session_facts']['last_product'] == 'antecipação'
    assert ctx['summary'] and ctx['summary'] == ctx['rolling_summary']
    state.append_message('memory-v3', 'cliente1988', 'user', 'Como funciona o Pix?')
    ctx = state.get_conversation_context('memory-v3', 'cliente1988')
    assert ctx['session_facts']['last_product'] == 'Pix'
    assert ctx['entities']['products'][-1] == 'Pix'


def test_old_database_migrates_memory_without_losing_existing_session(tmp_path):
    db = tmp_path / 'legacy.db'
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE sessions(session_id TEXT PRIMARY KEY,user_id TEXT NOT NULL,status TEXT NOT NULL DEFAULT 'active',rolling_summary TEXT NOT NULL DEFAULT '',active_topic TEXT NOT NULL DEFAULT '',last_agent TEXT NOT NULL DEFAULT '',turn_count INTEGER NOT NULL DEFAULT 0,created_at TEXT NOT NULL,updated_at TEXT NOT NULL)")
        conn.execute("INSERT INTO sessions(session_id,user_id,rolling_summary,created_at,updated_at) VALUES('legacy','cliente1988','Produto Get Smart mencionado','before','before')")
    ctx = StateStore(db).get_conversation_context('legacy', 'cliente1988')
    assert ctx['session_facts'] == {}
    assert ctx['summary'] == 'Produto Get Smart mencionado'

import importlib
from google.genai import types
from app.agents.orchestrator import MultiAgentOrchestrator
from app.agents.router_agent import RouterAgent
from app.agents.function_calling import RouterPlanningError
from app.schemas import ChatRequest, PlanStep, RoutingPlan, SpecialistOutput, ToolExecutionResult


@pytest.fixture
def orchestrator(state, monkeypatch):
    monkeypatch.setattr(importlib.import_module('app.agents.orchestrator'), 'state_store', state)
    monkeypatch.setattr(settings, 'GEMINI_API_KEY', None)
    return MultiAgentOrchestrator()


def plan(*agents):
    return RoutingPlan(primary_agent=agents[0], confidence=.9, reasoning='Consulta dependente',
        plan_steps=[PlanStep(step_order=i, target_agent=a, sub_query='Explique o código identificado', rationale='Consultar evidência')
                    for i,a in enumerate(agents, 1)])


def diagnosis():
    return SpecialistOutput(agent_name='CUSTOMER_SUPPORT_AGENT', response='Código observado: 91.',
        tools_called=['diagnose_terminal_issue'],
        tool_results=[ToolExecutionResult(tool_name='diagnose_terminal_issue', success=True, data={
            'terminal': {'terminal_id':'GET-SMART-0042'}, 'declined_transaction': {'decline_code':'91'}})],
        structured_data={'diagnose_terminal_issue': {'terminal': {'terminal_id':'GET-SMART-0042'}, 'declined_transaction': {'decline_code':'91'}}},
        evidence=[{'source_kind':'mock_operational','claims': {'decline_code':'91'}}])


def test_support_observation_supplies_real_code_to_dependent_knowledge(orchestrator, monkeypatch):
    monkeypatch.setattr(orchestrator.router, 'plan', lambda **kw: plan('CUSTOMER_SUPPORT_AGENT', 'KNOWLEDGE_AGENT'))
    monkeypatch.setattr(orchestrator.customer_support_agent, 'run', lambda **kw: diagnosis())
    def knowledge(**kw):
        observations = kw['conversation_context'].get('specialist_observations') or []
        code = observations[-1]['tool_results'][0]['data']['declined_transaction']['decline_code'] if observations else 'missing'
        return SpecialistOutput(agent_name='KNOWLEDGE_AGENT', response=f'Explicação obtida para código {code}.')
    monkeypatch.setattr(orchestrator.knowledge_agent, 'run', knowledge)
    result = orchestrator.process(ChatRequest(message='Verifique minha maquininha e explique o código identificado', user_id='cliente1988'))
    assert 'Explicação obtida para código 91.' in result.response
    assert result.evidence == [{'source_kind':'mock_operational','claims':{'decline_code':'91'}}]


def test_offline_supervisor_adds_dependent_specialist_from_observed_code(orchestrator, monkeypatch):
    monkeypatch.setattr(orchestrator.router, 'plan', lambda **kw: plan('CUSTOMER_SUPPORT_AGENT'))
    monkeypatch.setattr(orchestrator.customer_support_agent, 'run', lambda **kw: diagnosis())
    def knowledge(**kw):
        return SpecialistOutput(agent_name='KNOWLEDGE_AGENT', response='Consulta documentada: '+kw['message'])
    monkeypatch.setattr(orchestrator.knowledge_agent, 'run', knowledge)
    result = orchestrator.process(ChatRequest(message='Verifique minha maquininha e explique o código identificado', user_id='cliente1988'))
    assert 'KNOWLEDGE_AGENT' in result.agent_used
    assert 'código 91' in result.response.lower()
    assert any(s.action=='dynamic_replan' and s.details['trigger']=='dependent_observation' for s in result.trace)


def test_tool_failure_replans_to_human_and_keeps_failure_visible(orchestrator, monkeypatch):
    monkeypatch.setattr(orchestrator.router, 'plan', lambda **kw: plan('CUSTOMER_SUPPORT_AGENT'))
    monkeypatch.setattr(orchestrator.customer_support_agent, 'run', lambda **kw: SpecialistOutput(
        agent_name='CUSTOMER_SUPPORT_AGENT', response='Consulta privada indisponível.', abstained=True,
        tool_results=[ToolExecutionResult(tool_name='get_sales_settlement',success=False,error_code='TOOL_TIMEOUT',outcome_state='failed')]))
    monkeypatch.setattr(orchestrator.escalation_agent, 'run', lambda **kw: SpecialistOutput(
        agent_name='HUMAN_ESCALATION_AGENT', response='Revisão humana registrada.', structured_data={'protocol_number':'GET-ESC-TEST'}))
    result = orchestrator.process(ChatRequest(message='Consulte minhas vendas', user_id='cliente1988'))
    assert result.escalated is True
    assert result.abstained is True
    assert 'Consulta privada indisponível.' in result.response
    assert 'Revisão humana registrada.' in result.response
    assert any(s.action=='dynamic_replan' and s.details['trigger']=='operational_tool_failure' for s in result.trace)


def test_replan_does_not_repeat_same_agent_without_progress(orchestrator, monkeypatch):
    monkeypatch.setattr(orchestrator.router,'plan',lambda **kw: plan('CUSTOMER_SUPPORT_AGENT'))
    calls = []
    def support(**kw):
        calls.append(kw)
        return diagnosis()
    monkeypatch.setattr(orchestrator.customer_support_agent,'run',support)
    monkeypatch.setattr(orchestrator.router,'replan',lambda **kw: plan('CUSTOMER_SUPPORT_AGENT'))
    result=orchestrator.process(ChatRequest(message='Explique o código identificado da minha maquininha',user_id='cliente1988'))
    assert len(calls)==1
    assert any(s.action=='replan_no_progress' for s in result.trace)


def test_pending_action_proposal_is_exposed_without_claiming_execution(orchestrator, monkeypatch):
    monkeypatch.setattr(orchestrator.router,'plan',lambda **kw: plan('CUSTOMER_SUPPORT_AGENT'))
    proposal={'proposal_id':'proposal-1','payload_hash':'a'*64,'status':'PENDING_CONFIRMATION','terminal_id':'GET-SMART-0042'}
    monkeypatch.setattr(orchestrator.customer_support_agent,'run',lambda **kw: SpecialistOutput(
        agent_name='CUSTOMER_SUPPORT_AGENT',response='Confirme a proposta; nenhum comando enviado.',
        tool_results=[ToolExecutionResult(tool_name='execute_terminal_remote_action',success=True,data=proposal,outcome_state='pending')],
        structured_data={'execute_terminal_remote_action':proposal}))
    result=orchestrator.process(ChatRequest(message='Reinicie minha maquininha',user_id='cliente1988'))
    assert result.action_proposals == [proposal]


def test_router_native_replan_observes_results_and_returns_validated_delegation(monkeypatch):
    monkeypatch.setattr(settings,'GEMINI_API_KEY','synthetic-never-sent')
    class Provider:
        models=None
        def __init__(self): self.models=self
        def generate_content(self, **kw):
            serialized=' '.join(c.model_dump_json() for c in kw['contents'])
            assert 'decline_code' in serialized and '91' in serialized
            assert 'specialist_observations' in serialized
            return types.GenerateContentResponse(candidates=[types.Candidate(finish_reason='STOP',content=types.Content(role='model',parts=[types.Part(function_call=types.FunctionCall(name='KNOWLEDGE_AGENT',args={'sub_query':'Explique o código 91','rationale':'Código identificado no diagnóstico'}))]))])
    observed=[dict(diagnosis().model_dump(),completed=True)]
    result=RouterAgent(Provider()).replan('Explique o código identificado','cliente1988',observed)
    assert result.plan_steps[0].sub_query=='Explique o código 91'


def test_router_native_replan_provider_failure_never_falls_back(monkeypatch):
    monkeypatch.setattr(settings,'GEMINI_API_KEY','synthetic-never-sent')
    class Provider:
        models=None
        def __init__(self): self.models=self
        def generate_content(self, **kw): raise RuntimeError('provider transport failed')
    with pytest.raises(RouterPlanningError):
        RouterAgent(Provider()).replan('Explique o código','cliente1988',[dict(diagnosis().model_dump(),completed=True)])

from app.agents.function_calling import initial_contents


def test_verified_diagnostic_slots_persist_after_restart_and_reach_provider(orchestrator, state, monkeypatch):
    monkeypatch.setattr(settings,'MEMORY_WINDOW_MESSAGES',2)
    monkeypatch.setattr(orchestrator.router,'plan',lambda **kw: plan('CUSTOMER_SUPPORT_AGENT'))
    monkeypatch.setattr(orchestrator.customer_support_agent,'run',lambda **kw: diagnosis())
    orchestrator.process(ChatRequest(message='Verifique minha maquininha',user_id='cliente1988',session_id='observed-memory'))
    for text in ('Obrigado','Continuo aqui','Mais tarde','Olá de novo'):
        state.append_message('observed-memory','cliente1988','user',text)
    context=StateStore(state.db_path).get_conversation_context('observed-memory','cliente1988',limit=2)
    assert context['entities']['terminal_id']=='GET-SMART-0042'
    assert context['entities']['decline_code']=='91'
    payload=' '.join(c.model_dump_json() for c in initial_contents('E ela?',context))
    assert 'GET-SMART-0042' in payload and 'decline_code' in payload


def test_comparison_pair_survives_three_turns_without_turning_new_pix_into_old_product(state):
    state.append_message('products-v3','cliente1988','user','Qual a diferença entre Get Clássica e Get Smart?')
    state.append_message('products-v3','cliente1988','user','E qual delas tem impressora?')
    state.append_message('products-v3','cliente1988','user','E a outra?')
    context=state.get_conversation_context('products-v3','cliente1988')
    assert context['session_facts']['product_comparison']==['Get Clássica','Get Smart']
    assert context['entities']['products']==['Get Clássica','Get Smart']
    state.append_message('products-v3','cliente1988','user','Quero saber de Pix')
    assert state.get_conversation_context('products-v3','cliente1988')['session_facts']['last_product']=='Pix'


def test_native_replan_can_stop_explicitly_without_a_fabricated_delegation(monkeypatch):
    monkeypatch.setattr(settings,'GEMINI_API_KEY','synthetic-never-sent')
    class Provider:
        models=None
        def __init__(self):self.models=self
        def generate_content(self,**kw):
            return types.GenerateContentResponse(candidates=[types.Candidate(finish_reason='STOP',content=types.Content(role='model',parts=[types.Part(function_call=types.FunctionCall(name='finish_supervision',args={'rationale':'Sem próxima consulta necessária'}))]))])
    result=RouterAgent(Provider()).replan('Consulte o cadastro','cliente1988',[dict(diagnosis().model_dump(),completed=True)])
    assert result.plan_steps==[]


def test_replanning_failure_keeps_completed_data_and_marks_partial_result(orchestrator,monkeypatch):
    monkeypatch.setattr(orchestrator.router,'plan',lambda **kw:plan('CUSTOMER_SUPPORT_AGENT'))
    monkeypatch.setattr(orchestrator.customer_support_agent,'run',lambda **kw:diagnosis())
    def failed(**kw):raise RouterPlanningError('provider unavailable')
    monkeypatch.setattr(orchestrator.router,'replan',failed)
    result=orchestrator.process(ChatRequest(message='Explique código identificado',user_id='cliente1988'))
    assert 'Código observado: 91.' in result.response
    assert result.abstained is True
    assert 'replanejamento' in result.response.lower()

@pytest.mark.parametrize('question',[
    'Quem é o presidente do Brasil?',
    'Quanto está a libra hoje?',
    'O tempo em recife amanhã?',
    'What is the yen rate today?',
])
def test_offline_router_covers_external_intents_beyond_original_demo(question,monkeypatch):
    monkeypatch.setattr(settings,'GEMINI_API_KEY',None)
    result=RouterAgent().plan(question,'cliente1988')
    assert [s.target_agent for s in result.plan_steps]==['WEB_SEARCH_AGENT']


def test_offline_router_does_not_treat_cadastro_as_canadian_currency(monkeypatch):
    monkeypatch.setattr(settings,'GEMINI_API_KEY',None)
    result=RouterAgent().plan('Consulte meu cadastro','cliente1988')
    assert [s.target_agent for s in result.plan_steps]==['CUSTOMER_SUPPORT_AGENT']


def test_real_three_turn_comparison_never_answers_printer_with_unrelated_apps(orchestrator):
    for message in ('Qual a diferença entre Get Clássica e Get Smart?', 'E qual delas tem impressora?'):
        result=orchestrator.process(ChatRequest(message=message,user_id='cliente1988',session_id='printer-comparison'))
    # A printer-specific answer needs printer evidence; a related Get Store chunk is insufficient.
    assert result.abstained or any(word in result.response.lower() for word in ('impressora','impressão','imprimir'))
    result=orchestrator.process(ChatRequest(message='E a outra?',user_id='cliente1988',session_id='printer-comparison'))
    # Without a unique established referent, the safe result is a clarification/abstention.
    assert result.abstained or any(word in result.response.lower() for word in ('impressora','impressão','imprimir'))

@pytest.mark.parametrize('code',['INSUFFICIENT_TOOL_SCOPE','HITL_ACTIVE','IDEMPOTENCY_CONFLICT'])
def test_offline_feedback_does_not_turn_policy_denials_into_new_side_effects(code,monkeypatch):
    monkeypatch.setattr(settings,'GEMINI_API_KEY',None)
    observation=SpecialistOutput(agent_name='CUSTOMER_SUPPORT_AGENT',response='Operação negada.',abstained=True,
        tool_results=[ToolExecutionResult(tool_name='execute_terminal_remote_action',success=False,error_code=code)]).model_dump()
    result=RouterAgent().replan('Consulte o terminal','cliente1988',[observation])
    assert result.plan_steps==[]

@pytest.mark.parametrize('question',[
    'Please diagnose terminal GET-UNKNOWN-45678 and check its connectivity.',
    'Check the status of GET-UNKNOWN-76543.',
    'Can you inspect the signal of my terminal GET-UNKNOWN-22222?',
    'Verifique a conexão do equipamento GET-UNKNOWN-33333.',
])
def test_explicit_terminal_diagnostic_query_routes_private_support(question,monkeypatch):
    monkeypatch.setattr(settings,'GEMINI_API_KEY',None)
    result=RouterAgent().plan(question,'cliente1988')
    assert [s.target_agent for s in result.plan_steps]==['CUSTOMER_SUPPORT_AGENT']


@pytest.mark.parametrize('question',[
    'Where can I find the manual for GET-SMART-88219?',
    'What does the user guide say about connection for GET-SMART-88219?',
])
def test_terminal_id_inside_manual_request_does_not_force_account_lookup(question,monkeypatch):
    monkeypatch.setattr(settings,'GEMINI_API_KEY',None)
    result=RouterAgent().plan(question,'cliente1988')
    assert [s.target_agent for s in result.plan_steps]==['KNOWLEDGE_AGENT']

@pytest.mark.parametrize('question',[
    'My terminal has no signal.',
    'My POS is offline; please inspect it.',
    'Could you check my terminal connection?',
])
def test_operational_english_terminal_query_without_id_routes_support(question,monkeypatch):
    monkeypatch.setattr(settings,'GEMINI_API_KEY',None)
    assert RouterAgent().plan(question,'cliente1988').primary_agent=='CUSTOMER_SUPPORT_AGENT'

@pytest.mark.parametrize('question',[
    'When will the proceeds of my card takings from 2026-10-22 settle?',
    'Check settlement for sales dated 2026-10-22.',
    'Analise os recebimentos de 22/10/2026 da minha conta.',
    'Confira o repasse da venda de ontem.',
])
def test_private_dated_sales_queries_route_operational_data(question,monkeypatch):
    monkeypatch.setattr(settings,'GEMINI_API_KEY',None)
    assert RouterAgent().plan(question,'cliente1988').primary_agent=='CUSTOMER_SUPPORT_AGENT'


@pytest.mark.parametrize('question',[
    'What does sales settlement mean in the product documentation?',
    'Como funciona a liquidação no manual do produto?',
])
def test_generic_settlement_documentation_remains_public_knowledge(question,monkeypatch):
    monkeypatch.setattr(settings,'GEMINI_API_KEY',None)
    assert RouterAgent().plan(question,'cliente1988').primary_agent=='KNOWLEDGE_AGENT'

@pytest.mark.parametrize('question',[
    'Do I need a bank account to receive my sales by Pix?',
    'Preciso ter conta bancária para receber minhas vendas via Pix?',
])
def test_account_requirement_is_product_knowledge_even_when_sales_are_personal(question,monkeypatch):
    monkeypatch.setattr(settings,'GEMINI_API_KEY',None)
    assert RouterAgent().plan(question,'cliente1988').primary_agent=='KNOWLEDGE_AGENT'

@pytest.mark.parametrize('question',[
    'Check the status of GET-UNKNOWN-76543.',
    'Analise a conectividade de GET-UNKNOWN-76543.',
])
def test_real_unknown_terminal_keeps_requested_target_and_returns_not_found(orchestrator,question):
    result=orchestrator.process(ChatRequest(message=question,user_id='cliente1988',session_id='unknown-terminal-check'))
    assert 'diagnose_terminal_issue' in result.tools_called
    assert 'GET-UNKNOWN-76543' in result.response
    assert 'GET-SMART-88219' not in result.response
    assert 'GET-CLASSICA-10442' not in result.response
    assert 'não' in result.response.lower()
