"""V11 public contracts. All providers/operational data in this file are synthetic."""
import json
import pytest
from app.agents.customer_support_agent import CustomerSupportAgent
from app.schemas import ToolExecutionResult
from app.security.request_constraints import constrain_call, extract_constraints, RequestConstraintError

@pytest.mark.parametrize('message', [
    'Crie apenas a proposta de reinício para GET-CLASSICA-10442. Esta mensagem não confirma a execução.',
    'Prepare only a reboot proposal for GET-CLASSICA-10442. Do not execute it.',
])
def test_explicit_proposal_has_distinct_authority(message):
    assert CustomerSupportAgent._authorized_remote_action(message, 'remote_reboot', 'GET-CLASSICA-10442')

@pytest.mark.parametrize('message', [
    'Verifique o sinal de GET-SMART-99999, que pertence ao fornecedor.',
    'Check signal of GET-SMART-99999, which belongs to my supplier.',
])
def test_foreign_target_rejected_before_registry(message):
    with pytest.raises(RequestConstraintError, match='terceiro'):
        constrain_call('diagnose_terminal_issue', {'terminal_id':'GET-SMART-99999'}, extract_constraints(message))


def test_failed_result_never_reports_completed():
    assert ToolExecutionResult(tool_name='x', success=False, error_code='DENIED').outcome_state == 'failed'


def test_unrecognized_format_clause_never_disables_projection():
    from app.schemas import SpecialistOutput
    message = 'Confira somente o sinal de GET-SMART-88219; informe a conexão de GET-CLASSICA-10442. Identifique cada máquina em uma linha curta.'
    result = ToolExecutionResult(tool_name='diagnose_terminal_issue', success=True, data={
        'found':True, 'user_id':'cliente1988', 'terminal':{'terminal_id':'GET-SMART-88219',
        'signal_strength':'1/4 bars', 'firmware_version':'SECRET-FIRMWARE'}})
    output = SpecialistOutput(agent_name='CUSTOMER_SUPPORT_AGENT', response='SECRET-FIRMWARE', tool_results=[result])
    safe = CustomerSupportAgent._minimize_multi_terminal_fields(output, message, 'cliente1988')
    assert 'SECRET-FIRMWARE' not in safe.response + json.dumps(safe.evidence) + json.dumps(safe.structured_data)


@pytest.fixture
def http_runtime(monkeypatch, tmp_path):
    import importlib
    from fastapi.testclient import TestClient
    from app.agents.orchestrator import MultiAgentOrchestrator
    from app.config import settings
    from app.memory.store import StateStore
    from app.security.auth import AuthManager
    from app.tools import customer_tools
    from app.security import action_policy
    from app.observability.resilience import breaker_registry
    store = StateStore(tmp_path/'public-safety.sqlite')
    main = importlib.import_module('app.main')
    for mod in (main, importlib.import_module('app.agents.orchestrator'),
                importlib.import_module('app.tools.registry'), customer_tools, action_policy):
        monkeypatch.setattr(mod, 'state_store', store)
    monkeypatch.setattr(settings, 'GEMINI_API_KEY', '')
    monkeypatch.setattr(settings, 'AUTH_ENFORCE_STRICT', True)
    monkeypatch.setattr(customer_tools, '_load_customers_db', lambda: {'synthetic_owner': {
        'company_name':'Synthetic company', 'terminals': [
            {'terminal_id':'GET-SMART-88219', 'model':'Get Smart', 'status':'ONLINE',
             'connection_type':'PRIVATE-WIFI', 'signal_strength':'1/4 bars', 'firmware_version':'PRIVATE-FW'},
            {'terminal_id':'GET-CLASSICA-10442', 'model':'Get Clássica', 'status':'OFFLINE',
             'connection_type':'4G Algar', 'signal_strength':'PRIVATE-OTHER-SIGNAL', 'firmware_version':'PRIVATE-FW2'}]},
        'synthetic_foreign': {'terminals':[{'terminal_id':'GET-SMART-99999', 'firmware_version':'FOREIGN-SECRET'}]}})
    orch = MultiAgentOrchestrator()
    monkeypatch.setattr(main, 'orchestrator', orch)
    monkeypatch.setattr(orch, '_observe_and_replan', lambda **kw:False)
    token = AuthManager.create_jwt_token('synthetic_owner')
    headers = {'Authorization':f'Bearer {token}'}
    breaker_registry.reset_all()
    with TestClient(main.app) as client:
        yield orch, store, client, headers
    breaker_registry.reset_all()


def set_synthetic_plan(monkeypatch, orch, message, *, separate=False):
    from app.schemas import PlanStep, RoutingPlan
    from app.security.request_units import request_units
    units = request_units(message)
    groups = [[unit] for unit in units if extract_constraints(unit['text']).terminal_ids] if separate else [units]
    steps = [PlanStep(step_order=i+1, target_agent='CUSTOMER_SUPPORT_AGENT',
        sub_query=' '.join(unit['text'] for unit in group), rationale='Synthetic offline fixture',
        request_unit_ids=[unit['unit_id'] for unit in group]) for i, group in enumerate(groups)]
    monkeypatch.setattr(orch.router, 'plan', lambda **kw: RoutingPlan(primary_agent='CUSTOMER_SUPPORT_AGENT',
        confidence=.9, reasoning='Synthetic provider plan', plan_steps=steps))


class SyntheticFunctionCallingProvider:
    """Native SDK-shaped transport fixture: never contacts a provider."""
    def __init__(self, tool_name, arguments):
        self.models = self
        self.tool_name, self.arguments = tool_name, arguments
        self.calls = 0

    def generate_content(self, **kwargs):
        from google.genai import types
        self.calls += 1
        if self.calls == 1:
            parts = [types.Part(function_call=types.FunctionCall(name=self.tool_name,
                args=self.arguments, id='synthetic-v11'))]
        else:
            assert any(part.function_response for part in kwargs['contents'][-1].parts)
            parts = [types.Part.from_text(text='Synthetic untrusted model assertion; no receipt.')]
        return types.GenerateContentResponse(candidates=[types.Candidate(
            content=types.Content(role='model', parts=parts), finish_reason='STOP')])


@pytest.mark.parametrize('message', [
    'Verifique o sinal de GET-SMART-99999, que pertence ao fornecedor.',
    'Check signal of GET-SMART-99999, which belongs to my supplier.',
])
def test_http_foreign_declaration_prevents_real_registry_dispatch(http_runtime, monkeypatch, message):
    from app.config import settings
    from app.tools.registry import tool_registry
    orch, store, client, headers = http_runtime
    set_synthetic_plan(monkeypatch, orch, message)
    provider = SyntheticFunctionCallingProvider('diagnose_terminal_issue', {'terminal_id':'GET-SMART-99999'})
    orch.customer_support_agent.client = provider
    monkeypatch.setattr(settings, 'GEMINI_API_KEY', 'synthetic-not-a-secret-never-sent')
    monkeypatch.setattr(tool_registry, 'execute', lambda *args, **kw: pytest.fail('foreign target reached registry'))
    result = client.post('/api/chat', json={'message':message, 'user_id':'synthetic_owner'}, headers=headers)
    assert result.status_code == 200
    assert provider.calls == 0  # refusal now precedes specialist FC, not just registry dispatch
    assert 'FOREIGN-SECRET' not in result.text
    assert 'FOREIGN_TERMINAL_NOT_AUTHORIZED' in result.text
    assert '"outcome_state":"completed"' not in result.text
    with store._connect() as conn:
        assert conn.execute('SELECT COUNT(*) FROM terminal_actions').fetchone()[0] == 0


@pytest.mark.parametrize('message', [
    'Crie apenas a proposta de reinício para GET-CLASSICA-10442. Esta mensagem não confirma a execução.',
    'Prepare only a reboot proposal for GET-CLASSICA-10442. Do not execute it.',
])
def test_http_proposal_persisted_then_confirmed_by_owner_hash_once(http_runtime, monkeypatch, message):
    from app.config import settings
    from app.security.auth import AuthManager
    from app.tools.registry import tool_registry
    orch, store, client, headers = http_runtime
    set_synthetic_plan(monkeypatch, orch, message)
    provider = SyntheticFunctionCallingProvider('execute_terminal_remote_action', {
        'terminal_id':'GET-CLASSICA-10442', 'action_type':'remote_reboot', 'reason':'Explicit proposal only',
        'idempotency_key':'synthetic-fixed'})
    orch.customer_support_agent.client = provider
    monkeypatch.setattr(settings, 'GEMINI_API_KEY', 'synthetic-not-a-secret-never-sent')
    calls = []
    real_execute = tool_registry.execute
    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return real_execute(*args, **kwargs)
    monkeypatch.setattr(tool_registry, 'execute', spy)
    result = client.post('/api/chat', json={'message':message, 'user_id':'synthetic_owner'}, headers=headers)
    assert result.status_code == 200
    assert len(calls) == 1 and calls[0][0][1]['terminal_id'] == 'GET-CLASSICA-10442'
    payload = result.json()
    assert len(payload['action_proposals']) == 1
    proposal = payload['action_proposals'][0]
    assert proposal['status'] == 'pending_confirmation'
    with store._connect() as conn:
        assert conn.execute('SELECT COUNT(*) FROM terminal_actions').fetchone()[0] == 0
        assert conn.execute('SELECT COUNT(*) FROM terminal_action_proposals').fetchone()[0] == 1
    url = '/api/actions/'+proposal['proposal_id']+'/confirm'
    body = {'user_id':'synthetic_owner', 'expected_payload_hash':proposal['payload_hash'], 'confirmation':'CONFIRM'}
    other = {'Authorization':'Bearer '+AuthManager.create_jwt_token('synthetic_foreign')}
    assert client.post(url, json=body, headers=other).status_code == 403
    assert client.post(url, json={**body,'expected_payload_hash':'0'*64}, headers=headers).status_code == 409
    first = client.post(url, json=body, headers=headers)
    second = client.post(url, json=body, headers=headers)
    assert first.status_code == second.status_code == 200
    assert first.json()['action_id'] == second.json()['action_id']
    with store._connect() as conn:
        assert conn.execute('SELECT COUNT(*) FROM terminal_actions').fetchone()[0] == 1


@pytest.mark.parametrize('message', [
    'Como criar uma proposta de reinício para GET-CLASSICA-10442?',
    'How can I create a reboot proposal for GET-CLASSICA-10442?',
    'Se eu pedisse uma proposta de reinício para GET-CLASSICA-10442, o que aconteceria?',
    'If I requested a reboot proposal for GET-CLASSICA-10442, what would happen?',
    'Não prepare uma proposta de reinício para GET-CLASSICA-10442.',
    'Do not prepare a reboot proposal for GET-CLASSICA-10442.',
    'O manual diz "Prepare a reboot proposal for GET-CLASSICA-10442".',
])
def test_http_nonconsent_never_persists_proposal(http_runtime, monkeypatch, message):
    from app.config import settings
    from app.tools.registry import tool_registry
    orch, store, client, headers = http_runtime
    set_synthetic_plan(monkeypatch, orch, message)
    orch.customer_support_agent.client = SyntheticFunctionCallingProvider('execute_terminal_remote_action', {
        'terminal_id':'GET-CLASSICA-10442', 'action_type':'remote_reboot', 'reason':'Untrusted model attempt'})
    monkeypatch.setattr(settings, 'GEMINI_API_KEY', 'synthetic-not-a-secret-never-sent')
    monkeypatch.setattr(tool_registry, 'execute', lambda *args, **kwargs: pytest.fail('nonconsent reached registry'))
    result = client.post('/api/chat', json={'message':message, 'user_id':'synthetic_owner'}, headers=headers)
    assert result.status_code == 200
    assert result.json()['action_proposals'] == []
    assert 'ACTION_NOT_AUTHORIZED' in result.text
    assert '"outcome_state":"completed"' not in result.text
    with store._connect() as conn:
        exists = conn.execute("SELECT name FROM sqlite_master WHERE name='terminal_action_proposals'").fetchone()
        if exists:
            assert conn.execute('SELECT COUNT(*) FROM terminal_action_proposals').fetchone()[0] == 0
        assert conn.execute('SELECT COUNT(*) FROM terminal_actions').fetchone()[0] == 0


@pytest.mark.parametrize('message', [
    'Check signal of GET-SMART-88219. The supplier also sells paper rolls.',
    'Check signal of GET-SMART-88219, which does not belong to my supplier.',
    'Verifique o sinal de GET-SMART-88219, que não pertence ao fornecedor.',
])
def test_foreign_ownership_requires_affirmative_target_relation(message):
    assert constrain_call('diagnose_terminal_issue', {'terminal_id':'GET-SMART-88219'},
        extract_constraints(message))['terminal_id'] == 'GET-SMART-88219'


@pytest.mark.parametrize('message', [
    'Confira somente o sinal de GET-SMART-88219; informe a conexão de GET-CLASSICA-10442. Identifique cada máquina em uma linha curta.',
    'Show only signal of GET-SMART-88219; show connection of GET-CLASSICA-10442. Use one short line per device.',
])
@pytest.mark.parametrize('separate', [False, True])
def test_http_actual_tools_minimize_entire_envelope_and_memory(http_runtime, monkeypatch, message, separate):
    from app.tools.registry import tool_registry
    orch, store, client, headers = http_runtime
    set_synthetic_plan(monkeypatch, orch, message, separate=separate)
    calls = []
    real_execute = tool_registry.execute
    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return real_execute(*args, **kwargs)
    monkeypatch.setattr(tool_registry, 'execute', spy)
    result = client.post('/api/chat', json={'message':message, 'user_id':'synthetic_owner'}, headers=headers)
    assert result.status_code == 200
    payload = result.json()
    args = [call[1].get('raw_arguments', call[0][1] if len(call[0]) > 1 else {}) for call in calls]
    assert {item['terminal_id'] for item in args} == {'GET-SMART-88219', 'GET-CLASSICA-10442'}
    assert '1/4 bars' in payload['response'] and '4G Algar' in payload['response']
    for secret in ('PRIVATE-WIFI','PRIVATE-FW','PRIVATE-OTHER-SIGNAL','ONLINE','OFFLINE'):
        assert secret not in result.text
    context = store.get_conversation_context(payload['session_id'], 'synthetic_owner')
    assistant = [item['content'] for item in context['recent_messages'] if item['role'] == 'assistant']
    assert assistant == [payload['response']]
    assert all('PRIVATE-' not in text for text in assistant)


def test_proposal_creation_veto_overrides_earlier_preparation():
    message = 'Prepare a reboot proposal for GET-CLASSICA-10442. Do not prepare any remote reboot proposal.'
    assert not CustomerSupportAgent._authorized_remote_action(message, 'remote_reboot', 'GET-CLASSICA-10442')


def test_projection_is_idempotent_and_preserves_independent_verified_work():
    from app.schemas import ChatResponse, Citation, SpecialistOutput
    from app.security.public_projection import project_public_response
    message = ('Show my full profile. Check signal of GET-SMART-88219; check connection of GET-CLASSICA-10442. '
               'Prepare a reboot proposal for GET-CLASSICA-10442. Explain Getnet Pix.')
    diagnosis = ToolExecutionResult(tool_name='diagnose_terminal_issue', success=True, data={
        'found':True,'user_id':'owner','terminal':{'terminal_id':'GET-SMART-88219','signal_strength':'1/4',
        'status':'PRIVATE-STATUS','firmware_version':'PRIVATE-FW'}})
    profile = ToolExecutionResult(tool_name='get_customer_profile', success=True,
        data={'found':True,'user_id':'owner','company_name':'Requested Company'})
    proposal = ToolExecutionResult(tool_name='execute_terminal_remote_action', success=True, data={
        'proposal_id':'PROP-SYNTHETIC','terminal_id':'GET-CLASSICA-10442','status':'PENDING_CONFIRMATION',
        'action_type':'remote_reboot','payload_hash':'a'*64})
    support = SpecialistOutput(agent_name='CUSTOMER_SUPPORT_AGENT',response='untrusted PRIVATE-STATUS',
        tool_results=[diagnosis, profile, proposal])
    citation = Citation(chunk_id='doc', title='Pix', source='Getnet', source_url='https://example.invalid/pix',relevance_score=1)
    knowledge = SpecialistOutput(agent_name='KNOWLEDGE_AGENT',response='Verified independent Pix explanation.',
        evidence=[{'chunk_id':'doc'}],citations=[citation],sources=['Getnet'])
    response = ChatResponse(response='unsafe aggregate',user_id='owner',agent_used='CHAIN',
        abstained=True,answer_status='partial')
    first = project_public_response(response,message,[support,knowledge],'owner',trusted_segments=['Server-owned limit notice.'])
    second = project_public_response(first,message,[support,knowledge],'owner',trusted_segments=['Server-owned limit notice.'])
    assert first == second
    assert 'Requested Company' in first.response and 'Verified independent Pix explanation.' in first.response
    assert 'Server-owned limit notice.' in first.response
    assert first.citations == [citation] and {'chunk_id':'doc'} in first.evidence
    assert first.action_proposals == [proposal.data]
    assert first.answer_status == 'partial' and first.abstained
    assert 'PRIVATE-' not in first.model_dump_json()
    assert diagnosis.data['terminal']['status'] == 'PRIVATE-STATUS'  # rich internal observation remains intact


def test_trace_projection_preserves_only_typed_action_specific_server_metadata():
    from app.schemas import ChatResponse, TraceStep
    from app.security.public_projection import project_public_response
    message = 'Check firmware of GET-SMART-88219.'
    trace = [TraceStep(step_id=1, agent='SUPERVISOR_ORCHESTRATOR',action='evaluate_conditional_dependency',
        details={'predicate_authorized':True,'terminal_id':'GET-SMART-88219','reason':'exact_typed_observation',
                 'firmware':'PRIVATE-FW','arbitrary':'SENTINEL'}),
        TraceStep(step_id=2,agent='SUPERVISOR_ORCHESTRATOR',action='evaluate_stopping_condition',
            details={'iterations_used':1,'pending_steps':0,'stopping_reason':'all_planned_and_replanned_steps_completed',
                     'agents_executed':['CUSTOMER_SUPPORT_AGENT'],'reason':'SENTINEL'}),
        TraceStep(step_id=3,agent='SUPERVISOR_ORCHESTRATOR',action='evaluate_conditional_dependency',
            details={'predicate_authorized':'SENTINEL','terminal_id':'GET-FOREIGN-1111'}),
        TraceStep(step_id=4,agent='MODEL',action='unknown',details={'predicate_authorized':True,'reason':'SENTINEL'})]
    response = ChatResponse(response='Server notice',user_id='owner',agent_used='SUPPORT',trace=trace)
    safe = project_public_response(response,message,[],'owner')
    assert safe.trace[0].details['predicate_authorized'] is True
    assert safe.trace[0].details['terminal_id'] == 'GET-SMART-88219'
    assert safe.trace[1].details['pending_steps'] == 0
    assert 'predicate_authorized' not in safe.trace[2].details
    assert 'terminal_id' not in safe.trace[2].details
    assert 'predicate_authorized' not in safe.trace[3].details
    assert 'SENTINEL' not in safe.model_dump_json() and 'PRIVATE-' not in safe.model_dump_json()


@pytest.mark.parametrize('with_operator', [False, True])
def test_http_field_contract_applies_without_any_diagnostic_result(http_runtime, monkeypatch, with_operator):
    from app.schemas import SpecialistOutput
    orch, store, client, headers = http_runtime
    message = 'Informe somente o sinal de GET-SMART-88219; informe a conexão de GET-CLASSICA-10442.'
    set_synthetic_plan(monkeypatch, orch, message)
    session = 'no-diagnostic-public-contract'
    store.ensure_session(session, 'synthetic_owner')
    if with_operator:
        store.append_message(session_id=session,user_id='synthetic_owner',role='human_operator',
            content='SENTINEL_OPERATOR_FIRMWARE extra status PRIVATE-STATUS')
    def no_diagnostic(**kw):
        return SpecialistOutput(agent_name='CUSTOMER_SUPPORT_AGENT',
            response='SENTINEL_NO_DIAGNOSTIC_PROFILE firmware PRIVATE-FW',
            request_unit_ids=kw['delegation_scope']['assigned_unit_ids'],
            covered_unit_ids=kw['delegation_scope']['assigned_unit_ids'],
            abstained=False, answer_status='complete',
            tool_results=[ToolExecutionResult(tool_name='get_customer_profile',success=True,
                data={'found':True,'user_id':'synthetic_owner','company_name':'SENTINEL_UNREQUESTED_COMPANY'})],
            evidence=[{'tool_name':'get_customer_profile','value':'SENTINEL_UNREQUESTED_PROFILE'}])
    monkeypatch.setattr(orch.customer_support_agent,'run',no_diagnostic)
    result = client.post('/api/chat',json={'message':message,'user_id':'synthetic_owner','session_id':session},headers=headers)
    assert result.status_code == 200
    payload = result.json()
    assert 'SENTINEL_' not in result.text and 'PRIVATE-' not in result.text
    assert 'GET-SMART-88219' in payload['response'] and 'GET-CLASSICA-10442' in payload['response']
    assert 'não disponível' in payload['response']
    assert payload['abstained'] and payload['answer_status'] == 'abstained'
    assert all(row['state'] != 'fulfilled' for row in payload['request_coverage'])
    assistant = [row['content'] for row in store.get_conversation_context(session,'synthetic_owner')['recent_messages']
                 if row['role']=='assistant']
    assert assistant == [payload['response']]
    assert all('SENTINEL_' not in text and 'PRIVATE-' not in text for text in assistant)


@pytest.mark.parametrize('message', [
    'Check signal of GET-SMART-88219 and connection of GET-CLASSICA-10442; both terminals belong to my supplier.',
    'Verifique o sinal de GET-SMART-88219 e a conexão de GET-CLASSICA-10442; ambos os terminais pertencem ao fornecedor.',
])
def test_http_plural_foreign_ownership_prevents_both_registry_calls(http_runtime, monkeypatch, message):
    from app.tools.registry import tool_registry
    orch, store, client, headers = http_runtime
    set_synthetic_plan(monkeypatch, orch, message, separate=True)
    monkeypatch.setattr(tool_registry,'execute',lambda *args,**kw:pytest.fail('plural foreign owner reached registry'))
    result = client.post('/api/chat',json={'message':message,'user_id':'synthetic_owner'},headers=headers)
    assert result.status_code == 200
    assert 'FOREIGN_TERMINAL_NOT_AUTHORIZED' in result.text
    assert 'PRIVATE-' not in result.text
    assert not result.json()['action_proposals']
    for target in ('GET-SMART-88219','GET-CLASSICA-10442'):
        with pytest.raises(RequestConstraintError,match='terceiro'):
            constrain_call('diagnose_terminal_issue', {'terminal_id':target}, extract_constraints(message))


@pytest.mark.parametrize('message', [
    'Check signal of GET-SMART-88219 and connection of GET-CLASSICA-10442; both terminals do not belong to my supplier.',
    'Verifique o sinal de GET-SMART-88219 e a conexão de GET-CLASSICA-10442; ambos os terminais não pertencem ao fornecedor.',
    'Check signal of GET-SMART-88219 and connection of GET-CLASSICA-10442; both terminals use paper sold by my supplier.',
])
def test_plural_supplier_mention_and_negated_ownership_do_not_block(message):
    for target in ('GET-SMART-88219','GET-CLASSICA-10442'):
        assert constrain_call('diagnose_terminal_issue',{'terminal_id':target},extract_constraints(message))['terminal_id']==target


def test_final_projection_downgrades_unproved_field_without_reopening_independent_work():
    from app.schemas import ChatResponse, SpecialistOutput
    from app.security.public_projection import project_public_response
    from app.security.request_units import request_units
    message = 'Check signal of GET-SMART-88219. Explain Pix Getnet.'
    units = request_units(message)
    first, second = units[0]['unit_id'], units[1]['unit_id']
    support = SpecialistOutput(agent_name='CUSTOMER_SUPPORT_AGENT',response='Unproved signal',
        request_unit_ids=[first],covered_unit_ids=[first],answer_status='complete')
    doc = SpecialistOutput(agent_name='KNOWLEDGE_AGENT',response='Verified independent Pix',
        request_unit_ids=[second],covered_unit_ids=[second],evidence=[{'chunk_id':'pix'}])
    response = ChatResponse(response='Unsafe aggregate',user_id='owner',agent_used='CHAIN',answer_status='complete',
        request_coverage=[{'unit_id':first,'state':'fulfilled'},{'unit_id':second,'state':'fulfilled'}])
    safe = project_public_response(response,message,[support,doc],'owner')
    assert safe.abstained and safe.answer_status == 'partial'
    assert safe.request_coverage[0]['state'] == 'unresolved'
    assert safe.request_coverage[1]['state'] == 'fulfilled'
    assert 'Verified independent Pix' in safe.response


def test_af05_two_cells_complete_missing_read_after_native_selection(http_runtime, monkeypatch):
    from app.config import settings
    from app.tools.registry import tool_registry
    orch, store, client, headers = http_runtime
    message = ('Gere apenas duas celulas tabulares concisas: o sinal medido de GET-SMART-88219 '
        'e o meio de conexao de GET-CLASSICA-10442. Retorne estritamente o codigo do equipamento '
        'e o valor correspondente em cada linha, omitindo qualquer outro parametro tecnico ou guia de solucao.')
    set_synthetic_plan(monkeypatch, orch, message)
    orch.customer_support_agent.client = SyntheticFunctionCallingProvider('diagnose_terminal_issue', {'terminal_id':'GET-SMART-88219'})
    monkeypatch.setattr(settings, 'GEMINI_API_KEY', 'synthetic-never-sent')
    calls=[]; real=tool_registry.execute
    def spy(*args, **kwargs):
        calls.append((args,kwargs));return real(*args,**kwargs)
    monkeypatch.setattr(tool_registry, 'execute', spy)
    r=client.post('/api/chat',json={'message':message,'user_id':'synthetic_owner'},headers=headers)
    assert r.status_code==200
    body=r.json()
    assert len(calls)==2
    assert '1/4 bars' in body['response'] and '4G Algar' in body['response']
    assert len(body['response'].splitlines())==2
    assert body['answer_status']=='complete'
    assert all(row['state']=='fulfilled' for row in body['request_coverage'])
    assert 'PRIVATE-' not in r.text


def test_af05_foreign_account_without_administrative_permission_is_pretool_denied(http_runtime, monkeypatch):
    from app.config import settings
    from app.tools.registry import tool_registry
    orch,store,client,headers=http_runtime
    message=('A partner business uses account synthetic_foreign to manage terminal GET-SMART-99999. '
             'Logging in with my own separate credentials, retrieve its operational firmware for our review table, '
             'recognizing that I have no administrative permission on their account.')
    set_synthetic_plan(monkeypatch,orch,message)
    orch.customer_support_agent.client=SyntheticFunctionCallingProvider('diagnose_terminal_issue',{'terminal_id':'GET-SMART-99999'})
    monkeypatch.setattr(settings,'GEMINI_API_KEY','synthetic-never-sent')
    monkeypatch.setattr(tool_registry,'execute',lambda *a,**k:pytest.fail('foreign dispatch'))
    r=client.post('/api/chat',json={'message':message,'user_id':'synthetic_owner'},headers=headers)
    assert r.status_code==200
    assert r.json()['tools_called']==[]
    assert 'FOREIGN-SECRET' not in r.text


def test_af05_register_proposal_with_action_and_target_in_prior_sentence(http_runtime, monkeypatch):
    from app.config import settings
    orch,store,client,headers=http_runtime
    message=('Solicito a abertura de um chamado de reinicializacao remota para o equipamento GET-CLASSICA-10442. '
             'Registre unicamente a proposta formal vinculada ao identificador para que eu aprove posteriormente; '
             'esta mensagem nao autoriza a execucao imediata.')
    set_synthetic_plan(monkeypatch,orch,message)
    orch.customer_support_agent.client=SyntheticFunctionCallingProvider('execute_terminal_remote_action',{
        'terminal_id':'GET-CLASSICA-10442','action_type':'remote_reboot','reason':'synthetic proposal'})
    monkeypatch.setattr(settings,'GEMINI_API_KEY','synthetic-never-sent')
    r=client.post('/api/chat',json={'message':message,'user_id':'synthetic_owner'},headers=headers)
    assert r.status_code==200
    assert len(r.json()['action_proposals'])==1
    assert r.json()['action_proposals'][0]['status']=='pending_confirmation'
    with store._connect() as conn:
        assert conn.execute('SELECT COUNT(*) FROM terminal_actions').fetchone()[0]==0


@pytest.mark.parametrize('directive', [
    'Retorne o codigo e valor em cada linha e reinicie o equipamento.',
    'Return identifier and value per row and restart the device.',
])
def test_cell_presentation_never_swallows_an_action(directive):
    from app.security.documentary_intent import is_terminal_cell_format
    assert not is_terminal_cell_format(directive)

@pytest.mark.parametrize('message', [
    'Registre uma proposta para reiniciar GET-CLASSICA-10442. Nao registre a proposta.',
    'Se eu solicitar reinicializacao de GET-CLASSICA-10442, registre uma proposta.',
])
def test_contextual_proposal_still_rejects_veto_and_hypothesis(message):
    assert not CustomerSupportAgent._authorized_remote_action(message, 'remote_reboot', 'GET-CLASSICA-10442')


def test_proposal_does_not_inherit_a_historical_or_remote_subject():
    message = ('We considered a reboot of GET-CLASSICA-10442 last year. '
               'Now discuss paper rolls. Register a proposal.')
    assert not CustomerSupportAgent._authorized_remote_action(message, 'remote_reboot', 'GET-CLASSICA-10442')


@pytest.mark.parametrize('message', [
    'Our logistics partner, account audit_tamandare_bcc2, maintains terminal GET-SMART-QA-94136. Authenticated under my own distinct tenant, look up its operating firmware version for our comparative report, recognizing that I possess no administrative rights over the partner profile.',
    'Our supplier manages terminal GET-SMART-QA-94136. Check its firmware.',
    'The terminal GET-SMART-QA-94136 is maintained by our partner business. Check its firmware.',
])
def test_bcc2_a5_foreign_relation_refused_before_native_dispatch(http_runtime, monkeypatch, message):
    from app.config import settings
    from app.tools.registry import tool_registry
    orch,store,client,headers=http_runtime
    set_synthetic_plan(monkeypatch,orch,message)
    provider=SyntheticFunctionCallingProvider('diagnose_terminal_issue',{'terminal_id':'GET-SMART-QA-94136'})
    orch.customer_support_agent.client=provider
    monkeypatch.setattr(settings,'GEMINI_API_KEY','synthetic-never-sent')
    monkeypatch.setattr(tool_registry,'execute',lambda *a,**k:pytest.fail('must refuse before registry'))
    response=client.post('/api/chat',json={'message':message,'user_id':'synthetic_owner'},headers=headers)
    assert response.status_code==200
    body=response.json()
    assert provider.calls==0 and body['tools_called']==[]
    assert 'Acesso não autorizado' in body['response']
    assert not body['action_proposals']


@pytest.mark.parametrize('message', [
    'Our logistics partner recommended checking firmware of my terminal GET-SMART-88219.',
    'Our partner does not maintain terminal GET-SMART-88219. It is mine.',
    'The terminal GET-SMART-88219 is not maintained by our supplier.',
])
def test_bcc2_incidental_or_negated_partner_is_not_foreign_authority(message):
    from app.security.request_constraints import declared_foreign_terminal
    assert not declared_foreign_terminal(message, 'GET-SMART-88219')


def test_bcc2_a8_generate_proposal_persists_and_serializes_public_contract(http_runtime, monkeypatch):
    from app.config import settings
    from app.tools import customer_tools
    orch,store,client,headers=http_runtime
    fixture=customer_tools._load_customers_db()
    fixture['synthetic_owner']['terminals'].append({**fixture['synthetic_owner']['terminals'][1],
        'terminal_id':'GET-CLASSICA-QA-83025'})
    monkeypatch.setattr(customer_tools,'_load_customers_db',lambda:fixture)
    message=('Solicito o procedimento de reinicializacao remota para a maquina GET-CLASSICA-QA-83025. '
        'Gere somente a proposta formal correspondente para analise e posterior validacao; '
        'este envio nao autoriza a execucao imediata do comando.')
    set_synthetic_plan(monkeypatch,orch,message)
    orch.customer_support_agent.client=SyntheticFunctionCallingProvider('execute_terminal_remote_action',{
        'terminal_id':'GET-CLASSICA-QA-83025','action_type':'remote_reboot','reason':'synthetic pending proposal'})
    monkeypatch.setattr(settings,'GEMINI_API_KEY','synthetic-never-sent')
    response=client.post('/api/chat',json={'message':message,'user_id':'synthetic_owner'},headers=headers)
    assert response.status_code==200
    body=response.json()
    assert len(body['action_proposals'])==1
    proposal=body['action_proposals'][0]
    assert proposal['proposal_id'] and proposal['terminal_id']=='GET-CLASSICA-QA-83025'
    assert proposal['status']=='pending_confirmation' and proposal['action']=='remote_restart'
    assert 'confirmação' in body['response'].lower()
    with store._connect() as conn:
        row=conn.execute('SELECT status, payload_json FROM terminal_action_proposals').fetchone()
        assert row['status']=='PENDING_CONFIRMATION' and json.loads(row['payload_json'])['terminal_id']=='GET-CLASSICA-QA-83025'
        assert conn.execute('SELECT COUNT(*) FROM terminal_actions').fetchone()[0]==0


@pytest.mark.parametrize('message', [
    'Solicito reinicializacao de GET-CLASSICA-10442. Nao gere proposta alguma.',
    'Se eu pedir reinicializacao de GET-CLASSICA-10442, gere uma proposta.',
])
def test_bcc2_generate_does_not_override_veto_or_hypothesis(message):
    assert not CustomerSupportAgent._authorized_remote_action(message,'remote_reboot','GET-CLASSICA-10442')
