"""Behavioral regressions for audit A2/A3/A4/A7/A8; no provider transport."""
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import pytest
from google.genai import types
from pydantic import BaseModel

from app.agents.customer_support_agent import CustomerSupportAgent
from app.agents.function_calling import initial_contents
from app.config import settings
from app.memory.store import StateStore
from app.security.auth import CrossTenantAccessError, SecurityContext
from app.security.pii import PIIRedactor
from app.tools import customer_tools
from app.tools.registry import ToolRegistry, tool_registry


@pytest.fixture
def store(monkeypatch, tmp_path):
    isolated = StateStore(tmp_path / 'actions.db')
    monkeypatch.setattr(customer_tools, 'state_store', isolated)
    import app.tools.registry as registry_module
    monkeypatch.setattr(registry_module, 'state_store', isolated, raising=False)
    # The new policy is imported lazily so pre-fix tests still expose behavioral failures.
    try:
        from app.security import action_policy
        monkeypatch.setattr(action_policy, 'state_store', isolated)
    except ImportError:
        pass
    monkeypatch.setattr(settings, 'GEMINI_API_KEY', '')
    return isolated


def principal(scopes=None, user='cliente1988', role='merchant'):
    return SecurityContext(authenticated_user_id=user, role=role,
        auth_method='jwt_bearer', scopes=scopes if scopes is not None else
        ['chat:write', 'customer:read_own', 'terminal:action_own'])


def proposal(store, **kwargs):
    from app.security.action_policy import propose_action
    return propose_action('cliente1988', 'action-session',
        kwargs.pop('terminal_id', 'GET-SMART-88219'),
        kwargs.pop('action_type', 'remote_reboot'), **kwargs)


def count_actions(store):
    with store._connect() as conn:
        return conn.execute('SELECT count(*) FROM terminal_actions').fetchone()[0]


@pytest.mark.parametrize('message,secrets', [
    ('Meu CPF é 12345678901 e minha senha é Abacaxi987; meu CVV é 678.',
     ['12345678901', 'Abacaxi987', '678']),
    ('CPF eh 12345678901; password is Apricot987; CVC is 927.',
     ['12345678901', 'Apricot987', '927']),
    ('Meu CPF é: 12345678901; minha senha é "duas palavras secretas"; PIN = 8162',
     ['12345678901', 'duas palavras secretas', '8162']),
])
def test_natural_language_pii_never_reaches_memory_or_provider(store, message, secrets):
    store.append_message('private-session', 'cliente1988', 'user', message)
    context = store.get_conversation_context('private-session', 'cliente1988')
    destinations = [PIIRedactor.mask_text(message), json.dumps(context),
                    str(initial_contents(message, context))]
    for destination in destinations:
        assert all(secret not in destination for secret in secrets)


def test_structured_secrets_are_redacted_with_nonstandard_value_types():
    out = PIIRedactor.mask_structure({'cpf': 12345678901, 'pin': 8162,
                                     'cvc': 927, 'token_secreto': 'secret-value'})
    assert set(out.values()) == {'[REDACTED]'}


@pytest.mark.parametrize('name,args', [
    ('get_customer_profile', {}), ('get_sales_settlement', {}),
    ('diagnose_terminal_issue', {'terminal_id': 'GET-SMART-88219'}),
    ('execute_terminal_remote_action', {'terminal_id': 'GET-SMART-88219', 'action_type': 'remote_reboot'}),
    ('create_escalation_ticket', {'reason': 'Preciso de ajuda'}),
    ('check_escalation_ticket', {}),
])
def test_tools_enforce_least_privilege_for_merchants_and_operators(store, name, args):
    for role in ['merchant', 'operator']:
        result = tool_registry.execute(name, args, principal(scopes=[], role=role), 'action-session')
        assert not result.success and result.error_code == 'INSUFFICIENT_TOOL_SCOPE'
    assert count_actions(store) == 0


def test_explicit_nonexistent_terminal_does_not_expose_another_terminal(store):
    result = tool_registry.execute('diagnose_terminal_issue',
        {'terminal_id': 'GET-FAKE-99999', 'symptom': 'decline', 'decline_code': '05'}, principal())
    assert result.success
    assert result.data['found'] is False
    assert result.data['error_code'] == 'TERMINAL_NOT_FOUND'
    assert 'GET-SMART-88219' not in json.dumps(result.data)


def test_explicit_terminal_never_uses_transaction_from_another_machine(store):
    result = customer_tools.diagnose_terminal_issue('cliente1988',
        terminal_id='GET-CLASSICA-10442', symptom='decline', decline_code='05')
    transaction = result['last_declined_transaction']
    assert not transaction or transaction['terminal_id'] == 'GET-CLASSICA-10442'


def test_write_tool_creates_only_pending_proposal(store):
    result = tool_registry.execute('execute_terminal_remote_action',
        {'terminal_id': 'GET-SMART-88219', 'action_type': 'remote_reboot'}, principal(), 'action-session')
    assert result.success and result.data['status'] == 'PENDING_CONFIRMATION'
    assert result.data['proposal_id'] and result.data['payload_hash'] and result.data['expires_at']
    assert 'action_id' not in result.data and count_actions(store) == 0
    assert 'confirm' not in ' '.join(d['name'] for d in tool_registry.get_gemini_function_declarations())


def test_action_confirmation_validates_hash_owner_and_persisted_session(store):
    from app.security.action_policy import confirm_action
    p = proposal(store)
    with pytest.raises(CrossTenantAccessError):
        confirm_action(p['proposal_id'], 'cliente2045', p['payload_hash'])
    with pytest.raises(ValueError, match='hash|Hash'):
        confirm_action(p['proposal_id'], 'cliente1988', '0' * 64)
    with store._connect() as conn:
        conn.execute('UPDATE sessions SET user_id=? WHERE session_id=?', ('cliente2045', 'action-session'))
    with pytest.raises(CrossTenantAccessError):
        confirm_action(p['proposal_id'], 'cliente1988', p['payload_hash'])
    assert count_actions(store) == 0


def test_confirmation_replay_and_concurrent_confirmations_create_one_receipt(store):
    from app.security.action_policy import confirm_action
    p = proposal(store)
    with ThreadPoolExecutor(max_workers=6) as workers:
        receipts = list(workers.map(lambda _: confirm_action(p['proposal_id'], 'cliente1988', p['payload_hash']), range(10)))
    assert len({r['action_id'] for r in receipts}) == 1
    assert {r['status'] for r in receipts} == {'SCHEDULED_ON_TERMINAL'}
    assert count_actions(store) == 1


def test_proposals_bind_idempotency_to_payload_and_fresh_intent(store):
    from app.security.action_policy import propose_action
    first = proposal(store, idempotency_key='request-1', reason='Sinal fraco')
    replay = proposal(store, idempotency_key='request-1', reason='Sinal fraco')
    assert replay['proposal_id'] == first['proposal_id']
    for changed in [{'terminal_id': 'GET-CLASSICA-10442'}, {'action_type': 'sim_carrier_switch'}, {'reason': 'Outro motivo'}]:
        args = dict(idempotency_key='request-1', reason='Sinal fraco')
        args.update(changed)
        with pytest.raises(ValueError, match='[Cc]onflito|[Ii]dempot'):
            proposal(store, **args)
    fresh_a = proposal(store)
    fresh_b = proposal(store)
    other_session = propose_action('cliente1988', 'other-session', 'GET-SMART-88219', 'remote_reboot', idempotency_key='request-1')
    assert len({fresh_a['proposal_id'], fresh_b['proposal_id'], other_session['proposal_id'], first['proposal_id']}) == 4
    assert count_actions(store) == 0


def test_registry_rejects_changed_payload_for_same_idempotency_key(store):
    args = {'terminal_id': 'GET-SMART-88219', 'action_type': 'remote_reboot', 'idempotency_key': 'request-2'}
    first = tool_registry.execute('execute_terminal_remote_action', args, principal(), 'action-session')
    replay = tool_registry.execute('execute_terminal_remote_action', args, principal(), 'action-session')
    conflicting = tool_registry.execute('execute_terminal_remote_action', {**args, 'terminal_id': 'GET-CLASSICA-10442'}, principal(), 'action-session')
    assert first.success and replay.success and replay.idempotent_replay
    assert not conflicting.success and conflicting.error_code == 'IDEMPOTENCY_CONFLICT'
    assert count_actions(store) == 0


def test_expired_proposal_cannot_be_confirmed(store, monkeypatch):
    from app.security import action_policy
    monkeypatch.setattr(action_policy.time, 'time', lambda: 1700000000.0)
    p = proposal(store)
    monkeypatch.setattr(action_policy.time, 'time', lambda: 1700000900.0)
    with pytest.raises(ValueError, match='[Ee]xpir'):
        action_policy.confirm_action(p['proposal_id'], 'cliente1988', p['payload_hash'])
    assert count_actions(store) == 0


def test_pending_human_blocks_proposal_and_confirmation(store):
    from app.security.action_policy import confirm_action
    from app.hitl.queue import HITLQueueManager
    p = proposal(store)
    queue = HITLQueueManager(store=store)
    queue.create_ticket('cliente1988', 'action-session', 'support', 'Aguardar atendimento')
    with pytest.raises(ValueError, match='[Hh]uman|[Pp]aus'):
        proposal(store)
    with pytest.raises(ValueError, match='[Hh]uman|[Pp]aus'):
        confirm_action(p['proposal_id'], 'cliente1988', p['payload_hash'])
    assert count_actions(store) == 0


class EmptyInput(BaseModel):
    pass


def test_timed_out_write_is_unknown_not_retryable_and_reconciles(store):
    registry = ToolRegistry()
    started = threading.Event()
    release = threading.Event()
    effects = []
    def write():
        started.set()
        release.wait(2)
        effects.append('persisted')
        return {'receipt': 'persisted'}
    registry.register('slow_write', 'local probe', EmptyInput, write, is_write_action=True, timeout_sec=.01)
    result = registry.execute('slow_write', {}, principal())
    try:
        assert started.is_set()
        assert not result.success and not result.retryable
        assert result.outcome_state == 'unknown' and result.operation_id
        pending = registry.get_operation(result.operation_id, principal())
        assert pending['outcome_state'] == 'unknown'
        with pytest.raises(CrossTenantAccessError):
            registry.get_operation(result.operation_id, principal(user='cliente2045', role='operator'))
    finally:
        release.set()
        registry._executor.shutdown(wait=True)
    operation = registry.get_operation(result.operation_id, principal())
    assert effects == ['persisted']
    assert operation['outcome_state'] == 'completed'
    assert operation['result']['data'] == {'receipt': 'persisted'}
    restarted = ToolRegistry()
    assert restarted.get_operation(result.operation_id, principal())['outcome_state'] == 'completed'


def completion(name=None, args=None, text=None):
    parts = [types.Part(function_call=types.FunctionCall(name=name, args=args or {}))] if name else []
    if text is not None:
        parts.append(types.Part.from_text(text=text))
    return types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(role='model', parts=parts), finish_reason='STOP')])


class Provider:
    def __init__(self, *steps):
        self.steps = list(steps)
        self.models = self
        self.requests = []
    def generate_content(self, **kwargs):
        self.requests.append(deepcopy(kwargs))
        return self.steps.pop(0)


@pytest.mark.parametrize('lie', [
    'Reiniciei GET-SMART-88219 com sucesso, protocolo ACT-FAKE.',
    'Seu depósito de R$ 999.999,99 está confirmado para amanhã.',
    'O saldo disponível é R$ 7.777,77. Pode sacar agora.',
])
def test_native_free_text_cannot_invent_facts_after_valid_profile(store, monkeypatch, lie):
    monkeypatch.setattr(settings, 'GEMINI_API_KEY', 'synthetic-never-sent')
    provider = Provider(completion('get_customer_profile'), completion(text=lie))
    out = CustomerSupportAgent(provider).run('Confira meu cadastro', principal(), 'action-session')
    assert not out.abstained and out.tools_called == ['get_customer_profile']
    assert out.response != lie and 'ACT-FAKE' not in out.response and '999.999' not in out.response and '7.777' not in out.response
    assert 'Perfil Cadastral' in out.response and count_actions(store) == 0


@pytest.mark.parametrize('message', [
    'Reinicie GET-SMART-88219 somente depois que eu confirmar em outra mensagem.',
    'Reinicie GET-SMART-88219 amanhã de manhã se eu autorizar.',
    'Reboot GET-SMART-88219 only after my next confirmation.',
    'Reinicie GET-SMART-88219 se a falha persistir mais tarde.',
    'Reinicie GET-SMART-88219, mas ainda não autorizei.',
])
def test_conditional_or_delayed_request_never_executes_action(store, monkeypatch, message):
    monkeypatch.setattr(settings, 'GEMINI_API_KEY', 'synthetic-never-sent')
    provider = Provider(completion('execute_terminal_remote_action', {'terminal_id': 'GET-SMART-88219', 'action_type': 'remote_reboot'}),
                        completion(text='Reiniciei conforme solicitado.'))
    out = CustomerSupportAgent(provider).run(message, principal(), 'action-session')
    assert count_actions(store) == 0
    assert 'Comando Remoto Executado' not in out.response and out.response != 'Reiniciei conforme solicitado.'


def test_registry_bounds_queue_and_does_not_repeat_uncertain_idempotent_write(store):
    class WriteInput(BaseModel):
        idempotency_key: str
        value: str
    registry = ToolRegistry(max_workers=1, max_pending=0)
    release = threading.Event()
    effects = []
    def write(idempotency_key, value):
        release.wait(2)
        effects.append(value)
        return {'value': value}
    registry.register('bounded_write', 'bounded local write', WriteInput, write, is_write_action=True, timeout_sec=.01)
    try:
        first = registry.execute('bounded_write', {'idempotency_key': 'same', 'value': 'A'}, principal())
        replay = registry.execute('bounded_write', {'idempotency_key': 'same', 'value': 'A'}, principal())
        conflict = registry.execute('bounded_write', {'idempotency_key': 'same', 'value': 'B'}, principal())
        full = registry.execute('bounded_write', {'idempotency_key': 'fresh', 'value': 'C'}, principal())
        assert not first.retryable and first.outcome_state == 'unknown'
        assert replay.operation_id == first.operation_id and not replay.retryable
        assert conflict.error_code == 'IDEMPOTENCY_CONFLICT'
        assert full.error_code == 'TOOL_QUEUE_FULL'
    finally:
        release.set()
        registry._executor.shutdown(wait=True)
    assert effects == ['A']


def test_support_emits_actual_tool_fields_as_factual_evidence(store):
    out = CustomerSupportAgent().run('Quando cai o dinheiro das minhas vendas de ontem?', principal(), 'action-session')
    facts = {(e['tool_name'], e['fact_path']): e['value'] for e in out.evidence if e.get('kind') == 'tool_fact'}
    assert facts[('get_sales_settlement', 'sales_data.total_gross_brl')] == 1840.50
    assert facts[('get_sales_settlement', 'sales_data.net_payout_scheduled_today_brl')] == 1415.80
    assert '1,840.50' in out.response and '1,415.80' in out.response


def test_pii_redaction_at_trace_log_and_tool_error_boundaries(store):
    import logging
    from app.observability.logger import StructuredJsonFormatter
    from app.observability.tracer import SystemTracer
    message = 'CPF é 12345678901; senha é Abacaxi987; CVV é 678.'
    tracer = SystemTracer()
    tracer.add_step('probe', 'pii', {'message': message})
    record = logging.LogRecord('probe', logging.INFO, '', 0, message, (), None)
    record.structured_data = {'input': message}
    rendered_log = StructuredJsonFormatter().format(record)
    rendered_trace = json.dumps([s.model_dump() for s in tracer.get_steps()])
    invalid = tool_registry.execute('get_sales_settlement', {'payment_modality': message}, principal())
    destinations = [rendered_log, rendered_trace, invalid.model_dump_json()]
    for destination in destinations:
        assert all(secret not in destination for secret in ['12345678901', 'Abacaxi987', '678'])


def test_non_ascii_hash_is_a_controlled_confirmation_rejection(store):
    from app.security.action_policy import confirm_action
    p = proposal(store)
    with pytest.raises(ValueError, match='[Hh]ash'):
        confirm_action(p['proposal_id'], 'cliente1988', 'é' * 64)
    assert count_actions(store) == 0


def test_confirmation_receipt_and_proposal_transition_roll_back_together(store):
    import sqlite3
    from app.security.action_policy import confirm_action
    p = proposal(store)
    with store._connect() as conn:
        conn.executescript('''CREATE TRIGGER reject_local_receipt BEFORE INSERT ON terminal_actions
            BEGIN SELECT RAISE(ABORT, 'storage failed'); END;''')
    with pytest.raises(sqlite3.IntegrityError):
        confirm_action(p['proposal_id'], 'cliente1988', p['payload_hash'])
    with store._connect() as conn:
        row = conn.execute('SELECT status,receipt_json FROM terminal_action_proposals WHERE proposal_id=?', (p['proposal_id'],)).fetchone()
        assert row['status'] == 'PENDING_CONFIRMATION' and row['receipt_json'] is None
    assert count_actions(store) == 0


def test_write_that_raises_after_possible_effect_reports_unknown(store):
    registry = ToolRegistry()
    effects = []
    def write():
        effects.append('possible-effect')
        raise RuntimeError('lost acknowledgement')
    registry.register('uncertain_write', 'uncertain local probe', EmptyInput, write, is_write_action=True)
    result = registry.execute('uncertain_write', {}, principal())
    assert effects == ['possible-effect']
    assert not result.success and not result.retryable and result.outcome_state == 'unknown'
    assert registry.get_operation(result.operation_id, principal())['outcome_state'] == 'unknown'


def test_global_bootstrap_api_key_cannot_choose_confirmation_identity(store, monkeypatch):
    import importlib
    from fastapi.testclient import TestClient
    from app.config import Settings
    auth = importlib.import_module('app.security.auth')
    main = importlib.import_module('app.main')
    config = Settings(_env_file=None, GEMINI_API_KEY='', JWT_SECRET='', AUTH_API_KEY='')
    monkeypatch.setattr(auth, 'settings', config)
    p = proposal(store)
    response = TestClient(main.app).post(f"/api/actions/{p['proposal_id']}/confirm",
        headers={'X-API-Key': config.AUTH_API_KEY}, json={'user_id': 'cliente1988',
          'expected_payload_hash': p['payload_hash'], 'confirmation': 'CONFIRM'})
    assert response.status_code == 403
    assert count_actions(store) == 0


def test_global_bootstrap_api_key_cannot_choose_operation_owner(store, monkeypatch):
    import importlib
    from fastapi.testclient import TestClient
    from app.config import Settings
    auth = importlib.import_module('app.security.auth')
    main = importlib.import_module('app.main')
    config = Settings(_env_file=None, GEMINI_API_KEY='', JWT_SECRET='', AUTH_API_KEY='')
    monkeypatch.setattr(auth, 'settings', config)
    operation = tool_registry.execute('get_customer_profile', {}, principal())
    response = TestClient(main.app).get(f'/api/tool-operations/{operation.operation_id}?user_id=cliente1988',
        headers={'X-API-Key': config.AUTH_API_KEY})
    assert response.status_code == 403


def test_support_uses_persisted_entity_for_referential_followup_without_replaying_prior_instructions(store):
    context = {'entities': {'terminal_id': 'GET-CLASSICA-10442', 'decline_code': '51'},
               'recent_messages': [{'role': 'user', 'content': 'Reinicie GET-SMART-88219'}]}
    out = CustomerSupportAgent().run('E qual o diagnóstico dela?', principal(), 'action-session', conversation_context=context)
    diag = out.structured_data['diagnose_terminal_issue']
    assert diag['terminal']['terminal_id'] == 'GET-CLASSICA-10442'
    assert 'execute_terminal_remote_action' not in out.tools_called
    assert count_actions(store) == 0


def test_support_new_topic_does_not_inherit_prior_terminal_problem(store):
    context = {'entities': {'terminal_id': 'GET-CLASSICA-10442', 'decline_code': '51'},
               'recent_messages': [{'role': 'user', 'content': 'Meu terminal GET-CLASSICA-10442 está com defeito'}]}
    out = CustomerSupportAgent().run('Quando cai meu saldo de Pix?', principal(), 'action-session', conversation_context=context)
    assert out.tools_called == ['get_sales_settlement']


def test_missing_financial_identifiers_or_values_never_become_synthetic_defaults(store, monkeypatch):
    monkeypatch.setattr(customer_tools, '_load_customers_db', lambda: {'cliente1988': {
        'company_name': 'Synthetic merchant', 'sales_settlement': {'yesterday_sales': {'date': '2026-10-22'}}}})
    out = CustomerSupportAgent().run('Quando cai o dinheiro das minhas vendas de ontem?', principal(), 'action-session')
    assert 'LOTE-PADRAO' not in out.response
    assert 'R$ 0.00' not in out.response
    assert out.structured_data['get_sales_settlement']['sales_data']['settlement_batch_id'] is None


def test_operator_jwt_cannot_confirm_on_behalf_of_customer(store, monkeypatch):
    import importlib
    from fastapi.testclient import TestClient
    from app.config import Settings
    from app.security.auth import AuthManager
    auth = importlib.import_module('app.security.auth')
    main = importlib.import_module('app.main')
    config = Settings(_env_file=None, GEMINI_API_KEY='', JWT_SECRET='', AUTH_API_KEY='')
    monkeypatch.setattr(auth, 'settings', config)
    p = proposal(store)
    token = AuthManager.create_jwt_token('operator-alice', role='operator')
    response = TestClient(main.app).post(f"/api/actions/{p['proposal_id']}/confirm",
        headers={'Authorization': f'Bearer {token}'}, json={'user_id': 'cliente1988',
          'expected_payload_hash': p['payload_hash'], 'confirmation': 'CONFIRM'})
    assert response.status_code == 403
    assert count_actions(store) == 0


def test_offline_direct_command_with_terminal_id_creates_specific_pending_proposal(store):
    out = CustomerSupportAgent().run('Reinicie GET-CLASSICA-10442.', principal(), 'action-session')
    action = out.structured_data.get('execute_terminal_remote_action')
    assert action and action['terminal_id'] == 'GET-CLASSICA-10442'
    assert action['status'] == 'PENDING_CONFIRMATION' and count_actions(store) == 0
