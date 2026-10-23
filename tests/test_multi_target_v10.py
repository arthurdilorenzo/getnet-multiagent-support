"""Literal multi-target additions never transfer authority between clauses."""
import pytest

from app.agents.customer_support_agent import CustomerSupportAgent
from app.schemas import SpecialistOutput, ToolExecutionResult
from app.security.request_constraints import constrain_call, extract_constraints, RequestConstraintError
from app.security.request_units import request_units, validate_scope

WEST = 'GET-CLASSICA-QA-9627'
EAST = 'GET-SMART-QA-9513'


def observation(target, **fields):
    return ToolExecutionResult(tool_name='diagnose_terminal_issue', success=True, data={
        'found': True, 'user_id': 'synthetic', 'terminal': {
            'terminal_id': target, 'connection_type': 'Ethernet', 'signal_strength': '3/4', **fields}})


def coverage(message, results):
    output = SpecialistOutput(agent_name='CUSTOMER_SUPPORT_AGENT', response='facts', tool_results=results)
    return CustomerSupportAgent._with_unit_coverage(output, message, validate_scope(message), None, 'synthetic')


@pytest.mark.parametrize('message', [
    f'Give me the connection of {WEST} plus the signal strength of {EAST}.',
    f'Check connection of {WEST} plus signal of {EAST}.',
    f'Consulte conexão de {WEST} mais sinal de {EAST}.',
    f'Consulte conexão de {WEST} além de sinal de {EAST}.',
    f'Check signal of {EAST} as well as connection of {WEST}.',
])
def test_independent_literal_additive_reads(message):
    units = request_units(message)
    assert len(units) == 2
    assert ''.join(unit['text'] for unit in units) == message
    assert request_units(message) == units
    assert all(message[unit['start']:unit['end']] == unit['text'] for unit in units)
    for unit in units:
        target = extract_constraints(unit['text']).terminal_ids[0]
        assert constrain_call('diagnose_terminal_issue', {'terminal_id': target}, extract_constraints(message),
                              assigned_unit_ids=[unit['unit_id']])['terminal_id'] == target
    output = coverage(message, [observation(WEST), observation(EAST)])
    assert output.covered_unit_ids == ['u1', 'u2']
    assert output.answer_status == 'complete'


def test_scoped_addition_cannot_borrow_other_target():
    message = f'Check connection of {WEST} plus signal of {EAST}.'
    with pytest.raises(RequestConstraintError):
        constrain_call('diagnose_terminal_issue', {'terminal_id': EAST}, extract_constraints(message), assigned_unit_ids=['u1'])


@pytest.mark.parametrize('message', [
    f'Check connection of {WEST} plus signal of {EAST} if certified.',
    f'Do not check connection of {WEST} plus signal of {EAST}.',
    f'Check connection of {WEST} plus signal of {EAST} and guarantee security.',
])
def test_additive_boundary_does_not_erase_conditions_or_negation(message):
    assert not coverage(message, [observation(WEST), observation(EAST)]).covered_unit_ids


def test_missing_field_cannot_be_donated_by_other_target():
    message = f'Check connection of {WEST} plus signal of {EAST}.'
    output = coverage(message, [observation(WEST), observation(EAST, signal_strength=None)])
    assert output.covered_unit_ids == ['u1']
    assert output.unresolved_unit_ids == ['u2']


def test_plus_transaction_pairs_cannot_cross():
    message = f'Check code of TXN-A on {WEST} plus code of TXN-B on {EAST}.'
    with pytest.raises(RequestConstraintError):
        constrain_call('diagnose_terminal_issue', {'terminal_id': WEST, 'transaction_id': 'TXN-B'}, extract_constraints(message))


def test_ambiguous_elliptical_second_target_does_not_get_read_credit():
    message = f'Check connection of {WEST} plus {EAST}.'
    assert not coverage(message, [observation(WEST), observation(EAST)]).covered_unit_ids


@pytest.mark.parametrize('message', [
    'Explain the company Plus Payments.',
    'Explain the plan Mais Vantagens.',
    'Explain A e B Consulting.',
])
def test_connector_inside_proper_name_is_not_an_additive_read(message):
    assert not coverage(message, [observation(WEST)]).covered_unit_ids


@pytest.fixture
def local_tools(monkeypatch, tmp_path):
    import importlib
    import json
    from app.config import settings
    from app.memory.store import StateStore
    from app.tools import customer_tools
    from app.security import action_policy
    from app.observability.resilience import breaker_registry
    store = StateStore(tmp_path/'operations.sqlite')
    monkeypatch.setattr(customer_tools, 'state_store', store)
    monkeypatch.setattr(action_policy, 'state_store', store)
    monkeypatch.setattr(importlib.import_module('app.tools.registry'), 'state_store', store)
    def terminal(target, **fields):
        return {'terminal_id': target, 'model': 'Get Smart', 'status': 'ONLINE',
                'connection_type': 'Ethernet', 'signal_strength': '3/4', **fields}
    path = tmp_path/'customers.json'
    path.write_text(json.dumps({'customers': {
        'synthetic': {'company_name': 'Synthetic', 'terminals': [terminal(WEST), terminal(EAST, connection_type='WIFI')]},
        'foreign': {'company_name': 'Foreign', 'terminals': [terminal('GET-FOREIGN-V10-783', signal_strength='PRIVATE')]}}}))
    monkeypatch.setattr(settings, 'MOCK_CUSTOMERS_PATH', path)
    monkeypatch.setattr(settings, 'GEMINI_API_KEY', '')
    breaker_registry.reset_all()
    yield path
    breaker_registry.reset_all()


class SelectedToolsProvider:
    """SDK-shaped transport only; product owns registry, scope, facts and rendering."""
    def __init__(self, targets):
        self.models = self
        self.turns = 0
        self.targets = targets

    def generate_content(self, **kwargs):
        from google.genai import types
        self.turns += 1
        if self.turns == 1:
            parts = [types.Part(function_call=types.FunctionCall(name='diagnose_terminal_issue',
                args={'terminal_id': target, 'symptom': 'general'}, id=f'call-{index}'))
                for index, target in enumerate(self.targets)]
        else:
            parts = [types.Part.from_text(text='Untrusted model narrative: every signal is full.')]
        return types.GenerateContentResponse(candidates=[types.Candidate(
            content=types.Content(role='model', parts=parts), finish_reason='STOP')])


@pytest.mark.parametrize('native', [False, True])
@pytest.mark.parametrize('missing', [False, True])
def test_real_support_path_returns_each_targets_typed_facts(local_tools, monkeypatch, native, missing):
    import json
    from app.config import settings
    from app.security.auth import SecurityContext
    if missing:
        data = json.loads(local_tools.read_text())
        data['customers']['synthetic']['terminals'][1]['signal_strength'] = None
        local_tools.write_text(json.dumps(data))
    if native:
        monkeypatch.setattr(settings, 'GEMINI_API_KEY', 'synthetic-never-sent')
    provider = SelectedToolsProvider([WEST, EAST]) if native else None
    message = f'Give me the connection of {WEST} plus the signal strength of {EAST}.'
    output = CustomerSupportAgent(provider).run(message, SecurityContext(authenticated_user_id='synthetic'))
    assert [r.data['terminal']['terminal_id'] for r in output.tool_results] == [WEST, EAST]
    assert all(r.success for r in output.tool_results)
    assert output.covered_unit_ids == (['u1'] if missing else ['u1', 'u2'])
    assert output.answer_status == ('partial' if missing else 'complete')
    assert 'Ethernet' in output.response
    assert f'`{WEST}`' in output.response and f'`{EAST}`' in output.response
    if not missing:
        assert '3/4' in output.response
    assert 'every signal is full' not in output.response
    if native:
        assert provider.turns == 2


@pytest.mark.parametrize('target', ['GET-FOREIGN-V10-783', 'GET-UNREQUESTED-V10-345'])
def test_native_unrequested_target_is_denied_without_data(local_tools, monkeypatch, target):
    from app.config import settings
    from app.security.auth import SecurityContext
    monkeypatch.setattr(settings, 'GEMINI_API_KEY', 'synthetic-never-sent')
    message = f'Check connection of {WEST} plus signal of {EAST}.'
    output = CustomerSupportAgent(SelectedToolsProvider([WEST, target])).run(
        message, SecurityContext(authenticated_user_id='synthetic'))
    assert output.tool_results[0].success
    assert not output.tool_results[1].success
    assert output.tool_results[1].data is None
    assert output.tool_results[1].error_code == 'REQUEST_TARGET_MISMATCH'
    assert output.covered_unit_ids == ['u1']
    assert 'PRIVATE' not in output.response


def test_explicitly_requested_foreign_terminal_still_cannot_cross_tenant(local_tools, monkeypatch):
    from app.config import settings
    from app.security.auth import SecurityContext
    monkeypatch.setattr(settings, 'GEMINI_API_KEY', 'synthetic-never-sent')
    target = 'GET-FOREIGN-V10-783'
    message = f'Check connection of {WEST} plus signal of {target}.'
    output = CustomerSupportAgent(SelectedToolsProvider([WEST, target])).run(
        message, SecurityContext(authenticated_user_id='synthetic'))
    assert output.tool_results[0].success
    # The read completed with an explicit not-found result inside this tenant.
    assert output.tool_results[1].data['found'] is False
    assert 'terminal' not in output.tool_results[1].data
    assert output.covered_unit_ids == ['u1']
    assert output.unresolved_unit_ids == ['u2']
    assert 'PRIVATE' not in output.response


@pytest.mark.parametrize('message', [
    f'Check firmware plus connection of {WEST}.',
    f'Check connection of {WEST} plus signal.',
    f'Consulte conexão de {WEST} além de sinal.',
])
def test_additive_ellipsis_uses_only_single_proven_subject(message):
    results = [observation(WEST, firmware_version='v10.2')]
    assert coverage(message, results).covered_unit_ids == ['u1', 'u2']


def test_third_elliptical_property_cannot_choose_one_of_two_targets():
    message = f'Check connection of {WEST} plus signal of {EAST} plus firmware.'
    output = coverage(message, [observation(WEST, firmware_version='v10.2'), observation(EAST, firmware_version='v10.3')])
    assert output.covered_unit_ids == ['u1', 'u2']
    assert output.unresolved_unit_ids == ['u3']


@pytest.mark.parametrize('message', ['Explain Plus Payments.', 'Explain Mais Vantagens.', 'Explain Além de Tudo.'])
def test_new_additive_markers_do_not_split_unrecognized_proper_names(message):
    assert len(request_units(message)) == 1


@pytest.mark.parametrize('message', [
    f'Check connection of {WEST} plus signal of {EAST} unless the bank permits.',
    f'Consulte conexão de {WEST} além de sinal de {EAST} desde que certificado.',
    f'Check connection of {WEST} as well as signal of {EAST} only when certified.',
])
def test_conditional_additions_keep_literal_scope_and_deny_credit(message):
    assert ''.join(unit['text'] for unit in request_units(message)) == message
    assert not coverage(message, [observation(WEST), observation(EAST)]).covered_unit_ids


def test_additive_veto_does_not_authorize_remote_action_or_hide_negative_scope():
    message = f'Check connection of {WEST} plus do not reboot {EAST}.'
    assert not CustomerSupportAgent._authorized_remote_action(message, 'remote_reboot', EAST)
    assert not coverage(message, [observation(WEST), observation(EAST)]).covered_unit_ids
