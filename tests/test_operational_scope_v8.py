"""Canonical per-unit scope and shared-subject operational read contracts."""
import json

import pytest

from app.agents.customer_support_agent import CustomerSupportAgent
from app.config import settings
from app.schemas import SpecialistOutput, ToolExecutionResult
from app.security.auth import SecurityContext
from app.security.request_constraints import constrain_call, extract_constraints, RequestConstraintError
from app.security.request_units import request_units, validate_scope

WEST = 'GET-SYNTH-VALLEY-381'
EAST = 'GET-SYNTH-MESA-672'


def observation(target=WEST, **fields):
    return ToolExecutionResult(tool_name='diagnose_terminal_issue', success=True, data={
        'found': True, 'user_id': 'synthetic', 'terminal': {
            'terminal_id': target, 'model': 'Get Smart', 'status': 'ONLINE',
            'connection_type': 'WIFI', 'firmware_version': 'v8.2', **fields}})


def coverage(message, results, scope=None):
    output = SpecialistOutput(agent_name='CUSTOMER_SUPPORT_AGENT', response='facts', tool_results=results,
        structured_data={'observations': [r.data for r in results]})
    return CustomerSupportAgent._with_unit_coverage(output, message, validate_scope(message, scope), None, 'synthetic')


@pytest.mark.parametrize('message', [
    f'Consulte o status e a conexão de {WEST}.',
    f'Check status and connection of {WEST}.',
    f'Inspecione firmware e sinal do terminal {WEST}.',
    f'Inspect firmware and signal of device {WEST}.',
    f'Check firmware of {WEST} and status.',
])
def test_shared_subject_literal_units_are_covered_by_matching_fields(message):
    output = coverage(message, [observation(signal_strength='strong')])
    assert output.covered_unit_ids == [unit['unit_id'] for unit in request_units(message)]
    assert output.answer_status == 'complete'


@pytest.mark.parametrize('missing', [None, '', '   '])
def test_shared_subject_does_not_cover_missing_field(missing):
    message = f'Check status and connection of {WEST}.'
    output = coverage(message, [observation(status=missing)])
    assert output.covered_unit_ids == ['u2']
    assert output.unresolved_unit_ids == ['u1']


@pytest.mark.parametrize('message', [
    f'Check status and connection of {WEST} if certified.',
    f'Não consulte status e conexão de {WEST}.',
    f'Check status and connection of {WEST} means secure payments.',
    f'Check status and firmware security of {WEST}.',
])
def test_unknown_relation_or_condition_cannot_authorize_shared_subject(message):
    assert not coverage(message, [observation()]).covered_unit_ids


def test_no_cross_target_field_donation():
    message = f'Check firmware of {WEST} and connection of {EAST}.'
    output = coverage(message, [observation(WEST), observation(EAST, connection_type=None)])
    assert output.covered_unit_ids == ['u1']
    assert output.unresolved_unit_ids == ['u2']


@pytest.fixture
def local_tools(monkeypatch, tmp_path):
    import importlib
    from app.memory.store import StateStore
    from app.tools import customer_tools
    from app.security import action_policy
    from app.observability.resilience import breaker_registry
    store = StateStore(tmp_path/'operations.sqlite')
    monkeypatch.setattr(customer_tools, 'state_store', store)
    monkeypatch.setattr(action_policy, 'state_store', store)
    monkeypatch.setattr(importlib.import_module('app.tools.registry'), 'state_store', store)
    path = tmp_path/'customers.json'
    path.write_text(json.dumps({'customers': {'synthetic': {'company_name': 'Synthetic',
        'terminals': [observation(WEST).data['terminal'], observation(EAST).data['terminal']]}}}))
    monkeypatch.setattr(settings, 'MOCK_CUSTOMERS_PATH', path)
    monkeypatch.setattr(settings, 'GEMINI_API_KEY', '')
    breaker_registry.reset_all()
    yield
    breaker_registry.reset_all()


@pytest.mark.parametrize('message', [
    f'Check firmware of {WEST} and battery of {EAST}.',
    f'Consulte firmware de {WEST} e bateria de {EAST}.',
    f'Check firmware of {WEST}; check battery of {EAST}.',
])
def test_offline_multitarget_keeps_available_firmware_partial(local_tools, message):
    output = CustomerSupportAgent().run(message, SecurityContext(authenticated_user_id='synthetic'))
    assert len(output.tool_results) == 2
    assert all(result.success for result in output.tool_results)
    assert [result.data['terminal']['terminal_id'] for result in output.tool_results] == [WEST, EAST]
    assert output.covered_unit_ids == ['u1']
    assert output.unresolved_unit_ids == ['u2']
    assert output.answer_status == 'partial'
    assert 'v8.2' in output.response
    for target in (WEST, EAST):
        assert f'`{target}`' in output.response
    assert [fact['value'] for fact in output.evidence if fact.get('fact_path') == 'terminal.terminal_id'] == [WEST, EAST]


def test_offline_scoped_read_cannot_query_other_assigned_target(local_tools):
    message = f'Check firmware of {WEST} and connection of {EAST}.'
    output = CustomerSupportAgent().run(message, SecurityContext(authenticated_user_id='synthetic'),
        delegation_scope={'assigned_unit_ids': ['u1'], 'other_unit_ids': ['u2']})
    assert len(output.tool_results) == 1
    assert output.tool_results[0].data['terminal']['terminal_id'] == WEST


def test_explicit_target_requires_original_scoped_unit():
    original = f'Check firmware of {WEST} and connection of {EAST}.'
    with pytest.raises(RequestConstraintError):
        constrain_call('diagnose_terminal_issue', {'terminal_id': EAST}, extract_constraints(original),
            assigned_unit_ids=['u1'])


def test_multitarget_transaction_pairs_cannot_cross():
    original = f'Check decline code of TXN-SYNTH-A on {WEST} and decline code of TXN-SYNTH-B on {EAST}.'
    with pytest.raises(RequestConstraintError):
        constrain_call('diagnose_terminal_issue', {'terminal_id': WEST, 'transaction_id': 'TXN-SYNTH-B'}, extract_constraints(original))


@pytest.mark.parametrize('tool,args,message', [
    ('diagnose_terminal_issue', {}, f'Check firmware of {WEST} and {EAST}.'),
    ('diagnose_terminal_issue', {'terminal_id': WEST}, f'Check TXN-SYNTH-A and TXN-SYNTH-B on {WEST}.'),
    ('get_sales_settlement', {}, 'Check my sales on 2026-10-13 and 2026-10-14.'),
])
def test_ambiguous_target_transaction_or_date_stays_rejected(tool, args, message):
    with pytest.raises(RequestConstraintError):
        constrain_call(tool, args, extract_constraints(message))


@pytest.mark.parametrize('scoped', [False, True])
def test_native_model_selects_each_target_and_local_scope_gates_execution(monkeypatch, local_tools, scoped):
    from google.genai import types
    monkeypatch.setattr(settings, 'GEMINI_API_KEY', 'synthetic-never-sent')

    class Provider:
        def __init__(self):
            self.models = self
            self.turns = 0

        def generate_content(self, **kwargs):
            self.turns += 1
            if self.turns == 1:
                parts = [types.Part(function_call=types.FunctionCall(name='diagnose_terminal_issue',
                    args={'terminal_id': target, 'symptom': 'general'}, id=f'call-{index}'))
                    for index, target in enumerate((WEST, EAST))]
            else:
                responses = [part.function_response.response for part in kwargs['contents'][-1].parts]
                assert responses[0]['success']
                assert responses[1]['success'] is not scoped
                parts = [types.Part.from_text(text='Untrusted assertion that the battery is full.')]
            return types.GenerateContentResponse(candidates=[types.Candidate(
                content=types.Content(role='model', parts=parts), finish_reason='STOP')])

    message = f'Check firmware of {WEST} and battery of {EAST}.'
    provider = Provider()
    scope = {'assigned_unit_ids': ['u1'], 'other_unit_ids': ['u2']} if scoped else None
    output = CustomerSupportAgent(provider).run(message, SecurityContext(authenticated_user_id='synthetic'),
        delegation_scope=scope)
    assert provider.turns == 2
    assert output.semantic_mode == 'native_semantic'
    assert output.covered_unit_ids == ['u1']
    assert output.tool_results[0].data['terminal']['terminal_id'] == WEST
    if scoped:
        assert output.tool_results[1].error_code == 'REQUEST_TARGET_MISMATCH'
        assert output.tool_results[1].data is None
    else:
        assert output.tool_results[1].data['terminal']['terminal_id'] == EAST
        assert output.unresolved_unit_ids == ['u2']
    assert 'battery is full' not in output.response


@pytest.mark.parametrize('field', ['status', 'connection_type'])
def test_shared_subject_cannot_combine_fields_from_different_observations(field):
    message = f'Check status, connection of {WEST} and firmware of {WEST}.'
    first = observation(**{field: None})
    other = observation(**({'connection_type': None} if field == 'status' else {'status': None}))
    output = coverage(message, [first, other])
    assert 'u1' not in output.covered_unit_ids


def test_shared_subject_needs_literal_canonical_unit():
    from app.security.fulfillment import evaluate_operational_unit
    original = f'Check firmware and connection of {WEST}.'
    assert not evaluate_operational_unit('Check status', [observation()], original=original).fulfilled


@pytest.mark.parametrize('message', [
    f'Check firmware of {WEST} and status of {EAST}.',
    f'Check firmware of {EAST} and status of {WEST}.',
])
def test_ordered_multi_target_fields_do_not_move_between_units(message):
    output = coverage(message, [observation(WEST, firmware_version=None), observation(EAST, status=None)])
    expected = [] if message.startswith(f'Check firmware of {WEST}') else ['u1', 'u2']
    assert output.covered_unit_ids == expected


def test_explicit_transaction_pair_is_kept_on_selected_terminal():
    message = f'Check decline code of TXN-SYNTH-A on {WEST} and decline code of TXN-SYNTH-B on {EAST}.'
    result = constrain_call('diagnose_terminal_issue', {'terminal_id': WEST}, extract_constraints(message))
    assert result['transaction_id'] == 'TXN-SYNTH-A'
    assert result['terminal_id'] == WEST


def test_scoped_read_cannot_take_transaction_from_unassigned_unit():
    message = f'Check firmware of {WEST} and decline code of TXN-SYNTH-A on {WEST}.'
    with pytest.raises(RequestConstraintError):
        constrain_call('diagnose_terminal_issue', {'terminal_id': WEST, 'transaction_id': 'TXN-SYNTH-A'},
            extract_constraints(message), assigned_unit_ids=['u1'])
