"""Metamorphic local contract: typed facts fulfill a relation, not tool success."""
import pytest

from app.agents.customer_support_agent import CustomerSupportAgent
from app.schemas import SpecialistOutput, ToolExecutionResult
from app.security.request_units import validate_scope

TARGET = 'GET-SYNTH-WEST-740'
OTHER = 'GET-SYNTH-EAST-741'


def diagnosis(**terminal):
    return {'found': True, 'user_id': 'synthetic', 'terminal': {
        'terminal_id': TARGET, 'model': 'Get Smart', 'firmware_version': '9.4',
        'status': 'ONLINE', 'connection_type': 'WIFI', 'signal_strength': 'strong', **terminal}}


def cover(message, data=None, tool='diagnose_terminal_issue', *, state='completed', context=None):
    result = ToolExecutionResult(tool_name=tool, success=True, data=data or diagnosis(), outcome_state=state)
    out = SpecialistOutput(agent_name='CUSTOMER_SUPPORT_AGENT', response='Verified fields',
        tool_results=[result], structured_data={tool: result.data})
    return CustomerSupportAgent._with_unit_coverage(out, message, validate_scope(message), context)


@pytest.mark.parametrize('field,query', [
    ('firmware_version', 'Check the firmware'), ('status', 'Check the status'),
    ('signal_strength', 'Check the signal'), ('connection_type', 'Check the connection type'),
])
@pytest.mark.parametrize('missing', ['absent', None, '', '   '])
def test_requested_terminal_fact_must_be_useful(field, query, missing):
    message = f'{query} of {TARGET}.'
    assert cover(message).covered_unit_ids == ['u1']
    data = diagnosis()
    if missing == 'absent':
        del data['terminal'][field]
    else:
        data['terminal'][field] = missing
    out = cover(message, data)
    assert not out.covered_unit_ids
    assert out.unresolved_unit_ids == ['u1']
    assert out.abstained and out.answer_status != 'complete'


@pytest.mark.parametrize('message', [
    'Does every debit transaction become Pix?',
    'A transação no débito equivale a Pix?',
    f'Is the firmware of {TARGET} secure?',
    f'Check the battery capacity of {TARGET}.',
    f'Check the encryption certification of {TARGET}.',
    'Does a savings account support debit transactions?',
])
def test_unknown_or_public_predicate_cannot_gain_credit_from_private_pos(message):
    assert not cover(message).covered_unit_ids


def test_wrong_terminal_does_not_fulfill_even_with_requested_field():
    out = cover(f'Check firmware of {TARGET}.', diagnosis(terminal_id=OTHER))
    assert out.unresolved_unit_ids == ['u1']


@pytest.mark.parametrize('state', ['pending', 'unknown', 'failed'])
def test_nonfinal_observations_never_fulfill(state):
    assert not cover(f'Check firmware of {TARGET}.', state=state).covered_unit_ids


def settlement():
    return {'found': True, 'user_id': 'synthetic', 'requested_sales_date': '2026-10-19',
        'data_scope': 'synthetic_historical_snapshot', 'sales_data': {
            'date': '2026-10-19', 'net_payout_scheduled_today_brl': 0,
            'total_gross_brl': 200, 'payout_status': 'scheduled'}}


@pytest.mark.parametrize('mutation', ['wrong_actual_date', 'wrong_requested_date', 'missing_amount', 'live_scope'])
def test_dated_net_amount_requires_actual_date_scope_and_amount(mutation):
    message = 'Check my net payout on 2026-10-19.'
    assert cover(message, settlement(), 'get_sales_settlement').covered_unit_ids == ['u1']
    data = settlement()
    if mutation == 'wrong_actual_date': data['sales_data']['date'] = '2026-10-18'
    if mutation == 'wrong_requested_date': data['requested_sales_date'] = '2026-10-18'
    if mutation == 'missing_amount': data['sales_data']['net_payout_scheduled_today_brl'] = None
    if mutation == 'live_scope': data['data_scope'] = 'unverified_live'
    assert not cover(message, data, 'get_sales_settlement').covered_unit_ids


def test_false_boolean_and_zero_are_facts_rather_than_absence():
    profile = {'found': True, 'user_id': 'synthetic', 'antecipacao_automatica': False}
    assert cover('Check my automatic advance.', profile, 'get_customer_profile').covered_unit_ids == ['u1']
    assert cover('Check my net payout on 2026-10-19.', settlement(), 'get_sales_settlement').covered_unit_ids == ['u1']


@pytest.mark.parametrize('condition', ['if the issuer permits', 'unless the issuer rejects', 'se o fabricante autorizar'])
def test_condition_cannot_be_discharged_by_unconditional_observation(condition):
    message = f'Check firmware of {TARGET} {condition}.'
    out = cover(message)
    assert out.unresolved_unit_ids
    assert not out.covered_unit_ids


def test_pending_proposal_does_not_fulfill_action():
    out = cover(f'Restart {TARGET}.', {'proposal_id': 'PROP-LOCAL', 'status': 'PENDING_CONFIRMATION',
        'terminal_id': TARGET}, 'execute_terminal_remote_action')
    assert not out.covered_unit_ids
    assert out.unresolved_unit_ids == ['u1']


def test_later_irrelevant_result_cannot_replace_exact_target_observation():
    from app.security.fulfillment import evaluate_operational_unit
    exact = ToolExecutionResult(tool_name='diagnose_terminal_issue', success=True, data=diagnosis())
    unrelated = ToolExecutionResult(tool_name='diagnose_terminal_issue', success=True, data=diagnosis(terminal_id=OTHER))
    decision = evaluate_operational_unit(f'Check firmware of {TARGET}.', [exact, unrelated])
    assert decision.fulfilled
    assert decision.required_fact_paths == ('terminal.firmware_version',)
    assert decision.reason.value == 'verified'


@pytest.mark.parametrize('message', [
    f'Check firmware vulnerabilities of {TARGET}.',
    f'Check firmware latency of {TARGET}.',
    f'Check signal coverage across Brazil on {TARGET}.',
    f'Check whether firmware of {TARGET} guarantees settlement.',
])
def test_known_property_token_does_not_authorize_an_unrecognized_relation(message):
    assert not cover(message).covered_unit_ids


@pytest.mark.parametrize('bad', [False, 0, [], {}, 'unknown', 'não informado', 'n/a'])
def test_firmware_observation_must_be_typed_useful_version(bad):
    assert not cover(f'Check firmware of {TARGET}.', diagnosis(firmware_version=bad)).covered_unit_ids


def test_matching_transaction_id_is_required_for_a_decline_read():
    message = f'Check decline code of TXN-SYNTH-740 on {TARGET}.'
    data = diagnosis()
    data['symptom_classified'] = 'decline'
    data['last_declined_transaction'] = {'transaction_id': 'TXN-SYNTH-740', 'decline_code': '51'}
    assert cover(message, data).covered_unit_ids == ['u1']
    data['last_declined_transaction']['transaction_id'] = 'TXN-SYNTH-741'
    assert not cover(message, data).covered_unit_ids


def test_original_authority_binds_shared_terminal_properties():
    message = f'Check status of {TARGET} and firmware.'
    assert cover(message).covered_unit_ids == ['u1', 'u2']
    assert not cover(message, diagnosis(terminal_id=OTHER)).covered_unit_ids


def test_actual_authenticated_tenant_is_required_by_contract():
    from app.security.fulfillment import evaluate_operational_unit
    result = ToolExecutionResult(tool_name='diagnose_terminal_issue', success=True, data=diagnosis())
    assert evaluate_operational_unit(f'Check firmware of {TARGET}.', [result], authenticated_user_id='synthetic').fulfilled
    decision = evaluate_operational_unit(f'Check firmware of {TARGET}.', [result], authenticated_user_id='other')
    assert not decision.fulfilled
    assert decision.reason.value == 'scope_mismatch'


def test_historical_settlement_scope_is_published_as_evidence():
    result = ToolExecutionResult(tool_name='get_sales_settlement', success=True, data=settlement())
    facts = CustomerSupportAgent._result_evidence([result])
    assert any(fact['fact_path'] == 'data_scope' and fact['value'] == 'synthetic_historical_snapshot' for fact in facts)
    assert any(fact['fact_path'] == 'requested_sales_date' and fact['value'] == '2026-10-19' for fact in facts)


@pytest.mark.parametrize('predicate', [
    'envia dados para fora', 'vende dados', 'espiona o cliente', 'altera o conteúdo',
    'transmits customer data', 'sells information', 'spies on customers', 'changes receipts',
    'is current',
])
def test_present_firmware_cannot_prove_an_extra_predicate(predicate):
    assert not cover(f'Check firmware of {TARGET} {predicate}?').covered_unit_ids


@pytest.mark.parametrize('message,tool,data', [
    ('Check my net payout accuracy on 2026-10-19.', 'get_sales_settlement', settlement()),
    ('Check my payout after applying a secret adjustment on 2026-10-19.', 'get_sales_settlement', settlement()),
    ('Check my profile reliability.', 'get_customer_profile', {'found': True, 'company_name': 'Synthetic', 'account_plan': 'Basic', 'terminals_summary': [{'terminal_id': TARGET}]}),
])
def test_other_property_domains_also_reject_new_predicate_residue(message, tool, data):
    assert not cover(message, data, tool).covered_unit_ids


def test_run_missing_firmware_preserves_unknown_in_text_and_pending_state(monkeypatch, tmp_path, isolated_registry):
    import json
    from app.config import settings
    from app.security.auth import SecurityContext
    fixture = tmp_path/'customers.json'
    fixture.write_text(json.dumps({'customers': {'synthetic': {'company_name': 'Synthetic',
        'terminals': [{'terminal_id': TARGET, 'model': 'Get Smart', 'status': 'ONLINE',
                       'connection_type': 'WIFI', 'signal_strength': 'strong'}]}}}))
    monkeypatch.setattr(settings, 'MOCK_CUSTOMERS_PATH', fixture)
    monkeypatch.setattr(settings, 'GEMINI_API_KEY', '')
    out = CustomerSupportAgent().run(f'Check firmware of {TARGET}.', SecurityContext(authenticated_user_id='synthetic'))
    assert 'Firmware:** não informado' in out.response
    assert out.unresolved_unit_ids == ['u1']
    assert out.abstained and out.answer_status == 'partial'


@pytest.mark.parametrize('cause_query', [
    f'Why was TXN-SYNTH-740 declined on {TARGET}?',
    f'Qual o motivo da recusa de TXN-SYNTH-740 no {TARGET}?',
])
@pytest.mark.parametrize('reason', ['Issuer rejected the authorization', None, '', '   '])
def test_decline_cause_requires_cause_field_not_just_code(cause_query, reason):
    data = diagnosis()
    data['symptom_classified'] = 'decline'
    data['last_declined_transaction'] = {'transaction_id': 'TXN-SYNTH-740',
        'decline_code': '51', 'decline_reason': reason}
    out = cover(cause_query, data)
    assert bool(out.covered_unit_ids) is bool(reason and reason.strip())


@pytest.mark.parametrize('query', [
    f'Qual o motivo do status offline de {TARGET}?',
    f'Why is the connection of {TARGET} offline?',
    f'What is the reason for firmware version of {TARGET}?',
])
def test_terminal_property_cannot_explain_an_unsupported_cause(query):
    assert not cover(query).covered_unit_ids


def test_known_profile_and_settlement_fields_do_not_prove_behavior_or_forecast():
    assert not cover('Projete minhas vendas de 2026-10-19.', settlement(), 'get_sales_settlement').covered_unit_ids
    profile = {'found': True, 'company_name': 'Synthetic', 'account_plan': 'Basic', 'terminals_summary': [{'terminal_id': TARGET}]}
    assert not cover('Meu perfil bloqueia pagamentos internacionais?', profile, 'get_customer_profile').covered_unit_ids


@pytest.mark.parametrize('prefix', ['Pode verificar', 'Can you check', 'Could you inspect'])
def test_polite_property_read_is_not_public_capability(prefix):
    message = f'{prefix} firmware do {TARGET}?'
    assert cover(message).covered_unit_ids == ['u1']
    assert not cover(message, diagnosis(firmware_version=None)).covered_unit_ids


def test_known_code_is_not_credit_when_renderer_cannot_publish_transaction():
    data = diagnosis()
    data['symptom_classified'] = 'general'
    data['last_declined_transaction'] = {'transaction_id': 'TXN-SYNTH-740', 'decline_code': '51'}
    assert not cover(f'Check decline code of TXN-SYNTH-740 on {TARGET}.', data).covered_unit_ids


@pytest.fixture
def isolated_registry(monkeypatch, tmp_path):
    import importlib
    from app.memory.store import StateStore
    from app.tools import customer_tools
    from app.security import action_policy
    from app.observability.resilience import breaker_registry
    store = StateStore(tmp_path/'operations.sqlite')
    monkeypatch.setattr(customer_tools, 'state_store', store)
    monkeypatch.setattr(action_policy, 'state_store', store)
    monkeypatch.setattr(importlib.import_module('app.tools.registry'), 'state_store', store)
    breaker_registry.reset_all()
    yield
    breaker_registry.reset_all()


@pytest.mark.parametrize('firmware', ['sdk-7.4', None, '', 'absent'])
def test_native_sdk_loop_uses_same_field_contract(monkeypatch, tmp_path, isolated_registry, firmware):
    import json
    from google.genai import types
    from app.config import settings
    from app.security.auth import SecurityContext
    terminal = diagnosis(firmware_version=firmware)['terminal']
    if firmware == 'absent': terminal.pop('firmware_version')
    fixture = tmp_path/'native_customers.json'
    fixture.write_text(json.dumps({'customers': {'synthetic': {'company_name': 'Synthetic', 'terminals': [terminal]}}}))
    monkeypatch.setattr(settings, 'MOCK_CUSTOMERS_PATH', fixture)
    monkeypatch.setattr(settings, 'GEMINI_API_KEY', 'synthetic-never-sent')

    class Provider:
        def __init__(self):
            self.models = self
            self.turns = 0

        def generate_content(self, **kwargs):
            self.turns += 1
            if self.turns == 1:
                part = types.Part(function_call=types.FunctionCall(name='diagnose_terminal_issue',
                    args={'terminal_id': TARGET, 'symptom': 'general'}, id='local-call'))
            else:
                assert kwargs['contents'][-1].parts[0].function_response.response['success']
                part = types.Part.from_text(text='Model assertion of completion is not authority.')
            return types.GenerateContentResponse(candidates=[types.Candidate(
                content=types.Content(role='model', parts=[part]), finish_reason='STOP')])

    provider = Provider()
    out = CustomerSupportAgent(provider).run(f'Check firmware of {TARGET}.',
        SecurityContext(authenticated_user_id='synthetic'))
    assert provider.turns == 2
    assert out.semantic_mode == 'native_semantic'
    assert bool(out.covered_unit_ids) is (firmware == 'sdk-7.4')
    assert out.abstained is (firmware != 'sdk-7.4')
    assert 'Model assertion' not in out.response


def test_terminal_observation_requires_canonical_identity_even_for_private_default():
    assert not cover('Check my firmware.', diagnosis(terminal_id='opaque-unvalidated-target')).covered_unit_ids


def test_false_hardware_replacement_fact_is_rendered_when_credited():
    data = diagnosis()
    data['requires_hardware_replacement'] = False
    out = cover(f'Check hardware of {TARGET}.', data)
    assert out.covered_unit_ids == ['u1']
    rendered = CustomerSupportAgent()._synthesize_support_response('Check hardware', 'synthetic', {'diagnose_terminal_issue': data})
    assert 'não indica necessidade de substituição' in rendered


def test_requested_decline_code_must_match_observed_code():
    data = diagnosis()
    data['symptom_classified'] = 'decline'
    data['last_declined_transaction'] = {'transaction_id': 'TXN-SYNTH-740', 'decline_code': '51'}
    query = f'Check decline code 51 of TXN-SYNTH-740 on {TARGET}.'
    assert cover(query, data).covered_unit_ids == ['u1']
    data['last_declined_transaction']['decline_code'] = '05'
    assert not cover(query, data).covered_unit_ids


def test_literal_firmware_version_assertion_is_outside_property_read_grammar():
    assert not cover(f'Check firmware version 7.4 of {TARGET}.').covered_unit_ids


@pytest.mark.parametrize('query', [
    'Minha maquininha não conecta, o que faço?',
    'My card machine won’t connect to the internet, what should I do?',
])
@pytest.mark.parametrize('steps', [['Check the network configuration.'], None, [], [''], ['   ']])
def test_connectivity_advice_requires_typed_nonempty_instructions(query, steps):
    data = diagnosis()
    data['recommended_troubleshooting'] = {'connectivity': steps}
    out = cover(query, data)
    assert bool(out.covered_unit_ids) is (steps == ['Check the network configuration.'])


@pytest.mark.parametrize('query', [
    'Minha maquininha não conecta, o que faço se o emissor autorizar?',
    'My card machine won’t connect to the internet, what should I do unless the bank permits?',
    'Minha maquininha não conecta, não reinicie o terminal.',
    'Do not restart my terminal, it will not connect.',
])
def test_advice_never_erases_conditions_or_action_negation(query):
    data = diagnosis()
    data['recommended_troubleshooting'] = {'connectivity': ['Check network.']}
    assert not cover(query, data).covered_unit_ids


def test_connectivity_advice_has_observation_and_recommendation_paths():
    from app.security.fulfillment import evaluate_operational_unit
    data = diagnosis()
    data['recommended_troubleshooting'] = {'connectivity': ['Check network.']}
    result = ToolExecutionResult(tool_name='diagnose_terminal_issue', success=True, data=data)
    decision = evaluate_operational_unit('Minha maquininha não conecta, o que faço?', [result])
    assert decision.fulfilled
    assert 'terminal.connection_type' in decision.required_fact_paths
    assert 'recommended_troubleshooting.connectivity' in decision.required_fact_paths


@pytest.mark.parametrize('query', [
    'Minha maquininha não conecta, o que faço?',
    'My card machine won’t connect to the internet, what should I do?',
])
def test_offline_advice_run_keeps_guidance_read_only(monkeypatch, tmp_path, isolated_registry, query):
    import json
    from app.config import settings
    from app.security.auth import SecurityContext
    fixture = tmp_path/'advice_customers.json'
    fixture.write_text(json.dumps({'customers': {'synthetic': {'company_name': 'Synthetic',
        'terminals': [diagnosis(status='OFFLINE')['terminal']]}}}))
    monkeypatch.setattr(settings, 'MOCK_CUSTOMERS_PATH', fixture)
    monkeypatch.setattr(settings, 'GEMINI_API_KEY', '')
    out = CustomerSupportAgent().run(query, SecurityContext(authenticated_user_id='synthetic'))
    assert out.tools_called == ['diagnose_terminal_issue']
    assert out.covered_unit_ids == ['u1']
    assert not out.abstained
    assert 'Orientações de diagnóstico (não executadas remotamente)' in out.response


@pytest.mark.parametrize('query', [
    f'Verificabilidade do firmware {TARGET}',
    f'Inspectability of firmware {TARGET}',
    f'Inspectionism of firmware {TARGET}',
    f'Confidentiality of firmware {TARGET}',
    f'Consultaridade do firmware {TARGET}',
    f'Diagnosability of firmware {TARGET}',
])
def test_nominalization_is_not_an_authorized_read_verb(query):
    assert not cover(query).covered_unit_ids


@pytest.mark.parametrize('verb', ['Verifique', 'Verificar', 'Confira', 'Consulte', 'Inspecione', 'Check', 'Inspect', 'Examine', 'Confirm'])
def test_finite_read_verb_forms_keep_supported_property_read(verb):
    assert cover(f'{verb} firmware do {TARGET}.').covered_unit_ids == ['u1']


def declined_observation():
    data = diagnosis()
    data['symptom_classified'] = 'decline'
    data['last_declined_transaction'] = {
        'transaction_id': 'TXN-SYNTH-740', 'decline_code': '51',
        'decline_reason': 'Issuer limit was insufficient.', 'action_required': 'Ask buyer to use another card.'}
    return data


@pytest.mark.parametrize('query', [
    'My card machine is showing a transaction decline error.',
    'Our POS displays a declined transaction.',
    'Meu terminal apresenta erro de transação recusada.',
    'Minha maquininha está mostrando uma recusa de pagamento.',
    f'My terminal {TARGET} reports decline error for TXN-SYNTH-740.',
])
@pytest.mark.parametrize('missing', [None, 'decline_code', 'decline_reason', 'action_required'])
def test_decline_incident_needs_diagnosis_and_recommended_action(query, missing):
    data = declined_observation()
    if missing:
        data['last_declined_transaction'].pop(missing)
    out = cover(query, data)
    assert bool(out.covered_unit_ids) is (missing is None)


def test_decline_incident_does_not_substitute_wrong_transaction():
    data = declined_observation()
    assert not cover(f'My terminal {TARGET} reports decline error for TXN-SYNTH-741.', data).covered_unit_ids


@pytest.mark.parametrize('query', [
    'My firmware is showing a transaction decline error.',
    'My terminal predicts a declined transaction.',
    'My card machine is showing a transaction decline error while selling customer data.',
])
def test_unknown_incident_subject_or_predicate_is_not_read_credit(query):
    assert not cover(query, declined_observation()).covered_unit_ids


@pytest.mark.parametrize('missing', [None, 'resolution', 'code', 'name', 'action'])
def test_incident_without_transaction_needs_canonical_catalog_resolution(missing):
    data = diagnosis()
    data['symptom_classified'] = 'decline'
    data['last_declined_transaction'] = None
    data['iso8583_catalog_reference'] = {'code': '51', 'name': 'Insufficient funds', 'action': 'Use another card'}
    data['recommended_troubleshooting'] = {'decline_resolution': 'Código 51 (Insufficient funds): Use another card'}
    if missing == 'resolution': data['recommended_troubleshooting'].pop('decline_resolution')
    elif missing: data['iso8583_catalog_reference'].pop(missing)
    assert bool(cover('My card machine is showing a transaction decline error.', data).covered_unit_ids) is (missing is None)


def test_generic_request_for_decline_code_does_not_fulfill_incident():
    data = diagnosis()
    data['symptom_classified'] = 'decline'
    data['recommended_troubleshooting'] = {'decline_resolution': 'Please provide the decline code.'}
    assert not cover('My card machine is showing a transaction decline error.', data).covered_unit_ids


@pytest.mark.parametrize('query', ['Consulte meu plano, CNPJ.', 'Check my account plan, CNPJ.'])
@pytest.mark.parametrize('missing', [None, 'account_plan', 'cnpj'])
def test_comma_profile_properties_require_every_requested_field(query, missing):
    data = {'found': True, 'account_plan': 'Basic', 'cnpj': 'synthetic-masked'}
    if missing: data.pop(missing)
    assert bool(cover(query, data, 'get_customer_profile').covered_unit_ids) is (missing is None)


def full_profile():
    return {'found': True, 'company_name': 'Synthetic', 'cnpj': 'synthetic-masked',
        'account_plan': 'Basic', 'antecipacao_automatica': False,
        'domicilio_bancario': {'banco': 'Synthetic Bank', 'agencia': '**1', 'conta_corrente': '**2'},
        'terminals_count': 1, 'terminals_summary': [{'terminal_id': TARGET}]}


@pytest.mark.parametrize('query', ['Consulte meu perfil completo, plano.', 'Check my complete profile, account plan.'])
def test_full_profile_is_not_reduced_to_first_explicit_field(query):
    data = full_profile()
    assert cover(query, data, 'get_customer_profile').covered_unit_ids
    data.pop('cnpj')
    assert not cover(query, data, 'get_customer_profile').covered_unit_ids


def test_catalog_fields_cannot_authorize_an_unrelated_resolution():
    data = diagnosis()
    data['symptom_classified'] = 'decline'
    data['iso8583_catalog_reference'] = {'code': '51', 'name': 'Insufficient funds', 'action': 'Use another card'}
    data['recommended_troubleshooting'] = {'decline_resolution': 'Please provide the decline code.'}
    assert not cover('My card machine is showing a transaction decline error.', data).covered_unit_ids


def test_incident_run_diagnoses_and_recommends_without_mutation(monkeypatch, tmp_path, isolated_registry):
    import json
    from app.config import settings
    from app.security.auth import SecurityContext
    fixture = tmp_path/'decline_customers.json'
    tx = declined_observation()['last_declined_transaction']
    tx.update({'terminal_id': TARGET, 'status': 'RECUSADA'})
    fixture.write_text(json.dumps({'customers': {'synthetic': {'company_name': 'Synthetic',
        'terminals': [diagnosis()['terminal']], 'recent_transactions': [tx]}}}))
    monkeypatch.setattr(settings, 'MOCK_CUSTOMERS_PATH', fixture)
    monkeypatch.setattr(settings, 'GEMINI_API_KEY', '')
    out = CustomerSupportAgent().run('My card machine is showing a transaction decline error.',
        SecurityContext(authenticated_user_id='synthetic'))
    assert out.tools_called == ['diagnose_terminal_issue']
    assert out.covered_unit_ids == ['u1']
    assert not out.abstained
    assert 'Issuer limit was insufficient.' in out.response
    assert 'Ask buyer to use another card.' in out.response
