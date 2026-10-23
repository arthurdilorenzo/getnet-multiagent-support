"""CNPJ redaction by explicit semantics, across real local egress boundaries."""
import json
import logging
from copy import deepcopy

import pytest
from google.genai import types

from app.security.pii import PIIRedactor


RAW = '11222333000181'  # Synthetic fixture, no real customer identity.
MASKED = '11.***.***/0001-**'


@pytest.mark.parametrize('value', [RAW, '11.222.333/0001-81', 11222333000181])
def test_explicit_cnpj_masks_raw_formatted_and_numeric(value):
    assert PIIRedactor.mask_cnpj(value) == MASKED


def test_cnpj_leading_zero_is_preserved_for_string_and_numeric_loss_fails_closed():
    assert PIIRedactor.mask_cnpj('01222333000181') == '01.***.***/0001-**'
    # JSON numbers cannot retain a leading zero; do not invent a document.
    assert PIIRedactor.mask_structure({'cnpj': 1222333000181}) == {'cnpj': '[REDACTED]'}
    assert PIIRedactor.mask_structure({'cnpj': '[REDACTED]'}) == {'cnpj': '[REDACTED]'}


@pytest.mark.parametrize('label', [
    'CNPJ: ', 'cnpj = ', 'Meu CNPJ é ', 'CNPJ eh: ', 'CNPJ is ',
    'CNPJ (Mascarado): ', '**CNPJ**: ', '"cnpj": "',
    'Company tax ID: ', 'Brazilian tax ID = ', 'Business tax number is ',
])
def test_labeled_text_masks_cnpj_without_changing_surrounding_text(label):
    assert PIIRedactor.mask_text(label + RAW) == label + MASKED


@pytest.mark.parametrize('key', [
    'cnpj', 'CNPJ', 'company_cnpj', 'cnpj_number', 'numero_cnpj',
    'Brazilian Tax ID', 'business-tax-id', 'corporate_tax_id', 'tax_id',
])
@pytest.mark.parametrize('value', [RAW, 11222333000181])
def test_recognized_cnpj_fields_mask_string_and_numeric_values(key, value):
    assert PIIRedactor.mask_structure({key: value}) == {key: MASKED}


def test_nested_cnpj_field_keeps_context_for_sequences_and_is_idempotent():
    payload = {'profile': [{'cnpj': [RAW, 11222333000181, None]}],
               'note': 'CNPJ (Mascarado): ' + RAW}
    expected = {'profile': [{'cnpj': [MASKED, MASKED, None]}],
                'note': 'CNPJ (Mascarado): ' + MASKED}
    once = PIIRedactor.mask_structure(payload)
    assert once == expected
    assert PIIRedactor.mask_structure(once) == expected
    assert payload['profile'][0]['cnpj'][0] == RAW


@pytest.mark.parametrize('field', ['idempotency_key', 'session_id', 'terminal_id',
                                    'proposal_id', 'transaction_id', 'payload_hash'])
def test_digit_count_alone_never_changes_opaque_identifiers(field):
    for identifier in [RAW, '41111111111111', 'GET-' + RAW, 'PROP-' + RAW + '-AA', 'a' + RAW + 'f']:
        assert PIIRedactor.mask_structure({field: identifier}) == {field: identifier}
        assert PIIRedactor.mask_text(field + ': ' + identifier) == field + ': ' + identifier
    assert PIIRedactor.mask_text('CNPJ: GET-' + RAW) == 'CNPJ: GET-' + RAW
    assert PIIRedactor.mask_text('CNPJ: ' + RAW + '0') == 'CNPJ: ' + RAW + '0'


def test_cnpj_redaction_preserves_existing_cpf_pan_and_secret_protection():
    message = 'CPF: 12345678901; card 4111 1111 1111 1111; password is segredo; CNPJ: ' + RAW
    masked = PIIRedactor.mask_text(message)
    assert all(secret not in masked for secret in ['12345678901', '4111 1111 1111 1111', 'segredo', RAW])
    assert PIIRedactor.mask_text(masked) == masked


@pytest.mark.parametrize('cnpj', [RAW, 11222333000181, '11.222.333/0001-81'])
def test_profile_cnpj_is_masked_before_provider_memory_trace_log_and_response(monkeypatch, tmp_path, cnpj):
    from app.agents.customer_support_agent import CustomerSupportAgent
    from app.agents.function_calling import initial_contents
    from app.config import settings
    from app.memory.store import StateStore
    from app.observability.logger import StructuredJsonFormatter
    from app.observability.tracer import SystemTracer
    from app.security.auth import SecurityContext
    from app.tools import customer_tools
    import app.tools.registry as registry_module

    store = StateStore(tmp_path / 'pii.db')
    monkeypatch.setattr(registry_module, 'state_store', store)
    monkeypatch.setattr(customer_tools, '_load_customers_db', lambda: {
        'synthetic-owner': {'company_name': 'Synthetic', 'cnpj': cnpj,
                            'terminals': [], 'domicilio_bancario': {}}})
    monkeypatch.setattr(settings, 'GEMINI_API_KEY', '')
    principal = SecurityContext(authenticated_user_id='synthetic-owner', role='merchant',
        auth_method='jwt_bearer', scopes=['customer:read_own', 'chat:write'])
    message = 'Confira meu cadastro'
    tracer = SystemTracer(session_id='pii-v8', user_id='synthetic-owner')

    class ProviderTransport:
        """Only the external SDK transport is replaced; all tool/render code is real."""
        def __init__(self):
            self.models = self
            self.requests = []

        def generate_content(self, **kwargs):
            self.requests.append(deepcopy(kwargs))
            if len(self.requests) == 1:
                part = types.Part(function_call=types.FunctionCall(name='get_customer_profile', args={}))
            else:
                part = types.Part.from_text(text='Perfil consultado.')
            return types.GenerateContentResponse(candidates=[types.Candidate(
                content=types.Content(role='model', parts=[part]), finish_reason='STOP')])

    provider = ProviderTransport()
    agent = CustomerSupportAgent(provider)
    # Exercise the native loop directly with an injected fake transport and empty key.
    output = agent._run_with_gemini(message, principal, 'pii-v8', None, tracer, message)
    assert output.tools_called == ['get_customer_profile']
    assert output.tool_results[0].data['cnpj'] == MASKED
    assert MASKED in output.response
    assert len(provider.requests) == 2
    provider_payload = json.dumps([c.model_dump(mode='json')
        for c in provider.requests[1]['contents']], ensure_ascii=False)
    assert MASKED in provider_payload

    store.append_message('pii-v8', 'synthetic-owner', 'assistant', output.response)
    store.append_message('pii-v8', 'synthetic-owner', 'user', 'CNPJ: ' + RAW)
    context = store.get_conversation_context('pii-v8', 'synthetic-owner')
    tracer.add_step('probe', 'profile', {'profile': {'cnpj': cnpj}})
    record = logging.LogRecord('probe', logging.INFO, '', 0, 'CNPJ: ' + RAW, (), None)
    record.structured_data = {'profile': {'cnpj': cnpj}}
    destinations = [provider_payload, output.model_dump_json(), json.dumps(context),
        str(initial_contents('CNPJ: ' + RAW, context)),
        json.dumps([s.model_dump() for s in tracer.get_steps()]),
        StructuredJsonFormatter().format(record)]
    for destination in destinations:
        assert RAW not in destination and '11.222.333/0001-81' not in destination
