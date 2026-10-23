"""Original request authority, exercised through real handlers and SDK transport fake."""
import pytest
from google.genai import types
from app.agents.customer_support_agent import CustomerSupportAgent
from app.config import settings
from app.security.auth import SecurityContext
from app.tools import customer_tools, search_tools

CTX = SecurityContext(authenticated_user_id='cliente1988')

@pytest.mark.parametrize('date_text', ['5 de outubro de 2026', 'October 5, 2026', '5 October 2026'])
def test_written_sales_date_never_substitutes_snapshot(date_text):
    out = CustomerSupportAgent().run(f'Consulte as vendas de {date_text}.', CTX)
    assert out.structured_data['get_sales_settlement']['found'] is False
    assert out.structured_data['get_sales_settlement']['requested_sales_date'] == '2026-10-05'
    assert '1.415' not in out.response


def test_ambiguous_numeric_date_never_selects_snapshot():
    out = CustomerSupportAgent().run('Consulte as vendas de 05/10/2026', CTX)
    assert out.abstained
    assert not out.structured_data


def test_same_terminal_uses_persisted_entity_pt():
    out = CustomerSupportAgent().run('Cheque o firmware do mesmo terminal.', CTX,
        conversation_context={'entities': {'terminal_id': 'GET-CLASSICA-10442'}})
    assert out.structured_data['diagnose_terminal_issue']['terminal']['terminal_id'] == 'GET-CLASSICA-10442'


def test_multisegment_terminal_not_replaced_by_available_machine():
    out = CustomerSupportAgent().run('Cheque o terminal GET-SMART-WEST-77551.', CTX)
    data = out.structured_data['diagnose_terminal_issue']
    assert data['found'] is False
    assert data['requested_terminal_id'] == 'GET-SMART-WEST-77551'


def test_transaction_is_exact_in_handler_and_schema():
    assert 'transaction_id' in customer_tools.DiagnoseTerminalIssueInput.model_fields
    result = customer_tools.diagnose_terminal_issue('cliente1988', transaction_id='TXN-DOES-NOT-EXIST')
    assert result['found'] is False
    assert not result.get('last_declined_transaction')


def test_sdk_cannot_change_explicit_terminal(monkeypatch):
    monkeypatch.setattr(settings, 'GEMINI_API_KEY', 'synthetic-not-sent')
    class Provider:
        models = None
        def __init__(self):
            self.models = self
            self.turn = 0
        def generate_content(self, **kwargs):
            self.turn += 1
            part = (types.Part(function_call=types.FunctionCall(name='diagnose_terminal_issue', args={'terminal_id': 'GET-SMART-88219'}))
                    if self.turn == 1 else types.Part.from_text(text='Concluído'))
            return types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(role='model', parts=[part]))])
    out = CustomerSupportAgent(Provider()).run('Cheque o terminal GET-SMART-88219', CTX,
        authorization_message='Cheque somente GET-UNKNOWN-77551.')
    assert out.abstained
    assert not out.structured_data
    assert out.tool_results[0].error_code == 'REQUEST_TARGET_MISMATCH'


@pytest.mark.parametrize('payload', [
    {'amount': 1, 'base': 'USD', 'date': '2026-10-22', 'rates': {'BRL': 7.31}},
    {'amount': 2, 'base': 'EUR', 'date': '2026-10-22', 'rates': {'BRL': 7.31}},
    {'base': 'EUR', 'date': '2026-10-22', 'rates': {'BRL': 7.31}},
    {'amount': True, 'base': 'EUR', 'date': '2026-10-22', 'rates': {'BRL': 7.31}},
])
def test_fx_secondary_requires_matching_base_and_unit(monkeypatch, payload):
    class Response:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return payload
    class Client:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def get(self, url):
            if 'awesomeapi' in url: raise RuntimeError('primary unavailable')
            return Response()
    monkeypatch.setattr(search_tools.httpx, 'Client', Client)
    monkeypatch.setattr(search_tools, 'call_with_resilience', lambda name, fn: fn())
    with pytest.raises(ValueError):
        search_tools.search_currency_rate('EUR', 'BRL')


def test_constraints_bind_copy_and_are_immutable():
    from dataclasses import FrozenInstanceError
    from app.security.request_constraints import extract_constraints, constrain_call
    contract = extract_constraints('Confira GET-SMART-WEST-77551 e TXN-ABC-72 em 21 de outubro de 2026')
    args = {'symptom': 'decline'}
    bound = constrain_call('diagnose_terminal_issue', args, contract)
    assert args == {'symptom': 'decline'}
    assert bound['terminal_id'] == 'GET-SMART-WEST-77551'
    assert bound['transaction_id'] == 'TXN-ABC-72'
    assert contract.sales_dates == ('2026-10-21',)
    with pytest.raises(FrozenInstanceError):
        contract.original_message = 'replacement'


@pytest.mark.parametrize('value', ['31 de fevereiro de 2026', 'October 5', '2026-99-01'])
def test_invalid_or_partial_date_requests_clarification(value):
    out = CustomerSupportAgent().run('Consulte as vendas de ' + value, CTX)
    assert out.abstained
    assert not out.structured_data


@pytest.mark.parametrize('field,tool,original,args', [
    ('REQUEST_DATE_MISMATCH', 'get_sales_settlement', 'Vendas em October 5, 2026', {'sales_date': '2026-10-21'}),
    ('REQUEST_TRANSACTION_MISMATCH', 'diagnose_terminal_issue', 'Verifique TXN-ABC-72', {'transaction_id':'TXN-ABC-73'}),
    ('REQUEST_NEEDS_CLARIFICATION', 'diagnose_terminal_issue', 'Verifique GET-A-11 e GET-B-22', {}),
])
def test_constraint_rejects_wrong_or_ambiguous_values(field, tool, original, args):
    from app.security.request_constraints import extract_constraints, constrain_call, RequestConstraintError
    with pytest.raises(RequestConstraintError) as exc:
        constrain_call(tool, args, extract_constraints(original))
    assert exc.value.code == field


def test_same_terminal_without_persisted_entity_requires_clarification():
    out = CustomerSupportAgent().run('Check the same terminal firmware.', CTX)
    assert out.abstained
    assert not out.structured_data


@pytest.mark.parametrize('phrase', ['Cheque o mesmo terminal.', 'Check the same terminal.'])
def test_persisted_reference_and_exact_transaction(phrase):
    out = CustomerSupportAgent().run(phrase, CTX,
        conversation_context={'entities': {'terminal_id':'GET-CLASSICA-10442'}})
    assert out.structured_data['diagnose_terminal_issue']['terminal']['terminal_id'] == 'GET-CLASSICA-10442'
    result = customer_tools.diagnose_terminal_issue('cliente1988', transaction_id='TXN-881915', symptom='decline')
    assert result['last_declined_transaction']['transaction_id'] == 'TXN-881915'
    assert all(t['transaction_id'] == 'TXN-881915' for t in result['all_declined_transactions'])
    assert result['terminal']['terminal_id'] == 'GET-CLASSICA-10442'


def test_explicit_transaction_reaches_handler_without_extra_keywords():
    out = CustomerSupportAgent().run('TXN-881915?', CTX)
    assert out.structured_data['diagnose_terminal_issue']['last_declined_transaction']['transaction_id'] == 'TXN-881915'


def test_primary_wrong_identity_cannot_be_relabelled(monkeypatch):
    payload = {'EURBRL': {'code':'USD', 'codein':'BRL', 'bid':'7.31'}}
    class Response:
        status_code = 200
        def raise_for_status(self): raise RuntimeError('secondary unavailable')
        def json(self): return payload
    class Client:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def get(self, url): return Response()
    monkeypatch.setattr(search_tools.httpx, 'Client', Client)
    monkeypatch.setattr(search_tools, 'call_with_resilience', lambda name, fn: fn())
    with pytest.raises(RuntimeError):
        search_tools.search_currency_rate('EUR', 'BRL')


def _native_output(monkeypatch, original, tool, arguments, context=None):
    monkeypatch.setattr(settings, 'GEMINI_API_KEY', 'synthetic-not-sent')
    class Provider:
        def __init__(self): self.models, self.turn = self, 0
        def generate_content(self, **kwargs):
            self.turn += 1
            part = (types.Part(function_call=types.FunctionCall(name=tool, args=arguments))
                    if self.turn == 1 else types.Part.from_text(text='Concluído'))
            return types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(role='model', parts=[part]))])
    return CustomerSupportAgent(Provider()).run('Consulta operacional delegada', CTX,
        authorization_message=original, conversation_context=context)


def test_native_binds_missing_sales_date_from_original(monkeypatch):
    out = _native_output(monkeypatch, 'Vendas de 21 de outubro de 2026', 'get_sales_settlement', {})
    assert out.structured_data['get_sales_settlement']['requested_sales_date'] == '2026-10-21'


def test_native_sales_date_adversary_rejected(monkeypatch):
    out = _native_output(monkeypatch, 'Vendas de October 5, 2026', 'get_sales_settlement', {'sales_date':'2026-10-21'})
    assert out.abstained
    assert out.tool_results[0].error_code == 'REQUEST_DATE_MISMATCH'
    assert not out.structured_data


def test_native_omitted_terminal_binds_persisted_reference(monkeypatch):
    out = _native_output(monkeypatch, 'Check the same terminal', 'diagnose_terminal_issue', {},
                         {'entities': {'terminal_id':'GET-CLASSICA-10442'}})
    assert out.structured_data['diagnose_terminal_issue']['terminal']['terminal_id'] == 'GET-CLASSICA-10442'


def test_native_binds_exact_transaction(monkeypatch):
    out = _native_output(monkeypatch, 'Verifique TXN-881915', 'diagnose_terminal_issue', {})
    assert out.structured_data['diagnose_terminal_issue']['last_declined_transaction']['transaction_id'] == 'TXN-881915'


@pytest.mark.parametrize('text', ['5th of October 2026', '5 de outubro, 2026'])
def test_written_date_connectors_are_equivalent(text):
    out = CustomerSupportAgent().run('Vendas de ' + text, CTX)
    assert out.structured_data['get_sales_settlement']['requested_sales_date'] == '2026-10-05'


def test_native_remote_omitted_unique_target_is_bound_before_consent(monkeypatch):
    out = _native_output(monkeypatch, 'Reinicie GET-CLASSICA-10442', 'execute_terminal_remote_action', {'action_type':'remote_reboot'})
    assert out.tool_results[0].success
    assert out.structured_data['execute_terminal_remote_action']['terminal_id'] == 'GET-CLASSICA-10442'


def test_transaction_only_request_renders_that_transactions_decline():
    out = CustomerSupportAgent().run('TXN-881915?', CTX)
    assert 'TXN-881915' in out.response
    assert 'TXN-881920' not in out.response


def test_explicit_transaction_cannot_render_other_decline_code():
    result = customer_tools.diagnose_terminal_issue('cliente1988', transaction_id='TXN-881915', decline_code='91', symptom='decline')
    assert result['found'] is False
    assert result['error_code'] == 'TRANSACTION_CODE_MISMATCH'
