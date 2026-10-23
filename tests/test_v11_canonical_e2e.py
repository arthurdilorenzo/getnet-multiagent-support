"""Offline synthetic FC transport + real serialized /api/chat regressions.

Providers below are fixtures, never Gemini clients. Handlers use local corpus or
synthetic tenant data; the Web failure is injected before any network access.
"""
import importlib
import json

import pytest
from fastapi.testclient import TestClient
from google.genai import types

from app.agents.orchestrator import MultiAgentOrchestrator
from app.config import settings
from app.memory.store import StateStore
from app.security.request_units import request_units
from app.tools.registry import tool_registry
from app.schemas import ToolExecutionResult

TARGET = 'GET-SMART-QA-65172'


class SyntheticFC:
    """Explicitly synthetic provider exercising native FC parsing offline."""
    def __init__(self, calls):
        self.models = self
        self.calls = calls
        self.requests = []

    def generate_content(self, **kwargs):
        self.requests.append(kwargs)
        return types.GenerateContentResponse(candidates=[types.Candidate(
            finish_reason='STOP', content=types.Content(role='model', parts=[
                types.Part(function_call=types.FunctionCall(name=name, args=args))
                for name, args in self.calls]))])


@pytest.fixture
def boundary(monkeypatch, tmp_path):
    state = StateStore(tmp_path / 'canonical-e2e.sqlite')
    for name in ('app.agents.orchestrator', 'app.main', 'app.tools.customer_tools',
                 'app.security.action_policy', 'app.tools.registry'):
        monkeypatch.setattr(importlib.import_module(name), 'state_store', state)
    monkeypatch.setattr(settings, 'GEMINI_API_KEY', '')
    monkeypatch.setattr(settings, 'AUTH_ENFORCE_STRICT', False)
    fixture = tmp_path / 'customers.json'
    fixture.write_text(json.dumps({'customers': {'synthetic': {'company_name': 'Synthetic',
        'terminals': [{'terminal_id': TARGET, 'model': 'Get Smart', 'status': 'Online',
        'connection_type': 'Wi-Fi', 'signal_strength': '2/4 bars', 'firmware_version': 'lab-46.2'}]}}}))
    monkeypatch.setattr(settings, 'MOCK_CUSTOMERS_PATH', fixture)
    orch = MultiAgentOrchestrator()
    main = importlib.import_module('app.main')
    monkeypatch.setattr(main, 'orchestrator', orch)
    observed = []
    real_execute = tool_registry.execute

    def execute(**kwargs):
        observed.append((kwargs['tool_name'], dict(kwargs['raw_arguments'])))
        if kwargs['tool_name'] == 'web_search_general':
            return ToolExecutionResult(tool_name='web_search_general', success=False,
                outcome_state='failed', error_code='EXTERNAL_SERVICE_UNAVAILABLE', error_message='synthetic offline failure')
        return real_execute(**kwargs)

    monkeypatch.setattr(tool_registry, 'execute', execute)

    def chat(message, query=None, agent='KNOWLEDGE_AGENT', *, replan=False):
        ids = [u['unit_id'] for u in request_units(message)]
        provider = SyntheticFC([(agent, {'sub_query': query or message,
            'rationale': 'synthetic offline canonical regression', 'request_unit_ids': ids})])
        orch.router.client = provider
        real_plan = orch.router.plan
        def native_plan(**kwargs):
            with monkeypatch.context() as config:
                config.setattr(settings, 'GEMINI_API_KEY', 'synthetic-never-sent')
                return real_plan(**kwargs)
        monkeypatch.setattr(orch.router, 'plan', native_plan)
        if not replan:
            monkeypatch.setattr(orch, '_observe_and_replan', lambda **kwargs: False)
        response = TestClient(main.app).post('/api/chat', json={'message': message, 'user_id': 'synthetic'})
        assert response.status_code == 200, response.text
        assert provider.requests
        return response.json(), observed, orch
    return chat


@pytest.mark.parametrize('message', [
    'Name the account types accepted for Pix in Brazil. Answer in bullet points.',
    'Quais categorias de conta podem receber Pix na Getnet? Responda em tópicos.',
    'Name the account types accepted for Pix in Brazil. Answer using Getnet internal material, without replacing it with institutional participation rules.',
    'Quais categorias de conta podem receber Pix na Getnet? Responda pelo material interno da Getnet, sem substituir isso por regras de participação de instituições no Pix.',
])
def test_source_preference_is_attached_and_fulfilled_in_http(boundary, monkeypatch, message):
    scripted_native_evidence(monkeypatch, message, 'conta corrente, poupança')
    body, calls, _ = boundary(message)
    assert body['answer_status'] == 'complete'
    assert len(body['request_coverage']) == 1
    assert all(row['state'] == 'fulfilled' for row in body['request_coverage'])
    assert body['citations'] and body['evidence']
    assert [name for name, _ in calls] == ['search_getnet_knowledge']
    assert not body['action_proposals']


@pytest.mark.parametrize('context,task', [
    ('There is an Algar SIM in the reader packaging.', 'Explain safe installation and communication testing.'),
    ('Achei um chip Algar na embalagem do leitor.', 'Explique a instalação segura e o teste de comunicação.'),
])
def test_context_and_task_survive_context_only_fc_query(boundary, monkeypatch, context, task):
    scripted_native_evidence(monkeypatch, context, 'parte metálica para baixo')
    body, calls, _ = boundary(context + ' ' + task, query=context)
    queries = [args['query'] for name, args in calls if name == 'search_getnet_knowledge']
    assert queries and all(task in query and context in query for query in queries)
    assert any(row['action'] == 'call:search_getnet_knowledge' for row in body['trace'])
    assert 'desligada' in body['response'] and 'metálica para baixo' in body['response']
    assert 'teste de comunicação' in body['response'] and body['citations']
    assert body['evidence'] and body['answer_status'] == 'complete'
    assert not body['action_proposals']


@pytest.mark.parametrize('message', [
    f'If firmware of {TARGET} is lab-46.9 and status is Online, explain Getnet Crediário advance-payment costs.',
    f'Se o firmware do terminal {TARGET} for lab-46.9 e o status for Online, explique os custos de antecipação do Crediário Getnet.',
    f'If firmware of {TARGET} is lab-46.9 or status is Online, explain Getnet Crediário advance-payment costs.',
])
def test_unknown_or_false_predicate_still_reads_exact_target_without_dependent_lookup(boundary, message):
    body, calls, _ = boundary(message)
    assert [name for name, _ in calls] == ['diagnose_terminal_issue']
    assert calls[0][1]['terminal_id'] == TARGET
    assert 'lab-46.2' in body['response'] and 'Online' in body['response']
    assert 'não' in body['response'].lower()
    assert body['answer_status'] == 'abstained'
    assert all(row['text'].strip(' ;,.!?') for row in body['request_coverage'])
    assert not body['action_proposals']


@pytest.mark.parametrize('message', [
    'Is there a Getnet outage in Campinas right now? Consult current sources and state if it cannot be confirmed.',
    'Há uma pane da Getnet em Campinas agora? Consulte fontes atuais e informe se não houver confirmação.',
])
def test_failed_current_incident_lookup_answers_specific_nonconfirmation(boundary, message):
    body, calls, _ = boundary(message, agent='WEB_SEARCH_AGENT')
    assert calls and all(name == 'web_search_general' for name, _ in calls)
    assert all('Getnet' in args['query'] and 'Campinas' in args['query'] for _, args in calls)
    assert 'Getnet' in body['response'] and 'Campinas' in body['response']
    assert 'não confirm' in body['response'].lower()
    assert body['answer_status'] == 'abstained'
    assert not body['citations'] and not body['action_proposals']


@pytest.mark.parametrize('message,qualifier', [
    ('Explain Getnet Crediário timing and advance-payment costs in Brazil. For the Norwegian shop, leave applicability open unless a current Norway-specific external source supports it.',
     'For the Norwegian shop, leave applicability open unless a current Norway-specific external source supports it.'),
    ('Explique prazo e custo de antecipação do Crediário Getnet no Brasil. Para a loja na Noruega, mantenha a aplicabilidade em aberto sem fonte externa atual específica do país.',
     'Para a loja na Noruega, mantenha a aplicabilidade em aberto sem fonte externa atual específica do país.'),
])
def test_country_qualifier_retains_product_in_real_web_arguments(boundary, message, qualifier):
    body, calls, _ = boundary(message, query=qualifier, agent='WEB_SEARCH_AGENT')
    web = [args['query'] for name, args in calls if name == 'web_search_general']
    assert web and all('Crediário' in query and 'Getnet' in query for query in web)
    assert not body['action_proposals']


class SyntheticEvidenceFC(SyntheticFC):
    """Scripted evidence decisions validate real spans/schema, not LLM quality."""
    def __init__(self, query, marker):
        super().__init__([('search_getnet_knowledge', {'query': query})])
        self.marker = marker

    def generate_content(self, **kwargs):
        schema = kwargs['config'].response_json_schema
        if schema is None:
            return super().generate_content(**kwargs)
        self.requests.append(kwargs)
        payload = json.loads(kwargs['contents'][0].parts[0].text)
        units = payload['request_units']
        assigned = set(payload['delegation_scope']['assigned_unit_ids'])
        if 'units' in schema['properties']:
            spans = [span for chunk in payload['candidates'] for span in chunk['spans']
                     if self.marker in span['quote']]
            assert spans, 'real corpus must contain requested literal evidence'
            decisions = {}
            for unit in units:
                uid = unit['unit_id']
                foreign = any(token in unit['text'].lower() for token in ('norway', 'norwegian', 'noruega'))
                kind = 'delegated' if uid not in assigned else 'unresolved' if foreign else 'requirement'
                decisions[uid] = {'kind': kind,
                    'decision': 'supported' if kind == 'requirement' else 'unknown' if kind == 'unresolved' else 'not_applicable',
                    'subject': 'literal local product' if kind == 'requirement' else '',
                    'relation': 'documented fact' if kind == 'requirement' else '', 'value': '', 'unit': '',
                    'qualifiers': [], 'span_ids': [spans[0]['span_id']] if kind == 'requirement' else []}
            content = {'request_hash': payload['request_hash'], 'units': decisions}
        else:
            content = {'request_hash': payload['request_hash'], 'complete': all(
                row['kind'] != 'unresolved' for row in payload['analysis']['units'].values()),
                'checks': {unit['unit_id']: {'verdict': 'accept', 'issues': [], 'affected_unit_ids': []} for unit in units}}
        return types.GenerateContentResponse(candidates=[types.Candidate(finish_reason='STOP',
            content=types.Content(role='model', parts=[types.Part.from_text(text=json.dumps(content))]))])


def scripted_native_evidence(monkeypatch, query, marker):
    from app.agents.knowledge_agent import KnowledgeAgent
    original = KnowledgeAgent.run
    providers = []
    def run(self, **kwargs):
        self.client = SyntheticEvidenceFC(query, marker)
        providers.append(self.client)
        with monkeypatch.context() as config:
            config.setattr(settings, 'GEMINI_API_KEY', 'synthetic-never-sent')
            return original(self, **kwargs)
    monkeypatch.setattr(KnowledgeAgent, 'run', run)
    return providers


def test_public_context_never_absorbs_later_private_need():
    message = 'There is an Algar SIM in the reader box. Explain installation and communication testing. Inspect my balance.'
    units = request_units(message)
    assert not any('SIM' in unit['text'] and 'balance' in unit['text'] for unit in units)
    from app.security.documentary_intent import is_documentary_qualifier
    assert not is_documentary_qualifier('Answer using Getnet internal material and list the fees for Crediário.')
    assert not is_documentary_qualifier('Responda pelo material interno da Getnet e informe as tarifas do Crediário.')


@pytest.mark.parametrize('message', [
    f'If firmware of {TARGET} is lab-46.2 and status is Online, explain Getnet Crediário advance-payment costs.',
    f'Se o firmware do terminal {TARGET} for lab-46.2 e o status for Online, explique os custos de antecipação do Crediário Getnet.',
])
def test_exact_predicate_does_not_depend_on_consequence_wording(boundary, message):
    body, calls, _ = boundary(message)
    assert [name for name, _ in calls] == ['diagnose_terminal_issue', 'search_getnet_knowledge']
    gate = next(row for row in body['trace'] if row['action'] == 'evaluate_conditional_dependency')
    assert gate['details']['predicate_authorized'] is True


@pytest.mark.parametrize('brazil,foreign', [
    ('Explain Getnet Crediário settlement timing, fee and advance costs in Brazil.',
     'For the Norwegian shop, leave applicability open unless a current Norway-specific external source supports it.'),
    ('Explique o prazo, a taxa e o custo de antecipação do Crediário Getnet no Brasil.',
     'Para a loja na Noruega, mantenha a aplicabilidade em aberto sem fonte externa atual específica do país.'),
])
def test_real_replan_restores_product_and_keeps_grounded_brazil_after_web_failure(boundary, monkeypatch, brazil, foreign):
    from app.agents.router_agent import RouterAgent
    message = brazil + ' ' + foreign
    scripted_native_evidence(monkeypatch, brazil, 'taxa única')
    original_replan = RouterAgent.replan
    replans = []
    def native_replan(self, **kwargs):
        ids = [unit['unit_id'] for unit in request_units(message) if foreign in unit['text']]
        # Norway may split at a conditional connector; preserve all foreign IDs.
        ids = [unit['unit_id'] for unit in request_units(message) if unit['start'] >= len(brazil)]
        self.client = SyntheticFC([('WEB_SEARCH_AGENT', {'sub_query': foreign,
            'rationale': 'synthetic fixture pending foreign source', 'request_unit_ids': ids})])
        with monkeypatch.context() as config:
            config.setattr(settings, 'GEMINI_API_KEY', 'synthetic-never-sent')
            plan = original_replan(self, **kwargs)
        replans.extend(self.client.requests)
        return plan
    monkeypatch.setattr(RouterAgent, 'replan', native_replan)
    body, calls, _ = boundary(message, replan=True)
    assert replans
    assert [name for name, _ in calls] == ['search_getnet_knowledge', 'web_search_general']
    web_query = calls[-1][1]['query']
    assert 'Getnet' in web_query and 'Crediário' in web_query
    assert ('Norway' in web_query or 'Norwegian' in web_query or 'Noruega' in web_query)
    assert '2 dias úteis' in body['response'] and 'taxa única' in body['response']
    assert 'sem custo de antecipação' in body['response']
    assert body['answer_status'] == 'partial' and body['citations'] and body['evidence']
    assert all(row['state'] == ('fulfilled' if row['start'] < len(brazil) else 'unresolved')
               for row in body['request_coverage'])
    assert not body['action_proposals']


def test_documentary_query_never_absorbs_independent_private_need(boundary, monkeypatch):
    context = 'There is an Algar SIM in the reader box.'
    task = 'Explain safe installation and communication testing.'
    scripted_native_evidence(monkeypatch, context, 'parte metálica para baixo')
    body, calls, _ = boundary(context + ' ' + task + ' Inspect my balance.', query=context)
    local = [args['query'] for name, args in calls if name == 'search_getnet_knowledge']
    assert local and all('balance' not in query.lower() for query in local)
    assert all(row['state'] != 'fulfilled' for row in body['request_coverage'] if 'balance' in row['text'])


@pytest.mark.parametrize('owner', [None, 'foreign-tenant'])
def test_missing_or_foreign_observation_owner_never_authorizes_or_renders_values(boundary, monkeypatch, owner):
    from app.agents.customer_support_agent import CustomerSupportAgent
    original = CustomerSupportAgent.run
    def corrupted(self, **kwargs):
        output = original(self, **kwargs)
        for result in output.tool_results:
            if isinstance(result.data, dict):
                result.data['user_id'] = owner
        return output
    monkeypatch.setattr(CustomerSupportAgent, 'run', corrupted)
    message = f'If firmware of {TARGET} is lab-46.2 and status is Online, explain Crediário fees.'
    body, calls, _ = boundary(message)
    assert [name for name, _ in calls] == ['diagnose_terminal_issue']
    assert 'Observado em' not in body['response']
    assert body['answer_status'] == 'abstained'
    gate = next(row for row in body['trace'] if row['action'] == 'evaluate_conditional_dependency')
    assert gate['details']['predicate_authorized'] is False


@pytest.mark.parametrize('message,factual', [
    ('Explain Getnet Pix eligibility. Answer using Getnet material: what is the cancellation fee?', 'cancellation fee'),
    ('Explique a elegibilidade do Pix Getnet. Responda pelo material Getnet: qual é a tarifa de cancelamento?', 'tarifa de cancelamento'),
])
def test_source_directive_with_new_question_preserves_factual_task(message, factual):
    from app.agents.knowledge_agent import KnowledgeAgent
    from app.schemas import ToolCallSpec
    units = request_units(message)
    scope = {'assigned_unit_ids': [unit['unit_id'] for unit in units], 'other_unit_ids': []}
    calls = KnowledgeAgent._bind_documentary_scopes(
        [ToolCallSpec(tool_name='search_getnet_knowledge', arguments={'query': message})],
        message, scope, message)
    assert factual in calls[0][0].arguments['query']
    assert len(units) == 2
