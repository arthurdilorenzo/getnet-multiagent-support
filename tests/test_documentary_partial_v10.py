"""V10 regression contracts; native replies are synthetic, never provider quality proof."""
import json
from types import SimpleNamespace

import pytest
from google.genai import types

from app.rag.semantic import (SemanticEvidenceAuthorizer, candidate_spans, content_hash)
from app.security.request_units import request_units


@pytest.fixture(autouse=True)
def reset_synthetic_breaker():
    from app.observability.resilience import breaker_registry
    breaker_registry.reset_all()
    yield
    breaker_registry.reset_all()


def source():
    body = ('No Brasil, o Crediário permite até 48 parcelas, conforme condições do emissor.\n'
            'O Pix pode levar até um dia corrido, conforme análise de segurança.')
    return {'chunk_id': 'v10-synthetic', 'content': body, 'content_hash': content_hash(body),
            'source': 'synthetic', 'source_url': 'https://example.test/v10'}


def protocol(message, issue='scope_mismatch', affected='self', *, missing=False, kind='requirement'):
    units = request_units(message)
    assert len(units) >= 2
    first, second = units[0]['unit_id'], units[-1]['unit_id']
    def requirement(spans, decision='supported'):
        return dict(kind='requirement', decision=decision, subject='Getnet', relation='installments',
                    value='', unit='', qualifiers=[], span_ids=spans)
    analysis = {'request_hash': content_hash(message), 'units': {
        first: requirement([candidate_spans(source())[0]['span_id']]),
        second: requirement([], 'unknown')}}
    for unit in units[1:-1]:
        analysis['units'][unit['unit_id']] = dict(kind='courtesy', decision='not_applicable', subject='', relation='', value='', unit='', qualifiers=[], span_ids=[])
    if kind != 'requirement':
        analysis['units'][second] = dict(kind=kind, decision='not_applicable', subject='', relation='',
                                         value='', unit='', qualifiers=[], span_ids=[])
    checks = {first: dict(verdict='accept', issues=[]), second: dict(verdict='reject', issues=[issue])}
    for unit in units[1:-1]:
        checks[unit['unit_id']] = dict(verdict='accept', issues=[])
    if not missing:
        checks[first]['affected_unit_ids'] = []
        checks[second]['affected_unit_ids'] = [second] if affected == 'self' else affected
    verification = {'request_hash': content_hash(message), 'checks': checks, 'complete': False}
    return analysis, verification, first, second


def authorize(message, **kwargs):
    analysis, verification, first, second = protocol(message, **kwargs)
    replies = iter([analysis, verification]); requests = []
    def generate_content(**request):
        requests.append(request)
        return SimpleNamespace(candidates=[SimpleNamespace(finish_reason=types.FinishReason.STOP,
            content=types.Content(role='model', parts=[types.Part.from_text(text=json.dumps(next(replies)))]))])
    client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    return SemanticEvidenceAuthorizer(client).authorize(message, [source()]), first, second, requests


def test_independent_country_rejection_keeps_verified_brazilian_fact():
    message = 'Explain Crediário installments in Brazil; confirm the terms in Norway.'
    out, first, second, requests = authorize(message)
    assert out.status == 'partial'
    assert out.covered_unit_ids == [first]
    assert out.unresolved_unit_ids == [second]
    assert out.approved_chunks[0]['approved_spans'][0] == candidate_spans(source())[0]
    assert len(requests) == 2


@pytest.mark.parametrize('issue', ['omitted_condition', 'omitted_exception', 'scope_mismatch', 'polarity_mismatch', 'invalid_delegation'])
def test_missing_dependency_scope_preserves_conservative_global_veto(issue):
    message = 'Explain Crediário installments; provided my company is abroad.'
    out, first, second, _ = authorize(message, issue=issue, missing=True)
    assert out.status == 'abstained'
    assert not out.covered_unit_ids and not out.approved_chunks
    assert set(out.unresolved_unit_ids) == {first, second}


@pytest.mark.parametrize('issue', ['omitted_condition', 'omitted_exception', 'scope_mismatch', 'polarity_mismatch'])
def test_dependency_on_earlier_fact_invalidates_that_fact(issue):
    message = 'Explain Crediário installments; provided my company is abroad.'
    ids = [unit['unit_id'] for unit in request_units(message)]
    out, _, _, _ = authorize(message, issue=issue, affected=ids)
    assert out.status == 'abstained' and not out.approved_chunks


@pytest.mark.parametrize('affected', [[], ['invented'], ['u2', 'u2'], ['u1']])
def test_invalid_dependency_scope_is_protocol_failure(affected):
    out, _, _, _ = authorize('Explain Crediário in Brazil; confirm Norway.', affected=affected)
    assert out.status == 'abstained' and not out.approved_chunks
    assert out.error_code == 'DEPENDENCY_SCOPE'


def test_condition_hidden_as_context_cannot_use_local_dependency_escape():
    out, _, _, _ = authorize('Explain Crediário installments; for a foreign resident.',
                             issue='omitted_condition', kind='context')
    assert out.status == 'abstained' and not out.approved_chunks


@pytest.mark.parametrize('message', [
    'Explain Brazilian Crediário; only if those terms apply in Norway.',
    'Explique o Crediário brasileiro; desde que as condições valham na Noruega.',
    'Explain Brazilian Crediário; except when the issuer refuses it.',
])
def test_explicit_attached_condition_cannot_underdeclare_affected_fact(message):
    out, _, _, _ = authorize(message, issue='omitted_condition')
    assert out.status == 'abstained' and not out.approved_chunks


def completion(*calls, text=None):
    parts = [types.Part(function_call=types.FunctionCall(name=name, args=args)) for name, args in calls]
    if text is not None:
        parts.append(types.Part.from_text(text=json.dumps(text)))
    return types.GenerateContentResponse(candidates=[types.Candidate(
        content=types.Content(role='model', parts=parts), finish_reason='STOP')])


class Provider:
    def __init__(self, *replies):
        self.replies = iter(replies)
        self.requests = []
        self.models = self

    def generate_content(self, **kwargs):
        self.requests.append(kwargs)
        reply = next(self.replies)
        return reply(kwargs) if callable(reply) else reply


@pytest.mark.parametrize('message', [
    'Where do I insert the Algar SIM in the Getnet terminal and how do I test connectivity?',
    'Como colocar o chip Algar na Getnet e testar a conexão?',
    'Explain the SIM installation steps for my Getnet device.',
])
def test_documentary_native_priority_instructions_preserve_native_choice(monkeypatch, message):
    from app.agents.router_agent import RouterAgent
    from app.agents.knowledge_agent import KnowledgeAgent
    from app.config import settings
    monkeypatch.setattr(settings, 'GEMINI_API_KEY', 'synthetic-never-sent')
    def route(request):
        instruction = request['config'].system_instruction
        assert 'documentação Getnet tem prioridade' in instruction
        return completion(('KNOWLEDGE_AGENT', dict(sub_query=message, rationale='manual',
                           request_unit_ids=[u['unit_id'] for u in request_units(message)])))
    provider = Provider(route)
    plan = RouterAgent(provider).plan(message, 'synthetic')
    assert plan.primary_agent == 'KNOWLEDGE_AGENT'
    def choose(request):
        assert 'documentação Getnet tem prioridade' in request['config'].system_instruction
        return completion(('search_getnet_knowledge', {'query': message}))
    specialist = KnowledgeAgent(Provider(choose))
    assert specialist._plan_native(message)[0].tool_name == 'search_getnet_knowledge'


@pytest.mark.parametrize('message', [
    'Where do I insert the Algar SIM in the Getnet terminal and how do I test connectivity?',
    'Como colocar o chip Algar na Getnet e testar a conexão?',
    'Explain the SIM installation steps for my Getnet device.',
])
def test_native_web_proposal_moves_to_knowledge_when_local_getnet_evidence_is_relevant(monkeypatch, message):
    from app.agents.router_agent import RouterAgent
    from app.config import settings
    monkeypatch.setattr(settings, 'GEMINI_API_KEY', 'synthetic-never-sent')
    provider = Provider(completion(('WEB_SEARCH_AGENT', dict(sub_query=message, rationale='external update'))))
    plan = RouterAgent(provider).plan(message, 'synthetic')
    assert plan.primary_agent == 'KNOWLEDGE_AGENT'
    assert all(step.target_agent == 'KNOWLEDGE_AGENT' for step in plan.plan_steps)
    assert {uid for step in plan.plan_steps for uid in step.request_unit_ids} == {
        unit['unit_id'] for unit in request_units(message)
    }


def test_native_web_choice_is_preserved_without_relevant_local_document(monkeypatch):
    from app.agents.router_agent import RouterAgent
    from app.config import settings
    monkeypatch.setattr(settings, 'GEMINI_API_KEY', 'synthetic-never-sent')
    message = 'What is the current weather in Porto Alegre tomorrow?'
    provider = Provider(completion(('WEB_SEARCH_AGENT', dict(sub_query=message, rationale='current forecast'))))
    plan = RouterAgent(provider).plan(message, 'synthetic')
    assert plan.primary_agent == 'WEB_SEARCH_AGENT'


def test_native_web_proposal_is_not_rebound_to_unverified_candidate(monkeypatch):
    from app.agents.router_agent import RouterAgent
    from app.config import settings
    from app.rag.retriever import retriever
    monkeypatch.setattr(settings, 'GEMINI_API_KEY', 'synthetic-never-sent')
    message = 'Where do I insert the Algar SIM in the Getnet terminal and how do I test connectivity?'
    monkeypatch.setattr(retriever, 'retrieve_candidates', lambda **kwargs: {
        'candidate_chunks': [{
            'source_url': 'https://site.getnet.com.br/duvidas/',
            'origin_type': 'live_web_scrape',
            'provenance': {'verification_status': 'unreviewed'},
            'relevance_score': 0.99,
        }]
    })
    provider = Provider(completion(('WEB_SEARCH_AGENT', dict(sub_query=message, rationale='external update'))))
    plan = RouterAgent(provider).plan(message, 'synthetic')
    assert plan.primary_agent == 'WEB_SEARCH_AGENT'


def test_mixed_firmware_has_separate_native_delegations(monkeypatch):
    from app.agents.router_agent import RouterAgent
    from app.config import settings
    monkeypatch.setattr(settings, 'GEMINI_API_KEY', 'synthetic-never-sent')
    message = 'Explique o prazo do Pix Getnet; consulte o firmware de GET-CLASSICA-QA-9627.'
    units = request_units(message)
    private = units[-1]['unit_id']; public = [u['unit_id'] for u in units[:-1]]
    def route(request):
        assert 'firmware instalado' in request['config'].system_instruction
        return completion(
            ('KNOWLEDGE_AGENT', dict(sub_query=units[0]['text'], rationale='documento', request_unit_ids=public)),
            ('CUSTOMER_SUPPORT_AGENT', dict(sub_query=units[-1]['text'], rationale='estado privado', request_unit_ids=[private])))
    plan = RouterAgent(Provider(route)).plan(message, 'synthetic')
    assert [(s.target_agent, s.request_unit_ids) for s in plan.plan_steps] == [
        ('KNOWLEDGE_AGENT', public), ('CUSTOMER_SUPPORT_AGENT', [private])]


def test_real_document_partial_is_rendered_and_reported_in_ledger(monkeypatch):
    from app.agents.knowledge_agent import KnowledgeAgent
    from app.config import settings
    from app.security.auth import SecurityContext
    from app.security.request_units import RequestCoverageLedger
    from app.tools.search_tools import retriever
    monkeypatch.setattr(settings, 'GEMINI_API_KEY', 'synthetic-never-sent')
    # Existing local corpus, unchanged. Retrieval, registry, rendering and ledger are real.
    message = 'Explain Crediário installments in Brazil; confirm the terms in Norway.'
    chunk = next(c for c in retriever.retrieve_candidates(query=message)['candidate_chunks'] if 'até 48x' in c['content'])
    monkeypatch.setattr(__import__(__name__, fromlist=['source']), 'source', lambda: chunk)
    analysis, verification, first, second = protocol(message)
    span = next(s for s in candidate_spans(chunk) if 'até 48x' in s['quote'])
    analysis['units'][first]['span_ids'] = [span['span_id']]
    provider = Provider(completion(('search_getnet_knowledge', {'query': message})),
                        completion(text=analysis), completion(text=verification))
    out = KnowledgeAgent(provider).run(message, SecurityContext(authenticated_user_id='synthetic'))
    assert out.answer_status == 'partial' and out.abstained
    assert span['quote'] in out.response and 'Norway' in out.response
    assert out.covered_unit_ids == [first] and out.unresolved_unit_ids == [second]
    assert out.citations and out.evidence and len(provider.requests) == 3
    ledger = RequestCoverageLedger(message)
    ledger.assign('KNOWLEDGE_AGENT', out.request_unit_ids)
    assert ledger.report('KNOWLEDGE_AGENT', out.request_unit_ids, out.covered_unit_ids,
                         out.unresolved_unit_ids, len(out.evidence))
    assert [row['state'] for row in ledger.snapshot()] == ['fulfilled', 'unresolved']


def test_native_verifier_schema_requests_explicit_dependency_scope():
    from app.rag.semantic import compact_schema
    schema = compact_schema('semantic_verification', content_hash('request'), ['u1', 'u2'], ['source:p0'])
    check = schema['properties']['checks']['properties']['u1']
    assert 'affected_unit_ids' in check['required']
    array = next(branch for branch in check['properties']['affected_unit_ids']['anyOf'] if branch['type'] == 'array')
    assert array['items']['enum'] == ['u1', 'u2']


def test_native_planning_receives_original_and_scope_despite_reduced_subquery(monkeypatch):
    from app.agents.knowledge_agent import KnowledgeAgent
    from app.config import settings
    from app.security.auth import SecurityContext
    monkeypatch.setattr(settings, 'GEMINI_API_KEY', 'synthetic-never-sent')
    original = 'Explain Pix Getnet; inspect firmware on my terminal.'
    units = request_units(original)
    scope = {'assigned_unit_ids': [units[0]['unit_id']], 'other_unit_ids': [u['unit_id'] for u in units[1:]]}
    requests = []
    def choose(request):
        requests.append(request)
        text = '\n'.join(p.text for c in request['contents'] for p in c.parts if p.text)
        assert original in text
        assert json.dumps(scope, ensure_ascii=False) in text
        assert all(unit['text'] in text for unit in units)
        return completion(('search_getnet_knowledge', {'query': 'Pix Getnet'}))
    # Stop after native planning: malformed semantic transport is deliberately
    # fail-closed, while the planning request itself must preserve full scope.
    provider = Provider(choose, completion(text={}))
    out = KnowledgeAgent(provider).run('Pix Getnet', SecurityContext(authenticated_user_id='synthetic'),
                                      request_message=original, delegation_scope=scope)
    assert requests and out.tools_called == ['search_getnet_knowledge']
