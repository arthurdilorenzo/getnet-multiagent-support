"""Fresh synthetic relation tests; no reserved evaluation questions are used."""
import hashlib
from dataclasses import replace

import pytest

from app.rag.answerability import assess


LINK = 'O Link de Pagamento permite cobrar clientes à distância. O Link de Pagamento aceita crédito e Pix; as condições dependem do plano contratado.'


@pytest.mark.parametrize('query', [
    'Por gentileza, explique como funciona o Link de Pagamento.',
    'Tenho fregueses em outra cidade. Como o Link de Pagamento permite cobrar?',
    'Please explain payment link for my small shop.',
])
def test_incidental_context_does_not_become_a_property(query):
    assert assess(query, LINK)['answerable']


def test_cash_is_a_payment_requirement_not_scaffolding():
    assert not assess('O Link de Pagamento permite cobrar cliente usando dinheiro?', LINK)['answerable']


def test_pix_as_an_object_does_not_acquire_links_card_capability():
    assert not assess('O Pix aceita cartão?', LINK)['answerable']


def test_two_subjects_require_two_supported_relations():
    assert not assess('O Link de Pagamento e o Pix aceitam cartão?', LINK)['answerable']


def test_attribute_is_bound_to_its_subject():
    source = 'A Get Clássica possui teclado físico. A Get Smart possui câmera.'
    assert assess('A Get Clássica possui teclado físico?', source)['answerable']
    assert not assess('A Get Smart possui teclado físico?', source)['answerable']


@pytest.mark.parametrize('query', [
    'Posso usar o Link de Pagamento na Argentina?',
    'Como cancelar o Link de Pagamento?',
    'A Get Smart possui sensor de temperatura?',
])
def test_absent_country_operation_and_unknown_attribute_are_unknown(query):
    assert not assess(query, LINK + ' A Get Smart possui câmera.')['answerable']


def test_amount_without_its_unit_is_not_support():
    source = 'O Crediário permite parcelar em até 48x. O Crediário recebe em até 2 dias úteis.'
    assert not assess('O Crediário recebe em até 48 dias?', source)['answerable']
    assert assess('O Crediário permite parcelar em até 48x?', source)['answerable']


def test_negative_fact_is_not_rendered_as_an_affirmative_fact():
    source = 'A Get Smart não possui teclado físico; o recurso depende do modelo contratado.'
    result = assess('A Get Smart possui teclado físico?', source)
    assert result['answerable']
    assert result['coverage'][0]['facts'][0]['polarity'] is False
    assert result['answer_snippets'] == [source]


def test_conditions_are_preserved_in_exact_extraction():
    result = assess('O Link de Pagamento aceita Pix?', LINK)
    assert result['answerable']
    assert 'condições dependem do plano contratado' in ' '.join(result['answer_snippets'])


def test_compound_request_does_not_get_global_approval_from_one_relation():
    assert not assess('O Link de Pagamento aceita Pix e como cancelo a cobrança?', LINK)['answerable']


def test_catalog_rejects_changed_hash_and_offsets():
    from app.rag.facts import EvidenceFact, FactCatalog
    source = 'A Get Smart possui câmera.'
    fact = EvidenceFact(fact_id='synthetic-camera', subject='smart', relation='hardware_feature',
                        value='camera', chunk_id='synthetic', content_hash=hashlib.sha256(source.encode()).hexdigest(),
                        start=0, end=len(source), excerpt=source)
    assert FactCatalog([fact]).for_chunk('synthetic', source) == [fact]
    assert FactCatalog([fact]).for_chunk('synthetic', source + ' Texto novo.') == []
    assert FactCatalog([replace(fact, start=2)]).for_chunk('synthetic', source) == []


def test_original_request_survives_provider_query_rewrite(monkeypatch):
    from app.agents.knowledge_agent import KnowledgeAgent
    from app.schemas import ToolCallSpec
    from app.security.auth import SecurityContext
    agent = KnowledgeAgent()
    monkeypatch.setattr(agent, '_plan_tool_calls', lambda **kwargs: [ToolCallSpec(tool_name='search_getnet_knowledge', arguments={'query':'Como funciona Pix?'})])
    out = agent.run('Como funciona Pix?', SecurityContext(authenticated_user_id='synthetic'), request_message='Posso usar Pix na Argentina?')
    assert out.abstained


def test_retriever_can_cover_two_relations_from_distinct_chunks(tmp_path):
    from app.rag.retriever import HybridKnowledgeRetriever
    (tmp_path/'synthetic.md').write_text('## Hardware\nA Get Smart possui câmera.\n\n## Software\nA Get Smart possui Android.', encoding='utf-8')
    retriever = HybridKnowledgeRetriever(data_dir=tmp_path, corpus_path=tmp_path/'c.json', vector_index_path=tmp_path/'v.json', min_relevance_threshold=0)
    result = retriever.retrieve_hybrid('A Get Smart possui câmera e Android?')
    assert not result['abstained']
    assert len(result['results']) == 2
    assert len(result['answerability']['coverage']) == 2


def test_renderer_rejects_tampered_approved_span():
    from app.agents.knowledge_agent import KnowledgeAgent
    from app.rag.facts import extract_facts
    from app.schemas import ToolExecutionResult
    source = 'A Get Smart possui câmera.'
    fact = extract_facts('synthetic',source)[-1].to_dict()
    fact['excerpt'] = 'A Get Smart possui sensor térmico.'
    data = {'chunks':[{'chunk_id':'synthetic','content':source,'answer_snippets':[fact['excerpt']], 'approved_facts':[fact]}]}
    text = KnowledgeAgent()._synthesize_from_tool_results('A Get Smart possui câmera?', [ToolExecutionResult(tool_name='search_getnet_knowledge',success=True,data=data)], [])
    assert 'sensor térmico' not in text
    assert 'não encontrei' in text.lower()


@pytest.mark.parametrize('query,city', [
    ('Chove amanhã no Recife?', 'Recife'),
    ('For Manaus tomorrow, forecast the rain probability.', 'Manaus'),
])
def test_weather_city_uses_preposition_without_a_default(query, city):
    from app.agents.knowledge_agent import KnowledgeAgent
    call = KnowledgeAgent()._plan_tool_calls(query, True)[0]
    assert call.arguments['city'].casefold() == city.casefold()


@pytest.mark.parametrize('query', [
    'Como funciona o Link de Pagamento com cobrança recorrente?',
    'Como funciona o Link de Pagamento no Japão?',
    'Qual a taxa do Link de Pagamento em 2031-03-04?',
    'O Link de Pagamento cobra 0,75%?',
])
def test_qualifiers_unknown_scope_date_and_decimal_are_not_erased(query):
    assert not assess(query, LINK)['answerable']


@pytest.mark.parametrize('original,tool,args', [
    ('Qual a cotação de EUR para BRL?', 'get_currency_exchange_rate', {'base_currency':'GBP','target_currency':'BRL'}),
    ('Qual a cotação de EUR para JPY?', 'get_currency_exchange_rate', {'base_currency':'EUR','target_currency':'USD'}),
    ('Chove amanhã no Recife?', 'get_weather_forecast', {'city':'Manaus','days_ahead':1}),
    ('Chove amanhã no Recife?', 'get_weather_forecast', {'city':'Recife','days_ahead':0}),
])
def test_public_calls_cannot_rewrite_explicit_original_slots(monkeypatch, original, tool, args):
    from app.agents.knowledge_agent import KnowledgeAgent
    from app.schemas import ToolCallSpec
    from app.security.auth import SecurityContext
    from app.tools.registry import tool_registry
    calls = []
    def forbidden_execute(**kwargs):
        calls.append(kwargs)
        pytest.fail('A divergent public call reached the tool registry.')
    monkeypatch.setattr(tool_registry,'execute',forbidden_execute)
    agent = KnowledgeAgent()
    monkeypatch.setattr(agent,'_plan_tool_calls',lambda **kwargs: [ToolCallSpec(tool_name=tool,arguments=args)])
    result=agent.run('consulta reformulada',SecurityContext(authenticated_user_id='synthetic'),True,request_message=original)
    assert result.abstained
    assert not calls


def test_brand_affiliation_is_context_not_an_unknown_property():
    source = 'A antecipação de recebíveis adianta vendas de crédito antes do prazo de compensação.'
    assert assess('Please explain receivables advance with Getnet.',source)['answerable']


def test_decimal_fee_constraint_keeps_both_amount_and_unit():
    source = 'O Link de Pagamento tem taxa de 0,25% conforme o plano contratado.'
    assert not assess('Qual a taxa de 0,75% do Link de Pagamento?',source)['answerable']
    assert assess('O Link de Pagamento tem taxa de 0,25%?',source)['answerable']


def test_compound_sentence_keeps_each_attribute_on_its_own_subject():
    source = 'A Get Smart possui câmera e a Get Clássica possui teclado físico.'
    assert not assess('A Get Smart possui teclado físico?',source)['answerable']
    assert assess('A Get Clássica possui teclado físico?',source)['answerable']


def test_contrastive_clause_does_not_inherit_negation():
    source = 'A Get Smart não possui teclado físico, mas possui Android.'
    result=assess('A Get Smart possui Android?',source)
    assert result['answerable']
    assert all(f['polarity'] for c in result['coverage'] for f in c['facts'])
