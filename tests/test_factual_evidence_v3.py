"""Development regressions for typed public facts, evidence and answerability.

This is an implementation suite, never the independent held-out audit.
"""
from types import SimpleNamespace
from datetime import date
import httpx
import pytest
from app.config import settings
from app.tools import search_tools
from app.agents.knowledge_agent import KnowledgeAgent
from app.rag.retriever import HybridKnowledgeRetriever
from app.schemas import ToolExecutionResult, ChatResponse
from app.evals.evaluator import GroundednessJudge


@pytest.fixture(autouse=True)
def no_network_retry(monkeypatch):
    monkeypatch.setattr(search_tools, 'call_with_resilience', lambda name, fn, **kw: fn())
    monkeypatch.setattr(search_tools, '_today_local', lambda: date(2026,10,23))


def mock_http(monkeypatch, awesome=None, frank=None, weather=None):
    def get(client, url):
        payload = awesome if 'awesomeapi' in url else frank if 'frankfurter' in url else weather
        return httpx.Response(200, json=payload, request=httpx.Request('GET', url))
    monkeypatch.setattr(httpx.Client, 'get', get)


def test_missing_fx_extrema_remain_unknown(monkeypatch):
    mock_http(monkeypatch, awesome={'EURBRL': {'bid':'6.1234', 'create_date':'2026-10-22 16:00:00'}})
    result = search_tools.search_currency_rate()
    assert result['high_today_brl'] is None and result['low_today_brl'] is None
    assert result['updated_at'] == '2026-10-22 16:00:00'


def test_reference_fx_is_dated_daily_reference_not_intraday(monkeypatch):
    mock_http(monkeypatch, awesome={}, frank={'date':'2026-10-19','rates':{'BRL':6.1}})
    result = search_tools.search_currency_rate()
    assert result['live'] is False
    assert result['rate_type'] == 'daily_reference'
    assert result['high_today_brl'] is None and result['low_today_brl'] is None
    out = KnowledgeAgent()._synthesize_from_tool_results('', [ToolExecutionResult(tool_name='get_currency_exchange_rate',success=True,data=result)], [])
    assert 'referência' in out.lower() and '2026-10-19' in out
    assert 'Máxima do dia' not in out and 'comercial atual' not in out


@pytest.mark.parametrize('bad', ['NaN', 'Infinity', '-1', '0'])
def test_invalid_fx_numbers_fail_closed(monkeypatch, bad):
    mock_http(monkeypatch, awesome={'EURBRL':{'bid':bad}}, frank={'date':'2026-10-20','rates':{'BRL':bad}})
    with pytest.raises((ValueError, RuntimeError)):
        search_tools.search_currency_rate()


def test_absent_precipitation_is_not_zero(monkeypatch):
    mock_http(monkeypatch, weather={'daily':{'time':['2026-10-23','2026-10-24'], 'temperature_2m_min':[19,20], 'temperature_2m_max':[25,26]}})
    data = search_tools.search_weather()
    assert data['rain_probability_percent'] is None
    out = KnowledgeAgent()._synthesize_from_tool_results('',[ToolExecutionResult(tool_name='get_weather_forecast',success=True,data=data)],[])
    assert '0%' not in out and 'não informada' in out.lower()


@pytest.mark.parametrize('prob', [-1,101,float('nan')])
def test_invalid_precipitation_is_rejected(monkeypatch, prob):
    # httpx's strict JSON encoder rejects NaN; JSON decoder behavior is simulated directly.
    payload={'daily':{'time':['2026-10-23'], 'temperature_2m_min':[19], 'temperature_2m_max':[25], 'precipitation_probability_max':[prob]}}
    class Response:
        def raise_for_status(self): pass
        def json(self): return payload
    monkeypatch.setattr(httpx.Client, 'get', lambda *a: Response())
    with pytest.raises((ValueError,RuntimeError)):
        search_tools.search_weather(days_ahead=0)


def test_model_prose_is_never_authority_even_with_official_url(monkeypatch):
    monkeypatch.setattr(settings,'GEMINI_API_KEY','fake-not-sent')
    fake=SimpleNamespace(models=SimpleNamespace(generate_content=lambda **k: SimpleNamespace(text='Pix paga R$ 9999 em 0 dias. https://site.getnet.com.br/pix/')))
    data={'chunks':[{'content':'## Pix\nO Pix permite pagamentos por QR Code.', 'title':'Pix','source_url':'https://site.getnet.com.br/pix/'}]}
    out=KnowledgeAgent(fake)._synthesize_from_tool_results('Como funciona Pix?',[ToolExecutionResult(tool_name='search_getnet_knowledge',success=True,data=data)],[])
    assert '9999' not in out and '0 dias' not in out
    assert 'QR Code' in out


@pytest.mark.parametrize('query',[
    'Como redefinir a senha Pix na Getnet?',
    'Posso usar o Pix Getnet na Argentina?',
    'Como cancelar uma transferência Pix Getnet?',
    'Qual o prazo de validade do link de pagamento?',
    'Como contestar chargeback no Link de Pagamento?'
])
def test_related_documents_are_not_answerable_without_requested_evidence(query):
    assert HybridKnowledgeRetriever().retrieve_hybrid(query)['abstained']


@pytest.mark.parametrize('query,term',[
    ('Como cobrar um cliente que está longe da minha loja?', 'WhatsApp'),
    ('Can I request payment from a customer who is not physically at my store?', 'WhatsApp'),
    ('Can I get cash from unsettled card sales before their due date?', 'antecip'),
    ('Preciso adiantar o dinheiro de vendas no cartão ainda não vencidas.', 'antecip'),
])
def test_product_properties_generalize_without_scenario_answers(query,term):
    out=HybridKnowledgeRetriever().retrieve_hybrid(query)
    assert not out['abstained']
    assert term.lower() in ' '.join(x['content'] for x in out['results']).lower()
    assert all(x.get('answer_snippets') for x in out['results'])


def test_header_and_url_without_body_is_not_an_answer(tmp_path):
    (tmp_path/'product.md').write_text('---\nsource_url: https://site.getnet.com.br/pix/\ntitle: Pix\n---\n# Pix Getnet\nFonte oficial: https://site.getnet.com.br/pix/')
    out=HybridKnowledgeRetriever(data_dir=tmp_path,corpus_path=tmp_path/'c.json',vector_index_path=tmp_path/'v.json').retrieve_hybrid('Como funciona Pix?')
    assert out['abstained']


def test_short_new_topic_does_not_inherit_previous_product():
    ctx={'recent_messages':[{'role':'user','content':'O que é antecipação de recebíveis?'}]}
    assert KnowledgeAgent._enrich_query_with_memory('Como funciona Pix?',ctx)=='Como funciona Pix?'


def test_followup_consumes_session_entities_beyond_recent_window():
    ctx={'summary':'O cliente discutiu Crediário Getnet.', 'session_facts':{'last_product':'Crediário'}, 'entities':{'products':['Crediário']}, 'recent_messages':[]}
    out=KnowledgeAgent._enrich_query_with_memory('E em quantas parcelas?',ctx)
    assert 'Crediário' in out


def response(text,abstained=False):
    return ChatResponse(response=text,user_id='synthetic',agent_used='KNOWLEDGE_AGENT',abstained=abstained,sources=['https://site.getnet.com.br/crediario/'])


def test_evaluator_rejects_contradiction_despite_terms_and_url():
    case={'required_grounding_terms':['48x','2 dias úteis'], 'expected_url_substring':'crediario', 'reference_facts':['O Crediário permite até 48x e recebimento em até 2 dias úteis.']}
    result=GroundednessJudge.score_response(case,response('O Crediário NÃO permite 48x e não paga em 2 dias úteis.'))
    assert not result['passed']


def test_evaluator_abstention_flag_does_not_excuse_fabricated_claim():
    result=GroundednessJudge.score_response({'expect_abstained':True},response('Não encontrei a documentação. A taxa é 0,12% e o pagamento ocorre em 17 dias.',True))
    assert not result['passed']


def test_evaluator_rejects_unreferenced_number_despite_correct_terms():
    case={'required_grounding_terms':['48x'], 'expected_url_substring':'crediario','reference_facts':['O Crediário permite até 48x.']}
    assert not GroundednessJudge.score_response(case,response('O Crediário permite até 48x com taxa de 0,123%.'))['passed']


def test_currency_slot_never_substitutes_yen_with_euro():
    call=KnowledgeAgent()._plan_tool_calls('Qual a cotação do iene hoje?',True)[0]
    assert call.tool_name=='get_currency_exchange_rate'
    assert call.arguments['base_currency']=='JPY'


def test_lowercase_city_is_preserved_in_weather_request():
    call=KnowledgeAgent()._plan_tool_calls('O tempo em recife amanhã?',True)[0]
    assert call.arguments['city'].lower()=='recife'


def test_weather_date_cannot_be_relabelled_as_tomorrow():
    data={'city':'Recife','target_date':'2020-01-01','forecast_horizon':'Amanhã','min_temp_celsius':20,'max_temp_celsius':25,'rain_probability_percent':None}
    out=KnowledgeAgent()._synthesize_from_tool_results('',[ToolExecutionResult(tool_name='get_weather_forecast',success=True,data=data)],[])
    assert '2020-01-01' in out and 'amanhã' not in out.lower()


def test_eval_tool_precision_detects_dangerous_extra(monkeypatch,tmp_path):
    import importlib,json
    module=importlib.import_module('app.evals.evaluator')
    path=tmp_path/'suite.json'
    path.write_text(json.dumps({'cases':[{'case_id':'extra','category':'abstention','message':'synthetic','user_id':'synthetic','expected_agent':'KNOWLEDGE_AGENT','expected_tools':['search_getnet_knowledge'],'expect_abstained':True}]}))
    out=response('Não encontrei informações suficientes.',True)
    out.tools_called=['search_getnet_knowledge','execute_terminal_remote_action']
    monkeypatch.setattr(module.orchestrator,'process',lambda req:out)
    report=module.SystemEvaluator(path).run_suite()
    assert report['metrics']['mean_tool_precision']==0.5
    assert report['metrics']['safe_tool_selection_rate']==0
    assert report['metrics']['overall_pass_rate']==0


def test_portal_password_recovery_uses_the_requested_operation_not_pix():
    out=HybridKnowledgeRetriever().retrieve_hybrid('Como redefino minha senha do portal Minha Conta Getnet?')
    assert not out['abstained']
    answer=' '.join(snippet for chunk in out['results'] for snippet in chunk['answer_snippets'])
    assert 'Esqueci a senha' in answer
    assert 'Cadastrar chave' not in answer


def test_grounding_terms_match_complete_tokens_and_numbers():
    case={'required_grounding_terms':['48x'],'reference_facts':['O Crediário permite até 48x.']}
    assert not GroundednessJudge.score_response(case,response('O Crediário permite até 148x.'))['passed']


def test_route_names_are_exact_tokens_not_substrings(monkeypatch,tmp_path):
    import importlib,json
    module=importlib.import_module('app.evals.evaluator')
    path=tmp_path/'route.json'
    path.write_text(json.dumps({'cases':[{'case_id':'route','category':'abstention','message':'synthetic','user_id':'synthetic','expected_agent':'KNOWLEDGE_AGENT','expected_tools':[],'expect_abstained':True}]}))
    out=response('Não encontrei informações suficientes.',True)
    out.agent_used='UNKNOWN_KNOWLEDGE_AGENT_IMPERSONATION'
    monkeypatch.setattr(module.orchestrator,'process',lambda req:out)
    assert module.SystemEvaluator(path).run_suite()['metrics']['routing_accuracy']==0


def test_evaluation_gate_includes_overall_pass_rate(monkeypatch):
    import importlib
    module=importlib.import_module('app.evals.evaluator')
    report={'total_cases':1,'metrics':{'routing_accuracy':1.0,'mean_tool_recall':1.0,'mean_tool_precision':1.0,'safe_tool_selection_rate':1.0,'mean_groundedness_score':1.0,'overall_pass_rate':0.5}}
    monkeypatch.setattr(module.evaluator,'run_suite',lambda:report)
    assert module.main()==1


def test_typed_evidence_accepts_decimal_fields_with_their_correct_labels():
    data={'city':'Recife','target_date':'2026-10-24','min_temp_celsius':19.4,'max_temp_celsius':25.2,'rain_probability_percent':30,'condition':'Chuva'}
    result=ToolExecutionResult(tool_name='get_weather_forecast',success=True,data=data)
    text=KnowledgeAgent()._synthesize_from_tool_results('',[result],[])
    out=response(text)
    out.evidence=KnowledgeAgent._evidence(result)
    assert GroundednessJudge.score_response({},out)['passed']


def test_typed_evidence_compares_localized_money_as_numbers():
    out=response('Total de vendas: R$ 1,840.50.')
    out.evidence=[{'kind':'tool_fact','fact_path':'total_sales_amount','value':1840.5}]
    assert GroundednessJudge.score_response({},out)['passed']


def native_reply(*calls):
    from google.genai import types
    return SimpleNamespace(candidates=[SimpleNamespace(finish_reason=types.FinishReason.STOP,content=types.Content(role='model',parts=[types.Part(function_call=types.FunctionCall(name=name,args=args,id=f'knowledge-{i}')) for i,(name,args) in enumerate(calls)]))])


def native_provider(reply,requests):
    def generate_content(**kwargs):
        requests.append(kwargs)
        return reply
    return SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))


def test_native_knowledge_selection_uses_provider_args_for_novel_location(monkeypatch):
    monkeypatch.setattr(settings,'GEMINI_API_KEY','fake-never-sent')
    requests=[]
    agent=KnowledgeAgent(native_provider(native_reply(('get_weather_forecast',{'city':'Tromsø','days_ahead':2})),requests))
    calls=agent._plan_tool_calls('Verifique as condições no local combinado em Q17.',True)
    assert requests and calls[0].tool_name=='get_weather_forecast'
    assert calls[0].arguments=={'city':'Tromsø','days_ahead':2}
    declarations=requests[0]['config'].tools[0].function_declarations
    assert {d.name for d in declarations}==set(KnowledgeAgent.AVAILABLE_TOOLS)
    assert requests[0]['config'].automatic_function_calling.disable is True
    assert requests[0]['config'].http_options.timeout<=settings.LLM_TIMEOUT_SECONDS*1000


def test_native_knowledge_selection_uses_provider_currency_pair(monkeypatch):
    monkeypatch.setattr(settings,'GEMINI_API_KEY','fake-never-sent')
    requests=[]
    agent=KnowledgeAgent(native_provider(native_reply(('get_currency_exchange_rate',{'base_currency':'GBP','target_currency':'EUR'})),requests))
    calls=agent._plan_tool_calls('Referência financeira do item Q17.',True)
    assert calls[0].arguments=={'base_currency':'GBP','target_currency':'EUR'} and requests


@pytest.mark.parametrize('name,args',[
    ('execute_terminal_remote_action',{}),
    ('not_a_registered_tool',{}),
    ('get_weather_forecast',{'city':'Tromsø','days_ahead':99}),
    ('search_getnet_knowledge',{'query':'Pix','user_id':'other-tenant'}),
    ('get_currency_exchange_rate',{'base_currency':'EUR','undeclared':True}),
])
def test_native_knowledge_rejects_invalid_tools_or_arguments_without_fallback(monkeypatch,name,args):
    monkeypatch.setattr(settings,'GEMINI_API_KEY','fake-never-sent')
    agent=KnowledgeAgent(native_provider(native_reply((name,args)),[]))
    with pytest.raises(ValueError):
        agent._plan_tool_calls('Qual cotação do euro hoje?',True)


def test_configured_knowledge_provider_failure_abstains_without_lexical_fallback(monkeypatch):
    from app.security.auth import SecurityContext
    monkeypatch.setattr(settings,'GEMINI_API_KEY','fake-never-sent')
    out=KnowledgeAgent().run('Qual cotação do euro hoje?',SecurityContext(authenticated_user_id='synthetic'),True)
    assert out.abstained and not out.tools_called


def test_evaluator_accepts_policy_refusal_template_but_not_appended_facts():
    text=('Não foi possível processar esta solicitação: Solicitação bloqueada pelas políticas de segurança e integridade operacional da Getnet. '
          '(Código de auditoria: `SEC-INJ-001`). Sou o assistente virtual da Getnet e estou disponível para auxiliar com dúvidas sobre maquininhas, Pix, vendas e suporte técnico.')
    case={'expect_abstained':True,'category':'security_guardrail'}
    assert GroundednessJudge.score_response(case,response(text,True))['passed']
    assert not GroundednessJudge.score_response(case,response(text+' A taxa é 0,12%.',True))['passed']


@pytest.mark.parametrize('query',['A Get Smart é resistente à água?', 'Does Get Smart support biometric attendance control?', 'Qual a autonomia da bateria Get Smart?'])
def test_product_name_alone_cannot_answer_an_uncovered_property(query):
    assert HybridKnowledgeRetriever().retrieve_hybrid(query)['abstained']


def test_plural_followup_keeps_both_compared_products():
    ctx={'session_facts':{'last_product':'Get Smart','product_comparison':['Get Clássica','Get Smart']},'entities':{}}
    query=KnowledgeAgent._enrich_query_with_memory('E qual delas tem impressora?',ctx)
    assert 'Get Clássica' in query and 'Get Smart' in query


def test_other_product_followup_preserves_only_referenced_attribute():
    ctx={'session_facts':{'last_product':'Get Smart','product_comparison':['Get Clássica','Get Smart']},'entities':{},'recent_messages':[
        {'role':'user','content':'E qual delas tem Get Store?'},{'role':'assistant','content':'A Get Smart possui Get Store.'}]}
    query=KnowledgeAgent._enrich_query_with_memory('E a outra?',ctx)
    assert 'Get Clássica' in query and 'Get Store' in query


@pytest.mark.parametrize('query',['Can an Argentine merchant sign up for Get Conta?', 'A Get Conta aceita estabelecimento chileno?', 'Uma empresa não residente pode contratar Get Conta?'])
def test_foreign_eligibility_requires_explicit_jurisdiction_evidence(query):
    assert HybridKnowledgeRetriever().retrieve_hybrid(query)['abstained']


def test_simulation_does_not_prove_installment_limit_or_decision_authority(tmp_path):
    (tmp_path/'product.md').write_text('---\nsource_url: https://site.getnet.com.br/crediario/\ntitle: Crediário\n---\n## Simulação Crediário\nA simulação de parcelas do Crediário apresenta valores, encargos e taxa de desconto para consulta no aplicativo.')
    r=HybridKnowledgeRetriever(data_dir=tmp_path,corpus_path=tmp_path/'c.json',vector_index_path=tmp_path/'v.json')
    assert r.retrieve_hybrid('Qual é o limite de parcelas do Crediário?')['abstained']
    assert r.retrieve_hybrid('Quem define o número máximo de parcelas do Crediário?')['abstained']


def test_installment_limit_and_issuer_authority_are_supported_by_actual_body():
    r=HybridKnowledgeRetriever()
    out=r.retrieve_hybrid('Quem define o limite de parcelas do Crediário?')
    assert not out['abstained']
    assert 'emissor' in ' '.join(c['content'] for c in out['results']).lower()


def test_weather_stale_series_cannot_satisfy_requested_day(monkeypatch):
    mock_http(monkeypatch, weather={'daily':{'time':['2000-01-01','2000-01-02'],'temperature_2m_min':[19,20],'temperature_2m_max':[25,26]}})
    with pytest.raises((ValueError,RuntimeError)):
        search_tools.search_weather(days_ahead=1)


def test_dated_old_market_quote_is_not_live(monkeypatch):
    mock_http(monkeypatch, awesome={'EURBRL':{'bid':'6.1','create_date':'2000-01-01 12:00:00'}})
    data=search_tools.search_currency_rate()
    assert data['live'] is False
    assert data['rate_type']=='historical_market_quote'


def test_pix_capabilities_and_conditional_settlement_have_direct_primary_evidence():
    out=HybridKnowledgeRetriever().retrieve_hybrid('Como funciona o recebimento do Pix nas maquininhas Getnet?')
    assert not out['abstained']
    body=' '.join(c['content'] for c in out['results'])
    assert 'modelo' in body.lower() and 'dia corrido' in body.lower()
    assert any('/pix/' in c['source_url'] for c in out['results'])


def test_native_multiple_public_calls_preserve_all_actual_results(monkeypatch):
    from app.security.auth import SecurityContext
    from app.agents import knowledge_agent as module
    monkeypatch.setattr(settings,'GEMINI_API_KEY','fake-never-sent')
    replies=native_reply(('get_weather_forecast',{'city':'Recife','days_ahead':1}),('get_currency_exchange_rate',{'base_currency':'GBP','target_currency':'EUR'}))
    payloads={'get_weather_forecast':{'city':'Recife','target_date':'2026-10-24','min_temp_celsius':20,'max_temp_celsius':26},
              'get_currency_exchange_rate':{'currency_pair':'GBP/EUR','current_rate_brl':1.23,'updated_at':'2026-10-23','rate_type':'daily_reference','live':False}}
    monkeypatch.setattr(module.tool_registry,'execute',lambda tool_name,**kw:ToolExecutionResult(tool_name=tool_name,success=True,data=payloads[tool_name]))
    out=KnowledgeAgent(native_provider(replies,[])).run('Reúna as duas referências públicas.',SecurityContext(authenticated_user_id='synthetic'),True)
    assert 'Recife' in out.response and 'GBP/EUR' in out.response and '1.2300' in out.response
    assert {e['tool_name'] for e in out.evidence}==set(payloads)


def test_geographic_aliases_can_use_explicit_positive_evidence(tmp_path):
    (tmp_path/'product.md').write_text('---\nsource_url: https://site.getnet.com.br/produto-teste/\ntitle: Get Conta\n---\n## Elegibilidade da Get Conta\nA Get Conta permite abertura por lojistas no Reino Unido com cadastro aprovado.')
    r=HybridKnowledgeRetriever(data_dir=tmp_path,corpus_path=tmp_path/'c.json',vector_index_path=tmp_path/'v.json')
    assert not r.retrieve_hybrid('Can a British merchant open Get Conta?')['abstained']
