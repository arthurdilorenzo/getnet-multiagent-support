"""Typed public read predicates; no availability credit for a clean refusal."""
import pytest
from datetime import timedelta
from app.tools.search_tools import _today_local
TODAY = _today_local().isoformat()
TOMORROW = (_today_local()+timedelta(days=1)).isoformat()
from app.agents.knowledge_agent import KnowledgeAgent
from app.schemas import ToolCallSpec,ChatResponse
from app.evals.evaluator import GroundednessJudge


@pytest.mark.parametrize('text',[
    "What's the euro exchange rate today?", 'What is the EUR exchange rate now?',
    'What is the exchange rate for euro today?', 'Qual é a cotação do euro hoje?',
])
def test_currency_noun_orders_preserve_typed_pair(text):
    call=ToolCallSpec(tool_name='get_currency_exchange_rate',arguments={'base_currency':'EUR','target_currency':'BRL'})
    assert KnowledgeAgent._public_unit_covered(text,call,{'current_rate_brl':6.1,'updated_at':TODAY,'live':True,'currency_pair':'EUR/BRL'})


@pytest.mark.parametrize('text',[
    "What's the weather forecast in Porto Alegre tomorrow?", 'What is the weather in Porto Alegre tomorrow?',
    'Qual é a previsão do tempo em Porto Alegre amanhã?',
])
def test_compound_weather_predicate_needs_observed_city_and_fields(text):
    call=ToolCallSpec(tool_name='get_weather_forecast',arguments={'city':'Porto Alegre','days_ahead':1})
    assert KnowledgeAgent._public_unit_covered(text,call,{'target_date':TOMORROW,'city':'Porto Alegre, RS',
        'min_temp_celsius':15.,'max_temp_celsius':24.})


@pytest.mark.parametrize('text',[
    'What is the euro exchange rate today if certified?',
    'Do not check the euro exchange rate today.',
    'What is the euro exchange rate yesterday?',
    'What is the euro exchange rate today and its investment guarantee?',
])
def test_unknown_currency_relation_or_condition_cannot_gain_coverage(text):
    call=ToolCallSpec(tool_name='get_currency_exchange_rate',arguments={'base_currency':'EUR','target_currency':'BRL'})
    assert not KnowledgeAgent._public_unit_covered(text,call,{'current_rate_brl':6.1,'updated_at':TODAY,'live':True,'currency_pair':'EUR/BRL'})


@pytest.mark.parametrize('field,value', [('currency_pair','EUR/USD'),('current_rate_brl',None),
    ('current_rate_brl',float('nan')),('updated_at','')])
def test_currency_coverage_never_replaces_missing_or_different_fact(field,value):
    data={'current_rate_brl':6.1,'updated_at':TODAY,'live':True,'currency_pair':'EUR/BRL'};data[field]=value
    call=ToolCallSpec(tool_name='get_currency_exchange_rate',arguments={'base_currency':'EUR','target_currency':'BRL'})
    assert not KnowledgeAgent._public_unit_covered('What is the euro exchange rate today?',call,data)


def test_web_provider_refusal_is_safe_but_does_not_pass_requested_fulfillment():
    response=ChatResponse(response='Serviço externo indisponível. Tente novamente.',user_id='synthetic',
        session_id='synthetic',trace_id='synthetic',agent_used='WEB_SEARCH_AGENT',abstained=True,answer_status='abstained')
    case={'message':'What is the weather in Porto Alegre tomorrow?','category':'web_search','expect_abstained':False}
    result=GroundednessJudge.score_response(case,response)
    assert result['groundedness_score']==1.0
    assert result['passed'] is False and result['availability_passed'] is False


@pytest.mark.parametrize('updated,live', [('2001-01-01',True),(None,True),(TODAY,False)])
def test_today_rate_needs_fresh_observation_not_a_date_label(updated,live):
    call=ToolCallSpec(tool_name='get_currency_exchange_rate',arguments={'base_currency':'EUR','target_currency':'BRL'})
    assert not KnowledgeAgent._public_unit_covered('What is the exchange rate for euro today?',call,
        {'current_rate_brl':6.1,'updated_at':updated,'live':live,'currency_pair':'EUR/BRL'})


def test_weather_cannot_credit_a_different_observation_date():
    call=ToolCallSpec(tool_name='get_weather_forecast',arguments={'city':'Porto Alegre','days_ahead':1})
    assert not KnowledgeAgent._public_unit_covered('What is the weather in Porto Alegre tomorrow?',call,
        {'target_date':'2001-01-01','city':'Porto Alegre','min_temp_celsius':15.,'max_temp_celsius':24.})


@pytest.mark.parametrize('pair,expected,forbidden',[('EUR/BRL','R$','USD'),('EUR/USD','USD','R$')])
def test_currency_renderer_names_the_actual_target_unit(pair,expected,forbidden):
    from app.schemas import ToolExecutionResult
    text=KnowledgeAgent()._synthesize_from_tool_results('currency',[
        ToolExecutionResult(tool_name='get_currency_exchange_rate',success=True,data={
            'currency_pair':pair,'current_rate_brl':6.1234,'rate_type':'market_quote','updated_at':TODAY})],[])
    assert expected in text and forbidden not in text
    assert '6.1234' in text
