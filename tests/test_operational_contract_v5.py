"""Synthetic compositional operational cases; no reserved evaluation fixtures."""
from types import SimpleNamespace
import pytest
from google.genai import types
from app.agents.customer_support_agent import CustomerSupportAgent
from app.agents.router_agent import RouterAgent
from app.memory.store import StateStore
from app.security.auth import SecurityContext
from app.config import Settings, settings
from app.observability.tracer import SystemTracer
from app.agents.function_calling import native_config, initial_contents, generate_turn

CTX = SecurityContext(authenticated_user_id='cliente1988')


def test_compound_terminal_persists_across_compaction_and_neutral_agent(tmp_path):
    store = StateStore(tmp_path/'memory.sqlite')
    store.append_message('compound', 'cliente1988', 'user', 'Inspect GET-DEVICE-REGION-731 and TXN-SYNTH-992')
    for i in range(12):
        store.append_message('compound', 'cliente1988', 'user', f'Thank you {i}')
        store.append_message('compound', 'cliente1988', 'assistant', 'Acknowledged', agent_used='KNOWLEDGE_AGENT')
    context = store.get_conversation_context('compound', 'cliente1988')
    assert context['entities']['terminal_id'] == 'GET-DEVICE-REGION-731'
    assert context['entities']['transaction_id'] == 'TXN-SYNTH-992'
    assert context['rolling_summary']
    plan = RouterAgent().plan('Check the same terminal firmware.', 'cliente1988', context)
    assert plan.primary_agent == 'CUSTOMER_SUPPORT_AGENT'
    out = CustomerSupportAgent().run('Check the same terminal firmware.', CTX, conversation_context=context)
    assert out.structured_data['diagnose_terminal_issue']['requested_terminal_id'] == 'GET-DEVICE-REGION-731'


def test_compound_tool_referent_uses_shared_pattern(tmp_path):
    store = StateStore(tmp_path/'tools.sqlite')
    store.ensure_session('tool', 'cliente1988')
    store.remember_tool_referents('tool', 'cliente1988', [{'success':True,'tool_name':'diagnose_terminal_issue', 'data':{'terminal':{'terminal_id':'GET-DEVICE-WEST-731'}}}])
    assert store.get_conversation_context('tool','cliente1988')['entities']['terminal_id'] == 'GET-DEVICE-WEST-731'


@pytest.mark.parametrize('message', [
    'Consulte o valor líquido em 17 de setembro de 2026.',
    'Check our net payout on September 17, 2026.',
    'How much were our proceeds on 2026-09-17?',
])
def test_dated_payout_classification_is_shared_by_router_and_support(message):
    plan = RouterAgent().plan(message, 'cliente1988')
    assert plan.primary_agent == 'CUSTOMER_SUPPORT_AGENT'
    out = CustomerSupportAgent().run(message, CTX)
    assert out.tools_called == ['get_sales_settlement']
    data = out.structured_data['get_sales_settlement']
    assert data['requested_sales_date'] == '2026-09-17'
    assert not data['found']


@pytest.mark.parametrize('message', [
    'Como funciona o repasse de vendas da Getnet?',
    'How does settlement of sales work with Getnet?',
    'Preciso de conta bancária para receber minhas vendas via Pix?',
])
def test_commercial_question_does_not_become_private_snapshot(message):
    assert RouterAgent().plan(message, 'cliente1988').primary_agent == 'KNOWLEDGE_AGENT'
    out = CustomerSupportAgent().run(message, CTX)
    assert not out.tools_called
    assert out.abstained


def test_missing_transaction_is_identified_as_transaction_in_response():
    out = CustomerSupportAgent().run('Inspect TXN-SYNTH-9191', CTX)
    assert 'TXN-SYNTH-9191' in out.response
    assert 'TRANSACTION_NOT_FOUND' in out.response
    assert 'terminal solicitado `não informado`' not in out.response


def test_sdk_timeout_default_validated_without_env_file():
    assert Settings(_env_file=None, GEMINI_API_KEY='').LLM_TIMEOUT_SECONDS == 30
    for value in (0, -1, float('inf'), 61):
        with pytest.raises(ValueError):
            Settings(_env_file=None, GEMINI_API_KEY='', LLM_TIMEOUT_SECONDS=value)


@pytest.mark.parametrize('model,supported', [('gemini-3.1-flash-lite',True),('gemini-2.5-flash',False),('gemini-3.1-pro',False)])
def test_minimal_thinking_is_explicit_only_for_supported_model(monkeypatch, model, supported):
    monkeypatch.setattr(settings,'GEMINI_MODEL',model)
    config = native_config([], 'system')
    assert bool(config.thinking_config) == supported
    if supported: assert config.thinking_config.thinking_level == types.ThinkingLevel.MINIMAL


def test_generation_instruments_safe_metadata_and_thought_tokens(monkeypatch):
    captured=[]
    response = types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(role='model',parts=[types.Part.from_text(text='ok')]),finish_reason='STOP')],usage_metadata=types.GenerateContentResponseUsageMetadata(prompt_token_count=5,candidates_token_count=2,thoughts_token_count=7))
    client=SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kw: captured.append(kw) or response))
    tracer=SystemTracer()
    monkeypatch.setattr(tracer,'get_total_execution_time_ms',lambda:45000)
    generate_turn(client,initial_contents('sensitive input',None),native_config([],'system'),tracer,phase='semantic_analysis')
    assert captured[0]['config'].http_options.timeout == 15000
    assert captured[0]['config'].http_options.retry_options.attempts == 1
    step=next(s for s in tracer.steps if s.action=='provider_generation')
    assert step.details['phase']=='semantic_analysis'
    assert step.details['thoughts_token_count']==7
    assert step.details['finish_reason']=='STOP'
    assert step.details['duration_ms']>=0
    assert 'sensitive input' not in str(step.details)
    assert tracer.get_token_usage().completion_tokens == 9


def test_separate_documentary_and_private_clauses_keep_both_intents():
    message='Explique como funciona o repasse; consulte meu valor líquido em 17 de setembro de 2026.'
    from app.security.operational_intent import classify_operational_intent
    intent=classify_operational_intent(message)
    assert intent.documentary_scope and intent.private_scope
    assert intent.intents == ('settlement',)
    agents={s.target_agent for s in RouterAgent().plan(message,'cliente1988').plan_steps}
    assert agents == {'KNOWLEDGE_AGENT','CUSTOMER_SUPPORT_AGENT'}


def test_offline_multiple_terminal_targets_requires_clarification():
    out=CustomerSupportAgent().run('Inspect GET-DEVICE-WEST-731 and GET-DEVICE-EAST-732',CTX)
    assert out.abstained
    assert not out.structured_data
    assert out.tool_results[0].error_code=='REQUEST_NEEDS_CLARIFICATION'


def test_multiple_referents_do_not_persist_an_arbitrary_last_id(tmp_path):
    store=StateStore(tmp_path/'multiple.sqlite')
    store.append_message('multi','cliente1988','user','Check GET-DEVICE-WEST-731')
    store.append_message('multi','cliente1988','user','Compare GET-DEVICE-WEST-731 and GET-DEVICE-EAST-732')
    assert 'terminal_id' not in store.get_conversation_context('multi','cliente1988')['entities']


def test_failed_generation_reports_phase_duration_without_payload(monkeypatch):
    def fail(**kwargs): raise RuntimeError('secret-content-not-to-log')
    client=SimpleNamespace(models=SimpleNamespace(generate_content=fail))
    tracer=SystemTracer()
    from app.observability.resilience import breaker_registry
    breaker_registry.reset_all()
    with pytest.raises(Exception):
        generate_turn(client, initial_contents('secret-prompt-not-to-log',None),native_config([],'secret-system'),tracer,phase='support')
    step=next(s for s in tracer.steps if s.action=='provider_generation')
    assert step.status=='error'
    assert step.details['phase']=='support'
    assert step.details['error_type']=='ExternalServiceUnavailableError'
    assert step.details['provider_error_type']=='RuntimeError'
    assert step.details['provider_error_code'] is None
    assert 'secret-' not in str(step.details)
    breaker_registry.reset_all()
