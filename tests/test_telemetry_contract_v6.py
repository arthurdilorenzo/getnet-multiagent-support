from app.observability.metrics import MetricsCollector
from app.schemas import ChatResponse, PlanStep, SpecialistOutput, TokenUsage


def test_delegation_and_coverage_defaults_do_not_share_mutable_state():
    step = PlanStep(step_order=1, target_agent='KNOWLEDGE_AGENT', sub_query='Pix', rationale='Documentação')
    other = step.model_copy(deep=True)
    step.request_unit_ids.append('u1')
    assert other.request_unit_ids == []
    out = SpecialistOutput(agent_name='KNOWLEDGE_AGENT', response='Pix',
                           request_unit_ids=['u1', 'u2'], covered_unit_ids=['u1'],
                           unresolved_unit_ids=['u2'])
    assert out.model_dump()['unresolved_unit_ids'] == ['u2']
    reply = ChatResponse(response='Pix', user_id='synthetic', agent_used=out.agent_name,
                         request_coverage=[{'unit_id': 'u2', 'status': 'unresolved'}])
    assert reply.model_dump()['request_coverage'][0]['status'] == 'unresolved'


def test_metrics_separate_transport_outcome_from_declared_completion():
    collector = MetricsCollector()
    common = dict(agent_used='KNOWLEDGE_AGENT', tools_called=[], latency_ms=10,
                  token_usage=TokenUsage())
    collector.record_request(**common, semantic_mode='native_semantic', answer_status='partial')
    collector.record_request(**common, semantic_mode='limited_offline', answer_status='abstained', abstained=True)
    collector.record_request(**common, semantic_mode='native_semantic', answer_status='complete', failed=True)
    collector.record_request(**common)
    data = collector.snapshot()
    assert data['requests']['successful'] == 3
    grouped = data['requests_by_mode_and_status']
    assert grouped['native_semantic']['partial'] == {'total': 1, 'successful': 1, 'failed': 0, 'abstained': 0}
    assert grouped['native_semantic']['complete']['failed'] == 1
    assert grouped['limited_offline']['abstained']['abstained'] == 1
    assert grouped['unclassified']['unclassified']['total'] == 1
    assert 'declarados' in data['completion_metadata_limitations']


def test_metrics_do_not_create_unbounded_labels_from_model_text():
    collector = MetricsCollector()
    collector.record_request('KNOWLEDGE_AGENT', [], 1, TokenUsage(),
                             semantic_mode='private-user-text', answer_status='invented')
    grouped = collector.snapshot()['requests_by_mode_and_status']
    assert set(grouped) == {'unclassified'}
    assert set(grouped['unclassified']) == {'unclassified'}


def test_generation_records_payload_sizes_without_contents():
    from types import SimpleNamespace
    from google.genai import types
    from app.agents.function_calling import generate_turn, initial_contents, native_config
    from app.observability.tracer import SystemTracer
    response = types.GenerateContentResponse(candidates=[types.Candidate(
        finish_reason='STOP', content=types.Content(role='model', parts=[types.Part.from_text(text='ok')]))])
    client = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kwargs: response))
    tracer = SystemTracer()
    config = native_config([], 'private instruction')
    config.response_json_schema = {'type': 'object', 'properties': {'decision': {'type': 'string'}}}
    generate_turn(client, initial_contents('private request', None), config, tracer, phase='semantic_analysis')
    details = next(step.details for step in tracer.steps if step.action == 'provider_generation')
    assert details['request_bytes'] > 0
    assert details['response_schema_bytes'] > 0
    assert details['estimated_prompt_tokens'] > 0
    assert details['tokens_consumed_before_call'] == 0
    assert 'private' not in str(details)


def test_budget_includes_schema_and_instruction_before_network(monkeypatch):
    import pytest
    from types import SimpleNamespace
    from app.agents.function_calling import generate_turn, initial_contents, native_config, ProviderTurnError
    from app.config import settings
    calls = []
    client = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kw: calls.append(kw)))
    monkeypatch.setattr(settings, 'LLM_REQUEST_TOKEN_BUDGET', settings.LLM_MAX_OUTPUT_TOKENS + 200)
    config = native_config([], 'x' * 4000)
    config.response_json_schema = {'description': 'y' * 4000, 'type': 'object'}
    with pytest.raises(ProviderTurnError) as error:
        generate_turn(client, initial_contents('short', None), config)
    assert error.value.code == 'TOKEN_BUDGET'
    assert not calls


def test_finish_rejection_retains_safe_reason_and_counts_consumed_usage():
    import pytest
    from types import SimpleNamespace
    from google.genai import types
    from app.agents.function_calling import generate_turn, initial_contents, native_config, ProviderTurnError
    from app.observability.tracer import SystemTracer
    response = types.GenerateContentResponse(candidates=[types.Candidate(finish_reason='MAX_TOKENS')],
        usage_metadata=types.GenerateContentResponseUsageMetadata(prompt_token_count=3, candidates_token_count=2))
    client = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kwargs: response))
    tracer = SystemTracer()
    with pytest.raises(ProviderTurnError) as error:
        generate_turn(client, initial_contents('private request', None), native_config([], 'system'), tracer)
    assert error.value.code == 'SDK_FINISH_REASON'
    assert tracer.get_token_usage().total_tokens == 5
    details = next(step.details for step in tracer.steps if step.action == 'provider_generation')
    assert details['turn_error_code'] == 'SDK_FINISH_REASON'
    assert 'private request' not in str(details)
