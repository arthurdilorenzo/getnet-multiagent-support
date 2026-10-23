"""V9 provider boundaries; no network, payload body, or fallback involved."""
from types import SimpleNamespace
import pytest
from google.genai import types
from app.agents.function_calling import generate_turn, initial_contents, native_config, ProviderTurnError
from app.config import settings, Settings
from app.observability.resilience import breaker_registry
from app.observability.tracer import SystemTracer

@pytest.fixture(autouse=True)
def clean_breakers():
    breaker_registry.reset_all()
    yield
    breaker_registry.reset_all()

def details(tracer):return next(s.details for s in tracer.steps if s.action=='provider_generation')
def client_for(fn):return SimpleNamespace(models=SimpleNamespace(generate_content=fn))
def stopped():return types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(role='model',parts=[types.Part.from_text(text='synthetic')]),finish_reason='STOP')])

def test_local_open_breaker_is_distinct_from_external_failure():
    breaker=breaker_registry.get('gemini_api')
    for _ in range(breaker.failure_threshold):breaker.record_failure()
    def unexpected(**kw):pytest.fail('open breaker must not contact SDK')
    tracer=SystemTracer()
    with pytest.raises(Exception):generate_turn(client_for(unexpected),initial_contents('synthetic',None),native_config([],'synthetic'),tracer)
    d=details(tracer)
    assert d['failure_stage']=='local_breaker'
    assert d['provider_error_type'] is None and d['provider_error_code'] is None
    assert d['call_started'] is False

@pytest.mark.parametrize('status',[429,503,504])
def test_external_status_is_safe_and_not_protocol_or_breaker(status):
    class ProviderFailure(Exception):code=status
    calls=[]
    def fail(**kw):
        calls.append(kw)
        raise ProviderFailure('credentials and provider body must not be retained')
    tracer=SystemTracer()
    with pytest.raises(Exception):generate_turn(client_for(fail),initial_contents('private synthetic prompt',None),native_config([],'private synthetic system'),tracer)
    d=details(tracer)
    assert d['failure_stage']=='provider' and d['provider_error_code']==status
    assert d['turn_error_code'] is None
    assert len(calls)==1
    assert 'private' not in str(d) and 'credentials' not in str(d)

def test_protocol_error_is_not_attributed_to_external_provider_status():
    tracer=SystemTracer()
    with pytest.raises(ProviderTurnError):generate_turn(client_for(lambda **kw:types.GenerateContentResponse(candidates=[])),initial_contents('synthetic',None),native_config([],'synthetic'),tracer)
    d=details(tracer)
    assert d['failure_stage']=='protocol' and d['turn_error_code']=='SDK_CANDIDATE_COUNT'
    assert d['provider_error_type'] is None and d['provider_error_code'] is None

@pytest.mark.parametrize('model,expected',[('gemini-3.5-flash-lite','MINIMAL'),('gemini-3.1-flash-lite','MINIMAL'),('gemini-2.5-flash-lite',None),('gemini-3-pro',None),('unknown-alias',None)])
def test_minimal_thinking_is_allowlisted_by_supported_family(monkeypatch,model,expected):
    monkeypatch.setattr(settings,'GEMINI_MODEL',model)
    config=native_config([],'synthetic')
    actual=config.thinking_config.thinking_level.value if config.thinking_config else None
    assert actual==expected

def test_turn_preserves_deadline_output_tokens_and_zero_retries(monkeypatch):
    monkeypatch.setattr(settings,'GEMINI_MODEL','gemini-3.1-flash-lite')
    tracer=SystemTracer();tracer._start_time-=40
    observed={}
    def success(**kw):observed.update(kw);return stopped()
    generate_turn(client_for(success),initial_contents('synthetic',None),native_config([],'synthetic'),tracer)
    config=observed['config']
    assert 0 < config.http_options.timeout <= 20000
    assert config.http_options.retry_options.attempts==1
    assert config.max_output_tokens==2048
    assert settings.LLM_REQUEST_TOKEN_BUDGET==32000

def test_token_budget_preflight_does_not_contact_provider(monkeypatch):
    monkeypatch.setattr(settings,'LLM_REQUEST_TOKEN_BUDGET',2048)
    def unexpected(**kw):pytest.fail('budget failure must not contact SDK')
    tracer=SystemTracer()
    with pytest.raises(ProviderTurnError):generate_turn(client_for(unexpected),initial_contents('synthetic',None),native_config([],'synthetic'),tracer)
    d=details(tracer)
    assert d['failure_stage']=='preflight' and d['turn_error_code']=='TOKEN_BUDGET'
    assert d['call_started'] is False and d['provider_error_code'] is None

def test_explicit_model_override_is_preserved():
    assert Settings(_env_file=None,GEMINI_MODEL='explicit-synthetic-override').GEMINI_MODEL=='explicit-synthetic-override'


def test_expired_request_deadline_does_not_contact_provider():
    tracer=SystemTracer();tracer._start_time-=61
    def unexpected(**kw):pytest.fail('expired request must not contact SDK')
    with pytest.raises(TimeoutError):generate_turn(client_for(unexpected),initial_contents('synthetic',None),native_config([],'synthetic'),tracer)
    d=details(tracer)
    assert d['failure_stage']=='preflight' and d['turn_error_code']=='DEADLINE'
    assert d['call_started'] is False and d['provider_error_type'] is None


def test_default_model_matches_completed_preregistered_diagnostic(monkeypatch):
    monkeypatch.delenv('GEMINI_MODEL',raising=False)
    # _env_file=None excludes the workstation's old explicit model override.
    assert Settings(_env_file=None).GEMINI_MODEL=='gemini-3.5-flash-lite'
