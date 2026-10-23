import json
from types import SimpleNamespace

import pytest
from google.genai import types

from app.agents.function_calling import generate_turn, initial_contents, native_config
from app.config import settings
from app.observability.tracer import SystemTracer


def test_summary_facts_and_entities_reach_provider_as_untrusted_data():
    context = {"rolling_summary": "merchant uses Smart", "session_facts": {"product": "Smart"},
               "entities": {"terminal_id": "GET-SMART-1"}, "recent_messages": []}
    serialized = json.dumps([c.model_dump(mode="json") for c in initial_contents("And this one?", context)])
    assert "merchant uses Smart" in serialized
    assert "GET-SMART-1" in serialized
    assert "dados" in serialized.lower()


def test_model_config_bounds_output_tokens():
    config = native_config([], "system")
    assert 0 < config.max_output_tokens <= settings.LLM_MAX_OUTPUT_TOKENS


def test_request_deadline_prevents_provider_call(monkeypatch):
    called = []
    client = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kw: called.append(kw)))
    tracer = SystemTracer()
    monkeypatch.setattr(tracer, "get_total_execution_time_ms", lambda: 9999999)
    with pytest.raises(TimeoutError):
        generate_turn(client, initial_contents("hello", None), native_config([], "system"), tracer)
    assert not called


def test_token_budget_prevents_provider_call(monkeypatch):
    called = []
    client = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kw: called.append(kw)))
    monkeypatch.setattr(settings, "LLM_REQUEST_TOKEN_BUDGET", 10)
    with pytest.raises(ValueError, match="budget"):
        generate_turn(client, initial_contents("x" * 1000, None), native_config([], "system"), SystemTracer())
    assert not called
