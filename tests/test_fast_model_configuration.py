"""Fast model configuration applies to native tools and structured verification."""
from types import SimpleNamespace

import pytest
from google.genai import types

from app.agents.function_calling import generate_turn, initial_contents, native_config
from app.config import settings
from app.observability.resilience import breaker_registry
from app.observability.tracer import SystemTracer


@pytest.mark.parametrize("structured", [False, True], ids=["native-tools", "evidence-review"])
def test_flash_36_minimal_thinking_reaches_actual_generation_boundary(monkeypatch, structured):
    monkeypatch.setattr(settings, "GEMINI_MODEL", "gemini-3.6-flash")
    breaker_registry.reset_all()
    sent = []
    def generate(**kwargs):
        sent.append(kwargs)
        return types.GenerateContentResponse(candidates=[types.Candidate(finish_reason="STOP",
            content=types.Content(role="model", parts=[types.Part.from_text(text='{"ok":true}')]))])
    config = (types.GenerateContentConfig(response_mime_type="application/json",
                response_json_schema={"type":"object"}, max_output_tokens=2048)
              if structured else native_config([], "synthetic"))
    tracer = SystemTracer()
    try:
        generate_turn(SimpleNamespace(models=SimpleNamespace(generate_content=generate)),
                      initial_contents("synthetic", None), config, tracer,
                      phase="semantic_verification" if structured else "routing")
        assert len(sent) == 1
        assert sent[0]["model"] == "gemini-3.6-flash"
        assert sent[0]["config"].thinking_config is not None
        assert sent[0]["config"].thinking_config.thinking_level == types.ThinkingLevel.MINIMAL
        assert sent[0]["config"].http_options.retry_options.attempts == 1
        assert tracer.steps[-1].details["model"] == "gemini-3.6-flash"
    finally:
        breaker_registry.reset_all()
