from unittest.mock import Mock
import pytest
from google.genai import types

from app.observability.tracer import SystemTracer
from app.v2.contracts import Plan, PlannedRequirement
from app.v2.model import GeminiModel


def test_native_adapter_uses_real_sdk_protocol_with_injected_client_no_network():
    client = Mock()
    client.models.generate_content.return_value = types.GenerateContentResponse(candidates=[
        types.Candidate(finish_reason=types.FinishReason.STOP, content=types.Content(role="model", parts=[
            types.Part.from_text(text='{"requirements":[{"requirement_id":"req_1","requirement_summary":"produto"}],"steps":[{"kind":"knowledge","query":"produto"}],"entities":{}}')]))])
    response = GeminiModel(client).generate(Plan, {"original_request": "Produto?"},
        phase="routing", tracer=SystemTracer())
    assert response.steps[0].query == "produto"
    assert response.requirements[0].requirement_id == "req_1"
    config = client.models.generate_content.call_args.kwargs["config"]
    assert config.response_mime_type == "application/json"
    assert config.http_options.retry_options.attempts == 1


def test_provider_invalid_json_raises_validation_error():
    client = Mock()
    client.models.generate_content.return_value = types.GenerateContentResponse(candidates=[
        types.Candidate(finish_reason=types.FinishReason.STOP, content=types.Content(role="model", parts=[
            types.Part.from_text(text='{"broken_json": true}')]))])
    with pytest.raises(Exception):
        GeminiModel(client).generate(Plan, {"original_request": "Produto?"}, phase="routing", tracer=SystemTracer())
