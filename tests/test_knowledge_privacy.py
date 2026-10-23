"""User PII is redacted at the optional knowledge synthesis provider boundary."""

from types import SimpleNamespace
from app.agents.knowledge_agent import KnowledgeAgent
from app.config import settings
from app.schemas import ToolExecutionResult


def test_knowledge_synthesis_redacts_user_pii_before_external_call(monkeypatch):
    requests = []
    def generate_content(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(text="Pix permite pagamentos digitais.")
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "test-key-not-sent")
    client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    KnowledgeAgent(client)._synthesize_from_tool_results(
        user_message="Como funciona Pix? CPF 123.456.789-01, senha: privatePass123",
        tool_results=[ToolExecutionResult(tool_name="search_getnet_knowledge", success=True,
                                         data={"chunks": [{"content": "Pix permite pagamentos digitais."}]})],
        citations=[],
    )
    assert len(requests) == 1
    assert "123.456.789-01" not in requests[0]["contents"]
    assert "privatePass123" not in requests[0]["contents"]
