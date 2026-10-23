"""Pix evidence routing and independent Brazil/Norway coverage."""
import importlib
import pytest
from google.genai import types

from app.agents.knowledge_agent import KnowledgeAgent
from app.agents.orchestrator import MultiAgentOrchestrator
from app.config import settings
from app.memory.store import StateStore
from app.schemas import ChatRequest, PlanStep, RoutingPlan, SpecialistOutput
from app.security.auth import SecurityContext
from app.security.request_units import request_units


P06 = ("Compare as condições publicadas para recebimento do Crediário no Brasil — prazo, taxa, "
       "custo de antecipação — com o que a documentação informa sobre os mesmos termos na Noruega.")


def test_brazil_norway_comparison_becomes_country_scoped_units():
    units = request_units(P06)
    assert len(units) == 2
    assert "Brasil" in units[0]["text"]
    assert "Noruega" not in units[0]["text"]
    assert "Noruega" in units[1]["text"]
    assert "Brasil" not in units[1]["text"]
    assert "".join(unit["text"] for unit in units) == P06


def completion(tool_name, arguments):
    return types.GenerateContentResponse(candidates=[types.Candidate(finish_reason="STOP",
        content=types.Content(role="model", parts=[types.Part(function_call=types.FunctionCall(
            name=tool_name, args=arguments))]))])


class Provider:
    def __init__(self, tool_name):
        self.tool_name = tool_name
        self.models = self

    def generate_content(self, **kwargs):
        return completion(self.tool_name, {"query": "Pix account categories in Brazil", "max_results": 3})


def test_getnet_pix_account_eligibility_overrides_native_web_choice_only_with_verified_local_evidence(monkeypatch):
    from app.security.auth import SecurityContext
    from app.schemas import ToolExecutionResult
    from app.tools import registry as registry_module
    message = "Name the account types accepted for Pix in Brazil."
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "synthetic-never-sent")
    agent = KnowledgeAgent(Provider("web_search_general"))
    executed = []
    def fail_closed_execute(**kwargs):
        executed.append(kwargs["tool_name"])
        return ToolExecutionResult(tool_name=kwargs["tool_name"], success=False,
                                   error_code="SYNTHETIC_STOP", error_message="stop after selection")
    monkeypatch.setattr(registry_module.tool_registry, "execute", fail_closed_execute)
    out = agent.run(message, SecurityContext("synthetic"))
    assert executed == ["search_getnet_knowledge"]
    assert out.tools_called == ["search_getnet_knowledge"]


def test_explicit_banco_central_pix_regulation_remains_web(monkeypatch):
    from app.security.auth import SecurityContext
    from app.schemas import ToolExecutionResult
    from app.tools import registry as registry_module
    message = "Find the current Banco Central regulation on Pix account eligibility in Brazil."
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "synthetic-never-sent")
    agent = KnowledgeAgent(Provider("web_search_general"))
    executed = []
    def fail_closed_execute(**kwargs):
        executed.append(kwargs["tool_name"])
        return ToolExecutionResult(tool_name=kwargs["tool_name"], success=False,
                                   error_code="SYNTHETIC_STOP", error_message="stop after selection")
    monkeypatch.setattr(registry_module.tool_registry, "execute", fail_closed_execute)
    out = agent.run(message, SecurityContext("synthetic"))
    assert executed == ["web_search_general"]
    assert out.tools_called == ["web_search_general"]


def test_partial_brazil_facts_survive_unsupported_norway_without_replan(monkeypatch, tmp_path):
    module = importlib.import_module("app.agents.orchestrator")
    monkeypatch.setattr(module, "state_store", StateStore(tmp_path / "crediario-v11.sqlite"))
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "")
    monkeypatch.setattr(settings, "AUTH_ENFORCE_STRICT", False)
    units = request_units(P06)
    brazil, norway = units
    plan = RoutingPlan(primary_agent="KNOWLEDGE_AGENT", confidence=0.9, reasoning="independent country clauses",
        plan_steps=[
            PlanStep(step_order=1, target_agent="KNOWLEDGE_AGENT", sub_query=brazil["text"],
                rationale="Brazil documentary terms", request_unit_ids=[brazil["unit_id"]]),
            PlanStep(step_order=2, target_agent="KNOWLEDGE_AGENT", sub_query=norway["text"],
                rationale="Norway documentary coverage", request_unit_ids=[norway["unit_id"]]),
        ])
    orch = MultiAgentOrchestrator()
    monkeypatch.setattr(orch.router, "plan", lambda **kwargs: plan)
    monkeypatch.setattr(orch.router, "replan", lambda **kwargs: pytest.fail("unsupported Norway needs no gainless replan"))
    calls = []

    def partial(**kwargs):
        calls.append(kwargs["target_agent"])
        assert kwargs["target_agent"] == "KNOWLEDGE_AGENT"
        return SpecialistOutput(agent_name="KNOWLEDGE_AGENT",
            response=("Brasil: recebimento em até dois dias úteis, taxa única e sem custo de antecipação. "
                      "A documentação consultada não informa os termos para a Noruega."),
            semantic_mode="native_semantic", answer_status="partial", abstained=True,
            request_unit_ids=[brazil["unit_id"], norway["unit_id"]],
            covered_unit_ids=[brazil["unit_id"]], unresolved_unit_ids=[norway["unit_id"]],
            citations=[{"chunk_id": "getnet_crediario_s1_9ea682", "title": "Crediário", "source": "Getnet",
                        "source_url": "https://site.getnet.com.br/crediario/", "relevance_score": 0.9,
                        "excerpt": "recebe em até 2 dias úteis, com taxa única e sem custo de antecipação"}],
            evidence=[{"chunk_id": "getnet_crediario_s1_9ea682", "country": "Brazil"}],
            structured_data={"unsupported_units": [norway["unit_id"]]})

    monkeypatch.setattr(orch, "_delegate_to_specialist", partial)
    out = orch.process(ChatRequest(message=P06, user_id="synthetic", session_id="crediario-v11"),
                       SecurityContext("synthetic"))

    assert "até dois dias úteis" in out.response
    assert "taxa única" in out.response
    assert "sem custo de antecipação" in out.response
    assert "não informa os termos para a Noruega" in out.response
    assert out.answer_status == "partial"
    assert [(row["unit_id"], row["state"]) for row in out.request_coverage] == [
        (brazil["unit_id"], "fulfilled"), (norway["unit_id"], "unresolved")]
    assert [citation.chunk_id for citation in out.citations] == ["getnet_crediario_s1_9ea682"]
    assert calls == ["KNOWLEDGE_AGENT"]

