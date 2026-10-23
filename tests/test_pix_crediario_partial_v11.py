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

    # Only native model decisions are scripted. Retrieval, authorization, span
    # validation, citations and response composition all execute production code.
    import json
    from app.rag.semantic import content_hash
    class EvidenceProvider:
        models = None
        def __init__(self):
            self.models = self
        def generate_content(self, **kwargs):
            calls.append(kwargs)
            schema = kwargs["config"].response_json_schema
            if schema is None:
                return completion("search_getnet_knowledge", {"query": P06})
            payload = json.loads(kwargs["contents"][0].parts[0].text)
            if "units" in schema["properties"]:
                spans = [span for chunk in payload["candidates"] for span in chunk["spans"]
                         if "taxa única" in span["quote"] and "antecipação" in span["quote"]]
                assert spans, "production retrieval must return the Brazilian Crediário span"
                self.span = spans[0]
                response = {"request_hash": content_hash(P06), "units": {
                    brazil["unit_id"]: {"kind":"requirement", "decision":"supported", "subject":"Crediário Brasil",
                        "relation":"recebimento", "value":"", "unit":"", "qualifiers":[], "span_ids":[self.span["span_id"]]},
                    norway["unit_id"]: {"kind":"unresolved", "decision":"unknown", "subject":"", "relation":"",
                        "value":"", "unit":"", "qualifiers":[], "span_ids":[]}}}
            else:
                response = {"request_hash": content_hash(P06), "complete": False, "checks": {
                    u["unit_id"]: {"verdict":"accept", "issues":[], "affected_unit_ids":[]} for u in units}}
            return types.GenerateContentResponse(candidates=[types.Candidate(finish_reason="STOP",
                content=types.Content(role="model", parts=[types.Part.from_text(text=json.dumps(response))]))])
    provider = EvidenceProvider()
    orch.knowledge_agent.client = provider
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "synthetic-never-sent")
    out = orch.process(ChatRequest(message=P06, user_id="synthetic", session_id="crediario-v11"),
                       SecurityContext("synthetic"))

    assert "2 dias úteis" in out.response
    assert "taxa única" in out.response
    assert "sem custo de antecipação" in out.response
    assert "Noruega" in out.response
    assert out.answer_status == "partial"
    assert [(row["unit_id"], row["state"]) for row in out.request_coverage] == [
        (brazil["unit_id"], "fulfilled"), (norway["unit_id"], "unresolved")]
    assert [citation.chunk_id for citation in out.citations] == ["getnet_crediario_s1_9ea682"]
    assert len(calls) == 3
    assert out.tools_called == ["search_getnet_knowledge"]
    assert out.evidence
    assert provider.span["quote"] in out.response
    assert provider.span["quote"] in out.citations[0].excerpt
    assert all("Noruega" not in c.excerpt for c in out.citations)



@pytest.mark.parametrize("variant", ["extra", "dependent", "prefix_dependency", "intervening", "mixed"])
def test_country_coalescing_only_merges_exact_adjacent_independent_units(variant):
    message = P06 + ("; explique o Pix." if variant in {"extra", "intervening", "mixed"} else "")
    if variant == "dependent":
        message = P06.replace("Compare as condições", "Somente quando autorizado, compare as condições")
    if variant == "prefix_dependency":
        message = "Somente quando autorizado; " + P06
    units = request_units(message)
    steps = [PlanStep(step_order=i+1, target_agent="KNOWLEDGE_AGENT", sub_query=u["text"],
        rationale="unit", request_unit_ids=[u["unit_id"]]) for i,u in enumerate(units)]
    if variant == "intervening":
        steps = [steps[0], steps[2], steps[1]]
    if variant == "mixed":
        steps[0].request_unit_ids.append(units[2]["unit_id"])
        steps = steps[:2]
    plan = RoutingPlan(primary_agent="KNOWLEDGE_AGENT", confidence=.9, reasoning="scope", plan_steps=steps)
    result = MultiAgentOrchestrator._coalesce_country_knowledge_plan(plan, message)
    if variant == "extra":
        assert [s.request_unit_ids for s in result.plan_steps] == [["u1", "u2"], ["u3"]]
        assert "Pix" not in result.plan_steps[0].sub_query
    else:
        assert [s.model_dump() for s in result.plan_steps] == [s.model_dump() for s in steps]


class BatchProvider:
    def __init__(self, calls):
        self.calls = calls
        self.models = self
    def generate_content(self, **kwargs):
        return types.GenerateContentResponse(candidates=[types.Candidate(finish_reason="STOP",
            content=types.Content(role="model", parts=[types.Part(function_call=types.FunctionCall(
                name=name, args=args, id=f"call-{index}")) for index, (name, args) in enumerate(self.calls)]))])


@pytest.mark.parametrize("mixed", [False, True])
def test_verified_pix_correction_handles_native_batches_without_dropping_unrelated_calls(monkeypatch, mixed):
    from app.schemas import ToolExecutionResult
    from app.tools.registry import tool_registry
    pix = "Name the account types accepted for Pix in Brazil."
    external = "Find the current external rules for Pix account eligibility in Brazil."
    message = pix + (" " + external + " Find weather in Porto Alegre." if mixed else "")
    proposals = [("web_search_general", {"query": "Pix account categories in Brazil"}),
                 ("web_search_general", {"query": "Pix accepted account types"})]
    if mixed:
        proposals += [("web_search_general", {"query": external}),
                      ("get_weather_forecast", {"city": "Porto Alegre", "days_ahead": 0})]
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "synthetic-never-sent")
    executed = []
    def stop(**kwargs):
        executed.append((kwargs["tool_name"], kwargs["raw_arguments"]))
        return ToolExecutionResult(tool_name=kwargs["tool_name"], success=False, error_code="SYNTHETIC_STOP")
    monkeypatch.setattr(tool_registry, "execute", stop)
    out = KnowledgeAgent(BatchProvider(proposals)).run(message, SecurityContext("synthetic"))
    assert [name for name, _ in executed] == (["search_getnet_knowledge", "web_search_general", "get_weather_forecast"]
                                             if mixed else ["search_getnet_knowledge"])
    assert executed[0][1]["query"] == pix
    if mixed:
        assert executed[1][1]["query"] == external
        assert executed[2][1] == {"city": "Porto Alegre", "days_ahead": 0}
    assert out.tools_called == [name for name, _ in executed]


@pytest.mark.parametrize("message", [
    "Find the current external rules for Pix account eligibility in Brazil.",
    "Pesquise as regras externas atuais de elegibilidade de contas Pix no Brasil.",
    "Use an external source for Pix account eligibility in Brazil.",
])
def test_router_and_knowledge_preserve_the_same_explicit_external_pix_intent(monkeypatch, message):
    from app.agents.router_agent import RouterAgent
    from app.schemas import ToolExecutionResult
    from app.tools.registry import tool_registry
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "synthetic-never-sent")
    plan = RoutingPlan(primary_agent="WEB_SEARCH_AGENT", confidence=.9, reasoning="external",
        plan_steps=[PlanStep(step_order=1, target_agent="WEB_SEARCH_AGENT", sub_query=message,
            rationale="external", request_unit_ids=["u1"])])
    revised = RouterAgent._prefer_local_documentary_evidence(plan, message)
    assert revised.plan_steps[0].target_agent == "WEB_SEARCH_AGENT"
    executed = []
    def stop(**kwargs):
        executed.append(kwargs["tool_name"])
        return ToolExecutionResult(tool_name=kwargs["tool_name"], success=False, error_code="SYNTHETIC_STOP")
    monkeypatch.setattr(tool_registry, "execute", stop)
    KnowledgeAgent(BatchProvider([("web_search_general", {"query": message})])).run(message, SecurityContext("synthetic"))
    assert executed == ["web_search_general"]


def test_pix_correction_preserves_conjoined_documentary_qualifiers(monkeypatch):
    from app.schemas import ToolExecutionResult
    from app.tools.registry import tool_registry
    message = "Name the account types accepted for Pix and the eligibility restrictions in Brazil."
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "synthetic-never-sent")
    executed = []
    def stop(**kwargs):
        executed.append(kwargs["raw_arguments"]["query"])
        return ToolExecutionResult(tool_name=kwargs["tool_name"], success=False, error_code="SYNTHETIC_STOP")
    monkeypatch.setattr(tool_registry, "execute", stop)
    KnowledgeAgent(BatchProvider([("web_search_general", {"query": "Pix account categories"})])).run(
        message, SecurityContext("synthetic"))
    assert executed == [message]


def test_external_source_qualifier_survives_provider_query_omission_at_both_boundaries(monkeypatch):
    from app.agents.router_agent import RouterAgent
    from app.schemas import ToolExecutionResult
    from app.tools.registry import tool_registry
    message = "Name the account types accepted for Pix and use an external source."
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "synthetic-never-sent")
    plan = RoutingPlan(primary_agent="WEB_SEARCH_AGENT", confidence=.9, reasoning="external",
        plan_steps=[PlanStep(step_order=1, target_agent="WEB_SEARCH_AGENT", sub_query=message,
            rationale="external", request_unit_ids=["u1", "u2"])])
    assert all(step.target_agent == "WEB_SEARCH_AGENT" for step in
               RouterAgent._prefer_local_documentary_evidence(plan, message).plan_steps)
    executed = []
    def stop(**kwargs):
        executed.append(kwargs["tool_name"])
        return ToolExecutionResult(tool_name=kwargs["tool_name"], success=False, error_code="SYNTHETIC_STOP")
    monkeypatch.setattr(tool_registry, "execute", stop)
    KnowledgeAgent(BatchProvider([("web_search_general", {"query": "Pix account categories"})])).run(
        message, SecurityContext("synthetic"))
    assert executed == ["web_search_general"]


def test_pix_only_assignment_binds_vague_or_invented_external_queries_to_actual_product_intent(monkeypatch):
    from app.schemas import ToolExecutionResult
    from app.tools.registry import tool_registry
    message = "Name the account types accepted for Pix in Brazil."
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "synthetic-never-sent")
    executed = []
    def stop(**kwargs):
        executed.append((kwargs["tool_name"], kwargs["raw_arguments"]["query"]))
        return ToolExecutionResult(tool_name=kwargs["tool_name"], success=False, error_code="SYNTHETIC_STOP")
    monkeypatch.setattr(tool_registry, "execute", stop)
    KnowledgeAgent(BatchProvider([
        ("web_search_general", {"query": "Pix Brazil"}),
        ("web_search_general", {"query": "external rules on Pix account eligibility"}),
    ])).run(message, SecurityContext("synthetic"))
    assert executed == [("search_getnet_knowledge", message)]
