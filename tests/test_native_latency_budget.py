"""Native latency fixes retain provider deadlines and both evidence passes."""
import json
from types import SimpleNamespace
from collections import deque

import pytest
from google.genai import types

from app.agents.function_calling import generate_turn, native_config, initial_contents
from app.agents.knowledge_agent import KnowledgeAgent
from app.config import settings
from app.observability.resilience import breaker_registry
from app.observability.tracer import SystemTracer
from app.security.auth import SecurityContext
from app.security.request_units import request_units, validate_scope


@pytest.fixture(autouse=True)
def reset_breaker():
    breaker_registry.reset_all()
    yield
    breaker_registry.reset_all()


def stopped(text="synthetic"):
    return types.GenerateContentResponse(candidates=[types.Candidate(
        content=types.Content(role="model", parts=[types.Part.from_text(text=text)]), finish_reason="STOP")])


@pytest.mark.parametrize("remaining", [0.1, 7.0, 9.999])
def test_insufficient_provider_deadline_never_sends_invalid_request(monkeypatch, remaining):
    tracer = SystemTracer()
    monkeypatch.setattr(tracer, "get_total_execution_time_ms", lambda: (60-remaining)*1000)
    sent = []
    def generate(**kwargs):
        sent.append(kwargs)
        return stopped()
    with pytest.raises(TimeoutError):
        generate_turn(SimpleNamespace(models=SimpleNamespace(generate_content=generate)),
                      initial_contents("synthetic", None), native_config([], "synthetic"), tracer,
                      phase="replanning")
    assert not sent
    details = tracer.steps[-1].details
    assert details["failure_stage"] == "preflight"
    assert details["turn_error_code"] == "DEADLINE"
    assert details["call_started"] is False


def test_ten_seconds_is_allowed_without_increasing_remaining_budget(monkeypatch):
    tracer = SystemTracer()
    monkeypatch.setattr(tracer, "get_total_execution_time_ms", lambda: 50000)
    sent = []
    def generate(**kwargs):
        sent.append(kwargs["config"].http_options.timeout)
        return stopped()
    generate_turn(SimpleNamespace(models=SimpleNamespace(generate_content=generate)),
                  initial_contents("synthetic", None), native_config([], "synthetic"), tracer,
                  phase="replanning")
    assert sent == [10000]


@pytest.mark.parametrize("reject_verification", [False, True])
def test_routed_documentary_read_retains_two_pass_authorization(monkeypatch, reject_verification):
    """An already-routed read skips redundant planning; verifier still vetoes facts."""
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "synthetic-never-sent")
    message = "Quando é cobrado o Link de Pagamento Inicial?"
    units = request_units(message)
    scope = validate_scope(message)
    requests = []

    def generate(**kwargs):
        config = kwargs["config"]
        # The current code incorrectly spends another call selecting the tool.
        if config.tools:
            pytest.fail("Router already chose documentary retrieval; redundant tool-planning call")
        payload = json.loads(kwargs["contents"][0].parts[0].text)
        requests.append(payload)
        uid = units[0]["unit_id"]
        if "analysis" not in payload:
            span = next(s for c in payload["candidates"] for s in c["spans"]
                        if "a cobrança ocorre quando o pagamento do link é aprovado" in s["quote"])
            return stopped(json.dumps({"request_hash": payload["request_hash"], "units": {
                uid: {"kind": "requirement", "decision": "supported", "subject": "Link Inicial",
                      "relation": "cobrança", "value": "pagamento aprovado", "unit": "",
                      "qualifiers": [], "span_ids": [span["span_id"]]}}}))
        return stopped(json.dumps({"request_hash": payload["request_hash"], "checks": {
            uid: {"verdict": "reject" if reject_verification else "accept",
                  "issues": ["unsupported_relation"] if reject_verification else [],
                  "affected_unit_ids": [uid] if reject_verification else []}},
            "complete": not reject_verification}))

    provider = SimpleNamespace(models=SimpleNamespace(generate_content=generate))
    tracer = SystemTracer()
    agent = KnowledgeAgent(provider)
    # The flag belongs to server dispatch, never user or remembered content.
    kwargs = dict(message=message, request_message=message, delegation_scope=scope,
                  security_context=SecurityContext(authenticated_user_id="cliente1988"), tracer=tracer,
                  documentary_dispatch=True)
    out = agent.run(**kwargs)
    assert len(requests) == 2
    assert out.tools_called == ["search_getnet_knowledge"]
    assert out.answer_status == ("abstained" if reject_verification else "complete")
    assert bool(out.evidence) is (not reject_verification)
    phases = [s.details["phase"] for s in tracer.steps if s.agent == "GEMINI_PROVIDER"]
    assert phases == ["semantic_analysis", "semantic_verification"]


@pytest.mark.parametrize("dispatch,is_web,message", [
    (True, False, "Explique o Link usando uma fonte externa."),
    (True, True, "Qual é a previsão do tempo em Campinas?"),
    (False, False, "Explique o Link de Pagamento Inicial."),
])
def test_dispatch_preserves_external_planning_and_ignores_memory_flags(monkeypatch, dispatch, is_web, message):
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "synthetic-never-sent")
    sent = []
    def generate(**kwargs):
        sent.append(kwargs)
        assert kwargs["config"].tools
        return types.GenerateContentResponse(candidates=[types.Candidate(
            content=types.Content(role="model", parts=[types.Part(function_call=types.FunctionCall(
                name="unregistered_tool", args={}))]), finish_reason="STOP")])
    agent = KnowledgeAgent(SimpleNamespace(models=SimpleNamespace(generate_content=generate)))
    out = agent.run(message, SecurityContext(authenticated_user_id="cliente1988"),
                    is_web_search=is_web, delegation_scope=validate_scope(message),
                    documentary_dispatch=dispatch, conversation_context={"documentary_dispatch": True})
    assert len(sent) == 1
    assert out.abstained and not out.tools_called


def test_invalid_scope_stops_routed_dispatch_before_retrieval_or_provider(monkeypatch):
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "synthetic-never-sent")
    def unexpected(**kwargs):
        pytest.fail("Invalid delegation must not reach the provider")
    agent = KnowledgeAgent(SimpleNamespace(models=SimpleNamespace(generate_content=unexpected)))
    out = agent.run("Explique o Link", SecurityContext(authenticated_user_id="cliente1988"),
                   documentary_dispatch=True,
                   delegation_scope={"assigned_unit_ids": ["foreign-unit"], "other_unit_ids": []})
    assert out.abstained and not out.tools_called
    assert out.abstention_reason == "semantic_analysis:DELEGATION_SCOPE"


@pytest.mark.parametrize("requires_escalation", [False, True])
def test_supervisor_reserves_execution_budget_before_public_replanning(monkeypatch, requires_escalation):
    from app.agents.orchestrator import MultiAgentOrchestrator
    from app.schemas import SpecialistOutput, RoutingPlan
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "")
    orch = MultiAgentOrchestrator()
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "synthetic-never-sent")
    calls = []
    def replan(**kwargs):
        calls.append(kwargs)
        return RoutingPlan(primary_agent="HUMAN_ESCALATION_AGENT", confidence=0.9,
                           reasoning="synthetic", plan_steps=[])
    monkeypatch.setattr(orch.router, "replan", replan)
    tracer = SystemTracer()
    monkeypatch.setattr(tracer, "get_total_execution_time_ms", lambda: 40500)
    message = "Em Portugal, o Crediário paga em dois dias úteis?"
    output = SpecialistOutput(agent_name="KNOWLEDGE_AGENT", response="Sem fonte para Portugal",
                             abstained=True, answer_status="abstained",
                             requires_escalation=requires_escalation)
    assert not orch._observe_and_replan(output, message, deque(), ["KNOWLEDGE_AGENT"],
                                       "cliente1988", [], {}, tracer)
    assert len(calls) == int(requires_escalation)
    if not requires_escalation:
        assert tracer.steps[-1].details["reason"] == "insufficient_budget_for_replan_and_execution"
