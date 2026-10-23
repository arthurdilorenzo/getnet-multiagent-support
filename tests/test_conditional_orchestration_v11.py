"""Server-owned gates for conditional documentary work."""
import importlib

import pytest

from app.agents.orchestrator import MultiAgentOrchestrator
from app.config import settings
from app.memory.store import StateStore
from app.schemas import ChatRequest, PlanStep, RoutingPlan, SpecialistOutput, ToolExecutionResult
from app.security.auth import SecurityContext


TARGET = "GET-SMART-QA-73146"
MESSAGE = f"O recebimento do Crediário só deve ser explicado quando o {TARGET} tiver firmware qa-12.08 com conexão online."


def knowledge_plan(message):
    from app.security.request_units import request_units
    ids = [unit["unit_id"] for unit in request_units(message)]
    return RoutingPlan(primary_agent="KNOWLEDGE_AGENT", confidence=0.9, reasoning="synthetic plan",
        plan_steps=[PlanStep(step_order=1, target_agent="KNOWLEDGE_AGENT", sub_query=message,
            rationale="model proposal", request_unit_ids=ids)])


def make_orchestrator(monkeypatch, tmp_path, message=MESSAGE):
    module = importlib.import_module("app.agents.orchestrator")
    state = StateStore(tmp_path / "conditional-v11.sqlite")
    monkeypatch.setattr(module, "state_store", state)
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "")
    monkeypatch.setattr(settings, "AUTH_ENFORCE_STRICT", False)
    orch = MultiAgentOrchestrator()
    monkeypatch.setattr(orch.router, "plan", lambda **kwargs: knowledge_plan(message))
    return orch


def diagnostic(*, firmware="qa-12.07", status="Online", success=True, target=TARGET):
    data = {"requested_terminal_id": target, "found": True,
            "terminal": {"terminal_id": target, "firmware_version": firmware, "status": status}}
    return ToolExecutionResult(tool_name="diagnose_terminal_issue", success=success,
        data=data if success else None, error_code=None if success else "TOOL_UNAVAILABLE",
        outcome_state="completed" if success else "failed")


@pytest.mark.parametrize("result", [
    diagnostic(firmware="qa-12.07"),
    diagnostic(firmware=None),
    diagnostic(status="Offline"),
    diagnostic(success=False),
])
def test_failed_or_unknown_terminal_prerequisite_never_calls_knowledge(monkeypatch, tmp_path, result):
    orch = make_orchestrator(monkeypatch, tmp_path)
    calls = []

    def delegate(**kwargs):
        calls.append(kwargs["target_agent"])
        assert kwargs["target_agent"] == "CUSTOMER_SUPPORT_AGENT"
        assert TARGET in kwargs["sub_query"]
        return SpecialistOutput(agent_name="CUSTOMER_SUPPORT_AGENT", response="read performed",
            tools_called=["diagnose_terminal_issue"], tool_results=[result])

    monkeypatch.setattr(orch, "_delegate_to_specialist", delegate)
    out = orch.process(ChatRequest(message=MESSAGE, user_id="synthetic", session_id="cond-false"),
                       SecurityContext("synthetic"))

    assert calls == ["CUSTOMER_SUPPORT_AGENT"]
    assert "search_getnet_knowledge" not in out.tools_called
    assert "web_search_general" not in out.tools_called
    assert "não consultei os termos do Crediário" in out.response
    assert out.answer_status == "abstained"


def test_exact_terminal_predicate_allows_knowledge_only_after_read(monkeypatch, tmp_path):
    orch = make_orchestrator(monkeypatch, tmp_path)
    calls = []
    unit_id = "u1"

    def delegate(**kwargs):
        target_agent = kwargs["target_agent"]
        calls.append(target_agent)
        if target_agent == "CUSTOMER_SUPPORT_AGENT":
            return SpecialistOutput(agent_name=target_agent, response="internal prerequisite",
                tools_called=["diagnose_terminal_issue"], tool_results=[diagnostic(firmware="qa-12.08")])
        assert target_agent == "KNOWLEDGE_AGENT"
        return SpecialistOutput(agent_name=target_agent,
            response="O Crediário tem recebimento em até dois dias úteis, taxa única e sem custo de antecipação.",
            semantic_mode="limited_offline", answer_status="complete", request_unit_ids=[unit_id],
            covered_unit_ids=[unit_id], evidence=[{"chunk_id": "getnet_crediario_s1_9ea682"}],
            tools_called=["search_getnet_knowledge"])

    monkeypatch.setattr(orch, "_delegate_to_specialist", delegate)
    out = orch.process(ChatRequest(message=MESSAGE, user_id="synthetic", session_id="cond-true"),
                       SecurityContext("synthetic"))

    assert calls == ["CUSTOMER_SUPPORT_AGENT", "KNOWLEDGE_AGENT"]
    assert out.answer_status == "complete"
    assert out.request_coverage[0]["state"] == "fulfilled"


def test_unrecognized_added_condition_blocks_all_downstream_search(monkeypatch, tmp_path):
    message = MESSAGE[:-1] + ", caso a conta esteja regular."
    orch = make_orchestrator(monkeypatch, tmp_path, message)
    calls = []
    monkeypatch.setattr(orch, "_delegate_to_specialist", lambda **kwargs: calls.append(kwargs["target_agent"]))

    out = orch.process(ChatRequest(message=message, user_id="synthetic", session_id="cond-unknown"),
                       SecurityContext("synthetic"))

    assert calls == []
    assert "search_getnet_knowledge" not in out.tools_called
    assert "web_search_general" not in out.tools_called
    assert out.answer_status == "abstained"


@pytest.mark.parametrize("predicate", ["false", "unknown"])
@pytest.mark.parametrize("plan_shape", ["mixed", "separate"])
def test_blocked_dependency_cannot_return_through_independent_abstention_replan(monkeypatch, tmp_path, predicate, plan_shape):
    from app.security.request_units import request_units
    message = "Consulte a cotação EUR BRL; " + MESSAGE
    orch = make_orchestrator(monkeypatch, tmp_path, message)
    units = request_units(message)
    independent = units[0]["unit_id"]
    dependent = next(u["unit_id"] for u in units if "Crediário" in u["text"])
    if predicate == "false":
        # Task 1's finite parser deliberately rejects mixed-sentence grammar.
        # Exercise the typed false-observation gate separately from that parser.
        dependency = orch._conditional_crediario_dependency(MESSAGE)
        dependency["unit_ids"] = [dependent]
        monkeypatch.setattr(orch, "_conditional_crediario_dependency", lambda text: dependency)
    steps = [PlanStep(step_order=1, target_agent="WEB_SEARCH_AGENT",
        sub_query=message if plan_shape == "mixed" else units[0]["text"], rationale="independent lookup",
        request_unit_ids=[independent, dependent] if plan_shape == "mixed" else [independent])]
    if plan_shape == "separate":
        steps.append(PlanStep(step_order=2, target_agent="KNOWLEDGE_AGENT", sub_query=MESSAGE,
            rationale="dependent lookup", request_unit_ids=[dependent]))
    monkeypatch.setattr(orch.router, "plan", lambda **kwargs: RoutingPlan(
        primary_agent="WEB_SEARCH_AGENT", confidence=.9, reasoning="mixed scope", plan_steps=steps))
    replans = []
    def replan(**kwargs):
        replans.append(kwargs)
        return RoutingPlan(primary_agent="KNOWLEDGE_AGENT", confidence=.9, reasoning="unsafe proposal",
            plan_steps=[PlanStep(step_order=1, target_agent="KNOWLEDGE_AGENT", sub_query=MESSAGE,
                rationale="retry dependent", request_unit_ids=[dependent])])
    monkeypatch.setattr(orch.router, "replan", replan)
    calls = []
    def delegate(**kwargs):
        calls.append(kwargs["target_agent"])
        if kwargs["target_agent"] == "CUSTOMER_SUPPORT_AGENT":
            return SpecialistOutput(agent_name="CUSTOMER_SUPPORT_AGENT", response="read performed",
                tools_called=["diagnose_terminal_issue"], tool_results=[diagnostic()])
        assert kwargs["target_agent"] == "WEB_SEARCH_AGENT"
        assert "Crediário" not in kwargs["sub_query"]
        assert "firmware" not in kwargs["sub_query"]
        assert kwargs["delegation_scope"]["assigned_unit_ids"] == [independent]
        return SpecialistOutput(agent_name="WEB_SEARCH_AGENT", response="independent lookup unavailable",
            abstained=True, answer_status="abstained", request_unit_ids=[independent],
            unresolved_unit_ids=[independent])
    monkeypatch.setattr(orch, "_delegate_to_specialist", delegate)
    out = orch.process(ChatRequest(message=message, user_id="synthetic", session_id="mixed"), SecurityContext("synthetic"))
    assert replans
    assert calls == (["CUSTOMER_SUPPORT_AGENT"] if predicate == "false" else []) + ["WEB_SEARCH_AGENT"]
    assert "search_getnet_knowledge" not in out.tools_called
    assert next(r for r in out.request_coverage if r["unit_id"] == dependent)["state"] == "unresolved"


@pytest.mark.parametrize("workflow_source", ["initial", "replan"])
@pytest.mark.parametrize("trigger", ["critical", "failure"])
def test_blocked_condition_preserves_authorized_workflow_handoff(monkeypatch, tmp_path, workflow_source, trigger):
    _run_blocked_condition_handoff(monkeypatch, tmp_path, workflow_source, trigger, expected=True)


def test_blocked_condition_does_not_authorize_unrequested_workflow_handoff(monkeypatch, tmp_path):
    _run_blocked_condition_handoff(monkeypatch, tmp_path, "replan", "abstention", expected=False)


def _run_blocked_condition_handoff(monkeypatch, tmp_path, workflow_source, trigger, *, expected):
    from app.security.request_units import request_units
    from app.tools.registry import tool_registry
    message = "Verifique meu terminal; " + MESSAGE
    orch = make_orchestrator(monkeypatch, tmp_path, message)
    units = request_units(message)
    independent = units[0]["unit_id"]
    dependent = next(unit["unit_id"] for unit in units if "Crediário" in unit["text"])
    handoff = PlanStep(step_order=2, target_agent="HUMAN_ESCALATION_AGENT",
        sub_query="Encaminhe ao suporte humano", rationale="Operational workflow handoff", request_unit_ids=[])
    initial = [PlanStep(step_order=1, target_agent="CUSTOMER_SUPPORT_AGENT", sub_query=units[0]["text"],
        rationale="Independent diagnosis", request_unit_ids=[independent])]
    if workflow_source == "initial":
        initial.append(handoff)
    monkeypatch.setattr(orch.router, "plan", lambda **kwargs: RoutingPlan(
        primary_agent="CUSTOMER_SUPPORT_AGENT", confidence=.9, reasoning="diagnosis", plan_steps=initial))
    monkeypatch.setattr(orch.router, "replan", lambda **kwargs: RoutingPlan(
        primary_agent="HUMAN_ESCALATION_AGENT", confidence=.9, reasoning="workflow with forbidden factual retry",
        plan_steps=[handoff, PlanStep(step_order=3, target_agent="KNOWLEDGE_AGENT", sub_query=MESSAGE,
            rationale="Must remain blocked", request_unit_ids=[dependent])]))
    monkeypatch.setattr(orch.customer_support_agent, "run", lambda **kwargs: SpecialistOutput(
        agent_name="CUSTOMER_SUPPORT_AGENT", response="Independent diagnosis needs attention",
        requires_escalation=trigger == "critical", abstained=True, request_unit_ids=[independent],
        unresolved_unit_ids=[independent], tools_called=["diagnose_terminal_issue"] if trigger != "abstention" else [],
        tool_results=[diagnostic(success=trigger != "failure")] if trigger != "abstention" else []))
    monkeypatch.setattr(orch.knowledge_agent, "run", lambda **kwargs: pytest.fail("Blocked factual lookup executed"))
    tools = []
    def execute(**kwargs):
        tools.append(kwargs["tool_name"])
        assert kwargs["tool_name"] == "create_escalation_ticket"
        return ToolExecutionResult(tool_name="create_escalation_ticket", success=True, outcome_state="completed",
            data={"ticket_id":"GET-ESC-CONDITIONAL-V11", "status":"OPEN"})
    # The real escalation agent and orchestration run; isolate only ticket storage.
    monkeypatch.setattr(tool_registry, "execute", execute)
    out = orch.process(ChatRequest(message=message, user_id="synthetic", session_id="handoff-blocked"),
                       SecurityContext("synthetic"))
    assert out.escalated is expected
    assert out.ticket_id == ("GET-ESC-CONDITIONAL-V11" if expected else None)
    assert tools == (["create_escalation_ticket"] if expected else [])
    assert "search_getnet_knowledge" not in out.tools_called
    assert "web_search_general" not in out.tools_called
    assert next(row for row in out.request_coverage if row["unit_id"] == dependent)["state"] == "unresolved"


@pytest.mark.parametrize("message,recognized", [
    (f"If firmware of {TARGET} is qa-12.08 and status is Online, tell me the eligible benefits.", True),
    (f"Se o firmware do terminal {TARGET} for qa-12.08 e o status estiver online, informe os benefícios do Crediário.", False),
    (f"Explain the benefits provided that firmware of {TARGET} is newer than qa-12.08.", False),
    (f"Informe os benefícios quando o status do terminal {TARGET} estiver disponível.", False),
])
def test_terminal_dependency_intent_blocks_search_outside_authorization_grammar(monkeypatch, tmp_path, message, recognized):
    orch = make_orchestrator(monkeypatch, tmp_path, message)
    calls = []
    def delegate(**kwargs):
        calls.append(kwargs["target_agent"])
        assert kwargs["target_agent"] == "CUSTOMER_SUPPORT_AGENT", "dependent factual lookup escaped the gate"
        return SpecialistOutput(agent_name="CUSTOMER_SUPPORT_AGENT", response="internal read",
            tools_called=["diagnose_terminal_issue"], tool_results=[diagnostic()])
    monkeypatch.setattr(orch, "_delegate_to_specialist", delegate)
    out = orch.process(ChatRequest(message=message, user_id="synthetic"), SecurityContext("synthetic"))
    assert calls == (["CUSTOMER_SUPPORT_AGENT"] if recognized else [])
    assert all(row["state"] == "unresolved" for row in out.request_coverage)
    assert not {"search_getnet_knowledge", "web_search_general"} & set(out.tools_called)
    assert out.answer_status == "abstained"


def test_supported_if_benefits_lookup_requires_exact_positive_read(monkeypatch, tmp_path):
    message = f"If firmware of {TARGET} is qa-12.08 and status is Online, tell me the eligible benefits."
    orch = make_orchestrator(monkeypatch, tmp_path, message)
    calls = []
    def delegate(**kwargs):
        calls.append(kwargs["target_agent"])
        if kwargs["target_agent"] == "CUSTOMER_SUPPORT_AGENT":
            return SpecialistOutput(agent_name="CUSTOMER_SUPPORT_AGENT", response="internal read",
                tools_called=["diagnose_terminal_issue"], tool_results=[diagnostic(firmware="qa-12.08")])
        assert calls == ["CUSTOMER_SUPPORT_AGENT", "KNOWLEDGE_AGENT"]
        return SpecialistOutput(agent_name="KNOWLEDGE_AGENT", response="No verified benefit evidence",
            abstained=True, request_unit_ids=kwargs["delegation_scope"]["assigned_unit_ids"])
    monkeypatch.setattr(orch, "_delegate_to_specialist", delegate)
    monkeypatch.setattr(orch, "_observe_and_replan", lambda **kwargs: False)
    orch.process(ChatRequest(message=message, user_id="synthetic"), SecurityContext("synthetic"))
    assert calls == ["CUSTOMER_SUPPORT_AGENT", "KNOWLEDGE_AGENT"]


@pytest.mark.parametrize("tool_name,independent", [
    ("search_getnet_knowledge", "Name the account types accepted for Pix in Brazil"),
    ("web_search_general", "Find current external rules for Pix account eligibility in Brazil"),
])
def test_actual_registry_query_excludes_blocked_condition_even_when_native_query_repeats_it(monkeypatch, tmp_path, tool_name, independent):
    from google.genai import types
    from app.tools import registry as registry_module
    message = independent + "; " + MESSAGE
    orch = make_orchestrator(monkeypatch, tmp_path, message)
    monkeypatch.setattr(orch, "_observe_and_replan", lambda **kwargs: False)
    planning = []
    class Provider:
        def __init__(self): self.models = self
        def generate_content(self, **kwargs):
            planning.append(kwargs)
            return types.GenerateContentResponse(candidates=[types.Candidate(finish_reason="STOP",
                content=types.Content(role="model", parts=[types.Part(function_call=types.FunctionCall(
                    name=tool_name, args={"query": message}))]))])
    orch.knowledge_agent.client = Provider()
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "synthetic-never-sent")
    monkeypatch.setattr(registry_module, "state_store", StateStore(tmp_path / "registry.sqlite"))
    executed = []
    def handler(**kwargs):
        executed.append(kwargs)
        raise RuntimeError("Synthetic stop after the real registry validated/dispatched the read")
    tool = registry_module.tool_registry.get_tool(tool_name)
    monkeypatch.setattr(tool, "handler", handler)
    out = orch.process(ChatRequest(message=message, user_id="synthetic"), SecurityContext("synthetic"))
    assert len(executed) == 1
    assert executed[0]["query"] == independent
    assert all("Crediário" not in row["query"] and "firmware" not in row["query"] and TARGET not in row["query"]
               for row in executed)
    assert planning and message in str(planning[0]["contents"]), "authorization must keep the immutable full request"
    assert "conditional_lookup_blocked" in [row.action for row in out.trace]
    assert all(row["state"] == "unresolved" for row in out.request_coverage[1:])


def test_unsupported_postfixed_terminal_condition_keeps_previous_benefit_clause_blocked(monkeypatch, tmp_path):
    message = f"Explain the eligible benefits; only if firmware of {TARGET} is newer than qa-12.08."
    orch = make_orchestrator(monkeypatch, tmp_path, message)
    monkeypatch.setattr(orch, "_delegate_to_specialist", lambda **kwargs: pytest.fail("Unsupported dependent lookup escaped"))
    out = orch.process(ChatRequest(message=message, user_id="synthetic"), SecurityContext("synthetic"))
    assert all(row["state"] == "unresolved" for row in out.request_coverage)
    assert not out.tools_called
