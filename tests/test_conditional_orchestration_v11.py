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
