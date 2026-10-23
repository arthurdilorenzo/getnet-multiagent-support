"""Native Web proposals must respect verified Getnet Pix evidence and private scope."""
import pytest
from google.genai import types
import importlib

from app.agents.router_agent import RouterAgent
from app.agents.orchestrator import MultiAgentOrchestrator
from app.config import settings
from app.memory.store import StateStore
from app.schemas import ChatRequest, PlanStep, RoutingPlan, SpecialistOutput


class WebProposal:
    def __init__(self, assignments):
        self.assignments = assignments
        self.models = self

    def generate_content(self, **kwargs):
        parts = [types.Part(function_call=types.FunctionCall(
            name="WEB_SEARCH_AGENT",
            args={"sub_query": query, "rationale": "Model proposed public web", "request_unit_ids": ids},
        )) for query, ids in self.assignments]
        return types.GenerateContentResponse(candidates=[types.Candidate(
            content=types.Content(role="model", parts=parts), finish_reason="STOP")])


def test_verified_getnet_account_categories_override_model_web(monkeypatch):
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "synthetic-never-sent")
    message = "Name the account types accepted for Pix in Brazil."
    plan = RouterAgent(WebProposal([(message, ["u1"])] )).plan(message, "synthetic")
    assert [(s.target_agent, s.request_unit_ids) for s in plan.plan_steps] == [
        ("KNOWLEDGE_AGENT", ["u1"]),
    ]


@pytest.mark.parametrize("message", [
    "Find the current Banco Central regulation on Pix account eligibility in Brazil.",
    "Find the latest external regulations for Pix account eligibility in Brazil.",
])
def test_explicit_current_external_pix_regulation_stays_web(monkeypatch, message):
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "synthetic-never-sent")
    plan = RouterAgent(WebProposal([(message, ["u1"])] )).plan(message, "synthetic")
    assert [(s.target_agent, s.request_unit_ids) for s in plan.plan_steps] == [
        ("WEB_SEARCH_AGENT", ["u1"]),
    ]


def test_mixed_pix_and_private_firmware_get_distinct_assignments(monkeypatch):
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "synthetic-never-sent")
    message = ("Name the account types accepted for Pix in Brazil plus inspect the installed "
               "firmware on GET-CLASSICA-QA-84257.")
    plan = RouterAgent(WebProposal([(message, ["u1", "u2"])] )).plan(message, "synthetic")
    assert [(s.target_agent, s.request_unit_ids) for s in plan.plan_steps] == [
        ("KNOWLEDGE_AGENT", ["u1"]),
        ("CUSTOMER_SUPPORT_AGENT", ["u2"]),
    ]
    assert "firmware" not in plan.plan_steps[0].sub_query.lower()
    assert "GET-CLASSICA-QA-84257" not in plan.plan_steps[0].sub_query
    assert "Pix" not in plan.plan_steps[1].sub_query


def test_unrelated_current_weather_stays_web(monkeypatch):
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "synthetic-never-sent")
    message = "Find the current weather in Campinas."
    plan = RouterAgent(WebProposal([(message, ["u1"])] )).plan(message, "synthetic")
    assert plan.primary_agent == "WEB_SEARCH_AGENT"


def test_getnet_current_pix_account_rules_use_verified_product_source(monkeypatch):
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "synthetic-never-sent")
    message = "What are Getnet's current Pix account rules?"
    plan = RouterAgent(WebProposal([(message, ["u1"])] )).plan(message, "synthetic")
    assert [(s.target_agent, s.request_unit_ids) for s in plan.plan_steps] == [
        ("KNOWLEDGE_AGENT", ["u1"]),
    ]


def test_installed_firmware_read_stays_private_despite_manual_mention(monkeypatch):
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "synthetic-never-sent")
    message = "According to the manual, inspect currently installed firmware on GET-CLASSICA-QA-84257."
    plan = RouterAgent(WebProposal([(message, ["u1"])] )).plan(message, "synthetic")
    assert [(s.target_agent, s.request_unit_ids) for s in plan.plan_steps] == [
        ("CUSTOMER_SUPPORT_AGENT", ["u1"]),
    ]


def test_many_units_fit_execution_budget_without_losing_assignments(monkeypatch):
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "synthetic-never-sent")
    message = ("Name the account types accepted for Pix in Brazil; "
               "find current weather in Campinas; find current weather in Santos; "
               "find current weather in Recife; find current weather in Manaus.")
    plan = RouterAgent(WebProposal([(message, ["u1", "u2", "u3", "u4", "u5"])] )).plan(
        message, "synthetic")
    assert len(plan.plan_steps) <= settings.MAX_ORCHESTRATION_ITERATIONS
    assert [(s.target_agent, s.request_unit_ids) for s in plan.plan_steps] == [
        ("KNOWLEDGE_AGENT", ["u1"]),
        ("WEB_SEARCH_AGENT", ["u2", "u3", "u4", "u5"]),
    ]


def test_multiple_web_proposals_keep_sequential_dependencies(monkeypatch):
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "synthetic-never-sent")
    message = ("Name Pix account types in Brazil; find weather in Campinas; "
               "name Pix account types for Getnet; find weather in Santos; "
               "name account types accepted for Pix; find weather in Recife.")
    provider = WebProposal([
        ("first pair", ["u1", "u2"]),
        ("second pair", ["u3", "u4"]),
        ("third pair", ["u5", "u6"]),
    ])
    plan = RouterAgent(provider).plan(message, "synthetic")
    assert len(plan.plan_steps) == 6
    assert [(s.target_agent, s.request_unit_ids) for s in plan.plan_steps] == [
        ("KNOWLEDGE_AGENT", ["u1"]),
        ("WEB_SEARCH_AGENT", ["u2"]),
        ("KNOWLEDGE_AGENT", ["u3"]),
        ("WEB_SEARCH_AGENT", ["u4"]),
        ("KNOWLEDGE_AGENT", ["u5"]),
        ("WEB_SEARCH_AGENT", ["u6"]),
    ]


def test_nonadjacent_repeated_model_assignments_do_not_change_order(monkeypatch):
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "synthetic-never-sent")
    message = ("Name Pix account types in Brazil; find weather in Campinas; "
               "find weather in Santos; name account types accepted for Pix; "
               "find weather in Recife.")
    provider = WebProposal([
        ("first pair", ["u1", "u2"]),
        ("repeated first unit", ["u1", "u3"]),
        ("third pair", ["u4", "u5"]),
    ])
    plan = RouterAgent(provider).plan(message, "synthetic")
    assert [(s.target_agent, s.request_unit_ids) for s in plan.plan_steps] == [
        ("KNOWLEDGE_AGENT", ["u1"]),
        ("WEB_SEARCH_AGENT", ["u2"]),
        ("KNOWLEDGE_AGENT", ["u1"]),
        ("WEB_SEARCH_AGENT", ["u3"]),
        ("KNOWLEDGE_AGENT", ["u4"]),
        ("WEB_SEARCH_AGENT", ["u5"]),
    ]


def test_nonadjacent_web_steps_and_empty_id_workflow_keep_order_and_queries():
    plan = RoutingPlan(primary_agent="WEB_SEARCH_AGENT", confidence=0.9, reasoning="ordered work",
        plan_steps=[
            PlanStep(step_order=1, target_agent="WEB_SEARCH_AGENT", sub_query="weather first", rationale="one", request_unit_ids=["u1"]),
            PlanStep(step_order=2, target_agent="CUSTOMER_SUPPORT_AGENT", sub_query="private second", rationale="two", request_unit_ids=["u2"]),
            PlanStep(step_order=3, target_agent="KNOWLEDGE_AGENT", sub_query="documentation third", rationale="three", request_unit_ids=["u3"]),
            PlanStep(step_order=4, target_agent="WEB_SEARCH_AGENT", sub_query="weather fourth", rationale="four", request_unit_ids=["u4"]),
            PlanStep(step_order=5, target_agent="HUMAN_ESCALATION_AGENT", sub_query="handoff fifth", rationale="five", request_unit_ids=[]),
        ])
    message = ("Find weather in Campinas; inspect firmware on GET-CLASSICA-QA-84257; "
               "explain Pix Getnet; find weather in Santos; contact a human agent.")
    revised = RouterAgent._prefer_local_documentary_evidence(plan, message)
    assert [(s.target_agent, s.request_unit_ids) for s in revised.plan_steps] == [
        ("WEB_SEARCH_AGENT", ["u1"]),
        ("CUSTOMER_SUPPORT_AGENT", ["u2"]),
        ("KNOWLEDGE_AGENT", ["u3"]),
        ("WEB_SEARCH_AGENT", ["u4"]),
        ("HUMAN_ESCALATION_AGENT", []),
    ]
    assert revised.plan_steps[-1].sub_query == "handoff fifth"
    assert len(revised.plan_steps) > settings.MAX_ORCHESTRATION_ITERATIONS


def test_nonadjacent_empty_id_workflow_queries_are_both_preserved():
    plan = RoutingPlan(primary_agent="HUMAN_ESCALATION_AGENT", confidence=0.9, reasoning="ordered workflow",
        plan_steps=[
            PlanStep(step_order=1, target_agent="HUMAN_ESCALATION_AGENT", sub_query="first handoff", rationale="one", request_unit_ids=[]),
            PlanStep(step_order=2, target_agent="KNOWLEDGE_AGENT", sub_query="first explanation", rationale="two", request_unit_ids=[]),
            PlanStep(step_order=3, target_agent="WEB_SEARCH_AGENT", sub_query="external update", rationale="three", request_unit_ids=[]),
            PlanStep(step_order=4, target_agent="CUSTOMER_SUPPORT_AGENT", sub_query="private read", rationale="four", request_unit_ids=[]),
            PlanStep(step_order=5, target_agent="HUMAN_ESCALATION_AGENT", sub_query="second handoff", rationale="five", request_unit_ids=[]),
        ])
    revised = RouterAgent._prefer_local_documentary_evidence(plan, "Contact an agent for both stages.")
    assert [(step.target_agent, step.sub_query) for step in revised.plan_steps] == [
        ("HUMAN_ESCALATION_AGENT", "first handoff"),
        ("KNOWLEDGE_AGENT", "first explanation"),
        ("WEB_SEARCH_AGENT", "external update"),
        ("CUSTOMER_SUPPORT_AGENT", "private read"),
        ("HUMAN_ESCALATION_AGENT", "second handoff"),
    ]


def test_over_budget_ordered_plan_leaves_trailing_units_explicitly_pending(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "GEMINI_API_KEY", None)
    orchestrator_module = importlib.import_module("app.agents.orchestrator")
    monkeypatch.setattr(orchestrator_module, "state_store", StateStore(tmp_path / "route-limit.sqlite"))
    message = ("Name Pix account types in Brazil; find weather in Campinas; "
               "name Pix account types for Getnet; find weather in Santos; "
               "name account types accepted for Pix; find weather in Recife.")
    planned = RouterAgent._prefer_local_documentary_evidence(
        RoutingPlan(primary_agent="WEB_SEARCH_AGENT", confidence=0.9, reasoning="ordered",
                    plan_steps=[PlanStep(step_order=1, target_agent="WEB_SEARCH_AGENT",
                                         sub_query=message, rationale="proposed",
                                         request_unit_ids=[f"u{i}" for i in range(1, 7)])]),
        message,
    )
    orch = MultiAgentOrchestrator()
    monkeypatch.setattr(orch.router, "plan", lambda **kwargs: planned)
    monkeypatch.setattr(orch, "_observe_and_replan", lambda **kwargs: False)
    seen = []

    def delegate(**kwargs):
        ids = kwargs["delegation_scope"]["assigned_unit_ids"]
        seen.append((kwargs["target_agent"], ids))
        return SpecialistOutput(agent_name=kwargs["target_agent"], response="No verified evidence",
                                abstained=True, request_unit_ids=ids, unresolved_unit_ids=ids)

    monkeypatch.setattr(orch, "_delegate_to_specialist", delegate)
    response = orch.process(ChatRequest(message=message, user_id="synthetic"))
    assert seen == [
        ("KNOWLEDGE_AGENT", ["u1"]), ("WEB_SEARCH_AGENT", ["u2"]),
        ("KNOWLEDGE_AGENT", ["u3"]), ("WEB_SEARCH_AGENT", ["u4"]),
    ]
    stop = next(step for step in response.trace if step.action == "evaluate_stopping_condition")
    assert stop.details["stopping_reason"] == "iteration_limit"
    assert stop.details["pending_steps"] == 2
    assert [(row["unit_id"], row["state"]) for row in response.request_coverage[-2:]] == [
        ("u5", "delegated"), ("u6", "delegated"),
    ]
    assert response.abstained is True
