"""Native Web proposals must respect verified Getnet Pix evidence and private scope."""
import pytest
from google.genai import types

from app.agents.router_agent import RouterAgent
from app.config import settings


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


def test_multiple_web_proposals_compact_without_dropping_later_units(monkeypatch):
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
    assert len(plan.plan_steps) <= settings.MAX_ORCHESTRATION_ITERATIONS
    assert [(s.target_agent, s.request_unit_ids) for s in plan.plan_steps] == [
        ("KNOWLEDGE_AGENT", ["u1", "u3", "u5"]),
        ("WEB_SEARCH_AGENT", ["u2", "u4", "u6"]),
    ]


def test_compaction_deduplicates_repeated_model_unit_assignments(monkeypatch):
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
        ("KNOWLEDGE_AGENT", ["u1", "u4"]),
        ("WEB_SEARCH_AGENT", ["u2", "u3", "u5"]),
    ]
